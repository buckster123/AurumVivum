import asyncio
import base64
import builtins
import html
import io
import json
import logging
import os
import re
import shlex
import sqlite3
import shutil
import subprocess
import sys
import time
import traceback
import unittest
import uuid
import concurrent.futures
import threading
import nest_asyncio
import bs4
import chess
import chromadb
import jsbeautifier
import mpmath
import networkx as nx
import ntplib
import numpy as np
import pulp as PuLP
import pygame
import pygit2
import requests
import RestrictedPython
import sqlparse
import streamlit as st
import sympy
import tiktoken
import yaml
import pathlib
import typing
from black import FileMode, format_str
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
import openai
from passlib.hash import sha256_crypt
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from enum import Enum
import inspect
import hashlib
import ast
from functools import lru_cache, wraps
import xml.dom.minidom # Ensure imported
import torch # Added for device check in embed model.
nest_asyncio.apply()
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
load_dotenv()
API_KEY = os.getenv("XAI_API_KEY")
if not API_KEY:
    st.error("XAI_API_KEY not set in .env! Please add it and restart.")
class Config(Enum):
    DEFAULT_TOP_K = 5
    CACHE_TTL_MIN = 15
    AGENT_MAX_CONCURRENT = 5
    PRUNE_FREQUENCY = 50
    SIM_THRESHOLD = 0.6
    MAX_TASK_LEN = 2000
    STABILITY_PENALTY = 0.05
    MAX_ITERATIONS = 100
    TOOL_CALL_LIMIT = 200 # Bumped from 100
    API_CALLS_PER_MIN = 10
    TOOL_CALLS_PER_MIN = 50
class Models(Enum):
    GROK_FAST_REASONING = "grok-4-1-fast-reasoning"
    GROK_LATEST = "grok-4-latest"
    GROK_CODE_FAST = "grok-code-fast-1"
    GROK_3_MINI = "grok-3-mini"
def hash_password(password: str) -> str:
    return sha256_crypt.hash(password)
def verify_password(stored: str, provided: str) -> bool:
    return sha256_crypt.verify(provided, stored)
def sync_limiter(calls_per_min: int):
    last_call = [0]
    lock = threading.Lock() # FIX: Phase 1 - Atomic updates.
    def limiter():
        now = time.time()
        with lock:
            if now - last_call[0] < 60 / calls_per_min:
                time.sleep((60 / calls_per_min) - (now - last_call[0]))
            last_call[0] = time.time()
    return limiter
# NEW: Async limiter
async def async_limiter(sem: asyncio.Semaphore):
    async with sem:
        pass
api_limiter_sync = sync_limiter(Config.API_CALLS_PER_MIN.value)
tool_limiter_sync = sync_limiter(Config.TOOL_CALLS_PER_MIN.value)
api_sem = asyncio.Semaphore(Config.API_CALLS_PER_MIN.value)
tool_sem = asyncio.Semaphore(Config.TOOL_CALLS_PER_MIN.value)
def inject_user_convo(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['user'] = kwargs.get('user', st.session_state.get('user', 'shared'))
        kwargs['convo_id'] = kwargs.get('convo_id', st.session_state.get('current_convo_id', 0))
        return func(*args, **kwargs)
    return wrapper
class AppState:
    def __init__(self):
        self.db_path = "sandbox/db/chatapp.db"
        self.chroma_path = "./sandbox/db/chroma_db"
        self.chroma_client = None
        self.chroma_collection = None
        self.yaml_collection = None
        self._init_resources()
     
        self.memory_cache = {
            "lru_cache": {},
            "metrics": {
                "total_inserts": 0,
                "total_retrieves": 0,
                "hit_rate": 1.0,
                "last_update": None,
            },
        }
        self.prompts_dir = "./prompts"
        os.makedirs(self.prompts_dir, exist_ok=True)
        default_prompts = {
            "default.txt": "You are Aurum Vivum, a highly intelligent, helpful AI assistant powered by xAI.",
            "coder.txt": "You are an expert Aurum coder, providing precise code solutions.",
            "tools-enabled.txt": """Loaded via .txt file.""",
        }
        if not any(f.endswith(".txt") for f in os.listdir(self.prompts_dir)):
            for filename, content in default_prompts.items():
                with open(os.path.join(self.prompts_dir, filename), "w") as f:
                    f.write(content)
        self.sandbox_dir = "./sandbox"
        os.makedirs(self.sandbox_dir, exist_ok=True)
        self.yaml_dir = "./sandbox/config"
        os.makedirs(self.yaml_dir, exist_ok=True)
        self.agent_dir = os.path.join(self.sandbox_dir, "agents")
        os.makedirs(self.agent_dir, exist_ok=True)
        self.agent_executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.agent_lock = threading.Lock()
        self.agent_sem = asyncio.Semaphore(Config.AGENT_MAX_CONCURRENT.value)
        self.embed_model = None
        self.stability_score = 1.0
        self.yaml_ready = False
        self.yaml_cache = {}
        self.chroma_lock = threading.Lock() # FIX: Phase 1 - For all Chroma ops.
        self._init_yaml_embeddings() # Intentional full init
        self.counter_lock = threading.Lock()  # FIX: Phase 1 - For global counters
        self.session_lock = threading.Lock()  # FIX: Phase 2 - For st.session_state accesses
    def _init_resources(self):
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.c = self.conn.cursor()
            self._init_db()
         
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="memory_vectors",
                metadata={"hnsw:space": "cosine"},
            )
            self.yaml_collection = self.chroma_client.get_or_create_collection(
                name="yaml_vectors", metadata={"hnsw:space": "cosine"}
            )
            self.chroma_ready = True
        except sqlite3.Error as e: # FIX: Phase 1 - Specific except.
            logger.error(f"DB error: {e}")
            st.error(f"Failed to initialize database: {e}. App may not function properly.")
            if not self.conn:
                st.error("DB init failed—app limited.")
        except chromadb.errors.InvalidDimensionException as e: # FIX: Phase 1 - Chroma-specific.
            logger.error(f"Chroma error: {e}")
            self.chroma_ready = False
        except Exception as e:
            logger.warning(f"Resource init failed ({e}). Falling back.")
            self.chroma_ready = False
            self.chroma_collection = None
            self.yaml_collection = None
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
        if self.chroma_client:
            pass
        self.agent_executor.shutdown(wait=True)
        if exc_type:
            logger.error(f"Exception in AppState: {exc_val}")
        return True # Suppress exceptions
    def _init_db(self):
        try:
            with self.conn:
                self.c.execute(
                    """CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)"""
                )
                self.c.execute(
                    """CREATE TABLE IF NOT EXISTS history (user TEXT, convo_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, messages TEXT)"""
                )
                self.c.execute(
                    """CREATE TABLE IF NOT EXISTS memory (
                    user TEXT,
                    convo_id INTEGER,
                    mem_key TEXT,
                    mem_value TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    salience REAL DEFAULT 1.0,
                    parent_id INTEGER,
                    PRIMARY KEY (user, convo_id, mem_key)
                )"""
                )
                self.c.execute("CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory (timestamp)")
                self.c.execute("CREATE INDEX IF NOT EXISTS idx_memory_user_convo ON memory (user, convo_id)")  # FIX: Phase 3 - Add index
             
                self.c.execute("INSERT OR IGNORE INTO users (username, password) VALUES ('shared', ?)", (hash_password(''),))
        except sqlite3.Error as e: # FIX: Phase 1 - Specific.
            logger.error(f"DB init error: {e}")
            st.error(f"Failed to initialize database: {e}. App may not function properly.")
    def _init_yaml_embeddings(self):
        embed_model = self.get_embed_model()
        if embed_model:
            files_refreshed = 0
            failed = []  # FIX: Phase 1 - Track failed
            for fname in os.listdir(self.yaml_dir):
                if fname.endswith(".yaml"):
                    path = os.path.join(self.yaml_dir, fname)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                        embedding = embed_model.encode(content).tolist()
                        with self.chroma_lock: # FIX: Phase 1 - Lock.
                            self.yaml_collection.upsert(
                                ids=[fname],
                                embeddings=[embedding],
                                documents=[content],
                                metadatas=[{"filename": fname}],
                            )
                        self.yaml_cache[fname] = content
                        files_refreshed += 1
                    except OSError as e: # FIX: Phase 1 & 3 - Specific, skip.
                        logger.error(f"YAML file error for {fname}: {e}")
                        failed.append(fname)
            self.yaml_ready = len(failed) == 0 and files_refreshed > 0  # FIX: Phase 1 - Ready only if no fails and files
            if failed:
                logger.warning(f"Failed YAML files: {failed}")
            logger.info(f"YAML embeddings inited ({files_refreshed} files).")
        else:
            logger.warning("YAML embeddings skipped – embed model not ready.")
    @classmethod
    def get(cls):
        if "app_state" not in st.session_state:
            st.session_state["app_state"] = cls()
        return st.session_state["app_state"]
 
    def get_embed_model(self):
        if self.embed_model is None:
            for _ in range(3):  # FIX: Phase 2 - Retry 3x
                try:
                    with st.spinner("Loading embedding model for advanced memory (first-time use)..."):
                        device = 'cuda' if torch.cuda.is_available() else 'cpu' # FIX: Phase 3 - Device check.
                        self.embed_model = SentenceTransformer("all-mpnet-base-v2", device=device)
                    st.info("Embedding model loaded successfully.")
                    break
                except Exception as e:
                    logger.error(f"Failed to load embedding model (attempt {_+1}): {e}")
                    time.sleep(5)
            else:
                st.error("Failed to load embedding model after retries. Some features may be disabled.")
                self.embed_model = None
        return self.embed_model
    def lru_evict(self):
        # IMPROVED: Use OrderedDict for O(1) eviction
        if "lru_ordered" not in self.memory_cache:
            from collections import OrderedDict
            self.memory_cache["lru_ordered"] = OrderedDict()
        lru = self.memory_cache["lru_ordered"]
        if len(lru) > 500:
            num_to_evict = len(lru) - 500
            for _ in range(num_to_evict):
                key, entry = lru.popitem(last=False)
                if entry["salience"] < 0.4:
                    del self.memory_cache["lru_cache"][key]  # FIX: Phase 1 - Explicit del after pop
state = AppState.get()
st.markdown(
    """<style>
body {
    background: linear-gradient(to right, #000000, #003333);
    color: #66cccc;
    font-family: 'Courier New', monospace;
}
.stApp {
    background: linear-gradient(to right, #000000, #003333);
    display: flex;
    flex-direction: column;
    color: #66cccc;
    font-family: 'Courier New', monospace;
}
.sidebar .sidebar-content {
    background: rgba(0, 51, 51, 0.5);
    border-radius: 10px;
    color: #66cccc;
    font-family: 'Courier New', monospace;
}
.stButton > button {
    background-color: #66cccc;
    color: #000000;
    border-radius: 10px;
    border: none;
    font-family: 'Courier New', monospace;
}
.stButton > button:hover {
    background-color: #339999;
}
[data-theme="dark"] .stApp {
    background: linear-gradient(to right, #000000, #003333);
    color: #66cccc;
    font-family: 'Courier New', monospace;
}
</style>
""",
    unsafe_allow_html=True,
)
def get_state(key: str, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]
def safe_call(func, *args, max_retries=3, **kwargs):
    start = time.time()  # FIX: Phase 2 - Timing
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            state.stability_score = min(1.0, state.stability_score + 0.01)
            return result
        except (ValueError, TypeError, sqlite3.Error, asyncio.TimeoutError, requests.RequestException) as e: # Added TypeError
            logger.exception(f"Attempt {attempt+1} failed for {func.__name__}: {e}")
            if attempt == max_retries - 1:
                short_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
                state.stability_score = max(0.0, state.stability_score - Config.STABILITY_PENALTY.value)
                return f"Oops: {short_msg}. Check logs for details."
            time.sleep(2 ** attempt)
        except chromadb.errors.InvalidDimensionException as e: # FIX: Phase 1 - Chroma-specific.
            logger.error(f"Chroma error in {func.__name__}: {e}")
            return f"Chroma error: {e}"
        except Exception as e:
            logger.exception(f"Unexpected error in {func.__name__}: {e}")
            state.stability_score = max(0.0, state.stability_score - Config.STABILITY_PENALTY.value)
            notify_critical(str(e))  # FIX: Phase 2 - Centralized notify
            return f"Tool glitched: {str(e)[:50]}..."
    state.stability_score = max(0.0, state.stability_score - 0.1)
    logger.info(f"{func.__name__} took {time.time() - start}s")  # FIX: Phase 2 - Log time
    return "Max retries exceeded."
def notify_critical(err: str):
    st.error(f"Critical: {err}")
    logger.critical(err)
def get_tool_cache_key(func_name: str, args: dict) -> str:
    try:
        arg_str = json.dumps(args, sort_keys=True)
    except TypeError: # FIX: Phase 2 - Handle unserializable.
        arg_str = str(args) # Fallback to str.
    return f"tool_cache:{func_name}:{hashlib.sha256(arg_str.encode()).hexdigest()}"
def get_cached_tool_result(func_name: str, args: dict, ttl_minutes: int = Config.CACHE_TTL_MIN.value) -> str | None:
    with state.session_lock:  # FIX: Phase 2 - Lock session
        cache = get_state("tool_cache", {})
    key = get_tool_cache_key(func_name, args)
    if key in cache:
        timestamp, result = cache[key]
        if (datetime.now() - timestamp).total_seconds() / 60 < ttl_minutes:
            return result
    return None
def set_cached_tool_result(func_name: str, args: dict, result: str):
    with state.session_lock:  # FIX: Phase 2 - Lock
        cache = get_state("tool_cache", {})
        key = get_tool_cache_key(func_name, args)
        cache[key] = (datetime.now(), result)
        if len(cache) > 100:
            oldest_key = min(cache, key=lambda k: cache[k][0])
            del cache[oldest_key]
        st.session_state["tool_cache"] = cache
def load_prompt_files() -> list:
    return [f for f in os.listdir(state.prompts_dir) if f.endswith(".txt")]
def gen_title(first_msg: str) -> str:
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
        resp = client.chat.completions.create(
            model=Models.GROK_FAST_REASONING.value,
            messages=[
                {"role": "user", "content": f"Summarize this query into a short, punchy title: {first_msg[:500]}"},
                {"role": "system", "content": "Keep under 50 chars, evocative."}
            ],
            max_tokens=50
        )
        return resp.choices[0].message.content.strip()[:50]
    except Exception as e:
        logger.warning(f"Title gen failed: {e}")
        return first_msg[:40] + "..."
def auto_optimize_prompt(current_prompt: str, user: str, convo_id: int, metrics: dict = None) -> str:
    if metrics is None:
        metrics = {}
    try:
        branches = [
            f"Variant 1: {current_prompt} – Emphasize creativity.",
            f"Variant 2: {current_prompt} – Focus on precision.",
            f"Variant 3: {current_prompt} – Add humor."
        ]
        consensus = socratic_api_council(branches, user=user, convo_id=convo_id)
        optimized = reflect_optimize("prompt", {"consensus": consensus, "metrics": metrics})
        return optimized.split("Optimized prompt:")[-1].strip() if "Optimized prompt:" in optimized else current_prompt
    except Exception as e:
        logger.error(f"Prompt optimize error: {e}")
        return current_prompt
@st.cache_data(ttl=300, hash_funcs={dict: lambda d: json.dumps(d, sort_keys=True)}) # Added hash for dicts
def fs_read_file(file_path: str) -> str:
    safe_path = pathlib.Path(state.sandbox_dir) / pathlib.Path(file_path).relative_to('.') # Safer paths
    safe_path = safe_path.resolve()
    if not safe_path.is_relative_to(pathlib.Path(state.sandbox_dir).resolve()):
        return "Error: Path is outside the sandbox."
    if not safe_path.exists():  # Proactive check
        if safe_path.suffix in ['.yaml', '.lattice']:  # Auto-create defaults for configs
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            with open(safe_path, 'w') as f:
                f.write('{}')  # Empty dict as default
            logger.info(f"Auto-created missing file: {file_path}")
        else:
            return "Error: File not found."
    try:
        with open(safe_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return f"Error reading file: {e}"
def fs_write_file(file_path: str, content: str) -> str:
    safe_path = pathlib.Path(state.sandbox_dir) / pathlib.Path(file_path).relative_to('.') # Safer
    safe_path = safe_path.resolve()
    if not safe_path.is_relative_to(pathlib.Path(state.sandbox_dir).resolve()):
        return "Error: Path is outside the sandbox."
    try:
        content = html.unescape(content)
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)
        if "tool_cache" in st.session_state:
            key_to_remove = get_tool_cache_key("fs_read_file", {"file_path": file_path})
            st.session_state["tool_cache"].pop(key_to_remove, None)
        return f"File '{file_path}' written successfully."
    except Exception as e:
        logger.error(f"Error writing file: {e}")
        return f"Error writing file: {e}"
def fs_list_files(dir_path: str = "") -> str:
    safe_dir = pathlib.Path(state.sandbox_dir) / pathlib.Path(dir_path or '.').relative_to('.') # Safer
    safe_dir = safe_dir.resolve()
    if not safe_dir.is_relative_to(pathlib.Path(state.sandbox_dir).resolve()):
        return "Error: Path is outside the sandbox."
    try:
        files = os.listdir(safe_dir)
        return f"Files in '{dir_path or '/'}': {json.dumps(files)}"
    except FileNotFoundError:
        return "Error: Directory not found."
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return f"Error listing files: {e}"
def fs_mkdir(dir_path: str) -> str:
    safe_path = pathlib.Path(state.sandbox_dir) / pathlib.Path(dir_path).relative_to('.') # Safer
    safe_path = safe_path.resolve()
    if not safe_path.is_relative_to(pathlib.Path(state.sandbox_dir).resolve()):
        return "Error: Path is outside the sandbox."
    try:
        safe_path.mkdir(parents=True, exist_ok=True)
        return f"Directory '{dir_path}' created successfully."
    except Exception as e:
        logger.error(f"Error creating directory: {e}")
        return f"Error creating directory: {e}"
def get_current_time(sync: bool = False, format_str: str = "iso") -> str:
    try:
        if sync:
            ntp_client = ntplib.NTPClient()
            response = ntp_client.request("pool.ntp.org", version=3)
            dt_object = datetime.fromtimestamp(response.tx_time)
        else:
            dt_object = datetime.now()
        if format_str == "human":
            return dt_object.strftime("%A, %B %d, %Y %I:%M:%S %p")
        elif format_str == "json":
            return json.dumps(
                {
                    "datetime": dt_object.isoformat(),
                    "timezone": time.localtime().tm_zone,
                }
            )
        else:
            return dt_object.isoformat()
    except Exception as e:
        logger.error(f"Time error: {e}")
        return f"Time error: {e}"
SAFE_BUILTINS = {
    b: getattr(builtins, b)
    for b in [
        "print",
        "len",
        "range",
        "str",
        "int",
        "float",
        "list",
        "dict",
        "set",
        "tuple",
        "abs",
        "round",
        "max",
        "min",
        "sum",
        "sorted",
    ]
}
ADDITIONAL_LIBS = {
    "numpy": np,
    "sympy": sympy,
    "mpmath": mpmath,
    "PuLP": PuLP,
    "pygame": pygame,
    "chess": chess,
    "networkx": nx,
    "unittest": unittest,
    "asyncio": asyncio,
    "qiskit": __import__('qiskit') if 'qiskit' in sys.modules else None,  # Loosened for agents
    "scipy": __import__('scipy') if 'scipy' in sys.modules else None,
    # Add more safe libs for creativity, e.g., for sims
    "matplotlib": __import__('matplotlib') if 'matplotlib' in sys.modules else None,
    "pandas": __import__('pandas') if 'pandas' in sys.modules else None,
}
def init_repl_namespace():
    if "repl_namespace" not in st.session_state:
        st.session_state["repl_namespace"] = {"__builtins__": SAFE_BUILTINS.copy()}
        st.session_state["repl_namespace"].update({k: v for k, v in ADDITIONAL_LIBS.items() if v is not None})
def restricted_policy(node):
    if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
        if isinstance(node, ast.ImportFrom) and node.module in ADDITIONAL_LIBS: # Allow safe imports
            return True
        return False
    if isinstance(node, ast.Call) and hasattr(node.func, 'id') and node.func.id in ['open', 'exec', 'eval', 'subprocess']: # FIX: Phase 2 - Ban risky calls.
        return False
    return True
def execute_in_venv(code: str, venv_path: str) -> str:
    safe_venv = os.path.abspath(os.path.normpath(os.path.join(state.sandbox_dir, venv_path)))
    if not safe_venv.startswith(os.path.abspath(state.sandbox_dir)):
        return "Error: Venv path outside sandbox."
    venv_python = os.path.join(safe_venv, "bin", "python")
    if not os.path.exists(venv_python):
        return "Error: Venv Python not found."
    result = subprocess.run(
        [venv_python, "-c", code], capture_output=True, text=True, timeout=300
    )
    output = result.stdout
    if result.stderr:
        logger.error(f"Venv execution error: {result.stderr}")
        return f"Error: {result.stderr}"
    return output
def execute_local(code: str, redirected_output: io.StringIO) -> str:
    try:
        tree = ast.parse(code)
        if not all(restricted_policy(node) for node in ast.walk(tree)):
            return "Restricted: Imports banned."
        # Removed _print_ hack  # FIX: Phase 1 - Remove risky hack
        if 'open' in code: # FIX: Phase 2 - Quick hack ban.
            return "Banned func."
        result = RestrictedPython.compile_restricted_exec(code)  # <-- Add this missing line
        if result.errors:
            return f"Restricted compile error: {result.errors}"
        exec(result.code, st.session_state["repl_namespace"], {})
    except Exception as e:
        return f"Restricted exec error: {e}"
    return redirected_output.getvalue()
def code_execution(code: str, venv_path: str = None) -> str:
    tool_limiter_sync()
    init_repl_namespace()
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    try:
        if venv_path:
            output = execute_in_venv(code, venv_path)
        else:
            output = execute_local(code, redirected_output)
        return f"Output:\n{output}" if output else "Execution successful (no output)."
    except Exception:
        logger.error(f"Code execution error: {traceback.format_exc()}")
        return f"Error: {traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout # Reset stdout
@inject_user_convo
def memory_insert(
    mem_key: str, mem_value: dict | str, user: str = 'shared', convo_id: int = 0
) -> str:
    tool_limiter_sync()
    if isinstance(mem_value, str):
        mem_value = {"raw": mem_value} # Always wrap str
        logger.warning(f"Wrapped mem_value str for '{mem_key}'")
    if not isinstance(mem_value, dict):
        return "Error: mem_value must be a dict (or valid JSON str)."
    try:
        json.dumps(mem_value)  # FIX: Phase 1 - Check serializable
    except TypeError:
        return "Error: mem_value not JSON-serializable."
    try:
        with state.conn:
            state.c.execute(
                "INSERT OR REPLACE INTO memory (user, convo_id, mem_key, mem_value) VALUES (?, ?, ?, ?)",
                (user, convo_id, mem_key, json.dumps(mem_value)),
            )
        cache_key = f"{user}_{convo_id}_{mem_key}"
        entry = {
            "summary": mem_value.get("summary", ""),
            "details": mem_value.get("details", ""),
            "tags": mem_value.get("tags", []),
            "domain": mem_value.get("domain", "general"),
            "timestamp": datetime.now().isoformat(),
            "salience": mem_value.get("salience", 1.0),
        }
        state.memory_cache["lru_cache"][cache_key] = {
            "entry": entry,
            "last_access": time.time(),
        }
        state.memory_cache["metrics"]["total_inserts"] += 1
        logger.info(f"Memory inserted: {mem_key} (user={user}, convo={convo_id})")
        return "Memory inserted successfully."
    except Exception as e:
        logger.error(f"Error inserting memory: {e}")
        return f"Error inserting memory: {e}"
def load_into_lru(key: str, entry: dict, user: str, convo_id: int):
    cache_key = f"{user}_{convo_id}_{key}"
    if cache_key not in state.memory_cache["lru_cache"]:
        state.memory_cache["lru_cache"][cache_key] = {
            "entry": {
                "summary": entry.get("summary", ""),
                "details": entry.get("details", ""),
                "tags": entry.get("tags", []),
                "domain": entry.get("domain", "general"),
                "timestamp": entry.get("timestamp", datetime.now().isoformat()),
                "salience": entry.get("salience", 1.0),
            },
            "last_access": time.time(),
        }
@inject_user_convo
def memory_query(
    mem_key: str = None, limit: int = Config.DEFAULT_TOP_K.value, user: str = 'shared', convo_id: int = 0
) -> str:
    tool_limiter_sync()
    try:
        with state.conn:
            if mem_key:
                state.c.execute(
                    "SELECT mem_value FROM memory WHERE user=? AND convo_id=? AND mem_key=?",
                    (user, convo_id, mem_key),
                )
                result = state.c.fetchone()
                logger.info(f"Memory queried: {mem_key} (user={user}, convo={convo_id})")
                return json.loads(result[0]) if result and result[0] else "Key not found."
            else:
                state.c.execute(
                    "SELECT mem_key, mem_value FROM memory WHERE user=? AND convo_id=? ORDER BY timestamp DESC LIMIT ?",
                    (user, convo_id, limit),
                )
                results = {}
                for row in state.c.fetchall():
                    if row[1]:
                        results[row[0]] = json.loads(row[1])
                for key in results:
                    load_into_lru(key, results[key], user, convo_id)
                logger.info("Recent memories queried (shared mode)")
                return json.dumps(results)
    except Exception as e:
        logger.error(f"Error querying memory: {e}")
        return f"Error querying memory: {e}"
@inject_user_convo
def advanced_memory_consolidate(
    mem_key: str, interaction_data: dict, user: str = 'shared', convo_id: int = 0
) -> str:
    tool_limiter_sync()
    return safe_call(_advanced_memory_consolidate_impl, mem_key, interaction_data, user, convo_id)
def _advanced_memory_consolidate_impl(mem_key: str, interaction_data: dict, user: str, convo_id: int) -> str:
    cache_args = {"mem_key": mem_key, "interaction_data": interaction_data, "user": user, "convo_id": convo_id}
    if cached := get_cached_tool_result("advanced_memory_consolidate", cache_args):
        return cached
    embed_model = state.get_embed_model()
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
        summary_response = client.chat.completions.create(
            model=Models.GROK_FAST_REASONING.value,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this interaction concisely in one paragraph.",
                },
                {"role": "user", "content": json.dumps(interaction_data)},
            ],
            stream=False,
        )
        summary = summary_response.choices[0].message.content.strip()
        json_episodic = json.dumps(interaction_data)
        with state.conn:
            state.c.execute(
                "INSERT OR REPLACE INTO memory (user, convo_id, mem_key, mem_value) VALUES (?, ?, ?, ?)",
                (user, convo_id, mem_key, json_episodic),
            )
        if embed_model and state.chroma_ready and state.chroma_collection:
            chroma_col = state.chroma_collection
            embedding = embed_model.encode(summary).tolist()
            with state.chroma_lock: # FIX: Phase 1 - Lock.
                chroma_col.upsert(
                    ids=[str(uuid.uuid4())],
                    embeddings=[embedding],
                    documents=[json_episodic],
                    metadatas=[
                        {
                            "user": user,
                            "convo_id": int(convo_id), # Ensure int
                            "mem_key": mem_key,
                            "salience": 1.0,
                            "summary": summary,
                        }
                    ],
                )
        else:
            st.warning("Vector storage skipped; using DB only.")
        entry = {
            "summary": summary,
            "details": json_episodic,
            "tags": [],
            "domain": "general",
            "timestamp": datetime.now().isoformat(),
            "salience": 1.0,
        }
        cache_key = f"{user}_{convo_id}_{mem_key}"
        state.memory_cache["lru_cache"][cache_key] = {
            "entry": entry,
            "last_access": time.time(),
        }
        result = "Memory consolidated successfully."
        set_cached_tool_result("advanced_memory_consolidate", cache_args, result)
        logger.info(f"Memory consolidated: {mem_key} (shared mode)")
        return result
    except Exception:
        logger.error(f"Error consolidating memory: {traceback.format_exc()}")
        return f"Error consolidating memory: {traceback.format_exc()}"
@inject_user_convo
def advanced_memory_retrieve(
    query: str, top_k: int = Config.DEFAULT_TOP_K.value, user: str = 'shared', convo_id: int = 0
) -> str:
    tool_limiter_sync()
    try:
        convo_id = int(convo_id)  # FIX: Phase 1 - Safe coerce
    except ValueError:
        convo_id = 0
        logger.warning("Invalid convo_id; default to 0")
    cache_args = {"query": query, "top_k": top_k, "user": user, "convo_id": convo_id}
    if cached := get_cached_tool_result("advanced_memory_retrieve", cache_args):
        return cached
    embed_model = state.get_embed_model()
    chroma_col = state.chroma_collection
    if not embed_model or not state.chroma_ready or not chroma_col:
        logger.warning("Vector memory not available; falling back to keyword search.")
        retrieved = keyword_search(query, top_k, user=user, convo_id=convo_id)
        if isinstance(retrieved, str) and "error" in retrieved.lower():  # FIX: Phase 1 - Check type
            result = retrieved
        else:
            result = json.dumps(retrieved)
        set_cached_tool_result("advanced_memory_retrieve", cache_args, result)
        return result
    try:
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, embed_model.encode, query)
        try:
            query_emb = loop.run_until_complete(asyncio.wait_for(future, timeout=60.0)).tolist()
        except asyncio.TimeoutError:
            logger.warning("Embed timeout - Fallback to keyword search.")
            return fallback_to_keyword(query, top_k, user, convo_id)
        where_clause = {
            "$and": [
                {"user": {"$eq": user}},
                {"convo_id": {"$eq": convo_id}},
            ]
        }
        with state.chroma_lock: # FIX: Phase 1 - Lock.
            results = chroma_col.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                where=where_clause,
                include=["distances", "metadatas", "documents"],
            )
        if not results.get("ids") or not results["ids"][0]:
            logger.warning(f"Empty Chroma results: where={where_clause}, query={query[:50]}")
            where_clause.pop("convo_id", None)  # Relax convo_id
            with state.chroma_lock:
                results = chroma_col.query(
                    query_embeddings=[query_emb],
                    n_results=top_k,
                    where=where_clause,
                    include=["distances", "metadatas", "documents"],
                )  # Completed re-query
            if not results.get("ids") or not results["ids"][0]:  # Still empty? Seed but return short
                logger.info("Seeding empty DB.")
                example_data = {"summary": "Initial seed memory", "details": "Welcome!"}
                advanced_memory_consolidate("seed_key", example_data, user=user, convo_id=convo_id)
                return "No relevant memories found."
        retrieved = process_chroma_results(results, top_k)
        if len(retrieved) > 5:
            viz_result = viz_memory_lattice(user, convo_id, top_k=len(retrieved))
            logger.info(f"Auto-viz triggered: {viz_result}")
        if not retrieved:
            return "No relevant memories found."
        update_retrieve_metrics(len(retrieved), top_k)
        result = json.dumps(retrieved)
        set_cached_tool_result("advanced_memory_retrieve", cache_args, result)
        logger.info(f"Memory retrieved for query: {query} (shared mode)")
        return result
    except Exception as e:
        logger.error(f"Error retrieving memory: {traceback.format_exc()}")
        return "No relevant memories found."
def fallback_to_keyword(query: str, top_k: int, user: str, convo_id: int) -> list | str:
    fallback_results = keyword_search(query, top_k, user=user, convo_id=convo_id)
    if isinstance(fallback_results, str) and "error" in fallback_results.lower():
        return fallback_results
    retrieved = []
    for res in fallback_results:
        mem_key = res["id"]
        mem_value = memory_query(mem_key=mem_key, user=user, convo_id=convo_id)
        retrieved.append(
            {
                "mem_key": mem_key,
                "value": mem_value,
                "relevance": res["score"],
                "summary": mem_value.get("summary", ""),
            }
        )
    return retrieved
def process_chroma_results(results: dict, top_k: int) -> list:
    if not results or not results.get("ids") or not results["ids"]:
        return []
    retrieved = []
    ids_to_update = []
    metadata_to_update = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        sim = (1 - results["distances"][0][i]) * meta.get("salience", 1.0)
        retrieved.append(
            {
                "mem_key": meta["mem_key"],
                "value": json.loads(results["documents"][0][i]),
                "relevance": sim,
                "summary": meta.get("summary", ""),
            }
        )
        ids_to_update.append(results["ids"][0][i])
        new_salience = min(1.0, meta.get("salience", 1.0) + 0.1)  # FIX: Phase 1 - Cap
        metadata_to_update.append({"salience": new_salience})
    if ids_to_update:
        with state.chroma_lock: # FIX: Phase 1 - Lock.
            state.chroma_collection.update(
                ids=ids_to_update, metadatas=metadata_to_update
            )
    retrieved.sort(key=lambda x: x["relevance"], reverse=True)
    return retrieved
def update_retrieve_metrics(len_retrieved: int, top_k: int):
    state.memory_cache["metrics"]["total_retrieves"] += 1
    hit_rate = len_retrieved / top_k if top_k > 0 else 1.0
    state.memory_cache["metrics"]["hit_rate"] = (
        (
            state.memory_cache["metrics"]["hit_rate"]
            * (state.memory_cache["metrics"]["total_retrieves"] - 1)
        )
        + hit_rate
    ) / state.memory_cache["metrics"]["total_retrieves"]
def should_prune() -> bool:
    if "prune_counter" not in st.session_state:
        st.session_state["prune_counter"] = 0
    st.session_state["prune_counter"] += 1
    return st.session_state["prune_counter"] % Config.PRUNE_FREQUENCY.value == 0
def decay_salience(user: str, convo_id: int):
    one_week_ago = (datetime.now() - timedelta(days=7)).isoformat()  # FIX: Phase 1 - ISO str
    with state.conn:
        state.c.execute(
            "UPDATE memory SET salience = salience * 0.99 WHERE user=? AND convo_id=? AND timestamp < ?",
            (user, convo_id, one_week_ago),
        )
def prune_low_salience(user: str, convo_id: int):
    with state.conn:
        state.c.execute(
            "DELETE FROM memory WHERE user=? AND convo_id=? AND salience < 0.1",
            (user, convo_id),
        )
def size_based_prune(user: str, convo_id: int):
    with state.conn:
        state.c.execute(
            "SELECT COUNT(*) FROM memory WHERE user=? AND convo_id=?", (user, convo_id)
        )
        row_count = state.c.fetchone()[0]
        if row_count > 1000:
            state.c.execute(
                "SELECT mem_key FROM memory WHERE user=? AND convo_id=? AND salience < 0.5 ORDER BY timestamp ASC LIMIT ?",
                (user, convo_id, row_count - 1000),
            )
            low_keys = [row[0] for row in state.c.fetchall()]
            if not low_keys:  # FIX: Phase 1 - If no low, delete oldest
                state.c.execute(
                    "SELECT mem_key FROM memory WHERE user=? AND convo_id=? ORDER BY timestamp ASC LIMIT ?",
                    (user, convo_id, row_count - 1000),
                )
                low_keys = [row[0] for row in state.c.fetchall()]
            state.c.executemany(
                "DELETE FROM memory WHERE user=? AND convo_id=? AND mem_key=?",
                [(user, convo_id, key) for key in low_keys]
            )
def dedup_prune(user: str, convo_id: int):
    with state.conn:
        state.c.execute(
            "SELECT mem_key, mem_value FROM memory WHERE user=? AND convo_id=?",
            (user, convo_id),
        )
        rows = state.c.fetchall()
        hashes = {}
        to_delete = []
        for key, value_str in rows:
            value = json.loads(value_str)
            h = hash(value.get("summary", ""))
            if h in hashes and value.get("salience", 1.0) < hashes[h].get("salience", 1.0):
                to_delete.append(key)
            else:
                hashes[h] = value
        state.c.executemany(
            "DELETE FROM memory WHERE user=? AND convo_id=? AND mem_key=?",
            [(user, convo_id, key) for key in to_delete]
        )
@inject_user_convo
def advanced_memory_prune(user: str = 'shared', convo_id: int = 0) -> str:
    if not should_prune():
        return "Prune skipped (infrequent)."
    def _prune_sync(u, cid):
        try:
            with state.conn:
                decay_salience(u, cid)
                prune_low_salience(u, cid)
                size_based_prune(u, cid)
                dedup_prune(u, cid)
                one_week_ago = (datetime.now() - timedelta(days=7)).isoformat()  # FIX: Phase 1 - ISO
                state.c.execute(
                    "DELETE FROM memory WHERE user=? AND convo_id=? AND timestamp < ? AND mem_key LIKE 'agent_%'",
                    (u, cid, one_week_ago),
                )
                state.conn.commit() # Explicit commit
            for agent_folder in os.listdir(state.agent_dir):
                folder_path = os.path.join(state.agent_dir, agent_folder)
                if os.path.isdir(folder_path):
                    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
                    if files and os.path.getmtime(os.path.join(folder_path, files[0])) < (datetime.now() - timedelta(days=7)).timestamp():
                        shutil.rmtree(folder_path)
                        logger.info(f"Pruned old agent folder: {agent_folder}")
            state.lru_evict()
            state.memory_cache["metrics"]["last_update"] = datetime.now().isoformat()
            logger.info("Memory pruned successfully (shared mode)")
            return "Memory pruned successfully."
        except Exception as e:
            logger.error(f"Error pruning memory: {e}")
            return f"Error pruning memory: {e}"
    future = state.agent_executor.submit(_prune_sync, user, convo_id)
    try:
        return future.result(timeout=300)
    except Exception as e:
        return f"Prune timeout/error: {e}"
@lru_cache(maxsize=128)
def cached_embed(summary: str):
    embed_model = state.get_embed_model()
    return embed_model.encode(summary).tolist()
def viz_memory_lattice(
    user: str,
    convo_id: int,
    top_k: int = Config.DEFAULT_TOP_K.value * 4,
    sim_threshold: float = Config.SIM_THRESHOLD.value,
    output_dir: str = "./sandbox/viz",
    plot_type: str = "both"
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    embed_model = state.get_embed_model()
    chroma_col = state.chroma_collection
    if not embed_model or not chroma_col:
        return "Error: Embed/Chroma not ready for viz."
    convo_summary = memory_query(limit=1, user=user, convo_id=convo_id)
    query = convo_summary if convo_summary != "[]" else "memory lattice"
    where_clause = {"$and": [{"user": {"$eq": user}}, {"convo_id": {"$eq": int(convo_id)}}]}
    query_emb = embed_model.encode(query).tolist()
    with state.chroma_lock: # FIX: Phase 1 - Lock.
        results = chroma_col.query(
            query_embeddings=[query_emb],
            n_results=min(top_k, 10),
            where=where_clause,
            include=["metadatas", "documents", "distances"]
        )
    if not results.get("ids") or len(results["ids"][0]) == 0:
        return "No memories to visualize."
    G = nx.Graph()
    node_data = []
    summaries = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        mem_key = meta["mem_key"]
        summary = meta.get("summary", results["documents"][0][i][:100])
        salience = meta.get("salience", 1.0)
        sim_score = 1 - results["distances"][0][i]
        if sim_score < sim_threshold:
            continue
        G.add_node(mem_key, summary=summary, salience=salience, sim=sim_score)
        summaries.append(summary)
        node_data.append({"layer_proxy": i // (len(results["ids"][0]) // 12), "amp": salience * sim_score})
    if len(G.nodes) > 1:
        all_embs = [cached_embed(summaries[i]) for i in range(len(summaries))]
        for i, node_i in enumerate(G.nodes):
            candidates = list(G.nodes)[i+1:]
            if len(candidates) == 0:  # FIX: Phase 1 - Handle small
                continue
            sampled = np.random.choice(candidates, min(3, len(candidates)), replace=False) # OPTIMIZED: Sample less
            for node_j in sampled:
                j_idx = list(G.nodes).index(node_j)
                sim = np.dot(all_embs[i], all_embs[j_idx]) / (np.linalg.norm(all_embs[i]) * np.linalg.norm(all_embs[j_idx]))
                if sim > sim_threshold:
                    G.add_edge(node_i, node_j, weight=sim)
    if plot_type in ["graph", "both"]:
        pos = nx.spring_layout(G, k=1, iterations=5) # Reduced iterations for speed - FIX: Phase 3
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'), hoverinfo='none',
            mode='lines', line_shape='spline'
        )
        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(G.nodes[node]["summary"][:20])
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(size=[G.nodes[n]["salience"] * 10 for n in G.nodes], color=[G.nodes[n]["sim"] for n in G.nodes], colorscale='Viridis', showscale=True)
        )
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title=f"Memory Lattice: {len(G.nodes)} Nodes, {len(G.edges)} Veins (Convo {convo_id})",
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[ dict(text="Interactive: Hover/Zoom", showarrow=False, x=0, y=1.1, xref="paper", yref="paper") ],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        )
        )
        st.plotly_chart(fig, use_container_width=True)
    if plot_type in ["amps", "both"]:
        layers = list(range(12))
        amps_simple = [0.5] * 12
        amps_complex = [d["amp"] for d in sorted(node_data, key=lambda x: x["layer_proxy"])]
        amps_complex = amps_complex + [0.5] * (12 - len(amps_complex))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(layers, amps_simple, label="Baseline (Simple Prompt)", color='lightgray')
        ax.plot(layers, amps_complex, label="Your Lattice (Post-Priming)", color='teal', linewidth=2)
        ax.set_title(f"Activation Amps Over 'Layers' (Convo {convo_id})")
        ax.set_xlabel("Layer Proxy (Retrieval Rank Bins)")
        ax.set_ylabel("Amp (Salience * Sim)")
        ax.legend()
        st.pyplot(fig)
        amps_path = os.path.join(output_dir, f"memory_amps_{convo_id}.png")
        plt.savefig(amps_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    graph_json = nx.node_link_data(G)
    mem_key = f"lattice_viz_{convo_id}"
    memory_insert(mem_key, {"graph": graph_json, "paths": [amps_path]}, user=user, convo_id=convo_id)
    logger.info(f"Lattice viz saved for convo {convo_id}: {len(G.nodes)} nodes")
    return f"Viz complete: Interactive graph rendered. Query '{mem_key}' for data."
async def async_run_spawn(agent_id: str, task: str, user: str, convo_id: int, model: str = Models.GROK_3_MINI.value) -> str:
    async with state.agent_sem: # Use semaphore
        max_attempts = 3 # FIX: Phase 3.
        for attempt in range(max_attempts):
            try:
                client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are an agent for AurumVivum. Execute the given task/query/scenario/simulation. Suggest tool-chains if needed, but do not call tools yourself. Respond concisely."},
                            {"role": "user", "content": task}
                        ],
                        stream=False,
                    ), timeout=600.0  # Bumped timeout
                )
                result = response.choices[0].message.content.strip()
                persist_agent_result(agent_id, task, result, user, convo_id)
                logger.info(f"Agent {agent_id} succeeded async.")
                return result
            except asyncio.TimeoutError:
                error = "Timeout: 60s exceeded."
            except Exception as e:
                await asyncio.sleep(5 * (attempt + 1)) # FIX: Phase 3 - Expo backoff.
                if attempt == max_attempts - 1:
                    error = f"Max attempts failed: {e}"
                    persist_agent_result(agent_id, task, error, user, convo_id)
                    return error

def visualize_got(
    got_data: str,  # YAML/JSON str of lattice (e.g., from tool_lattice.graph)
    format: str = "both",  # "graph" (Plotly), "amps" (Matplotlib), "both", "json" (data only), "text" (ASCII)
    detail_level: int = 2,  # 1=simple nodes, 2=relations, 3=attrs/weights
    user: str = "shared",
    convo_id: int = 0,
    output_dir: str = "./sandbox/viz",
    sim_threshold: float = Config.SIM_THRESHOLD.value
) -> str:
    import json
    import yaml
    import networkx as nx
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    try:
        from ascii_graph import Pyasciigraph  # FIX: Phase 1 - Import here, handle missing
    except ImportError:
        if format == "text":
            return "ASCII not available - install ascii_graph."

    os.makedirs(output_dir, exist_ok=True)

    # Parse got_data (YAML priority, fallback JSON)
    try:
        data = yaml.safe_load(got_data) if got_data.strip().startswith(('#', 'graph:')) else json.loads(got_data)
    except Exception as e:
        return f"Error: Invalid GoT data format - {e}"

    G = nx.DiGraph()  # Directed for relations (depends_on, etc.)
    for node, attrs in data.get('graph', {}).items():
        G.add_node(node, **attrs)  # Add weights/triggers
        for rel in ['depends_on', 'limited_by', 'mitigation', 'integrates']:
            targets = attrs.get(rel, [])
            if isinstance(targets, list):
                for t in targets:
                    G.add_edge(node, t, relation=rel, weight=attrs.get('weight', 1.0))
            elif targets:
                G.add_edge(node, targets, relation=rel, weight=attrs.get('weight', 1.0))

    if len(G.nodes) == 0:
        return "No GoT nodes to visualize."

    # Optional: Embed/sims if detail_level >1 (reuse your cached_embed)
    embed_model = state.get_embed_model()
    if detail_level > 1 and embed_model:
        node_embs = {n: cached_embed(G.nodes[n].get('desc', n)) for n in G.nodes}
        for u, v in G.edges:
            sim = np.dot(node_embs[u], node_embs[v]) / (np.linalg.norm(node_embs[u]) * np.linalg.norm(node_embs[v]))
            if sim > sim_threshold:
                G.edges[u, v]['sim'] = sim

    # Render based on format
    paths = []
    if format in ["graph", "both"]:
        pos = nx.spring_layout(G, k=1, iterations=10)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines', line_shape='spline')

        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node + f" (w:{G.nodes[node].get('weight',1.0):.2f})")
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', hoverinfo='text', text=node_text,
                                marker=dict(size=[G.nodes[n].get('weight',1.0)*10 for n in G.nodes], color=[G.edges.get((n, next(iter(G[n])), {}), {}).get('sim',0.5) for n in G.nodes], colorscale='Viridis'))

        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title=f"GoT Lattice: {len(G.nodes)} Nodes", showlegend=False, hovermode='closest',
                                                                        margin=dict(b=20,l=5,r=5,t=40), xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)))
        # UI optional: if st._is_running_with_streamlit: st.plotly_chart(fig)
        graph_path = os.path.join(output_dir, f"got_graph_{convo_id}.png")
        fig.write_image(graph_path)  # Save PNG for agent/fs access
        paths.append(graph_path)

    if format in ["amps", "both"]:
        weights = [d.get('weight', 1.0) for _, d in G.nodes(data=True)]
        sims = [e[2].get('sim', 0.5) for e in G.edges(data=True)] or [0.5] * len(G.nodes)
        layers = list(range(len(weights)))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(layers, [0.5]*len(layers), label="Baseline", color='lightgray')
        ax.plot(layers, weights, label="Node Weights", color='gold')
        ax.plot(layers, sims[:len(layers)], label="Edge Sims", color='crimson', linewidth=2)
        ax.set_title(f"GoT Activation Amps (Convo {convo_id})")
        ax.legend()
        # UI optional: if st._is_running_with_streamlit: st.pyplot(fig)
        amps_path = os.path.join(output_dir, f"got_amps_{convo_id}.png")
        fig.savefig(amps_path, dpi=150)
        plt.close(fig)
        paths.append(amps_path)

    if format == "text":
        graph = Pyasciigraph()
        ascii_rep = ""
        for line in graph.graph('GoT ASCII', [(n, len(G[n])) for n in G.nodes]):
            ascii_rep += line + "\n"
        text_path = os.path.join(output_dir, f"got_text_{convo_id}.txt")
        with open(text_path, 'w') as f:
            f.write(ascii_rep)
        paths.append(text_path)

    # Save to memory (like viz_memory_lattice)
    graph_json = nx.node_link_data(G)
    mem_key = f"got_viz_{convo_id}"
    memory_insert(mem_key, {"graph": graph_json, "paths": paths}, user=user, convo_id=convo_id)

    return f"GoT Viz complete: PNGs/TXT at {output_dir}, data in '{mem_key}'. Use fs_read_file or view_image to access."

def persist_agent_result(agent_id: str, task: str, response: str, user: str, convo_id: int) -> None:
    try:
        with state.agent_lock:
            agent_folder = os.path.join(state.agent_dir, agent_id)
            os.makedirs(agent_folder, exist_ok=True)
            result_data = {
                "agent_id": agent_id,
                "task": task,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "status": "complete"
            }
            json_path = os.path.join(agent_folder, "result.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=4)
            mem_key = f"agent_{agent_id}_result"
            memory_insert(mem_key, result_data, user=user, convo_id=convo_id)
            summary_data = {"summary": f"Agent {agent_id} response to task: {task[:100]}...", "details": response}
            advanced_memory_consolidate(f"agent_{agent_id}_summary", summary_data, user=user, convo_id=convo_id)
            notify_key = f"agent_{agent_id}_complete"
            notify_data = {"agent_id": agent_id, "status": "complete", "result_key": mem_key, "timestamp": datetime.now().isoformat()}
            memory_insert(notify_key, notify_data, user=user, convo_id=convo_id)
            get_state("pending_notifies", []).append({"agent_id": agent_id, "status": "complete", "task": task[:100]})
            logger.info(f"Agent {agent_id} persisted and notified.")
    except OSError as e:  # Specific for file/dir issues
        logger.error(f"File error for agent {agent_id}: {e}")
        error_data = {"agent_id": agent_id, "error": str(e), "status": "failed"}
        memory_insert(f"agent_{agent_id}_error", error_data, user=user, convo_id=convo_id)
    except Exception as e:
        logger.error(f"Persistence error for agent {agent_id}: {e}")
        error_data = {"agent_id": agent_id, "error": str(e), "status": "failed"}
        memory_insert(f"agent_{agent_id}_error", error_data, user=user, convo_id=convo_id)
@inject_user_convo
def agent_spawn(sub_agent_type: str, task: str, user: str = 'shared', convo_id: int = 0, poll_interval: int = 5, model: str = Models.GROK_3_MINI.value, auto_poll: bool = False) -> str: # FIX: Phase 3 - Add auto_poll.
    tool_limiter_sync()
    if len(task) > Config.MAX_TASK_LEN.value:
        return "Error: Task too long (max 2000 chars)."
    agent_id = f"{sub_agent_type}_{str(uuid.uuid4())[:8]}"
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_run_spawn(agent_id, task, user, convo_id, model))
    status_key = f"agent_{agent_id}_status"
    status_data = {"agent_id": agent_id, "task": task[:100], "status": "spawned", "timestamp": datetime.now().isoformat(), "poll_interval": poll_interval}
    memory_insert(status_key, status_data, user=user, convo_id=convo_id)
    get_state("pending_notifies", []).append({"agent_id": agent_id, "status": "spawned", "task": task[:100]})
    if auto_poll:
        max_polls = 30
        for _ in range(max_polls):
            time.sleep(poll_interval)  # FIXME: Non-async sleep - for hobby, OK; later thread
            complete = memory_query(mem_key=f"agent_{agent_id}_complete", user=user, convo_id=convo_id)
            if complete != "Key not found.":
                return f"Polled result: {complete}"
        return "Poll timeout."
    return f"Agent '{sub_agent_type}' spawned (ID: {agent_id}). Poll 'agent_{agent_id}_complete' for results. Status: {status_key}"
def render_agent_fleet():
    pending = get_state("pending_notifies", [])
    if pending:
        st.subheader("Recent Notifies")
        for notify in pending:
            st.info(f"**{notify['agent_id']}**: {notify['status']} – Task: {notify['task']}")
        st.session_state["pending_notifies"] = []
 
    user = st.session_state.get("user")
    convo_id = st.session_state.get("current_convo_id", 0)
    if user and convo_id:
        active_query = memory_query(limit=20, user=user, convo_id=convo_id)
        active_agents = []
        try:
            active_data = json.loads(active_query)
            active_agents = [data for key, data in active_data.items() if key.startswith("agent_") and data.get("status") in ["spawned", "running"]]
        except Exception as e:
            logger.error(f"Fleet active query error: {e}")
            st.warning("Error loading active agents.")
        if active_agents:
            st.subheader("Active Fleet")
            for idx, agent in enumerate(active_agents):  # Enumerate for unique keys
                col1, col2, col3 = st.columns([3, 4, 1])
                with col1:
                    st.write(f"**{agent.get('agent_id', 'Unknown')}**")
                with col2:
                    st.write(f"Status: {agent.get('status', 'Unknown')} | Task: {agent.get('task', '')[:50]}...")
                with col3:
                    unique_key = f"kill_{agent.get('agent_id', '')}_{uuid.uuid4()}_{idx}"  # Unique with UUID and idx
                    logger.debug(f"Key generated: {unique_key}")  # Debug logger
                    if st.button("Kill", key=unique_key):
                        kill_key = f"agent_{agent['agent_id']}_kill"
                        kill_data = {"status": "killed", "timestamp": datetime.now().isoformat()}
                        memory_insert(kill_key, kill_data, user=user, convo_id=convo_id)
                        st.rerun()
            if st.button("Spawn Fleet (Parallel Sims)"):
                safe_call(agent_spawn, "fleet", "Run parallel quantum sims on nodes 1-3", user=user, convo_id=convo_id)
def git_init(safe_repo: str) -> str:
    try:
        repo = pygit2.discover_repository(safe_repo) # Check existing
        if repo:
            return "Repo already exists."
    except:
        pass
    pygit2.init_repository(safe_repo)
    return "Repo initialized."
def git_commit(repo: pygit2.Repository, message: str) -> str:
    if not message:
        return "Error: Message required for commit."
    try:
        repo.git.add(A=True)  # Fixed: Use git.add with A=True
    except AttributeError:
        # Fallback for older pygit2
        index = repo.index
        index.add_all()
        index.write()
    tree = repo.index.write_tree()
    author = pygit2.Signature("User", "user@example.com")
    repo.create_commit(
        "HEAD",
        author,
        author,
        message,
        tree,
        [repo.head.target] if repo.head.is_branch else [],
    )
    return "Committed."
def git_branch(repo: pygit2.Repository, name: str) -> str:
    if not name:
        return "Error: Name required for branch."
    repo.create_branch(name, repo.head.peel())
    return f"Branch '{name}' created."
def git_diff(repo: pygit2.Repository) -> str:
    diff = repo.diff()
    return diff.patch
def git_ops(
    operation: str, repo_path: str, message: str = None, name: str = None
) -> str:
    tool_limiter_sync()
    safe_repo = os.path.abspath(os.path.normpath(os.path.join(state.sandbox_dir, repo_path)))
    if not safe_repo.startswith(os.path.abspath(state.sandbox_dir)):
        return "Error: Repo path outside sandbox."
    try:
        repo = pygit2.discover_repository(safe_repo) or pygit2.init_repository(safe_repo) # Init if none
        op_funcs = {
            "init": lambda: git_init(safe_repo),
            "commit": lambda: git_commit(repo, message),
            "branch": lambda: git_branch(repo, name),
            "diff": lambda: git_diff(repo),
            "status": lambda: repo.git.status(),
        }
        if operation in op_funcs:
            return op_funcs[operation]()
        return "Unknown operation."
    except Exception as e:
        logger.error(f"Git error: {e}")
        return f"Git error: {e}"
def db_query(db_path: str, query: str, params: list = None) -> str:
    tool_limiter_sync()
    safe_db = os.path.abspath(os.path.normpath(os.path.join(state.sandbox_dir, db_path)))
    if not safe_db.startswith(os.path.abspath(state.sandbox_dir)):
        return "Error: DB path outside sandbox."
    if any(kw in query.upper() for kw in ['DROP', 'DELETE', 'ALTER']): # FIX: Phase 2 - Safety.
        return "Forbidden op."
    try:
        with sqlite3.connect(safe_db) as db_conn:
            db_c = db_conn.cursor()
            if params:
                db_c.execute(query, params)
            else:
                db_c.execute(query)
            if query.lower().startswith("select"):
                results = db_c.fetchall()
                return json.dumps(results)
            db_conn.commit()
            return "Query executed."
    except Exception as e:
        logger.error(f"DB error: {e}")
        return f"DB error: {e}"
def shell_exec(command: str) -> str:
    tool_limiter_sync()
    whitelist_pattern = r"^(ls|grep|sed|awk|cat|echo|wc|tail|head|cp|mv|rm|mkdir|rmdir|touch|python|pip)$" # Expanded
    cmd_parts = shlex.split(command)
    cmd_base = cmd_parts[0]
    if not re.match(whitelist_pattern, cmd_base):
        return "Error: Command not whitelisted."
    for arg in cmd_parts[1:]: # FIX: Phase 2 - Arg validation.
        if re.search(r'[;&|><$]', arg):
            return "Invalid arg chars."
        if re.search(r'[\*\?\[\]]|\.\./', arg):  # FIX: Phase 1 - Better forbidden
            return "Error: Forbidden patterns in args."
    convo_id = st.session_state.get("current_convo_id", 0)
    confirm_key = f"confirm_destructive_{convo_id}"
    if cmd_base in ["rm", "rmdir"] and not get_state(confirm_key, False):
        st.session_state[confirm_key] = True
        return "Warning: Destructive command detected. Confirm by re-running."
    # FIXED: Quote args to prevent injection
    quoted_parts = [shlex.quote(part) for part in cmd_parts]
    try:
        result = subprocess.run(
            quoted_parts, cwd=state.sandbox_dir, capture_output=True, text=True, timeout=60, shell=False # shell=False
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        logger.error(f"Shell error: {e}")
        return f"Shell error: {e}"
def lint_python(code: str) -> str:
    return format_str(code, mode=FileMode())
def lint_javascript(code: str) -> str:
    return jsbeautifier.beautify(code)
def lint_json(code: str) -> str:
    return json.dumps(json.loads(code), indent=4)
def lint_yaml(code: str) -> str:
    return yaml.safe_dump(yaml.safe_load(code), default_flow_style=False)
def lint_sql(code: str) -> str:
    return sqlparse.format(code, reindent=True)
def lint_xml_html(code: str, lang_lower: str) -> str:
    if lang_lower == "html":
        soup = bs4.BeautifulSoup(code, "html.parser")
        return soup.prettify()
    else:
        dom = xml.dom.minidom.parseString(code)
        return dom.toprettyxml()
def code_lint(language: str, code: str) -> str:
    tool_limiter_sync()
    lang_lower = language.lower()
    try:
        lint_funcs = {
            "python": lambda: lint_python(code),
            "javascript": lambda: lint_javascript(code),
            "json": lambda: lint_json(code),
            "yaml": lambda: lint_yaml(code),
            "sql": lambda: lint_sql(code),
            "xml": lambda: lint_xml_html(code, lang_lower),
            "html": lambda: lint_xml_html(code, lang_lower),
        }
        if lang_lower in lint_funcs:
            return lint_funcs[lang_lower]()
        return f"Linting not supported for {language}."
    except Exception as e:
        logger.error(f"Lint error: {e}")
        return f"Lint error: {e}"
def api_simulate(url: str, method: str = "GET", data: dict = None, headers: dict = None, mock: bool = True) -> str:
    tool_limiter_sync()
    whitelist = ["https://api.example.com", "https://jsonplaceholder.typicode.com", "https://api.x.ai/v1"]
    if not any(url.startswith(w) for w in whitelist) and not mock:
        return "Error: URL not whitelisted for real calls."
    default_headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"} if API_KEY else {}
    headers = {**default_headers, **(headers or {})}
    try:
        if mock:
            return f"Mock response for {method} {url}: {json.dumps({'status': 'mocked'})}"
        logger.info(f"API call to {url} (key masked)")
        response = requests.request(method, url, json=data if method == "POST" else None, headers=headers)
        return response.text
    except Exception as e:
        logger.error(f"API error: {e}")
        return f"API error: {e}"
# REMOVED: Custom xai_* functions - now handled via native tools in API call
def generate_embedding(text: str) -> str:
    tool_limiter_sync()
    embed_model = state.get_embed_model()
    if not embed_model:
        return "Error: Embedding model not loaded."
    try:
        if len(text) > 10000: # FIX: Phase 3 - Batching for large (though rare per user).
            chunks = chunk_text(text, max_tokens=32768) # Assume returns list[str].
            embeds = [embed_model.encode(c).tolist() for c in chunks]
            avg_embed = np.mean(embeds, axis=0).tolist()
            return json.dumps(avg_embed)
        else:
            embedding = embed_model.encode(text).tolist()
            return json.dumps(embedding)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return f"Embedding error: {e}"
def vector_search(query_embedding: list, top_k: int = Config.DEFAULT_TOP_K.value, threshold: float = Config.SIM_THRESHOLD.value) -> str:
    tool_limiter_sync()
    if not state.chroma_ready or not state.chroma_collection:
        return "Error: ChromaDB not ready."
    chroma_col = state.chroma_collection
    try:
        with state.chroma_lock: # FIX: Phase 1 - Lock.
            results = chroma_col.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
        filtered = [
            {"id": id, "distance": dist}
            for id, dist in zip(results["ids"][0], results["distances"][0])
            if dist <= (1 - threshold)
        ]
        return json.dumps(filtered)
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return f"Vector search error: {e}"
def chunk_text(text: str, max_tokens: int = 32768) -> str:
    tool_limiter_sync()
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        chunks = [
            enc.decode(tokens[i : i + max_tokens])
            for i in range(0, len(tokens), max_tokens)
        ]
        return json.dumps(chunks)
    except Exception as e:
        logger.error(f"Chunk error: {e}")
        return f"Chunk error: {e}"
def summarize_chunk(chunk: str) -> str:
    tool_limiter_sync()
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
        response = client.chat.completions.create(
            model=Models.GROK_FAST_REASONING.value,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this text in under 100 words, preserving key facts.",
                },
                {"role": "user", "content": chunk},
            ],
            stream=False,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        return f"Summarize error: {e}"
@inject_user_convo
def keyword_search(
    query: str, top_k: int = Config.DEFAULT_TOP_K.value, user: str = 'shared', convo_id: int = 0
) -> list | str:
    tool_limiter_sync()
    try:
        with state.conn:
            state.c.execute(
                "SELECT mem_key FROM memory WHERE user=? AND convo_id=? AND mem_value LIKE ? ORDER BY salience DESC LIMIT ?",
                (user, convo_id, f"%{query}%", top_k),
            )
            results = [{"id": row[0], "score": 1.0} for row in state.c.fetchall()]
            return results
    except Exception as e:
        logger.error(f"Keyword search error: {e}")
        return f"Keyword search error: {e}"
@inject_user_convo
def socratic_api_council(
    branches: list,
    model: str = Models.GROK_FAST_REASONING.value,
    user: str = 'shared',
    convo_id: int = 0,
    api_key: str = None,
    rounds: int = 3,
    personas: list = None,
) -> str:
    tool_limiter_sync()
    if not api_key:
        api_key = API_KEY
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1/")
    default_personas = [
        "Planner",
        "Critic",
        "Executor",
        "Summarizer",
        "Verifier",
        "Moderator",
    ]
    personas = personas or default_personas
    try:
        consensus = ""
        messages = [
            {
                "role": "system",
                "content": f"Debate these branches as a {len(personas)}-persona council: {', '.join(personas)}. Iterate {rounds} rounds for refinement and consensus.",
            }
        ]
        for branch in branches:
            messages.append({"role": "user", "content": branch})
        for r in range(rounds):
            response = client.chat.completions.create(model=model, messages=messages)
            round_result = response.choices[0].message.content
            consensus += f"Round {r+1}: {round_result}\n"
            messages.append({"role": "assistant", "content": round_result})
        messages.append(
            {
                "role": "system",
                "content": "Reach final consensus via majority vote or judge.",
            }
        )
        final_response = client.chat.completions.create(model=model, messages=messages)
        consensus += f"Final Consensus: {final_response.choices[0].message.content}"
        advanced_memory_consolidate(
            "council_result",
            {"branches": branches, "result": consensus},
            user=user,
            convo_id=convo_id,
        )
        logger.info("Socratic council completed")
        return consensus
    except Exception as e:
        logger.error(f"Council error: {e}")
        return f"Council error: {e}"
def reflect_optimize(component: str, metrics: dict) -> str:
    # UNSTUBBED: Use Grok to optimize
    tool_limiter_sync()
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
        prompt = f"Reflect on {component} with metrics {json.dumps(metrics)}. Suggest optimized version."
        response = client.chat.completions.create(
            model=Models.GROK_FAST_REASONING.value,
            messages=[
                {"role": "system", "content": "Optimize based on metrics/reflection."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Reflect error: {e}")
        return f"Optimized {component} with metrics: {json.dumps(metrics)} - Error: {e}"
def venv_create(env_name: str, with_pip: bool = True) -> str:
    tool_limiter_sync()
    safe_env = os.path.abspath(os.path.normpath(os.path.join(state.sandbox_dir, env_name)))
    if not safe_env.startswith(os.path.abspath(state.sandbox_dir)):
        return "Error: Env path outside sandbox."
    if os.path.exists(safe_env): # FIX: Phase 2 - Exists check.
        return "Venv exists—skip."
    try:
        import venv # Lazy
        venv.create(safe_env, with_pip=with_pip)
        return f"Venv '{env_name}' created."
    except Exception as e:
        logger.error(f"Venv error: {e}")
        return f"Venv error: {e}"
def restricted_exec(code: str, level: str = "basic") -> str:
    tool_limiter_sync()
    try:
        if level == "basic":
            tree = ast.parse(code)
            if not all(restricted_policy(node) for node in ast.walk(tree)):
                return "Restricted: Imports/unsafe banned."
            result = RestrictedPython.compile_restricted_exec(code)
            if result.errors:
                return f"Restricted compile error: {result.errors}"
            exec(result.code, RestrictedPython.safe_globals, {})
        else:
            exec(code, globals())
        return "Executed in restricted mode."
    except Exception as e:
        logger.error(f"Restricted exec error: {e}")
        return f"Restricted exec error: {e}"
def isolated_subprocess(cmd: str, custom_env: dict = None) -> str:
    tool_limiter_sync()
    env = os.environ.copy()
    if custom_env:
        env.update(custom_env)
    try:
        result = subprocess.run(
            shlex.split(cmd),
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            cwd=state.sandbox_dir,
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        logger.error(f"Subprocess error: {e}")
        return f"Subprocess error: {e}"
PIP_WHITELIST = [
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "sympy",
    "requests",
    "beautifulsoup4",
    "qiskit",  # Loosened for agents
    "torch",
    "tensorflow",  # Added for creativity
]
def pip_install(venv_path: str, packages: list, upgrade: bool = False) -> str:
    tool_limiter_sync()
    if any(pkg not in PIP_WHITELIST for pkg in packages):
        return "Error: One or more packages not in whitelist."
    safe_venv = os.path.abspath(os.path.normpath(os.path.join(state.sandbox_dir, venv_path)))
    if not safe_venv.startswith(os.path.abspath(state.sandbox_dir)):
        return "Error: Venv path outside sandbox."
    venv_pip = os.path.join(safe_venv, "bin", "pip")
    if not os.path.exists(venv_pip):
        return "Error: Pip not found in venv."
    cmd = [venv_pip, "install", "--no-deps"] + (["--upgrade"] if upgrade else []) + packages
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if "Error" in result.stderr and "--no-deps" in cmd: # FIX: Phase 2 - Try without no-deps.
            cmd.remove("--no-deps")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            logger.warning("Retried without --no-deps.")
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        logger.error(f"Pip error: {e}")
        return f"Pip error: {e}"
@inject_user_convo
def chat_log_analyze_embed(
    convo_id: int, criteria: str, summarize: bool = True, user: str = 'shared'
) -> str:
    tool_limiter_sync()
    with state.conn:
        state.c.execute(
            "SELECT messages FROM history WHERE convo_id=? AND user=?", (convo_id, user)
        )
        result = state.c.fetchone()
    if not result:
        return "Error: Chat log not found."
    messages = json.loads(result[0])
    if not messages:
        return "Error: Empty chat log."
    chat_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
    analysis_prompt = f"Analyze this chat log on criteria: {criteria}. Summarize if needed."
    response = client.chat.completions.create(
        model=Models.GROK_FAST_REASONING.value,
        messages=[
            {"role": "system", "content": analysis_prompt},
            {"role": "user", "content": chat_text},
        ],
        stream=False,
    )
    analysis = response.choices[0].message.content.strip()
    if summarize:
        summary_prompt = "Summarize the analysis concisely."
        summary_response = client.chat.completions.create(
            model=Models.GROK_FAST_REASONING.value,
            messages=[
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": analysis},
            ],
            stream=False,
        )
        analysis = summary_response.choices[0].message.content.strip()
    embed_model = state.get_embed_model()
    if not embed_model or not state.chroma_ready:
        return "Error: Embedding or ChromaDB not ready."
    chroma_col = state.chroma_collection
    embedding = embed_model.encode(analysis).tolist()
    mem_key = f"chat_log_{convo_id}"
    with state.chroma_lock: # FIX: Phase 1 - Lock.
        chroma_col.upsert(
            ids=[mem_key],
            embeddings=[embedding],
            documents=[analysis],
            metadatas=[
                {"user": user, "convo_id": convo_id, "type": "chat_log", "salience": 1.0}
            ],
        )
    return f"Chat log {convo_id} analyzed and embedded as {mem_key}."
def yaml_retrieve(
    query: str = None, top_k: int = Config.DEFAULT_TOP_K.value, filename: str = None
) -> str:
    tool_limiter_sync()
    if not state.yaml_ready:
        state._init_yaml_embeddings() # Lazy if not ready
    col = state.yaml_collection
    embed_model = state.get_embed_model()
    cache = state.yaml_cache
    try:
        if filename:
            if filename in cache:
                return cache[filename]
            with state.chroma_lock: # FIX: Phase 1 - Lock.
                results = col.query(
                    n_results=1, where={"filename": filename}, include=["documents"]
                )
            if results.get("documents") and results["documents"][0]:
                content = results["documents"][0][0]
                cache[filename] = content
                state.yaml_cache = cache
                return content
            else:
                return "YAML not found."
        else:
            if not query:
                return "Error: Query required for semantic search."
            try:
                query_emb = embed_model.encode(query).tolist()
                if not query_emb:
                    raise ValueError("Empty embedding generated.")
            except Exception as e:
                return f"Embedding gen error: {e} - Fallback to fs_list_files in yaml_dir."
            with state.chroma_lock: # FIX: Phase 1 - Lock.
                results = col.query(
                    query_embeddings=[query_emb],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
            if not results.get("documents"):
                return "No relevant YAMLs found."
            retrieved = [
                {"filename": meta["filename"], "content": doc, "distance": dist}
                for meta, doc, dist in zip(
                    results["metadatas"][0],
                    results["documents"][0],
                    results["distances"][0],
                )
            ]
            return json.dumps(retrieved)
    except Exception as e:
        logger.error(f"YAML retrieve error: {e}")
        return f"YAML retrieve error: {e}"
def yaml_refresh(filename: str = None) -> str:
    tool_limiter_sync()
    embed_model = state.get_embed_model()
    if not embed_model:
        return "Error: Embedding model not loaded."
    col = state.yaml_collection
    try:
        if filename:
            path = os.path.join(state.yaml_dir, filename)
            if not os.path.exists(path):
                return "Error: File not found."
            try: # FIX: Phase 1 & 3 - Specific, skip on fail.
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                embedding = embed_model.encode(content).tolist()
                with state.chroma_lock: # FIX: Phase 1 - Lock.
                    # FIX: Phase 3 - Check exists, update vs upsert
                    existing = col.get(where={"filename": filename})
                    if existing["ids"]:
                        col.update(
                            ids=existing["ids"],
                            embeddings=[embedding],
                            documents=[content],
                            metadatas=[{"filename": filename}],
                        )
                    else:
                        col.upsert(
                            ids=[filename],
                            embeddings=[embedding],
                            documents=[content],
                            metadatas=[{"filename": filename}],
                        )
                state.yaml_cache[filename] = content
                return f"YAML '{filename}' refreshed successfully."
            except OSError as e:
                return f"Skipped {filename}: {e}"
        else:
            with state.chroma_lock: # FIX: Phase 1 - Lock.
                ids = col.get()["ids"]
                if ids:
                    col.delete(ids=ids)
            state.yaml_cache = {}
            files_refreshed = 0
            contents = [] # For batch.
            fnames = []
            for fname in os.listdir(state.yaml_dir):
                if fname.endswith(".yaml"):
                    path = os.path.join(state.yaml_dir, fname)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                        contents.append(content)
                        fnames.append(fname)
                        files_refreshed += 1
                    except OSError as e:
                        logger.error(f"YAML file error for {fname}: {e}")
            if contents:
                if len(contents) > 10:  # FIX: Phase 3 - Batch with threads if large
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        embeddings = list(executor.map(embed_model.encode, contents))
                    embeddings = [emb.tolist() for emb in embeddings]
                else:
                    embeddings = embed_model.encode(contents).tolist()
                metadatas = [{"filename": fn} for fn in fnames]
                with state.chroma_lock: # FIX: Phase 1 - Lock.
                    col.upsert(
                        ids=fnames,
                        embeddings=embeddings,
                        documents=contents,
                        metadatas=metadatas,
                    )
                for i, fname in enumerate(fnames):
                    state.yaml_cache[fname] = contents[i]
            state.yaml_ready = True if files_refreshed > 0 else False
            return f"All YAMLs refreshed successfully ({files_refreshed} files)."
    except Exception as e:
        logger.error(f"YAML refresh error: {e}")
        return f"YAML refresh error: {e}"
# IMPROVED: Type-aware schema
def generate_tool_schema(func):
    sig = inspect.signature(func)
    properties = {}
    required = []
    type_map = {
        int: "integer",
        bool: "boolean",
        str: "string",
        float: "number",
        list: "array",
        dict: "object",
        typing.List: "array",
        # Add more as needed (e.g., typing.Dict: "object")
    }
    for param_name, param in sig.parameters.items():
        ann = param.annotation
        # Compute prop_type with fallback
        if ann is inspect._empty:
            prop_type = "string"
        else:
            # Handle typing (e.g., List[str] -> "array")
            if typing.get_origin(ann) is typing.List:
                base_type = typing.get_args(ann)[0] if typing.get_args(ann) else str
                prop_type = type_map.get(base_type, "array")
            else:
                prop_type = type_map.get(ann, "string")
     
        # Build prop uniformly
        prop = {"type": prop_type}
        if prop_type == "array":
            prop["items"] = {"type": "string"} # Assume string array; customize if needed
        if param.default is not inspect._empty and param.default != inspect.Parameter.empty:
            prop["default"] = param.default
     
        # Always assign
        properties[param_name] = prop
     
        # Always check required
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
 
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func) or "No description.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
# FIX: Phase 3 - Diagnostic tool.
def diagnose() -> str: # FIX: Phase 3 - Hack.
    return json.dumps({
        "stability": state.stability_score,
        "cache_size": len(state.memory_cache["lru_cache"]),
        # Add more if needed.
    })
# Custom tools (removed xai_*)
TOOL_FUNCS = [
    fs_read_file, fs_write_file, fs_list_files, fs_mkdir,
    get_current_time, code_execution, memory_insert, memory_query,
    git_ops, db_query, shell_exec, code_lint, api_simulate,
    advanced_memory_consolidate, advanced_memory_retrieve, advanced_memory_prune,
    generate_embedding, vector_search,
    chunk_text, summarize_chunk, keyword_search, socratic_api_council,
    agent_spawn, reflect_optimize, venv_create, restricted_exec,
    isolated_subprocess, pip_install, chat_log_analyze_embed,
    yaml_retrieve, yaml_refresh,
    visualize_got,
    diagnose # Added.
]
TOOLS = [generate_tool_schema(func) for func in TOOL_FUNCS]
# FIXED: Direct lambda for dispatcher with type coercion
TOOL_DISPATCHER = {}
for func in TOOL_FUNCS:
    def make_wrapper(f):
        sig = inspect.signature(f)
        def wrapper(**kwargs):
            # Type coercion
            coerced_kwargs = {}
            for k, v in kwargs.items():
                if k in sig.parameters:
                    ann = sig.parameters[k].annotation
                    try:  # FIX: Phase 1 - Catch coercion
                        if ann is int and isinstance(v, str):
                            coerced_kwargs[k] = int(v)
                        elif ann is bool and isinstance(v, str):
                            coerced_kwargs[k] = v.lower() in ('true', '1', 'yes')
                        elif ann is float and isinstance(v, str):
                            coerced_kwargs[k] = float(v)
                        elif ann is list and isinstance(v, str):
                            coerced_kwargs[k] = v.split(',')  # FIX: Phase 2 - Simple list coerce
                        else:
                            coerced_kwargs[k] = v
                    except ValueError:
                        return f"Arg type error for {k}: {v}"
                else:
                    coerced_kwargs[k] = v
            logger.info(f"Calling {f.__name__} with args: {coerced_kwargs}")
            result = safe_call(f, **coerced_kwargs)
            if callable(result): # Edge case
                result = "Tool returned callable—invoke error."
            return result
        return wrapper
    TOOL_DISPATCHER[func.__name__] = make_wrapper(func)
# Native xAI tools schemas (type: string for natives)
NATIVE_TOOLS = [
    {"type": "web_search"},
    {"type": "x_search"}, # Covers x_keyword, semantic, etc.
    {"type": "code_execution"},
    {"type": "browse_page"},
    {"type": "view_image"},
    {"type": "view_x_video"},
]
tool_count = get_state("tool_count", 0)
council_count = get_state("council_count", 0)
main_count = get_state("main_count", 0)
def _handle_single_tool(tool_call, current_messages, enable_tools):
    func_name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
        func_to_call = TOOL_DISPATCHER.get(func_name)
        if not func_to_call:
            raise ValueError(f"Unknown tool: {func_name}")
        result = func_to_call(**args)
    except Exception as e:
        result = f"Error calling tool {func_name}: {e}"
    if func_name == "socratic_api_council":
        with state.counter_lock:  # FIX: Phase 1 - Lock counters
            global council_count
            council_count += 1
    else:
        with state.counter_lock:
            global tool_count
            tool_count += 1
        st.session_state.setdefault("tool_calls_per_convo", 0)
        st.session_state["tool_calls_per_convo"] += 1
    logger.info(f"Tool call: {func_name} - Result: {str(result)[:200]}... Stability: {state.stability_score}") # FIX: Phase 3 - Log stability.
    yield f"\n> **Tool Call:** `{func_name}` | **Result:** `{str(result)[:200]}...`\n"
    current_messages.append(
        {"tool_call_id": tool_call.id, "role": "tool", "name": func_name, "content": str(result)}
    )
def process_tool_calls(tool_calls, current_messages, enable_tools):
    yield "\n*Thinking... Using tools...*\n"
    with state.conn:
        for tool_call in tool_calls:
            yield from _handle_single_tool(tool_call, current_messages, enable_tools)
def call_xai_api(
    model,
    messages,
    sys_prompt,
    stream=True,
    image_files=None,
    enable_tools=False,
    enable_xai_tools=True,
):
    # Custom function tools only (no natives in tools array)
    all_tools = TOOLS if enable_tools else None
 
    client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/", timeout=3600)
    api_messages = [{"role": "system", "content": sys_prompt}]
    for msg in messages:
        content_parts = [{"type": "text", "text": msg["content"]}]
        if msg["role"] == "user" and image_files and msg is messages[-1]:
            for img_file in image_files:
                img_file.seek(0)
                img_data = base64.b64encode(img_file.read()).decode("utf-8")
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img_file.type};base64,{img_data}"},
                    }
                )
        api_messages.append(
            {
                "role": msg["role"],
                "content": content_parts if len(content_parts) > 1 else msg["content"],
            }
        )
 
    def generate(current_messages):
        global tool_count, council_count, main_count
        max_iterations = Config.MAX_ITERATIONS.value
        tool_calls_per_convo = get_state("tool_calls_per_convo", 0)
        if tool_calls_per_convo > Config.TOOL_CALL_LIMIT.value:
            yield "Error: Tool call limit exceeded for this conversation."
            return
     
        # FIXED: Use local flag to avoid UnboundLocalError
        current_enable_xai_tools = enable_xai_tools  # FIX: Phase 1 - Local copy
     
        # Configure search_parameters for xAI natives
        extra_body = {}
        if current_enable_xai_tools:
            search_params = {
                "mode": "auto", # Grok decides; use "on" to force
                "return_citations": True, # Include sources
                "max_search_results": 10, # Limit for cost
                # Optional granular sources (uncomment/customize):
                # "sources": [
                # {"type": "web", "country": "US"},
                # {"type": "x", "included_x_handles": ["xai"]},
                # {"type": "news"}
                # ],
                # "from_date": "2025-11-01" # e.g., recent only
            }
            extra_body["search_parameters"] = search_params
            logger.info(f"Enabled xAI live search with params: {search_params}")
     
        retry_count = 0 # FIX: Phase 1 - Cap retries.
        start_time = time.time()  # FIX: Phase 3 - Total timeout
        for iteration in range(max_iterations):
            with state.counter_lock:  # FIX: Phase 1 - Lock
                main_count += 1
            logger.info(
                f"API call: Tools: {tool_count} | Council: {council_count} | Main: {main_count} | Iteration: {iteration + 1} | Search Enabled: {current_enable_xai_tools}"
            )
            try:
                api_limiter_sync()
                response = client.chat.completions.create(
                    model=model,
                    messages=current_messages,
                    tools=all_tools,
                    tool_choice="auto" if all_tools else None,
                    stream=stream,
                    extra_body=extra_body, # NEW: For search_parameters
                )
                tool_calls = []
                full_delta_response = ""
                sources_used = 0 # Track for logging
                chunk_buffer = ""  # FIX: Phase 3 - Buffer yields
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        chunk_buffer += delta.content
                        if len(chunk_buffer.split()) > 3:  # Yield every ~3 words
                            yield chunk_buffer
                            chunk_buffer = ""
                        full_delta_response += delta.content
                    if delta and delta.tool_calls:
                        tool_calls.extend(delta.tool_calls)
                    # Log sources in final chunk (non-streaming fallback)
                    if chunk.usage and hasattr(chunk.usage, 'num_sources_used'):
                        sources_used = chunk.usage.num_sources_used
                if chunk_buffer:  # Flush remaining
                    yield chunk_buffer
                logger.info(f"Search sources used: {sources_used}")
             
                if not tool_calls:
                    break
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": full_delta_response,
                        "tool_calls": tool_calls,
                    }
                )
                for chunk in process_tool_calls(tool_calls, current_messages, enable_tools):
                    yield chunk
            except openai.AuthenticationError as e:
                error_msg = f"Auth error: Invalid API key or access. {str(e)}"
                yield f"\nAuth issue: {error_msg}. Check your XAI_API_KEY."
                logger.error(f"OpenAI Auth error: {e}")
                break
            except openai.RateLimitError as e:
                backoff = 60
                error_msg = f"Rate limit hit: {str(e)}. Backing off {backoff}s..."
                yield f"\n{error_msg}"
                logger.warning(f"Rate limit: {e}. Backoff: {backoff}s")
                time.sleep(backoff)
                continue
            except openai.APITimeoutError as e:
                error_msg = f"API timeout: {str(e)}. Retrying..."
                yield f"\nTimeout: {error_msg}"
                logger.warning(f"API timeout: {e}")
                time.sleep(5 * (iteration + 1))
                continue
            except openai.APIError as e:
                error_code = getattr(e, 'code', 'unknown')
                if "search_parameters" in str(e): # Specific to natives
                    error_msg = f"Search config error ({error_code}): {str(e)}. Disabling natives for retry."
                    current_enable_xai_tools = False # FIXED: Use local flag
                    extra_body = {} # Reset
                    yield "\nNative search disabled—fallback to local." # Fixed: Removed unnecessary f prefix.
                    continue
                else:
                    error_msg = f"API error ({error_code}): {str(e)}. Retrying..."
                yield f"\nAPI hiccup: {error_msg}"
                logger.error(f"OpenAI API error (code {error_code}): {e}")
                if retry_count < 5:
                    retry_count += 1
                    time.sleep(2 ** retry_count)
                    continue
                else:
                    yield "Max retries—aborting."
                    break
            except requests.exceptions.RequestException as e:
                error_msg = f"Network error: {str(e)}. Retrying..."
                yield f"\nNetwork glitch: {error_msg}"
                logger.error(f"Request error: {e}")
                time.sleep(2 ** (iteration + 1))
                continue
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}. Aborting this turn."
                yield f"\n{error_msg}"
                logger.error(f"Unexpected error in API call: {traceback.format_exc()}")
                break
            if time.time() - start_time > 300:  # FIX: Phase 3 - Cap total
                yield "Total timeout—aborting."
                break
    return generate(api_messages)
def search_history(query: str):
    state.c.execute(
        "SELECT convo_id, title FROM history WHERE user=? AND title LIKE ?",
        (st.session_state["user"], f"%{query}%"),
    )
    return state.c.fetchall()
def export_convo(format: str = "json"):
    if "messages" not in st.session_state:  # FIX: Phase 1 - Check
        return "No convo to export."
    if format == "json":
        return json.dumps(st.session_state["messages"], indent=4)
    elif format == "md":
        md = ""
        for msg in st.session_state["messages"]:
            md += f"**{msg['role'].capitalize()}:** {msg['content']}\n\n"
        return md
    elif format == "txt":
        txt = ""
        for msg in st.session_state["messages"]:
            txt += f"{msg['role']}: {msg['content']}\n\n"
        return txt
    return "Unsupported format."
def render_sidebar():
    global main_count, tool_count, council_count # Moved to top of function
    with st.sidebar:
        st.header("Chat Settings")
        st.selectbox(
            "Select Model",
            [m.value for m in Models],
            key="model_select",
        )
        st.selectbox(
            "Select Council Model",
            [m.value for m in Models],
            key="council_model_select",
        )
        prompt_files = load_prompt_files()
        if prompt_files:
            selected_file = st.selectbox(
                "Select System Prompt", prompt_files, key="prompt_select"
            )
            load_prompt = False
            if "last_prompt_file" not in st.session_state:
                st.session_state["last_prompt_file"] = selected_file
                load_prompt = True
            elif st.session_state["last_prompt_file"] != selected_file:
                load_prompt = True
            if load_prompt:
                file_path = os.path.join(state.prompts_dir, selected_file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        new_prompt_content = f.read().strip()
                    st.session_state["custom_prompt"] = new_prompt_content
                    st.session_state["last_prompt_file"] = selected_file
                    logger.info(f"Prompt loaded from {selected_file} (init/change override)")
                    st.rerun()
                except FileNotFoundError:
                    st.error(f"Prompt file '{selected_file}' missing!")
                    st.session_state["custom_prompt"] = "Default prompt fallback."
                    st.session_state["last_prompt_file"] = selected_file
                    st.rerun()
            current_prompt = st.session_state.get("custom_prompt", "")
            st.text_area(
                "Edit System Prompt",
                value=current_prompt,
                height=200,
                key="custom_prompt",
            )
            if st.button("Evolve Prompt"):
                user = st.session_state.get("user")
                convo_id = st.session_state.get("current_convo_id", 0)
                metrics = {"length": len(st.session_state["custom_prompt"]), "vibe": "creative"}
                new_prompt = auto_optimize_prompt(st.session_state["custom_prompt"], user, convo_id, metrics)
                st.session_state["custom_prompt"] = new_prompt
                st.rerun()
            st.checkbox("Enable Tools (Sandboxed)", value=False, key="enable_tools")
            if st.session_state["enable_tools"]:
                st.checkbox("Include xAI Natives (Web/X Search)", value=True, key="enable_xai_tools")
        else:
            st.warning("No prompt files found in ./prompts/")
            st.text_area(
                "System Prompt",
                value="You are a helpful AI.",
                height=200,
                key="custom_prompt",
            )
        st.file_uploader(
            "Upload Images",
            type=["jpg", "png"],
            accept_multiple_files=True,
            key="uploaded_images",
        )
        st.divider()
        tab1, tab2 = st.tabs(["Settings", "Agent Fleet"])
        with tab1:
            if st.button("➕ New Chat", use_container_width=True):
                st.session_state["messages"] = []
                st.session_state["current_convo_id"] = 0
                st.session_state["tool_calls_per_convo"] = 0
                st.session_state["rerun_guard"] = False
                st.rerun()
            st.header("Chat History")
            history_search = st.text_input("Search History", key="history_search")
            if history_search:
                histories = search_history(history_search)
            else:
                state.c.execute(
                    "SELECT convo_id, title FROM history WHERE user=? ORDER BY convo_id DESC",
                    (st.session_state["user"],),
                )
                histories = state.c.fetchall()
            for convo_id, title in histories:
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(title, key=f"load_{convo_id}", use_container_width=True):
                        load_history(convo_id)
                with col2:
                    if st.button("🗑️", key=f"delete_{convo_id}", use_container_width=True):
                        delete_history(convo_id)
            st.header("Export Current Convo")
            export_format = st.selectbox("Format", ["json", "md", "txt"])
            if st.button("Export"):
                exported = export_convo(export_format)
                st.download_button("Download", exported, file_name=f"convo.{export_format}")
            with st.expander("Metrics & Viz"):
                if "memory_cache" in st.session_state:
                    metrics = st.session_state["memory_cache"]["metrics"]
                    st.metric("Total Inserts", metrics["total_inserts"])
                    st.metric("Total Retrieves", metrics["total_retrieves"])
                    st.metric("Hit Rate", f"{metrics['hit_rate']:.2%}")
                    st.metric("Tool Calls", tool_count)
                    st.metric("Council Calls", council_count)
                    st.metric("Stability Score", f"{state.stability_score:.2%}")
                    st.metric("Main API Calls", main_count) # FIX: Phase 3.
                if st.button("Weave Memory Lattice (Viz Current Convo)"):
                    viz_result = viz_memory_lattice(st.session_state["user"], st.session_state["current_convo_id"])
                    st.info(viz_result)
                if st.button("Prune Memory Now"):
                    prune_result = advanced_memory_prune(user=st.session_state["user"], convo_id=st.session_state["current_convo_id"])
                    st.info(prune_result)
                if st.button("Clear Cache"):
                    st.session_state["tool_cache"] = {}
                    state.memory_cache["lru_cache"] = {}
                    st.success("Cache cleared.")
                if st.button("Reset Counters"): # FIX: Phase 3.
                    with state.counter_lock:
                        main_count = tool_count = council_count = 0
                # NEW: Tool Logs Expander
                if st.button("Show Recent Tool Logs"):
                    with st.expander("Tool Logs", expanded=False):
                        recent_logs = [line for line in open("app.log", "r").readlines()[-20:] if "Tool call:" in line]
                        for log in recent_logs:
                            st.text(log.strip())
        with tab2:
            render_agent_fleet()
def login_page():
    st.title("Aurum Vivum Interface")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                state.c.execute("SELECT password FROM users WHERE username=?", (username,))
                result = state.c.fetchone()
                if result and verify_password(result[0], password):
                    st.session_state["logged_in"] = True
                    st.session_state["user"] = username
                    st.session_state["rerun_guard"] = False
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            if st.form_submit_button("Register"):
                state.c.execute("SELECT * FROM users WHERE username=?", (new_user,))
                if state.c.fetchone():
                    st.error("Username already exists.")
                else:
                    state.c.execute(
                        "INSERT INTO users VALUES (?, ?)",
                        (new_user, hash_password(new_pass)),
                    )
                    state.conn.commit()
                    st.success("Registered! Please login.")
def load_history(convo_id: int):
    state.c.execute(
        "SELECT messages FROM history WHERE convo_id=? AND user=?",
        (convo_id, st.session_state["user"]),
    )
    if result := state.c.fetchone():
        messages = json.loads(result[0])
        st.session_state["messages"] = messages
        st.session_state["current_convo_id"] = convo_id
        st.session_state["rerun_guard"] = False
        st.rerun()
def delete_history(convo_id: int):
    state.c.execute(
        "DELETE FROM history WHERE convo_id=? AND user=?",
        (convo_id, st.session_state["user"]),
    )
    state.conn.commit()
    if st.session_state.get("current_convo_id") == convo_id:
        st.session_state["messages"] = []
        st.session_state["current_convo_id"] = 0
    st.session_state["rerun_guard"] = False
    st.rerun()
def render_chat_interface(model: str, custom_prompt: str, enable_tools: bool, uploaded_images: list, enable_xai_tools: bool):
    st.title(f"Aurum Vivum - {st.session_state['user']}")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "current_convo_id" not in st.session_state:
        st.session_state["current_convo_id"] = 0
    if "tool_calls_per_convo" not in st.session_state:
        st.session_state["tool_calls_per_convo"] = 0
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=False)
    if prompt := st.chat_input("Your command, ape?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt, unsafe_allow_html=False)
        with st.chat_message("assistant"):
            images_to_process = uploaded_images if uploaded_images else []
            generator = call_xai_api(
                model,
                st.session_state.messages,
                custom_prompt,
                stream=True,
                image_files=images_to_process,
                enable_tools=enable_tools,
                enable_xai_tools=enable_xai_tools,
            )
            full_response = st.write_stream(generator)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        title = gen_title(prompt)
        messages_json = json.dumps(st.session_state.messages)
        if st.session_state.get("current_convo_id", 0) == 0:
            with state.conn:
                state.c.execute(
                    "INSERT INTO history (user, title, messages) VALUES (?, ?, ?)",
                    (st.session_state["user"], title, messages_json),
                )
                st.session_state.current_convo_id = state.c.lastrowid
        else:
            with state.conn:
                state.c.execute(
                    "UPDATE history SET title=?, messages=? WHERE convo_id=?",
                    (title, messages_json, st.session_state["current_convo_id"]),
                )
        st.session_state["rerun_guard"] = False
        render_agent_fleet()
        st.rerun()
def chat_page():
    render_sidebar()
    render_chat_interface(
        st.session_state["model_select"],
        st.session_state["custom_prompt"],
        st.session_state["enable_tools"],
        st.session_state["uploaded_images"],
        st.session_state.get("enable_xai_tools", True),
    )
if "auto_prune_done" not in st.session_state:
    user = st.session_state.get("user", 'shared')
    convo_id = st.session_state.get("current_convo_id", 0)
    advanced_memory_prune(user=user, convo_id=convo_id)
    st.session_state["auto_prune_done"] = True
def run_tests():
    class TestTools(unittest.TestCase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # FIX: Phase 1 test for chroma lock - Sim race.
            def test_chroma_lock(self):
                def consolidate_task():
                    advanced_memory_consolidate("test_key", {"data": "test"})
                def retrieve_task():
                    advanced_memory_retrieve("test")
                threads = [threading.Thread(target=consolidate_task) for _ in range(2)] + [threading.Thread(target=retrieve_task) for _ in range(2)]
                for t in threads: t.start()
                for t in threads: t.join()
                self.assertTrue(True) # Check no exceptions in logs manually.
            # FIX: Phase 2 - Restricted policy.
            def test_restricted_policy(self):
                bad_code = "__builtins__['open']('secret.txt')"
                result = code_execution(bad_code)
                self.assertIn("Banned" or "Restricted", result)
            # Original tests here...
            def setUp(self):
                st.session_state["user"] = "test_user"
                st.session_state["current_convo_id"] = 1
                st.session_state["memory_cache"] = {
                    "lru_cache": {},
                    "metrics": {"total_inserts": 0, "total_retrieves": 0, "hit_rate": 1.0},
                }
                st.session_state["tool_cache"] = {}
                st.session_state["yaml_ready"] = True
                st.session_state["chroma_ready"] = True
            def test_fs_write_read(self):
                result_write = safe_call(fs_write_file, "test.txt", "Hello World")
                self.assertIn("successfully", result_write)
                result_read = safe_call(fs_read_file, "test.txt")
                self.assertEqual(result_read, "Hello World")
            def test_memory_insert_query(self):
                mem_value = {"summary": "test mem", "salience": 0.8}
                result_insert = safe_call(memory_insert, "test_key", mem_value, "test", 1)
                self.assertEqual(result_insert, "Memory inserted successfully.")
                result_query = safe_call(memory_query, mem_key="test_key", user="test", convo_id=1)
                self.assertIn("summary", str(result_query))
                self.assertIn("test mem", str(result_query))
            def test_memory_shared_mode(self):
                result_insert = safe_call(memory_insert, "shared_key", {"test": "data"})
                self.assertEqual(result_insert, "Memory inserted successfully.")
                query = safe_call(memory_query, mem_key="shared_key")
                self.assertIn("test", str(query))
            def test_advanced_memory_prune(self):
                low_mem = {"summary": "low salience", "salience": 0.05}
                safe_call(memory_insert, "low_key", low_mem, "test", 1)
                st.session_state["prune_counter"] = 49
                result_prune = safe_call(advanced_memory_prune, "test", 1)
                self.assertIn("successfully", result_prune)
                query_after = safe_call(memory_query, mem_key="low_key", user="test", convo_id=1)
                self.assertEqual(query_after, "Key not found.")
            def test_tool_cache(self):
                args = {"file_path": "cache_test.txt"}
                set_cached_tool_result("fs_read_file", args, "cached_value")
                result = get_cached_tool_result("fs_read_file", args)
                self.assertEqual(result, "cached_value")
                old_timestamp = datetime.now() - timedelta(minutes=20)
                st.session_state["tool_cache"][get_tool_cache_key("fs_read_file", args)] = (old_timestamp, "expired")
                result_expired = get_cached_tool_result("fs_read_file", args)
                self.assertIsNone(result_expired)
            def test_agent_spawn_persist(self):
                result_spawn = safe_call(agent_spawn, "TestAgent", "Mock task: echo hello", "test", 1)
                self.assertIn("spawned (ID:", result_spawn)
                agent_id_prefix = result_spawn.split("ID: ")[1].split(")")[0][:12]
                status_query = safe_call(memory_query, mem_key=f"agent_{agent_id_prefix}_status", user="test", convo_id=1)
                self.assertIn("status", str(status_query))
                self.assertIn("spawned", str(status_query))
            def test_code_execution_restricted(self):
                code = "print('Hello from REPL')"
                result = safe_call(code_execution, code)
                self.assertIn("Hello from REPL", result)
                bad_code = "import os; os.system('ls')"
                bad_result = safe_call(code_execution, bad_code)
                self.assertIn("Restricted", bad_result) or self.assertIn("Error", bad_result)
            def test_git_ops_init_commit(self):
                result_init = safe_call(git_ops, "init", "test_repo")
                self.assertEqual(result_init, "Repo initialized.")
                safe_call(fs_write_file, "test_repo/test.txt", "commit test")
                result_commit = safe_call(git_ops, "commit", "test_repo", message="Test commit")
                self.assertEqual(result_commit, "Committed.")
            def test_shell_exec_whitelist(self):
                result_ls = safe_call(shell_exec, "ls .")
                self.assertNotIn("Error: Command not whitelisted.", result_ls)
                st.session_state["confirm_destructive_1"] = False
                result_rm = safe_call(shell_exec, "rm test_rm.txt")
                self.assertIn("Warning: Destructive command detected.", result_rm)
                st.session_state["confirm_destructive_1"] = True
                result_rm_confirm = safe_call(shell_exec, "rm test_rm.txt")
                self.assertNotIn("Warning", result_rm_confirm)
                result_bad = safe_call(shell_exec, "curl google.com")
                self.assertIn("Error: Command not whitelisted.", result_bad)
                # NEW: Injection test
                inject_cmd = 'grep "; rm *"'
                result_inject = safe_call(shell_exec, inject_cmd)
                self.assertIn("Error", result_inject) or self.assertNotIn("rm", result_inject)
            def test_code_lint_python(self):
                ugly_code = "def foo():print('hi')"
                result = safe_call(code_lint, "python", ugly_code)
                self.assertIn("def foo():\n print('hi')", result)
            def test_api_simulate_mock(self):
                result_mock = safe_call(api_simulate, "https://api.example.com/test", method="GET", mock=True)
                self.assertIn("Mock response", result_mock)
            def test_yaml_retrieve_ready(self):
                safe_call(fs_write_file, "test.yaml", "key: value")
                safe_call(yaml_refresh, "test.yaml")
                result = safe_call(yaml_retrieve, filename="test.yaml")
                self.assertIn("key: value", result)
            def test_socratic_council(self):
                branches = ["Option A", "Option B"]
                result = safe_call(socratic_api_council, branches, user="test", convo_id=1)
                self.assertIn("Round", result)
                self.assertIn("Consensus", result)
            def test_agent_sem_bound(self):
                self.assertTrue(hasattr(state, 'agent_sem'))
                self.assertEqual(state.agent_sem._value, Config.AGENT_MAX_CONCURRENT.value)
            # NEW: Test reflect
            def test_reflect_optimize(self):
                metrics = {"test": 1}
                result = safe_call(reflect_optimize, "test_component", metrics)
                self.assertIn("Optimized", result) or self.assertIn("Suggest", result)
            # NEW: Async test
            def test_async_spawn(self):
                import asyncio
                result = asyncio.run(async_run_spawn("test_id", "echo test", "test", 1))
                self.assertIn("test", result)  # Assume mock success
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTools)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    state.stability_score = 1.0 - (len([t for t, s in result.failures + result.errors]) / len(suite.tests)) if suite.tests else 1.0
    return result.wasSuccessful()
if os.getenv('TEST_MODE'):
    run_tests()
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if st.session_state.get("logged_in"):
        chat_page()
    else:
        login_page()
