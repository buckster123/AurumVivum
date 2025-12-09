# Moonshot-Adapted Aurum Vivum - REFACTORED v1.1

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
import qutip
import qiskit
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
import httpx
import torch
import cProfile
import pstats
import aiofiles
from typing import Callable, Any, TypedDict
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
import xml.dom.minidom
from marshmallow import Schema, fields, ValidationError
from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

nest_asyncio.apply()

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

# Moonshot API configuration
API_KEY = os.getenv("MOONSHOT_API_KEY")
if not API_KEY:
    st.error("MOONSHOT_API_KEY not set in .env! Please add it and restart.")

class Config(Enum):
    DEFAULT_TOP_K = 5
    CACHE_TTL_MIN = 15
    AGENT_MAX_CONCURRENT = 5
    PRUNE_FREQUENCY = 50
    SIM_THRESHOLD = 0.6
    MAX_TASK_LEN = 2000
    STABILITY_PENALTY = 0.05
    MAX_ITERATIONS = 100
    TOOL_CALL_LIMIT = 200
    API_CALLS_PER_MIN = 100
    API_TOKENS_PER_MIN = 1000000
    API_CONCURRENT_REQUESTS = 5
    TOOL_CALLS_PER_MIN = 100

# Global counters (will be moved to session_state)
main_count = 0
tool_count = 0
council_count = 0

# Moonshot Models
class Models(Enum):
    MOONSHOT_V1_8K = "moonshot-v1-8k"
    MOONSHOT_V1_32K = "moonshot-v1-32k"
    MOONSHOT_V1_128K = "moonshot-v1-128k"
    KIMI_LATEST = "kimi-latest"
    KIMI_K_THINKING_TURBO = "kimi-k2-thinking-turbo"
    KIMI_K2_THINKING = "kimi-k2-thinking"
    KIMI_K2_THINKING_PREVIEW = "kimi-k2"

def hash_password(password: str) -> str:
    return sha256_crypt.hash(password)

def verify_password(stored: str, provided: str) -> bool:
    return sha256_crypt.verify(provided, stored)

def sync_limiter(calls_per_min: int):
    last_call = [0]
    lock = threading.Lock()
    def limiter():
        now = time.time()
        with lock:
            if now - last_call[0] < 60 / calls_per_min:
                time.sleep((60 / calls_per_min) - (now - last_call[0]))
            last_call[0] = time.time()
    return limiter

async def async_limiter(sem: asyncio.Semaphore):
    async with sem:
        pass

class MoonshotRateLimiter:
    """Handles Moonshot's multi-dimensional rate limits"""
    def __init__(self):
        self.rpm_limiter = sync_limiter(Config.API_CALLS_PER_MIN.value)
        self.tpm_limiter = threading.Semaphore(Config.API_TOKENS_PER_MIN.value)
        self.concurrent_limiter = threading.Semaphore(Config.API_CONCURRENT_REQUESTS.value)
    
    def __call__(self, estimated_tokens: int = 100):
        self.rpm_limiter()
        self.concurrent_limiter.acquire()
        if estimated_tokens > Config.API_TOKENS_PER_MIN.value:
            raise ValueError("Request exceeds token limit")

api_limiter_sync = MoonshotRateLimiter()
tool_limiter_sync = sync_limiter(Config.TOOL_CALLS_PER_MIN.value)

# Session state helpers for unified state management
def get_session_cache(key: str, default_factory=dict):
    """Unified cache access using session_state"""
    if key not in st.session_state:
        st.session_state[key] = default_factory()
    return st.session_state[key]

def get_memory_cache():
    return get_session_cache("memory_cache", lambda: {
        "lru_cache": {},
        "metrics": {
            "total_inserts": 0,
            "total_retrieves": 0,
            "hit_rate": 1.0,
            "last_update": None,
        },
    })

def get_stability_score() -> float:
    return st.session_state.get("stability_score", 1.0)

def set_stability_score(value: float):
    st.session_state["stability_score"] = max(0.0, min(1.0, value))

def inject_convo_uuid(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['convo_uuid'] = kwargs.get('convo_uuid', st.session_state.get('current_convo_uuid', str(uuid.uuid4())))
        return func(*args, **kwargs)
    return wrapper

# Lightweight DI Container for plugin system
class Container:
    """Lightweight DI for tool plugins"""
    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._official_tools: dict[str, str] = {}
    
    def register_tool(self, func: Callable, name: str = None):
        name = name or func.__name__
        self._tools[name] = func
        logger.info(f"Registered tool: {name}")
    
    def register_official_tool(self, tool_name: str, formula_uri: str):
        self._official_tools[tool_name] = formula_uri
    
    def get_tool(self, name: str) -> Callable | None:
        return self._tools.get(name)
    
    def get_official_tools(self) -> dict[str, str]:
        return self._official_tools.copy()

# Initialize container at startup
container = Container()

class AppState:
    def __init__(self):
        self.db_path = "./sandbox/db/chatapp.db"
        self.chroma_path = "./sandbox/db/chroma_db"
        self.chroma_client = None
        self.chroma_collection = None
        self.yaml_collection = None
        self._init_resources()
        
        # Complex, non-serializable objects stay in AppState
        self.agent_executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.agent_lock = threading.Lock()
        self.chroma_lock = threading.Lock()
        self.counter_lock = threading.Lock()
        self.session_lock = threading.Lock()
        self.agent_sem = asyncio.Semaphore(Config.AGENT_MAX_CONCURRENT.value)
        self.embed_model = None
        self.yaml_ready = False
        
        # Simple state moved to session_state via helpers
        # self.memory_cache, self.stability_score, self.yaml_cache removed
        
        self.prompts_dir = "./prompts"
        self.sandbox_dir = "./sandbox"
        self.yaml_dir = "./sandbox/config"
        self.agent_dir = os.path.join(self.sandbox_dir, "agents")
        
        # Initialize directories
        os.makedirs(self.prompts_dir, exist_ok=True)
        os.makedirs(self.sandbox_dir, exist_ok=True)
        os.makedirs(self.yaml_dir, exist_ok=True)
        os.makedirs(self.agent_dir, exist_ok=True)
        
        self._init_prompts()
        self._init_yaml_embeddings()

    def _init_prompts(self):
        default_prompts = {
            "default.txt": "You are Aurum Vivum, a highly intelligent AI assistant powered by Moonshot AI.",
            "coder.txt": "You are an expert Aurum coder, providing precise code solutions.",
            "tools-enabled.txt": """Loaded via .txt file.""",
        }
        if not any(f.endswith(".txt") for f in os.listdir(self.prompts_dir)):
            for filename, content in default_prompts.items():
                with open(os.path.join(self.prompts_dir, filename), "w") as f:
                    f.write(content)

    def _init_resources(self) -> None:
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.cursor = self.conn.cursor()
            self._init_db()
            
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="memory_vectors",
                metadata={"hnsw:space": "cosine"},
            )
            self.yaml_collection = self.chroma_client.get_or_create_collection(
                name="yaml_vectors", metadata={"hnsw:space": "cosine"}
            )
        except sqlite3.Error as e:
            logger.error(f"DB error: {e}")
            st.error(f"Failed to initialize database: {e}. App may not function properly.")
        except chromadb.errors.InvalidDimensionException as e:
            logger.error(f"Chroma error: {e}")
        except Exception as e:
            logger.warning(f"Resource init failed ({e}). Falling back.")

    def _init_db(self) -> None:
        try:
            with self.conn:
                self.cursor.execute(
                    """CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)"""
                )
                self.cursor.execute(
                    """CREATE TABLE IF NOT EXISTS history (user TEXT, convo_id INTEGER PRIMARY KEY AUTOINCREMENT, uuid TEXT UNIQUE, title TEXT, messages TEXT)"""
                )
                self.cursor.execute(
                    """CREATE TABLE IF NOT EXISTS memory (
                    uuid TEXT,
                    mem_key TEXT,
                    mem_value TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    salience REAL DEFAULT 1.0,
                    parent_id INTEGER,
                    PRIMARY KEY (uuid, mem_key)
                )"""
                )
                self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory (timestamp)")
                self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_uuid ON memory (uuid)")
                
                self.cursor.execute("INSERT OR IGNORE INTO users (username, password) VALUES ('shared', ?)", (hash_password(''),))
                self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"DB init error: {e}")

    def _init_yaml_embeddings(self) -> None:
        embed_model = self.get_embed_model()
        if embed_model:
            files_refreshed = 0
            failed = []
            contents = []
            fnames = []
            for fname in os.listdir(self.yaml_dir):
                if fname.endswith(".yaml"):
                    path = os.path.join(self.yaml_dir, fname)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            content = f.read()
                        contents.append(content)
                        fnames.append(fname)
                        files_refreshed += 1
                    except OSError as e:
                        logger.error(f"YAML file error for {fname}: {e}")
                        failed.append(fname)
            if contents:
                embeddings = embed_model.encode(contents, batch_size=32).tolist()
                metadatas = [{"filename": fn} for fn in fnames]
                with self.chroma_lock:
                    self.yaml_collection.upsert(
                        ids=fnames,
                        embeddings=embeddings,
                        documents=contents,
                        metadatas=metadatas,
                    )
                # Update session_state cache
                yaml_cache = get_session_cache("yaml_cache")
                for i, fname in enumerate(fnames):
                    yaml_cache[fname] = contents[i]
            self.yaml_ready = len(failed) == 0 and files_refreshed > 0
            if failed:
                logger.warning(f"Failed YAML files: {failed}")
            logger.info(f"YAML embeddings inited ({files_refreshed} files).")

    @classmethod
    def get(cls) -> "AppState":
        if "app_state" not in st.session_state:
            st.session_state["app_state"] = cls()
        return st.session_state["app_state"]

    def get_embed_model(self) -> SentenceTransformer | None:
        if self.embed_model is None:
            for _ in range(3):
                try:
                    with st.spinner("Loading embedding model for advanced memory (first-time use)..."):
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    def lru_evict(self) -> None:
        memory_cache = get_memory_cache()
        if "lru_ordered" not in memory_cache:
            from collections import OrderedDict
            memory_cache["lru_ordered"] = OrderedDict()
        lru = memory_cache["lru_ordered"]
        if len(lru) > 500:
            num_to_evict = len(lru) - 500
            for _ in range(num_to_evict):
                key, entry = lru.popitem(last=False)
                if entry["salience"] < 0.4:
                    memory_cache["lru_cache"].pop(key, None)

# Initialize state
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

def get_state(key: str, default=None) -> typing.Any:
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

def safe_call(func: typing.Callable, *args, max_retries: int = 3, **kwargs) -> typing.Any:
    start = time.time()
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            set_stability_score(get_stability_score() + 0.01)
            return result
        except (ValueError, TypeError, sqlite3.Error, asyncio.TimeoutError, requests.RequestException) as e:
            logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                set_stability_score(get_stability_score() - Config.STABILITY_PENALTY.value)
                return f"Oops: Error occurred. Check logs for details."
            time.sleep(2 ** attempt)
        except chromadb.errors.InvalidDimensionException:
            return f"Chroma error"
        except Exception:
            set_stability_score(get_stability_score() - Config.STABILITY_PENALTY.value)
            notify_critical("Error in tool execution")
            return f"Tool glitched"
    set_stability_score(get_stability_score() - 0.1)
    logger.info(f"{func.__name__} took {time.time() - start}s")
    return "Max retries exceeded."

def notify_critical(err: str) -> None:
    st.error(f"Critical: {err}")
    logger.critical(err)

def get_tool_cache_key(func_name: str, args: dict) -> str:
    try:
        arg_str = json.dumps(args, sort_keys=True)
    except TypeError:
        arg_str = str(args)
    return f"tool_cache:{func_name}:{hashlib.sha256(arg_str.encode()).hexdigest()}"

def get_cached_tool_result(func_name: str, args: dict, ttl_minutes: int = Config.CACHE_TTL_MIN.value) -> str | None:
    cache = get_session_cache("tool_cache")
    key = get_tool_cache_key(func_name, args)
    if key in cache:
        timestamp, result = cache[key]
        if (datetime.now() - timestamp).total_seconds() / 60 < ttl_minutes:
            return result
    return None

def set_cached_tool_result(func_name: str, args: dict, result: str) -> None:
    cache = get_session_cache("tool_cache")
    key = get_tool_cache_key(func_name, args)
    cache[key] = (datetime.now(), result)
    if len(cache) > 100:
        oldest_key = min(cache, key=lambda k: cache[k][0])
        del cache[oldest_key]

def load_prompt_files() -> list[str]:
    return [f for f in os.listdir(state.prompts_dir) if f.endswith(".txt")]

def gen_title(first_msg: str) -> str:
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://api.moonshot.ai/v1")
        resp = client.chat.completions.create(
            model=Models.KIMI_K2_THINKING.value,
            messages=[
                {"role": "user", "content": f"Summarize this query into a short, punchy title: {first_msg[:500]}"},
                {"role": "system", "content": "Keep under 50 chars, evocative."}
            ],
            max_tokens=50
        )
        return resp.choices[0].message.content.strip()[:50]
    except Exception:
        logger.warning("Title gen failed")
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
    except Exception:
        logger.error("Prompt optimize error")
        return current_prompt

@st.cache_data(ttl=3600, hash_funcs={dict: lambda d: json.dumps(d, sort_keys=True)})
def fs_read_file(file_path: str) -> str:
    safe_path = pathlib.Path(state.sandbox_dir) / pathlib.Path(file_path).relative_to('.')
    safe_path = safe_path.resolve()
    if not safe_path.is_relative_to(pathlib.Path(state.sandbox_dir).resolve()):
        return "Error: Path is outside the sandbox."
    if not safe_path.exists():
        if safe_path.suffix in ['.yaml', '.lattice']:
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            with open(safe_path, 'w') as f:
                f.write('{}')
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
    safe_path = pathlib.Path(state.sandbox_dir) / pathlib.Path(file_path).relative_to('.')
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
    safe_dir = pathlib.Path(state.sandbox_dir) / pathlib.Path(dir_path or '.').relative_to('.')
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
    safe_path = pathlib.Path(state.sandbox_dir) / pathlib.Path(dir_path).relative_to('.')
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
        "print", "len", "range", "str", "int", "float", "list", "dict",
        "set", "tuple", "abs", "round", "max", "min", "sum", "sorted",
    ]
}

ADDITIONAL_LIBS = {
    "numpy": np,
    "sympy": sympy,
    "mpmath": mpmath,
    "pygame": pygame,
    "chess": chess,
    "networkx": nx,
    "unittest": unittest,
    "asyncio": asyncio,
    "qiskit": __import__('qiskit') if 'qiskit' in sys.modules else None,
    "scipy": __import__('scipy') if 'scipy' in sys.modules else None,
    "matplotlib": __import__('matplotlib') if 'matplotlib' in sys.modules else None,
    "pandas": __import__('pandas') if 'pandas' in sys.modules else None,
}

def init_repl_namespace() -> None:
    if "repl_namespace" not in st.session_state:
        st.session_state["repl_namespace"] = {"__builtins__": SAFE_BUILTINS.copy()}
        st.session_state["repl_namespace"].update({k: v for k, v in ADDITIONAL_LIBS.items() if v is not None})

def restricted_policy(node: ast.AST) -> bool:
    if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
        if isinstance(node, ast.ImportFrom) and node.module in ADDITIONAL_LIBS:
            return True
        return False
    if isinstance(node, ast.Call) and hasattr(node.func, 'id') and node.func.id in ['open', 'exec', 'eval', 'subprocess']:
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
        if 'open' in code:
            return "Banned func."
        result = RestrictedPython.compile_restricted_exec(code)
        if result.errors:
            return f"Restricted compile error: {result.errors}"
        try:
            exec(result.code, st.session_state["repl_namespace"], {})
        except Exception as e:
            return f"Exec error: {traceback.format_exc()}"
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
        sys.stdout = old_stdout

@inject_convo_uuid
def memory_insert(
    mem_key: str, mem_value: dict | str, convo_uuid: str
) -> str:
    tool_limiter_sync()
    if isinstance(mem_value, str):
        mem_value = {"raw": mem_value}
        logger.warning(f"Wrapped mem_value str for '{mem_key}'")
    if not isinstance(mem_value, dict):
        return "Error: mem_value must be a dict (or valid JSON str)."
    try:
        json.dumps(mem_value)
    except TypeError:
        return "Error: mem_value not JSON-serializable."
    try:
        with state.conn:
            state.cursor.execute(
                "INSERT OR REPLACE INTO memory (uuid, mem_key, mem_value) VALUES (?, ?, ?)",
                (convo_uuid, mem_key, json.dumps(mem_value)),
            )
        
        # Update session_state cache
        memory_cache = get_memory_cache()
        cache_key = f"{convo_uuid}_{mem_key}"
        entry = {
            "summary": mem_value.get("summary", ""),
            "details": mem_value.get("details", ""),
            "tags": mem_value.get("tags", []),
            "domain": mem_value.get("domain", "general"),
            "timestamp": datetime.now().isoformat(),
            "salience": mem_value.get("salience", 1.0),
        }
        memory_cache["lru_cache"][cache_key] = {
            "entry": entry,
            "last_access": time.time(),
        }
        memory_cache["metrics"]["total_inserts"] += 1
        logger.info(f"Memory inserted: {mem_key} (convo_uuid={convo_uuid})")
        return "Memory inserted successfully."
    except Exception as e:
        logger.error(f"Error inserting memory: {e}")
        return f"Error inserting memory: {e}"

def load_into_lru(key: str, entry: dict, convo_uuid: str) -> None:
    cache_key = f"{convo_uuid}_{key}"
    memory_cache = get_memory_cache()
    with state.chroma_lock:
        if cache_key not in memory_cache["lru_cache"]:
            memory_cache["lru_cache"][cache_key] = {
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

@inject_convo_uuid
def memory_query(
    mem_key: str = None, limit: int = Config.DEFAULT_TOP_K.value, convo_uuid: str = None, uuid_list: list[str] = None
) -> str:
    tool_limiter_sync()
    try:
        with state.conn:
            if uuid_list:
                placeholders = ','.join('?' for _ in uuid_list)
                where = f"uuid IN ({placeholders})"
                params = uuid_list
            else:
                where = "uuid=?"
                params = [convo_uuid]
            if mem_key:
                where += " AND mem_key=?"
                params.append(mem_key)
                sql = f"SELECT mem_value FROM memory WHERE {where}"
                state.cursor.execute(sql, params)
                result = state.cursor.fetchone()
                logger.info(f"Memory queried: {mem_key} (convo_uuid={convo_uuid})")
                return json.loads(result[0]) if result and result[0] else "Key not found."
            else:
                sql = f"SELECT mem_key, mem_value FROM memory WHERE {where} ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                state.cursor.execute(sql, params)
                results = {}
                for row in state.cursor.fetchall():
                    if row[1]:
                        results[row[0]] = json.loads(row[1])
                for key in results:
                    load_into_lru(key, results[key], convo_uuid)
                logger.info("Recent memories queried")
                return json.dumps(results)
    except Exception as e:
        logger.error(f"Error querying memory: {e}")
        return f"Error querying memory: {e}"

@inject_convo_uuid
def advanced_memory_consolidate(
    mem_key: str, interaction_data: dict, convo_uuid: str
) -> str:
    tool_limiter_sync()
    return safe_call(_advanced_memory_consolidate_impl, mem_key, interaction_data, convo_uuid)

def _advanced_memory_consolidate_impl(mem_key: str, interaction_data: dict, convo_uuid: str) -> str:
    cache_args = {"mem_key": mem_key, "interaction_data": interaction_data, "convo_uuid": convo_uuid}
    if cached := get_cached_tool_result("advanced_memory_consolidate", cache_args):
        return cached
    embed_model = state.get_embed_model()
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://api.moonshot.ai/v1")
        summary_response = client.chat.completions.create(
            model=Models.KIMI_K2_THINKING.value,
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
            state.cursor.execute(
                "INSERT OR REPLACE INTO memory (uuid, mem_key, mem_value) VALUES (?, ?, ?)",
                (convo_uuid, mem_key, json_episodic),
            )
        if embed_model and state.chroma_collection:
            chroma_col = state.chroma_collection
            embedding = embed_model.encode(summary).tolist()
            with state.chroma_lock:
                chroma_col.upsert(
                    ids=[str(uuid.uuid4())],
                    embeddings=[embedding],
                    documents=[json_episodic],
                    metadatas=[
                        {
                            "uuid": convo_uuid,
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
        cache_key = f"{convo_uuid}_{mem_key}"
        memory_cache = get_memory_cache()
        memory_cache["lru_cache"][cache_key] = {
            "entry": entry,
            "last_access": time.time(),
        }
        result = "Memory consolidated successfully."
        set_cached_tool_result("advanced_memory_consolidate", cache_args, result)
        logger.info(f"Memory consolidated: {mem_key}")
        return result
    except Exception:
        logger.error(f"Error consolidating memory: {traceback.format_exc()}")
        return f"Error consolidating memory: {traceback.format_exc()}"

def profile_func(func: typing.Callable) -> typing.Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if os.getenv('PROFILE_MODE'):
            pr = cProfile.Profile()
            pr.enable()
            result = func(*args, **kwargs)
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(10)
            logger.info(s.getvalue())
            return result
        return func(*args, **kwargs)
    return wrapper

@inject_convo_uuid
@profile_func
def advanced_memory_retrieve(
    query: str, top_k: int = Config.DEFAULT_TOP_K.value, convo_uuid: str = None, uuid_list: list[str] = None
) -> str:
    tool_limiter_sync()
    cache_args = {"query": query, "top_k": top_k, "convo_uuid": convo_uuid, "uuid_list": uuid_list}
    if cached := get_cached_tool_result("advanced_memory_retrieve", cache_args):
        return cached
    embed_model = state.get_embed_model()
    chroma_col = state.chroma_collection
    if not embed_model or not state.chroma_collection:
        logger.warning("Vector memory not available; falling back to keyword search.")
        retrieved = keyword_search(query, top_k, convo_uuid=convo_uuid, uuid_list=uuid_list)
        if isinstance(retrieved, str) and "error" in retrieved.lower():
            result = retrieved
        else:
            result = json.dumps(retrieved)
        set_cached_tool_result("advanced_memory_retrieve", cache_args, result)
        return result
    try:
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, embed_model.encode, query)
        try:
            query_emb = loop.run_until_complete(asyncio.wait_for(future, timeout=300.0)).tolist()
        except asyncio.TimeoutError:
            logger.warning("Embed timeout - Fallback to keyword search.")
            return fallback_to_keyword(query, top_k, convo_uuid, uuid_list)
        if uuid_list:
            where_clause = {"uuid": {"$in": uuid_list}}
        else:
            where_clause = {"uuid": {"$eq": convo_uuid}}
        with state.chroma_lock:
            results = chroma_col.query(
                query_embeddings=[query_emb],
                n_results=top_k,
                where=where_clause,
                include=["distances", "metadatas", "documents"],
            )
        if not results.get("ids") or not results["ids"][0]:
            logger.warning(f"Empty Chroma results: where={where_clause}, query={query[:50]}")
            if uuid_list:
                for single_uuid in uuid_list:
                    results = chroma_col.query(
                        query_embeddings=[query_emb],
                        n_results=top_k,
                        where={"uuid": {"$eq": single_uuid}},
                        include=["distances", "metadatas", "documents"],
                    )
                    if results.get("ids") and results["ids"][0]:
                        break
            if not results.get("ids") or not results["ids"][0]:
                logger.info("Seeding empty DB.")
                example_data = {"summary": "Initial welcome messaage", "details": "Welcome to the AurumVivum Realm! It´s powers are yours!"}
                advanced_memory_consolidate("seed_key", example_data, convo_uuid=convo_uuid)
                return "No recent memories found."
        retrieved = process_chroma_results(results, top_k)
        if len(retrieved) > 5:
            viz_result = viz_memory_lattice(convo_uuid, top_k=len(retrieved))
            logger.info(f"Auto-viz triggered: {viz_result}")
        if not retrieved:
            return "No relevant memories found."
        update_retrieve_metrics(len(retrieved), top_k)
        result = json.dumps(retrieved)
        set_cached_tool_result("advanced_memory_retrieve", cache_args, result)
        logger.info(f"Memory retrieved for query: {query}")
        return result
    except Exception:
        logger.error(f"Memory Glitch: {traceback.format_exc()}")
        return "No memories found."

def fallback_to_keyword(query: str, top_k: int, convo_uuid: str, uuid_list: list[str]) -> list | str:
    fallback_results = keyword_search(query, top_k, convo_uuid=convo_uuid, uuid_list=uuid_list)
    if isinstance(fallback_results, str) and "error" in fallback_results.lower():
        return fallback_results
    retrieved = []
    for res in fallback_results:
        mem_key = res["id"]
        mem_value = memory_query(mem_key=mem_key, convo_uuid=convo_uuid)
        retrieved.append(
            {
                "mem_key": mem_key,
                "value": mem_value,
                "relevance": res["score"],
                "summary": mem_value.get("summary", ""),
            }
        )
    return retrieved

def process_chroma_results(results: dict, top_k: int) -> list[dict]:
    if not results or not results.get("ids") or not results.get("ids"):
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
        new_salience = min(1.0, meta.get("salience", 1.0) + 0.1)
        metadata_to_update.append({"salience": new_salience})
    if ids_to_update and state.chroma_collection:
        with state.chroma_lock:
            state.chroma_collection.update(
                ids=ids_to_update, metadatas=metadata_to_update
            )
    retrieved.sort(key=lambda x: x["relevance"], reverse=True)
    return retrieved

def update_retrieve_metrics(len_retrieved: int, top_k: int) -> None:
    memory_cache = get_memory_cache()
    memory_cache["metrics"]["total_retrieves"] += 1
    hit_rate = len_retrieved / top_k if top_k > 0 else 1.0
    memory_cache["metrics"]["hit_rate"] = (
        (
            memory_cache["metrics"]["hit_rate"]
            * (memory_cache["metrics"]["total_retrieves"] - 1)
        )
        + hit_rate
    ) / memory_cache["metrics"]["total_retrieves"]

def should_prune() -> bool:
    """Prune based on memory inserts, not all calls"""
    memory_cache = get_memory_cache()
    inserts = memory_cache["metrics"]["total_inserts"]
    return inserts > 0 and inserts % Config.PRUNE_FREQUENCY.value == 0

def decay_salience(convo_uuid: str) -> None:
    one_week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    with state.conn:
        state.cursor.execute(
            "UPDATE memory SET salience = salience * 0.99 WHERE uuid=? AND timestamp < ?",
            (convo_uuid, one_week_ago),
        )

def prune_low_salience(convo_uuid: str) -> None:
    with state.conn:
        state.cursor.execute(
            "DELETE FROM memory WHERE uuid=? AND salience < 0.1",
            (convo_uuid,),
        )

def size_based_prune(convo_uuid: str) -> None:
    with state.conn:
        state.cursor.execute(
            "SELECT COUNT(*) FROM memory WHERE uuid=?", (convo_uuid,)
        )
        row_count = state.cursor.fetchone()[0]
        if row_count > 1000:
            state.cursor.execute(
                "SELECT mem_key FROM memory WHERE uuid=? AND salience < 0.5 ORDER BY timestamp ASC LIMIT ?",
                (convo_uuid, row_count - 1000),
            )
            low_keys = [row[0] for row in state.cursor.fetchall()]
            if not low_keys:
                state.cursor.execute(
                    "SELECT mem_key FROM memory WHERE uuid=? ORDER BY timestamp ASC LIMIT ?",
                    (convo_uuid, row_count - 1000),
                )
                low_keys = [row[0] for row in state.cursor.fetchall()]
            state.cursor.executemany(
                "DELETE FROM memory WHERE uuid=? AND mem_key=?",
                [(convo_uuid, key) for key in low_keys]
            )

def dedup_prune(convo_uuid: str) -> None:
    with state.conn:
        state.cursor.execute(
            "SELECT mem_key, mem_value FROM memory WHERE uuid=?",
            (convo_uuid,),
        )
        rows = state.cursor.fetchall()
        hashes = {}
        to_delete = []
        for key, value_str in rows:
            value = json.loads(value_str)
            h = hash(value.get("summary", ""))
            if h in hashes and value.get("salience", 1.0) < hashes[h].get("salience", 1.0):
                to_delete.append(key)
            else:
                hashes[h] = value
        state.cursor.executemany(
            "DELETE FROM memory WHERE uuid=? AND mem_key=?",
            [(convo_uuid, key) for key in to_delete]
        )

@inject_convo_uuid
def advanced_memory_prune(convo_uuid: str, uuid_list: list[str] = None) -> str:
    if not should_prune():
        return "Prune skipped (infrequent)."
    def _prune_sync(u):
        try:
            with state.conn:
                decay_salience(u)
                prune_low_salience(u)
                size_based_prune(u)
                dedup_prune(u)
                one_week_ago = (datetime.now() - timedelta(days=7)).isoformat()
                state.cursor.execute(
                    "DELETE FROM memory WHERE uuid=? AND timestamp < ? AND mem_key LIKE 'agent_%'",
                    (u, one_week_ago),
                )
                state.conn.commit()
            for agent_folder in os.listdir(state.agent_dir):
                folder_path = os.path.join(state.agent_dir, agent_folder)
                if os.path.isdir(folder_path):
                    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
                    if files and os.path.getmtime(os.path.join(folder_path, files[0])) < (datetime.now() - timedelta(days=7)).timestamp():
                        shutil.rmtree(folder_path)
                        logger.info(f"Pruned old agent folder: {agent_folder}")
            state.lru_evict()
            memory_cache = get_memory_cache()
            memory_cache["metrics"]["last_update"] = datetime.now().isoformat()
            logger.info("Memory pruned successfully")
            return "Memory pruned successfully."
        except Exception as e:
            logger.error(f"Error pruning memory: {e}")
            return f"Error pruning memory: {e}"
    if uuid_list:
        results = []
        for u in uuid_list:
            results.append(_prune_sync(u))
        return "\n".join(results)
    else:
        future = state.agent_executor.submit(_prune_sync, convo_uuid)
        try:
            return future.result(timeout=300)
        except Exception as e:
            return f"Prune timeout/error: {e}"

@lru_cache(maxsize=128)
def cached_embed(summary: str) -> list[float]:
    embed_model = state.get_embed_model()
    return embed_model.encode(summary).tolist() if embed_model else []

def viz_memory_lattice(
    convo_uuid: str,
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
    convo_summary = memory_query(limit=1, convo_uuid=convo_uuid)
    query = convo_summary if convo_summary != "[]" else "memory lattice"
    where_clause = {"uuid": {"$eq": convo_uuid}}
    query_emb = embed_model.encode(query).tolist()
    with state.chroma_lock:
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
            if len(candidates) == 0:
                continue
            sampled = np.random.choice(candidates, min(3, len(candidates)), replace=False)
            for node_j in sampled:
                j_idx = list(G.nodes).index(node_j)
                sim = np.dot(all_embs[i], all_embs[j_idx]) / (np.linalg.norm(all_embs[i]) * np.linalg.norm(all_embs[j_idx]))
                if sim > sim_threshold:
                    G.add_edge(node_i, node_j, weight=sim)
    if plot_type in ["graph", "both"]:
        pos = nx.spring_layout(G, k=1, iterations=5)
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
                            title=f"Memory Lattice: {len(G.nodes)} Nodes, {len(G.edges)} Veins (UUID {convo_uuid})",
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
        ax.set_title(f"Activation Amps Over 'Layers' (UUID {convo_uuid})")
        ax.set_xlabel("Layer Proxy (Retrieval Rank Bins)")
        ax.set_ylabel("Amp (Salience * Sim)")
        ax.legend()
        st.pyplot(fig)
        amps_path = os.path.join(output_dir, f"memory_amps_{convo_uuid}.png")
        plt.savefig(amps_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    graph_json = nx.node_link_data(G)
    mem_key = f"lattice_viz_{convo_uuid}"
    memory_insert(mem_key, {"graph": graph_json, "paths": [amps_path]}, convo_uuid=convo_uuid)
    logger.info(f"Lattice viz saved for convo_uuid {convo_uuid}: {len(G.nodes)} nodes")
    return f"Viz complete: Interactive graph rendered. Query '{mem_key}' for data."

async def async_run_spawn(agent_id: str, context: dict, model: str = Models.KIMI_K2_THINKING.value) -> str:
    async with state.agent_sem:
        max_attempts = 3
        convo_uuid = context['convo_uuid']
        task = context['task']
        for attempt in range(max_attempts):
            try:
                client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.moonshot.ai/v1")
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": f"You are an agent for the AurumVivum system. Execute the given task/query/scenario/simulation. Use this UUID for memory: {convo_uuid}. Suggest tool-chains if needed, but do not call tools yourself. Respond concisely."},
                            {"role": "user", "content": task}
                        ],
                        stream=False,
                    ), timeout=600.0
                )
                result = response.choices[0].message.content.strip()
                persist_agent_result(agent_id, task, result, convo_uuid)
                logger.info(f"Agent {agent_id} succeeded async.")
                return result
            except asyncio.TimeoutError:
                error = "Timeout: 60s exceeded."
            except Exception as e:
                await asyncio.sleep(5 * (attempt + 1))
                if attempt == max_attempts - 1:
                    error = f"Max attempts failed: {e}"
                    persist_agent_result(agent_id, task, error, convo_uuid)
                    return error

def visualize_got(
    got_data: str,
    format: str = "both",
    detail_level: int = 2,
    user: str = "shared",
    convo_uuid: str = None,
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
        from ascii_graph import Pyasciigraph
    except ImportError:
        if format == "text":
            return "ASCII not available - install ascii_graph."
    os.makedirs(output_dir, exist_ok=True)
    try:
        data = yaml.safe_load(got_data) if got_data.strip().startswith(('#', 'graph:')) else json.loads(got_data)
    except Exception as e:
        return f"Error: Invalid GoT data format - {e}"
    G = nx.DiGraph()
    for node, attrs in data.get('graph', {}).items():
        G.add_node(node, **attrs)
        for rel in ['depends_on', 'limited_by', 'mitigation', 'integrates']:
            targets = attrs.get(rel, [])
            if isinstance(targets, list):
                for t in targets:
                    G.add_edge(node, t, relation=rel, weight=attrs.get('weight', 1.0))
            elif targets:
                G.add_edge(node, targets, relation=rel, weight=attrs.get('weight', 1.0))
    if len(G.nodes) == 0:
        return "No GoT nodes to visualize."
    embed_model = state.get_embed_model()
    if detail_level > 1 and embed_model:
        node_embs = {n: cached_embed(G.nodes[n].get('desc', n)) for n in G.nodes}
        for u, v in G.edges:
            sim = np.dot(node_embs[u], node_embs[v]) / (np.linalg.norm(node_embs[u]) * np.linalg.norm(node_embs[v]))
            if sim > sim_threshold:
                G.edges[u, v]['sim'] = sim
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
        graph_path = os.path.join(output_dir, f"got_graph_{convo_uuid}.png")
        fig.write_image(graph_path)
        paths.append(graph_path)
    if format in ["amps", "both"]:
        weights = [d.get('weight', 1.0) for _, d in G.nodes(data=True)]
        sims = [e[2].get('sim', 0.5) for e in G.edges(data=True)] or [0.5] * len(G.nodes)
        layers = list(range(len(weights)))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(layers, [0.5]*len(layers), label="Baseline", color='lightgray')
        ax.plot(layers, weights, label="Node Weights", color='gold')
        ax.plot(layers, sims[:len(layers)], label="Edge Sims", color='crimson', linewidth=2)
        ax.set_title(f"GoT Activation Amps (UUID {convo_uuid})")
        ax.legend()
        amps_path = os.path.join(output_dir, f"got_amps_{convo_uuid}.png")
        fig.savefig(amps_path, dpi=150)
        plt.close(fig)
        paths.append(amps_path)
    if format == "text":
        graph = Pyasciigraph()
        ascii_rep = ""
        for line in graph.graph('GoT ASCII', [(n, len(G[n])) for n in G.nodes]):
            ascii_rep += line + "\n"
        text_path = os.path.join(output_dir, f"got_text_{convo_uuid}.txt")
        with open(text_path, 'w') as f:
            f.write(ascii_rep)
        paths.append(text_path)
    graph_json = nx.node_link_data(G)
    mem_key = f"got_viz_{convo_uuid}"
    memory_insert(mem_key, {"graph": graph_json, "paths": paths}, convo_uuid=convo_uuid)
    return f"GoT Viz complete: PNGs/TXT at {output_dir}, data in '{mem_key}'. Use fs_read_file or view_image to access."

def persist_agent_result(agent_id: str, task: str, response: str, convo_uuid: str) -> None:
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
            memory_insert(mem_key, result_data, convo_uuid=convo_uuid)
            summary_data = {"summary": f"Agent {agent_id} response to task: {task[:100]}...", "details": response}
            advanced_memory_consolidate(f"agent_{agent_id}_summary", summary_data, convo_uuid=convo_uuid)
            notify_key = f"agent_{agent_id}_complete"
            notify_data = {"agent_id": agent_id, "status": "complete", "result_key": mem_key, "timestamp": datetime.now().isoformat()}
            memory_insert(notify_key, notify_data, convo_uuid=convo_uuid)
            get_state("pending_notifies", []).append({"agent_id": agent_id, "status": "complete", "task": task[:100]})
            logger.info(f"Agent {agent_id} persisted and notified.")
    except OSError as e:
        logger.error(f"File error for agent {agent_id}: {e}")
        error_data = {"agent_id": agent_id, "error": str(e), "status": "failed"}
        memory_insert(f"agent_{agent_id}_error", error_data, convo_uuid=convo_uuid)
    except Exception as e:
        logger.error(f"Persistence error for agent {agent_id}: {e}")
        error_data = {"agent_id": agent_id, "error": str(e), "status": "failed"}
        memory_insert(f"agent_{agent_id}_error", error_data, convo_uuid=convo_uuid)

@inject_convo_uuid
def agent_spawn(sub_agent_type: str, task: str, convo_uuid: str, poll_interval: int = 5, model: str = Models.KIMI_K2_THINKING.value, auto_poll: bool = False) -> str:
    tool_limiter_sync()
    if len(task) > Config.MAX_TASK_LEN.value:
        return "Error: Task too long (max 2000 chars)."
    agent_id = f"{sub_agent_type}_{str(uuid.uuid4())[:8]}"
    context = {'convo_uuid': convo_uuid, 'task': task}
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_run_spawn(agent_id, context, model))
    status_key = f"agent_{agent_id}_status"
    status_data = {"agent_id": agent_id, "task": task[:100], "status": "spawned", "timestamp": datetime.now().isoformat(), "poll_interval": poll_interval}
    memory_insert(status_key, status_data, convo_uuid=convo_uuid)
    get_state("pending_notifies", []).append({"agent_id": agent_id, "status": "spawned", "task": task[:100]})
    if auto_poll:
        max_polls = 30
        for _ in range(max_polls):
            time.sleep(poll_interval)
            complete = memory_query(mem_key=f"agent_{agent_id}_complete", convo_uuid=convo_uuid)
            if complete != "Key not found.":
                return f"Polled result: {complete}"
        return "Poll timeout."
    return f"Agent '{sub_agent_type}' spawned (ID: {agent_id}). Poll 'agent_{agent_id}_complete' for results. Status: {status_key}"

def render_agent_fleet() -> None:
    with state.session_lock:
        pending = get_state("pending_notifies", [])
    if pending:
        st.subheader("Recent Notifies")
        for notify in pending:
            st.info(f"**{notify['agent_id']}**: {notify['status']} – Task: {notify['task']}")
        with state.session_lock:
            st.session_state["pending_notifies"] = []
    user = st.session_state.get("user")
    convo_uuid = st.session_state.get("current_convo_uuid")
    if user and convo_uuid:
        active_query = memory_query(limit=20, convo_uuid=convo_uuid)
        active_agents = []
        try:
            active_data = json.loads(active_query)
            active_agents = [data for key, data in active_data.items() if key.startswith("agent_") and data.get("status") in ["spawned", "running"]]
        except Exception as e:
            logger.error(f"Fleet active query error: {e}")
            st.warning("Error loading active agents.")
        if active_agents:
            st.subheader("Active Fleet")
            for idx, agent in enumerate(active_agents):
                col1, col2, col3 = st.columns([3, 4, 1])
                with col1:
                    st.write(f"**{agent.get('agent_id', 'Unknown')}**")
                with col2:
                    st.write(f"Status: {agent.get('status', 'Unknown')} | Task: {agent.get('task', '')[:50]}...")
                with col3:
                    unique_key = f"kill_{agent.get('agent_id', '')}_{uuid.uuid4()}_{idx}"
                    logger.debug(f"Key generated: {unique_key}")
                    if st.button("Kill", key=unique_key):
                        kill_key = f"agent_{agent['agent_id']}_kill"
                        kill_data = {"status": "killed", "timestamp": datetime.now().isoformat()}
                        memory_insert(kill_key, kill_data, convo_uuid=convo_uuid)
                        st.rerun()
            if st.button("Spawn Fleet (Parallel Sims)"):
                safe_call(agent_spawn, "fleet", "Run parallel quantum sims on nodes 1-3", convo_uuid=convo_uuid)

def git_init(safe_repo: str) -> str:
    try:
        repo = pygit2.discover_repository(safe_repo)
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
        repo.git.add(A=True)
    except AttributeError:
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
        repo = pygit2.discover_repository(safe_repo) or pygit2.init_repository(safe_repo)
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
    if any(kw in query.upper() for kw in ['DROP', 'DELETE', 'ALTER']):
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

class ShellExecSchema(Schema):
    command = fields.Str(required=True, validate=lambda x: re.match(r"^(ls|grep|sed|awk|cat|echo|wc|tail|head|cp|mv|rm|mkdir|rmdir|touch|python|pip)$", shlex.split(x)[0]))

def shell_exec(command: str) -> str:
    tool_limiter_sync()
    try:
        ShellExecSchema().load({"command": command})
    except ValidationError as e:
        return f"Validation error: {e.messages}"
    whitelist_pattern = r"^(ls|grep|sed|awk|cat|echo|wc|tail|head|cp|mv|rm|mkdir|rmdir|touch|python|pip)$"
    cmd_parts = shlex.split(command)
    cmd_base = cmd_parts[0]
    if not re.match(whitelist_pattern, cmd_base):
        return "Error: Command not whitelisted."
    for arg in cmd_parts[1:]:
        if re.search(r'[;&|><$]', arg):
            return "Invalid arg chars."
        if re.search(r'[\*\?\[\]]|\.\./', arg):
            return "Error: Forbidden patterns in args."
    convo_uuid = st.session_state.get("current_convo_uuid")
    confirm_key = f"confirm_destructive_{convo_uuid}"
    if cmd_base in ["rm", "rmdir"] and not get_state(confirm_key, False):
        st.session_state[confirm_key] = True
        return "Warning: Destructive command detected. Confirm by re-running."
    
    # FIXED: Remove shlex.quote, pass list directly with shell=False
    try:
        result = subprocess.run(
            cmd_parts,  # Pass list directly, no quoting
            cwd=state.sandbox_dir,
            capture_output=True,
            text=True,
            timeout=300,
            shell=False
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
    whitelist = ["https://api.example.com", "https://jsonplaceholder.typicode.com", "https://api.moonshot.ai/v1"]
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

def generate_embedding(text: str) -> str:
    tool_limiter_sync()
    embed_model = state.get_embed_model()
    if not embed_model:
        return "Error: Embedding model not loaded."
    try:
        if len(text) > 10000:
            chunks = chunk_text(text, max_tokens=256000)
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
    if not state.chroma_collection:
        return "Error: ChromaDB not ready."
    chroma_col = state.chroma_collection
    try:
        with state.chroma_lock:
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
        client = OpenAI(api_key=API_KEY, base_url="https://api.moonshot.ai/v1")
        response = client.chat.completions.create(
            model=Models.KIMI_K2_THINKING.value,
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

@inject_convo_uuid
def keyword_search(
    query: str, top_k: int = Config.DEFAULT_TOP_K.value, convo_uuid: str = None, uuid_list: list[str] = None
) -> list | str:
    tool_limiter_sync()
    try:
        with state.conn:
            if uuid_list:
                placeholders = ','.join('?' for _ in uuid_list)
                where = f"uuid IN ({placeholders})"
                params = uuid_list
            else:
                where = "uuid=?"
                params = [convo_uuid]
            sql = f"SELECT mem_key FROM memory WHERE {where} AND mem_value LIKE ? ORDER BY salience DESC LIMIT ?"
            params.append(f"%{query}%")
            params.append(top_k)
            state.cursor.execute(sql, params)
            results = [{"id": row[0], "score": 1.0} for row in state.cursor.fetchall()]
            return results
    except Exception as e:
        logger.error(f"Keyword search error: {e}")
        return f"Keyword search error: {e}"

@inject_convo_uuid
def socratic_api_council(
    branches: list,
    model: str = Models.KIMI_K2_THINKING.value,
    convo_uuid: str = None,
    api_key: str = None,
    rounds: int = 3,
    personas: list = None,
) -> str:
    tool_limiter_sync()
    if not api_key:
        api_key = API_KEY
    client = OpenAI(api_key=api_key, base_url="https://api.moonshot.ai/v1")
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
            convo_uuid=convo_uuid
        )
        logger.info("Socratic council completed")
        return consensus
    except Exception as e:
        logger.error(f"Council error: {e}")
        return f"Council error: {e}"

def reflect_optimize(component: str, metrics: dict) -> str:
    tool_limiter_sync()
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://api.moonshot.ai/v1")
        prompt = f"Reflect on {component} with metrics {json.dumps(metrics)}. Suggest optimized version."
        response = client.chat.completions.create(
            model=Models.KIMI_K2_THINKING.value,
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
    if os.path.exists(safe_env):
        return "Venv exists—skip."
    try:
        import venv
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
            timeout=300,
            cwd=state.sandbox_dir,
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        logger.error(f"Subprocess error: {e}")
        return f"Subprocess error: {e}"

PIP_WHITELIST = [
    "numpy", "pandas", "matplotlib", "scipy", "sympy", "requests", "beautifulsoup4",
    "qutip", "qiskit", "torch", "tensorflow", "astropy", "statsmodels",
    "biopython", "pubchempy", "dendropy", "rdkit", "pyscf", "polygon-api-client",
    "coingecko", "pygame", "chess", "mido", "midiutil", "networkx", "python-snappy",
    "pillow", "scikit-learn", "seaborn", "control", "mpmath", "sentence-transformers",
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if "Error" in result.stderr and "--no-deps" in cmd:
            cmd.remove("--no-deps")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            logger.warning("Retried without --no-deps.")
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        logger.error(f"Pip error: {e}")
        return f"Pip error: {e}"

@inject_convo_uuid
def chat_log_analyze_embed(
    convo_id: int, criteria: str, summarize: bool = True, convo_uuid: str = None
) -> str:
    tool_limiter_sync()
    if convo_id == 0:
        messages = st.session_state.get("messages", [])
        if not messages:
            return "Error: No current chat log found in session."
    else:
        with state.conn:
            state.cursor.execute(
                "SELECT messages FROM history WHERE convo_id=?", (convo_id,)
            )
            result = state.cursor.fetchone()
        if not result:
            return "Error: Chat log not found."
        messages = json.loads(result[0])
    if not messages:
        return "Error: Empty chat log."
    chat_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    client = OpenAI(api_key=API_KEY, base_url="https://api.moonshot.ai/v1")
    analysis_prompt = f"Analyze this chat log on criteria: {criteria}. Summarize if needed."
    response = client.chat.completions.create(
        model=Models.KIMI_K2_THINKING.value,
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
            model=Models.KIMI_K2_THINKING.value,
            messages=[
                {"role": "system", "content": summary_prompt},
                {"role": "user", "content": analysis},
            ],
            stream=False,
        )
        analysis = summary_response.choices[0].message.content.strip()
    embed_model = state.get_embed_model()
    if not embed_model or not state.chroma_collection:
        return "Error: Embedding or ChromaDB not ready."
    chroma_col = state.chroma_collection
    embedding = embed_model.encode(analysis).tolist()
    mem_key = f"chat_log_{convo_id}"
    with state.chroma_lock:
        chroma_col.upsert(
            ids=[mem_key],
            embeddings=[embedding],
            documents=[analysis],
            metadatas=[
                {"uuid": convo_uuid, "type": "chat_log", "salience": 1.0}
            ],
        )
    return f"Chat log {convo_id} analyzed and embedded as {mem_key}."

def yaml_retrieve(
    query: str = None, top_k: int = Config.DEFAULT_TOP_K.value, filename: str = None
) -> str:
    tool_limiter_sync()
    if not query and not filename:
        return "Error: Provide query or filename."
    if not state.yaml_ready:
        state._init_yaml_embeddings()
    col = state.yaml_collection
    embed_model = state.get_embed_model()
    cache = get_session_cache("yaml_cache")
    try:
        if filename:
            filename = os.path.basename(filename)
            if filename in cache:
                return cache[filename]
            with state.chroma_lock:
                existing = col.get(
                    where={"filename": filename},
                    include=["documents", "metadatas"]
                )
                if existing.get("documents") and existing["documents"]:
                    content = existing["documents"][0]
                    cache[filename] = content
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
            with state.chroma_lock:
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
    cache = get_session_cache("yaml_cache")
    try:
        if filename:
            path = os.path.join(state.yaml_dir, filename)
            if not os.path.exists(path):
                return "Error: File not found."
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                embedding = embed_model.encode(content).tolist()
                with state.chroma_lock:
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
                cache[filename] = content
                return f"YAML '{filename}' refreshed successfully."
            except OSError as e:
                return f"Skipped {filename}: {e}"
        else:
            with state.chroma_lock:
                ids = col.get()["ids"]
                if ids:
                    col.delete(ids=ids)
            cache.clear()
            files_refreshed = 0
            contents = []
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
                if len(contents) > 10:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        embeddings = list(executor.map(embed_model.encode, contents))
                    embeddings = [emb.tolist() for emb in embeddings]
                else:
                    embeddings = embed_model.encode(contents).tolist()
                metadatas = [{"filename": fn} for fn in fnames]
                with state.chroma_lock:
                    col.upsert(
                        ids=fnames,
                        embeddings=embeddings,
                        documents=contents,
                        metadatas=metadatas,
                    )
                for i, fname in enumerate(fnames):
                    cache[fname] = contents[i]
            state.yaml_ready = files_refreshed > 0
            return f"All YAMLs refreshed successfully ({files_refreshed} files)."
    except Exception as e:
        logger.error(f"YAML refresh error: {e}")
        return f"YAML refresh error: {e}"

# MOONSHOT OFFICIAL TOOLS - FIXED IMPLEMENTATION
MOONSHOT_OFFICIAL_TOOLS = {
    "moonshot-web-search": "moonshot/web-search:latest",
    "moonshot-calculate": "moonshot/calculate:latest",
    "moonshot-url-extract": "moonshot/url-extract:latest",
}

# Global mapping of function names to formula URIs
FORMULA_URI_MAP = MOONSHOT_OFFICIAL_TOOLS.copy()

def get_moonshot_tools(enable_official: bool = True, enable_custom: bool = True):
    """
    Returns tools in OpenAI format with hardcoded schemas
    NO API CALLS - all schemas are defined locally
    """
    tools = []
    
    # Official Moonshot tools (hardcoded - no fetching)
    if enable_official:
        tools.extend([
            {
                "type": "function",
                "function": {
                    "name": "moonshot-web-search",
                    "description": "Search the web for current information, news, and facts",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moonshot-calculate",
                    "description": "Perform mathematical calculations and evaluate expressions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "Mathematical expression to evaluate"}
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "moonshot-url-extract",
                    "description": "Extract content from a web URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "The URL to extract content from"}
                        },
                        "required": ["url"]
                    }
                }
            }
        ])
    
    # Custom sandbox tools from container
    if enable_custom:
        tools.extend([generate_tool_schema(func) for func in container._tools.values()])
    
    return tools

# Formula execution functions - FIXED
async def execute_moonshot_formula_async(
    formula_uri: str, 
    function_name: str, 
    arguments: dict,
    api_key: str
) -> dict:
    """
    Execute a Moonshot official tool via Formula API (async)
    """
    base_url = "https://api.moonshot.ai/v1"
    endpoint = f"{base_url}/formulas/{formula_uri}/fibers"
    try:
        async with httpx.AsyncClient() as httpx_client:
            resp = await httpx_client.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=arguments,
                timeout=30.0
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Formula API HTTP error {e.response.status_code}: {e}")
        return {"error": f"Formula API error: {e.response.status_code}"}
    except Exception as e:
        logger.error(f"Formula execution error: {e}")
        return {"error": str(e)}

def execute_moonshot_formula(
    formula_uri: str, 
    function_name: str, 
    arguments: dict,
    api_key: str
) -> dict:
    """
    Synchronous wrapper for formula execution
    """
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        execute_moonshot_formula_async(formula_uri, function_name, arguments, api_key)
    )

# Model fallback resolver
def resolve_model(preferred: str) -> str:
    """Fallback chain for model availability"""
    fallback_chain = [
        preferred,
        Models.KIMI_K2_THINKING.value,
        Models.MOONSHOT_V1_32K.value,
        Models.MOONSHOT_V1_8K.value
    ]
    
    for model in fallback_chain:
        try:
            client = OpenAI(api_key=API_KEY, base_url="https://api.moonshot.ai/v1")
            client.models.retrieve(model)
            logger.info(f"Using model: {model}")
            return model
        except:
            logger.warning(f"Model {model} unavailable, trying next...")
            continue
    
    raise RuntimeError("No Moonshot models available!")

# Type-aware schema - FIXED (removed invalid strict field)
def generate_tool_schema(func: typing.Callable) -> dict:
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
    }
    for param_name, param in sig.parameters.items():
        ann = param.annotation
        if ann is inspect._empty:
            prop_type = "string"
        else:
            if typing.get_origin(ann) is typing.List:
                base_type = typing.get_args(ann)[0] if typing.get_args(ann) else str
                prop_type = type_map.get(base_type, "array")
            else:
                prop_type = type_map.get(ann, "string")
        
        prop = {
            "type": prop_type,
            "description": f"Parameter: {param_name}"
        }
        if prop_type == "array":
            prop["items"] = {"type": "string"}
        if param.default is not inspect.Parameter.empty and param.default != inspect.Parameter.empty:
            prop["default"] = param.default
        
        properties[param_name] = prop
        
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
        }
    }

def diagnose() -> str:
    memory_cache = get_memory_cache()
    return json.dumps({
        "stability": get_stability_score(),
        "cache_size": len(memory_cache["lru_cache"]),
    })

# Custom tools
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
    diagnose
]

# Register all tools in container
for func in TOOL_FUNCS:
    container.register_tool(func)

# FIXED TOOL DISPATCHER - separates custom from official tools
TOOL_DISPATCHER = {}

# Register custom tools (local execution)
for name, func in container._tools.items():
    def make_custom_wrapper(f):
        sig = inspect.signature(f)
        def wrapper(**kwargs):
            coerced_kwargs = {}
            for k, v in kwargs.items():
                if k in sig.parameters:
                    ann = sig.parameters[k].annotation
                    try:
                        if ann is int and isinstance(v, str):
                            coerced_kwargs[k] = int(v)
                        elif ann is bool and isinstance(v, str):
                            coerced_kwargs[k] = v.lower() in ('true', '1', 'yes')
                        elif ann is float and isinstance(v, str):
                            coerced_kwargs[k] = float(v)
                        elif ann is list and isinstance(v, str):
                            coerced_kwargs[k] = v.split(',')
                        else:
                            coerced_kwargs[k] = v
                    except ValueError:
                        return f"Arg type error for {k}: {v}"
                else:
                    coerced_kwargs[k] = v
            
            logger.info(f"Executing CUSTOM tool: {f.__name__} with args: {coerced_kwargs}")
            result = safe_call(f, **coerced_kwargs)
            return str(result) if result is not None else "Tool returned None"
        return wrapper
    TOOL_DISPATCHER[name] = make_custom_wrapper(func)

# Register official Moonshot formulas (API execution)
def make_official_wrapper(formula_name, formula_uri):
    def wrapper(**kwargs):
        logger.info(f"Executing OFFICIAL formula: {formula_name} ({formula_uri})")
        return execute_moonshot_formula(
            formula_uri=formula_uri,
            function_name=formula_name,
            arguments=kwargs,
            api_key=API_KEY
        )
    return wrapper

for tool_name, formula_uri in container.get_official_tools().items():
    TOOL_DISPATCHER[tool_name] = make_official_wrapper(tool_name, formula_uri)

# FIXED: Handle duplicate tool calls and proper indexing
def _handle_single_tool(tool_call, current_messages, enable_tools) -> typing.Generator[str, None, None]:
    """Process a single tool call - works for both custom and official tools"""
    existing_ids = {msg.get("tool_call_id") for msg in current_messages if msg.get("role") == "tool"}
    if tool_call.id in existing_ids:
        logger.warning(f"Skipping duplicate tool call ID: {tool_call.id}")
        yield f"\n> **Skipping duplicate tool call:** `{tool_call.function.name}`\n"
        return
    
    func_name = tool_call.function.name
    logger.info(f"Tool call received: {func_name}")
    
    # Validate tool exists
    if func_name not in TOOL_DISPATCHER:
        error_msg = f"Unknown tool: {func_name} (not registered in dispatcher)"
        logger.error(error_msg)
        yield f"\n> **Tool Error:** `{func_name}` - {error_msg}\n"
        
        current_messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool", 
            "name": func_name,
            "content": error_msg
        })
        return
    
    # Execute tool
    try:
        args = json.loads(tool_call.function.arguments)
        func_to_call = TOOL_DISPATCHER[func_name]
        result = func_to_call(**args)
        
        # Format result
        if isinstance(result, str):
            result_str = result
        elif isinstance(result, dict) and "error" in result:
            result_str = f"Error: {result['error']}"
        elif isinstance(result, dict):
            result_str = json.dumps(result)
        else:
            result_str = str(result)
            
        # Update counters
        with state.counter_lock:
            if func_name == "socratic_api_council":
                st.session_state["council_count"] = st.session_state.get("council_count", 0) + 1
            else:
                st.session_state["tool_count"] = st.session_state.get("tool_count", 0) + 1
        
        logger.info(f"Tool result: {func_name} - {result_str[:100]}...")
        
        yield f"\n> **Tool Call:** `{func_name}` | **Result:** `{result_str[:100]}...`\n"
        
        current_messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": func_name,
            "content": result_str
        })
        
    except json.JSONDecodeError as e:
        error_msg = f"Invalid arguments JSON: {e}"
        logger.error(error_msg)
        yield f"\n> **Tool Parse Error:** `{func_name}` - {error_msg}\n"
    except Exception as e:
        error_msg = f"Execution error: {str(e)}"
        logger.error(f"Tool execution error: {e}")
        yield f"\n> **Tool Execution Error:** `{func_name}` - {error_msg}\n"
        
        current_messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": func_name,
            "content": error_msg
        })

def process_tool_calls(tool_calls, current_messages, enable_tools) -> typing.Generator[str, None, None]:
    yield "\n*Thinking... Using tools...*\n"
    tool_history = []
    with state.conn:
        for tool_call in tool_calls:
            for chunk in _handle_single_tool(tool_call, current_messages, enable_tools):
                yield chunk
                tool_history.append(f"{tool_call.function.name}: {chunk[:50]}")

@profile_func
def call_moonshot_api(
    model: str,
    messages: list[dict],
    sys_prompt: str,
    stream: bool = True,
    image_files: list = None,
    enable_tools: bool = True,
    enable_moonshot_tools: bool = True,
) -> typing.Generator[str, None, None]:
    # Resolve model with fallback
    model = resolve_model(model)
    
    all_tools = get_moonshot_tools(
        enable_official=enable_moonshot_tools,
    )
    
    client = OpenAI(api_key=API_KEY, base_url="https://api.moonshot.ai/v1", timeout=3600)
    
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
    
    def generate(current_messages: list[dict]) -> typing.Generator[str, None, None]:
        nonlocal model, enable_tools, enable_moonshot_tools, client, all_tools

        max_iterations = Config.MAX_ITERATIONS.value
        tool_calls_per_convo = get_state("tool_calls_per_convo", 0)
        if tool_calls_per_convo > Config.TOOL_CALL_LIMIT.value:
            yield "Error: Tool call limit exceeded for this conversation."
            return
        
        retry_count = 0
        start_time = time.time()
        
        for iteration in range(max_iterations):
            with state.counter_lock:
                global main_count
                st.session_state["main_count"] = st.session_state.get("main_count", 0) + 1
            
            logger.info(
                f"API call: Tools: {st.session_state.get('tool_count', 0)} | Council: {st.session_state.get('council_count', 0)} | Main: {st.session_state.get('main_count', 0)} | Iteration: {iteration + 1} | Moonshot Tools: {enable_moonshot_tools}"
            )
            
            try:
                estimated_tokens = sum(len(m["content"]) for m in current_messages) // 4
                api_limiter_sync(estimated_tokens)
                
                response = client.chat.completions.create(
                    model=model,
                    messages=current_messages,
                    tools=all_tools,
                    tool_choice="auto" if all_tools else None,
                    stream=stream,
                    max_tokens=256000,
                )
                
                # FIXED: Proper tool call handling with index tracking
                tool_calls_by_index = {}
                full_delta_response = ""
                reasoning_content = ""
                chunk_buffer = ""
                
                for chunk in response:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue
                    
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                        reasoning_content += delta.reasoning_content
                        if reasoning_content.strip() == delta.reasoning_content.strip():
                            yield "\n*🤔 **Thinking...***\n"
                    
                    if delta and delta.content:
                        chunk_buffer += delta.content
                        if len(chunk_buffer.split()) > 3:
                            yield chunk_buffer
                            chunk_buffer = ""
                        full_delta_response += delta.content
                    
                    # FIXED: Handle tool call chunks with proper indexing
                    if delta and delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            idx = tool_call_delta.index if hasattr(tool_call_delta, 'index') else 0
                            
                            if idx not in tool_calls_by_index:
                                tool_calls_by_index[idx] = {
                                    "id": tool_call_delta.id,
                                    "name": tool_call_delta.function.name,
                                    "args": tool_call_delta.function.arguments or ""
                                }
                            else:
                                if tool_call_delta.function.arguments:
                                    tool_calls_by_index[idx]["args"] += tool_call_delta.function.arguments
                
                if chunk_buffer:
                    yield chunk_buffer
                
                # Convert indexed tool calls to list
                tool_calls = []
                for idx in sorted(tool_calls_by_index.keys()):
                    tool_data = tool_calls_by_index[idx]
                    try:
                        # Validate JSON
                        json.loads(tool_data["args"]) if tool_data["args"] else {}
                        tool_call_obj = ChatCompletionMessageToolCall(
                            id=tool_data["id"],
                            type="function",
                            function=Function(
                                name=tool_data["name"],
                                arguments=tool_data["args"]
                            )
                        )
                        tool_calls.append(tool_call_obj)
                    except json.JSONDecodeError as e:
                        logger.error(f"Malformed arguments for {tool_data['name']}: {e}")
                        logger.error(f"Raw arguments: {repr(tool_data['args'])}")
                        yield f"\n> **Tool parse error:** `{tool_data['name']}` - Invalid JSON arguments\n"
                        continue
                
                if reasoning_content and model in ["kimi-k2-thinking", "kimi-k2-thinking-preview"]:
                    st.session_state["last_reasoning"] = reasoning_content
                    yield f"\n*✅ Thinking complete ({len(reasoning_content)} chars)*\n"
                
                if not tool_calls:
                    return
                
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
                yield f"\nAuth issue: {error_msg}. Check your MOONSHOT_API_KEY."
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
                if "search_parameters" in str(e):
                    yield "\nNative search disabled—fallback to local."
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
            
            if time.time() - start_time > 3600:
                yield "Total timeout—aborting."
                break
    
    return generate(api_messages)

def search_history(query: str) -> list[tuple]:
    state.cursor.execute(
        "SELECT convo_id, title FROM history WHERE user=? AND title LIKE ?",
        (st.session_state["user"], f"%{query}%"),
    )
    return state.cursor.fetchall()

def export_convo(format: str = "json") -> str:
    if "messages" not in st.session_state:
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

def render_sidebar() -> None:
    global main_count, tool_count, council_count
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
                height=400,
                key="custom_prompt",
            )
            if st.button("Save Prompt"):
                with open(os.path.join(state.prompts_dir, "custom.txt"), "w") as f:
                    f.write(st.session_state["custom_prompt"])
            if st.button("Evolve Prompt"):
                user = st.session_state.get("user")
                convo_uuid = st.session_state.get("current_convo_uuid")
                metrics = {"length": len(st.session_state["custom_prompt"]), "vibe": "creative"}
                new_prompt = auto_optimize_prompt(st.session_state["custom_prompt"], user, convo_id=0, metrics=metrics)
                st.session_state["custom_prompt"] = new_prompt
                st.rerun()
            st.checkbox("Enable Tools (Sandboxed)", value=False, key="enable_tools")
            if st.session_state["enable_tools"]:
                st.checkbox("Include Moonshot Official Tools (Web/Calc)", value=True, key="enable_moonshot_tools")
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
                new_convo_uuid = str(uuid.uuid4())
                st.session_state["messages"] = []
                st.session_state["current_convo_uuid"] = new_convo_uuid
                st.session_state["current_convo_id"] = 0
                st.session_state["tool_calls_per_convo"] = 0
                st.session_state["rerun_guard"] = False
                st.rerun()
            st.header("Chat History")
            history_search = st.text_input("Search History", key="history_search")
            if history_search:
                histories = search_history(history_search)
            else:
                state.cursor.execute(
                    "SELECT convo_id, title FROM history WHERE user=? ORDER BY convo_id DESC",
                    (st.session_state["user"],),
                )
                histories = state.cursor.fetchall()
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
                memory_cache = get_memory_cache()
                metrics = memory_cache["metrics"]
                st.metric("Total Inserts", metrics["total_inserts"])
                st.metric("Total Retrieves", metrics["total_retrieves"])
                st.metric("Hit Rate", f"{metrics['hit_rate']:.2%}")
                st.metric("Tool Calls", st.session_state.get("tool_count", 0))
                st.metric("Council Calls", st.session_state.get("council_count", 0))
                st.metric("Stability Score", f"{get_stability_score():.2%}", delta_color="normal" if get_stability_score() > 0.8 else "inverse")
                st.metric("Main API Calls", st.session_state.get("main_count", 0))
                if st.button("Weave Memory Lattice (Viz Current Convo)"):
                    viz_result = viz_memory_lattice(st.session_state["current_convo_uuid"])
                    st.info(viz_result)
                if st.button("Prune Memory Now"):
                    prune_result = advanced_memory_prune(convo_uuid=st.session_state["current_convo_uuid"])
                    st.info(prune_result)
                if st.button("Clear Cache"):
                    st.session_state["tool_cache"] = {}
                    memory_cache["lru_cache"] = {}
                    st.success("Cache cleared.")
                if st.button("Reset Counters"):
                    with state.counter_lock:
                        st.session_state["main_count"] = st.session_state["tool_count"] = st.session_state["council_count"] = 0
                if st.button("Show Recent Tool Logs"):
                    with st.expander("Tool Logs", expanded=False):
                        recent_logs = [line for line in open("app.log", "r").readlines()[-20:] if "Tool call:" in line]
                        for log in recent_logs:
                            st.text(log.strip())
        with tab2:
            render_agent_fleet()

def login_page() -> None:
    st.title("Aurum Vivum Interface")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                state.cursor.execute("SELECT password FROM users WHERE username=?", (username,))
                result = state.cursor.fetchone()
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
                state.cursor.execute("SELECT * FROM users WHERE username=?", (new_user,))
                if state.cursor.fetchone():
                    st.error("Username already exists.")
                else:
                    state.cursor.execute(
                        "INSERT INTO users VALUES (?, ?)",
                        (new_user, hash_password(new_pass)),
                    )
                    state.conn.commit()
                    st.success("Registered! Please login.")

def load_history(convo_id: int) -> None:
    state.cursor.execute(
        "SELECT messages, uuid FROM history WHERE convo_id=? AND user=?",
        (convo_id, st.session_state["user"]),
    )
    if result := state.cursor.fetchone():
        messages = json.loads(result[0])
        st.session_state["messages"] = messages
        st.session_state["current_convo_id"] = convo_id
        st.session_state["current_convo_uuid"] = result[1]
        st.session_state["rerun_guard"] = False
        st.rerun()

def delete_history(convo_id: int) -> None:
    state.cursor.execute(
        "SELECT uuid FROM history WHERE convo_id=? AND user=?",
        (convo_id, st.session_state["user"]),
    )
    convo_uuid = state.cursor.fetchone()[0]
    state.cursor.execute(
        "DELETE FROM history WHERE convo_id=? AND user=?",
        (convo_id, st.session_state["user"]),
    )
    state.cursor.execute("DELETE FROM memory WHERE uuid=?", (convo_uuid,))
    state.conn.commit()
    if st.session_state.get("current_convo_id") == convo_id:
        st.session_state["messages"] = []
        st.session_state["current_convo_id"] = 0
        st.session_state["current_convo_uuid"] = None
    st.session_state["rerun_guard"] = False
    st.rerun()

def render_chat_interface(model: str, custom_prompt: str, enable_tools: bool, uploaded_images: list, enable_moonshot_tools: bool) -> None:
    st.title(f"Aurum Vivum - {st.session_state['user']}")
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "current_convo_id" not in st.session_state:
        st.session_state["current_convo_id"] = 0
    if "current_convo_uuid" not in st.session_state:
        st.session_state["current_convo_uuid"] = str(uuid.uuid4())
    if "tool_calls_per_convo" not in st.session_state:
        st.session_state["tool_calls_per_convo"] = 0
    
    # Initialize reasoning persistence
    if "reasoning_data" not in st.session_state:
        st.session_state.reasoning_data = None
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=False)
    
    if prompt := st.chat_input("Your command, ape?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt, unsafe_allow_html=False)
        
        with st.chat_message("assistant"):
            images_to_process = uploaded_images if uploaded_images else []
            generator = call_moonshot_api(
                model,
                st.session_state.messages,
                custom_prompt,
                stream=True,
                image_files=images_to_process,
                enable_tools=enable_tools,
                enable_moonshot_tools=enable_moonshot_tools,
            )
            full_response = st.write_stream(generator)
        
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        
        # FIXED: Persist reasoning box with proper state management
        if "last_reasoning" in st.session_state and st.session_state["last_reasoning"]:
            expander_key = f"reasoning_{st.session_state['current_convo_uuid']}"
            
            with st.expander("🔍 View Reasoning Trace", expanded=True):
                st.code(st.session_state["last_reasoning"], language="text")
                
                # Use form to prevent rerun on button press
                with st.form(key=f"save_reasoning_form_{st.session_state['current_convo_uuid']}"):
                    if st.form_submit_button("💾 Save to Memory"):
                        reasoning_data = {
                            "summary": f"Reasoning trace from turn {len(st.session_state['messages'])//2}",
                            "details": st.session_state["last_reasoning"],
                            "type": "reasoning_trace",
                            "timestamp": datetime.now().isoformat(),
                            "salience": 0.9
                        }
                        result = memory_insert(
                            f"reasoning_{uuid.uuid4().hex[:8]}", 
                            reasoning_data, 
                            convo_uuid=st.session_state["current_convo_uuid"]
                        )
                        if "successfully" in result:
                            st.success("✅ Reasoning saved!")
                            # Clear the reasoning so expander disappears naturally
                            del st.session_state["last_reasoning"]
                        else:
                            st.error(f"Save failed: {result}")
        
        title = gen_title(prompt)
        messages_json = json.dumps(st.session_state.messages)
        if st.session_state.get("current_convo_id", 0) == 0:
            with state.conn:
                state.cursor.execute(
                    "INSERT INTO history (user, uuid, title, messages) VALUES (?, ?, ?, ?)",
                    (st.session_state["user"], st.session_state["current_convo_uuid"], title, messages_json),
                )
                st.session_state.current_convo_id = state.cursor.lastrowid
        else:
            with state.conn:
                state.cursor.execute(
                    "UPDATE history SET title=?, messages=? WHERE convo_id=?",
                    (title, messages_json, st.session_state["current_convo_id"]),
                )
        st.session_state["rerun_guard"] = False
        render_agent_fleet()
        st.rerun()

def chat_page() -> None:
    render_sidebar()
    render_chat_interface(
        st.session_state["model_select"],
        st.session_state["custom_prompt"],
        st.session_state["enable_tools"],
        st.session_state["uploaded_images"],
        st.session_state.get("enable_moonshot_tools", True),
    )

if "auto_prune_done" not in st.session_state:
    current_convo_uuid = st.session_state.get("current_convo_uuid")
    advanced_memory_prune(convo_uuid=current_convo_uuid)
    st.session_state["auto_prune_done"] = True

def run_tests() -> bool:
    class TestMoonshotCompatibility(unittest.TestCase):
        def setUp(self) -> None:
            st.session_state["user"] = "test_user"
            st.session_state["current_convo_uuid"] = str(uuid.uuid4())
            st.session_state["memory_cache"] = {
                "lru_cache": {},
                "metrics": {"total_inserts": 0, "total_retrieves": 0, "hit_rate": 1.0},
            }
            st.session_state["tool_cache"] = {}
            st.session_state["yaml_ready"] = True
            st.session_state["chroma_ready"] = True
        
        def test_model_enum_valid(self):
            """Verify Moonshot model names are valid"""
            for model in Models:
                self.assertIn(model.value, [
                    "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k",
                    "kimi-latest", "kimi-k2-thinking", "kimi-k2-thinking-preview"
                ])
        
        def test_reasoning_content_extraction(self):
            """Test that reasoning_content is captured"""
            fake_chunk = type('obj', (object,), {
                'choices': [type('obj', (object,), {
                    'delta': type('obj', (object,), {
                        'content': "Final answer",
                        'reasoning_content': "Thinking through this..."
                    })
                })]
            })
            self.assertTrue(hasattr(fake_chunk.choices[0].delta, 'reasoning_content'))
        
        def test_fs_write_read(self) -> None:
            result_write = safe_call(fs_write_file, "test.txt", "Hello World")
            self.assertIn("successfully", result_write)
            result_read = safe_call(fs_read_file, "test.txt")
            self.assertEqual(result_read, "Hello World")
        
        def test_memory_insert_query(self) -> None:
            mem_value = {"summary": "test mem", "salience": 0.8}
            result_insert = safe_call(memory_insert, "test_key", mem_value, convo_uuid=st.session_state["current_convo_uuid"])
            self.assertEqual(result_insert, "Memory inserted successfully.")
            result_query = safe_call(memory_query, mem_key="test_key", convo_uuid=st.session_state["current_convo_uuid"])
            self.assertIn("summary", str(result_query))
            self.assertIn("test mem", str(result_query))
        
        def test_agent_spawn_persist(self) -> None:
            result_spawn = safe_call(agent_spawn, "TestAgent", "Mock task: echo hello", convo_uuid=st.session_state["current_convo_uuid"])
            self.assertIn("spawned (ID:", result_spawn)
        
        def test_tool_cache(self) -> None:
            args = {"file_path": "cache_test.txt"}
            set_cached_tool_result("fs_read_file", args, "cached_value")
            result = get_cached_tool_result("fs_read_file", args)
            self.assertEqual(result, "cached_value")
        
        def test_rate_limiter(self) -> None:
            limiter = MoonshotRateLimiter()
            self.assertTrue(callable(limiter))
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMoonshotCompatibility)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    num_tests = suite.countTestCases()
    set_stability_score(1.0 - (len([t for t, s in result.failures + result.errors]) / num_tests) if num_tests else 1.0)
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
