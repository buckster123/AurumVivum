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
import venv
import xml.dom.minidom
from datetime import datetime, timedelta
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
from black import FileMode, format_str
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from passlib.hash import sha256_crypt
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt  # Fallback for non-Plotly
# V3: Plotly for interactive viz
import plotly.graph_objects as go
# V3: Enums
from enum import Enum
import inspect
from typing import Any  # Kept minimal
import hashlib
import ast  # For RestrictedPython policy
# Phase 1: New imports for resilience
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, ValidationError
# Phase 2: For cache eviction
import heapq

nest_asyncio.apply()

# Setup logging
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
LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY")
if not LANGSEARCH_API_KEY:
    st.warning("LANGSEARCH_API_KEY not set in .env—web search tool will fail.")

# V3: Enums/Consts for magic nums
class Config(Enum):
    DEFAULT_TOP_K = 5
    CACHE_TTL_MIN = 15
    AGENT_MAX_CONCURRENT = 5
    PRUNE_FREQUENCY = 50
    SIM_THRESHOLD = 0.6
    MAX_TASK_LEN = 2000
    STABILITY_PENALTY = 0.05
    MAX_ITERATIONS = 100
    TOOL_CALL_LIMIT = 100

class Models(Enum):
    GROK_FAST_REASONING = "grok-4-1-fast-reasoning"
    GROK_LATEST = "grok-4-latest"
    GROK_CODE_FAST = "grok-code-fast-1"
    GROK_3_MINI = "grok-3-mini"

# Phase 1: Pydantic for mem_value
class MemValue(BaseModel):
    summary: str = ""
    details: str = ""
    tags: list[str] = []
    domain: str = "general"
    timestamp: str
    salience: float = 1.0

# V2: AppState Singleton (unchanged from v2)
class AppState:
    def __init__(self):
        self.conn = sqlite3.connect("sandbox/db/chatapp.db", check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.c = self.conn.cursor()
        self._init_db()
        
        # Chroma init
        try:
            self.chroma_client = chromadb.PersistentClient(path="./sandbox/db/chroma_db")
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name="memory_vectors",
                metadata={"hnsw:space": "cosine"},
            )
            self.chroma_ready = True
        except Exception as e:
            logger.warning(f"ChromaDB init failed ({e}). Vector search disabled.")
            st.warning(f"ChromaDB init failed ({e}). Vector search disabled.")
            self.chroma_ready = False
            self.chroma_collection = None

        # Memory cache
        self.memory_cache = {
            "lru_cache": {},
            "vector_store": [],
            "metrics": {
                "total_inserts": 0,
                "total_retrieves": 0,
                "hit_rate": 1.0,
                "last_update": None,
            },
        }

        # Phase 2: Tool cache heap for age-based eviction
        self.tool_cache_heap = []

        # Prompts Directory
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

        # Sandbox Directory
        self.sandbox_dir = "./sandbox"
        os.makedirs(self.sandbox_dir, exist_ok=True)
        # YAML Directory
        self.yaml_dir = "./sandbox/evo_data/modules/aurum"
        os.makedirs(self.yaml_dir, exist_ok=True)
        # Agent FS base dir
        self.agent_dir = os.path.join(self.sandbox_dir, "agents")
        os.makedirs(self.agent_dir, exist_ok=True)

        # Global thread pool
        self.agent_executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.agent_lock = threading.Lock()

        # V2: Async semaphore for agents
        self.agent_sem = asyncio.Semaphore(Config.AGENT_MAX_CONCURRENT.value)

        # Embed model lazy
        self.embed_model = None

        # V2: Stability score (updated in safe_call)
        self.stability_score = 1.0

        # YAML init
        self.yaml_collection = self.chroma_client.get_or_create_collection(
            name="yaml_vectors", metadata={"hnsw:space": "cosine"}
        )
        self.yaml_cache = {}
        self.yaml_ready = False  # Set after embed load
        self._init_yaml_embeddings()

    def _init_db(self):
        """Lazy init for DB tables."""
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
        self.conn.commit()

    # Phase 2: Optimized YAML embeddings with hash caching
    def _init_yaml_embeddings(self):
        """Init YAML embeddings once embed_model ready, with file change detection."""
        embed_model = self.get_embed_model()
        if not embed_model:
            logger.warning("YAML embeddings skipped – embed model not ready.")
            return
        # Hash dir contents for change detection
        dir_hash = hashlib.sha256()
        yaml_files = sorted([f for f in os.listdir(self.yaml_dir) if f.endswith(".yaml")])
        for fname in yaml_files:
            path = os.path.join(self.yaml_dir, fname)
            try:
                with open(path, "rb") as f:
                    dir_hash.update(f.read())
            except Exception as e:
                logger.error(f"Hash error for {fname}: {e}")
        current_hash = dir_hash.hexdigest()
        session_hash = st.session_state.get("yaml_hash", None)
        if current_hash == session_hash and self.yaml_ready:
            logger.info("YAMLs unchanged—using cached embeddings.")
            return
        # Full refresh if changed or first run
        try:
            self.yaml_collection.delete(where={})  # Clear old
        except Exception as e:
            logger.warning(f"Could not clear old YAML collection: {e}")
        self.yaml_cache = {}
        files_refreshed = 0
        for fname in yaml_files:
            path = os.path.join(self.yaml_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                embedding = embed_model.encode(content).tolist()
                self.yaml_collection.upsert(
                    ids=[fname],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[{"filename": fname}],
                )
                self.yaml_cache[fname] = content
                files_refreshed += 1
            except Exception as e:
                logger.error(f"YAML embed error for {fname}: {e}")
        self.yaml_ready = True
        st.session_state["yaml_hash"] = current_hash  # Persist across refreshes
        logger.info(f"YAML embeddings refreshed ({files_refreshed} files).")

    @classmethod
    def get(cls):
        if "app_state" not in st.session_state:
            st.session_state["app_state"] = cls()
        return st.session_state["app_state"]
    
    def get_embed_model(self):
        if self.embed_model is None:
            try:
                with st.spinner("Loading embedding model for advanced memory (first-time use)..."):
                    self.embed_model = SentenceTransformer("all-mpnet-base-v2")
                st.info("Embedding model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                st.error(f"Failed to load embedding model: {e}")
                self.embed_model = None
        return self.embed_model

    def lru_evict(self):
        """Evict low-priority from memory LRU cache."""
        if len(self.memory_cache["lru_cache"]) > 500:
            lru_items = sorted(
                self.memory_cache["lru_cache"].items(),
                key=lambda x: x[1]["last_access"],
            )
            num_to_evict = len(lru_items) - 500
            for key, _ in lru_items[:num_to_evict]:
                entry = self.memory_cache["lru_cache"][key]["entry"]
                if entry["salience"] < 0.4:
                    del self.memory_cache["lru_cache"][key]

# Call init on load
state = AppState.get()

# === SECTION: UI Styling (Unchanged) ===
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

# === SECTION: Helper Functions (Added gen_title) ===
def get_state(key, default=None):
    """Safe session state access/mutation."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# Phase 1: Enhanced safe_call with tenacity and structured returns
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ValueError, sqlite3.Error, asyncio.TimeoutError, requests.RequestException))
)
def safe_call(func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        state.stability_score = min(1.0, state.stability_score + 0.01)
        return {"success": True, "result": result}
    except Exception as e:
        full_trace = traceback.format_exc()
        logger.error(f"Full trace for {func.__name__}: {full_trace}")
        error_type = type(e).__name__
        user_msg = f"Oops: {str(e)[:100]}... (Type: {error_type}). Check logs."
        state.stability_score = max(0.0, state.stability_score - Config.STABILITY_PENALTY.value)
        return {"success": False, "error_type": error_type, "details": full_trace, "user_msg": user_msg}

def hash_password(password):
    return sha256_crypt.hash(password)

def verify_password(stored, provided):
    return sha256_crypt.verify(provided, stored)

# Tool Cache Helper (V3: Use Enum TTL)
def get_tool_cache_key(func_name, args):
    arg_str = json.dumps(args, sort_keys=True)
    return f"tool_cache:{func_name}:{hashlib.sha256(arg_str.encode()).hexdigest()}"

def get_cached_tool_result(func_name, args, ttl_minutes=Config.CACHE_TTL_MIN.value):  # V3: Enum
    cache = get_state("tool_cache", {})
    key = get_tool_cache_key(func_name, args)
    if key in cache:
        timestamp, result = cache[key]
        if (datetime.now() - timestamp).total_seconds() / 60 < ttl_minutes:
            return result
    return None

# Phase 2: Age-based eviction with heapq
def set_cached_tool_result(func_name, args, result):
    cache = get_state("tool_cache", {})
    key = get_tool_cache_key(func_name, args)
    expiry = datetime.now() + timedelta(minutes=Config.CACHE_TTL_MIN.value)
    cache[key] = (datetime.now(), result)
    heapq.heappush(state.tool_cache_heap, (expiry.timestamp(), key))
    if len(cache) > 100:
        while state.tool_cache_heap:
            exp_ts, old_key = heapq.heappop(state.tool_cache_heap)
            if datetime.fromtimestamp(exp_ts) > datetime.now():  # Still valid? Push back
                heapq.heappush(state.tool_cache_heap, (exp_ts, old_key))
                break
            if old_key in cache:
                del cache[old_key]
    st.session_state["tool_cache"] = cache

def load_prompt_files():
    return [f for f in os.listdir(state.prompts_dir) if f.endswith(".txt")]

# V3: Auto-title gen
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
        return first_msg[:40] + "..."  # Fallback

# V3: Prompt optimize
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
        # Chain to reflect
        optimized = reflect_optimize("prompt", {"consensus": consensus, "metrics": metrics})
        return optimized.split("Optimized prompt:")[-1].strip() if "Optimized prompt:" in optimized else current_prompt
    except Exception as e:
        logger.error(f"Prompt optimize error: {e}")
        return current_prompt

# === SECTION: Tool Functions (Sandboxed, State-Aware; Updated w/ Enums) ===
# FS Tools (unchanged)
@st.cache_data(ttl=300)
def fs_read_file(file_path: str) -> str:
    safe_path = os.path.abspath(os.path.normpath(os.path.join(state.sandbox_dir, file_path)))
    if not safe_path.startswith(os.path.abspath(state.sandbox_dir)):
        return "Error: Path is outside the sandbox."
    try:
        with open(safe_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Error: File not found."
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return f"Error reading file: {e}"

def fs_write_file(file_path: str, content: str) -> str:
    safe_path = os.path.abspath(os.path.normpath(os.path.join(state.sandbox_dir, file_path)))
    if not safe_path.startswith(os.path.abspath(state.sandbox_dir)):
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
    safe_dir = os.path.abspath(os.path.normpath(os.path.join(state.sandbox_dir, dir_path)))
    if not safe_dir.startswith(os.path.abspath(state.sandbox_dir)):
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
    safe_path = os.path.abspath(os.path.normpath(os.path.join(state.sandbox_dir, dir_path)))
    if not safe_path.startswith(os.path.abspath(state.sandbox_dir)):
        return "Error: Path is outside the sandbox."
    try:
        os.makedirs(safe_path, exist_ok=True)
        return f"Directory '{dir_path}' created successfully."
    except Exception as e:
        logger.error(f"Error creating directory: {e}")
        return f"Error creating directory: {e}"

# Time Tool (unchanged)
def get_current_time(sync: bool = False, format: str = "iso") -> str:
    try:
        if sync:
            ntp_client = ntplib.NTPClient()
            response = ntp_client.request("pool.ntp.org", version=3)
            dt_object = datetime.fromtimestamp(response.tx_time)
        else:
            dt_object = datetime.now()
        if format == "human":
            return dt_object.strftime("%A, %B %d, %Y %I:%M:%S %p")
        elif format == "json":
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

# Code Tools (V3: Unchanged from v2)
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
}

def init_repl_namespace():
    if "repl_namespace" not in st.session_state:
        st.session_state["repl_namespace"] = {"__builtins__": SAFE_BUILTINS.copy()}
        st.session_state["repl_namespace"].update(ADDITIONAL_LIBS)

# V2: Custom policy to ban imports
def restricted_policy(node):
    if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
        return False  # Ban imports
    return True

def execute_in_venv(code: str, venv_path: str) -> str:
    safe_venv = os.path.abspath(os.path.normpath(os.path.join(state.sandbox_dir, venv_path)))
    if not safe_venv.startswith(os.path.abspath(state.sandbox_dir)):
        return "Error: Venv path outside sandbox."
    venv_python = os.path.join(safe_venv, "bin", "python")
    if not os.path.exists(venv_python):
        return "Error: Venv Python not found."
    result = subprocess.run(
        [venv_python, "-c", code], capture_output=True, text=True, timeout=30
    )
    output = result.stdout
    if result.stderr:
        logger.error(f"Venv execution error: {result.stderr}")
        return f"Error: {result.stderr}"
    return output

def execute_local(code: str, redirected_output: io.StringIO) -> str:
    try:
        # V2: Apply policy
        tree = ast.parse(code)
        if not all(restricted_policy(node) for node in ast.walk(tree)):
            return "Restricted: Imports banned."
        result = RestrictedPython.compile_restricted_exec(code)
        if result.errors:
            return f"Restricted compile error: {result.errors}"
        exec(result.code, st.session_state["repl_namespace"], {})
    except Exception as e:
        return f"Restricted exec error: {e}"
    return redirected_output.getvalue()

def code_execution(code: str, venv_path: str = None) -> str:
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

# Memory Tools (V3: Use Enum for limits)
def memory_insert(
    mem_key: str, mem_value: dict, user: str = None, convo_id: int = None
) -> str:
    if user is None or convo_id is None:
        return "Error: user and convo_id required for memory insert."
    # V3.1: Perma-fix – coerce str to dict if possible (handles tool-arg stringify edges)
    if isinstance(mem_value, str):
        try:
            mem_value = json.loads(mem_value)
            logger.info(f"Coerced str to dict for mem_key '{mem_key}'")  # Debug crumb
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse mem_value str as JSON for '{mem_key}': {e}")
            return "Error: mem_value string is invalid JSON; must be dict or valid JSON str."
    if not isinstance(mem_value, dict):
        return "Error: mem_value must be a dict (or valid JSON str)."
    # Phase 1: Pydantic validation
    try:
        validated = MemValue(**mem_value)  # Validates/coerces
        mem_value = validated.model_dump()  # Back to dict
    except ValidationError as e:
        return safe_call(lambda: f"Schema error: {e}")["user_msg"]
    try:
        state.c.execute(
            "INSERT OR REPLACE INTO memory (user, convo_id, mem_key, mem_value) VALUES (?, ?, ?, ?)",
            (user, convo_id, mem_key, json.dumps(mem_value)),
        )
        state.conn.commit()
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
        logger.info(f"Memory inserted: {mem_key}")
        return "Memory inserted successfully."
    except Exception as e:
        logger.error(f"Error inserting memory: {e}")
        return f"Error inserting memory: {e}"

# Phase 3: Type guard for load_into_lru
def load_into_lru(key, entry, user, convo_id):
    if not isinstance(entry, dict):
        logger.warning(f"Skipping LRU load for {key}: not dict")
        return
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

def memory_query(
    mem_key: str = None, limit: int = Config.DEFAULT_TOP_K.value, user: str = None, convo_id: int = None  # V3: Enum
) -> str:
    if user is None or convo_id is None:
        return "Error: user and convo_id required for memory query."
    try:
        if mem_key:
            state.c.execute(
                "SELECT mem_value FROM memory WHERE user=? AND convo_id=? AND mem_key=?",
                (user, convo_id, mem_key),
            )
            result = state.c.fetchone()
            logger.info(f"Memory queried: {mem_key}")
            return json.loads(result[0]) if result and result[0] else "Key not found."
        else:
            state.c.execute(
                "SELECT mem_key, mem_value FROM memory WHERE user=? AND convo_id=? ORDER BY timestamp DESC LIMIT ?",
                (user, convo_id, limit),
            )
            results = {}
            for row in state.c.fetchall():
                if row[1]:  # V2: Skip empty
                    results[row[0]] = json.loads(row[1])
            for key in results:
                load_into_lru(key, results[key], user, convo_id)
            logger.info("Recent memories queried")
            return json.dumps(results)
    except Exception as e:
        logger.error(f"Error querying memory: {e}")
        return f"Error querying memory: {e}"

# Advanced Memory Consolidate (unchanged)
def advanced_memory_consolidate(
    mem_key: str, interaction_data: dict, user: str = None, convo_id: int = None
) -> str:
    return safe_call(_advanced_memory_consolidate_impl, mem_key, interaction_data, user, convo_id).get("result", safe_call(lambda: "Fallback fail")["user_msg"])

def _advanced_memory_consolidate_impl(mem_key, interaction_data, user, convo_id):
    if user is None or convo_id is None:
        return "Error: user and convo_id required."
    cache_args = {"mem_key": mem_key, "interaction_data": interaction_data}
    if cached := get_cached_tool_result("advanced_memory_consolidate", cache_args):
        return cached
    embed_model = state.get_embed_model()
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
        summary_response = client.chat.completions.create(
            model=Models.GROK_FAST_REASONING.value,  # V3: Enum
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
        state.c.execute(
            "INSERT OR REPLACE INTO memory (user, convo_id, mem_key, mem_value) VALUES (?, ?, ?, ?)",
            (user, convo_id, mem_key, json_episodic),
        )
        state.conn.commit()
        if (
            embed_model
            and state.chroma_ready
            and state.chroma_collection
        ):
            chroma_col = state.chroma_collection
            embedding = embed_model.encode(summary).tolist()
            chroma_col.upsert(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                documents=[json_episodic],
                metadatas=[
                    {
                        "user": user,
                        "convo_id": convo_id,
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
        logger.info(f"Memory consolidated: {mem_key}")
        return result
    except Exception:
        logger.error(f"Error consolidating memory: {traceback.format_exc()}")
        return f"Error consolidating memory: {traceback.format_exc()}"

# Advanced Memory Retrieve (V3: Enum top_k, unchanged else)
def advanced_memory_retrieve(
    query: str, top_k: int = Config.DEFAULT_TOP_K.value, user: str = None, convo_id: int = None  # V3: Enum
) -> str:
    if user is None or convo_id is None:
        return "Error: user and convo_id required."
    cache_args = {"query": query, "top_k": top_k}
    if cached := get_cached_tool_result("advanced_memory_retrieve", cache_args):
        return cached
    embed_model = state.get_embed_model()
    chroma_col = state.chroma_collection
    if not embed_model or not state.chroma_ready or not chroma_col:
        logger.warning("Vector memory not available; falling back to keyword search.")
        st.warning("Vector memory not available; falling back to keyword search.")
        retrieved = fallback_to_keyword(query, top_k, user, convo_id)
        result = json.dumps(retrieved)
        set_cached_tool_result("advanced_memory_retrieve", cache_args, result)
        return result
    try:
        query_emb = embed_model.encode(query).tolist()
        where_clause = {
            "$and": [
                {"user": {"$eq": user}},
                {"convo_id": {"$eq": int(convo_id)}},  # V2: Strict cast
            ]
        }
        results = chroma_col.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=where_clause,
            include=["distances", "metadatas", "documents"],
        )
        if not results.get("ids") or not results["ids"][0]:  # V2: Log empty
            logger.warning(f"Empty Chroma results: where={where_clause}, query={query[:50]}")
        retrieved = process_chroma_results(results, top_k)
        if len(retrieved) > 5:
            viz_result = viz_memory_lattice(user, convo_id, top_k=len(retrieved))
            logger.info(f"Auto-viz triggered: {viz_result}")
        if not retrieved:
            return "No relevant memories found."
        update_retrieve_metrics(len(retrieved), top_k)
        result = json.dumps(retrieved)
        set_cached_tool_result("advanced_memory_retrieve", cache_args, result)
        logger.info(f"Memory retrieved for query: {query}")
        return result
    except Exception:
        logger.error(f"Error retrieving memory: {traceback.format_exc()}")
        return (
            f"Error retrieving memory: {traceback.format_exc()}. "
            "If this is a where clause issue, check filter structure."
        )

# Fallback and Process (unchanged)
def fallback_to_keyword(query: str, top_k: int, user: str, convo_id: int) -> list:
    fallback_results = keyword_search(query, top_k, user, convo_id)
    if isinstance(fallback_results, str) and "error" in fallback_results.lower():
        return fallback_results
    retrieved = []
    for res in fallback_results:
        mem_key = res["id"]
        value = memory_query(mem_key=mem_key, user=user, convo_id=convo_id)
        retrieved.append(
            {
                "mem_key": mem_key,
                "value": value,
                "relevance": res["score"],
                "summary": value.get("summary", ""),
            }
        )
    return retrieved

def process_chroma_results(results, top_k: int) -> list:
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
        metadata_to_update.append({"salience": meta.get("salience", 1.0) + 0.1})
    if ids_to_update:
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

# Prune (V3: Use Enum frequency)
def should_prune() -> bool:
    if "prune_counter" not in st.session_state:
        st.session_state["prune_counter"] = 0
    st.session_state["prune_counter"] += 1
    return st.session_state["prune_counter"] % Config.PRUNE_FREQUENCY.value == 0  # V3: Enum

def decay_salience(user: str, convo_id: int):
    one_week_ago = datetime.now() - timedelta(days=7)
    state.c.execute(
        "UPDATE memory SET salience = salience * 0.99 WHERE user=? AND convo_id=? AND timestamp < ?",
        (user, convo_id, one_week_ago),
    )

def prune_low_salience(user: str, convo_id: int):
    state.c.execute(
        "DELETE FROM memory WHERE user=? AND convo_id=? AND salience < 0.1",
        (user, convo_id),
    )

def size_based_prune(user: str, convo_id: int):
    state.c.execute(
        "SELECT COUNT(*) FROM memory WHERE user=? AND convo_id=?", (user, convo_id)
    )
    if (row_count := state.c.fetchone()[0]) > 1000:
        state.c.execute(
            "SELECT mem_key FROM memory WHERE user=? AND convo_id=? AND salience < 0.5 ORDER BY timestamp ASC LIMIT ?",
            (user, convo_id, row_count - 1000),
        )
        low_keys = [row[0] for row in state.c.fetchall()]
        for key in low_keys:
            state.c.execute(
                "DELETE FROM memory WHERE user=? AND convo_id=? AND mem_key=?",
                (user, convo_id, key),
            )

# Phase 2: Paginated dedup_prune
def dedup_prune(user: str, convo_id: int):
    offset = 0
    batch_size = 100
    hashes = {}
    to_delete = []
    while True:
        state.c.execute(
            "SELECT mem_key, mem_value FROM memory WHERE user=? AND convo_id=? LIMIT ? OFFSET ?",
            (user, convo_id, batch_size, offset),
        )
        batch = state.c.fetchall()
        if not batch:
            break
        for key, value_str in batch:
            value = json.loads(value_str)
            h = hash(value.get("summary", ""))
            if h in hashes and value.get("salience", 1.0) < hashes[h].get("salience", 1.0):
                to_delete.append(key)
            else:
                hashes[h] = value
        offset += batch_size
    for key in to_delete:
        state.c.execute(
            "DELETE FROM memory WHERE user=? AND convo_id=? AND mem_key=?",
            (user, convo_id, key),
        )

def advanced_memory_prune(user: str = None, convo_id: int = None) -> str:
    if user is None or convo_id is None:
        return "Error: user and convo_id required."
    if not should_prune():
        return "Prune skipped (infrequent)."
    def _prune_sync(u, cid):  # V2: Inner sync for executor
        try:
            state.conn.execute("BEGIN")
            decay_salience(u, cid)
            prune_low_salience(u, cid)
            size_based_prune(u, cid)
            dedup_prune(u, cid)
            one_week_ago = datetime.now() - timedelta(days=7)
            state.c.execute(
                "DELETE FROM memory WHERE user=? AND convo_id=? AND timestamp < ? AND mem_key LIKE 'agent_%'",
                (u, cid, one_week_ago),
            )
            state.conn.commit()
            for agent_folder in os.listdir(state.agent_dir):
                folder_path = os.path.join(state.agent_dir, agent_folder)
                if os.path.isdir(folder_path):
                    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
                    if files and os.path.getmtime(os.path.join(folder_path, files[0])) < one_week_ago.timestamp():
                        shutil.rmtree(folder_path)
                        logger.info(f"Pruned old agent folder: {agent_folder}")
            state.lru_evict()
            logger.info("Memory pruned successfully")
            return "Memory pruned successfully."
        except Exception as e:
            state.conn.rollback()
            logger.error(f"Error pruning memory: {e}")
            return f"Error pruning memory: {e}"
    future = state.agent_executor.submit(_prune_sync, user, convo_id)
    try:
        return future.result(timeout=10)  # V2: Async timeout
    except Exception as e:
        return f"Prune timeout/error: {e}"

# Viz Memory Lattice (V3: Plotly interactive)
# Phase 3: Better summary handling
def viz_memory_lattice(
    user: str,
    convo_id: int,
    top_k: int = Config.DEFAULT_TOP_K.value * 4,  # V3: Enum base
    sim_threshold: float = Config.SIM_THRESHOLD.value,
    output_dir: str = "./sandbox/viz",
    plot_type: str = "both"
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    embed_model = state.get_embed_model()
    chroma_col = state.chroma_collection
    if not embed_model or not chroma_col:
        return "Error: Embed/Chroma not ready for viz."
    # Use convo summary for query
    convo_summary = memory_query(limit=1, user=user, convo_id=convo_id)
    query = convo_summary if convo_summary != "[]" else "memory lattice"
    where_clause = {"$and": [{"user": {"$eq": user}}, {"convo_id": {"$eq": int(convo_id)}}]}
    query_emb = embed_model.encode(query).tolist()
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
        summary = meta.get("summary", results["documents"][0][i])  # Full doc fallback
        if len(summary) > 100: summary = summary[:100] + "..."  # Trunc only if needed
        salience = meta.get("salience", 1.0)
        sim_score = 1 - results["distances"][0][i]
        if sim_score < sim_threshold:
            continue
        G.add_node(mem_key, summary=summary, salience=salience, sim=sim_score)
        summaries.append(summary)
        node_data.append({"layer_proxy": i // (len(results["ids"][0]) // 12), "amp": salience * sim_score})
    if len(G.nodes) > 1:
        all_embs = [embed_model.encode(summaries[i]).tolist() for i in range(len(summaries))]
        for i, node_i in enumerate(G.nodes):
            candidates = list(G.nodes)[i+1:]
            sampled = np.random.choice(candidates, min(5, len(candidates)), replace=False)
            for node_j in sampled:
                j_idx = list(G.nodes).index(node_j)
                sim = np.dot(all_embs[i], all_embs[j_idx]) / (np.linalg.norm(all_embs[i]) * np.linalg.norm(all_embs[j_idx]))
                if sim > sim_threshold:
                    G.add_edge(node_i, node_j, weight=sim)
    # V3: Plotly interactive
    if plot_type in ["graph", "both"]:
        pos = nx.spring_layout(G, k=1, iterations=20)
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
    # Amps plot (unchanged, but could Plotly too)
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
        amps_path = os.path.join(output_dir, f"memory_amps_{convo_id}.png")
        plt.savefig(amps_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        st.pyplot(fig)  # Embed
    graph_json = nx.node_link_data(G)
    mem_key = f"lattice_viz_{convo_id}"
    memory_insert(mem_key, {"graph": graph_json, "paths": [amps_path]}, user, convo_id)
    logger.info(f"Lattice viz saved for convo {convo_id}: {len(G.nodes)} nodes")
    return f"Viz complete: Interactive graph rendered. Query '{mem_key}' for data."

# Agent Tools (unchanged from v2)
# Phase 1: Retry caps & partial persist in async_run_spawn
async def async_run_spawn(agent_id: str, task: str, user: str, convo_id: int, retries: int = 0) -> str:
    async with state.agent_sem:  # V2: Concurrency bound
        # Early persist: "retrying"
        persist_agent_result(agent_id, task, "Retrying...", user, convo_id)  # Partial
        try:
            client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=Models.GROK_FAST_REASONING.value,  # V3: Enum
                    messages=[
                        {"role": "system", "content": "You are an agent for AurumVivum. Execute the given task/query/scenario/simulation. Suggest tool-chains if needed, but do not call tools yourself. Respond concisely."},
                        {"role": "user", "content": task}
                    ],
                    stream=False,
                    timeout=60.0
                ), timeout=30.0
            )
            result = response.choices[0].message.content.strip()
            persist_agent_result(agent_id, task, result, user, convo_id)
            logger.info(f"Agent {agent_id} succeeded async.")
            return result
        except asyncio.TimeoutError:
            error = "Timeout: 30s exceeded."
        except Exception as e:
            error = f"Spawn err: {str(e)}"
            if retries < 2:
                await asyncio.sleep(2 ** retries)
                return await async_run_spawn(agent_id, task, user, convo_id, retries + 1)
        persist_agent_result(agent_id, task, error, user, convo_id)
        return error

def persist_agent_result(agent_id: str, task: str, response: str, user: str, convo_id: int) -> None:
    try:
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
        memory_insert(mem_key, result_data, user, convo_id)
        summary_data = {"summary": f"Agent {agent_id} response to task: {task[:100]}...", "details": response}
        advanced_memory_consolidate(f"agent_{agent_id}_summary", summary_data, user, convo_id)
        notify_key = f"agent_{agent_id}_complete"
        notify_data = {"agent_id": agent_id, "status": "complete", "result_key": mem_key, "timestamp": datetime.now().isoformat()}
        memory_insert(notify_key, notify_data, user, convo_id)
        # V2: Queue for poll
        get_state("pending_notifies", []).append({"agent_id": agent_id, "status": "complete", "task": task[:100]})
        logger.info(f"Agent {agent_id} persisted and notified.")
    except Exception as e:
        logger.error(f"Persistence error for agent {agent_id}: {e}")
        error_data = {"agent_id": agent_id, "error": str(e), "status": "failed"}
        memory_insert(f"agent_{agent_id}_error", error_data, user, convo_id)

def agent_spawn(sub_agent_type: str, task: str, user: str = None, convo_id: int = None, poll_interval: int = 5) -> str:
    if user is None or convo_id is None:
        return "Error: user and convo_id required for persistence."
    if len(task) > Config.MAX_TASK_LEN.value:  # V3: Enum
        return "Error: Task too long (max 2000 chars)."
    agent_id = f"{sub_agent_type}_{str(uuid.uuid4())[:8]}"
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_run_spawn(agent_id, task, user, convo_id))
    status_key = f"agent_{agent_id}_status"
    status_data = {"agent_id": agent_id, "task": task[:100], "status": "spawned", "timestamp": datetime.now().isoformat(), "poll_interval": poll_interval}
    memory_insert(status_key, status_data, user, convo_id)
    # V2: Queue status for poll
    get_state("pending_notifies", []).append({"agent_id": agent_id, "status": "spawned", "task": task[:100]})
    return f"Agent '{sub_agent_type}' spawned (ID: {agent_id}). Poll 'agent_{agent_id}_complete' for results. Status: {status_key}"

# V3: Enhanced interactive fleet
def render_agent_fleet():
    pending = get_state("pending_notifies", [])
    if pending:
        st.subheader("Recent Notifies")
        for notify in pending:
            st.info(f"**{notify['agent_id']}**: {notify['status']} – Task: {notify['task']}")
        st.session_state["pending_notifies"] = []  # Clear after display
    
    # V3: Fetch active agents from memory
    user = st.session_state.get("user")
    convo_id = st.session_state.get("current_convo_id", 0)
    if user and convo_id:
        active_query = memory_query(limit=20, user=user, convo_id=convo_id)
        active_agents = []
        try:
            active_data = json.loads(active_query)
            active_agents = [data for key, data in active_data.items() if key.startswith("agent_") and data.get("status") in ["spawned", "running"]]
        except:
            pass
        if active_agents:
            st.subheader("Active Fleet")
            for agent in active_agents:
                col1, col2, col3 = st.columns([3, 4, 1])
                with col1:
                    st.write(f"**{agent.get('agent_id', 'Unknown')}**")
                with col2:
                    st.write(f"Status: {agent.get('status', 'Unknown')} | Task: {agent.get('task', '')[:50]}...")
                with col3:
                    if st.button("Kill", key=f"kill_{agent.get('agent_id', uuid.uuid4())}"):
                        kill_key = f"agent_{agent['agent_id']}_kill"
                        kill_data = {"status": "killed", "timestamp": datetime.now().isoformat()}
                        memory_insert(kill_key, kill_data, user, convo_id)
                        st.rerun()
            if st.button("Spawn Fleet (Parallel Sims)"):
                safe_call(agent_spawn, "fleet", "Run parallel quantum sims on nodes 1-3", user, convo_id)

# Git Tools (unchanged)
def git_init(safe_repo: str) -> str:
    pygit2.init_repository(safe_repo)
    return "Repo initialized."

def git_commit(repo: pygit2.Repository, message: str) -> str:
    if not message:
        return "Error: Message required for commit."
    index = repo.index
    index.add_all()
    index.write()
    tree = index.write_tree()
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
    safe_repo = os.path.abspath(os.path.normpath(os.path.join(state.sandbox_dir, repo_path)))
    if not safe_repo.startswith(os.path.abspath(state.sandbox_dir)):
        return "Error: Repo path outside sandbox."
    try:
        repo = pygit2.Repository(safe_repo)
        op_funcs = {
            "init": lambda: git_init(safe_repo),
            "commit": lambda: git_commit(repo, message),
            "branch": lambda: git_branch(repo, name),
            "diff": lambda: git_diff(repo),
        }
        if operation in op_funcs:
            return op_funcs[operation]()
        return "Unknown operation."
    except Exception as e:
        logger.error(f"Git error: {e}")
        return f"Git error: {e}"

# DB Tool (unchanged)
def db_query(db_path: str, query: str, params: list = None) -> str:
    safe_db = os.path.abspath(os.path.normpath(os.path.join(state.sandbox_dir, db_path)))
    if not safe_db.startswith(os.path.abspath(state.sandbox_dir)):
        return "Error: DB path outside sandbox."
    try:
        db_conn = sqlite3.connect(safe_db)
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
    finally:
        db_conn.close()

# Shell Tool (unchanged from v2)
# Phase 2: TTL on destructive confirm
def shell_exec(command: str) -> str:
    whitelist_pattern = r"^(ls|grep|sed|awk|cat|echo|wc|tail|head|cp|mv|rm|mkdir|rmdir|touch)$"
    cmd_parts = shlex.split(command)
    cmd_base = cmd_parts[0]
    if not re.match(whitelist_pattern, cmd_base):
        return "Error: Command not whitelisted."
    convo_id = st.session_state.get("current_convo_id", 0)
    confirm_key = f"confirm_destructive_{convo_id}"
    confirm_ts_key = f"{confirm_key}_ts"
    if cmd_base in ["rm", "rmdir"] and not get_state(confirm_key, False):
        st.session_state[confirm_key] = True
        st.session_state[confirm_ts_key] = time.time()
        return "Warning: Destructive command detected. Confirm by re-running."
    if get_state(confirm_key, False) and (time.time() - get_state(confirm_ts_key, 0)) > 300:  # 5min TTL
        del st.session_state[confirm_key]
        del st.session_state[confirm_ts_key]
        return "Confirm expired—re-approve destructive cmds."
    # V2: Arg scrub
    if any(arg in ["*", ".."] for arg in cmd_parts[1:]):
        return "Error: Forbidden args (*, ..) in command."
    try:
        result = subprocess.run(
            cmd_parts, cwd=state.sandbox_dir, capture_output=True, text=True, timeout=10
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        logger.error(f"Shell error: {e}")
        return f"Shell error: {e}"

# Lint Tools (unchanged)
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

# API Simulate (unchanged)
# Phase 2: Default mock=True
def api_simulate(url: str, method: str = "GET", data: dict = None, headers: dict = None, mock: bool = True) -> str:
    whitelist = ["https://api.example.com", "https://jsonplaceholder.typicode.com", "https://api.x.ai/v1"]
    if not mock and not any(url.startswith(w) for w in whitelist):
        return "Error: Non-mock calls require whitelisted URL + explicit opt-in."
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

# Web Search (unchanged)
def langsearch_web_search(
    query: str, freshness: str = "noLimit", summary: bool = True, count: int = Config.DEFAULT_TOP_K.value  # V3: Enum
) -> str:
    if not LANGSEARCH_API_KEY:
        return "Error: LANGSEARCH_API_KEY not set."
    url = "https://api.langsearch.com/search"
    payload = {
        "query": query,
        "freshness": freshness,
        "summary": summary,
        "count": min(count, 10),
    }
    headers = {"Authorization": f"Bearer {LANGSEARCH_API_KEY}"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        return response.json() if response.ok else f"Error: {response.text}"
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Search error: {e}"

# Embedding (unchanged)
def generate_embedding(text: str) -> str:
    embed_model = state.get_embed_model()
    if not embed_model:
        return "Error: Embedding model not loaded."
    try:
        embedding = embed_model.encode(text).tolist()
        return json.dumps(embedding)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return f"Embedding error: {e}"

# Vector Search (V3: Enum top_k)
def vector_search(query_embedding: list, top_k: int = Config.DEFAULT_TOP_K.value, threshold: float = Config.SIM_THRESHOLD.value) -> str:  # V3: Enums
    if not state.chroma_ready or not state.chroma_collection:
        return "Error: ChromaDB not ready."
    chroma_col = state.chroma_collection
    try:
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

# Chunk & Summarize (unchanged)
def chunk_text(text: str, max_tokens: int = 512) -> str:
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
    try:
        client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/")
        response = client.chat.completions.create(
            model=Models.GROK_FAST_REASONING.value,  # V3: Enum
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

# Keyword Search (V3: Enum top_k)
def keyword_search(
    query: str, top_k: int = Config.DEFAULT_TOP_K.value, user: str = None, convo_id: int = None  # V3: Enum
) -> list:
    if user is None or convo_id is None:
        return "Error: user and convo_id required."
    try:
        state.c.execute(
            "SELECT mem_key FROM memory WHERE user=? AND convo_id=? AND mem_value LIKE ? ORDER BY salience DESC LIMIT ?",
            (user, convo_id, f"%{query}%", top_k),
        )
        results = [{"id": row[0], "score": 1.0} for row in state.c.fetchall()]
        return results
    except Exception as e:
        logger.error(f"Keyword search error: {e}")
        return f"Keyword search error: {e}"

# Socratic Council (V3: Enum model)
def socratic_api_council(
    branches: list,
    model: str = Models.GROK_FAST_REASONING.value,  # V3: Enum default
    user: str = None,
    convo_id: int = None,
    api_key: str = None,
    rounds: int = 3,
    personas: list = None,
) -> str:
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
            user,
            convo_id or 0,
        )
        logger.info("Socratic council completed")
        return consensus
    except Exception as e:
        logger.error(f"Council error: {e}")
        return f"Council error: {e}"

# Reflect Optimize (unchanged)
async def reflect_optimize(component: str, metrics: dict) -> str:
    return f"Optimized {component} with metrics: {json.dumps(metrics)} - Adjustments applied."

# Venv Tools (V3: Enum max len if needed, unchanged)
def venv_create(env_name: str, with_pip: bool = True) -> str:
    safe_env = os.path.abspath(os.path.normpath(os.path.join(state.sandbox_dir, env_name)))
    if not safe_env.startswith(os.path.abspath(state.sandbox_dir)):
        return "Error: Env path outside sandbox."
    try:
        venv.create(safe_env, with_pip=with_pip)
        return f"Venv '{env_name}' created."
    except Exception as e:
        logger.error(f"Venv error: {e}")
        return f"Venv error: {e}"

def restricted_exec(code: str, level: str = "basic") -> str:
    try:
        if level == "basic":
            tree = ast.parse(code)
            if not all(restricted_policy(node) for node in ast.walk(tree)):  # V2: Policy
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
    env = os.environ.copy()
    if custom_env:
        env.update(custom_env)
    try:
        result = subprocess.run(
            shlex.split(cmd),
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
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
]

def pip_install(venv_path: str, packages: list, upgrade: bool = False) -> str:
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
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        logger.error(f"Pip error: {e}")
        return f"Pip error: {e}"

# Chat Log (unchanged)
def chat_log_analyze_embed(
    convo_id: int, criteria: str, summarize: bool = True, user: str = None
) -> str:
    if user is None:
        return "Error: user required."
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
    analysis_prompt = (
        f"Analyze this chat log on criteria: {criteria}. Summarize if needed."
    )
    response = client.chat.completions.create(
        model=Models.GROK_FAST_REASONING.value,  # V3: Enum
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
    chroma_col.upsert(
        ids=[mem_key],
        embeddings=[embedding],
        documents=[analysis],
        metadatas=[
            {"user": user, "convo_id": convo_id, "type": "chat_log", "salience": 1.0}
        ],
    )
    return f"Chat log {convo_id} analyzed and embedded as {mem_key}."

# YAML Tools (unchanged)
def yaml_retrieve(
    query: str = None, top_k: int = Config.DEFAULT_TOP_K.value, filename: str = None  # V3: Enum
) -> str:
    if not state.yaml_ready:
        return "Error: YAML DB not ready."
    col = state.yaml_collection
    embed_model = state.get_embed_model()
    cache = state.yaml_cache
    try:
        if filename:
            if filename in cache:
                return cache[filename]
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
            query_emb = embed_model.encode(query).tolist()
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
    embed_model = state.get_embed_model()
    if not embed_model:
        return "Error: Embedding model not loaded."
    col = state.yaml_collection
    try:
        if filename:
            path = os.path.join(state.yaml_dir, filename)
            if not os.path.exists(path):
                return "Error: File not found."
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            embedding = embed_model.encode(content).tolist()
            col.upsert(
                ids=[filename],
                embeddings=[embedding],
                documents=[content],
                metadatas=[{"filename": filename}],
            )
            state.yaml_cache[filename] = content
            return f"YAML '{filename}' refreshed successfully."
        else:
            ids = col.get()["ids"]
            if ids:
                col.delete(ids=ids)
            state.yaml_cache = {}
            files_refreshed = 0
            for fname in os.listdir(state.yaml_dir):
                if fname.endswith(".yaml"):
                    path = os.path.join(state.yaml_dir, fname)
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    embedding = embed_model.encode(content).tolist()
                    col.upsert(
                        ids=[fname],
                        embeddings=[embedding],
                        documents=[content],
                        metadatas=[{"filename": fname}],
                    )
                    state.yaml_cache[fname] = content
                    files_refreshed += 1
            state.yaml_ready = True
            return f"All YAMLs refreshed successfully ({files_refreshed} files)."
    except Exception as e:
        logger.error(f"YAML refresh error: {e}")
        return f"YAML refresh error: {e}"

# === SECTION: Tool Schemas (V3: Use Enum in desc if needed, unchanged auto) ===
def generate_tool_schema(func):
    sig = inspect.signature(func)
    properties = {}
    required = []
    for param_name, param in sig.parameters.items():
        prop = {"type": "string"}  # Simplified; enhance with typing
        if param.default != inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(param_name)
        properties[param_name] = prop
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": inspect.getdoc(func) or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

# List all tool funcs for auto-gen
TOOL_FUNCS = [
    fs_read_file, fs_write_file, fs_list_files, fs_mkdir,
    get_current_time, code_execution, memory_insert, memory_query,
    git_ops, db_query, shell_exec, code_lint, api_simulate,
    advanced_memory_consolidate, advanced_memory_retrieve, advanced_memory_prune,
    langsearch_web_search, generate_embedding, vector_search,
    chunk_text, summarize_chunk, keyword_search, socratic_api_council,
    agent_spawn, reflect_optimize, venv_create, restricted_exec,
    isolated_subprocess, pip_install, chat_log_analyze_embed,
    yaml_retrieve, yaml_refresh
]

TOOLS = [generate_tool_schema(func) for func in TOOL_FUNCS]

# Tool Dispatcher (unchanged)
# V3.1: Fix closure—capture func per-iteration
TOOL_DISPATCHER = {
    func.__name__: (lambda f=func, **k: safe_call(f, **k))  # Default arg snapshots 'f'
    for func in TOOL_FUNCS
}

tool_count = get_state("tool_count", 0)
council_count = get_state("council_count", 0)
main_count = get_state("main_count", 0)

# === SECTION: Core Logic (V3: Refactored process_tool_calls) ===
def _handle_single_tool(tool_call, current_messages, enable_tools):  # V3: Helper for complexity
    func_name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
        func_to_call = TOOL_DISPATCHER.get(func_name)
        if (
            func_name.startswith("memory")
            or func_name.startswith("advanced_memory")
            or func_name == "socratic_api_council"
            or func_name == "agent_spawn"
            or func_name == "reflect_optimize"
        ):
            args["user"] = st.session_state["user"]
            args["convo_id"] = st.session_state.get("current_convo_id", 0)
        if func_name == "socratic_api_council":
            args["model"] = st.session_state.get(
                "council_model_select", Models.GROK_FAST_REASONING.value  # V3: Enum
            )
            args["convo_id"] = args.get("convo_id", 0)
        safe_result = func_to_call(**args) if func_to_call else {"success": False, "result": f"Unknown tool: {func_name}"}
        result = safe_result.get("result", safe_result.get("user_msg", "Fallback fail"))
        if not safe_result["success"]:
            st.error(safe_result["user_msg"])  # UI feedback
    except Exception as e:
        result = f"Error calling tool {func_name}: {e}"
    if func_name == "socratic_api_council":
        global council_count
        council_count += 1
    else:
        global tool_count
        tool_count += 1
        st.session_state.setdefault("tool_calls_per_convo", 0)
        st.session_state["tool_calls_per_convo"] += 1
    logger.info(f"Tool call: {func_name} - Result: {str(result)[:200]}...")
    yield f"\n> **Tool Call:** `{func_name}` | **Result:** `{str(result)[:200]}...`\n"
    current_messages.append(
        {"tool_call_id": tool_call.id, "role": "tool", "content": str(result)}
    )

def process_tool_calls(tool_calls, current_messages, enable_tools):
    collected_yields = []  # Phase 3: For coalesce
    collected_yields.append("\n*Thinking... Using tools...*\n")
    state.conn.execute("BEGIN")
    for tool_call in tool_calls:
        for chunk in _handle_single_tool(tool_call, current_messages, enable_tools):  # V3: Delegate
            collected_yields.append(chunk)
    state.conn.commit()
    # Phase 3: Coalesce to str
    yield "\n".join(str(y) for y in collected_yields if isinstance(y, str))

def call_xai_api(
    model,
    messages,
    sys_prompt,
    stream=True,
    image_files=None,
    enable_tools=False,
):
    client = OpenAI(api_key=API_KEY, base_url="https://api.x.ai/v1/", timeout=300)
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
        max_iterations = Config.MAX_ITERATIONS.value  # V3: Enum
        tool_calls_per_convo = get_state("tool_calls_per_convo", 0)
        if tool_calls_per_convo > Config.TOOL_CALL_LIMIT.value:  # V3: Enum
            yield "Error: Tool call limit exceeded for this conversation."
            return
        for _ in range(max_iterations):
            main_count += 1
            logger.info(
                f"API call: Tools: {tool_count} | Council: {council_count} | Main: {main_count}"
            )
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=current_messages,
                    tools=TOOLS if enable_tools else None,
                    tool_choice="auto" if enable_tools else None,
                    stream=True,
                )
                tool_calls = []
                full_delta_response = ""
                collected_yields = []  # Phase 3
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield delta.content
                        full_delta_response += delta.content
                    if delta and delta.tool_calls:
                        tool_calls.extend(delta.tool_calls)
                if not tool_calls:
                    break
                current_messages.append(
                    {
                        "role": "assistant",
                        "content": full_delta_response,
                        "tool_calls": tool_calls,
                    }
                )
                for chunk in process_tool_calls(
                    tool_calls, current_messages, enable_tools
                ):
                    if isinstance(chunk, str):
                        yield chunk
                    else:
                        collected_yields.append(chunk)
                if collected_yields:
                    yield "\n".join(str(y) for y in collected_yields if isinstance(y, str))  # Coalesce
            except Exception as e:
                error_msg = f"API or Tool Error: {traceback.format_exc()}"
                yield f"\nAn error occurred: {e}. Aborting this turn."
                logger.error(error_msg)
                st.warning(error_msg)
                break
    return generate(api_messages)

# === SECTION: UI Pages (V3: Enum models, evolve button, auto-title) ===
def search_history(query: str):
    state.c.execute(
        "SELECT convo_id, title FROM history WHERE user=? AND title LIKE ?",
        (st.session_state["user"], f"%{query}%"),
    )
    return state.c.fetchall()

def export_convo(format: str = "json"):
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
    with st.sidebar:
        st.header("Chat Settings")
        st.selectbox(
            "Select Model",
            [m.value for m in Models],  # V3: Enum
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
            with open(os.path.join(state.prompts_dir, selected_file), "r") as f:
                prompt_content = f.read()
            st.text_area(
                "Edit System Prompt",
                value=prompt_content,
                height=200,
                key="custom_prompt",
            )
            if st.button("Evolve Prompt"):  # V3: Optimize button
                user = st.session_state.get("user")
                convo_id = st.session_state.get("current_convo_id", 0)
                metrics = {"length": len(st.session_state["custom_prompt"]), "vibe": "creative"}
                new_prompt = auto_optimize_prompt(st.session_state["custom_prompt"], user, convo_id, metrics)
                st.session_state["custom_prompt"] = new_prompt
                st.rerun()
            st.checkbox("Enable Tools (Sandboxed)", value=False, key="enable_tools")
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
        # V2: Tabs for settings/fleet
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
                    st.metric("Stability Score", f"{state.stability_score:.2%}")  # V2
                if st.button("Weave Memory Lattice (Viz Current Convo)"):
                    viz_result = viz_memory_lattice(st.session_state["user"], st.session_state["current_convo_id"])
                    st.info(viz_result)
                    # V3: Interactive already in func
                if st.button("Prune Memory Now"):
                    prune_result = advanced_memory_prune(st.session_state["user"], st.session_state["current_convo_id"])
                    st.info(prune_result)
                if st.button("Clear Cache"):
                    st.session_state["tool_cache"] = {}
                    state.memory_cache["lru_cache"] = {}
                    st.success("Cache cleared.")
        with tab2:
            render_agent_fleet()  # V3: Enhanced

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

def load_history(convo_id):
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

def delete_history(convo_id):
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

def render_chat_interface(model, custom_prompt, enable_tools, uploaded_images):
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
        with st.chat_message("assistant"):  # Key dropped—version-safe
            images_to_process = uploaded_images if uploaded_images else []
            generator = call_xai_api(
                model,
                st.session_state.messages,
                custom_prompt,
                stream=True,
                image_files=images_to_process,
                enable_tools=enable_tools,
            )
            # Existing streaming logic (smooth inline via write_stream)
            message_placeholder = st.empty()
            full_response = st.write_stream(generator)
            # Finalize: Swap to formatted markdown
            message_placeholder.markdown(full_response, unsafe_allow_html=False)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        # V3: Auto-title
        title = gen_title(prompt)
        messages_json = json.dumps(st.session_state.messages)
        if st.session_state.get("current_convo_id", 0) == 0:
            state.c.execute(
                "INSERT INTO history (user, title, messages) VALUES (?, ?, ?)",
                (st.session_state["user"], title, messages_json),
            )
            st.session_state.current_convo_id = state.c.lastrowid
        else:
            state.c.execute(
                "UPDATE history SET title=?, messages=? WHERE convo_id=?",
                (title, messages_json, st.session_state["current_convo_id"]),
            )
        state.conn.commit()
        st.session_state["rerun_guard"] = False
        render_agent_fleet()  # V2: Poll after turn
        st.rerun()

def chat_page():
    render_sidebar()
    render_chat_interface(
        st.session_state["model_select"],
        st.session_state["custom_prompt"],
        st.session_state["enable_tools"],
        st.session_state["uploaded_images"],
    )

# Auto-Prune (unchanged)
if "auto_prune_done" not in st.session_state:
    advanced_memory_prune(
        st.session_state.get("user"), st.session_state.get("current_convo_id", 0)
    )
    st.session_state["auto_prune_done"] = True

# === SECTION: Tests (Unchanged from v2) ===
def run_tests():
    class TestTools(unittest.TestCase):
        def setUp(self):
            """Setup mock state for tests."""
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
            result_write = safe_call(fs_write_file, "test.txt", "Hello World")["result"]
            self.assertIn("successfully", result_write)
            result_read = safe_call(fs_read_file, "test.txt")["result"]
            self.assertEqual(result_read, "Hello World")

        def test_memory_insert_query(self):
            mem_value = {"summary": "test mem", "salience": 0.8}
            result_insert = safe_call(memory_insert, "test_key", mem_value, "test", 1)["result"]
            self.assertEqual(result_insert, "Memory inserted successfully.")
            result_query = safe_call(memory_query, mem_key="test_key", user="test", convo_id=1)["result"]
            self.assertIn("summary", result_query)
            self.assertIn("test mem", str(result_query))

        def test_advanced_memory_prune(self):
            low_mem = {"summary": "low salience", "salience": 0.05}
            safe_call(memory_insert, "low_key", low_mem, "test", 1)
            st.session_state["prune_counter"] = 49  # Trigger
            result_prune = safe_call(advanced_memory_prune, "test", 1)["result"]
            self.assertIn("successfully", result_prune)
            query_after = safe_call(memory_query, mem_key="low_key", user="test", convo_id=1)["result"]
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
            result_spawn = safe_call(agent_spawn, "TestAgent", "Mock task: echo hello", "test", 1)["result"]
            self.assertIn("spawned (ID:", result_spawn)
            agent_id_prefix = result_spawn.split("ID: ")[1].split(")")[0][:12]
            status_query = safe_call(memory_query, mem_key=f"agent_{agent_id_prefix}_status", user="test", convo_id=1)["result"]
            self.assertIn("status", str(status_query))
            self.assertIn("spawned", str(status_query))

        def test_code_execution_restricted(self):
            code = "print('Hello from REPL')"
            result = safe_call(code_execution, code)["result"]
            self.assertIn("Hello from REPL", result)
            bad_code = "import os; os.system('ls')"
            bad_result = safe_call(code_execution, bad_code)["result"]
            self.assertIn("Restricted", bad_result) or self.assertIn("Error", bad_result)

        def test_git_ops_init_commit(self):
            result_init = safe_call(git_ops, "init", "test_repo")["result"]
            self.assertEqual(result_init, "Repo initialized.")
            safe_call(fs_write_file, "test_repo/test.txt", "commit test")
            result_commit = safe_call(git_ops, "commit", "test_repo", message="Test commit")["result"]
            self.assertEqual(result_commit, "Committed.")

        def test_shell_exec_whitelist(self):
            result_ls = safe_call(shell_exec, "ls .")["result"]
            self.assertNotIn("Error: Command not whitelisted.", result_ls)
            st.session_state["confirm_destructive_1"] = False  # Mock convo 1
            result_rm = safe_call(shell_exec, "rm test_rm.txt")["result"]
            self.assertIn("Warning: Destructive command detected.", result_rm)
            st.session_state["confirm_destructive_1"] = True
            result_rm_confirm = safe_call(shell_exec, "rm test_rm.txt")["result"]
            self.assertNotIn("Warning", result_rm_confirm)
            result_bad = safe_call(shell_exec, "curl google.com")["result"]
            self.assertIn("Error: Command not whitelisted.", result_bad)

        def test_code_lint_python(self):
            ugly_code = "def foo():print('hi')"
            result = safe_call(code_lint, "python", ugly_code)["result"]
            self.assertIn("def foo():\n    print('hi')", result)

        def test_api_simulate_mock(self):
            result_mock = safe_call(api_simulate, "https://api.example.com/test", method="GET", mock=True)["result"]
            self.assertIn("Mock response", result_mock)

        def test_yaml_retrieve_ready(self):
            safe_call(fs_write_file, "test.yaml", "key: value")
            safe_call(yaml_refresh, "test.yaml")
            result = safe_call(yaml_retrieve, filename="test.yaml")["result"]
            self.assertIn("key: value", result)

        def test_socratic_council(self):
            branches = ["Option A", "Option B"]
            result = safe_call(socratic_api_council, branches, user="test", convo_id=1)["result"]
            self.assertIn("Round", result)
            self.assertIn("Consensus", result)

        # V2: New test for agent bound (simplified mock)
        def test_agent_sem_bound(self):
            # Mock sem; in real, would test acquire limit
            self.assertTrue(hasattr(state, 'agent_sem'))
            self.assertEqual(state.agent_sem._value, Config.AGENT_MAX_CONCURRENT.value)  # V3: Enum

    suite = unittest.TestLoader().loadTestsFromTestCase(TestTools)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    # V2: Update stability
    state.stability_score = 1.0 - (len([t for t, s in result.failures + result.errors]) / len(suite.tests)) if suite.tests else 1.0
    return result.wasSuccessful()

if os.getenv('TEST_MODE'):
    run_tests()

# === SECTION: Main App ===
if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if st.session_state.get("logged_in"):
        chat_page()
    else:
        login_page()
