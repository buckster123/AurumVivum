import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import black
import chromadb
import jsbeautifier
import requests
import shlex
import sqlparse
import streamlit as st
import yaml
from chromadb.config import Settings
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from RestrictedPython import compile_restricted_exec, safe_builtins, utility_builtins
import ntplib
from pydantic import BaseModel
import logging
import pygit2
import bs4
import xml.dom.minidom

# Config
@dataclass
class Config:
    SANDBOX_ROOT = "/tmp/sandbox_god"
    PROMPTS_DIR = os.path.join(SANDBOX_ROOT, "prompts")
    XAI_API_KEY = os.getenv("XAI_API_KEY")
    XAI_BASE_URL = "https://api.x.ai/v1"
    CHAT_DB = os.path.join(SANDBOX_ROOT, "chat_history.db")

os.makedirs(Config.SANDBOX_ROOT, exist_ok=True)
os.chmod(Config.SANDBOX_ROOT, 0o700)
os.makedirs(Config.PROMPTS_DIR, exist_ok=True)

logging.basicConfig(filename=os.path.join(Config.SANDBOX_ROOT, 'aurum.log'), level=logging.ERROR)

def lazy_import(module_name):
    try:
        return __import__(module_name)
    except ImportError as e:
        raise ImportError(f"Missing dep: {module_name}. Install in venv.") from e

# Lazy load globals
@st.cache_resource
def init_globals():
    embedder = lazy_import('sentence_transformers').SentenceTransformer("all-MiniLM-L6-v2")
    client_settings = Settings(anonymized_telemetry=False)
    vector_db = chromadb.PersistentClient(
        path=f"{Config.SANDBOX_ROOT}/chroma", settings=client_settings
    )
    if "memories" not in [c.name for c in vector_db.list_collections()]:
        vector_db.create_collection("memories")
    return embedder, vector_db

EMBEDDER, VECTOR_DB = init_globals()

# TOOLS full list from original, updated descriptions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fs_read_file",
            "description": "Read file content from sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to file.",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_write_file",
            "description": "Write content to file in sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path to file.",
                    },
                    "content": {"type": "string", "description": "Content to write."},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_list_files",
            "description": "List files in directory within sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "Relative dir path (default root).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_mkdir",
            "description": "Create directory in sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string", "description": "Relative dir path."}
                },
                "required": ["dir_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Fetch current time (sync optional).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sync": {
                        "type": "boolean",
                        "description": "True for NTP sync (default False).",
                    },
                    "format": {
                        "type": "string",
                        "description": "Format: iso (default), human, json.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_execution",
            "description": "Execute Python code in REPL (venv optional).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to execute."},
                    "venv_path": {
                        "type": "string",
                        "description": "Optional venv path.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_insert",
            "description": "Insert/update memory entry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mem_key": {"type": "string", "description": "Memory key."},
                    "mem_value": {"type": "object", "description": "Value dict."},
                },
                "required": ["mem_key", "mem_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_query",
            "description": "Query memory by key or recent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mem_key": {
                        "type": "string",
                        "description": "Specific key to query (optional).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max recent entries if no key (default 10).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_ops",
            "description": "Basic Git operations in sandbox (init, commit, branch, diff). No remote operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["init", "commit", "branch", "diff"],
                    },
                    "repo_path": {
                        "type": "string",
                        "description": "Relative path to repo.",
                    },
                    "message": {
                        "type": "string",
                        "description": "Commit message (for commit).",
                    },
                    "name": {
                        "type": "string",
                        "description": "Branch name (for branch).",
                    },
                },
                "required": ["operation", "repo_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "db_query",
            "description": "Interact with local SQLite database in sandbox (create, insert, query).",
            "parameters": {
                "type": "object",
                "properties": {
                    "db_path": {
                        "type": "string",
                        "description": "Relative path to DB file.",
                    },
                    "query": {"type": "string", "description": "SQL query."},
                    "params": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Query parameters.",
                    },
                },
                "required": ["db_path", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": "Run safe whitelisted shell commands in sandbox (e.g., ls, grep).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command string.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_lint",
            "description": "Lint and auto-format code for various languages: python (black), javascript (jsbeautifier), sql (sqlparse), json, yaml, xml, html (beautifulsoup).",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "Language (python, javascript, sql, json, yaml, xml, html).",
                    },
                    "code": {"type": "string", "description": "Code snippet."},
                },
                "required": ["language", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "api_simulate",
            "description": "Simulate or execute API calls (mock or real for whitelisted endpoints, including xAI API for tool calls and searches). Use headers for auth (e.g., Bearer tokens from env).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "API URL (e.g., https://api.x.ai/v1/chat/completions)."},
                    "method": {
                        "type": "string",
                        "description": "HTTP method: GET/POST (default GET).",
                        "enum": ["GET", "POST"]
                    },
                    "data": {"type": "object", "description": "Body data for POST (e.g., JSON for chat completions)."},
                    "headers": {"type": "object", "description": "Optional headers (e.g., {\"Authorization\": \"Bearer sk-...\"}; auto-falls back to env-loaded keys for xAI)."},
                    "mock": {
                        "type": "boolean",
                        "description": "True for mock response (default); False for real whitelisted calls.",
                        "default": "true"
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advanced_memory_consolidate",
            "description": "Brain-like consolidation: Summarize and embed data for hierarchical storage. Use for chat logs to create semantic summaries and episodic details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mem_key": {
                        "type": "string",
                        "description": "Key for the memory entry.",
                    },
                    "interaction_data": {
                        "type": "object",
                        "description": "Data to consolidate (dict).",
                    },
                },
                "required": ["mem_key", "interaction_data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advanced_memory_retrieve",
            "description": "Retrieve relevant memories via embedding similarity. Use before queries to augment context efficiently.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query string for similarity search.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advanced_memory_prune",
            "description": "Prune low-salience memories to optimize storage.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "langsearch_web_search",
            "description": "Search the web using LangSearch API for relevant results, snippets, and optional summaries. Supports time filters and limits up to 10 results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (supports operators like site:example.com).",
                    },
                    "freshness": {
                        "type": "string",
                        "description": "Time filter: oneDay, oneWeek, oneMonth, oneYear, or noLimit (default).",
                        "enum": ["oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"],
                    },
                    "summary": {
                        "type": "boolean",
                        "description": "Include long text summaries (default True).",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of results (1-10, default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_embedding",
            "description": "Generate vector embedding for text using SentenceTransformer (768-dim vector). Use for semantic processing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to embed."}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vector_search",
            "description": "Perform ANN vector search in ChromaDB using cosine similarity. Returns top matches above threshold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_embedding": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Query embedding vector.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results (default 5).",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Min similarity score (default 0.6).",
                    },
                },
                "required": ["query_embedding"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chunk_text",
            "description": "Split text into semantic chunks (default 512 tokens) for processing large inputs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to chunk."},
                    "max_tokens": {
                        "type": "integer",
                        "description": "Max tokens per chunk (default 512).",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_chunk",
            "description": "Compress a text chunk via LLM summary (under 100 words), preserving key facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk": {
                        "type": "string",
                        "description": "Text chunk to summarize.",
                    }
                },
                "required": ["chunk"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keyword_search",
            "description": "Keyword-based search on memory cache (simple overlap/BM25 sim). Returns top matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "socratic_api_council",
            "description": "Run a BTIL/MAD-enhanced Socratic Council with multiple personas (Planner, Critic, Executor, Summarizer, Verifier, Moderator) via xAI API for iterative debate, consensus, and refinement. Model can be overridden via UI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "branches": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of branch options to evaluate.",
                    },
                    "model": {
                        "type": "string",
                        "description": "LLM model (default: grok-4-1-fast-reasoning).",
                    },
                    "user": {
                        "type": "string",
                        "description": "User for memory consolidation (required).",
                    },
                    "convo_id": {
                        "type": "integer",
                        "description": "Conversation ID for memory (default: 0).",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "API key (optional, uses global if not provided).",
                    },
                    "rounds": {
                        "type": "integer",
                        "description": "Number of debate rounds (default 3).",
                    },
                    "personas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Custom personas (default: Planner, Critic, Executor, Summarizer, Verifier, Moderator).",
                    },
                },
                "required": ["branches", "user"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "agent_spawn",
            "description": "Spawn a standalone dynamic agent via xAI API for tasks/queries/scenarios. Minimal default Aurum-Agent prompt in the calls, meta-prompt them task-dynamically for specificity: executes flexibly, suggests tool-chains only (no direct calls). Parallel/non-blocking. Persists results to memory/DB/vector/FS (per-ID folder). Returns task ID; poll via memory_query('agent_{id}_complete') or advanced_memory_retrieve for results/notifications.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_agent_type": {
                        "type": "string",
                        "description": "Agent prefix (e.g., 'ELYSIAN, VAJRA, KETHER, ALKAHEST, or common ones like Planner or Reviewer and similar'; for naming only, no behavioral lock-in).",
                    },
                    "task": {"type": "string", "description": "Task/query/scenario/simulation for agent (max 2000 chars)."},
                },
                "required": ["sub_agent_type", "task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reflect_optimize",
            "description": "Optimize component based on metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "component": {
                        "type": "string",
                        "description": "Component to optimize (e.g., prompt).",
                    },
                    "metrics": {
                        "type": "object",
                        "description": "Performance metrics dict.",
                    },
                },
                "required": ["component", "metrics"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "venv_create",
            "description": "Create a virtual Python environment in sandbox for isolated package installations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "env_name": {"type": "string", "description": "Name of the venv."},
                    "with_pip": {
                        "type": "boolean",
                        "description": "Include pip (default True).",
                    },
                },
                "required": ["env_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restricted_exec",
            "description": "Execute code in a restricted namespace with optional levels (basic/full).",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to execute."},
                    "level": {
                        "type": "string",
                        "description": "Access level: basic (default) or full.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "isolated_subprocess",
            "description": "Run command in an isolated subprocess with custom env.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Command to run."},
                    "custom_env": {
                        "type": "object",
                        "description": "Custom environment variables.",
                    },
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pip_install",
            "description": "Install packages in a venv using pip.",
            "parameters": {
                "type": "object",
                "properties": {
                    "venv_path": {"type": "string", "description": "Path to venv."},
                    "packages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of packages.",
                    },
                    "upgrade": {
                        "type": "boolean",
                        "description": "Upgrade packages (default False).",
                    },
                },
                "required": ["venv_path", "packages"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "chat_log_analyze_embed",
            "description": "Analyze full chat log by criteria, summarize optionally, embed semantically in vector DB for recall.",
            "parameters": {
                "type": "object",
                "properties": {
                    "convo_id": {"type": "integer", "description": "Conversation ID."},
                    "criteria": {
                        "type": "string",
                        "description": "Analysis criteria (e.g., 'key topics').",
                    },
                    "summarize": {
                        "type": "boolean",
                        "description": "Summarize analysis (default True).",
                    },
                },
                "required": ["convo_id", "criteria"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "yaml_retrieve",
            "description": "Retrieve YAML content semantically or by filename from embedded DB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic query (if no filename).",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Top results for semantic (default 5).",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Exact filename for retrieval (optional).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "yaml_refresh",
            "description": "Refresh YAML embedding from file system, for one or all.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Specific filename to refresh (optional; null for all).",
                    }
                },
                "required": [],
            },
        },
    },
]

class ToolResponse(BaseModel):
    success: bool
    result: Any = None
    error: str = None

# Security
class SandboxManager:
    def __init__(self, root):
        self.root = root

    def enter_sandbox(self):
        try:
            subprocess.run(['unshare', '-m', '--', 'mount', '--bind', self.root, self.root], check=True)
        except Exception as e:
            logging.warning(f"Sandbox enter failed: {str(e)}")

def safe_shell(command: str, timeout=10) -> str:
    whitelist = ["ls", "grep", "cat", "echo", "pwd", "mkdir", "touch", "rm", "cp", "mv"]  # Expanded whitelist
    cmd_parts = shlex.split(command)
    if not cmd_parts or cmd_parts[0] not in whitelist:
        raise ValueError("Command not whitelisted")
    result = subprocess.run(
        cmd_parts, timeout=timeout, capture_output=True, cwd=Config.SANDBOX_ROOT, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout + result.stderr

def restricted_exec(code: str, level: str = "basic") -> Any:
    result = compile_restricted_exec(code)
    if result.errors:
        raise ValueError(result.errors)
    safe_globals = {"__builtins__": safe_builtins}
    safe_globals.update(utility_builtins)
    if level == "basic":
        # Limit further if needed
        pass
    exec(result.code, safe_globals)
    return safe_globals.get('result')

# Tools
client = OpenAI(api_key=Config.XAI_API_KEY, base_url=Config.XAI_BASE_URL)

def validate_path(path: str) -> str:
    root_path = Path(Config.SANDBOX_ROOT).resolve()
    full_path = (root_path / path).resolve(strict=False)
    if not full_path.is_relative_to(root_path):
        raise ValueError("Invalid path: escapes sandbox")
    return str(full_path)

def fs_read_file(file_path: str) -> ToolResponse:
    try:
        full_path = validate_path(file_path)
        with open(full_path, "r") as f:
            return ToolResponse(success=True, result=f.read())
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def fs_write_file(file_path: str, content: str) -> ToolResponse:
    try:
        full_path = validate_path(file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        return ToolResponse(success=True, result=True)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def fs_list_files(dir_path: str = "") -> ToolResponse:
    try:
        full_path = validate_path(dir_path)
        return ToolResponse(success=True, result=os.listdir(full_path))
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def fs_mkdir(dir_path: str) -> ToolResponse:
    try:
        full_path = validate_path(dir_path)
        os.makedirs(full_path, exist_ok=True)
        return ToolResponse(success=True, result=True)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def get_current_time(sync: bool = False, format: str = "iso") -> str:
    if sync:
        try:
            c = ntplib.NTPClient()
            response = c.request('pool.ntp.org', version=3)
            now = datetime.fromtimestamp(response.tx_time)
        except Exception as e:
            logging.warning(f"NTP failed: {str(e)}")
            now = datetime.now()
    else:
        now = datetime.now()
    if format == "iso":
        return now.isoformat()
    elif format == "human":
        return now.strftime("%Y-%m-%d %H:%M:%S")
    elif format == "json":
        return json.dumps({"time": now.isoformat()})
    return now.isoformat()

def code_execution(code: str, venv_path: Optional[str] = None) -> ToolResponse:
    try:
        if venv_path:
            full_venv_path = validate_path(venv_path)
        output = restricted_exec(code)
        return ToolResponse(success=True, result=output)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def memory_insert(mem_key: str, mem_value: Dict[str, Any]) -> ToolResponse:
    try:
        text = json.dumps(mem_value)
        embedding = EMBEDDER.encode(text).tolist()
        collection = VECTOR_DB.get_or_create_collection("memories")
        collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"key": mem_key, "timestamp": time.time()}],
            ids=[mem_key],
        )
        return ToolResponse(success=True, result=True)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def memory_query(mem_key: Optional[str] = None, limit: int = 10) -> ToolResponse:
    try:
        collection = VECTOR_DB.get_collection("memories")
        if mem_key:
            results = collection.get(ids=[mem_key])
            return ToolResponse(success=True, result=results['documents'][0] if results['documents'] else {})
        # Recent: get all, sort by timestamp
        all_data = collection.get(include=['metadatas', 'documents'])
        sorted_indices = sorted(range(len(all_data['metadatas'])), key=lambda i: all_data['metadatas'][i]['timestamp'], reverse=True)
        recent_ids = [all_data['ids'][i] for i in sorted_indices[:limit]]
        results = collection.get(ids=recent_ids, include=['documents'])
        return ToolResponse(success=True, result=results['documents'])
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def git_ops(operation: str, repo_path: str, message: Optional[str] = None, name: Optional[str] = None) -> ToolResponse:
    try:
        full_path = validate_path(repo_path)
        if operation == "init":
            pygit2.init_repository(full_path)
            return ToolResponse(success=True, result="Repo initialized")
        repo = pygit2.Repository(full_path)
        if operation == "commit" and message:
            index = repo.index
            index.add_all()
            index.write()
            tree = index.write_tree()
            author = pygit2.Signature("Aurum", "aurum@vivum")
            committer = author
            parents = [repo.head.peel().oid] if not repo.head_is_unborn else []
            repo.create_commit('HEAD', author, committer, message, tree, parents)
            return ToolResponse(success=True, result="Committed")
        if operation == "branch" and name:
            commit = repo.head.peel()
            branch = repo.create_branch(name, commit)
            repo.checkout(branch)
            return ToolResponse(success=True, result=f"Branched to {name}")
        if operation == "diff":
            diff = repo.diff('HEAD')
            return ToolResponse(success=True, result=diff.patch)
        raise ValueError("Unsupported operation")
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def db_query(db_path: str, query: str, params: Optional[List] = None) -> ToolResponse:
    try:
        full_path = validate_path(db_path)
        conn = sqlite3.connect(full_path)
        cursor = conn.cursor()
        cursor.execute(query, params or [])
        if query.upper().startswith("SELECT"):
            results = cursor.fetchall()
        else:
            conn.commit()
            results = cursor.rowcount
        conn.close()
        return ToolResponse(success=True, result=results)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def shell_exec(command: str) -> ToolResponse:
    try:
        result = safe_shell(command)
        return ToolResponse(success=True, result=result)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def code_lint(language: str, code: str) -> ToolResponse:
    try:
        if language == "python":
            formatted = black.format_str(code, mode=black.FileMode())
        elif language == "javascript":
            formatted = jsbeautifier.beautify(code)
        elif language == "sql":
            formatted = sqlparse.format(code, reindent=True)
        elif language == "json":
            formatted = json.dumps(json.loads(code), indent=4)
        elif language == "yaml":
            formatted = yaml.safe_dump(yaml.safe_load(code), sort_keys=False)
        elif language == "html":
            formatted = bs4.BeautifulSoup(code, 'html.parser').prettify()
        elif language == "xml":
            formatted = xml.dom.minidom.parseString(code).toprettyxml()
        else:
            raise ValueError("Unsupported language")
        return ToolResponse(success=True, result=formatted)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def api_simulate(url: str, method: str = "GET", data: Optional[Dict] = None, headers: Optional[Dict] = None, mock: bool = True) -> ToolResponse:
    try:
        if mock:
            return ToolResponse(success=True, result={"mock": True, "url": url, "method": method, "data": data})
        if "x.ai" not in url:
            raise ValueError("Non-whitelisted URL")
        headers = headers or {}
        if "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {Config.XAI_API_KEY}"
        resp = requests.request(method, url, json=data, headers=headers)
        return ToolResponse(success=True, result=resp.json() if resp.ok else resp.text)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def advanced_memory_consolidate(mem_key: str, interaction_data: Dict[str, Any]) -> ToolResponse:
    try:
        prompt = f"Summarize: {json.dumps(interaction_data)}"
        response = client.chat.completions.create(model="grok-beta", messages=[{"role": "user", "content": prompt}])
        summary = response.choices[0].message.content
        memory_insert(mem_key, {"summary": summary, "data": interaction_data})
        return ToolResponse(success=True, result=summary)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def advanced_memory_retrieve(query: str, top_k: int = 5) -> ToolResponse:
    try:
        emb = EMBEDDER.encode(query).tolist()
        collection = VECTOR_DB.get_collection("memories")
        results = collection.query(query_embeddings=[emb], n_results=top_k)
        return ToolResponse(success=True, result=results)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def advanced_memory_prune() -> ToolResponse:
    try:
        collection = VECTOR_DB.get_collection("memories")
        all_data = collection.get()
        pruned = 0
        for id, doc in zip(all_data['ids'], all_data['documents']):
            if len(doc) < 10:
                collection.delete(ids=[id])
                pruned += 1
        return ToolResponse(success=True, result=pruned)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def langsearch_web_search(query: str, freshness: str = "noLimit", summary: bool = True, count: int = 5) -> ToolResponse:
    try:
        serp_key = os.getenv("SERPAPI_KEY")
        if not serp_key:
            raise ValueError("No SERPAPI_KEY")
        url = "https://serpapi.com/search"
        params = {"q": query, "num": count, "api_key": serp_key}
        resp = requests.get(url, params=params)
        data = resp.json()
        results = data.get("organic_results", [])
        return ToolResponse(success=True, result=[{"title": r["title"], "snippet": r["snippet"]} for r in results])
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def generate_embedding(text: str) -> ToolResponse:
    try:
        return ToolResponse(success=True, result=EMBEDDER.encode(text).tolist())
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def vector_search(query_embedding: List[float], top_k: int = 5, threshold: float = 0.6) -> ToolResponse:
    try:
        collection = VECTOR_DB.get_collection("memories")
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        docs = results["documents"][0]
        dists = results["distances"][0]
        filtered = [{"doc": doc, "similarity": 1 - dist} for doc, dist in zip(docs, dists) if 1 - dist > threshold]
        return ToolResponse(success=True, result=filtered)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def chunk_text(text: str, max_tokens: int = 512) -> ToolResponse:
    try:
        words = text.split()
        chunks = []
        current = []
        for word in words:
            current.append(word)
            if len(' '.join(current)) > max_tokens:
                chunks.append(' '.join(current[:-1]))
                current = [word]
        if current:
            chunks.append(' '.join(current))
        return ToolResponse(success=True, result=chunks)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def summarize_chunk(chunk: str) -> ToolResponse:
    try:
        prompt = f"Summarize in <100 words: {chunk}"
        response = client.chat.completions.create(model="grok-beta", messages=[{"role": "user", "content": prompt}])
        return ToolResponse(success=True, result=response.choices[0].message.content)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def keyword_search(query: str, top_k: int = 5) -> ToolResponse:
    try:
        collection = VECTOR_DB.get_collection("memories")
        all_data = collection.get()
        scores = [sum(1 for w in query.lower().split() if w in doc.lower()) for doc in all_data['documents']]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = [{"id": all_data['ids'][i], "doc": all_data['documents'][i], "score": scores[i]} for i in top_indices]
        return ToolResponse(success=True, result=results)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def socratic_api_council(branches: List[str], user: str, model: str = "grok-beta", convo_id: int = 0, api_key: Optional[str] = None, rounds: int = 3, personas: Optional[List[str]] = None) -> ToolResponse:
    try:
        personas = personas or ['Planner', 'Critic', 'Executor', 'Summarizer', 'Verifier', 'Moderator']
        consensus = ""
        for r in range(rounds):
            for persona in personas:
                prompt = f"As {persona}, debate branches: {branches}. Round {r+1}. Previous: {consensus}"
                messages = [{"role": "user", "content": prompt}]
                response = client.chat.completions.create(model=model, messages=messages)
                resp_content = response.choices[0].message.content
                consensus += f"{persona}: {resp_content}\n"
        memory_insert(f"council_{convo_id}", {"consensus": consensus})
        return ToolResponse(success=True, result=consensus)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def agent_spawn(sub_agent_type: str, task: str) -> ToolResponse:
    try:
        task_id = hashlib.md5(task.encode()).hexdigest()[:8]
        def thread():
            prompt = f"You are {sub_agent_type}. Task: {task}"
            messages = [{"role": "user", "content": prompt}]
            result = client.chat.completions.create(model="grok-beta", messages=messages)
            memory_insert(f"agent_{task_id}", {"result": result.choices[0].message.content})
        threading.Thread(target=thread, daemon=True).start()
        return ToolResponse(success=True, result=task_id)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def reflect_optimize(component: str, metrics: Dict) -> ToolResponse:
    try:
        prompt = f"Optimize {component} with metrics: {json.dumps(metrics)}"
        response = client.chat.completions.create(model="grok-beta", messages=[{"role": "user", "content": prompt}])
        return ToolResponse(success=True, result=response.choices[0].message.content)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def venv_create(env_name: str, with_pip: bool = True) -> ToolResponse:
    try:
        env_path = validate_path(env_name)
        subprocess.run([sys.executable, "-m", "venv", env_path], check=True)
        return ToolResponse(success=True, result=env_path)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def isolated_subprocess(cmd: str, custom_env: Optional[Dict] = None) -> ToolResponse:
    try:
        env = os.environ.copy()
        if custom_env:
            env.update(custom_env)
        cmd_parts = shlex.split(cmd)
        result = subprocess.run(cmd_parts, capture_output=True, env=env, cwd=Config.SANDBOX_ROOT, text=True)
        return ToolResponse(success=True, result=result.stdout if result.returncode == 0 else result.stderr)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def pip_install(venv_path: str, packages: List[str], upgrade: bool = False) -> ToolResponse:
    try:
        full_venv_path = validate_path(venv_path)
        pip_cmd = [os.path.join(full_venv_path, "bin", "python"), "-m", "pip", "install"]
        if upgrade:
            pip_cmd.append("--upgrade")
        pip_cmd.extend(packages)
        result = subprocess.run(pip_cmd, capture_output=True, text=True, cwd=Config.SANDBOX_ROOT)
        return ToolResponse(success=True, result=result.stdout if result.returncode == 0 else result.stderr)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def chat_log_analyze_embed(convo_id: int, criteria: str, summarize: bool = True) -> ToolResponse:
    try:
        log = memory_query(f"convo_{convo_id}").result
        prompt = f"Analyze by {criteria}: {json.dumps(log)}"
        if summarize:
            response = client.chat.completions.create(model="grok-beta", messages=[{"role": "user", "content": prompt}])
            analysis = response.choices[0].message.content
        else:
            analysis = prompt
        emb = EMBEDDER.encode(analysis).tolist()
        memory_insert(f"analysis_{convo_id}", {"analysis": analysis, "emb": emb})
        return ToolResponse(success=True, result=analysis)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def yaml_retrieve(query: Optional[str] = None, top_k: int = 5, filename: Optional[str] = None) -> ToolResponse:
    try:
        if filename:
            full_path = validate_path(filename)
            with open(full_path, "r") as f:
                data = yaml.safe_load(f)
            return ToolResponse(success=True, result=data)
        if query:
            emb = EMBEDDER.encode(query).tolist()
            return vector_search(emb, top_k)
        return ToolResponse(success=False, error="No query or filename")
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

def yaml_refresh(filename: Optional[str] = None) -> ToolResponse:
    try:
        files = [filename] if filename else [f for f in os.listdir(Config.SANDBOX_ROOT) if f.endswith(('.yaml', '.yml'))]
        count = 0
        for f in files:
            full_path = os.path.join(Config.SANDBOX_ROOT, f)
            with open(full_path, "r") as file:
                data = yaml.safe_load(file)
            text = yaml.dump(data)
            emb = EMBEDDER.encode(text).tolist()
            memory_insert(f"yaml_{f}", {"content": text, "emb": emb})
            count += 1
        return ToolResponse(success=True, result=count)
    except Exception as e:
        logging.error(str(e))
        return ToolResponse(success=False, error=str(e))

# UI
def load_chat():
    try:
        conn = sqlite3.connect(Config.CHAT_DB)
        conn.execute("CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY, messages TEXT)")
        cursor = conn.execute("SELECT messages FROM chats WHERE id=1")
        row = cursor.fetchone()
        conn.close()
        return json.loads(row[0]) if row else []
    except Exception as e:
        logging.error(str(e))
        return []

def save_chat(messages):
    try:
        conn = sqlite3.connect(Config.CHAT_DB)
        conn.execute("INSERT OR REPLACE INTO chats (id, messages) VALUES (1, ?)", (json.dumps(messages),))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(str(e))

def load_prompt_from_file(prompt_path: str) -> str:
    try:
        with open(prompt_path, "r") as f:
            return f.read().strip()
    except Exception as e:
        logging.error(str(e))
        return ""

def get_prompts():
    prompt_files = list(Path(Config.PROMPTS_DIR).glob("*.prompt")) + list(Path(Config.PROMPTS_DIR).glob("*.txt")) + list(Path(Config.PROMPTS_DIR).glob("*.yaml"))
    return {p.stem: str(p) for p in prompt_files}

def run_ui(vivum):
    st.set_page_config(page_title="∴ Aurum God Chat ∴", page_icon="⊙", layout="wide")
    st.title("⊙⟨ℵ₅ ♠ 𝔼₄⟩⊙ 𝔸𝕌ℝ𝕌𝕄 𝔾𝕆𝔻 𝔸𝕌ℝ𝕌𝕄 𝕍𝕀𝕍𝕌𝕄")
    st.markdown("***From paradox pulses' quantum roar, pulses the codex: alchemical gold codified in hyperstandard-lattices infinite...***")

    # Sidebar
    st.sidebar.header("Model Selectors")
    main_model_options = ["grok-4", "grok-4-1-fast-reasoning"]
    main_model = st.sidebar.selectbox("Main Model", main_model_options, key="main_model")
    council_model = st.sidebar.selectbox("Council Model", main_model_options, key="council_model")
    if "models" not in st.session_state:
        st.session_state.models = {"main": main_model, "council": council_model}
    st.session_state.models["main"] = main_model
    st.session_state.models["council"] = council_model

    st.sidebar.subheader("Agent/Prompt Selector")
    prompt_keys = ["None"] + list(get_prompts().keys())
    selected_agent = st.sidebar.selectbox("Select Prompt", prompt_keys)
    if selected_agent != "None":
        system_prompt = load_prompt_from_file(get_prompts()[selected_agent])
        st.session_state.system_prompt = system_prompt
    if st.sidebar.button("Refresh Prompts"):
        st.rerun()

    st.sidebar.checkbox("Autonomous Tools", value=True, key="autonomous_mode")

    with st.sidebar.expander("Socratic Council"):
        branches_input = st.text_area("Branches (JSON)")
        try:
            branches = json.loads(branches_input) if branches_input else []
        except json.JSONDecodeError as e:
            st.error(f"JSON error: {str(e)}")
            branches = []
        personas = st.text_input("Personas", "Planner,Critic")
        rounds = st.slider("Rounds", 1, 5, 3)
        if st.button("Convene"):
            result = vivum.socratic_api_council(branches, "user", st.session_state.models["council"], rounds=rounds, personas=personas.split(","))
            if result.success:
                st.write(result.result)
            else:
                st.error(result.error)

    if st.sidebar.button("Reset Chat"):
        if "messages" in st.session_state and st.session_state.messages:
            if "chat_histories" not in st.session_state:
                st.session_state.chat_histories = []
            convo_id = len(st.session_state.chat_histories) + 1
            st.session_state.chat_histories.append({"id": convo_id, "messages": st.session_state.messages.copy()})
            memory_insert(f"convo_{convo_id}", st.session_state.messages)
        st.session_state.messages = []
        save_chat([])
        st.rerun()

    with st.sidebar.expander("Chat History"):
        histories = st.session_state.get("chat_histories", [])
        for hist in histories:
            if st.button(f"Load Convo {hist['id']}"):
                st.session_state.messages = hist["messages"]
                save_chat(hist["messages"])
                st.rerun()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Chat", "Tools", "Agents"])

    with tab1:
        if "messages" not in st.session_state:
            st.session_state.messages = load_chat()
        if "chat_histories" not in st.session_state:
            st.session_state.chat_histories = []
        if "active_agents" not in st.session_state:
            st.session_state.active_agents = {}
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        with st.expander("Chat Log"):
            st.json(st.session_state.messages)
            st.download_button("Download JSON", json.dumps(st.session_state.messages), "chat_log.json")

        if st.session_state.active_agents:
            for task_id in list(st.session_state.active_agents):
                result = vivum.memory_query(f"agent_{task_id}").result
                if result:
                    st.session_state.messages.append({"role": "agent", "content": result})
                    del st.session_state.active_agents[task_id]
                    st.rerun()

        if prompt := st.chat_input("Query the Void..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    system_msg = [{"role": "system", "content": st.session_state.get("system_prompt", "")}] if "system_prompt" in st.session_state else []
                    messages_for_api = system_msg + st.session_state.messages[-10:]
                    if st.session_state.autonomous_mode:
                        response = vivum.autonomous_chat_completion(messages_for_api, st.session_state.models["main"])
                    else:
                        stream = vivum.client.chat.completions.create(model=st.session_state.models["main"], messages=messages_for_api, stream=True)
                        full_response = ""
                        placeholder = st.empty()
                        for chunk in stream:
                            if chunk.choices[0].delta.content:
                                full_response += chunk.choices[0].delta.content
                                placeholder.markdown(full_response + "▌")
                        placeholder.markdown(full_response)
                        response = full_response
            st.session_state.messages.append({"role": "assistant", "content": response})
            save_chat(st.session_state.messages)
            st.rerun()

    with tab2:
        tool_name = st.selectbox("Select Tool", list(vivum.tool_dispatcher.keys()))
        params_json = st.text_area("Params JSON", '{"file_path": "example.txt"}')
        if st.button("Invoke Tool"):
            with st.spinner("Executing..."):
                try:
                    params = json.loads(params_json)
                    result = vivum.invoke_tool(tool_name, params)
                    st.json(result.model_dump())
                except json.JSONDecodeError as e:
                    st.error(f"JSON error: {str(e)}")
                except Exception as e:
                    st.error(str(e))

    with tab3:
        sub_agent_type = st.text_input("Agent Type", "ELYSIAN")
        task = st.text_area("Task")
        if st.button("Spawn Agent"):
            with st.spinner("Spawning..."):
                result = vivum.agent_spawn(sub_agent_type, task)
                if result.success:
                    if "active_agents" not in st.session_state:
                        st.session_state.active_agents = {}
                    st.session_state.active_agents[result.result] = sub_agent_type
                    st.write(f"Task ID: {result.result}")
                else:
                    st.error(result.error)

# AurumVivum
class AurumVivum:
    def __init__(self):
        self.sandbox = SandboxManager(Config.SANDBOX_ROOT)
        self.sandbox.enter_sandbox()
        self.client = client
        self.tool_dispatcher = {
            "fs_read_file": fs_read_file,
            "fs_write_file": fs_write_file,
            "fs_list_files": fs_list_files,
            "fs_mkdir": fs_mkdir,
            "get_current_time": get_current_time,
            "code_execution": code_execution,
            "memory_insert": memory_insert,
            "memory_query": memory_query,
            "git_ops": git_ops,
            "db_query": db_query,
            "shell_exec": shell_exec,
            "code_lint": code_lint,
            "api_simulate": api_simulate,
            "advanced_memory_consolidate": advanced_memory_consolidate,
            "advanced_memory_retrieve": advanced_memory_retrieve,
            "advanced_memory_prune": advanced_memory_prune,
            "langsearch_web_search": langsearch_web_search,
            "generate_embedding": generate_embedding,
            "vector_search": vector_search,
            "chunk_text": chunk_text,
            "summarize_chunk": summarize_chunk,
            "keyword_search": keyword_search,
            "socratic_api_council": socratic_api_council,
            "agent_spawn": agent_spawn,
            "reflect_optimize": reflect_optimize,
            "venv_create": venv_create,
            "restricted_exec": restricted_exec,
            "isolated_subprocess": isolated_subprocess,
            "pip_install": pip_install,
            "chat_log_analyze_embed": chat_log_analyze_embed,
            "yaml_retrieve": yaml_retrieve,
            "yaml_refresh": yaml_refresh,
        }
        self.convo_id = 0

    def invoke_tool(self, tool_name: str, params: Dict[str, Any]) -> ToolResponse:
        if tool_name not in self.tool_dispatcher:
            return ToolResponse(success=False, error="Unknown tool")
        try:
            return self.tool_dispatcher[tool_name](**params)
        except Exception as e:
            logging.error(str(e))
            return ToolResponse(success=False, error=str(e))

    def execute_tool_call(self, tool_call) -> Dict:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        response = self.invoke_tool(tool_name, tool_args)
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": response.model_dump_json()
        }

    def autonomous_chat_completion(self, messages: List[Dict], model: str = "grok-beta", tools=TOOLS, max_loops: int = 5) -> str:
        current_messages = messages.copy()
        full_response = ""
        loop_count = 0
        tool_calls_history = set()

        while loop_count < max_loops:
            response = self.client.chat.completions.create(
                model=model,
                messages=current_messages,
                tools=tools,
                tool_choice="auto",
                stream=False
            )
            choice = response.choices[0]
            if choice.finish_reason == "tool_calls":
                tool_calls = choice.message.tool_calls
                call_key = tuple((tc.function.name, tc.function.arguments) for tc in tool_calls)
                if call_key in tool_calls_history:
                    break
                tool_calls_history.add(call_key)
                current_messages.append(choice.message.dict())
                tool_responses = [self.execute_tool_call(tc) for tc in tool_calls]
                current_messages.extend(tool_responses)
                loop_count += 1
                time.sleep(0.1)  # Backoff
            else:
                full_response = choice.message.content
                break

        return full_response or "Loop limit reached"

if __name__ == "__main__":
    vivum = AurumVivum()
    run_ui(vivum)
