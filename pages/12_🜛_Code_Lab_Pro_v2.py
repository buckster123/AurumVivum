# pages/02_Code_Lab_Pro_v2.py
import streamlit as st
from streamlit_ace import st_ace
import pathlib
import os
import json
import sys
import io
import subprocess
import uuid
import threading
from typing import Optional, Dict, List, Tuple, Any
import sqlite3
import asyncio
import numpy
import builtins
import time
import re
import difflib
import ast
from datetime import datetime, timedelta
from functools import lru_cache, wraps
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, asdict
import shutil
import importlib.util
import traceback
import hashlib  # Added for task hashing

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler("/tmp/codelab.log", maxBytes=10*1024*1024, backupCount=3)
    ]
)
logger = logging.getLogger("CodeLabProV2")

# ============================================================================
# CRITICAL FIX: Robust TOOL_DISPATCHER Access
# ============================================================================
def _get_main_tool_dispatcher():
    """
    Access TOOL_DISPATCHER from main app using multiple strategies.
    This is the KEY FIX for tool availability in multi-page Streamlit apps.
    """
    start_time = time.time()
    
    # STRATEGY 1: Direct from session state (main app should store this)
    if "TOOL_DISPATCHER" in st.session_state:
        dispatcher = st.session_state.TOOL_DISPATCHER
        if dispatcher and isinstance(dispatcher, dict) and len(dispatcher) > 0:
            logger.info(f"[STRATEGY 1] Got TOOL_DISPATCHER from session state ({len(dispatcher)} tools)")
            return dispatcher
    
    # STRATEGY 2: Try lazy import of main module (avoids circular imports)
    try:
        main_script_names = ['main', 'app']
        
        for script_name in main_script_names:
            script_path = f"{script_name}.py"
            if os.path.exists(script_path):
                try:
                    spec = importlib.util.spec_from_file_location(script_name, script_path)
                    if spec and spec.loader:
                        main_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(main_module)
                        
                        dispatcher = getattr(main_module, 'TOOL_DISPATCHER', None)
                        if dispatcher and isinstance(dispatcher, dict) and len(dispatcher) > 0:
                            logger.info(f"[STRATEGY 2] Loaded TOOL_DISPATCHER from {script_path} ({len(dispatcher)} tools)")
                            # Cache it for future calls
                            st.session_state.TOOL_DISPATCHER = dispatcher
                            return dispatcher
                except Exception as e:
                    logger.debug(f"Failed to load {script_path}: {e}")
                    continue
    except Exception as e:
        logger.debug(f"[STRATEGY 2] Lazy import failed: {e}")
    
    # STRATEGY 3: Check sys.modules (if main app already imported it)
    try:
        for module_name in ['main', 'app']:
            if module_name in sys.modules:
                main_mod = sys.modules[module_name]
                dispatcher = getattr(main_mod, 'TOOL_DISPATCHER', None)
                if dispatcher and isinstance(dispatcher, dict) and len(dispatcher) > 0:
                    logger.info(f"[STRATEGY 3] Got TOOL_DISPATCHER from sys.modules['{module_name}']")
                    st.session_state.TOOL_DISPATCHER = dispatcher
                    return dispatcher
    except Exception as e:
        logger.debug(f"[STRATEGY 3] sys.modules method failed: {e}")
    
    # STRATEGY 4: Check __main__ (unlikely in pages/ but as last resort)
    try:
        main_mod = sys.modules.get('__main__')
        if main_mod:
            dispatcher = getattr(main_mod, 'TOOL_DISPATCHER', None)
            if dispatcher and isinstance(dispatcher, dict) and len(dispatcher) > 0:
                logger.info("[STRATEGY 4] Got TOOL_DISPATCHER from __main__ (unexpected!)")
                st.session_state.TOOL_DISPATCHER = dispatcher
                return dispatcher
    except Exception as e:
        logger.debug(f"[STRATEGY 4] __main__ method failed: {e}")
    
    logger.warning("All strategies failed to get TOOL_DISPATCHER - tools will be unavailable")
    logger.info(f"Dispatcher discovery took {time.time() - start_time:.3f}s")
    return None

def _safe_tool_call(func_name: str, **kwargs) -> Optional[Any]:
    """
    Safely call a tool from main app's dispatcher with 120s timeout (FIX #1)
    """
    dispatcher = _get_main_tool_dispatcher()
    
    if not dispatcher:
        return "Error: Tool dispatcher unavailable"
        
    if func_name not in dispatcher:
        available = list(dispatcher.keys())
        return f"Error: Tool '{func_name}' not found. Available: {', '.join(available[:5])}..."
    
    try:
        result = [None]
        def _call():
            try:
                logger.info(f"Executing tool: {func_name} with args {list(kwargs.keys())}")
                result[0] = dispatcher[func_name](**kwargs)
            except Exception as e:
                result[0] = f"Tool execution error: {e}"
                logger.error(f"Tool {func_name} failed: {e}", exc_info=True)
        
        thread = threading.Thread(target=_call)
        thread.start()
        thread.join(timeout=600)  # ← FIX #1: Increased to 120 seconds
        
        if thread.is_alive():
            logger.warning(f"Tool '{func_name}' timed out after 120s")
            return f"Error: Tool '{func_name}' timed out after 120s"
            
        return result[0]
    except Exception as e:
        logger.error(f"Tool wrapper error: {e}")
        return f"Error: {e}"

# ============================================================================
# STANDALONE MODE FALLBACKS
# ============================================================================
def _get_sandbox_dir() -> str:
    """Get sandbox dir from main app state or default"""
    try:
        if os.path.exists("main.py"):
            spec = importlib.util.spec_from_file_location("main", "main.py")
            if spec and spec.loader:
                main_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(main_module)
                if hasattr(main_module, 'state') and hasattr(main_module.state, 'sandbox_dir'):
                    return main_module.state.sandbox_dir
    except Exception as e:
        logger.debug(f"Could not access main app state for sandbox: {e}")
    
    if "app_state" in st.session_state and hasattr(st.session_state.app_state, 'sandbox_dir'):
        return st.session_state.app_state.sandbox_dir
    return "./sandbox"

def _get_state(key: str, default: Any = None) -> Any:
    """Safe session state getter with logging"""
    value = st.session_state.get(key, default)
    logger.debug(f"Session state get: {key} = {type(value).__name__}")
    return value

# ============================================================================
# ASYNC EXECUTION ENGINE
# ============================================================================
class AsyncExecutor:
    """Async subprocess executor with resource limits"""
    
    @staticmethod
    async def execute_code(
        code: str, 
        timeout: int = 30, 
        memory_mb: int = 256,
        cwd: Optional[str] = None
    ) -> Tuple[str, str, int]:
        """Execute code in isolated subprocess"""
        def set_limits():
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024,) * 2)
            except (ImportError, AttributeError):
                pass
        
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                preexec_fn=set_limits,
                limit=1024 * 512
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
            return stdout.decode('utf-8', errors='replace'), \
                   stderr.decode('utf-8', errors='replace'), \
                   process.returncode
                   
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return "", "Execution timed out", -1
        except Exception as e:
            logger.error(f"Execution error: {e}", exc_info=True)
            return "", str(e), -1

# ============================================================================
# STATE MANAGEMENT WITH PERSISTENCE
# ============================================================================
@dataclass
class TabState:
    """Represents a single editor tab"""
    path: str
    content: str
    is_dirty: bool = False
    last_saved: float = 0.0
    undo_stack: List[str] = None
    redo_stack: List[str] = None
    
    def __post_init__(self):
        if self.undo_stack is None:
            self.undo_stack = []
        if self.redo_stack is None:
            self.redo_stack = []

class CodeLabStateV2:
    """Enhanced state manager with SQLite persistence"""
    
    def __init__(self):
        self.sandbox_dir = _get_sandbox_dir()
        self._abs_sandbox = pathlib.Path(self.sandbox_dir).resolve()
        self._ensure_sandbox()
        self._init_db()
        self._init_agent_results_table()  # ← NEW: Agent results table
        self._restore_session()
        
    def _ensure_sandbox(self):
        """Create sandbox directory if missing"""
        self._abs_sandbox.mkdir(parents=True, exist_ok=True)
        
    def _init_db(self):
        """Initialize SQLite for state persistence"""
        db_path = self._abs_sandbox / "./db" / "codelab" / "session_v2.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tabs (
                path TEXT PRIMARY KEY,
                content TEXT,
                is_dirty INTEGER,
                last_saved REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        try:
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS search_index 
                USING fts5(path UNINDEXED, content, tokenize='trigram')
            """)
        except sqlite3.OperationalError:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS search_index (
                    path TEXT PRIMARY KEY,
                    content TEXT
                )
            """)
        
        logger.info(f"Database initialized at {db_path}")
        
    def _init_agent_results_table(self):
        """Create table for agent results in Code-Lab's own DB"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_results (
                agent_id TEXT PRIMARY KEY,
                result_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
        logger.info("Agent results table initialized in Code-Lab DB")
        
    def store_agent_result(self, agent_id: str, result_text: str):
        """Store agent result in Code-Lab's DB"""
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO agent_results (agent_id, result_text) VALUES (?, ?)",
                (agent_id, result_text)
            )
            self.conn.commit()
            logger.info(f"Stored agent result for {agent_id} in Code-Lab DB")
        except sqlite3.Error as e:
            logger.error(f"Failed to store agent result: {e}")
            
    def get_agent_result(self, agent_id: str) -> Optional[str]:
        """Retrieve agent result from Code-Lab's DB"""
        try:
            cursor = self.conn.execute(
                "SELECT result_text FROM agent_results WHERE agent_id = ?",
                (agent_id,)
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve agent result: {e}")
            return None
        
    def _restore_session(self):
        """Restore saved session state from SQLite - CRITICAL for F5 survival"""
        st.session_state.code_editor_tabs = {}
        
        try:
            cursor = self.conn.execute("SELECT path, content FROM tabs WHERE is_dirty = 0")
            tabs = {}
            for path, content in cursor:
                tabs[path] = TabState(path=path, content=content)
            
            st.session_state.code_editor_tabs = tabs
            logger.info(f"Restored {len(tabs)} tabs from database")
            
            active_file = st.session_state.get("code_active_file")
            if active_file and active_file not in tabs:
                logger.info(f"Active file {active_file} no longer exists, clearing")
                st.session_state.code_active_file = None
                
        except sqlite3.Error as e:
            logger.error(f"Session restore error: {e}")
            st.session_state.code_editor_tabs = {}
            
    def persist_tab(self, tab: TabState):
        """Persist tab to SQLite"""
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO tabs (path, content, is_dirty, last_saved) VALUES (?, ?, ?, ?)",
                (tab.path, tab.content, int(tab.is_dirty), tab.last_saved)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to persist tab {tab.path}: {e}")
        
    def _validate_path(self, path: pathlib.Path) -> bool:
        """Strict sandbox validation"""
        try:
            full_path = (self._abs_sandbox / path).resolve()
            return str(full_path).startswith(str(self._abs_sandbox))
        except Exception:
            return False
            
    def list_files(self, path: str = "") -> List[str]:
        """List files with caching"""
        @lru_cache(maxsize=128)
        def _cached_list(path_key: str) -> List[str]:
            try:
                target = pathlib.Path(self.sandbox_dir) / path
                if not target.exists():
                    return []
                
                files = []
                for item in target.iterdir():
                    try:
                        rel = item.relative_to(self.sandbox_dir)
                        if self._validate_path(rel):
                            files.append(str(rel) + ("/" if item.is_dir() else ""))
                    except ValueError:
                        continue
                        
                return sorted(files)
            except Exception as e:
                logger.error(f"File list error: {e}")
                return []
                
        return _cached_list(path)
    
    def read_file(self, file_path: str) -> Optional[str]:
        """Read file with validation and caching"""
        if not self._validate_path(pathlib.Path(file_path)):
            logger.warning(f"Path validation failed: {file_path}")
            return None
            
        result = _safe_tool_call("fs_read_file", file_path=file_path)
        if result and "Error" not in result:
            self._index_file(file_path, result)
            return result
        
        try:
            safe_path = self._abs_sandbox / file_path
            with open(safe_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                self._index_file(file_path, content)
                return content
        except Exception as e:
            logger.error(f"Read error: {e}")
            return None
    
    def write_file(self, file_path: str, content: str) -> bool:
        """Write file with validation"""
        if not self._validate_path(pathlib.Path(file_path)):
            logger.warning(f"Write path validation failed: {file_path}")
            return False
            
        result = _safe_tool_call("fs_write_file", file_path=file_path, content=content)
        if result and "successfully" in result:
            self._index_file(file_path, content)
            return True
        
        try:
            safe_path = self._abs_sandbox / file_path
            safe_path.parent.mkdir(parents=True, exist_ok=True)
            with open(safe_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self._index_file(file_path, content)
            return True
        except Exception as e:
            logger.error(f"Write error: {e}")
            return False
            
    def _index_file(self, path: str, content: str):
        """Update search index"""
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO search_index (path, content) VALUES (?, ?)",
                (path, content)
            )
            self.conn.commit()
        except sqlite3.OperationalError:
            self.conn.execute(
                "INSERT OR REPLACE INTO search_index (path, content) VALUES (?, ?)",
                (path, content)
            )
            self.conn.commit()
        
    def search_files(self, query: str, limit: int = 50) -> List[Tuple[str, str]]:
        """Search files using FTS"""
        try:
            cursor = self.conn.execute(
                "SELECT path, snippet(search_index, 2, '<mark>', '</mark>', '...', 10) "
                "FROM search_index WHERE content MATCH ? LIMIT ?",
                (query, limit)
            )
            return cursor.fetchall()
        except sqlite3.OperationalError:
            cursor = self.conn.execute(
                "SELECT path, content FROM search_index WHERE content LIKE ? LIMIT ?",
                (f"%{query}%", limit)
            )
            results = []
            for path, content in cursor:
                idx = content.lower().find(query.lower())
                snippet = content[max(0, idx-20):idx+20]
                results.append((path, snippet))
            return results
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file with recycle bin"""
        if not self._validate_path(pathlib.Path(file_path)):
            logger.warning(f"Delete path validation failed: {file_path}")
            return False
            
        try:
            safe_path = self._abs_sandbox / file_path
            if safe_path.exists():
                trash_dir = self._abs_sandbox / ".codelab" / "recycle"
                trash_dir.mkdir(parents=True, exist_ok=True)
                trash_path = trash_dir / f"{file_path.replace('/', '_')}_{int(time.time())}"
                trash_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(safe_path), str(trash_path))
                
                self.conn.execute("DELETE FROM search_index WHERE path = ?", (file_path,))
                self.conn.commit()
                logger.info(f"Deleted file: {file_path}")
                return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
        return False

    def get_repl_namespace(self) -> dict:
        """Get REPL namespace with module imports"""
        if "repl_namespace" in st.session_state:
            return st.session_state.repl_namespace
        
        safe_builtins = {
            b: getattr(builtins, b)
            for b in [
                "print", "len", "range", "str", "int", "float", "list", "dict",
                "set", "tuple", "abs", "round", "max", "min", "sum", "sorted",
                "enumerate", "zip", "map", "filter", "any", "all", "bool",
                "type", "isinstance", "hasattr", "getattr", "pow",
                "enumerate", "zip", "reversed", "chr", "ord", "hex"
            ]
        }
        
        namespace = {"__builtins__": safe_builtins}
        
        modules = [
            ("numpy", "np"), ("sympy", "sympy"), ("mpmath", "mpmath"),
            ("networkx", "nx"), ("chess", "chess"), ("pygame", "pygame"),
            ("qutip", "qutip"), ("qiskit", "qiskit"), ("torch", "torch"),
            ("scipy", "scipy"), ("pandas", "pd"), ("sklearn", "sklearn")
        ]
        
        for module_name, alias in modules:
            try:
                module = __import__(module_name)
                namespace[alias] = module
                namespace[module_name] = module
            except ImportError:
                pass
        
        try:
            import matplotlib.pyplot as plt
            namespace["matplotlib"] = __import__('matplotlib')
            namespace["plt"] = plt
        except ImportError:
            pass
        
        st.session_state.repl_namespace = namespace
        return namespace

# ============================================================================
# AST ANALYZER
# ============================================================================
class CodeAnalyzer(ast.NodeVisitor):
    """AST-based code analyzer"""
    
    def __init__(self):
        self.metrics = {
            "functions": 0,
            "classes": 0,
            "imports": [],
            "complexity": 0,
            "lines": 0,
            "comments": 0
        }
        
    def visit_FunctionDef(self, node):
        self.metrics["functions"] += 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With)):
                self.metrics["complexity"] += 1
        self.generic_visit(node)
        
    def visit_ClassDef(self, node):
        self.metrics["classes"] += 1
        self.generic_visit(node)
        
    def visit_Import(self, node):
        self.metrics["imports"].extend(alias.name for alias in node.names)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        if node.module:
            self.metrics["imports"].extend(f"{node.module}.{alias.name}" for alias in node.names)
        self.generic_visit(node)
        
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            self.metrics["comments"] += 1
        self.generic_visit(node)

# ============================================================================
# ENHANCED DEBUGGER
# ============================================================================
class EnhancedDebugger:
    """AST-based debugger with watch expressions"""
    
    def __init__(self, code: str, namespace: dict):
        self.code = code
        self.namespace = namespace.copy()
        self.tree = ast.parse(code)
        self.instructions = list(self._flatten_ast(self.tree))
        self.current_inst = 0
        self.breakpoints = set()
        self.watch_expressions: Dict[str, str] = {}
        self.output = ""
        self.paused = False
        
    def _flatten_ast(self, tree: ast.Module) -> List[ast.AST]:
        """Flatten to executable instructions"""
        instructions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign, 
                                ast.AugAssign, ast.If, ast.For, ast.While,
                                ast.FunctionDef, ast.ClassDef, ast.Return)):
                instructions.append(node)
        return sorted(instructions, key=lambda x: getattr(x, 'lineno', 0))
        
    def toggle_breakpoint(self, line: int):
        """Toggle breakpoint on line"""
        if line in self.breakpoints:
            self.breakpoints.remove(line)
        else:
            self.breakpoints.add(line)
    
    def add_watch(self, expression: str):
        """Add watch expression"""
        self.watch_expressions[expression] = ""
        
    def remove_watch(self, expression: str):
        """Remove watch expression"""
        self.watch_expressions.pop(expression, None)
        
    def eval_watch(self, expression: str) -> str:
        """Evaluate watch expression"""
        try:
            result = eval(expression, self.namespace)
            return str(result)[:200]
        except Exception as e:
            return f"Error: {e}"
        
    def step(self) -> Tuple[bool, Optional[Dict]]:
        """Execute one instruction"""
        if self.current_inst >= len(self.instructions):
            return False, {}
            
        inst = self.instructions[self.current_inst]
        
        if not hasattr(inst, 'lineno'):
            self.current_inst += 1
            return self.current_inst < len(self.instructions), {}
            
        line_no = inst.lineno
        
        # Check breakpoint
        if line_no in self.breakpoints:
            self.paused = True
            return True, self._get_namespace_snapshot()
        
        # Execute
        try:
            module = ast.Module([inst], type_ignores=[])
            code_obj = compile(module, '<debugger>', 'exec')
            
            old_stdout = sys.stdout
            captured = io.StringIO()
            sys.stdout = captured
            
            exec(code_obj, self.namespace)
            
            sys.stdout = old_stdout
            self.output = captured.getvalue()
            self.current_inst += 1
            
            # Check if next is breakpoint
            should_pause = self.current_inst < len(self.instructions) and \
                          hasattr(self.instructions[self.current_inst], 'lineno') and \
                          self.instructions[self.current_inst].lineno in self.breakpoints
                          
            return should_pause, self._get_namespace_snapshot()
            
        except Exception as e:
            sys.stdout = old_stdout
            self.output = f"Error at line {line_no}: {e}"
            self.paused = True
            return True, {"__error__": str(e)}
            
    def run_to_completion(self) -> Tuple[str, Dict]:
        """Run all code"""
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured
        
        try:
            for i in range(len(self.instructions)):
                self.current_inst = i
                paused, _ = self.step()
                if paused:
                    break
                    
            sys.stdout = old_stdout
            return captured.getvalue() + self.output, self._get_namespace_snapshot()
            
        except Exception as e:
            sys.stdout = old_stdout
            return f"Fatal error: {e}", self._get_namespace_snapshot()
            
    def _get_namespace_snapshot(self) -> Dict:
        """Get safe namespace snapshot"""
        return {
            k: v for k, v in self.namespace.items() 
            if not k.startswith('_') and k not in ['__builtins__'] and len(str(v)) < 1000
        }

# ============================================================================
# AGENT INTEGRATION
# ============================================================================
class AgentIntegration:
    """Enhanced agent integration"""
    
    def __init__(self):
        self.conversation_history: List[Dict] = []
        
    def spawn_agent(self, task: str, file_path: str, code: str, context: Optional[Dict] = None) -> Optional[str]:
        """Spawn agent with context"""
        convo_uuid = _get_state("current_convo_uuid", str(uuid.uuid4()))
        
        # Use tool dispatcher
        if "agent_spawn" not in (_get_main_tool_dispatcher() or {}):
            logger.error("agent_spawn not available in dispatcher")
            return None
            
        analysis = self._analyze_code_context(code)
        
        agent_task = f"""
        You are a pair-programming AI assistant. Analyze the code and provide actionable suggestions.
        
        FILE: {file_path}
        CODE CONTEXT:
        ```python
        {code[:2000]}
        ```
        
        CODE METRICS:
        - Lines: {analysis['lines']}
        - Functions: {analysis['functions']}
        - Complexity: {analysis['complexity']}
        - Imports: {', '.join(analysis['imports'][:5])}
        
        TASK: {task}
        
        FORMAT:
        - [LINE X] [REPLACE|INSERT] code here
        - Explanation: ...
        """
        
        result = _safe_tool_call(
            "agent_spawn",
            sub_agent_type="coder",
            task=agent_task,
            convo_uuid=convo_uuid,
            model="kimi-k2-thinking"
        )
        
        if result:
            self.conversation_history.append({
                "task": task,
                "file": file_path,
                "result": result,
                "timestamp": time.time()
            })
            
        return result
        
    def _analyze_code_context(self, code: str) -> Dict:
        """Analyze code for context"""
        try:
            tree = ast.parse(code)
            analyzer = CodeAnalyzer()
            analyzer.visit(tree)
            return analyzer.metrics
        except SyntaxError as e:
            return {"error": str(e)}
            
    @staticmethod
    def parse_suggestions(response: str) -> List[Dict]:
        """Parse agent suggestions"""
        suggestions = []
        lines = response.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            if "[LINE" in line and "]" in line:
                try:
                    match = re.search(r'\[LINE\s*(\d+)\]', line)
                    if match:
                        line_num = int(match.group(1))
                        
                        action = "INSERT"
                        if "[REPLACE" in line:
                            action = "REPLACE"
                            
                        code_start = line.find("]") + 1
                        suggestion_code = line[code_start:].strip()
                        
                        explanation = ""
                        if i + 1 < len(lines) and ("Explanation:" in lines[i+1] or "解释" in lines[i+1]):
                            explanation = lines[i+1].split(":", 1)[1].strip()
                            i += 1
                            
                        suggestions.append({
                            "line": line_num,
                            "action": action,
                            "code": suggestion_code,
                            "explanation": explanation,
                            "accepted": False,
                            "id": str(uuid.uuid4())[:8]
                        })
                except Exception as e:
                    logger.error(f"Failed to parse suggestion: {e}")
            i += 1
            
        return suggestions
        
    @staticmethod
    def apply_suggestion(original: str, suggestion: Dict) -> str:
        """Apply suggestion with syntax validation"""
        lines = original.splitlines()
        line_num = suggestion["line"] - 1
        
        if suggestion["action"] == "REPLACE" and line_num < len(lines):
            lines[line_num] = suggestion["code"]
        elif suggestion["action"] == "INSERT" and line_num <= len(lines):
            lines.insert(line_num, suggestion["code"])
            
        new_code = "\n".join(lines)
        
        try:
            ast.parse(new_code)
            return new_code
        except SyntaxError as e:
            raise ValueError(f"Suggestion would break syntax: {e}")

# ============================================================================
# UI COMPONENTS
# ============================================================================
class FileBrowser:
    """Enhanced file browser"""
    
    def __init__(self, state: CodeLabStateV2):
        self.state = state
        self.expanded_folders = _get_state("expanded_folders", set())
        
    def render(self):
        """Render file browser sidebar"""
        st.sidebar.header("📂 Sandbox Explorer")
        
        # Search
        search_query = st.sidebar.text_input("🔍 Search files", placeholder="Fuzzy search...")
        if search_query:
            results = self.state.search_files(search_query)
            if results:
                st.sidebar.markdown("### Search Results")
                for path, snippet in results[:10]:
                    if st.sidebar.button(f"📄 {path}", key=f"search_{path}"):
                        st.session_state["workspace_selected_file"] = path
                        st.rerun()
                st.sidebar.markdown("---")
        
        # Directory tree
        self._render_directory(pathlib.Path(self.state.sandbox_dir), level=0)
        
        # File operations
        self._render_file_operations()
        
    def _render_directory(self, directory: pathlib.Path, level: int = 0):
        """Render directory recursively"""
        indent = "    " * level
        
        try:
            items = self.state.list_files(str(directory.relative_to(self.state.sandbox_dir)))
            folders = [i for i in items if i.endswith('/')]
            files = [i for i in items if not i.endswith('/')]
            
            # Folders
            for folder in folders:
                folder_name = folder.strip('/').split('/')[-1]
                folder_key = f"folder_{folder}_{level}"
                
                is_expanded = folder_key in self.expanded_folders
                if st.sidebar.checkbox(
                    f"{indent}📁 {folder_name}/",
                    value=is_expanded,
                    key=folder_key
                ):
                    self.expanded_folders.add(folder_key)
                    self._render_directory(
                        pathlib.Path(self.state.sandbox_dir) / folder.strip('/'),
                        level + 1
                    )
                elif folder_key in self.expanded_folders:
                    self.expanded_folders.remove(folder_key)
            
            # Files
            for file in files:
                file_name = file.split('/')[-1]
                if st.sidebar.button(
                    f"{indent}📄 {file_name}",
                    key=f"file_{file}_{level}"
                ):
                    st.session_state["workspace_selected_file"] = file
                    st.rerun()
                    
        except Exception as e:
            logger.error(f"Directory render error: {e}")
            
    def _render_file_operations(self):
        """Render file operation controls"""
        st.sidebar.markdown("---")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            new_name = st.sidebar.text_input("New file", placeholder="filename.py")
            if st.sidebar.button("Create") and new_name.strip():
                new_path = new_name.strip()
                if self.state._validate_path(pathlib.Path(new_path)):
                    full_path = self.state._abs_sandbox / new_path
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.touch()
                    st.session_state["workspace_selected_file"] = new_path
                    st.rerun()
                else:
                    st.sidebar.error("Invalid path")
        
        with col2:
            if st.sidebar.button("Refresh"):
                st.rerun()

class Editor:
    """Enhanced code editor"""
    
    def __init__(self, state: CodeLabStateV2):
        self.state = state
        self.agent = AgentIntegration()
        self.auto_save_timer = _get_state("auto_save_last", time.time())
        
    def render(self):
        """Render main editor area"""
        st.header("🎨 Code Editor Pro")
        
        # Auto-save every 30 seconds
        if time.time() - self.auto_save_timer > 30:
            self._auto_save()
            st.session_state.auto_save_last = time.time()
            
        # Load selected file
        self._load_selected_file()
        
        # Check if any tabs exist
        tab_count = self.state.conn.execute("SELECT COUNT(*) FROM tabs").fetchone()[0]
        if tab_count == 0:
            st.info("📂 Select a file from the browser to start editing")
            return
            
        # Render tabs
        self._render_tabs()
        
    def _load_selected_file(self):
        """Auto-load selected file"""
        selected = st.session_state.get("workspace_selected_file")
        if not selected:
            return
            
        # Check if already open
        if self.state.conn.execute("SELECT 1 FROM tabs WHERE path = ?", (selected,)).fetchone():
            st.session_state.code_active_file = selected
            return
            
        # Load content
        content = self.state.read_file(selected)
        if content is not None:
            try:
                self.state.conn.execute(
                    "INSERT INTO tabs (path, content, is_dirty, last_saved) VALUES (?, ?, 0, ?)",
                    (selected, content, time.time())
                )
                self.state.conn.commit()
                st.session_state.code_active_file = selected
                logger.info(f"Loaded file: {selected}")
                st.rerun()
            except sqlite3.Error as e:
                logger.error(f"Failed to insert tab: {e}")
        else:
            st.error(f"Could not load file: {selected}")
            st.session_state.workspace_selected_file = None
            
    def _render_tabs(self):
        """Render tabbed interface"""
        tabs_data = self.state.conn.execute("SELECT path, content FROM tabs ORDER BY path").fetchall()
        if not tabs_data:
            return
            
        tab_paths = [path for path, _ in tabs_data]
        
        # Handle active file that was closed
        active_file = st.session_state.get("code_active_file")
        if active_file and active_file not in tab_paths:
            logger.info(f"Active file {active_file} no longer exists, clearing")
            st.session_state.code_active_file = None
            
        tabs = st.tabs(tab_paths + ["+"])
        
        for i, (path, content) in enumerate(tabs_data):
            with tabs[i]:
                self._render_editor_tab(path, content)
                
        # Handle new tab click
        if len(tabs) > len(tab_paths) and tabs[len(tab_paths)].button("New File"):
            st.session_state["workspace_selected_file"] = None
            st.rerun()
            
    def _render_editor_tab(self, file_path: str, content: str):
        """Render individual tab content"""
        col_actions, col_status = st.columns([4, 1])
        
        with col_actions:
            self._render_action_buttons(file_path)
            
        with col_status:
            # Check if row exists
            row = self.state.conn.execute(
                "SELECT is_dirty FROM tabs WHERE path = ?", (file_path,)
            ).fetchone()
            if row is None:
                status = "⚪ Closed"
            else:
                status = "🔴 Unsaved" if row[0] else "🟢 Saved"
            st.markdown(f"**{status}**")
                
        # Editor
        self._render_ace_editor(file_path, content)
        
    def _render_action_buttons(self, file_path: str):
        """Render action buttons"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("💾 Save", key=f"save_{file_path}"):
                self._save_file(file_path)
                
        with col2:
            if file_path.endswith('.py'):
                if st.button("▶️ Run", key=f"run_{file_path}"):
                    self._execute_file(file_path)
                    
        with col3:
            if st.button("🐛 Debug", key=f"debug_{file_path}"):
                self._debug_file(file_path)
                
        with col4:
            if st.button("🤖 Agent", key=f"agent_{file_path}"):
                st.session_state.show_agent_panel = True
                
        with col5:
            if st.button("❌ Close", key=f"close_{file_path}"):
                self._close_tab(file_path)
                
    def _render_ace_editor(self, file_path: str, content: str):
        """Render ACE editor"""
        ext = pathlib.Path(file_path).suffix.lower()
        lang_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".html": "html", ".css": "css", ".json": "json",
            ".md": "markdown", ".yaml": "yaml", ".yml": "yaml",
            ".txt": "text", ".sql": "sql", ".xml": "xml",
            ".sh": "sh", ".bash": "sh", ".zsh": "sh"
        }
        language = lang_map.get(ext, "text")
        
        # Get current content from database
        row = self.state.conn.execute(
            "SELECT content FROM tabs WHERE path = ?", (file_path,)
        ).fetchone()
        current_content = row[0] if row else content
        
        editor_content = st_ace(
            value=current_content,
            language=language,
            theme="monokai",
            font_size=14,
            tab_size=4,
            show_gutter=True,
            wrap=True,
            auto_update=False,
            height=500,
            key=f"ace_{file_path}"
        )
        
        # Only update if content actually changed
        if editor_content != current_content:
            try:
                self.state.conn.execute(
                    "UPDATE tabs SET content = ?, is_dirty = 1 WHERE path = ?",
                    (editor_content, file_path)
                )
                self.state.conn.commit()
                st.session_state.code_active_file = file_path
            except sqlite3.Error as e:
                logger.error(f"Failed to update tab: {e}")
            
    def _save_file(self, file_path: str):
        """Save file"""
        row = self.state.conn.execute(
            "SELECT content FROM tabs WHERE path = ?", (file_path,)
        ).fetchone()
        if row is None:
            st.error("File not found in editor")
            return
            
        content = row[0]
        
        if self.state.write_file(file_path, content):
            try:
                self.state.conn.execute(
                    "UPDATE tabs SET is_dirty = 0, last_saved = ? WHERE path = ?",
                    (time.time(), file_path)
                )
                self.state.conn.commit()
                st.success("✅ Saved!")
                st.rerun()
            except sqlite3.Error as e:
                logger.error(f"Failed to update save state: {e}")
        else:
            st.error("Save failed")
            
    def _execute_file(self, file_path: str):
        """Execute file"""
        row = self.state.conn.execute(
            "SELECT content FROM tabs WHERE path = ?", (file_path,)
        ).fetchone()
        if row is None:
            st.error("File not found")
            return
            
        content = row[0]
        
        async def run():
            return await AsyncExecutor.execute_code(
                content,
                cwd=str(self.state._abs_sandbox),
                timeout=60
            )
        
        stdout, stderr, code = asyncio.run(run())
        
        output = f"Exit code: {code}\n"
        if stdout:
            output += f"STDOUT:\n{stdout}\n"
        if stderr:
            output += f"STDERR:\n{stderr}\n"
            
        st.session_state.code_debug_session = {
            "file": file_path,
            "output": output,
            "type": "execution"
        }
        st.rerun()
        
    def _debug_file(self, file_path: str):
        """Debug file"""
        row = self.state.conn.execute(
            "SELECT content FROM tabs WHERE path = ?", (file_path,)
        ).fetchone()
        if row is None:
            st.error("File not found")
            return
            
        content = row[0]
        
        debugger = EnhancedDebugger(content, self.state.get_repl_namespace())
        st.session_state.code_debug_session = {
            "file": file_path,
            "debugger": debugger,
            "type": "debug"
        }
        st.rerun()
        
    def _close_tab(self, file_path: str):
        """Close tab with unsaved check"""
        row = self.state.conn.execute(
            "SELECT is_dirty FROM tabs WHERE path = ?", (file_path,)
        ).fetchone()
        
        if row is None:
            st.error("Tab already closed")
            return
            
        is_dirty = row[0]
        if is_dirty:
            if not st.checkbox("Discard unsaved changes?", key=f"discard_{file_path}"):
                return
                
        try:
            self.state.conn.execute("DELETE FROM tabs WHERE path = ?", (file_path,))
            self.state.conn.commit()
            
            # Clear active file if this was it
            if st.session_state.get("code_active_file") == file_path:
                st.session_state.code_active_file = None
                
            logger.info(f"Closed tab: {file_path}")
            st.rerun()
        except sqlite3.Error as e:
            logger.error(f"Failed to close tab: {e}")
        
    def _auto_save(self):
        """Auto-save dirty tabs"""
        try:
            cursor = self.state.conn.execute(
                "SELECT path, content FROM tabs WHERE is_dirty = 1"
            )
            for path, content in cursor:
                self.state.write_file(path, content)
                self.state.conn.execute(
                    "UPDATE tabs SET is_dirty = 0, last_saved = ? WHERE path = ?",
                    (time.time(), path)
                )
            self.state.conn.commit()
            logger.info(f"Auto-saved {cursor.rowcount} tabs")
        except sqlite3.Error as e:
            logger.error(f"Auto-save error: {e}")

class DebuggerPanel:
    """Enhanced debugger panel"""
    
    def render(self):
        """Render debugger panel"""
        if not st.session_state.code_debug_session:
            return
            
        st.markdown("---")
        st.subheader("🐛 Debugger")
        
        session = st.session_state.code_debug_session
        
        if session["type"] == "execution":
            self._render_execution_output(session)
        elif session["type"] == "debug" and "debugger" in session:
            self._render_debug_controls(session["debugger"])
            
    def _render_execution_output(self, session: Dict):
        """Render execution output"""
        st.code(session["output"], language="text")
        
    def _render_debug_controls(self, debugger: EnhancedDebugger):
        """Render debug controls"""
        if debugger.current_inst < len(debugger.instructions):
            current_line = debugger.instructions[debugger.current_inst].lineno
            st.info(f"Current line: {current_line}")
        
        self._render_code_with_breakpoints(debugger)
        self._render_control_buttons(debugger)
        self._render_watch_expressions(debugger)
        self._render_variables(debugger)
        
    def _render_code_with_breakpoints(self, debugger: EnhancedDebugger):
        """Render code with breakpoints"""
        source_lines = debugger.code.splitlines()
        
        for i, line in enumerate(source_lines, 1):
            col1, col2 = st.columns([1, 20])
            
            with col1:
                is_bp = i in debugger.breakpoints
                if st.checkbox(
                    "⚫" if is_bp else "⚪",
                    value=is_bp,
                    key=f"bp_{i}",
                    label_visibility="collapsed"
                ):
                    debugger.toggle_breakpoint(i)
                    st.rerun()
                    
            with col2:
                if debugger.current_inst < len(debugger.instructions) and \
                   debugger.instructions[debugger.current_inst].lineno == i:
                    st.markdown(f"**→ {i:3} | {line}**")
                else:
                    st.code(f"{i:3} | {line}", language=None)
                    
    def _render_control_buttons(self, debugger: EnhancedDebugger):
        """Render control buttons"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Step Over"):
                debugger.step()
                st.rerun()
                
        with col2:
            if st.button("Continue"):
                while debugger.current_inst < len(debugger.instructions):
                    should_pause, _ = debugger.step()
                    if should_pause:
                        break
                st.rerun()
                
        with col3:
            if st.button("Stop"):
                st.session_state.code_debug_session = None
                st.rerun()
                
    def _render_watch_expressions(self, debugger: EnhancedDebugger):
        """Render watch expressions"""
        st.markdown("---")
        st.subheader("👁️ Watch Expressions")
        
        new_watch = st.text_input("Add watch", placeholder="variable_name")
        if st.button("Add") and new_watch:
            debugger.add_watch(new_watch)
            st.rerun()
            
        for expr in list(debugger.watch_expressions.keys()):
            col1, col2 = st.columns([3, 1])
            with col1:
                value = debugger.eval_watch(expr)
                st.text(f"{expr}: {value}")
            with col2:
                if st.button("Remove", key=f"remove_watch_{expr}"):
                    debugger.remove_watch(expr)
                    st.rerun()
                    
    def _render_variables(self, debugger: EnhancedDebugger):
        """Render variables"""
        st.markdown("---")
        st.subheader("📊 Variables")
        
        snapshot = debugger._get_namespace_snapshot()
        if snapshot:
            for var, val in snapshot.items():
                if var != "__error__":
                    st.text(f"{var}: {val}")
        else:
            st.info("No local variables")

class AgentPanel:
    """Enhanced agent panel with DB bridging"""
    
    def __init__(self, state: CodeLabStateV2):
        self.state = state
        self.agent = AgentIntegration()
        
    def render(self):
        """Render agent panel"""
        if not _get_state("show_agent_panel", False):
            return
            
        st.sidebar.markdown("---")
        st.sidebar.header("🤖 Agent Pair-Programmer")
        
        task = st.sidebar.text_area(
            "What do you want the agent to help with?",
            "Review this code for bugs and suggest improvements",
            height=100
        )
        
        with st.sidebar.expander("Advanced Options"):
            focus = st.selectbox("Focus area", ["General Review", "Bug Detection", "Performance", "Security", "Testing"])
            depth = st.slider("Analysis depth", 1, 5, 3)
            
        if st.sidebar.button("Spawn Agent", type="primary"):
            self._spawn_agent(task, focus, depth)
            
        # Load suggestions from Code-Lab's DB for rendering
        active_file = st.session_state.code_active_file
        if active_file:
            # Generate agent_id for this file+task
            task_hash = hashlib.md5(task.encode()).hexdigest()[:8]
            agent_id = f"agent_coder_{active_file.replace('/', '_')}_{task_hash}"
            cached_result = self.state.get_agent_result(agent_id)
            if cached_result:
                suggestions = self.agent.parse_suggestions(cached_result)
                if suggestions:
                    self._render_suggestions(suggestions)
            
    def _spawn_agent(self, task: str, focus: str, depth: int):
        """Spawn agent, bridge DBs, store in Code-Lab DB"""
        active_file = st.session_state.code_active_file
        if not active_file:
            st.sidebar.error("No active file")
            return
            
        row = self.state.conn.execute("SELECT content FROM tabs WHERE path = ?", (active_file,)).fetchone()
        if row is None:
            st.sidebar.error("File content not available")
            return
            
        code = row[0]
        convo_uuid = st.session_state.get("current_convo_uuid", str(uuid.uuid4()))
        
        # Generate unique agent_id for this file+task
        task_hash = hashlib.md5(task.encode()).hexdigest()[:8]
        agent_id = f"agent_coder_{active_file.replace('/', '_')}_{task_hash}"
        
        context = {"focus": focus, "depth": depth}
        
        with st.spinner("Agent analyzing (may take up to 2 minutes)..."):
            # 1. Spawn and wait for completion
            status_result = _safe_tool_call(
                "agent_spawn",
                sub_agent_type="coder",
                task=task,
                convo_uuid=convo_uuid,
                model="kimi-k2-thinking",
                auto_poll=True,
                poll_interval=3
            )
            
            if not status_result or "Error" in status_result:
                logger.error(f"Agent spawn failed: {status_result}")
                st.sidebar.error(f"Agent failed: {status_result}")
                return
            
            # 2. Extract agent_id from status
            agent_id_match = re.search(r"ID:\s*(agent_[\w]+)", status_result)
            if not agent_id_match:
                logger.warning(f"Use Memory Explorer or Main Chat Agent View Results: {status_result}")
                st.sidebar.warning("Agent completed Results Persisted")
                return
                
            main_agent_id = agent_id_match.group(1)
            logger.info(f"Agent completed with ID: {main_agent_id}. Fetching from main DB...")
            
            # 3. Fetch result from main app's DB (chatapp.db)
            full_result = _safe_tool_call(
                "memory_query",
                mem_key=f"{main_agent_id}_result",
                convo_uuid=convo_uuid
            )
            
            # Handle different return formats
            result_text = None
            if isinstance(full_result, dict) and "response" in full_result:
                result_text = full_result["response"]
            elif isinstance(full_result, str) and "Key not found" not in full_result:
                result_text = full_result
            else:
                # Final fallback - try alternative key
                full_result = _safe_tool_call(
                    "memory_query",
                    mem_key=f"agent_{main_agent_id}_result",
                    convo_uuid=convo_uuid
                )
                if isinstance(full_result, dict) and "response" in full_result:
                    result_text = full_result["response"]
                elif isinstance(full_result, str) and "Key not found" not in full_result:
                    result_text = full_result
            
            if result_text:
                # 4. Store in Code-Lab's own DB for UI rendering
                self.state.store_agent_result(agent_id, result_text)
                
                # 5. Parse and render
                suggestions = self.agent.parse_suggestions(result_text)
                if suggestions:
                    st.session_state.code_agent_suggestions = suggestions
                    st.sidebar.success(f"✅ {len(suggestions)} suggestions ready!")
                    st.rerun()  # Force UI refresh to show suggestions
                else:
                    st.sidebar.info("Agent returned no actionable suggestions")
            else:
                logger.error(f"Could not fetch agent result: {full_result}")
                st.sidebar.error("Agent completed but no result data found")

    def _render_suggestions(self, suggestions: List[Dict]):
        """Render suggestions"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("💡 Suggestions")
        
        for sugg in suggestions:
            with st.sidebar.expander(f"Line {sugg['line']}: {sugg['code'][:40]}...", expanded=False):
                st.code(sugg['code'], language="python")
                if sugg['explanation']:
                    st.text(sugg['explanation'])
                    
                col1, col2 = st.sidebar.columns(2)
                
                with col1:
                    if st.button("✅ Accept", key=f"accept_{sugg['id']}"):
                        try:
                            active_file = st.session_state.code_active_file
                            row = self.state.conn.execute("SELECT content FROM tabs WHERE path = ?", (active_file,)).fetchone()
                            if row is None:
                                st.sidebar.error("File not available")
                                continue
                                
                            original = row[0]
                            new_code = self.agent.apply_suggestion(original, sugg)
                            
                            self.state.conn.execute(
                                "UPDATE tabs SET content = ?, is_dirty = 1 WHERE path = ?",
                                (new_code, active_file)
                            )
                            self.state.conn.commit()
                            
                            suggestions.remove(sugg)
                            st.session_state.code_agent_suggestions = suggestions
                            st.rerun()
                        except ValueError as e:
                            st.sidebar.error(str(e))
                            
                with col2:
                    if st.button("❌ Dismiss", key=f"dismiss_{sugg['id']}"):
                        suggestions.remove(sugg)
                        st.session_state.code_agent_suggestions = suggestions
                        st.rerun()

class GitPanel:
    """Enhanced Git panel with sandbox path (FIX #3)"""
    
    def __init__(self, state: CodeLabStateV2):
        self.state = state
        
    def render(self):
        """Render Git panel"""
        st.sidebar.header("🌿 Git Operations")
        
        # ← FIX #3: Use sandbox directory as default repo path
        repo_path = st.sidebar.text_input("Repository path", value=self.state.sandbox_dir)
        
        if not self.state._validate_path(pathlib.Path(repo_path)):
            st.sidebar.error("Repository path outside sandbox!")
            return
            
        if st.sidebar.button("Refresh Status"):
            status = _safe_tool_call("git_ops", operation="status", repo_path=repo_path)
            if status and "Error" not in status:
                st.sidebar.code(status)
            else:
                st.sidebar.error(f"Git status failed: {status}")
                
        with st.sidebar.expander("Branch Management"):
            new_branch = st.text_input("New branch name")
            if st.button("Create Branch") and new_branch:
                result = _safe_tool_call(
                    "git_ops", 
                    operation="branch", 
                    repo_path=repo_path, 
                    name=new_branch
                )
                if result and "Error" not in result:
                    st.sidebar.success(result)
                else:
                    st.sidebar.error(f"Branch creation failed: {result}")
                    
        with st.sidebar.expander("Commit"):
            col1, col2 = st.sidebar.columns([2, 1])
            with col1:
                commit_msg = st.sidebar.text_input("Commit message", placeholder="Describe changes")
            with col2:
                if st.button("Commit") and commit_msg:
                    result = _safe_tool_call(
                        "git_ops", 
                        operation="commit", 
                        repo_path=repo_path, 
                        message=commit_msg
                    )
                    if result and "Error" not in result:
                        st.sidebar.success(result)
                    else:
                        st.sidebar.error(f"Commit failed: {result}")
                        
        if st.sidebar.button("Stage All"):
            result = _safe_tool_call("git_ops", operation="stage_all", repo_path=repo_path)
            if result and "Error" not in result:
                st.sidebar.code(result)
            else:
                st.sidebar.error(f"Staging failed: {result}")

class StaticAnalysisPanel:
    """Static analysis and linting"""
    
    def __init__(self, state: CodeLabStateV2):
        self.state = state
        
    def render(self):
        """Render analysis panel"""
        st.sidebar.header("🔍 Static Analysis")
        
        active_file = st.session_state.code_active_file
        if not active_file:
            st.sidebar.info("Open a file to analyze")
            return
            
        if not self.state.conn.execute("SELECT 1 FROM tabs WHERE path = ?", (active_file,)).fetchone():
            st.sidebar.warning("File not open in editor")
            return
            
        if st.sidebar.button("Analyze Now"):
            content = self._get_file_content(active_file)
            if content:
                analysis = self._analyze_code(content)
                self._render_analysis_results(analysis)
            
    def _get_file_content(self, file_path: str) -> Optional[str]:
        """Get file content with null check"""
        row = self.state.conn.execute(
            "SELECT content FROM tabs WHERE path = ?", (file_path,)
        ).fetchone()
        return row[0] if row else None
            
    def _analyze_code(self, code: str) -> Dict:
        """Analyze code using AST"""
        try:
            tree = ast.parse(code)
            analyzer = CodeAnalyzer()
            analyzer.visit(tree)
            return analyzer.metrics
        except SyntaxError as e:
            return {"error": str(e)}
            
    def _render_analysis_results(self, analysis: Dict):
        """Render analysis results"""
        if "error" in analysis:
            st.sidebar.error(f"Syntax Error: {analysis['error']}")
            return
            
        st.sidebar.metric("Functions", analysis["functions"])
        st.sidebar.metric("Classes", analysis["classes"])
        st.sidebar.metric("Complexity", analysis["complexity"])
        
        if analysis["imports"]:
            st.sidebar.markdown("**Imports:**")
            for imp in analysis["imports"][:5]:
                st.sidebar.text(f"• {imp}")

# ============================================================================
# MAIN PAGE - ENTRY POINT
# ============================================================================
def main():
    st.set_page_config(
        page_title="Code-Lab-Pro v2",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # CRITICAL FIX: Initialize ALL session state keys with defaults
    session_defaults = {
        "code_editor_tabs": {},
        "code_active_file": None,
        "show_agent_panel": False,
        "code_debug_session": None,
        "code_agent_suggestions": [],
        "workspace_selected_file": None,
        "auto_save_last": time.time(),
        "expanded_folders": set(),
        "tool_calls_per_convo": 0,
        "last_reasoning": None
    }
    
    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
            logger.debug(f"Initialized session state: {key} = {default_value}")
    
    # Initialize UI components
    state = CodeLabStateV2()
    file_browser = FileBrowser(state)
    editor = Editor(state)
    debugger = DebuggerPanel()
    agent_panel = AgentPanel(state)
    git_panel = GitPanel(state)
    analysis_panel = StaticAnalysisPanel(state)
    
    # Layout
    col_sidebar, col_main = st.columns([1, 4])
    
    with col_sidebar:
        file_browser.render()
        git_panel.render()
        analysis_panel.render()
        
    with col_main:
        editor.render()
        debugger.render()
        
    # Floating panels
    agent_panel.render()
    
    # Footer metrics
    st.sidebar.markdown("---")
    try:
        tab_count = state.conn.execute("SELECT COUNT(*) FROM tabs").fetchone()[0]
        st.sidebar.metric("Open Tabs", tab_count)
        debug_status = "Active" if st.session_state.code_debug_session else "Inactive"
        st.sidebar.metric("Debug Session", debug_status)
        
        # Add cleanup button
        if st.sidebar.button("🧹 Cleanup Old Data"):
            cleanup_old_agents()
            st.sidebar.success("Cleanup complete!")
            
    except Exception as e:
        logger.error(f"Footer metrics error: {e}")

# ============================================================================
# CLEANUP FUNCTION - Integrated at bottom
# ============================================================================
def cleanup_old_agents():
    """Clean agent results older than 30 days from both DBs"""
    try:
        logger.info("Starting cleanup of old agent data...")
        
        # Clean Code-Lab DB
        state = CodeLabStateV2()
        state.conn.execute(
            "DELETE FROM agent_results WHERE created_at < datetime('now', '-30 days')"
        )
        state.conn.commit()
        logger.info("Cleaned old agent results from Code-Lab DB")
        
        # Clean tabs older than 90 days
        state.conn.execute(
            "DELETE FROM tabs WHERE last_saved < ? AND is_dirty = 0",
            (time.time() - 7776000,)  # 90 days in seconds
        )
        state.conn.commit()
        logger.info("Cleaned old tabs from Code-Lab DB")
        
        # Clean main DB (optional - commented out to be safe)
        # _safe_tool_call(
        #     "advanced_memory_prune", 
        #     convo_uuid=st.session_state.get("current_convo_uuid", "")
        # )
        # logger.info("Triggered prune in main DB")
        
        st.sidebar.success("✅ Cleanup complete! Old data removed.")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        st.sidebar.error(f"Cleanup error: {e}")

if __name__ == "__main__":
    main()
