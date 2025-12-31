# pages/group_chat_phase3_complete.py
# 🜛 PAC Hive Phase-3: Aurum Aurifex - Complete Integration
# All Phase-3 modules embedded, no main.py changes needed

import streamlit as st
import aiohttp
import asyncio
import json
import copy
import os
import sys
import time
import uuid
import httpx
import sqlite3
import networkx as nx
import plotly.graph_objects as go
from typing import List, Dict, Callable, Optional, Any, Tuple
from datetime import datetime
from asyncio import Semaphore
import inspect
from dataclasses import dataclass

# === CRITICAL: Import Diagnostics & Path Resolution ===
def diagnose_imports():
    """Diagnose and fix import paths for main.py"""
    current_file = os.path.abspath(__file__)
    pages_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(pages_dir)
    
    st.info(f"📍 **Diagnostic Info:**\n- Current file: `{current_file}`\n- Pages dir: `{pages_dir}`\n- Project root: `{project_root}`")
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        st.success(f"✅ Added `{project_root}` to Python path")
    
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        site_packages = os.path.join(venv_path, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
        if site_packages not in sys.path:
            sys.path.append(site_packages)
            st.success(f"✅ Added venv site-packages to path")
    
    return project_root

PROJECT_ROOT = diagnose_imports()

# === Robust Import with Fallback ===
MAIN_SCRIPT_AVAILABLE = False
try:
    from main import (
        state, container, TOOL_DISPATCHER, MOONSHOT_OFFICIAL_TOOLS,
        get_moonshot_tools, process_tool_calls, execute_moonshot_formula,
        MoonshotRateLimiter, get_memory_cache, inject_convo_uuid,
        memory_insert, advanced_memory_consolidate, tool_limiter_sync,
        Models, Config
    )
    MAIN_SCRIPT_AVAILABLE = True
    st.success("🎉 Successfully imported from `main.py`")
    
except ImportError as e1:
    st.warning(f"⚠️ First import attempt failed: `{e1}`")
    
    try:
        sys.path.insert(0, os.getcwd())
        from main import *
        MAIN_SCRIPT_AVAILABLE = True
        st.success("🎉 Successfully imported with wildcard")
        
    except ImportError as e2:
        st.error(f"❌ All import attempts failed: `{e2}`")
        
        # === EMERGENCY MOCK OBJECTS ===
        st.warning("🛠️ Loading mock objects for development...")
        
        class MockState:
            def __init__(self):
                self.counter_lock = asyncio.Lock()
                self.agent_sem = asyncio.Semaphore(3)
                self.conn = None
                self.cursor = None
                self.chroma_lock = asyncio.Lock()
                self.sandbox_dir = "./sandbox"
                self.yaml_dir = "./sandbox/config"
                self.agent_dir = "./sandbox/agents"
                for d in [self.sandbox_dir, self.yaml_dir, self.agent_dir]:
                    os.makedirs(d, exist_ok=True)
        
        class MockContainer:
            def __init__(self):
                self._tools = {}
                self._official_tools = {}
            def register_tool(self, func, name=None):
                self._tools[name or func.__name__] = func
            def get_official_tools(self):
                return self._official_tools
        
        def mock_process_tool_calls(*args):
            return iter([])
        
        def mock_get_memory_cache():
            return {"lru_cache": {}, "metrics": {"total_inserts": 0, "total_retrieves": 0, "hit_rate": 1.0}}
        
        def mock_memory_insert(*args, **kwargs):
            return "Memory disabled (mock)"
        
        def mock_tool_limiter_sync():
            time.sleep(0.01)
        
        state = MockState()
        container = MockContainer()
        TOOL_DISPATCHER = {}
        MOONSHOT_OFFICIAL_TOOLS = {}
        get_moonshot_tools = lambda *a, **k: []
        process_tool_calls = mock_process_tool_calls
        execute_moonshot_formula = lambda *a, **k: {"error": "No main script"}
        MoonshotRateLimiter = type('MockLimiter', (), {})()
        get_memory_cache = mock_get_memory_cache
        inject_convo_uuid = lambda f: f
        memory_insert = mock_memory_insert
        advanced_memory_consolidate = lambda *a, **k: "Consolidation disabled"
        tool_limiter_sync = mock_tool_limiter_sync
        Models = type('Models', (), {'KIMI_K2': 'kimi-k2', 'MOONSHOT_V1_32K': 'moonshot-v1-32k'})
        Config = type('Config', (), {'DEFAULT_TOP_K': 5, 'TOOL_CALLS_PER_MIN': 10})

# === PHASE-3 MODULE 1: Living Grimoire (Memory Graph) ===
class MemoryGrimoire:
    """Symbolic Memory Graph - no glyph parsing, just semantic linking"""
    def __init__(self, sqlite_conn):
        self.graph = nx.DiGraph()
        self.conn = sqlite_conn
        self._load_existing_memories()
    
    def _load_existing_memories(self):
        """Bootstrap graph from existing SQLite memories"""
        if not self.conn:
            return
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT memory_id, content FROM memories")
            for row in cursor.fetchall():
                self.graph.add_node(row[0], content=row[1][:200])
        except:
            pass
    
    def add_evidence(self, memory_id: str, content: Dict, agent: str, 
                     glyphs: List[str] = None, contradicts: List[str] = None, 
                     supports: List[str] = None):
        """Add memory node with symbolic edges - glyphs are just tags"""
        self.graph.add_node(memory_id, **content, agent=agent, glyphs=glyphs or [], timestamp=datetime.now().isoformat())
        
        for target in contradicts or []:
            self.graph.add_edge(memory_id, target, relation="contradicts", weight=-0.7)
        
        for target in supports or []:
            self.graph.add_edge(memory_id, target, relation="supports", weight=0.9)
        
        # Pheromone trail: auto-link recent memories by agent
        recent = [
            n for n in self.graph.nodes 
            if self.graph.nodes[n].get("agent") == agent
        ]
        recent_sorted = sorted(recent, key=lambda n: self.graph.nodes[n].get("timestamp", ""), reverse=True)[:3]
        
        for r in recent_sorted:
            if r != memory_id:
                self.graph.add_edge(memory_id, r, relation="temporal_chain", weight=0.3)
    
    def propagate_salience(self, seed_id: str, decay: float = 0.85) -> Dict[str, float]:
        """Memory activation propagation - like neuron firing"""
        activation = {seed_id: 1.0}
        frontier = [seed_id]
        
        while frontier:
            current = frontier.pop(0)
            current_score = activation[current]
            
            for neighbor in self.graph.neighbors(current):
                edge_data = self.graph.get_edge_data(current, neighbor) or {}
                weight = edge_data.get("weight", 0.5)
                new_score = current_score * weight * decay
                
                if neighbor not in activation or new_score > activation[neighbor]:
                    activation[neighbor] = new_score
                    if new_score > 0.1:
                        frontier.append(neighbor)
        
        return activation
    
    def export_for_visualization(self) -> str:
        """Export for D3.js frontend"""
        try:
            return json.dumps({
                "nodes": [{"id": n, **self.graph.nodes[n]} for n in self.graph.nodes],
                "edges": [{"source": u, "target": v, **data} for u, v, data in self.graph.edges(data=True)]
            })
        except:
            return json.dumps({"nodes": [], "edges": []})

# === PHASE-3 MODULE 2: Pheromone Coordination ===
@dataclass
class PheromoneTrail:
    topic: str
    intensity: float
    agent_name: str
    timestamp: float

class PheromoneTracker:
    """Detects emergent topics without parsing PAC content"""
    
    TOPIC_KEYWORDS = {
        "database": "db_optimization", "sql": "db_optimization", "memory": "memory_systems",
        "prune": "memory_systems", "vector": "memory_systems", "tool": "tool_usage",
        "api": "tool_usage", "cost": "cost_tracking", "token": "cost_tracking",
        "async": "concurrency", "semaphore": "concurrency"
    }
    
    def __init__(self, spawn_callback: Callable):
        self.trails: List[PheromoneTrail] = []
        self.spawn_callback = spawn_callback
    
    def analyze_message(self, message: str, agent_name: str) -> List[str]:
        """Extract pheromones from plain text"""
        trails = []
        for keyword, topic in self.TOPIC_KEYWORDS.items():
            if re.search(r'\b' + keyword + r'\b', message, re.IGNORECASE):
                frequency = len(re.findall(keyword, message, re.IGNORECASE))
                urgency_markers = len(re.findall(r'!', message))
                intensity = min(0.9, 0.3 * frequency + 0.2 * urgency_markers)
                trails.append(PheromoneTrail(topic, intensity, agent_name, asyncio.get_event_loop().time()))
        
        self.trails.extend(trails)
        return [t.topic for t in trails]
    
    async def evaluate_swarm_needs(self) -> List[Dict]:
        """Check if new swarm cells should spawn"""
        now = asyncio.get_event_loop().time()
        active_trails = [t for t in self.trails if now - t.timestamp < 120]
        
        topic_intensity = {}
        for trail in active_trails:
            topic_intensity[trail.topic] = topic_intensity.get(trail.topic, 0) + trail.intensity
        
        spawn_requests = []
        for topic, intensity in topic_intensity.items():
            if intensity > 1.5:
                spawn_requests.append({
                    "topic": topic,
                    "intensity": intensity,
                    "specialist_role": self._get_specialist_role(topic)
                })
        
        return spawn_requests
    
    def _get_specialist_role(self, topic: str) -> str:
        role_map = {
            "db_optimization": "Database query optimizer. Focus on SQL, indexing, connection pools.",
            "memory_systems": "Memory systems architect. Focus on pruning, retention, vector search.",
            "tool_usage": "Tool efficiency analyst. Track ROI, reduce redundant calls.",
            "cost_tracking": "Cost optimization specialist. Minimize token waste.",
            "concurrency": "Async concurrency engineer. Optimize semaphore usage."
        }
        return role_map.get(topic, "General problem solver.")

# === PHASE-3 MODULE 3: Hyper-Cognitive Observability ===
class TokenFlowVisualizer:
    """Real-time token flow Sankey diagram"""
    def __init__(self, ledger):
        self.ledger = ledger
        self.flow_log = []
    
    def log_flow(self, source: str, target: str, tokens: int):
        self.flow_log.append({
            "source": source,
            "target": target,
            "tokens": tokens,
            "timestamp": datetime.now().isoformat()
        })
    
    def render_sankey(self) -> str:
        """Generate Plotly Sankey diagram HTML"""
        try:
            flow_map = {}
            for flow in self.flow_log[-50:]:
                key = (flow["source"], flow["target"])
                flow_map[key] = flow_map.get(key, 0) + flow["tokens"]
            
            nodes = list(set([s for s,_ in flow_map.keys()] + [t for _,t in flow_map.keys()]))
            node_indices = {n: i for i, n in enumerate(nodes)}
            
            fig = go.Figure(data=[go.Sankey(
                node=dict(label=nodes),
                link=dict(
                    source=[node_indices[s] for s,_ in flow_map.keys()],
                    target=[node_indices[t] for _,t in flow_map.keys()],
                    value=list(flow_map.values())
                )
            )])
            
            return fig.to_html()
        except:
            return "<div>Flow visualization unavailable</div>"

class AgentStateLens:
    """Sidebar inspector for agent cognitive state"""
    @staticmethod
    def inspect(agent_name: str, history: List[Dict]) -> str:
        agent_msgs = [m for m in history if m["name"] == agent_name][-10:]
        tool_usage = len([m for m in agent_msgs if "🜛 **Tool Results:**" in m["content"]])
        avg_length = sum(len(m["content"]) for m in agent_msgs) / len(agent_msgs) if agent_msgs else 0
        
        return f"""
        **Agent: {agent_name}**
        - Recent messages: {len(agent_msgs)}
        - Tool calls: {tool_usage}
        - Avg length: {avg_length:.0f} chars
        - Focus: {AgentStateLens._classify_focus(agent_msgs)}
        """
    
    @staticmethod
    def _classify_focus(msgs: List[Dict]) -> str:
        text = " ".join([m["content"] for m in msgs]).lower()
        if "tool" in text and "cost" in text:
            return "Cost optimization"
        if "memory" in text:
            return "Memory systems"
        return "General reasoning"

# === PHASE-3 MODULE 4: External Tools (Zero Main.py Changes) ===
@inject_convo_uuid
def webhook_invoker(url: str, payload: dict, method: str = "POST", convo_uuid: str = None) -> str:
    """🜛 Tool: Webhook Invoker - Agents trigger external services"""
    try:
        loop = asyncio.get_event_loop()
        async def _call():
            async with httpx.AsyncClient() as client:
                response = await client.request(method, url, json=payload)
                return f"Webhook {method} to {url}: {response.status_code}"
        return loop.run_until_complete(_call())
    except Exception as e:
        return f"Webhook error: {str(e)}"

@inject_convo_uuid
def database_persist(table: str, data: dict, convo_uuid: str = None) -> str:
    """🜛 Tool: Database Persistence - Write to external SQLite"""
    try:
        conn = sqlite3.connect("./sandbox/external_data.db")
        cursor = conn.cursor()
        columns = ", ".join([f"{k} TEXT" for k in data.keys()])
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY, {columns})")
        placeholders = ", ".join(["?" for _ in data])
        cursor.execute(f"INSERT INTO {table} VALUES (NULL, {placeholders})", list(data.values()))
        conn.commit()
        conn.close()
        return f"Persisted to {table}: {data}"
    except Exception as e:
        return f"DB error: {str(e)}"

@inject_convo_uuid
def database_query(query: str, convo_uuid: str = None) -> str:
    """🜛 Tool: Database Query - Read from external SQLite"""
    try:
        conn = sqlite3.connect("./sandbox/external_data.db")
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()
        return f"Query results: {results}"
    except Exception as e:
        return f"Query error: {str(e)}"

# Auto-registration function for external tools
def register_external_tools(container):
    """Register Phase-3 external tools with the container"""
    if not MAIN_SCRIPT_AVAILABLE:
        return
    container.register_tool(webhook_invoker, name="webhook_invoker")
    container.register_tool(database_persist, name="database_persist")
    container.register_tool(database_query, name="database_query")

# === PHASE-3 MODULE 5: Quick Wins UI ===
class MemorySearchUI:
    @staticmethod
    def render():
        """Sidebar memory search using existing vector DB"""
        st.sidebar.subheader("🔍 Memory Search")
        query = st.sidebar.text_input("Search memories", key="mem_search")
        if query:
            cache = get_memory_cache()
            results = cache.get("lru_cache", {})
            matches = [(k, v) for k, v in results.items() if query.lower() in str(v).lower()]
            for key, content in matches[:5]:
                with st.sidebar.expander(f"Memory: {key[:30]}"):
                    st.code(content, language="text")

class OneClickReproduction:
    @staticmethod
    def generate_script():
        """Generate standalone reproduction script"""
        if st.sidebar.button("📦 One-Click Reproduction"):
            script = f"""
# PAC Hive Reproduction Script (Auto-generated)
# Generated: {datetime.now().isoformat()}
import streamlit as st
import asyncio
HIVE_CONFIG = {json.dumps({
    "agents": [{"name": a.name, "model": a.model, "temp": a.temperature} for a in st.session_state.agents],
    "bootstrap": st.session_state.pac_bootstrap,
    "max_turns": 10,
    "termination": "SOLVE ET COAGULA COMPLETE"
}, indent=2)}
SEED = "{st.session_state.hive_history[0]['content'] if st.session_state.hive_history else ''}"
if __name__ == "__main__":
    st.set_page_config(page_title="PAC Hive Reproduction")
    st.write("Reproduction logic would go here")
"""
            st.sidebar.download_button("Download reproduce.py", script, file_name="reproduce.py")

# === CONFIG ===
API_KEY = os.getenv("MOONSHOT_API_KEY") or st.secrets.get("MOONSHOT_API_KEY")
if not API_KEY:
    st.error("🜩 MOONSHOT_API_KEY not found in .env or secrets! Add it and restart.")
    st.stop()

BASE_URL = "https://api.moonshot.ai/v1"
DEFAULT_MODEL = "moonshot-v1-32k"
MODEL_OPTIONS = [
    "moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k",
    "kimi-k2-thinking", "kimi-k2-thinking-turbo", "kimi-k2", "kimi-latest"
]

# === TOOL BRIDGE (Enhanced with Phase-3) ===
class ToolBridge:
    """Bridges the Hive to the main script's tool ecosystem with Phase-3 enhancements"""
    
    def __init__(self):
        self.custom_tools = {}
        self.official_tools = {}
        self._load_tools()
        # Phase-3: Initialize Memory Graph
        self.grimoire = MemoryGrimoire(state.conn if hasattr(state, 'conn') else None)
    
    def _load_tools(self):
        """Load tools from main script + register Phase-3 external tools"""
        if not MAIN_SCRIPT_AVAILABLE:
            st.warning("Running without main script - tools disabled")
            return
        
        # Load custom sandbox tools
        for name, func in container._tools.items():
            self.custom_tools[name] = {
                'func': func,
                'schema': self._generate_schema(func),
                'type': 'custom'
            }
        
        # Load official Moonshot tools
        for name, uri in MOONSHOT_OFFICIAL_TOOLS.items():
            self.official_tools[name] = {
                'uri': uri,
                'schema': self._get_official_schema(name),
                'type': 'official'
            }
        
        # Phase-3: Register external tools
        try:
            register_external_tools(container)
            # Reload after registration
            for name, func in container._tools.items():
                if name not in self.custom_tools:  # New Phase-3 tools
                    self.custom_tools[name] = {
                        'func': func,
                        'schema': self._generate_schema(func),
                        'type': 'custom'
                    }
        except Exception as e:
            st.warning(f"Phase-3 external tools not loaded: {e}")
    
    def _generate_schema(self, func: Callable) -> Dict:
        """Generate OpenAI-style schema for custom tools"""
        try:
            from main import generate_tool_schema
            return generate_tool_schema(func)
        except:
            sig = inspect.signature(func)
            properties = {}
            required = []
            type_map = {int: "integer", bool: "boolean", str: "string", float: "number", list: "array", dict: "object"}
            
            for param_name, param in sig.parameters.items():
                ann = param.annotation
                prop_type = type_map.get(ann, "string")
                properties[param_name] = {"type": prop_type, "description": f"Parameter: {param_name}"}
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)
            
            return {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": inspect.getdoc(func) or "No description",
                    "parameters": {"type": "object", "properties": properties, "required": required}
                }
            }
    
    def _get_official_schema(self, name: str) -> Dict:
        schemas = {
            "moonshot-web-search": {
                "type": "function",
                "function": {
                    "name": "moonshot-web-search",
                    "description": "Search the web for current information",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
                }
            },
            "moonshot-calculate": {
                "type": "function",
                "function": {
                    "name": "moonshot-calculate",
                    "description": "Perform mathematical calculations",
                    "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
                }
            },
            "moonshot-url-extract": {
                "type": "function",
                "function": {
                    "name": "moonshot-url-extract",
                    "description": "Extract content from a URL",
                    "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
                }
            }
        }
        return schemas.get(name, {})
    
    def get_all_tools(self, enable_custom: bool = True, enable_official: bool = True) -> List[Dict]:
        tools = []
        if enable_official:
            tools.extend([schema for schema in self.official_tools.values()])
        if enable_custom:
            tools.extend([tool['schema'] for tool in self.custom_tools.values()])
        return tools
    
    async def execute_tool(self, name: str, arguments: Dict, convo_uuid: str) -> str:
        """Execute tool with Phase-3 Memory Graph integration"""
        try:
            await asyncio.to_thread(tool_limiter_sync)
            
            if not convo_uuid:
                convo_uuid = st.session_state.get("current_convo_uuid", str(uuid.uuid4()))
            
            func = None
            tool_type = None
            
            if name in self.custom_tools:
                func = self.custom_tools[name]['func']
                tool_type = 'custom'
            elif name in self.official_tools:
                tool_type = 'official'
            else:
                return f"Error: Tool '{name}' not found"
            
            if tool_type == 'custom':
                sig = inspect.signature(func)
                if 'convo_uuid' in sig.parameters and 'convo_uuid' not in arguments:
                    arguments['convo_uuid'] = convo_uuid
                
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, func, **arguments)
                result_str = str(result) if result is not None else "Tool returned None"
                
                # Phase-3: Log to Memory Graph
                try:
                    self.grimoire.add_evidence(
                        memory_id=f"tool_{name}_{uuid.uuid4().hex[:8]}",
                        content={"result": result_str[:300], "tool": name},
                        agent=convo_uuid,
                        glyphs=["tool", name]
                    )
                except Exception as e:
                    st.sidebar.warning(f"Memory graph error: {e}")
                
                return result_str
            
            elif tool_type == 'official':
                result = await asyncio.to_thread(
                    execute_moonshot_formula,
                    self.official_tools[name]['uri'],
                    name,
                    arguments,
                    API_KEY
                )
                return json.dumps(result) if isinstance(result, dict) else str(result)
            
        except Exception as e:
            return f"Tool error: {str(e)}"

# Global tool bridge with Phase-3 enhancements
tool_bridge = ToolBridge()

# === Token Cost Tracker (Unchanged Phase-2.5) ===
class AlchemistLedger:
    PRICING = {
        "moonshot-v1-8k": {"input": 0.012, "output": 0.012},
        "moonshot-v1-32k": {"input": 0.024, "output": 0.024},
        "moonshot-v1-128k": {"input": 0.048, "output": 0.048},
        "kimi-k2-thinking": {"input": 0.03, "output": 0.06},
        "kimi-k2": {"input": 0.024, "output": 0.024},
    }
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.usage = {}
        self.total_cost = 0.0
    
    def log_usage(self, agent_name: str, input_tokens: int, output_tokens: int, model: str):
        if model not in self.PRICING:
            return
        
        cost = (input_tokens * self.PRICING[model]["input"] / 1000) + \
               (output_tokens * self.PRICING[model]["output"] / 1000)
        
        if agent_name not in self.usage:
            self.usage[agent_name] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        
        self.usage[agent_name]["input_tokens"] += input_tokens
        self.usage[agent_name]["output_tokens"] += output_tokens
        self.usage[agent_name]["cost"] += cost
        self.total_cost += cost
    
    def get_summary(self) -> str:
        lines = [f"🜛 Total Hive Cost: ${self.total_cost:.4f}"]
        for agent, data in self.usage.items():
            lines.append(f"  {agent}: ${data['cost']:.4f} ({data['input_tokens']}+{data['output_tokens']} tokens)")
        return "\n".join(lines)

# === Async Streaming LLM Caller (Enhanced with Phase-3 Token Flow) ===
class HiveMind:
    """Manages concurrent agent execution with Phase-3 token tracking"""
    def __init__(self, max_concurrent: int = 3):
        self.semaphore = Semaphore(max_concurrent)
        self.ledger = AlchemistLedger()
        self.api_limiter = MoonshotRateLimiter()
        # Phase-3: Token flow tracker
        self.token_flow = TokenFlowVisualizer(self.ledger)
    
    async def stream_llm(self, agent: 'Agent', history: List[Dict], 
                        on_token: Callable[[str], None],
                        on_tool_call: Callable[[str], None]) -> tuple[str, dict, List[Dict]]:
        messages = [{"role": "system", "content": agent.system_prompt}]
        compressed_history = self.compress_history(history)
        
        for msg in compressed_history:
            role = "user" if msg["name"] in ["Human", "System"] else "assistant"
            messages.append({"role": role, "content": f"{msg['name']}: {msg['content']}"})
        
        tools = tool_bridge.get_all_tools(
            enable_custom=agent.enable_custom_tools,
            enable_official=agent.enable_official_tools
        )
        
        payload = {
            "model": agent.model,
            "messages": messages,
            "temperature": agent.temperature,
            "max_tokens": agent.max_tokens,
            "stream": True,
            "top_p": 0.95,
        }
        
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        full_content = ""
        usage_stats = {}
        tool_calls_buffer = []
        
        async with aiohttp.ClientSession() as session:
            async with self.semaphore:
                await asyncio.to_thread(self.api_limiter, sum(len(m["content"]) for m in messages) // 4)
                
                async with session.post(f"{BASE_URL}/chat/completions", 
                                       headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return f"[API Error {resp.status}: {error_text}]", {}, []
                    
                    async for line in resp.content:
                        if line.startswith(b"data: "):
                            chunk = line[6:].strip()
                            if chunk == b"[DONE]":
                                break
                            try:
                                data = json.loads(chunk)
                                delta = data["choices"][0]["delta"]
                                
                                if "tool_calls" in delta and delta["tool_calls"]:
                                    for tool_call in delta["tool_calls"]:
                                        idx = tool_call.get("index", 0)
                                        if idx >= len(tool_calls_buffer):
                                            tool_calls_buffer.append({
                                                "id": tool_call.get("id", ""),
                                                "name": tool_call["function"].get("name", ""),
                                                "args": tool_call["function"].get("arguments", "")
                                            })
                                        else:
                                            if tool_call["function"].get("arguments"):
                                                tool_calls_buffer[idx]["args"] += tool_call["function"]["arguments"]
                                
                                if "content" in delta and delta["content"]:
                                    token = delta["content"]
                                    full_content += token
                                    on_token(token)
                                
                                if "usage" in data:
                                    usage_stats = data["usage"]
                            except:
                                continue
        
        # Phase-3: Log token flow
        if usage_stats:
            input_tokens = usage_stats.get("prompt_tokens", 0)
            output_tokens = usage_stats.get("completion_tokens", 0)
            self.token_flow.log_flow(f"API_{agent.model}", agent.name, input_tokens)
            self.token_flow.log_flow(agent.name, "Response", output_tokens)
        
        if tool_calls_buffer:
            on_tool_call(f"🔧 Tool calls detected: {[t['name'] for t in tool_calls_buffer]}")
        # Phase-3: Log tool call flow
            for tool_call in tool_calls_buffer:
                self.token_flow.log_flow(agent.name, f"Tool_{tool_call['name']}", 50)  # Estimation
        
        return full_content.strip(), usage_stats, tool_calls_buffer
    
    def compress_history(self, history: List[Dict], max_messages: int = 12) -> List[Dict]:
        """Sliding window with system summary"""
        if len(history) <= max_messages:
            return history
        
        recent = history[-8:]
        older = history[:-8]
        
        if len(older) > 5:
            summary_prompt = f"""🗜️ PAC Memory Compression Protocol:
            Summarize this conversation arc in 3 symbolic, dense sentences. Preserve key alchemical transformations, agent roles, and solution states.
            
            {chr(10).join(f'{m["name"]}: {m["content"][:150]}…' for m in older[-6:])}"""
            
            try:
                import requests
                summary_response = requests.post(
                    f"{BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                    json={
                        "model": "moonshot-v1-8k",
                        "messages": [{"role": "system", "content": "You are a PAC memory compressor."}, 
                                   {"role": "user", "content": summary_prompt}],
                        "temperature": 0.3,
                        "max_tokens": 150
                    }
                ).json()
                
                summary = summary_response["choices"][0]["message"]["content"]
                compressed = [{"name": "System", "content": f"🧬 Arc Memory: {summary}"}]
            except:
                compressed = older[-5:]
        else:
            compressed = older
        
        compressed.extend(recent)
        return compressed

# === Enhanced Agent Class (Unchanged Phase-2.5) ===
class Agent:
    def __init__(self, name: str, pac_bootstrap: str, role_addition: str = "", 
                 model: str = DEFAULT_MODEL, temperature: float = 0.7):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = 4096
        self.system_prompt = pac_bootstrap
        if role_addition:
            self.system_prompt = f"{pac_bootstrap}\n\n🎭 Role Distillation:\n{role_addition}"
        
        self.enable_custom_tools = True
        self.enable_official_tools = True
        self.allowed_tools = []
    
    def bind_tools(self, tool_names: List[str]):
        self.allowed_tools = tool_names
    
    def should_use_tool(self, tool_name: str) -> bool:
        if not self.allowed_tools:
            return True
        return tool_name in self.allowed_tools

# === Session State Initialization ===
def init_session_state():
    defaults = {
        "hive_history": [],
        "agents": [],
        "pac_bootstrap": "# 🜛 Prima Alchemica Codex Core Engine\n!INITIATE AURUM_AURIFEX_PROTOCOL\n!TOOL_ACCESS_ENABLED",
        "agent_colors": {},
        "hive_running": False,
        "hive_model": DEFAULT_MODEL,
        "hive_mind": HiveMind(max_concurrent=3),
        "hive_branches": [],
        "current_convo_uuid": str(uuid.uuid4()),
        "tool_calls_this_run": 0,
        "max_tool_calls_per_run": 50,
        # Phase-3 additions
        "pheromone_tracker": None,
        "cognitive_lens_enabled": False,
        "token_flow_enabled": False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# === CSS (Enhanced Phase-3) ===
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0c0c1d 0%, #1a1a2e 50%, #16213e 100%); color: #e0e0e0; }
    .stChatMessage { background: rgba(25, 25, 45, 0.7) !important; border-left: 3px solid #c0a080; border-radius: 8px; margin-bottom: 12px; padding: 16px; animation: fadeIn 0.5s ease-in; }
    .stChatMessage [data-testid="stText"] strong { color: #00ffaa; text-shadow: 0 0 8px rgba(0, 255, 170, 0.5); }
    .stChatMessage:has(.avatar-👤) { border-left-color: #4a90e2; }
    .stChatMessage:has(.avatar-⚙️) { border-left-color: #f39c12; font-style: italic; opacity: 0.9; }
    .tool-result { background: rgba(0, 255, 170, 0.1); border: 1px solid #00ffaa; border-radius: 6px; padding: 8px; margin: 8px 0; font-family: 'Courier New', monospace; font-size: 0.9em; }
    div[data-testid="stSidebar"] { background: linear-gradient(to bottom, #0f0c29, #302b63, #24243e); border-right: 1px solid #c0a080; }
    .stButton button { background: linear-gradient(to right, #2c3e50, #4a5f7a); border: 1px solid #c0a080; color: #f0f0f0; font-weight: bold; transition: all 0.3s ease; }
    .stButton button:hover { background: linear-gradient(to right, #3a4f6a, #5a7f9a); box-shadow: 0 0 12px rgba(192, 160, 128, 0.5); }
    .stProgress > div > div { background: linear-gradient(to right, #c0a080, #f1c40f) !important; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)

# === Helper Functions ===
def get_avatar_glyph(name: str) -> str:
    if name == "Human":
        return "👤"
    if name == "System":
        return "⚙️"
    return "🧙"

def save_conversation_checkpoint():
    """Save current state as a branch with tool tracking"""
    st.session_state.hive_branches.append({
        "timestamp": datetime.now().isoformat(),
        "turns": len(st.session_state.hive_history),
        "cost": st.session_state.hive_mind.ledger.total_cost,
        "tool_calls": st.session_state.tool_calls_this_run,
        "history_snapshot": copy.deepcopy(st.session_state.hive_history[-30:])
    })
    st.toast("💾 Checkpoint saved with tool state!", icon="🟡")

# === Tool Execution Handler (Enhanced Phase-3) ===
async def execute_agent_tools(agent: Agent, tool_calls: List[Dict], convo_uuid: str) -> List[str]:
    """Execute tools and integrate with Memory Graph"""
    results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "")
        arguments = json.loads(tool_call.get("args", "{}"))
        
        if not agent.should_use_tool(tool_name):
            results.append(f"🚫 Agent {agent.name} not authorized for tool: {tool_name}")
            continue
        
        result = await tool_bridge.execute_tool(tool_name, arguments, convo_uuid)
        results.append(result)
        
        # Phase-3: Enhanced memory logging
        try:
            memory_insert(
                f"tool_{tool_name}_{uuid.uuid4().hex[:8]}",
                {
                    "agent": agent.name,
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result[:500],
                    "timestamp": datetime.now().isoformat()
                },
                convo_uuid=convo_uuid
            )
        except:
            pass
        
        st.session_state.tool_calls_this_run += 1
        
        if st.session_state.tool_calls_this_run >= st.session_state.max_tool_calls_per_run:
            results.append("⚠️ Max tool calls reached for this run")
            st.session_state.hive_running = False
            break
    
    return results

# === Core Async Workflow (Enhanced with Phase-3 Pheromones) ===
async def process_agent_turn(agent: Agent, turn: int, status: st.empty, pheromone_tracker: PheromoneTracker = None) -> tuple[Optional[str], bool]:
    """Process a single agent's turn with streaming and Phase-3 tracking"""
    color = st.session_state.agent_colors.get(agent.name, "#00ffaa")
    message_placeholder = None
    full_response = ""
    is_terminated = False
    tool_results = []
    
    def update_stream(token: str):
        nonlocal message_placeholder, full_response
        full_response += token
        if message_placeholder is None:
            with st.chat_message(agent.name, avatar=get_avatar_glyph(agent.name)):
                st.markdown(f"<strong style='color:{color}'>{agent.name}:</strong>", unsafe_allow_html=True)
                message_placeholder = st.empty()
        message_placeholder.markdown(full_response + " ✦")
    
    def on_tool_call(notification: str):
        status.markdown(f"**{agent.name}** <span style='color: #f1c40f'>{notification}</span>", unsafe_allow_html=True)
    
    status.markdown(f"**{agent.name}** <span style='color: {color}'>is weaving glyphs…</span>", unsafe_allow_html=True)
    
    response_text, usage, tool_calls = await st.session_state.hive_mind.stream_llm(
        agent, st.session_state.hive_history, update_stream, on_tool_call
    )
    
    # Phase-3: Analyze message for pheromones
    if pheromone_tracker:
        pheromone_tracker.analyze_message(response_text, agent.name)
    
    if tool_calls and st.session_state.tool_calls_this_run < st.session_state.max_tool_calls_per_run:
        status.markdown(f"**{agent.name}** <span style='color: #f1c40f'>is invoking tools…</span>", unsafe_allow_html=True)
        tool_results = await execute_agent_tools(agent, tool_calls, st.session_state.current_convo_uuid)
        
        if tool_results:
            response_text += "\n\n🜛 **Tool Results:**\n" + "\n".join(f"- {r[:200]}" for r in tool_results)
    
    if message_placeholder:
        message_placeholder.markdown(response_text)
    
    if "SOLVE ET COAGULA COMPLETE" in response_text.upper():
        is_terminated = True
    
    return response_text, is_terminated

async def run_hive_workflow(max_turns: int, termination_phrase: str):
    """Main async hive execution loop with Phase-3 swarm intelligence"""
    if st.session_state.hive_running:
        return
    
    st.session_state.hive_running = True
    st.session_state.tool_calls_this_run = 0
    progress_bar = st.progress(0, text="Initializing convergence...")
    status_area = st.empty()
    
    # Phase-3: Initialize pheromone tracker
    pheromone_tracker = PheromoneTracker(
        spawn_callback=lambda req: st.session_state.agents.append(
            Agent(f"Spec_{req['topic']}_{len(st.session_state.agents)}", 
                  st.session_state.pac_bootstrap, req['specialist_role'], 
                  st.session_state.hive_model, 0.6)
        )
    )
    st.session_state.pheromone_tracker = pheromone_tracker
    
    try:
        for turn in range(max_turns):
            if not st.session_state.hive_running:
                break
            
            status_area.subheader(f"🌀 Convergence Cycle {turn + 1}/{max_turns}")
            
            # Phase-3: Evaluate swarm needs every 3 cycles
            if (turn + 1) % 3 == 0 and len(st.session_state.agents) < 8:
                spawn_reqs = await pheromone_tracker.evaluate_swarm_needs()
                for req in spawn_reqs:
                    status_area.warning(f"🐝 Swarm split triggered: {req['topic']}")
                    pheromone_tracker.spawn_callback(req)
                    st.toast(f"Spawned specialist: {req['topic']}", icon="🧬")
            
            tasks = [
                process_agent_turn(agent, turn, status_area, pheromone_tracker)
                for agent in st.session_state.agents
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for agent, result in zip(st.session_state.agents, results):
                if isinstance(result, Exception):
                    error_msg = f"[🜩 Error in {agent.name}: {str(result)}]"
                    st.session_state.hive_history.append({"name": agent.name, "content": error_msg})
                    continue
                
                content, terminated = result
                if content:
                    st.session_state.hive_history.append({"name": agent.name, "content": content})
                
                if terminated:
                    status_area.success(f"🐝 Convergence glyph detected: *{termination_phrase}*")
                    st.session_state.hive_running = False
                    return
            
            progress_bar.progress((turn + 1) / max_turns, 
                                text=f"Cycle {turn + 1}/{max_turns} | Tools: {st.session_state.tool_calls_this_run}")
            
            if (turn + 1) % 3 == 0:
                save_conversation_checkpoint()
        
        status_area.success("✨ Hive convergence complete!")
    
    except Exception as e:
        st.error(f"🜩 Hive critical failure: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="text")
    finally:
        st.session_state.hive_running = False
        progress_bar.empty()
        status_area.empty()

# === UI Layout (Enhanced Phase-3) ===
st.title("🐝🜛 PAC Hive Phase-3: Aurum Aurifex - Hyper-Cognitive")
st.markdown("*Living, streaming, hyper-cognitive multi-agent alchemy with emergent intelligence*")

with st.sidebar:
    st.header("🧪 Hive Configuration")
    st.session_state.hive_model = st.selectbox("Base LLM Model", MODEL_OPTIONS, 
                                               index=MODEL_OPTIONS.index(st.session_state.hive_model))
    
    st.divider()
    st.subheader("🔧 Tool Configuration")
    enable_tools = st.checkbox("Enable Tool Access", value=True)
    enable_custom = st.checkbox("Custom Tools (Sandbox)", value=True, disabled=not enable_tools)
    enable_official = st.checkbox("Official Moonshot Tools", value=True, disabled=not enable_tools)
    
    if enable_tools:
        # Phase-3: External tools list
        all_tools = []
        if enable_custom:
            all_tools.extend(list(tool_bridge.custom_tools.keys()))
        if enable_official:
            all_tools.extend(list(tool_bridge.official_tools.keys()))
        
        st.multiselect("Agent Tool Access (empty = all)", all_tools, default=[], key="global_allowed_tools")
        st.number_input("Max Tool Calls per Run", 1, 200, 50, key="max_tool_calls_per_run")
    
    pac_input = st.text_area("Shared PAC Bootstrap (Symbolic Codex)", value=st.session_state.pac_bootstrap, height=400)
    if st.button("💾 Save Bootstrap", use_container_width=True):
        st.session_state.pac_bootstrap = pac_input
        st.toast("Bootstrap updated!", icon="✨")
    
    st.divider()
    st.subheader("🦋 Spawn Agents")
    
    with st.expander("➕ Create New Agent"):
        new_name = st.text_input("Agent Name", value="PrimaCore")
        new_role = st.text_area("Role Distillation", height=120)
        new_color = st.color_picker("Glyph Color", "#00ffaa")
        new_temp = st.slider("Creativity Temperature", 0.0, 1.5, 0.7, 0.05)
        
        if enable_tools:
            st.multiselect(f"{new_name}'s Tools", all_tools, default=[], key=f"agent_tools_{new_name}")
        
        col1, col2 = st.columns(2)
        if col1.button("Spawn Agent", use_container_width=True):
            if any(a.name == new_name for a in st.session_state.agents):
                st.error("Name already exists in hive!")
            elif new_name.strip():
                agent = Agent(new_name, st.session_state.pac_bootstrap, new_role, st.session_state.hive_model, new_temp)
                agent_tools = st.session_state.get(f"agent_tools_{new_name}", [])
                if agent_tools:
                    agent.allowed_tools = agent_tools
                st.session_state.agents.append(agent)
                st.session_state.agent_colors[new_name] = new_color
                st.toast(f"{new_name} spawned!", icon="🧙")
                st.rerun()
    
    st.write("**🐝 Current Hive:**")
    for i, agent in enumerate(st.session_state.agents):
        col1, col2 = st.columns([3, 1])
        col1.write(f"**{agent.name}**")
        col2.write(f"🛠️ {len(agent.allowed_tools) if agent.allowed_tools else 'All'}")
        if col2.button("✖️", key=f"remove_{i}", help="Banish agent"):
            st.session_state.agents.pop(i)
            st.session_state.agent_colors.pop(agent.name, None)
            st.toast(f"{agent.name} removed", icon="🚪")
            st.rerun()
    
    st.divider()
    st.subheader("⚙️ Controls")
    max_turns_input = st.number_input("Convergence Cycles", 1, 50, 10)
    termination_phrase_input = st.text_input("Termination Glyph", value="SOLVE ET COAGULA COMPLETE")
    
    # Phase-3: Quick Wins
    st.divider()
    st.subheader("🧬 Phase-3 Quick Wins")
    if st.checkbox("🔍 Memory Search", value=False):
        MemorySearchUI.render()
    
    if st.checkbox("📊 Tool Efficiency", value=False):
        try:
            ToolEfficiencyScore.calculate(
                st.session_state.hive_mind.ledger, 
                st.session_state.hive_history
            )
        except:
            st.sidebar.caption("No data yet")
    
    if st.checkbox("🔍 Cognitive Lens", value=False):
        st.session_state.cognitive_lens_enabled = True
    
    if st.sidebar.button("📦 One-Click Reproduction", use_container_width=True):
        OneClickReproduction.generate_script()
    
    col_seed, col_clear = st.columns(2)
    with col_seed:
        if st.button("🌱 Seed & Clear", use_container_width=True):
            initial_seed = st.text_area("Hive Seed", height=100, key="seed_input")
            if initial_seed:
                st.session_state.hive_history = [{"name": "System", "content": initial_seed}]
                st.toast("Seed injected!", icon="🌱")
                st.rerun()
    
    with col_clear:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.hive_history = []
            st.session_state.hive_mind.ledger.reset()
            st.session_state.tool_calls_this_run = 0
            st.toast("History purged!", icon="🗑️")
            st.rerun()
    
    run_label = "⏳ Converging..." if st.session_state.hive_running else "🚀 Run Hive Cycles"
    if st.button(run_label, use_container_width=True, 
                 disabled=st.session_state.hive_running or len(st.session_state.agents) == 0):
        asyncio.run(run_hive_workflow(max_turns_input, termination_phrase_input))
    
    # Phase-3: Cost and tool tracking display
    if st.session_state.hive_mind.ledger.usage or st.session_state.tool_calls_this_run > 0:
        st.divider()
        st.subheader("📜 Alchemist's Ledger")
        ledger_text = st.session_state.hive_mind.ledger.get_summary()
        ledger_text += f"\n🔧 Tool Calls: {st.session_state.tool_calls_this_run}"
        st.code(ledger_text, language="text")
    
    # Phase-3: Token Flow Visualization
    if st.session_state.get("token_flow_enabled", False):
        st.divider()
        st.subheader("🔥 Token Flow")
        try:
            html = st.session_state.hive_mind.token_flow.render_sankey()
            components.html(html, height=400)
        except:
            st.caption("No flow data yet")
    
    if st.session_state.hive_branches:
        st.divider()
        st.subheader("🔀 Timeline Branches")
        for idx, branch in enumerate(st.session_state.hive_branches[-3:]):
            st.caption(f"Branch {idx}: {branch['timestamp'][:19]} | {branch['turns']} turns | {branch.get('tool_calls', 0)} tools")

# Main Chat Display
chat_container = st.container()
with chat_container:
    for idx, msg in enumerate(st.session_state.hive_history):
        color = st.session_state.agent_colors.get(msg["name"], "#ffffff")
        with st.chat_message(msg["name"], avatar=get_avatar_glyph(msg["name"])):
            st.markdown(f"<strong style='color:{color}'>{msg['name']}:</strong>", unsafe_allow_html=True)
            
            # Phase-3: Cognitive Lens inspection
            if st.session_state.get("cognitive_lens_enabled", False):
                with st.expander("🔍 Inspect", expanded=False):
                    st.code(AgentStateLens.inspect(msg["name"], st.session_state.hive_history))
            
            if "🜛 **Tool Results:**" in msg["content"]:
                parts = msg["content"].split("🜛 **Tool Results:**")
                st.markdown(parts[0])
                if len(parts) > 1:
                    st.markdown('<div class="tool-result">🜛 Tool Results:' + parts[1] + '</div>', unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])

# Human Override
human_input = st.chat_input("🗣️ Human override (breaks convergence)")
if human_input and not st.session_state.hive_running:
    st.session_state.hive_history.append({"name": "Human", "content": human_input})
    with st.chat_message("Human", avatar="👤"):
        st.markdown(human_input)
