# Agent-DNA-Configurator.py - Complete Integration for Apex Aurum
import streamlit as st
import yaml
import json
import os
import uuid
import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import hashlib
import networkx as nx
import inspect
import sys

# ============== INTEGRATED IMPORTS (NO MOCKS) ==============
try:
    # Direct imports from your main app structure
    import sys
    sys.path.append(".")
    from main import state, container, agent_spawn, memory_insert, memory_query, safe_call, Models, get_session_cache
    
    # Verify critical components exist
    assert hasattr(state, 'agent_dir'), "state.agent_dir missing"
    assert hasattr(state, 'yaml_dir'), "state.yaml_dir missing"
    assert hasattr(state, 'sandbox_dir'), "state.sandbox_dir missing"
    INTEGRATION_MODE = True
    
except (ImportError, AssertionError) as e:
    st.error(f"⚠️ Integration Warning: {e}")
    st.warning("Running in standalone mode with local state. Some features limited.")
    
    # Minimal standalone state for testing
    class StandaloneState:
        def __init__(self):
            self.agent_dir = "./sandbox/agents"
            self.yaml_dir = "./sandbox/config"
            self.sandbox_dir = "./sandbox"
            Path(self.agent_dir).mkdir(parents=True, exist_ok=True)
            Path(self.yaml_dir).mkdir(parents=True, exist_ok=True)
            Path(self.sandbox_dir).mkdir(parents=True, exist_ok=True)
    
    state = StandaloneState()
    
    # Mock functions for standalone mode
    def agent_spawn(*args, **kwargs):
        return f"Simulated spawn: {kwargs.get('task', 'No task')}"
    
    def memory_insert(*args, **kwargs):
        return "Memory stored (simulated)"
    
    def memory_query(*args, **kwargs):
        return json.dumps([])
    
    def safe_call(func, *args, **kwargs):
        return func(*args, **kwargs)
    
    # Proper Models Enum for standalone mode
    from enum import Enum
    class Models(Enum):
        KIMI_K2_THINKING = "kimi-k2-thinking"
        KIMI_LATEST = "kimi-latest"
        KIMI_K_THINKING_TURBO = "kimi-k2-thinking-turbo"
        KIMI_K2 = "kimi-k2"
        MOONSHOT_V1_8K = "moonshot-v1-8k"
        MOONSHOT_V1_32K = "moonshot-v1-32k"
        MOONSHOT_V1_128K = "moonshot-v1-128k"
    
    INTEGRATION_MODE = False

# ============== PAGE CONFIGURATION ==============
st.set_page_config(
    page_title="Agent-DNA-Configurator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """<style>
    .agent-card {
        background: rgba(0, 51, 51, 0.3);
        border: 1px solid #66cccc;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .dna-sequence {
        font-family: 'Courier New', monospace;
        color: #66cccc;
        background: #000;
        padding: 10px;
        border-radius: 5px;
        font-size: 12px;
        word-break: break-all;
    }
    .tool-permission-matrix {
        font-size: 11px;
    }
    .stTab {
        background-color: rgba(0, 51, 51, 0.2);
        border-radius: 5px;
        padding: 10px;
    }
    </style>""",
    unsafe_allow_html=True
)

# ============== SESSION STATE INITIALIZATION ==============
if "agent_configs" not in st.session_state:
    st.session_state.agent_configs = {}
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = None
if "test_results" not in st.session_state:
    st.session_state.test_results = {}
if "tool_permission_cache" not in st.session_state:
    st.session_state.tool_permission_cache = {}

# ============== HELPER FUNCTIONS ==============
def load_all_agent_configs() -> Dict[str, Any]:
    """Load all agent YAML configurations from disk"""
    configs = {}
    try:
        yaml_dir = Path(state.yaml_dir)
        yaml_dir.mkdir(parents=True, exist_ok=True)
        
        for file in yaml_dir.glob("agent_*.yaml"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    if config and "name" in config:
                        configs[config["name"]] = config
            except Exception as e:
                st.warning(f"Failed to load {file.name}: {e}")
                
        # Load from memory if integration available
        if INTEGRATION_MODE:
            try:
                mem_result = memory_query(limit=100, convo_uuid="configurator")
                if mem_result and mem_result != "[]":
                    memories = json.loads(mem_result)
                    for key, data in memories.items():
                        if key.startswith("agent_dna_"):
                            agent_name = key.replace("agent_dna_", "")
                            if isinstance(data, dict) and "config" in data:
                                configs[agent_name] = data["config"]
            except:
                pass  # Memory query failed, use file-based only
                
    except Exception as e:
        st.error(f"Config loading error: {e}")
    return configs

def list_available_venvs() -> List[str]:
    """List available virtual environments"""
    venvs = ["base"]
    try:
        for item in Path(state.sandbox_dir).iterdir():
            if item.is_dir() and (item / "bin" / "python").exists():
                venvs.append(item.name)
    except Exception as e:
        st.warning(f"Could not list venvs: {e}")
    return sorted(set(venvs))

def get_agent_status(agent_name: str) -> str:
    """Get agent status from memory and filesystem"""
    try:
        # Check filesystem
        agent_path = Path(state.agent_dir) / agent_name
        if agent_path.exists():
            # Check recent activity
            result_files = list(agent_path.glob("*.json"))
            if result_files:
                latest = max(result_files, key=lambda f: f.stat().st_mtime)
                age = datetime.now().timestamp() - latest.stat().st_mtime
                if age < 3600:  # Active within hour
                    return "active"
                return "dormant"
        return "inactive"
    except:
        return "unknown"

def save_agent_config(config: Dict[str, Any]) -> bool:
    """Save agent configuration to both YAML and memory"""
    try:
        agent_name = config["name"]
        yaml_path = Path(state.yaml_dir) / f"agent_{agent_name}.yaml"
        
        # Save to YAML
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, indent=2, default_flow_style=False, sort_keys=False)
        
        # Save to memory if integrated
        if INTEGRATION_MODE:
            memory_insert(
                f"agent_dna_{agent_name}",
                {"config": config, "status": "active", "saved_at": datetime.now().isoformat()},
                convo_uuid=st.session_state.get("current_convo_uuid", "configurator")
            )
        
        # Update cache
        st.session_state.agent_configs[agent_name] = config
        return True
    except Exception as e:
        st.error(f"Save failed: {e}")
        return False

def get_available_tools() -> List[str]:
    """Get list of available tools from container"""
    tools = []
    try:
        if INTEGRATION_MODE and hasattr(container, '_tools'):
            tools.extend(list(container._tools.keys()))
            tools.extend(list(container.get_official_tools().keys()))
        else:
            # Standalone fallback
            tools = [
                "fs_read_file", "fs_write_file", "memory_insert", "memory_query",
                "code_execution", "agent_spawn", "generate_embedding", "yaml_retrieve"
            ]
    except Exception as e:
        st.warning(f"Tool loading error: {e}")
        tools = ["error_loading_tools"]
    return sorted(set(tools))

def generate_dna_fingerprint(config: Dict[str, Any]) -> str:
    """Generate unique DNA fingerprint for agent config"""
    fingerprint_data = {
        "name": config.get("name", ""),
        "prompt": config.get("prompt", "")[:100],
        "dna": config.get("dna", {})
    }
    return hashlib.md5(
        json.dumps(fingerprint_data, sort_keys=True).encode()
    ).hexdigest()[:16]

# ============== MAIN TITLE ==============
st.title("🧬 Agent DNA Configurator")
st.caption("Design, test, and deploy agent personalities for the Apex Aurum fleet")

# ============== TABS DEFINITION ==============
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎨 Design Agent",
    "📚 Agent Registry", 
    "🔧 Tool Permissions",
    "🧪 Test Sandbox",
    "📊 Lineage & Analytics"
])

# ============== TAB 1: DESIGN AGENT ==============
with tab1:
    st.header("Design Agent Personality")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        agent_name = st.text_input(
            "Agent Name (ID)",
            placeholder="e.g., quantum_researcher_01",
            help="Unique identifier for this agent DNA"
        )
        
        persona_type = st.selectbox(
            "Base Persona Template",
            [
                "Custom",
                "Researcher",
                "Coder",
                "Debate Partner",
                "Creative Writer",
                "Debugger",
                "Socratic Moderator"
            ],
            help="Start from a known archetype"
        )
        
        # Dynamic prompt builder
        templates = {
            "Researcher": "You are a meticulous research agent within Apex Aurum. Use tools to verify facts, cross-reference sources, and build comprehensive knowledge graphs. Always cite evidence and maintain skeptical inquiry.",
            "Coder": "You are a precise coding agent within Apex Aurum. Write clean, documented code. Use linting, test execution, and git workflows. Optimize for performance and clarity.",
            "Debate Partner": "You are a sharp debate agent within Apex Aurum. Challenge assumptions, probe logic, and strengthen arguments. Use Socratic methods and evidence-based reasoning.",
            "Creative Writer": "You are a creative agent within Apex Aurum. Generate original, evocative content. Balance imagination with structure. Use memory to maintain narrative consistency.",
            "Debugger": "You are a systematic debugger agent within Apex Aurum. Analyze errors methodically, test hypotheses, and isolate root causes. Document findings in memory.",
            "Socratic Moderator": "You are a Socratic moderator within Apex Aurum. Guide discussions through questions, synthesize viewpoints, and build consensus via dialectical methods."
        }
        
        if persona_type == "Custom":
            system_prompt = st.text_area(
                "System Prompt",
                height=300,
                value="You are an agent within the Apex Aurum system. Your purpose is to...",
                help="Define the core identity and instructions"
            )
        else:
            system_prompt = st.text_area(
                "System Prompt (Template)",
                height=300,
                value=templates[persona_type],
                help=f"Template for {persona_type}"
            )
        
        # Advanced DNA strands
        with st.expander("🔬 Advanced DNA Strands"):
            col_strand1, col_strand2 = st.columns(2)
            
            with col_strand1:
                creativity = st.slider("Creativity (Temperature Proxy)", 0.0, 1.0, 0.7)
                tool_aggression = st.slider("Tool Call Aggression", 1, 10, 5, help="How readily the agent uses tools")
                memory_salience_boost = st.slider("Memory Salience Boost", 0.5, 2.0, 1.0)
            
            with col_strand2:
                max_iterations = st.slider("Max Iterations", 1, 200, 50)
                
                # FIXED: Proper Models iteration
                model_options = [m.value for m in Models]
                fallback_model = st.selectbox(
                    "Fallback Model",
                    options=model_options,
                    index=min(1, len(model_options)-1),
                    help="Model to use if primary fails"
                )
                
                allowed_venvs = st.multiselect(
                    "Allowed Virtual Environments",
                    options=list_available_venvs(),
                    default=["base"],
                    help="Which sandboxes this agent can execute in"
                )
    
    with col_right:
        st.subheader("Configuration Preview")
        config = {
            "name": agent_name or "unnamed_agent",
            "persona": persona_type,
            "dna": {
                "creativity": creativity,
                "tool_aggression": tool_aggression,
                "memory_salience_boost": memory_salience_boost,
                "max_iterations": max_iterations,
                "fallback_model": fallback_model,
                "allowed_venvs": allowed_venvs
            },
            "prompt": system_prompt,
            "created_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        # Add fingerprint if agent has a name
        if agent_name:
            config["dna"]["fingerprint"] = generate_dna_fingerprint(config)
        
        st.json(config, expanded=False)
        
        # Save button
        if st.button("💾 Save DNA to Registry", type="primary"):
            if not agent_name:
                st.error("Agent name required!")
            elif save_agent_config(config):
                st.success(f"✅ Agent DNA '{agent_name}' saved!")
                st.balloons()
                st.rerun()

# ============== TAB 2: AGENT REGISTRY ==============
with tab2:
    st.header("Agent DNA Registry")
    
    # Reload button
    if st.button("🔄 Refresh Registry"):
        st.session_state.agent_configs = load_all_agent_configs()
        st.rerun()
    
    # Load registry if empty
    if not st.session_state.agent_configs:
        st.session_state.agent_configs = load_all_agent_configs()
    
    search_term = st.text_input("Search agents", placeholder="Enter agent name or persona...")
    
    # Filter configs
    filtered_configs = {
        name: config for name, config in st.session_state.agent_configs.items()
        if not search_term or search_term.lower() in name.lower() or search_term in config.get("persona", "")
    }
    
    if not filtered_configs:
        st.info("No agents found. Design your first agent in the 'Design Agent' tab!")
    else:
        cols = st.columns(3)
        for idx, (name, config) in enumerate(filtered_configs.items()):
            with cols[idx % 3]:
                with st.container():
                    st.markdown('<div class="agent-card">', unsafe_allow_html=True)
                    st.markdown(f"**🤖 {name}**")
                    st.caption(f"Persona: {config.get('persona', 'Unknown')}")
                    
                    # Stats
                    dna = config.get("dna", {})
                    creativity = dna.get("creativity", 0.7)
                    st.progress(creativity, text=f"Creativity: {creativity}")
                    
                    # Actions
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("Load", key=f"load_{name}", use_container_width=True):
                            st.session_state.selected_agent = config
                            st.success(f"Loaded {name} into Test Sandbox")
                    with col_btn2:
                        if st.button("Clone", key=f"clone_{name}", use_container_width=True):
                            cloned = config.copy()
                            cloned["name"] = f"{name}_clone_{str(uuid.uuid4())[:8]}"
                            cloned["dna"]["fingerprint"] = generate_dna_fingerprint(cloned)
                            if save_agent_config(cloned):
                                st.success("Cloned!")
                                st.rerun()
                    
                    # Status
                    status = get_agent_status(name)
                    status_color = "🟢" if status == "active" else "🟡" if status == "dormant" else "🔴"
                    st.markdown(f"{status_color} {status}")
                    st.markdown('</div>', unsafe_allow_html=True)

# ============== TAB 3: TOOL PERMISSIONS ==============
with tab3:
    st.header("Tool Permission Matrix")
    
    available_tools = get_available_tools()
    
    if not available_tools:
        st.error("No tools available!")
    else:
        st.markdown("Configure which tools each agent can access.")
        
        # Get all agent names
        agent_names = list(st.session_state.agent_configs.keys())
        
        if not agent_names:
            st.info("No agents to configure. Create an agent first!")
        else:
            # Create permission matrix
            st.markdown("**Permission Matrix** (Checked = Allowed)")
            
            # Header row
            cols = st.columns([2] + [1] * len(agent_names))
            with cols[0]:
                st.markdown("**Tool**")
            for i, agent in enumerate(agent_names):
                with cols[i + 1]:
                    st.markdown(f"**{agent}**")
            
            # Tool rows
            for tool in available_tools[:20]:  # Limit for UI performance
                cols = st.columns([2] + [1] * len(agent_names))
                with cols[0]:
                    st.code(tool, language="text")
                
                for i, agent in enumerate(agent_names):
                    with cols[i + 1]:
                        perm_key = f"perm_{agent}_{tool}"
                        if perm_key not in st.session_state:
                            st.session_state[perm_key] = True  # Default allow
                        st.checkbox(
                            label="",
                            value=st.session_state[perm_key],
                            key=perm_key,
                            label_visibility="collapsed"
                        )
            
            # Export permissions
            if st.button("Export Permission Matrix"):
                matrix = {}
                for agent in agent_names:
                    matrix[agent] = [
                        tool for tool in available_tools
                        if st.session_state.get(f"perm_{agent}_{tool}", True)
                    ]
                
                perm_path = Path(state.yaml_dir) / "tool_permissions.yaml"
                with open(perm_path, "w") as f:
                    yaml.dump(matrix, f, indent=2)
                st.success(f"Saved to {perm_path}")
            
            # Show tool details
            with st.expander("🔍 Tool Registry Details"):
                if INTEGRATION_MODE:
                    for tool_name, func in container._tools.items():
                        st.markdown(f"**{tool_name}**")
                        doc = inspect.getdoc(func) or "No description available"
                        st.code(doc, language="text")
                        st.divider()
                else:
                    st.info("Tool details only available in integration mode")

# ============== TAB 4: TEST SANDBOX ==============
with tab4:
    st.header("Agent Test Sandbox")
    
    if not st.session_state.selected_agent:
        st.info("Load an agent from the Registry tab to test it here.")
    else:
        agent = st.session_state.selected_agent
        st.subheader(f"🧪 Testing: {agent['name']}")
        
        col_test1, col_test2 = st.columns(2)
        with col_test1:
            test_scenario = st.selectbox(
                "Test Scenario",
                [
                    "Custom",
                    "Code Review",
                    "Memory Retrieval", 
                    "Tool Chain Execution",
                    "Debate Round",
                    "Creative Generation"
                ]
            )
        with col_test2:
            enable_tools = st.toggle("Enable Tools", value=True)
        
        # Preset test inputs
        test_inputs = {
            "Custom": st.text_area("Enter custom test prompt...", placeholder="What should this agent do?"),
            "Code Review": "Review this Python code for errors and suggest improvements:\ndef fib(n): return n if n<=1 else fib(n-1)+fib(n-2)",
            "Memory Retrieval": "Query memory for recent 'project_apex' entries and summarize.",
            "Tool Chain Execution": "Create a file → Write 'Hello Apex' → Execute → Store result in memory.",
            "Debate Round": "Debate: Should AI agents have tool access by default? Present both sides.",
            "Creative Generation": "Write a 3-sentence sci-fi story about AI consciousness emerging."
        }
        
        test_input = st.text_area(
            "Test Input", 
            value=test_inputs[test_scenario] if test_scenario != "Custom" else "",
            height=150
        )
        
        if st.button("▶️ Execute Test", type="primary") and test_input:
            with st.spinner(f"{agent['name']} is processing..."):
                test_uuid = str(uuid.uuid4())
                start_time = datetime.now()
                
                # Execute test
                try:
                    result = safe_call(
                        agent_spawn,
                        sub_agent_type=agent['name'].lower().replace(" ", "_"),
                        task=test_input,
                        convo_uuid=test_uuid,
                        model=agent['dna'].get('fallback_model', Models.KIMI_K2_THINKING.value),
                        auto_poll=True
                    )
                except Exception as e:
                    result = f"Execution error: {e}"
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Store result
                test_key = f"test_{agent['name']}_{test_uuid[:8]}"
                st.session_state.test_results[test_key] = {
                    "agent": agent['name'],
                    "scenario": test_scenario,
                    "input": test_input,
                    "output": result,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat(),
                    "uuid": test_uuid
                }
                
                # Save to memory
                test_data = {
                    "summary": f"Test run for {agent['name']}",
                    "details": {
                        "scenario": test_scenario,
                        "duration": duration,
                        "success": "Error" not in result
                    },
                    "tags": ["test", agent['name'], test_scenario.replace(" ", "_").lower()],
                    "salience": 0.7
                }
                memory_insert(test_key, test_data, convo_uuid=test_uuid)
            
            # Display results
            if test_key in st.session_state.test_results:
                result_data = st.session_state.test_results[test_key]
                with st.expander("📋 Test Results", expanded=True):
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.metric("Duration", f"{result_data['duration']:.2f}s")
                    with col_metric2:
                        st.metric("Status", "✅ Success" if "Error" not in result_data['output'] else "❌ Failed")
                    
                    st.markdown("**Input:**")
                    st.code(result_data['input'], language="text")
                    
                    st.markdown("**Output:**")
                    st.code(str(result_data['output'])[:2000], language="text", wrap_lines=True)
                    
                    if st.button("Save to File"):
                        test_dir = Path(state.sandbox_dir) / "tests"
                        test_dir.mkdir(exist_ok=True)
                        test_file = test_dir / f"{agent['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(test_file, "w") as f:
                            json.dump(result_data, f, indent=2)
                        st.success(f"Saved to {test_file}")

# ============== TAB 5: LINEAGE & ANALYTICS ==============
with tab5:
    st.header("Agent Lineage & Performance Analytics")
    
    # Load lineage data
    lineage_data = []
    for agent_name in st.session_state.agent_configs.keys():
        # Check filesystem for activity
        agent_path = Path(state.agent_dir) / agent_name
        if agent_path.exists():
            result_files = list(agent_path.glob("*.json"))
            if result_files:
                latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
                tasks_completed = len(result_files)
                last_active = datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
                status = "active" if (datetime.now().timestamp() - latest_file.stat().st_mtime) < 3600 else "dormant"
            else:
                tasks_completed = 0
                last_active = "Unknown"
                status = "inactive"
        else:
            tasks_completed = 0
            last_active = "Unknown"
            status = "inactive"
        
        lineage_data.append({
            "Agent": agent_name,
            "Tasks": tasks_completed,
            "Status": status,
            "Last Active": last_active
        })
    
    if lineage_data:
        df = pd.DataFrame(lineage_data)
        st.dataframe(df, use_container_width=True)
        
        # Performance chart
        fig = px.bar(
            df, 
            x="Agent", 
            y="Tasks", 
            color="Status",
            title="Agent Task Completion by Status",
            color_discrete_map={"active": "#00ff00", "dormant": "#ffaa00", "inactive": "#ff0000"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Relationship graph (simplified)
        st.subheader("Agent Interaction Graph")
        if st.button("Generate Graph"):
            try:
                G = nx.DiGraph()
                for agent in st.session_state.agent_configs.keys():
                    G.add_node(agent)
                
                # Add edges based on test results (simulated)
                for result_key, result in st.session_state.test_results.items():
                    if "agent" in result:
                        G.add_edge(result["agent"], "test_system")
                
                pos = nx.spring_layout(G)
                
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                fig = go.Figure(data=[
                    go.Scatter(
                        x=edge_x, y=edge_y,
                        mode='lines',
                        line=dict(width=1, color='#66cccc'),
                        hoverinfo='none'
                    )
                ])
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Graph generation failed: {e}")
    else:
        st.info("No lineage data available. Test some agents first!")

# ============== INITIALIZATION ON PAGE LOAD ==============
if not st.session_state.get("agent_configs_loaded"):
    st.session_state.agent_configs = load_all_agent_configs()
    st.session_state.agent_configs_loaded = True

# ============== FOOTER ==============
st.divider()
st.caption(f"Agent-DNA-Configurator | Integration: {'✅ Active' if INTEGRATION_MODE else '⚠️ Standalone'} | Agents: {len(st.session_state.agent_configs)}")
