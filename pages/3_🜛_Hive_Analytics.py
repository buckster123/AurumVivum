# pages/06_hive_analytics.py
# Hive Analytics: Visualize agent behavior, costs, and convergence patterns

import streamlit as st
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# === Import Diagnostics ===
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from main import state, get_memory_cache, Models
except ImportError:
    st.error("⚠️ Could not import main script. Ensure it's in the parent directory.")
    st.stop()

# === Session State ===
if "analytics_data" not in st.session_state:
    st.session_state.analytics_data = {
        "selected_convo_uuid": None,
        "analysis_cache": {},
        "viz_type": "overview"
    }

# === Core Analytics Functions ===
def extract_hive_metrics(convo_uuid: str) -> dict:
    """Extract metrics from hive conversation memory"""
    memory_cache = get_memory_cache()
    
    tool_calls = []
    cost_data = []
    agent_activity = {}
    
    try:
        with state.conn:
            state.cursor.execute(
                "SELECT mem_key, mem_value, timestamp FROM memory WHERE uuid=? AND mem_key LIKE 'tool_%'",
                (convo_uuid,)
            )
            tool_rows = state.cursor.fetchall()
            
            for row in tool_rows:
                try:
                    value = json.loads(row[1]) if isinstance(row[1], str) else row[1]
                    tool_calls.append({
                        "tool": value.get("tool", "unknown"),
                        "agent": value.get("agent", "unknown"),
                        "timestamp": row[2],
                        "result_length": len(value.get("result", ""))
                    })
                except: pass
            
            # Get agent activity
            state.cursor.execute(
                "SELECT mem_key, mem_value, timestamp FROM memory WHERE uuid=? AND mem_key LIKE 'agent_%'",
                (convo_uuid,)
            )
            agent_rows = state.cursor.fetchall()
            
            for row in agent_rows:
                try:
                    value = json.loads(row[1]) if isinstance(row[1], str) else row[1]
                    agent_name = value.get("agent_id", "unknown").split("_")[0]
                    if agent_name not in agent_activity:
                        agent_activity[agent_name] = 0
                    agent_activity[agent_name] += 1
                except: pass
    
    except Exception as e:
        st.error(f"Memory query failed: {e}")
    
    return {
        "tool_calls": tool_calls,
        "agent_activity": agent_activity,
        "total_tools": len(tool_calls)
    }

def analyze_convergence(history: list) -> dict:
    """Detect if/when the hive got stuck or converged"""
    # FIXED: Always return all keys
    defaults = {
        "status": "insufficient_data",
        "unique_ratio": 0.0,
        "termination_score": 0,
        "loop_detected": False
    }
    
    if len(history) < 10:
        return defaults
    
    # Check for repetition patterns
    last_10 = [msg["content"][:200] for msg in history[-10:]]
    unique_ratio = len(set(last_10)) / len(last_10)
    
    # Check for termination phrase proximity
    termination_phrases = ["solve et coagula", "complete", "finished", "done"]
    termination_score = sum(any(phrase in msg["content"].lower() for phrase in termination_phrases) for msg in history[-5:])
    
    # Detect loops (repeated tool calls)
    tool_pattern = [msg for msg in history if "Tool call" in msg["content"]]
    loop_detected = len(tool_pattern) > 3 and len(set([msg["content"][:100] for msg in tool_pattern[-3:]])) == 1
    
    return {
        "status": "converged" if termination_score > 2 else "looping" if loop_detected else "exploring",
        "unique_ratio": unique_ratio,
        "termination_score": termination_score,
        "loop_detected": loop_detected
    }

def generate_leaderboard(agent_activity: dict, tool_calls: list) -> pd.DataFrame:
    """Create agent performance leaderboard"""
    tool_counts = {}
    for call in tool_calls:
        agent = call["agent"]
        tool_counts[agent] = tool_counts.get(agent, 0) + 1
    
    df_data = []
    all_agents = set(agent_activity.keys()) | set(tool_counts.keys())
    for agent in all_agents:
        df_data.append({
            "Agent": agent,
            "Messages": agent_activity.get(agent, 0),
            "Tool Calls": tool_counts.get(agent, 0),
            "Score": agent_activity.get(agent, 0) * 2 + tool_counts.get(agent, 0) * 3
        })
    
    df = pd.DataFrame(df_data).sort_values("Score", ascending=False)
    return df

# === UI ===
st.title("🐝 Hive Analytics Dashboard")
st.markdown("*Agent behavior, costs, and convergence patterns*")

# Conversation selector
st.subheader("📊 Select Hive Run")
convo_uuids = []
try:
    with state.conn:
        state.cursor.execute("SELECT DISTINCT uuid FROM memory WHERE mem_key LIKE 'tool_%' ORDER BY timestamp DESC LIMIT 20")
        rows = state.cursor.fetchall()
        convo_uuids = [row[0] for row in rows if row[0]]
except:
    pass

if not convo_uuids:
    st.info("No hive runs found in memory. Run the hive first!")
    st.stop()

selected_uuid = st.selectbox("Choose a hive run:", convo_uuids, key="convo_selector")

if selected_uuid != st.session_state.analytics_data["selected_convo_uuid"]:
    st.session_state.analytics_data["selected_convo_uuid"] = selected_uuid
    st.session_state.analytics_data["analysis_cache"] = extract_hive_metrics(selected_uuid)

metrics = st.session_state.analytics_data["analysis_cache"]

# === Visualization Tabs ===
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔧 Tool Usage", "👑 Leaderboard", "🌀 Convergence"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tool Calls", metrics["total_tools"])
    with col2:
        unique_tools = len(set([t["tool"] for t in metrics["tool_calls"]]))
        st.metric("Unique Tools", unique_tools)
    with col3:
        st.metric("Active Agents", len(metrics["agent_activity"]))
    with col4:
        avg_tools_per_agent = metrics["total_tools"] / len(metrics["agent_activity"]) if metrics["agent_activity"] else 0
        st.metric("Avg Tools/Agent", f"{avg_tools_per_agent:.1f}")
    
    # Tool distribution pie chart
    if metrics["tool_calls"]:
        tool_counts = pd.DataFrame(metrics["tool_calls"])["tool"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        tool_counts.plot.pie(ax=ax, autopct='%1.1f%%', startangle=90, cmap="viridis")
        ax.set_ylabel("")
        ax.set_title("Tool Distribution")
        st.pyplot(fig)

with tab2:
    # Heatmap: Agent vs Tool usage
    if metrics["tool_calls"]:
        pivot_data = pd.DataFrame(metrics["tool_calls"]).pivot_table(
            index="agent", columns="tool", aggfunc="size", fill_value=0
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_data, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
        ax.set_title("Agent-Tool Usage Heatmap")
        st.pyplot(fig)
    
    # Timeline of tool calls
    if metrics["tool_calls"]:
        df_tools = pd.DataFrame(metrics["tool_calls"])
        df_tools["timestamp"] = pd.to_datetime(df_tools["timestamp"])
        df_tools = df_tools.sort_values("timestamp")
        
        fig, ax = plt.subplots(figsize=(12, 4))
        for agent in df_tools["agent"].unique():
            agent_data = df_tools[df_tools["agent"] == agent]
            ax.scatter(agent_data["timestamp"], [agent] * len(agent_data), label=agent, alpha=0.7)
        
        ax.set_title("Tool Call Timeline")
        ax.legend()
        st.pyplot(fig)

with tab3:
    # Leaderboard
    df_leaderboard = generate_leaderboard(metrics["agent_activity"], metrics["tool_calls"])
    st.dataframe(df_leaderboard, use_container_width=True)
    
    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    df_leaderboard.plot(x="Agent", y="Score", kind="bar", ax=ax, color="#00ffaa", legend=False)
    ax.set_title("Agent Performance Score")
    ax.set_ylabel("Score (Messages×2 + Tools×3)")
    st.pyplot(fig)

with tab4:
    # Convergence analysis
    history = []
    try:
        with state.conn:
            state.cursor.execute(
                "SELECT mem_value FROM memory WHERE uuid=? AND mem_key LIKE 'agent_%' ORDER BY timestamp",
                (selected_uuid,)
            )
            rows = state.cursor.fetchall()
            for row in rows:
                try:
                    value = json.loads(row[0])
                    history.append({"name": value.get("agent_id", "unknown").split("_")[0], "content": value.get("response", "")})
                except: pass
    except: pass
    
    convergence = analyze_convergence(history)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Convergence Status", convergence["status"].upper())
        st.metric("Message Diversity", f"{convergence['unique_ratio']:.1%}")
    
    with col2:
        if convergence["loop_detected"]:
            st.error("🔄 LOOP DETECTED IN LAST 3 TOOL CALLS")
        if convergence["termination_score"] > 2:
            st.success("✅ TERMINATION SIGNAL DETECTED")
    
    # Show last 5 messages for manual inspection
    st.subheader("Recent Activity")
    for i, msg in enumerate(history[-5:], start=1):
        with st.expander(f"{i}. {msg['name']}", expanded=i > 2):
            st.text(msg["content"][:300] + "...")

# === Export ===
st.divider()
if st.button("📥 Export Analytics Report"):
    report = {
        "hive_run_uuid": selected_uuid,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "convergence": convergence,
        "leaderboard": df_leaderboard.to_dict()
    }
    st.download_button(
        "Download JSON Report",
        json.dumps(report, indent=2),
        file_name=f"hive_analytics_{selected_uuid}.json",
        mime="application/json"
    )
