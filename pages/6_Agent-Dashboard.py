import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="Agent Dashboard", layout="centered")
st.title("🤖 Agent Fleet Dashboard")
st.markdown("Live oversight of spawned sub-agents. Status from shared memory—nothing drops on page switch.")

# Pull from shared session_state/memory (same as main)
pending = st.session_state.get("pending_notifies", [])

if pending:
    st.subheader("Recent Notifies")
    for n in pending:
        st.info(f"**{n['agent_id']}**: {n['status']} – {n['task']}")

# Query active/running agents from memory (agent_* keys)
try:
    # Dummy query all recent (adapt from your memory_query logic)
    # Since no direct tool access here, scan agent_dir for simplicity
    agent_dir = Path("./sandbox/agents")
    active_agents = []
    if agent_dir.exists():
        for folder in agent_dir.iterdir():
            if folder.is_dir():
                result_file = folder / "result.json"
                if result_file.exists():
                    data = json.loads(result_file.read_text())
                    active_agents.append({
                        "id": folder.name,
                        "task": data.get("task", "Unknown"),
                        "status": data.get("status", "unknown"),
                        "timestamp": data.get("timestamp", "?")
                    })
except Exception as e:
    st.warning(f"Scan error: {e}")
    active_agents = []

if active_agents:
    st.subheader(f"Active/Recent Agents ({len(active_agents)})")
    for agent in active_agents:
        with st.expander(f"{agent['id']} – {agent['status'].capitalize()} ({agent['timestamp'][:10]})"):
            st.write(f"**Task:** {agent['task']}")
            st.write(f"**Status:** {agent['status']}")
            if st.button(f"Kill {agent['id']}", key=f"kill_{agent['id']}"):
                # Insert kill key (mirrors main logic)
                kill_key = f"agent_{agent['id']}_kill"
                kill_data = {"status": "killed", "timestamp": "now"}
                # Can't direct memory_insert here (no AppState), but suggest chat command
                st.info("Switch to main chat and run: memory_insert for kill key (or add tool later)")
                st.code(f"Tool call: memory_insert(key='{kill_key}', value={kill_data})")
else:
    st.info("No agents detected—spawn some in main chat!")

st.button("Refresh", on_click=st.rerun)
