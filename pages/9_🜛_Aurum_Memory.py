import streamlit as st
from main import state, memory_query, memory_insert

st.title("🧠 Memory Curator")

# Search
query = st.text_input("Search memory")
if query:
    results = memory_query(query=query, top_k=10)
    st.json(results)

# View by conversation
st.subheader("Browse by Conversation")
with state.conn:
    state.cursor.execute("SELECT DISTINCT uuid FROM memory LIMIT 20")
    convos = [row[0] for row in state.cursor.fetchall()]
    
selected_convo = st.selectbox("Conversation", convos)
if selected_convo:
    results = memory_query(convo_uuid=selected_convo, limit=50)
    st.json(results)

# Manual insert (for bootstrapping)
st.subheader("Manual Memory Insert")
key = st.text_input("Key")
value = st.text_area("Value (JSON)")
if st.button("Insert"):
    memory_insert(key, json.loads(value))
    st.success("Inserted")
