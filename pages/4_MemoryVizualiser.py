import streamlit as st
import chromadb
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

st.set_page_config(page_title="Memory Lattice Viz", layout="centered")
st.title("🧠 Memory Lattice Visualizer")
st.markdown("Standalone viz for main agent memory (or extensions). Interactive graph + amps.")

# Paths from main app vibe
main_chroma_path = "./sandbox/db/chroma_db"
extensions_path = "./vector_extensions"

path_options = {"Main Agent Memory": main_chroma_path}
# Add extensions
import os
from pathlib import Path
for item in Path(extensions_path).iterdir():
    if item.is_dir():
        path_options[f"Extension: {item.name}"] = str(item)

selected_path = st.selectbox("Select DB Path", options=list(path_options.keys()))
actual_path = path_options[selected_path]

client = chromadb.PersistentClient(path=actual_path)
collections = client.list_collections()
col_name = st.selectbox("Collection", [c.name for c in collections])
col = client.get_collection(col_name)

top_k = st.slider("Top Nodes", 5, 50, 20)
sim_threshold = st.slider("Sim Threshold", 0.3, 0.9, 0.6)

if st.button("Weave Lattice"):
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight
    
    # Dummy query to pull recent/relevant
    results = col.query(query_texts=["memory overview"], n_results=top_k, include=["metadatas", "documents", "distances"])
    
    G = nx.Graph()
    summaries = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        summary = meta.get("summary", results["documents"][0][i][:100])
        salience = meta.get("salience", 1.0)
        sim = 1 - results["distances"][0][i]
        if sim < sim_threshold:
            continue
        key = meta.get("mem_key", f"node{i}")
        G.add_node(key, summary=summary, salience=salience, sim=sim)
        summaries.append(summary)
    
    if len(G.nodes) > 1:
        all_embs = [embed_model.encode(s).tolist() for s in summaries]
        import numpy as np
        for i, node_i in enumerate(G.nodes):
            for j in range(i+1, len(G.nodes)):
                node_j = list(G.nodes)[j]
                sim = np.dot(all_embs[i], all_embs[j]) / (np.linalg.norm(all_embs[i]) * np.linalg.norm(all_embs[j]) + 1e-8)
                if sim > sim_threshold:
                    G.add_edge(node_i, node_j, weight=sim)
    
    if G.nodes:
        pos = nx.spring_layout(G, k=1, iterations=20)
        edge_trace = go.Scatter(x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
        
        node_trace = go.Scatter(x=[], y=[], mode='markers+text', hoverinfo='text',
                                marker=dict(size=[G.nodes[n]["salience"]*20 for n in G.nodes],
                                            color=[G.nodes[n]["sim"] for n in G.nodes], colorscale='Viridis'))
        node_trace['text'] = [G.nodes[n]["summary"][:30] for n in G.nodes]
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
        
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(title=f"Lattice: {len(G.nodes)} Nodes", showlegend=False, hovermode='closest',
                                         xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False)))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No nodes above threshold.")
