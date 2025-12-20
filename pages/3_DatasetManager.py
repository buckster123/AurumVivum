import streamlit as st
import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

st.set_page_config(page_title="Dataset Manager", layout="centered")
st.title("🗂️ Vector Extensions Hub")
st.markdown("""
Manage your static "memory extension" datasets.  
List, preview, query, stats, delete—standalone utility.
""")

base_path = Path("./vector_extensions")
if not base_path.exists():
    st.info("No datasets yet—create some with the Tiny Vector Creator!")
    st.stop()

# Scan for timestamped/subfolders
datasets = []
for item in base_path.iterdir():
    if item.is_dir():
        client = chromadb.PersistentClient(path=str(item))
        collections = client.list_collections()
        for col in collections:
            datasets.append({"folder": item.name, "collection": col.name, "path": item})

if not datasets:
    st.info("Empty folders found—create datasets first.")
    st.stop()

# Dataset selector
selected = st.selectbox(
    "Select Dataset",
    options=range(len(datasets)),
    format_func=lambda i: f"{datasets[i]['folder']} / {datasets[i]['collection']} ({datasets[i]['path']})"
)

ds = datasets[selected]
st.subheader(f"Selected: {ds['folder']} / {ds['collection']}")

# Stats
client = chromadb.PersistentClient(path=str(ds["path"]))
col = client.get_collection(ds["collection"])
count = col.count()
st.info(f"Items/Chunks: {count}")

# Delete button
if st.button("🗑️ Delete This Dataset", type="secondary"):
    confirm = st.checkbox("Confirm delete (irreversible)")
    if confirm:
        client.delete_collection(ds["collection"])
        if len(client.list_collections()) == 0:
            import shutil
            shutil.rmtree(ds["path"])
        st.success("Deleted!")
        st.rerun()

# Preview query
st.subheader("Quick Semantic Preview")
query = st.text_input("Test query")
top_k = st.slider("Results", 1, 20, 5)

if query and st.button("Search"):
    # Load embedding func from collection (or fallback)
    try:
        embed_func = col._embedding_function
    except:
        model_name = "all-MiniLM-L6-v2"  # Fallback
        embed_func = SentenceTransformerEmbeddingFunction(model_name=model_name)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight fallback
    query_emb = model.encode(query).tolist() if embed_func is None else None
    
    results = col.query(
        query_texts=[query] if embed_func else None,
        query_embeddings=[query_emb] if embed_func is None else None,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]
        dist = results["distances"][0][i]
        with st.expander(f"Result {i+1} (dist: {dist:.3f}) – {meta.get('source','?')} {meta.get('page','')}".strip()):
            if meta.get("type") == "image" and meta.get("image_base64"):
                import base64
                from io import BytesIO
                from PIL import Image
                img_data = base64.b64decode(meta["image_base64"])
                img = Image.open(BytesIO(img_data))
                st.image(img, caption=f"Image from {meta['source']} p{meta.get('page','?')}")
            else:
                st.text(doc or "(empty chunk)")
            st.json(meta)
