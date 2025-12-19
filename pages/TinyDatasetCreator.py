# pages/vector_db_creator.py
import streamlit as st
import os
import tempfile
from pathlib import Path
import subprocess
from typing import List, Tuple

try:
    import ocrmypdf
    has_ocrmypdf = True
except ImportError:
    has_ocrmypdf = False

from pypdf import PdfReader
import docx2txt
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

st.set_page_config(page_title="Tiny Vector Dataset Creator", layout="centered")
st.title("🛠️ Tiny Vector Dataset Creator")
st.markdown("""
Create lightweight, persistent Chroma vector DBs from documents (PDF/TXT/MD/DOCX/HTML).  
Perfect for small "memory extensions" on your Pi 5—separate from main agent memory.

- Upload multiple files (up to ~100MB total recommended)
- Optional OCR for scanned PDFs
- Streamed extraction = low memory (great for 1500–2000 pages)
- Page-level metadata for PDFs (agent can reason → load exact pages later)
- One-click → ready-to-use static dataset
""")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    model_options = [
        "all-MiniLM-L6-v2",       # Default: fast, light, good quality
        "all-MiniLM-L3-v2",       # Tiny & fastest
        "all-mpnet-base-v2",      # Heavier (matches your main app model)
        "paraphrase-MiniLM-L3-v2"
    ]
    model_name = st.selectbox("Embedding model", model_options, index=0)
    
    chunk_size = st.slider("Chunk size (chars)", 500, 4000, 2000, help="Larger = fewer chunks = faster")
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 600, 200)
    
    db_path = st.text_input("Vector DB folder", value="./vector_extensions", help="Separate from main app DB")
    collection_name = st.text_input("Collection name", value="extension_docs")

uploaded_files = st.file_uploader(
    "Upload documents",
    accept_multiple_files=True,
    type=["pdf", "txt", "md", "markdown", "docx", "html", "htm"],
    key="uploaded_docs",
)

if uploaded_files:
    st.info(f"Uploaded {len(uploaded_files)} file(s)")

force_ocr = False
if has_ocrmypdf:
    force_ocr = st.checkbox("Force OCR on PDFs (slow—only for scanned/image-only)", value=False)
else:
    st.warning("⚠️ ocrmypdf not installed → OCR disabled (pip install ocrmypdf + system deps)")

if st.button("🚀 Create Dataset", type="primary"):
    if not uploaded_files:
        st.error("Upload files first!")
        st.stop()

    os.makedirs(db_path, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Phase 1: Save + OCR
        processed_files: List[Tuple[str, Path]] = []
        for i, up_file in enumerate(uploaded_files):
            status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {up_file.name}")
            saved = temp_path / up_file.name
            with open(saved, "wb") as f:
                f.write(up_file.getbuffer())
            
            final_path = saved
            if force_ocr and up_file.name.lower().endswith(".pdf"):
                ocr_path = temp_path / f"ocr_{up_file.name}"
                try:
                    ocrmypdf.ocr(str(saved), str(ocr_path), jobs=os.cpu_count() or 4, optimize=1)
                    final_path = ocr_path
                except Exception as e:
                    st.warning(f"OCR failed on {up_file.name}: {e} → using original")
            
            processed_files.append((up_file.name, final_path))
            progress_bar.progress(0.3 * (i + 1) / len(uploaded_files))

        # Phase 2: Streamed extraction + chunking
        status_text.text("Extracting & chunking (streamed, low RAM)...")
        documents = []
        metadatas = []
        ids = []
        global_chunk_id = 0
        overlap_buffer = ""
        
        for orig_name, file_path in processed_files:
            ext = Path(orig_name).suffix.lower()
            safe_name = Path(orig_name).stem.replace(".", "_")
            
            try:
                if ext == ".pdf":
                    reader = PdfReader(str(file_path))
                    for page_num, page in enumerate(reader.pages, start=1):
                        page_text = (page.extract_text() or "") 
                        page_text = overlap_buffer + page_text
                        
                        start = 0
                        while start < len(page_text):
                            end = start + chunk_size
                            chunk = page_text[start:end]
                            if chunk.strip():
                                documents.append(chunk)
                                metadatas.append({"source": orig_name, "page": page_num})
                                ids.append(f"{safe_name}_p{page_num}_c{global_chunk_id}")
                                global_chunk_id += 1
                            start = end - chunk_overlap
                        
                        if chunk_overlap and start < len(page_text):
                            overlap_buffer = page_text[-chunk_overlap:]
                        else:
                            overlap_buffer = ""
                
                elif ext in [".txt", ".md", ".markdown"]:
                    with open(file_path, "r", encoding="utf-8") as f:
                        buffer = ""
                        for line in f:
                            buffer += line
                            while len(buffer) >= chunk_size:
                                chunk = buffer[:chunk_size]
                                if chunk.strip():
                                    documents.append(chunk)
                                    metadatas.append({"source": orig_name})
                                    ids.append(f"{safe_name}_c{global_chunk_id}")
                                    global_chunk_id += 1
                                buffer = buffer[chunk_size - chunk_overlap:] if chunk_overlap else buffer[chunk_size:]
                        if buffer.strip():
                            documents.append(buffer)
                            metadatas.append({"source": orig_name})
                            ids.append(f"{safe_name}_c{global_chunk_id}")
                            global_chunk_id += 1
                
                elif ext == ".docx":
                    text = docx2txt.process(str(file_path))
                    start = 0
                    while start < len(text):
                        end = start + chunk_size
                        chunk = text[start:end]
                        if chunk.strip():
                            documents.append(chunk)
                            metadatas.append({"source": orig_name})
                            ids.append(f"{safe_name}_c{global_chunk_id}")
                            global_chunk_id += 1
                        start = end - chunk_overlap
                
                elif ext == ".html" or ext == ".htm":
                    soup = BeautifulSoup(file_path.read_text(encoding="utf-8"), "html.parser")
                    text = soup.get_text(separator="\n")
                    start = 0
                    while start < len(text):
                        end = start + chunk_size
                        chunk = text[start:end]
                        if chunk.strip():
                            documents.append(chunk)
                            metadatas.append({"source": orig_name})
                            ids.append(f"{safe_name}_c{global_chunk_id}")
                            global_chunk_id += 1
                        start = end - chunk_overlap
            
            except Exception as e:
                st.warning(f"Failed on {orig_name}: {e} → skipped")
                continue
        
        if not documents:
            st.error("No text extracted!")
            st.stop()

        # Phase 3: Embed + Chroma
        status_text.text("Embedding & indexing...")
        embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)
        
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        
        batch_size = 32
        total = len(documents)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            collection.add(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end]
            )
            progress_bar.progress(0.3 + 0.7 * (end / total))
        
        progress_bar.progress(1.0)
        st.success(f"✅ Dataset created! {total} chunks → `{os.path.abspath(db_path)}/{collection_name}`")
        st.balloons()
        st.info("Transfer the folder to your agent Pi and point a tool/collection at it for RAG extensions.")
