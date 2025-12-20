# pages/vector_db_creator.py
import streamlit as st
import os
import tempfile
from pathlib import Path
import io
import base64
from datetime import datetime

try:
    import ocrmypdf
    has_ocrmypdf = True
except ImportError:
    has_ocrmypdf = False

try:
    from pdf2image import convert_from_path
    from PIL import Image
    has_pdf2image = True
except ImportError:
    has_pdf2image = False

from pypdf import PdfReader
import docx2txt
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

st.set_page_config(page_title="Tiny Vector Dataset Creator", layout="centered")
st.title("🛠️ Tiny Vector Dataset Creator (v2 – Multimodal Ready)")
st.markdown("""
Create lightweight, persistent Chroma vector DBs from documents **and images**.  
Perfect for static "memory extensions" on your Pi 5 – text + optional vision (CLIP).

- Streamed/low-memory processing
- Page-level metadata for PDFs
- Separate toggles: OCR, image embedding, PDF image extraction
- Clean-run safety: auto-delete old collection or unique names/folders
- CLIP multimodal when images enabled (semantic text ↔ image search)
""")

# Sidebar settings
with st.sidebar:
    st.header("Core Settings")
    text_model_options = [
        "all-MiniLM-L6-v2",       # Fast default
        "all-MiniLM-L3-v2",
        "all-mpnet-base-v2",      # Higher quality (768-dim)
        "paraphrase-MiniLM-L3-v2"
    ]
    
    enable_images = st.checkbox("Enable image embedding (multimodal CLIP)", value=False)
    if enable_images:
        st.info("CLIP-ViT-B-32 will be used (512-dim, text+image compatible). Slower on image-heavy docs.")
    
    text_model_name = st.selectbox(
        "Text embedding model (used when images off, or for text when images on)",
        text_model_options,
        index=0
    )
    
    chunk_size = st.slider("Text chunk size (chars)", 500, 4000, 2000)
    chunk_overlap = st.slider("Text chunk overlap (chars)", 0, 600, 200)
    
    st.header("Processing Toggles")
    force_ocr = st.checkbox("Force OCR on PDFs (for scanned/image-only)", value=False, disabled=not has_ocrmypdf)
    if not has_ocrmypdf:
        st.caption("⚠️ Install ocrmypdf for OCR support")
    
    extract_pdf_images = st.checkbox("Extract images from PDFs", value=True, disabled=not enable_images or not has_pdf2image)
    if enable_images and not has_pdf2image:
        st.caption("⚠️ Install pdf2image + poppler-utils for PDF image extraction")
    
    if enable_images and extract_pdf_images:
        pdf_dpi = st.slider("PDF image extraction DPI (lower = faster/smaller)", 100, 300, 150)
        resize_images = st.checkbox("Resize extracted/uploaded images to max 512x512", value=True)
        max_resize = st.slider("Max resize dimension", 256, 1024, 512) if resize_images else None
    
    st.header("DB Settings")
    db_path = st.text_input("Base DB folder", value="./vector_extensions")
    use_timestamp_folder = st.checkbox("Auto-create timestamped subfolder (clean run)", value=True)
    collection_name = st.text_input("Collection name", value="docs" if not use_timestamp_folder else "main")
    
    auto_delete_old = st.checkbox("Auto-delete existing collection on start (prevents dim errors)", value=True)

# Dynamic file types
file_types = ["pdf", "txt", "md", "markdown", "docx", "html", "htm"]
if enable_images:
    file_types += ["jpg", "jpeg", "png", "gif", "bmp", "tiff"]

uploaded_files = st.file_uploader(
    "Upload documents/images",
    accept_multiple_files=True,
    type=file_types,
    key="uploaded_files"
)

if uploaded_files:
    st.info(f"Uploaded {len(uploaded_files)} file(s)")

if st.button("🚀 Create Dataset", type="primary"):
    if not uploaded_files:
        st.error("Upload files first!")
        st.stop()

    # Setup DB path (timestamped subfolder if enabled)
    actual_db_path = Path(db_path)
    if use_timestamp_folder:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_db_path = actual_db_path / timestamp
        collection_name = collection_name or "main"
    actual_db_path.mkdir(parents=True, exist_ok=True)
    st.info(f"DB path: `{actual_db_path.resolve()}` | Collection: `{collection_name}`")

    # Chroma client + optional auto-delete
    client = chromadb.PersistentClient(path=str(actual_db_path))
    if auto_delete_old and collection_name in [col.name for col in client.list_collections()]:
        client.delete_collection(collection_name)
        st.success(f"Deleted old collection '{collection_name}' for clean run")

    # Embedding setup
    if enable_images:
        clip_model_name = "clip-ViT-B-32"
        with st.spinner(f"Loading CLIP model {clip_model_name}..."):
            model = SentenceTransformer(clip_model_name)
        text_encoder = model
        image_encoder = model
        embedding_function = None  # Manual embedding
    else:
        embedding_function = SentenceTransformerEmbeddingFunction(model_name=text_model_name)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function if not enable_images else None
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Phase 1: Save files + OCR
        processed_files: list[tuple[str, Path]] = []
        for i, up_file in enumerate(uploaded_files):
            status_text.text(f"Saving/OCR {i+1}/{len(uploaded_files)}: {up_file.name}")
            saved = temp_path / up_file.name
            with open(saved, "wb") as f:
                f.write(up_file.getbuffer())
            
            final_path = saved
            if force_ocr and up_file.name.lower().endswith(".pdf"):
                ocr_path = temp_path / f"ocr_{up_file.name}"
                try:
                    ocrmypdf.ocr(str(saved), str(ocr_path), jobs=os.cpu_count() or 4, optimize=1)
                    final_path = ocr_path
                    st.caption(f"OCR complete on {up_file.name}")
                except Exception as e:
                    st.warning(f"OCR failed on {up_file.name}: {e} → using original")
            
            processed_files.append((up_file.name, final_path))
            progress_bar.progress(0.2 * (i + 1) / len(uploaded_files))

        # Phase 2: Extract + chunk + image handling
        status_text.text("Extracting text/images & chunking...")
        documents = []
        embeddings = [] if enable_images else None
        metadatas = []
        ids = []
        global_id = 0
        overlap_buffer = ""
        
        for orig_name, file_path in processed_files:
            ext = Path(orig_name).suffix.lower()
            safe_name = Path(orig_name).stem.replace(".", "_")
            
            try:
                # Pure image files
                if ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"] and enable_images:
                    img = Image.open(file_path).convert("RGB")
                    if resize_images:
                        img.thumbnail((max_resize, max_resize))
                    img_embedding = image_encoder.encode(img).tolist()
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format="PNG")
                    img_b64 = base64.b64encode(img_bytes.getvalue()).decode()
                    
                    documents.append("")  # No text
                    embeddings.append(img_embedding)
                    metadatas.append({"source": orig_name, "type": "image", "image_base64": img_b64})
                    ids.append(f"{safe_name}_img{global_id}")
                    global_id += 1
                    continue
                
                # PDFs (text + optional images)
                if ext == ".pdf":
                    reader = PdfReader(str(file_path))
                    for page_num, page in enumerate(reader.pages, start=1):
                        # Text chunking
                        page_text = (page.extract_text() or "")
                        page_text = overlap_buffer + page_text
                        start = 0
                        while start < len(page_text):
                            end = start + chunk_size
                            chunk = page_text[start:end]
                            if chunk.strip():
                                if enable_images:
                                    chunk_emb = text_encoder.encode(chunk).tolist()
                                    embeddings.append(chunk_emb)
                                documents.append(chunk)
                                metadatas.append({"source": orig_name, "page": page_num, "type": "text"})
                                ids.append(f"{safe_name}_p{page_num}_c{global_id}")
                                global_id += 1
                            start = end - chunk_overlap
                        if chunk_overlap and start < len(page_text):
                            overlap_buffer = page_text[-chunk_overlap:]
                        else:
                            overlap_buffer = ""
                        
                        # PDF image extraction
                        if enable_images and extract_pdf_images:
                            try:
                                pil_images = convert_from_path(str(file_path), dpi=pdf_dpi, first_page=page_num, last_page=page_num)
                                for img_num, pil_img in enumerate(pil_images):
                                    if resize_images:
                                        pil_img.thumbnail((max_resize, max_resize))
                                    img_emb = image_encoder.encode(pil_img).tolist()
                                    img_bytes = io.BytesIO()
                                    pil_img.save(img_bytes, format="PNG")
                                    img_b64 = base64.b64encode(img_bytes.getvalue()).decode()
                                    
                                    documents.append("")
                                    embeddings.append(img_emb)
                                    metadatas.append({
                                        "source": orig_name,
                                        "page": page_num,
                                        "type": "image",
                                        "image_num": img_num,
                                        "image_base64": img_b64
                                    })
                                    ids.append(f"{safe_name}_p{page_num}_img{img_num}")
                                    global_id += 1
                            except Exception as e:
                                st.caption(f"Image extraction failed on page {page_num}: {e}")
                    continue
                
                # Text-based files
                if ext in [".txt", ".md", ".markdown"]:
                    with open(file_path, "r", encoding="utf-8") as f:
                        buffer = ""
                        for line in f:
                            buffer += line
                            while len(buffer) >= chunk_size:
                                chunk = buffer[:chunk_size]
                                if chunk.strip():
                                    if enable_images:
                                        chunk_emb = text_encoder.encode(chunk).tolist()
                                        embeddings.append(chunk_emb)
                                    documents.append(chunk)
                                    metadatas.append({"source": orig_name, "type": "text"})
                                    ids.append(f"{safe_name}_c{global_id}")
                                    global_id += 1
                                buffer = buffer[chunk_size - chunk_overlap:] if chunk_overlap else buffer[chunk_size:]
                        if buffer.strip():
                            if enable_images:
                                chunk_emb = text_encoder.encode(buffer).tolist()
                                embeddings.append(chunk_emb)
                            documents.append(buffer)
                            metadatas.append({"source": orig_name, "type": "text"})
                            ids.append(f"{safe_name}_c{global_id}")
                            global_id += 1
                
                elif ext == ".docx":
                    text = docx2txt.process(str(file_path))
                    start = 0
                    while start < len(text):
                        end = start + chunk_size
                        chunk = text[start:end]
                        if chunk.strip():
                            if enable_images:
                                chunk_emb = text_encoder.encode(chunk).tolist()
                                embeddings.append(chunk_emb)
                            documents.append(chunk)
                            metadatas.append({"source": orig_name, "type": "text"})
                            ids.append(f"{safe_name}_c{global_id}")
                            global_id += 1
                        start = end - chunk_overlap
                
                elif ext in [".html", ".htm"]:
                    soup = BeautifulSoup(file_path.read_text(encoding="utf-8"), "html.parser")
                    text = soup.get_text(separator="\n")
                    start = 0
                    while start < len(text):
                        end = start + chunk_size
                        chunk = text[start:end]
                        if chunk.strip():
                            if enable_images:
                                chunk_emb = text_encoder.encode(chunk).tolist()
                                embeddings.append(chunk_emb)
                            documents.append(chunk)
                            metadatas.append({"source": orig_name, "type": "text"})
                            ids.append(f"{safe_name}_c{global_id}")
                            global_id += 1
                        start = end - chunk_overlap
            
            except Exception as e:
                st.warning(f"Failed processing {orig_name}: {e} → skipped")
                continue
        
        if not documents:
            st.error("No content extracted!")
            st.stop()

        # Phase 3: Add to Chroma
        status_text.text("Embedding & indexing...")
        batch_size = 16 if enable_images else 32  # Smaller batches for images
        
        total = len(documents)
        for start_idx in range(0, total, batch_size):
            end_idx = min(start_idx + batch_size, total)
            batch_docs = documents[start_idx:end_idx]
            batch_meta = metadatas[start_idx:end_idx]
            batch_ids = ids[start_idx:end_idx]
            
            if enable_images:
                batch_embs = []
                for j, meta in enumerate(batch_meta):
                    if meta["type"] == "image":
                        img_data = base64.b64decode(meta["image_base64"])
                        img = Image.open(io.BytesIO(img_data)).convert("RGB")
                        batch_embs.append(image_encoder.encode(img).tolist())
                    else:
                        batch_embs.append(text_encoder.encode(batch_docs[j]).tolist())
                collection.add(
                    embeddings=batch_embs,
                    documents=batch_docs,
                    metadatas=batch_meta,
                    ids=batch_ids
                )
            else:
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_meta,
                    ids=batch_ids
                )
            
            progress_bar.progress(0.2 + 0.8 * (end_idx / total))
        
        progress_bar.progress(1.0)
        st.success(f"✅ Dataset ready! {total} items → `{actual_db_path.resolve()}/{collection_name}`")
        st.balloons()
        st.info("""
        Tips:
        - Transfer folder to agent Pi
        - Query with text → retrieves text + relevant images
        - Agent can display images via base64 in metadata
        - Use unique folders/collections to avoid dim conflicts
        """)
