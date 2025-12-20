import streamlit as st
import os
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image

st.set_page_config(page_title="Sandbox Explorer", layout="centered")
st.title("📁 Sandbox Explorer")
st.markdown("Browse/edit sandbox files. View images (including multimodal base64). Safe navigation with callbacks.")

sandbox_path = Path("./sandbox").resolve()

# Initialize session_state path
if "current_path" not in st.session_state:
    st.session_state["current_path"] = str(sandbox_path)

current = Path(st.session_state["current_path"])

# Safety check
if not current.is_relative_to(sandbox_path):
    st.error("Stay in sandbox!")
    st.stop()

# Manual path input (no key conflict)
new_path = st.text_input("Current Path", value=str(current))
if new_path != str(current):
    safe_new = Path(new_path).resolve()
    if safe_new.is_relative_to(sandbox_path) and safe_new.exists():
        st.session_state["current_path"] = str(safe_new)
        st.rerun()
    else:
        st.warning("Invalid/safe path—staying put.")

# List items
items = list(current.iterdir())
items.sort(key=lambda x: (x.is_file(), x.name.lower()))

for item in items:
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        prefix = "📁" if item.is_dir() else "📄"
        st.write(f"{prefix} {item.name}")
    with col2:
        if item.is_dir():
            if st.button("Open", key=f"open_{item}"):
                st.session_state["current_path"] = str(item.resolve())
                st.rerun()
    with col3:
        if item.is_file() and st.button("View/Edit", key=f"edit_{item}"):
            st.session_state["edit_file"] = str(item.resolve())
            st.rerun()

# Parent navigation
if current != sandbox_path:
    if st.button("⬆️ Up one folder"):
        st.session_state["current_path"] = str(current.parent)
        st.rerun()

# File editor/viewer
if "edit_file" in st.session_state:
    file_path = Path(st.session_state["edit_file"])
    if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]:
        try:
            img = Image.open(file_path)
            st.image(img, caption=file_path.name)
        except:
            st.error("Failed to load image")
        if st.button("Close Viewer"):
            del st.session_state["edit_file"]
            st.rerun()
    else:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except:
            content = "(Binary or unreadable file)"
            st.info("Non-text file—showing raw preview not possible.")
        new_content = st.text_area("Edit File", content, height=500)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save Changes"):
                try:
                    file_path.write_text(new_content)
                    st.success("Saved!")
                except Exception as e:
                    st.error(f"Save failed: {e}")
        with col2:
            if st.button("Close Editor"):
                del st.session_state["edit_file"]
                st.rerun()

# Bonus: Quick Base64 Image Viewer
st.subheader("Quick Base64 Image Viewer (for multimodal chunks)")
b64 = st.text_area("Paste base64 string here")
if b64 and st.button("Render Image"):
    try:
        img_data = base64.b64decode(b64.strip())
        img = Image.open(BytesIO(img_data))
        st.image(img, caption="Rendered from base64")
    except Exception as e:
        st.error(f"Invalid base64: {e}")
