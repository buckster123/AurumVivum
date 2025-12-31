import streamlit as st
import psutil
import subprocess
import json
from datetime import datetime

st.title("🖥️ System Diagnostics")

# CPU/Memory/Disk
col1, col2, col3 = st.columns(3)
col1.metric("CPU", f"{psutil.cpu_percent()}%")
col2.metric("RAM", f"{psutil.virtual_memory().percent}%")
col3.metric("Disk", f"{psutil.disk_usage('/').percent}%")

# NVMe temp (if available)
try:
    temp = subprocess.run(['sudo', 'nvme', 'smart-log', '/dev/nvme0n1'], 
                         capture_output=True, text=True)
    st.code(temp.stdout)
except:
    st.info("NVMe temp monitoring not available")

# Service status
if st.button("Check Apex Aurum Service"):
    status = subprocess.run(['systemctl', 'is-active', 'apex-aurum'], 
                          capture_output=True, text=True)
    st.info(f"Service: {status.stdout.strip()}")

# Recent errors from log
st.subheader("Recent Errors")
with open("./app.log", "r") as f:
    errors = [line for line in f.readlines() if "ERROR" in line][-10:]
    st.code("\n".join(errors))
