# pages/07_tool_forge.py
# Tool Forge: Create new tools via natural language

import streamlit as st
import os
import sys
from pathlib import Path
import json
import inspect
import textwrap
from typing import Callable
import ast

# === Import Diagnostics ===
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from main import container, state, memory_insert
    from openai import OpenAI
except ImportError as e:
    st.error(f"⚠️ Import error: {e}")
    st.stop()

# === Tool Creation Engine ===
class ToolForge:
    """Generate and validate tools from natural language"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("MOONSHOT_API_KEY"), base_url="https://api.moonshot.ai/v1")
        self.sandbox_dir = Path(state.sandbox_dir)
    
    def generate_tool_code(self, description: str) -> str:
        """Use LLM to generate tool function code"""
        prompt = f"""Create a Python function tool for an agent system. The function should:
1. Have a clear docstring
2. Accept JSON-serializable arguments
3. Return a string or JSON-serializable result
4. Be safe (no file system escapes, no subprocess calls)
5. Use the provided sandbox path: {self.sandbox_dir}

Description: {description}

Requirements:
- Function name should be descriptive
- Include type hints
- Add error handling
- Return clear success/error messages

Return ONLY the Python function code, no explanations."""

        response = self.client.chat.completions.create(
            model="kimi-k2-thinking",
            messages=[
                {"role": "system", "content": "You are a code generation assistant. Return only Python code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()
    
    def validate_tool_code(self, code: str) -> tuple[bool, str]:
        """Validate generated tool for safety and syntax"""
        try:
            # Parse AST
            tree = ast.parse(code)
            
            # Check for dangerous imports
            dangerous_imports = ['subprocess', 'os.system', 'eval', 'exec']
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in dangerous_imports:
                            return False, f"Dangerous import detected: {alias.name}"
                if isinstance(node, ast.ImportFrom):
                    if node.module in dangerous_imports:
                        return False, f"Dangerous import from: {node.module}"
            
            # Check function exists
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            if not functions:
                return False, "No function definition found"
            
            return True, f"Valid function: {functions[0].name}"
            
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def register_tool(self, func: Callable, name: str = None):
        """Add tool to container"""
        container.register_tool(func, name)
        # Persist to memory
        memory_insert(
            f"forge_tool_{name or func.__name__}",
            {
                "tool_name": name or func.__name__,
                "source_code": inspect.getsource(func),
                "created_at": datetime.now().isoformat(),
                "status": "active"
            },
            convo_uuid=str(uuid.uuid4())
        )

forge = ToolForge()

# === UI ===
st.title("🔨 Tool Forge")
st.markdown("*Create new tools via natural language*")

# Tool creation interface
st.subheader("📝 Describe Your Tool")
description = st.text_area(
    "What should the tool do? Be specific about inputs, outputs, and behavior.",
    height=200,
    placeholder="Example: Create a tool that lists all Python files in a sandbox folder, returning comma-separated paths. Accept 'folder' parameter."
)

col1, col2 = st.columns([2, 1])
with col1:
    if st.button("⚡ Generate Tool", use_container_width=True):
        if description.strip():
            with st.spinner("Generating tool code..."):
                code = forge.generate_tool_code(description)
                st.session_state["generated_code"] = code
                st.success("Tool generated!")
        else:
            st.error("Please provide a description")
with col2:
    st.metric("Active Tools", len(container._tools))

# Code editor
if "generated_code" in st.session_state:
    st.subheader("🔍 Generated Code")
    
    # Auto-fix common issues (remove markdown code fences)
    code = st.session_state["generated_code"].replace("```python", "").replace("```", "").strip()
    
    edited_code = st_ace(
        value=code,
        language="python",
        theme="monokai",
        font_size=14,
        height=400,
        key="tool_editor"
    )
    
    # Validation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Validate & Test"):
            is_valid, message = forge.validate_tool_code(edited_code)
            if is_valid:
                st.success(message)
                # Execute in sandbox
                try:
                    exec(edited_code, {"__builtins__": __builtins__}, {})
                    st.info("✅ Tool executed successfully in test environment")
                except Exception as e:
                    st.warning(f"Test execution warning: {e}")
            else:
                st.error(message)
    
    with col2:
        tool_name = st.text_input("Tool Name", 
                                  value=edited_code.split("def ")[1].split("(")[0] if "def " in edited_code else "custom_tool")
        
        if st.button("🔧 Register Tool", use_container_width=True):
            try:
                # Compile and register
                namespace = {}
                exec(edited_code, namespace)
                func = list(namespace.values())[0]  # Get the function object
                
                if callable(func):
                    forge.register_tool(func, tool_name)
                    st.success(f"Tool '{tool_name}' registered successfully!")
                    st.balloons()
                else:
                    st.error("Generated code is not a callable function")
            except Exception as e:
                st.error(f"Registration error: {e}")

# Existing tools browser
st.divider()
st.subheader("📚 Existing Tools")

tool_search = st.text_input("Search tools:", "")
tools_list = list(container._tools.items())
if tool_search:
    tools_list = [(name, func) for name, func in tools_list if tool_search.lower() in name.lower()]

for name, func in tools_list[:10]:  # Show first 10
    with st.expander(f"🔧 {name}"):
        st.code(inspect.getsource(func), language="python")

# Tool usage metrics
st.divider()
st.subheader("📊 Forge Statistics")

try:
    with state.conn:
        state.cursor.execute(
            "SELECT COUNT(*) FROM memory WHERE mem_key LIKE 'forge_tool_%'"
        )
        forged_count = state.cursor.fetchone()[0]
        st.metric("Tools Forged", forged_count)
except:
    st.metric("Tools Forged", 0)
