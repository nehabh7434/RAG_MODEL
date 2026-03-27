# File: app.py
import streamlit as st
import sys
import os

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline.ambedkargpt1 import initialize_system, generate_answer

# Page Config
st.set_page_config(page_title="AmbedkarGPT", page_icon="📚", layout="wide")

# Title & Header
st.title("📚 AmbedkarGPT: SemRAG System")
st.markdown("### An AI Assistant for Dr. B.R. Ambedkar's Works")
st.markdown("---")

# Initialize System (Cached so it doesn't reload on every click)
@st.cache_resource
def load_engine():
    return initialize_system()

# Sidebar for System Status
with st.sidebar:
    st.header("⚙️ System Status")
    with st.spinner("Initializing Knowledge Graph..."):
        engine = load_engine()
    st.success("System Ready!")
    st.info("Architecture: SemRAG\nModel: Groq/Llama3-8b\nSearch: Local + Global")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question (e.g., 'What is the origin of Shudras?')..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking... (Searching Knowledge Graph)"):
            response = generate_answer(prompt, engine)
            st.markdown(response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})