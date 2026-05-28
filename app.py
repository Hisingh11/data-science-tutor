import streamlit as st
import re
import os
import json
from datetime import datetime
from utils.model_manager import get_model_manager
from utils.rag_engine import RAGEngine
from utils.interview import InterviewSystem
from utils.assignment import AssignmentManager
from utils.code_assistant import CodeAssistant
from utils.deep_research import DeepResearchEngine
from utils.auth import register_user, login_user
from utils.chat_history import create_chat_session, add_message, get_session_messages, get_user_sessions, delete_session
from utils.image_recognition import analyze_image

st.set_page_config(page_title="Data Science Tutor", page_icon="💬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppHeader {display: none;}
    .stDeployButton {display: none;}
    [data-testid="stSidebar"] { background-color: #1e1e2f; padding-top: 2rem; }
    [data-testid="stSidebar"] * { color: #e0e0e0; }
    .sidebar-header { padding: 1rem; font-size: 1.2rem; font-weight: bold; border-bottom: 1px solid #333; margin-bottom: 1rem; }
    .main .block-container { padding: 0 !important; max-width: 1000px !important; margin: 0 auto !important; }
    .stChatMessage { padding: 1rem 1.5rem; margin-bottom: 0; }
    .stChatMessageUser { background-color: #2b2d3e; border-radius: 1.5rem; color: #fff; }
    .stChatMessageAssistant { background-color: transparent; color: #e0e0e0; }
    .stChatInputContainer { position: fixed; bottom: 0; left: 0; right: 0; background: #1e1e2f; padding: 1rem 2rem; border-top: 1px solid #333; z-index: 1000; }
    .stChatInput textarea { background-color: #2a2a3f !important; color: white !important; border-radius: 2rem !important; border: 1px solid #444 !important; }
    .minimal-footer { position: fixed; bottom: 0.5rem; left: 0; right: 0; text-align: center; font-size: 0.7rem; color: #777; background: transparent; z-index: 1001; pointer-events: none; }
</style>
""", unsafe_allow_html=True)

if 'core_initialized' not in st.session_state:
    st.session_state.model_manager = get_model_manager()
    st.session_state.rag_engine = RAGEngine()
    st.session_state.interview_system = InterviewSystem(st.session_state.model_manager)
    st.session_state.assignment_manager = AssignmentManager(st.session_state.model_manager)
    st.session_state.code_assistant = CodeAssistant(st.session_state.model_manager)
    st.session_state.research_engine = DeepResearchEngine(st.session_state.model_manager)
    with st.spinner(""):
        st.session_state.rag_engine.load_initial_knowledge()
    st.session_state.core_initialized = True

# Authentication
if 'user_id' not in st.session_state:
    st.markdown('<div style="max-width:400px;margin:auto;margin-top:10rem;"><h2>Welcome to Data Science Tutor</h2>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            user = login_user(username, password)
            if user:
                st.session_state.user_id, st.session_state.username, st.session_state.full_name = user
                sessions = get_user_sessions(st.session_state.user_id)
                if not sessions:
                    new_id = create_chat_session(st.session_state.user_id, "New Chat")
                    st.session_state.current_session_id = new_id
                else:
                    st.session_state.current_session_id = sessions[0]['id']
                st.session_state.messages = get_session_messages(st.session_state.current_session_id)
                st.rerun()
            else:
                st.error("Invalid credentials")
    with tab2:
        new_username = st.text_input("Username", key="reg_username")
        new_password = st.text_input("Password", type="password", key="reg_password")
        fullname = st.text_input("Full Name", key="fullname")
        if st.button("Register"):
            ok, msg = register_user(new_username, new_password, fullname)
            if ok:
                st.success("Registered! Please login.")
            else:
                st.error(msg)
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown(f"<div class='sidebar-header'>👤 {st.session_state.full_name}</div>", unsafe_allow_html=True)
    if st.button("➕ New Chat", use_container_width=True):
        new_id = create_chat_session(st.session_state.user_id, "New Chat")
        st.session_state.current_session_id = new_id
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("### 📜 History")
    sessions = get_user_sessions(st.session_state.user_id)
    for sess in sessions:
        col1, col2 = st.columns([4,1])
        with col1:
            if st.button(sess['title'], key=f"session_{sess['id']}", use_container_width=True):
                st.session_state.current_session_id = sess['id']
                st.session_state.messages = get_session_messages(sess['id'])
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{sess['id']}"):
                delete_session(sess['id'])
                remaining = get_user_sessions(st.session_state.user_id)
                if remaining:
                    st.session_state.current_session_id = remaining[0]['id']
                    st.session_state.messages = get_session_messages(remaining[0]['id'])
                else:
                    new_id = create_chat_session(st.session_state.user_id, "New Chat")
                    st.session_state.current_session_id = new_id
                    st.session_state.messages = []
                st.rerun()
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        for key in ['user_id', 'username', 'full_name', 'messages', 'current_session_id']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Chat area
if 'messages' not in st.session_state:
    st.session_state.messages = []
    if st.session_state.get('current_session_id'):
        st.session_state.messages = get_session_messages(st.session_state.current_session_id)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("attachments"):
            for att in msg["attachments"]:
                if att.endswith(('.png','.jpg','.jpeg','.gif')):
                    st.image(att, width=200)
                else:
                    st.caption(f"📎 {os.path.basename(att)}")

# Input row
with st.container():
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        prompt = st.chat_input("Message Data Science Tutor...")
    with col2:
        uploaded_file = st.file_uploader("📎", type=["png","jpg","jpeg","pdf","txt"], key="attached_file", label_visibility="collapsed")

if prompt or uploaded_file:
    user_content = prompt if prompt else ""
    attachments = []
    image_description = ""

    if uploaded_file:
        os.makedirs(f"./data/uploads/{st.session_state.user_id}", exist_ok=True)
        file_path = f"./data/uploads/{st.session_state.user_id}/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        attachments.append(file_path)

        if uploaded_file.type.startswith("image/"):
            with st.spinner("Analyzing image..."):
                image_description = analyze_image(file_path, "Describe this image in detail. If it contains code or text, extract it.")
            if image_description:
                user_content += "\n\n[Image analysis]: " + image_description

    add_message(st.session_state.current_session_id, "user", user_content, attachments)
    st.session_state.messages.append({"role": "user", "content": user_content, "attachments": attachments})

    full_prompt = user_content
    if image_description:
        full_prompt = image_description + "\n\nUser query: " + (prompt if prompt else "")
    relevant_docs = st.session_state.rag_engine.search(full_prompt, n_results=3)
    context = "\n\n".join([doc['text'] for doc in relevant_docs])
    if context:
        assistant_reply = st.session_state.model_manager.generate_with_context(full_prompt, context, "reasoning", 0.7)
    else:
        assistant_reply = st.session_state.model_manager.generate(full_prompt, "reasoning", 0.7)

    add_message(st.session_state.current_session_id, "assistant", assistant_reply, [])
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.rerun()

st.markdown('<div class="minimal-footer">Made by Himanshu</div>', unsafe_allow_html=True)