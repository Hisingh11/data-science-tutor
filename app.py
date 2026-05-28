import streamlit as st
import hashlib
import sqlite3
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import List, Dict

# ----------------------------------------------------------------------
#  Database & Authentication (built‑in, no external import)
# ----------------------------------------------------------------------

DB_PATH = "./data/ds_tutor.db"
os.makedirs("./data", exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT,
        created_at TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        title TEXT,
        created_at TIMESTAMP,
        updated_at TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        role TEXT,
        content TEXT,
        timestamp TIMESTAMP,
        attachments TEXT,
        FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
    )''')
    conn.commit()
    conn.close()

init_db()

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def register_user(username: str, password: str, full_name: str = ""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, full_name, created_at) VALUES (?,?,?,?)",
                  (username, hash_password(password), full_name, datetime.now(timezone.utc)))
        conn.commit()
        conn.close()
        return True, "Registered"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username exists"

def login_user(username: str, password: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, username, full_name FROM users WHERE username=? AND password_hash=?", 
              (username, hash_password(password)))
    row = c.fetchone()
    conn.close()
    return row

def create_chat_session(user_id: int, title: str = "New Chat"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now(timezone.utc)
    c.execute("INSERT INTO chat_sessions (user_id, title, created_at, updated_at) VALUES (?,?,?,?)",
              (user_id, title, now, now))
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id

def add_message(session_id: int, role: str, content: str, attachments: list = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    attachments_json = json.dumps(attachments or [])
    c.execute("INSERT INTO chat_messages (session_id, role, content, timestamp, attachments) VALUES (?,?,?,?,?)",
              (session_id, role, content, datetime.now(timezone.utc), attachments_json))
    c.execute("UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (datetime.now(timezone.utc), session_id))
    conn.commit()
    conn.close()

def get_session_messages(session_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content, attachments FROM chat_messages WHERE session_id = ? ORDER BY timestamp", (session_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "attachments": json.loads(r[2]) if r[2] else []} for r in rows]

def get_user_sessions(user_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title, updated_at FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]

def delete_session(session_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
    c.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()

# ----------------------------------------------------------------------
#  Core AI components
# ----------------------------------------------------------------------
from utils.model_manager import get_model_manager
from utils.rag_engine import RAGEngine
from utils.interview import InterviewSystem
from utils.assignment import AssignmentManager
from utils.code_assistant import CodeAssistant
from utils.deep_research import DeepResearchEngine
from utils.image_recognition import analyze_image

# ========== NEW NAME AND LOGO ==========
st.set_page_config(page_title="Data Scientist BOT", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
# =======================================

# Clean modern UI with white input field
css_code = """
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppHeader {display: none;}
    .stDeployButton {display: none;}

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e1e2f;
        padding-top: 2rem;
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0;
    }
    .sidebar-header {
        padding: 1rem;
        font-size: 1.2rem;
        font-weight: bold;
        border-bottom: 1px solid #333;
        margin-bottom: 1rem;
    }

    /* Main chat area */
    .main .block-container {
        padding: 1rem 1rem 5rem 1rem !important;
        max-width: 900px !important;
        margin: 0 auto !important;
    }

    /* Message bubbles */
    .stChatMessage {
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        border-radius: 1.25rem;
        max-width: 85%;
    }
    .stChatMessageUser {
        background-color: #2b2d3e;
        color: #fff;
        margin-left: auto;
        border-bottom-right-radius: 0.25rem;
    }
    .stChatMessageAssistant {
        background-color: #f0f2f5;
        color: #111;
        margin-right: auto;
        border-bottom-left-radius: 0.25rem;
    }

    /* White input field */
    .stChatInput textarea {
        background-color: white !important;
        color: #1e1e2f !important;
        border-radius: 2rem !important;
        border: 1px solid #ccc !important;
        font-size: 0.9rem !important;
        padding: 0.5rem 1rem !important;
    }
    .stChatInput textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 1px #667eea !important;
    }
    .stChatInputContainer {
        background: #f5f5f5 !important;
        border-top: 1px solid #e0e0e0 !important;
    }

    /* File uploader */
    .stFileUploader > div:first-child {
        display: none;
    }
    .stFileUploader {
        margin: 0;
        padding: 0;
    }

    /* Footer */
    .minimal-footer {
        position: fixed;
        bottom: 0.25rem;
        left: 0;
        right: 0;
        text-align: center;
        font-size: 0.7rem;
        color: #777;
        background: transparent;
        z-index: 1001;
        pointer-events: none;
    }
</style>
"""
st.markdown(css_code, unsafe_allow_html=True)

# Core models
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

def create_guest_session():
    guest_id = f"guest_{uuid.uuid4().hex[:8]}"
    st.session_state.user_id = guest_id
    st.session_state.username = "Guest"
    st.session_state.full_name = "Guest User"
    st.session_state.is_guest = True
    st.session_state.current_session_id = f"guest_session_{uuid.uuid4().hex[:8]}"
    st.session_state.messages = []
    st.rerun()

# Authentication
if 'user_id' not in st.session_state:
    # New heading with robot emoji
    st.markdown('<div style="max-width:450px;margin:auto;margin-top:5rem;"><h2 style="text-align:center;">🤖 Data Scientist BOT</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚪 Continue as Guest", use_container_width=True):
            create_guest_session()
    with col2:
        st.markdown("#### or")
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary"):
            user = login_user(username, password)
            if user:
                st.session_state.user_id, st.session_state.username, st.session_state.full_name = user
                st.session_state.is_guest = False
                sessions = get_user_sessions(st.session_state.user_id)
                if not sessions:
                    new_id = create_chat_session(st.session_state.user_id, "New Chat")
                    st.session_state.current_session_id = new_id
                else:
                    st.session_state.current_session_id = sessions[0]['id']
                st.session_state.messages = get_session_messages(st.session_state.current_session_id)
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
    with tab2:
        new_username = st.text_input("Username", key="reg_username")
        new_password = st.text_input("Password", type="password", key="reg_password")
        fullname = st.text_input("Full Name", key="fullname")
        if st.button("Register"):
            if new_username and new_password:
                ok, msg = register_user(new_username, new_password, fullname)
                if ok:
                    st.success("✅ Registered! Please login.")
                else:
                    st.error(msg)
            else:
                st.warning("Username and password required.")
    st.stop()

# Sidebar (optional logo can be added here)
with st.sidebar:
    # You can add a logo image here – replace with your own URL or local file
    # st.image("https://img.icons8.com/color/96/000000/robot.png", width=60)
    st.markdown(f"<div class='sidebar-header'>🤖 {st.session_state.full_name} { '(Guest)' if st.session_state.get('is_guest') else ''}</div>", unsafe_allow_html=True)
    if not st.session_state.get('is_guest'):
        if st.button("➕ New Chat", use_container_width=True):
            new_id = create_chat_session(st.session_state.user_id, "New Chat")
            st.session_state.current_session_id = new_id
            st.session_state.messages = get_session_messages(new_id)
            st.rerun()
        st.markdown("---")
        st.markdown("### 📜 History")
        for sess in get_user_sessions(st.session_state.user_id):
            colA, colB = st.columns([4,1])
            with colA:
                if st.button(sess['title'], key=f"session_{sess['id']}", use_container_width=True):
                    st.session_state.current_session_id = sess['id']
                    st.session_state.messages = get_session_messages(sess['id'])
                    st.rerun()
            with colB:
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
    else:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.info("💡 Guest mode: chat history not saved.")
        st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        for k in ['user_id', 'username', 'full_name', 'messages', 'current_session_id', 'is_guest']:
            if k in st.session_state: del st.session_state[k]
        st.rerun()

# Chat messages
if 'messages' not in st.session_state:
    st.session_state.messages = []
    if not st.session_state.get('is_guest') and st.session_state.get('current_session_id'):
        st.session_state.messages = get_session_messages(st.session_state.current_session_id)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("attachments"):
            for a in msg["attachments"]:
                if a.endswith(('.png','.jpg','.jpeg','.gif')):
                    st.image(a, width=200)
                else:
                    st.caption(f"📎 {os.path.basename(a)}")

# Input row with paperclip
with st.container():
    col_input, col_attach = st.columns([0.85, 0.15])
    with col_input:
        prompt = st.chat_input("Message Data Scientist BOT...")
    with col_attach:
        uploaded_file = st.file_uploader(
            "📎", 
            type=["png","jpg","jpeg","pdf","txt"],
            key="paperclip_upload",
            label_visibility="collapsed"
        )

# Process input
if prompt or uploaded_file:
    user_content = prompt if prompt else ""
    attachments = []
    img_desc = ""
    if uploaded_file:
        if st.session_state.get('is_guest'):
            os.makedirs("./data/guest_uploads", exist_ok=True)
            path = f"./data/guest_uploads/{uploaded_file.name}"
        else:
            os.makedirs(f"./data/uploads/{st.session_state.user_id}", exist_ok=True)
            path = f"./data/uploads/{st.session_state.user_id}/{uploaded_file.name}"
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        attachments.append(path)
        if uploaded_file.type.startswith("image/"):
            with st.spinner("Analyzing image..."):
                img_desc = analyze_image(path, "Describe this image. If code/text extract it.")
            if img_desc:
                user_content += f"\n\n[Image analysis]: {img_desc}"
    # Save user message
    if st.session_state.get('is_guest'):
        st.session_state.messages.append({"role":"user","content":user_content,"attachments":attachments})
    else:
        add_message(st.session_state.current_session_id, "user", user_content, attachments)
        st.session_state.messages.append({"role":"user","content":user_content,"attachments":attachments})
    # Generate assistant reply
    full_prompt = user_content
    if img_desc:
        full_prompt = img_desc + "\n\nUser query: " + (prompt if prompt else "")
    docs = st.session_state.rag_engine.search(full_prompt, n_results=3)
    ctx = "\n\n".join([d['text'] for d in docs])
    if ctx:
        reply = st.session_state.model_manager.generate_with_context(full_prompt, ctx, "reasoning", 0.7)
    else:
        reply = st.session_state.model_manager.generate(full_prompt, "reasoning", 0.7)
    if st.session_state.get('is_guest'):
        st.session_state.messages.append({"role":"assistant","content":reply})
    else:
        add_message(st.session_state.current_session_id, "assistant", reply, [])
        st.session_state.messages.append({"role":"assistant","content":reply})
    st.rerun()

st.markdown('<div class="minimal-footer">Made by Himanshu</div>', unsafe_allow_html=True)