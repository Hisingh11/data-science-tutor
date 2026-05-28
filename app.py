import streamlit as st
from PIL import Image
import io
import os
from datetime import datetime
import json
import re

from utils.model_manager import get_model_manager
from utils.rag_engine import RAGEngine
from utils.interview import InterviewSystem
from utils.assignment import AssignmentManager
from utils.code_assistant import CodeAssistant
from utils.deep_research import DeepResearchEngine

# Page config
st.set_page_config(
    page_title="Data Science Tutor",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ChatGPT-style minimal UI
st.markdown("""
<style>
    /* Hide all default Streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppHeader {display: none;}
    .stDeployButton {display: none;}
    .stStatusWidget {display: none;}
    
    /* Remove padding and center content */
    .main > div {
        padding: 0rem;
    }
    
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        max-width: 800px !important;
        margin: 0 auto !important;
    }
    
    /* Chat messages container */
    .stChatMessage {
        padding: 1rem 1.5rem;
        margin-bottom: 0rem;
        border: none;
        background: transparent;
    }
    
    /* User message - right aligned, light grey */
    .stChatMessageUser {
        background-color: #f0f2f5;
        border-radius: 1.5rem;
        margin-left: 2rem;
    }
    
    /* Assistant message - left aligned, white */
    .stChatMessageAssistant {
        background-color: transparent;
        border-radius: 0rem;
        margin-right: 2rem;
    }
    
    /* Chat input container - fixed at bottom */
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem 2rem 2rem 2rem;
        border-top: 1px solid #e5e5e5;
        z-index: 1000;
    }
    
    /* Chat input field */
    .stChatInput textarea {
        border-radius: 1.5rem !important;
        border: 1px solid #e5e5e5 !important;
        background: white !important;
        font-size: 1rem !important;
        padding: 0.75rem 1.5rem !important;
        box-shadow: none !important;
    }
    
    .stChatInput textarea:focus {
        border-color: #10a37f !important;
        box-shadow: none !important;
    }
    
    /* Hide the default Streamlit chat input bar */
    .stChatInput > div {
        background: transparent !important;
    }
    
    /* Hide any success/warning/info boxes */
    .stAlert, .element-container:has(.stAlert) {
        display: none !important;
    }
    
    /* Hide the sidebar toggle button */
    .st-emotion-cache-1wmy9hl {
        display: none;
    }
    
    /* Typography */
    p, div, span {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Code blocks */
    pre {
        background: #1e1e1e !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        overflow-x: auto !important;
    }
    
    code {
        font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace !important;
        font-size: 0.85rem !important;
    }
    
    /* Footer - minimal */
    .minimal-footer {
        position: fixed;
        bottom: 0.5rem;
        left: 0;
        right: 0;
        text-align: center;
        font-size: 0.7rem;
        color: #aaa;
        background: transparent;
        z-index: 1001;
        pointer-events: none;
    }
    
    /* Remove any extra padding from chat list */
    .stChatMessageContent {
        padding: 0.25rem 0;
    }
    
    /* Avatar icons - optional */
    .stChatMessageAvatar {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.model_manager = get_model_manager()
    st.session_state.rag_engine = RAGEngine()
    st.session_state.interview_system = InterviewSystem(st.session_state.model_manager)
    st.session_state.assignment_manager = AssignmentManager(st.session_state.model_manager)
    st.session_state.code_assistant = CodeAssistant(st.session_state.model_manager)
    st.session_state.research_engine = DeepResearchEngine(st.session_state.model_manager)
    st.session_state.messages = []
    st.session_state.interview_active = False
    
    # Suppress the "Added documents" message
    with st.spinner(""):
        st.session_state.rag_engine.load_initial_knowledge()
    
    # Initial assistant greeting (minimal)
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! How can I help you with Data Science today?"
    })

# Helper functions
def detect_command(message):
    """Detect special commands"""
    message_lower = message.lower().strip()
    
    if re.search(r'^(practice|interview|mock interview)', message_lower):
        return "interview", None
    elif re.search(r'^(assignment|generate assignment|create assignment)', message_lower):
        topic_match = re.search(r'on\s+(\w+)', message_lower)
        topic = topic_match.group(1) if topic_match else None
        return "assignment", topic
    elif re.search(r'^(code|write code|generate code|implement)', message_lower):
        return "code", message
    elif re.search(r'^(research|deep research|search)', message_lower):
        topic = message_lower.replace('research', '').replace('deep research', '').replace('search', '').strip()
        return "research", topic if topic else None
    
    return "chat", message

def process_command(command_type, content):
    """Process detected commands"""
    if command_type == "interview":
        return st.session_state.interview_system.start_interview("Data Science Fundamentals", "beginner")
    elif command_type == "assignment":
        topic = content if content else "Data Science Fundamentals"
        assignment = st.session_state.assignment_manager.generate_assignment(topic, "intermediate", 10)
        return assignment
    elif command_type == "code":
        result = st.session_state.code_assistant.generate_code(content, "python")
        return result
    elif command_type == "research":
        topic = content if content else "latest developments in AI"
        research = st.session_state.research_engine.deep_research(topic)
        return research
    return None

def generate_response(user_message):
    """Generate AI response"""
    command_type, command_content = detect_command(user_message)
    
    if command_type != "chat":
        result = process_command(command_type, command_content)
        
        if command_type == "interview":
            next_q = st.session_state.interview_system.get_next_question()
            return f"🎯 **Interview mode**\n\n{next_q}\n\n_(Type your answer, or say 'end interview' to stop)_"
        
        elif command_type == "assignment":
            topic_name = result.get('topic', 'Data Science') if result else 'Data Science'
            q_count = len(result.get('questions', [])) if result else 0
            return f"📝 **Assignment generated**\n\nTopic: {topic_name}\nQuestions: {q_count}\n\n[Download button will appear below]"
        
        elif command_type == "code":
            code_text = result.get('code', '') if result else ''
            return f"💻 **Code**\n\n```python\n{code_text}\n```"
        
        elif command_type == "research":
            research_report = result.get('report', '')[:800] if result else ''
            return f"🔬 **Research summary**\n\n{research_report}"
    
    # Regular chat with RAG
    relevant_docs = st.session_state.rag_engine.search(user_message, n_results=3)
    context = "\n\n".join([doc['text'] for doc in relevant_docs])
    
    if context and len(context) > 100:
        response = st.session_state.model_manager.generate_with_context(
            user_message, context, "reasoning", 0.7
        )
    else:
        response = st.session_state.model_manager.generate(
            user_message, "reasoning", 0.7
        )
    
    return response

# Main chat interface - display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input (no extra buttons)
if prompt := st.chat_input("Message Data Science Tutor..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner(""):
            response = generate_response(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Minimal footer
st.markdown('<div class="minimal-footer">Made by Himanshu</div>', unsafe_allow_html=True)