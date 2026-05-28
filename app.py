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
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean technical UI
st.markdown("""
<style>
    /* Hide all default Streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppHeader {display: none;}
    .stDeployButton {display: none;}
    
    /* Remove padding from main container */
    .main > div {
        padding: 0rem 1rem;
    }
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 4rem !important;
        max-width: 900px !important;
        margin: 0 auto !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        border: none;
    }
    
    /* User message - dark technical style */
    .stChatMessageUser {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0;
        border-left: 3px solid #00d4ff;
    }
    
    /* Assistant message - light technical style */
    .stChatMessageAssistant {
        background: linear-gradient(135deg, #f5f5f7 0%, #e8e8ec 100%);
        color: #1a1a2e;
        border-left: 3px solid #667eea;
    }
    
    /* Chat input container */
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(10px);
        padding: 1rem 2rem;
        border-top: 1px solid rgba(0,0,0,0.1);
        z-index: 1000;
    }
    
    /* Chat input styling */
    .stChatInput input {
        border-radius: 2rem !important;
        border: 1px solid #e0e0e0 !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 1rem !important;
        background: white !important;
    }
    
    .stChatInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102,126,234,0.1) !important;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 2rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.3);
    }
    
    /* Code block styling */
    pre {
        background: #1e1e1e !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        overflow-x: auto !important;
    }
    
    code {
        font-family: 'Fira Code', 'Courier New', monospace !important;
        font-size: 0.85rem !important;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Footer */
    .tech-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: center;
        padding: 0.5rem;
        font-size: 0.75rem;
        font-family: 'Fira Code', monospace;
        color: #888;
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(5px);
        z-index: 999;
        letter-spacing: 0.5px;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 2px solid #f3f3f3;
        border-top: 2px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
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
    
    # Load knowledge base
    with st.spinner("Initializing AI Tutor..."):
        st.session_state.rag_engine.load_initial_knowledge()
    
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hello! I'm your AI Data Science Tutor.\n\nI can help you with:\n• 📊 Data Science concepts\n• 🎯 Mock interviews\n• 📝 Assignments\n• 💻 Code generation\n• 🔬 Deep research\n\nWhat would you like to learn today?"
    })

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
            return "🎯 **Interview Mode Active**\n\n" + str(next_q) + "\n\n*Type your answer to continue, or say 'end interview' to stop.*"
        
        elif command_type == "assignment":
            topic_name = result.get('topic', 'Data Science') if result else 'Data Science'
            q_count = len(result.get('questions', [])) if result else 0
            return "📝 **Assignment Generated**\n\n**Topic:** " + topic_name + "\n**Questions:** " + str(q_count) + "\n\nYou can download the assignment using the button below."
        
        elif command_type == "code":
            code_text = result.get('code', '') if result else ''
            return "💻 **Code Generated**\n\n```python\n" + code_text + "\n```"
        
        elif command_type == "research":
            research_report = result.get('report', '')[:800] if result else ''
            return "🔬 **Research Summary**\n\n" + research_report + "\n\n*Want me to dive deeper? Just ask!*"
    
    # Regular chat - use RAG
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

# Main chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about Data Science, ML, Gen AI, or Agentic AI..."):
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
st.markdown('<div class="tech-footer">Made by Himanshu</div>', unsafe_allow_html=True)