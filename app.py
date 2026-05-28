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

# Page config - must be first Streamlit command
st.set_page_config(
    page_title="Data Science Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for ChatGPT-style UI
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main > div {
        padding: 0rem 1rem;
    }
    
    /* Chat container */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* User message */
    .stChatMessageUser {
        background-color: #f0f2f6;
    }
    
    /* Assistant message */
    .stChatMessageAssistant {
        background-color: #ffffff;
    }
    
    /* Hide sidebar toggle button */
    .st-emotion-cache-1wmy9hl {
        display: none;
    }
    
    /* Center the chat input */
    .stChatInputContainer {
        padding: 1rem;
        background: white;
        border-top: 1px solid #e5e5e5;
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 1000;
    }
    
    /* Main content area with padding for fixed input */
    .main .block-container {
        padding-bottom: 100px;
    }
    
    /* Welcome header */
    .welcome-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
    }
    
    /* Command chips */
    .command-chip {
        display: inline-block;
        background: #f0f2f6;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        cursor: pointer;
        transition: background 0.2s;
    }
    
    .command-chip:hover {
        background: #e0e2e6;
    }
    
    /* Feature buttons */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        color: white;
        cursor: pointer;
        transition: transform 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
    }
    
    .loading-dots:after {
        content: '...';
        animation: dots 1.5s steps(4, end) infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: '.'; }
        40% { content: '..'; }
        60%, 100% { content: '...'; }
    }
    
    /* Code block styling */
    pre {
        background: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        overflow-x: auto;
        color: #d4d4d4;
    }
    
    code {
        font-family: 'Courier New', monospace;
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
    st.session_state.assignment_mode = False
    st.session_state.code_mode = False
    st.session_state.research_mode = False
    
    # Load knowledge base
    with st.spinner("Initializing AI Tutor..."):
        st.session_state.rag_engine.load_initial_knowledge()

# Helper function to detect commands
def detect_command(message):
    """Detect special commands in user message"""
    message_lower = message.lower().strip()
    
    # Interview commands
    if re.search(r'^(practice|interview|mock interview)', message_lower):
        return "interview", None
    elif re.search(r'start interview', message_lower):
        return "interview_start", None
    
    # Assignment commands
    elif re.search(r'^(assignment|generate assignment|create assignment)', message_lower):
        # Extract topic if present
        topic_match = re.search(r'on\s+(\w+)', message_lower)
        topic = topic_match.group(1) if topic_match else None
        return "assignment", topic
    
    # Code commands
    elif re.search(r'^(code|write code|generate code|implement)', message_lower):
        return "code", message
    
    # Research commands
    elif re.search(r'^(research|deep research|search|find out about)', message_lower):
        topic = message_lower.replace('research', '').replace('deep research', '').replace('search', '').strip()
        return "research", topic if topic else None
    
    # Default to chat
    return "chat", message

# Process commands
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

# Generate AI response
def generate_response(user_message):
    """Generate AI response using RAG"""
    # Detect command
    command_type, command_content = detect_command(user_message)
    
    if command_type != "chat":
        # Handle special commands
        result = process_command(command_type, command_content)
        if command_type == "interview":
            return f"""🎯 **Interview Mode Activated!**

I'll help you practice with interview questions. Let's start with a question:

**{st.session_state.interview_system.get_next_question()}**

Type your answer, and I'll provide feedback. To end the interview, type 'end interview'."""
        
        elif command_type == "assignment":
            return f"""📝 **Assignment Generated!**

**Topic:** {result.get('topic', 'Data Science')}
**Questions:** {len(result.get('questions', []))}
**Total Points:** {result.get('total_points', 0)}

The assignment has been generated. You can download it using the button below.

💡 *Type 'submit assignment' followed by your answers to get them graded.*"""
        
        elif command_type == "code":
            return f"""💻 **Code Generated!**

Here's the code you requested:

```python
{result.get('code', '')}