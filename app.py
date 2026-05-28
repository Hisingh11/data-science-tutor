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
    
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        text-align: center;
        padding: 0.5rem;
        background: rgba(255,255,255,0.9);
        font-size: 0.8rem;
        color: #666;
        z-index: 999;
        border-top: 1px solid #e5e5e5;
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

def detect_command(message):
    """Detect special commands in user message"""
    message_lower = message.lower().strip()
    
    if re.search(r'^(practice|interview|mock interview)', message_lower):
        return "interview", None
    elif re.search(r'start interview', message_lower):
        return "interview_start", None
    elif re.search(r'^(assignment|generate assignment|create assignment)', message_lower):
        topic_match = re.search(r'on\s+(\w+)', message_lower)
        topic = topic_match.group(1) if topic_match else None
        return "assignment", topic
    elif re.search(r'^(code|write code|generate code|implement)', message_lower):
        return "code", message
    elif re.search(r'^(research|deep research|search|find out about)', message_lower):
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
    """Generate AI response using RAG"""
    command_type, command_content = detect_command(user_message)
    
    if command_type != "chat":
        result = process_command(command_type, command_content)
        
        if command_type == "interview":
            next_q = st.session_state.interview_system.get_next_question()
            return "🎯 **Interview Mode Activated!**\n\nI'll help you practice with interview questions. Let's start with a question:\n\n**" + str(next_q) + "**\n\nType your answer, and I'll provide feedback. To end the interview, type 'end interview'."
        
        elif command_type == "assignment":
            topic_name = result.get('topic', 'Data Science') if result else 'Data Science'
            q_count = len(result.get('questions', [])) if result else 0
            points = result.get('total_points', 0) if result else 0
            return "📝 **Assignment Generated!**\n\n**Topic:** " + topic_name + "\n**Questions:** " + str(q_count) + "\n**Total Points:** " + str(points) + "\n\nThe assignment has been generated. You can download it using the button below.\n\n💡 *Type 'submit assignment' followed by your answers to get them graded.*"
        
        elif command_type == "code":
            code_text = result.get('code', '') if result else ''
            explanation = result.get('full_response', '')[:500] if result else ''
            return "💻 **Code Generated!**\n\nHere's the code you requested:\n\n```python\n" + code_text + "\n```\n\n**Explanation:** " + explanation + "\n\n💡 *Need help debugging? Paste your code and I'll help fix errors.*"
        
        elif command_type == "research":
            research_topic = result.get('topic', 'Topic') if result else 'Topic'
            research_report = result.get('report', '')[:1000] if result else ''
            takeaways = result.get('key_takeaways', '') if result else ''
            return "🔬 **Research Complete!**\n\n## 📄 Research Report: " + research_topic + "\n\n" + research_report + "\n\n**Key Takeaways:**\n" + takeaways + "\n\n💡 *Want me to dive deeper into any specific aspect? Just ask!*"
    
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

# Main UI
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-header">
        <h1>🎓 Data Science Tutor</h1>
        <p>Your Personal AI Assistant for Data Science, ML, Gen AI & Agentic AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Quick Actions")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("🎯 Practice Interview", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "I want to practice for an interview"
            })
            response = generate_response("I want to practice for an interview")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            st.rerun()
    
    with col2:
        if st.button("📝 Generate Assignment", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Generate an assignment on Data Science"
            })
            response = generate_response("Generate an assignment on Data Science")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            st.rerun()
    
    with col3:
        if st.button("💻 Write Code", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Help me write code for data analysis"
            })
            response = generate_response("Help me write code for data analysis")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            st.rerun()
    
    with col4:
        if st.button("🔬 Deep Research", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Research the latest trends in AI"
            })
            response = generate_response("Research the latest trends in AI")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            st.rerun()
    
    with col5:
        if st.button("📚 Learn Basics", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": "Explain machine learning basics"
            })
            response = generate_response("Explain machine learning basics")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 💡 What can I help you with?")
    st.markdown("""
    - **Ask any Data Science question** - Get detailed explanations with examples
    - **Practice interviews** - Type "practice interview" to start mock interviews
    - **Generate assignments** - Type "generate assignment on [topic]"
    - **Write code** - Type "write code for [task]" or "implement [function]"
    - **Deep research** - Type "research [topic]" for comprehensive analysis
    - **Debug code** - Paste your code and ask "what's wrong with this code?"
    """)

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about Data Science, ML, Gen AI, or Agentic AI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            response = generate_response(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Floating footer
st.markdown('<div class="footer">Made with ❤️ by Himanshu | Powered by Groq AI</div>', unsafe_allow_html=True)