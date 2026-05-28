import streamlit as st
from PIL import Image
import io
import os
from datetime import datetime
from utils.model_manager import get_model_manager
from utils.rag_engine import RAGEngine
from utils.interview import InterviewSystem
from utils.assignment import AssignmentManager
from utils.code_assistant import CodeAssistant
from utils.deep_research import DeepResearchEngine
from utils.auth import UserAuth, ChatHistoryManager
import json

# Page config
st.set_page_config(
    page_title="Data Science Tutor - Personalized Learning",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton button {
        width: 100%;
        border-radius: 5px;
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    .chat-message-user {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .chat-message-assistant {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        cursor: pointer;
        transition: transform 0.3s;
        text-align: center;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .sidebar-footer {
        position: fixed;
        bottom: 20px;
        left: 20px;
        right: 20px;
        text-align: center;
        font-size: 12px;
        color: #666;
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
    st.session_state.auth = UserAuth()
    st.session_state.history_manager = ChatHistoryManager()
    st.session_state.current_mode = "chat"  # chat, interview, assignment, code, research
    st.session_state.chat_history = []
    
    # Load knowledge base
    with st.spinner("Loading knowledge base..."):
        st.session_state.rag_engine.load_initial_knowledge()

# Authentication UI
if 'user' not in st.session_state:
    # Login/Register UI
    st.markdown('<div class="main-header"><h1>🎓 Data Science Tutor</h1><p>Your Personalized AI Learning Assistant</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔐 Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", type="primary"):
            result = st.session_state.auth.login(login_username, login_password)
            if result["success"]:
                st.session_state.user = result["user"]
                # Load user's chat history
                st.session_state.chat_history = []
                conversations = st.session_state.history_manager.load_conversations(login_username)
                if conversations:
                    st.session_state.chat_history = conversations[-1]["conversation"]
                st.rerun()
            else:
                st.error(result["error"])
    
    with col2:
        st.subheader("📝 Register")
        reg_username = st.text_input("Username", key="reg_username")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_name = st.text_input("Full Name", key="reg_name")
        reg_email = st.text_input("Email", key="reg_email")
        
        if st.button("Register"):
            if reg_username and reg_password and reg_name and reg_email:
                result = st.session_state.auth.register(reg_username, reg_password, reg_name, reg_email)
                if result["success"]:
                    st.success("Registration successful! Please login.")
                else:
                    st.error(result["error"])
            else:
                st.warning("Please fill all fields")
    
    st.stop()

# Main App UI (Logged in)
# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.markdown(f"### 👋 Welcome, {st.session_state.user['name']}!")
    
    st.markdown("---")
    
    # Mode Selection
    st.markdown("### 🎯 Learning Mode")
    modes = {
        "chat": "💬 AI Chat",
        "interview": "🎯 Mock Interview",
        "assignment": "📝 Assignments",
        "code": "💻 Code Assistant",
        "research": "🔬 Deep Research"
    }
    
    for mode_key, mode_label in modes.items():
        if st.button(mode_label, key=f"mode_{mode_key}"):
            st.session_state.current_mode = mode_key
            st.rerun()
    
    st.markdown("---")
    
    # Chat History
    st.markdown("### 📜 Chat History")
    conversations = st.session_state.history_manager.load_conversations(st.session_state.user['username'])
    
    if conversations:
        for idx, conv in enumerate(reversed(conversations[-5:])):  # Show last 5
            date = conv['timestamp'][:10]
            preview = conv['conversation'][0]['content'][:30] if conv['conversation'] else "Empty"
            if st.button(f"📅 {date}: {preview}...", key=f"hist_{idx}"):
                st.session_state.chat_history = conv['conversation']
                st.rerun()
    
    st.markdown("---")
    
    # User Actions
    if st.button("🚪 Logout"):
        # Save current conversation before logout
        if st.session_state.chat_history:
            st.session_state.history_manager.save_conversation(
                st.session_state.user['username'], 
                st.session_state.chat_history
            )
        st.session_state.auth.logout()
        st.rerun()
    
    # API Status
    if st.session_state.model_manager.api_key:
        st.success("✅ API Ready")
    else:
        st.error("❌ API Key Missing")
    
    st.markdown('<div class="sidebar-footer">Made with ❤️ by Himanshu<br>v2.0 - Personalized Learning</div>', unsafe_allow_html=True)

# Main Content Area
st.markdown(f'<div class="main-header"><h1>🎓 {modes[st.session_state.current_mode]}</h1><p>Your Personalized AI Learning Assistant</p></div>', unsafe_allow_html=True)

# Chat Interface (Unified)
if st.session_state.current_mode == "chat":
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask me anything about Data Science, ML, Gen AI, or Agentic AI...")
    
    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Search RAG for context
                relevant_docs = st.session_state.rag_engine.search(user_input, n_results=3)
                context = "\n\n".join([doc['text'] for doc in relevant_docs])
                
                # Personalize response with user name
                personalized_prompt = f"""User Name: {st.session_state.user['name']}
User Query: {user_input}

As a personalized AI tutor, address the user by name and provide helpful responses.
Context: {context if context else 'No additional context'}
"""
                
                if context and len(context) > 100:
                    response = st.session_state.model_manager.generate_with_context(
                        personalized_prompt, context, "reasoning", 0.7
                    )
                else:
                    response = st.session_state.model_manager.generate(
                        personalized_prompt, "reasoning", 0.7
                    )
                
                st.markdown(response)
                
                # Offer follow-up actions
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if st.button("🎯 Practice Interview", key="quick_interview"):
                        st.session_state.current_mode = "interview"
                        st.rerun()
                with col2:
                    if st.button("📝 Generate Assignment", key="quick_assignment"):
                        st.session_state.current_mode = "assignment"
                        st.rerun()
                with col3:
                    if st.button("💻 Get Code", key="quick_code"):
                        st.session_state.current_mode = "code"
                        st.rerun()
                with col4:
                    if st.button("🔬 Deep Research", key="quick_research"):
                        st.session_state.current_mode = "research"
                        st.rerun()
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Auto-save conversation after each exchange
        st.session_state.history_manager.save_conversation(
            st.session_state.user['username'],
            st.session_state.chat_history
        )

# Mock Interview Mode
elif st.session_state.current_mode == "interview":
    st.markdown("### 🎯 Mock Interview Practice")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        topic = st.selectbox("Topic", ["Data Science Fundamentals", "Machine Learning", "Generative AI", "Agentic AI", "Python & Coding"])
        difficulty = st.selectbox("Difficulty", ["beginner", "intermediate", "advanced"])
        
        if st.button("Start New Interview", type="primary"):
            if st.session_state.interview_system.start_interview(topic, difficulty):
                st.session_state.interview_active = True
                st.session_state.current_topic = topic
                st.rerun()
    
    with col2:
        if st.session_state.get('interview_active', False):
            question_index = st.session_state.interview_system.questions_asked
            questions = st.session_state.interview_system.QUESTION_BANK.get(
                st.session_state.current_topic, {}
            ).get(difficulty, [])
            
            if question_index < len(questions):
                st.markdown(f"**Question {question_index + 1} of {len(questions)}**")
                st.markdown(f"### {questions[question_index]}")
                
                answer = st.text_area("Your Answer:", height=200)
                
                if st.button("Submit Answer", type="primary"):
                    evaluation = st.session_state.interview_system.evaluate_answer(answer)
                    if evaluation:
                        st.markdown(f"**Score:** {evaluation.get('score', 0)}/10")
                        st.markdown(f"**Feedback:** {evaluation.get('feedback', '')}")
                        st.markdown(f"**Model Answer:** {evaluation.get('model_answer', '')}")
                        st.rerun()
                
                if st.button("End Interview"):
                    st.session_state.interview_active = False
                    st.rerun()
            else:
                st.success("🎉 Interview Completed!")
                st.balloons()
                st.session_state.interview_active = False
        else:
            st.info("Select a topic and difficulty, then click 'Start New Interview'")

# Assignment Mode
elif st.session_state.current_mode == "assignment":
    st.markdown("### 📝 Generate Assignment")
    
    topic = st.selectbox("Topic", ["Data Science Fundamentals", "Machine Learning", "Generative AI", "Agentic AI", "Python"])
    difficulty = st.selectbox("Difficulty", ["beginner", "intermediate", "advanced"])
    num_questions = st.slider("Number of Questions", 5, 30, 10)
    
    if st.button("Generate Assignment", type="primary"):
        with st.spinner("Generating assignment..."):
            assignment = st.session_state.assignment_manager.generate_assignment(topic, difficulty, num_questions)
            st.success(f"Assignment generated! ID: {assignment.get('assignment_id', 'Unknown')}")
            
            with st.expander("View Assignment"):
                for q in assignment.get('questions', []):
                    st.markdown(f"**Q{q.get('id')}** ({q.get('type')}, {q.get('points', 10)} pts)")
                    st.markdown(q.get('question', ''))
                    st.markdown("---")
            
            st.download_button(
                label="Download JSON",
                data=json.dumps(assignment, indent=2),
                file_name=f"assignment_{assignment.get('assignment_id', 'unknown')}.json"
            )

# Code Assistant Mode
elif st.session_state.current_mode == "code":
    st.markdown("### 💻 Code Assistant")
    
    problem = st.text_area("Describe what you want to code:", height=150)
    language = st.selectbox("Language", ["python", "sql", "r", "javascript"])
    
    if st.button("Generate Code", type="primary"):
        with st.spinner("Generating code..."):
            result = st.session_state.code_assistant.generate_code(problem, language)
            st.markdown("### Generated Code")
            st.code(result['code'], language=language)
            st.markdown("### Explanation")
            st.markdown(result['full_response'])

# Deep Research Mode
elif st.session_state.current_mode == "research":
    st.markdown("### 🔬 Deep Research")
    
    research_topic = st.text_input("Enter a topic to research:")
    
    if st.button("Conduct Research", type="primary"):
        if research_topic:
            with st.spinner(f"Researching {research_topic}..."):
                research = st.session_state.research_engine.deep_research(research_topic)
                st.markdown("### Research Report")
                st.markdown(research['report'])
                st.markdown("### Key Takeaways")
                st.markdown(research['key_takeaways'])

# Footer
st.markdown("---")