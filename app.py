import os
import streamlit as st

# For cloud deployment, use environment variables
if 'GROQ_API_KEY' not in st.secrets:
    # Try to get from environment
    api_key = os.environ.get('GROQ_API_KEY')
    if api_key:
        st.secrets['GROQ_API_KEY'] = api_key
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

st.set_page_config(
    page_title="Data Science Tutor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stButton button {
        width: 100%;
        border-radius: 5px;
    }
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .feedback-positive {
        background-color: #d4edda;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .feedback-negative {
        background-color: #f8d7da;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.model_manager = get_model_manager()
    st.session_state.rag_engine = RAGEngine()
    st.session_state.interview_system = InterviewSystem(st.session_state.model_manager)
    st.session_state.assignment_manager = AssignmentManager(st.session_state.model_manager)
    st.session_state.code_assistant = CodeAssistant(st.session_state.model_manager)
    st.session_state.research_engine = DeepResearchEngine(st.session_state.model_manager)
    st.session_state.current_topic = None
    st.session_state.interview_active = False
    st.session_state.chat_history = []
    
    with st.spinner("Loading knowledge base..."):
        count = st.session_state.rag_engine.load_initial_knowledge()
        st.session_state.kb_loaded = count

st.sidebar.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
st.sidebar.title("📚 Data Science Tutor")

nav_options = [
    "💬 AI Chat",
    "🎯 Mock Interview", 
    "📝 Assignments",
    "💻 Code Assistant",
    "🔬 Deep Research",
    "📖 Knowledge Base"
]

selected_nav = st.sidebar.radio("Navigation", nav_options)

st.sidebar.markdown("---")
if st.session_state.model_manager.api_key:
    st.sidebar.success("✅ API Ready")
else:
    st.sidebar.error("❌ GROQ_API_KEY missing")
    st.sidebar.info("Add your API key to .env file")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Topics Covered:**
- 📊 Data Science Fundamentals
- 🤖 Machine Learning
- 🧠 Generative AI  
- 🤝 Agentic AI
- 🐍 Python & Coding
- 📈 Statistics & Math
""")

def get_grade(percentage):
    if percentage >= 90:
        return "A+"
    elif percentage >= 80:
        return "A"
    elif percentage >= 70:
        return "B"
    elif percentage >= 60:
        return "C"
    else:
        return "F"

if selected_nav == "💬 AI Chat":
    st.markdown('<div class="main-header"><h1>💬 AI Tutor Chat</h1><p>Ask any Data Science question - get detailed answers with examples</p></div>', unsafe_allow_html=True)
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    user_query = st.chat_input("Ask your Data Science question here...")
    
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                relevant_docs = st.session_state.rag_engine.search(user_query, n_results=3)
                context = "\n\n".join([doc['text'] for doc in relevant_docs])
                
                if context and len(context) > 100:
                    response = st.session_state.model_manager.generate_with_context(
                        user_query, context, "reasoning", 0.7
                    )
                else:
                    response = st.session_state.model_manager.generate(
                        user_query, "reasoning", 0.7
                    )
                
                st.markdown(response)
                
                if relevant_docs:
                    with st.expander("📚 Sources from Knowledge Base"):
                        for i, doc in enumerate(relevant_docs):
                            st.markdown(f"**Source {i+1}** (relevance: {doc['relevance']:.2f})")
                            st.caption(doc['text'][:300] + "...")
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})

elif selected_nav == "🎯 Mock Interview":
    st.markdown('<div class="main-header"><h1>🎯 Mock Interview Practice</h1><p>Practice real interview questions with AI feedback</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Interview Setup")
        
        topic = st.selectbox(
            "Select Topic",
            ["Data Science Fundamentals", "Machine Learning", "Generative AI", "Agentic AI", "Python & Coding"]
        )
        
        difficulty = st.selectbox(
            "Difficulty Level",
            ["beginner", "intermediate", "advanced"]
        )
        
        if st.button("🎤 Start New Interview", type="primary"):
            if st.session_state.interview_system.start_interview(topic, difficulty):
                st.session_state.interview_active = True
                st.session_state.current_topic = topic
                st.rerun()
            else:
                st.error("Error starting interview. Please try again.")
        
        if st.session_state.interview_active:
            st.markdown("---")
            st.markdown(f"**Current Session:** {topic} ({difficulty})")
            st.markdown(f"**Questions Answered:** {st.session_state.interview_system.questions_asked}")
            st.markdown(f"**Current Score:** {st.session_state.interview_system.score}")
    
    with col2:
        if st.session_state.interview_active:
            st.subheader("Interview Session")
            
            question_index = st.session_state.interview_system.questions_asked
            questions = st.session_state.interview_system.QUESTION_BANK.get(
                st.session_state.current_topic, {}
            ).get(difficulty, [])
            
            if question_index < len(questions):
                current_question = questions[question_index]
                st.markdown(f"**Question {question_index + 1} of {len(questions)}**")
                st.markdown(f"### {current_question}")
                
                answer = st.text_area("Your Answer:", height=200, key="interview_answer")
                
                col_submit, col_end = st.columns(2)
                with col_submit:
                    if st.button("Submit Answer", type="primary"):
                        evaluation = st.session_state.interview_system.evaluate_answer(answer)
                        
                        if evaluation:
                            st.markdown("### 📊 Evaluation")
                            
                            score = evaluation.get('score', 0)
                            if score >= 7:
                                st.markdown(f'<div class="feedback-positive">✅ Score: {score}/10 - Good job!</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="feedback-negative">⚠️ Score: {score}/10 - Needs improvement</div>', unsafe_allow_html=True)
                            
                            st.markdown("**Feedback:**")
                            st.info(evaluation.get('feedback', 'No specific feedback'))
                            
                            st.markdown("**Model Answer:**")
                            st.markdown(evaluation.get('model_answer', 'Review the answer'))
                            
                            st.rerun()
                
                with col_end:
                    if st.button("❌ End Interview"):
                        summary, avg_score, percentage = st.session_state.interview_system.get_summary()
                        st.session_state.interview_active = False
                        st.markdown(summary)
                        st.balloons()
                        st.rerun()
            else:
                st.success("🎉 Interview Completed!")
                summary, avg_score, percentage = st.session_state.interview_system.get_summary()
                st.markdown(summary)
                st.balloons()
                
                if st.button("Start New Interview"):
                    st.session_state.interview_active = False
                    st.rerun()
        else:
            st.info("👈 Select a topic and difficulty, then click 'Start New Interview' to begin practicing.")

elif selected_nav == "📝 Assignments":
    st.markdown('<div class="main-header"><h1>📝 Assignments & Grading</h1><p>Complete comprehensive assignments and get AI-powered feedback</p></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📋 Generate Assignment", "📤 Submit Assignment", "📊 Your Progress"])
    
    with tab1:
        st.subheader("Generate New Assignment")
        
        assign_topic = st.selectbox("Topic", ["Data Science Fundamentals", "Machine Learning", "Generative AI", "Agentic AI", "Python"], key="assign_topic")
        assign_difficulty = st.selectbox("Difficulty", ["beginner", "intermediate", "advanced"], key="assign_diff")
        num_questions = st.slider("Number of Questions", 10, 50, 20, help="Generate 10-50 comprehensive questions")
        
        if st.button("📄 Generate Assignment", type="primary"):
            with st.spinner(f"Generating {num_questions} questions and creating PDF..."):
                assignment, pdf_bytes = st.session_state.assignment_manager.generate_assignment(
                    assign_topic, assign_difficulty, num_questions
                )
                
                st.session_state.current_assignment = assignment
                
                total_q = assignment.get('total_questions', num_questions)
                total_p = assignment.get('total_points', 0)
                
                st.success(f"✅ Assignment generated! ID: {assignment.get('assignment_id', 'Unknown')}")
                st.info(f"📊 Total Questions: {total_q} | Total Points: {total_p}")
                
                with st.expander("📖 Preview First 5 Questions"):
                    for q in assignment.get('questions', [])[:5]:
                        st.markdown(f"**Q{q.get('id')}** ({q.get('type')}, {q.get('points')} pts)")
                        st.markdown(f"{q.get('question')[:200]}...")
                        st.markdown("---")
                
                # Two columns for download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    if pdf_bytes:
                        st.download_button(
                            label="📥 Download PDF Assignment",
                            data=pdf_bytes,
                            file_name=f"assignment_{assignment.get('assignment_id', 'unknown')}.pdf",
                            mime="application/pdf",
                            type="primary"
                        )
                    else:
                        st.error("PDF generation failed. Please check reportlab installation.")
                
                with col2:
                    import json
                    st.download_button(
                        label="📥 Download JSON",
                        data=json.dumps(assignment, indent=2),
                        file_name=f"assignment_{assignment.get('assignment_id', 'unknown')}.json",
                        mime="application/json"
                    )
    
    with tab2:
        st.subheader("Submit Completed Assignment")
        
        assignment_id = st.text_input("Assignment ID", placeholder="Enter the assignment ID you received")
        uploaded_file = st.file_uploader("Upload your assignment (TXT file)", type=["txt"])
        text_submission = st.text_area("Or paste your answers here:", height=200)
        
        if st.button("Submit for Grading", type="primary"):
            if assignment_id:
                content = text_submission
                if uploaded_file:
                    content = st.session_state.assignment_manager.extract_text_from_file(uploaded_file)
                
                if content:
                    with st.spinner("Grading your submission..."):
                        grading = st.session_state.assignment_manager.grade_submission(assignment_id, content)
                        
                        if "error" not in grading:
                            st.markdown("### 📊 Grading Results")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Points", f"{grading.get('earned_points', 0)}/{grading.get('total_points', 0)}")
                            with col2:
                                st.metric("Percentage", f"{grading.get('percentage', 0):.1f}%")
                            with col3:
                                st.metric("Grade", get_grade(grading.get('percentage', 0)))
                            
                            st.markdown("**Overall Feedback:**")
                            st.info(grading.get('overall_feedback', 'No feedback provided'))
                            
                            if grading.get('strengths'):
                                st.markdown("**✅ Strengths:**")
                                for s in grading['strengths']:
                                    st.markdown(f"- {s}")
                            
                            if grading.get('weak_areas'):
                                st.markdown("**📚 Areas to Improve:**")
                                for w in grading['weak_areas']:
                                    st.markdown(f"- {w}")
                            
                            st.balloons()
                        else:
                            st.error(grading.get('error', 'Error grading submission'))
                else:
                    st.warning("Please provide your answers")
            else:
                st.warning("Please enter the Assignment ID")
    
    with tab3:
        st.subheader("Your Learning Progress")
        st.info("📈 Your progress is tracked across assignments. Complete more assignments to see detailed analytics!")

elif selected_nav == "💻 Code Assistant":
    st.markdown('<div class="main-header"><h1>💻 Code Assistant</h1><p>Generate code and debug errors</p></div>', unsafe_allow_html=True)
    
    mode = st.radio("Select Mode", ["Generate Code", "Debug Code"], horizontal=True)
    
    if mode == "Generate Code":
        problem = st.text_area("Describe what you want to code:", height=150)
        language = st.selectbox("Programming Language", ["python", "sql", "r"])
        
        if st.button("Generate Code", type="primary"):
            with st.spinner("Generating code..."):
                result = st.session_state.code_assistant.generate_code(problem, language)
                
                st.markdown("### Generated Code")
                st.code(result['code'], language=language)
                
                st.markdown("### Explanation")
                st.markdown(result['full_response'])
    
    elif mode == "Debug Code":
        code_input = st.text_area("Paste your code here:", height=300)
        
        if st.button("Debug Code", type="primary"):
            if code_input:
                with st.spinner("Analyzing code..."):
                    analysis = st.session_state.code_assistant.check_code(code_input)
                    st.markdown("### Code Analysis")
                    st.markdown(analysis)
            else:
                st.warning("Please paste some code to debug")

elif selected_nav == "🔬 Deep Research":
    st.markdown('<div class="main-header"><h1>🔬 Deep Research</h1><p>Research topics with web search and fact-checking</p></div>', unsafe_allow_html=True)
    
    research_mode = st.radio("Research Mode", ["Topic Research", "Fact Check"], horizontal=True)
    
    if research_mode == "Topic Research":
        research_topic = st.text_input("Enter a topic to research:")
        
        if st.button("Conduct Deep Research", type="primary"):
            if research_topic:
                with st.spinner(f"Researching {research_topic}..."):
                    research = st.session_state.research_engine.deep_research(research_topic)
                    
                    st.markdown("### 📄 Research Report")
                    st.markdown(research['report'])
                    
                    st.markdown("### 💡 Key Takeaways")
                    st.markdown(research['key_takeaways'])
            else:
                st.warning("Please enter a topic to research")
    
    elif research_mode == "Fact Check":
        claim = st.text_area("Enter a claim to fact-check:", height=100)
        
        if st.button("Verify Claim", type="primary"):
            if claim:
                with st.spinner("Searching and verifying..."):
                    result = st.session_state.research_engine.fact_check(claim)
                    
                    verdict = result.get('verdict', 'unknown')
                    if verdict == 'true':
                        st.success(f"✅ Verdict: TRUE (Confidence: {result.get('confidence', 0)}%)")
                    elif verdict == 'false':
                        st.error(f"❌ Verdict: FALSE (Confidence: {result.get('confidence', 0)}%)")
                    else:
                        st.info(f"❓ Verdict: {verdict}")
                    
                    st.markdown("### Evidence")
                    st.markdown(result.get('evidence', 'No evidence found'))
            else:
                st.warning("Please enter a claim to fact-check")

elif selected_nav == "📖 Knowledge Base":
    st.markdown('<div class="main-header"><h1>📖 Knowledge Base</h1><p>Curated Data Science learning resources</p></div>', unsafe_allow_html=True)
    
    st.info("The knowledge base contains curated content on Data Science, ML, Gen AI, and Agentic AI.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Science Fundamentals")
        st.markdown("""
        - Data Science Lifecycle (10 steps)
        - CRISP-DM methodology
        - Data cleaning and preprocessing
        - Exploratory Data Analysis (EDA)
        - Feature engineering techniques
        """)
        
        st.subheader("🤖 Machine Learning")
        st.markdown("""
        - Supervised vs Unsupervised Learning
        - Regression and Classification
        - Decision Trees & Random Forest
        - Gradient Boosting (XGBoost, LightGBM)
        - Neural Networks and Deep Learning
        """)
    
    with col2:
        st.subheader("🧠 Generative AI")
        st.markdown("""
        - Large Language Models (LLMs)
        - Transformer architecture
        - Attention mechanisms
        - Prompt engineering techniques
        - RAG (Retrieval Augmented Generation)
        """)
        
        st.subheader("🤝 Agentic AI")
        st.markdown("""
        - AI Agents and their components
        - ReAct (Reason + Act) pattern
        - Plan-and-Execute architecture
        - Multi-agent collaboration
        - Tool use and function calling
        """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Data Science Tutor v1.0 | Powered by Groq LLM API | Open Source</p>
</div>
""", unsafe_allow_html=True)