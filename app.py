import json
import os
import sqlite3
import uuid
import hashlib
from datetime import datetime, timezone

import streamlit as st

from utils.model_manager import get_model_manager
from utils.rag_engine import RAGEngine
from utils.interview import InterviewSystem
from utils.assignment import AssignmentManager
from utils.code_assistant import CodeAssistant
from utils.deep_research import DeepResearchEngine
from utils.image_recognition import analyze_image

DB_PATH = "./data/ds_tutor.db"
os.makedirs("./data", exist_ok=True)

TOPICS = [
    "Data Science Fundamentals",
    "Machine Learning",
    "Generative AI",
    "Agentic AI",
    "Python & Coding",
]
ASSIGNMENT_TOPICS = [
    "Data Science Fundamentals",
    "Machine Learning",
    "Generative AI",
    "Agentic AI",
    "Python",
]
DIFFICULTIES = ["beginner", "intermediate", "advanced"]


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
        c.execute(
            "INSERT INTO users (username, password_hash, full_name, created_at) VALUES (?,?,?,?)",
            (username, hash_password(password), full_name, datetime.now(timezone.utc)),
        )
        conn.commit()
        conn.close()
        return True, "Registered"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Username exists"


def login_user(username: str, password: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, username, full_name FROM users WHERE username=? AND password_hash=?",
        (username, hash_password(password)),
    )
    row = c.fetchone()
    conn.close()
    return row


def create_chat_session(user_id: int, title: str = "New Chat"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now(timezone.utc)
    c.execute(
        "INSERT INTO chat_sessions (user_id, title, created_at, updated_at) VALUES (?,?,?,?)",
        (user_id, title, now, now),
    )
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id


def update_session_title(session_id: int, title: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "UPDATE chat_sessions SET title = ? WHERE id = ? AND title = 'New Chat'",
        (title[:48], session_id),
    )
    conn.commit()
    conn.close()


def add_message(session_id: int, role: str, content: str, attachments: list = None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO chat_messages (session_id, role, content, timestamp, attachments) VALUES (?,?,?,?,?)",
        (session_id, role, content, datetime.now(timezone.utc), json.dumps(attachments or [])),
    )
    c.execute(
        "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
        (datetime.now(timezone.utc), session_id),
    )
    conn.commit()
    conn.close()


def get_session_messages(session_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT role, content, attachments FROM chat_messages WHERE session_id = ? ORDER BY timestamp",
        (session_id,),
    )
    rows = c.fetchall()
    conn.close()
    return [
        {"role": r[0], "content": r[1], "attachments": json.loads(r[2]) if r[2] else []}
        for r in rows
    ]


def get_user_sessions(user_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, title, updated_at FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,),
    )
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


st.set_page_config(
    page_title="Data Scientist BOT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stSidebar"] { background-color: #1e1e2f; }
    [data-testid="stSidebar"] * { color: #e0e0e0; }
    [data-testid="stSidebar"] .stButton button {
        background-color: #2a2a3f;
        color: white !important;
        border: 1px solid #444;
        border-radius: 0.5rem;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #3a3a4f;
        border-color: #667eea;
    }
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 5rem;
        max-width: 980px;
    }
    .stChatInput textarea {
        background-color: white !important;
        color: #1e1e2f !important;
        border-radius: 1.2rem !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


def ensure_core():
    if st.session_state.get("core_initialized"):
        return
    st.session_state.model_manager = get_model_manager()
    st.session_state.rag_engine = RAGEngine()
    st.session_state.rag_engine.load_initial_knowledge()
    st.session_state.interview_system = InterviewSystem(st.session_state.model_manager)
    st.session_state.assignment_manager = AssignmentManager(st.session_state.model_manager)
    st.session_state.code_assistant = CodeAssistant(st.session_state.model_manager)
    st.session_state.research_engine = DeepResearchEngine(st.session_state.model_manager)
    st.session_state.core_initialized = True


def create_guest_session():
    st.session_state.user_id = f"guest_{uuid.uuid4().hex[:8]}"
    st.session_state.username = "Guest"
    st.session_state.full_name = "Guest User"
    st.session_state.is_guest = True
    st.session_state.current_session_id = f"guest_session_{uuid.uuid4().hex[:8]}"
    st.session_state.messages = []
    st.rerun()


def save_exchange(user_content, attachments, reply):
    if st.session_state.get("is_guest"):
        st.session_state.messages.append(
            {"role": "user", "content": user_content, "attachments": attachments}
        )
        st.session_state.messages.append({"role": "assistant", "content": reply, "attachments": []})
        return
    session_id = st.session_state.current_session_id
    if not st.session_state.messages:
        title = user_content.strip().split("\n")[0][:48] or "New Chat"
        update_session_title(session_id, title)
    add_message(session_id, "user", user_content, attachments)
    add_message(session_id, "assistant", reply, [])
    st.session_state.messages = get_session_messages(session_id)


def render_login():
    st.markdown(
        "<h2 style='text-align:center;margin-top:4rem;'>Data Scientist BOT</h2>"
        "<p style='text-align:center;color:#555;'>Chat, mock interviews, assignments, code review, and research.</p>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1, 1])
    with left:
        if st.button("Continue as Guest", use_container_width=True):
            create_guest_session()
    with right:
        st.caption("Guest chats stay in this browser session only.")
    tab_login, tab_register = st.tabs(["Login", "Register"])
    with tab_login:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary"):
            user = login_user(username.strip(), password)
            if user:
                st.session_state.user_id, st.session_state.username, st.session_state.full_name = user
                st.session_state.is_guest = False
                sessions = get_user_sessions(st.session_state.user_id)
                if not sessions:
                    st.session_state.current_session_id = create_chat_session(st.session_state.user_id)
                else:
                    st.session_state.current_session_id = sessions[0]["id"]
                st.session_state.messages = get_session_messages(st.session_state.current_session_id)
                st.rerun()
            else:
                st.error("Invalid username or password.")
    with tab_register:
        new_username = st.text_input("Username", key="reg_username")
        new_password = st.text_input("Password", type="password", key="reg_password")
        fullname = st.text_input("Full name", key="fullname")
        if st.button("Create account"):
            if len(new_username.strip()) < 3 or len(new_password) < 6:
                st.warning("Use a username of at least 3 characters and a password of at least 6.")
            else:
                ok, msg = register_user(new_username.strip(), new_password, fullname.strip())
                if ok:
                    st.success("Account created. Log in to continue.")
                else:
                    st.error(msg)


def render_chat():
    messages = st.session_state.messages
    if not messages:
        st.markdown("### Ask a data science question")
        st.caption("Try: explain the bias-variance tradeoff, or attach a screenshot of an error.")
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for attachment in msg.get("attachments") or []:
                if str(attachment).lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                    if os.path.exists(attachment):
                        st.image(attachment, width=280)
                else:
                    st.caption(os.path.basename(str(attachment)))

    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0
    uploaded = st.file_uploader(
        "Attach an image, PDF, or text file",
        type=["png", "jpg", "jpeg", "gif", "webp", "pdf", "txt"],
        key=f"paperclip_{st.session_state.upload_key}",
    )
    if uploaded:
        st.caption(f"Attached {uploaded.name}. Send a message to include it.")
    prompt = st.chat_input("Message Data Scientist BOT...")
    if not prompt:
        return

    user_content = prompt
    attachments = []
    image_note = ""
    if uploaded:
        folder = (
            "./data/guest_uploads"
            if st.session_state.get("is_guest")
            else f"./data/uploads/{st.session_state.user_id}"
        )
        os.makedirs(folder, exist_ok=True)
        safe_name = f"{uuid.uuid4().hex[:8]}_{os.path.basename(uploaded.name)}"
        path = os.path.join(folder, safe_name)
        with open(path, "wb") as handle:
            handle.write(uploaded.getbuffer())
        attachments.append(path)
        if (uploaded.type or "").startswith("image/"):
            with st.spinner("Reading the image..."):
                image_note = analyze_image(path, "Describe this image. Extract any code or text you can see.")
            user_content += f"\n\n[Image analysis]: {image_note}"
        elif safe_name.lower().endswith(".txt"):
            user_content += "\n\n[Attached text]:\n" + uploaded.getvalue().decode("utf-8", errors="replace")[:6000]
        elif safe_name.lower().endswith(".pdf"):
            text = st.session_state.assignment_manager.extract_text_from_file(uploaded)
            user_content += f"\n\n[Attached PDF text]:\n{text[:6000]}"

    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
        if msg.get("role") in ("user", "assistant")
    ]
    full_prompt = user_content
    docs = st.session_state.rag_engine.search(full_prompt, n_results=3)
    context = "\n\n".join(doc["text"] for doc in docs)
    with st.spinner("Thinking..."):
        reply = st.session_state.model_manager.answer(
            full_prompt,
            history=history,
            context=context,
            model_type="reasoning",
            temperature=0.6,
        )
    save_exchange(user_content, attachments, reply)
    st.session_state.upload_key += 1
    st.rerun()


def render_interview():
    interview = st.session_state.interview_system
    st.markdown("### Mock interview")
    st.caption("Five questions, scored out of 10 each, with a model answer after every response.")
    topic = st.selectbox("Topic", TOPICS)
    difficulty = st.selectbox("Difficulty", DIFFICULTIES)
    if st.button("Start interview", type="primary"):
        if interview.start_interview(topic, difficulty):
            st.session_state.interview_feedback = None
            st.rerun()
        st.error("That topic is not available.")

    question = interview.get_next_question() if interview.current_topic else None
    if interview.current_topic and question:
        st.info(f"{interview.current_topic} · {interview.current_difficulty} · question {interview.questions_asked + 1} of {len(interview.questions)}")
        st.markdown(f"**{question}**")
        answer = st.text_area("Your answer", key=f"answer_{interview.questions_asked}", height=160)
        if st.button("Submit answer"):
            if not answer.strip():
                st.warning("Write an answer first.")
            else:
                with st.spinner("Reviewing your answer..."):
                    st.session_state.interview_feedback = interview.evaluate_answer(answer.strip())
                st.rerun()
    elif interview.current_topic:
        summary, _avg, percentage = interview.get_summary()
        st.success(summary)
        st.progress(min(percentage / 100, 1.0))

    feedback = st.session_state.get("interview_feedback")
    if feedback:
        st.metric("Score", f"{feedback.get('score', 0)} / 10")
        st.markdown(feedback.get("feedback", ""))
        strengths = feedback.get("strengths") or []
        improvements = feedback.get("improvements") or []
        if strengths:
            st.markdown("**What worked:** " + "; ".join(strengths))
        if improvements:
            st.markdown("**Improve:** " + "; ".join(improvements))
        if feedback.get("model_answer"):
            with st.expander("Model answer"):
                st.markdown(feedback["model_answer"])


def render_assignments():
    manager = st.session_state.assignment_manager
    st.markdown("### Assignments")
    topic = st.selectbox("Topic", ASSIGNMENT_TOPICS, key="assign_topic")
    difficulty = st.selectbox("Difficulty", DIFFICULTIES, key="assign_diff")
    count = st.slider("Questions", 6, 15, 9)
    if st.button("Generate assignment", type="primary"):
        with st.spinner("Building the assignment..."):
            assignment, pdf_bytes = manager.generate_assignment(
                topic, difficulty, count, student_id=str(st.session_state.user_id)
            )
        st.session_state.current_assignment = assignment
        st.session_state.current_assignment_pdf = pdf_bytes

    assignment = st.session_state.get("current_assignment")
    if assignment:
        st.markdown(
            f"**{assignment['topic']}** · {assignment['difficulty']} · "
            f"{assignment['total_questions']} questions · {assignment['total_points']} points"
        )
        st.caption(f"Assignment ID: {assignment['assignment_id']}")
        pdf_bytes = st.session_state.get("current_assignment_pdf") or b""
        if pdf_bytes.startswith(b"%PDF"):
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"{assignment['assignment_id']}.pdf",
                mime="application/pdf",
            )
        else:
            st.download_button(
                "Download text",
                data=pdf_bytes,
                file_name=f"{assignment['assignment_id']}.txt",
                mime="text/plain",
            )
        for question in assignment["questions"]:
            st.markdown(
                f"**{question['id']}. {question['question']}**  \n"
                f"{question['type']} · {question['points']} points"
            )

    st.markdown("#### Grade a submission")
    assignment_id = st.text_input(
        "Assignment ID",
        value=(assignment or {}).get("assignment_id", ""),
    )
    submission = st.text_area("Paste your answers", height=180)
    answer_file = st.file_uploader("Or upload answers", type=["txt", "pdf"], key="grade_upload")
    if st.button("Grade"):
        file_text = manager.extract_text_from_file(answer_file) if answer_file else ""
        with st.spinner("Grading..."):
            result = manager.grade_submission(assignment_id.strip(), submission, file_text)
        if result.get("error"):
            st.error(result["error"])
        else:
            if result.get("percentage") is not None:
                st.metric("Score", f"{result['earned_points']} / {result['total_points']} ({result['percentage']}%)")
            st.markdown(result.get("overall_feedback", ""))
            if result.get("strengths"):
                st.markdown("**Strengths:** " + "; ".join(result["strengths"]))
            if result.get("weak_areas"):
                st.markdown("**Work on:** " + "; ".join(result["weak_areas"]))


def render_code():
    assistant = st.session_state.code_assistant
    st.markdown("### Code assistant")
    st.caption("Generates and reviews code. It does not run the code on the server.")
    language = st.selectbox("Language", ["python", "sql", "r"])
    problem = st.text_area("What should the code do?", height=120, placeholder="Write a function that ...")
    if st.button("Generate code", type="primary"):
        if not problem.strip():
            st.warning("Describe the problem first.")
        else:
            with st.spinner("Writing code..."):
                st.session_state.generated_code = assistant.generate_code(problem.strip(), language)
    generated = st.session_state.get("generated_code")
    if generated:
        st.markdown(generated.get("full_response") or generated.get("code") or "")

    st.markdown("#### Review code")
    code = st.text_area("Paste code to review", height=180, key="review_code")
    if st.button("Review"):
        if not code.strip():
            st.warning("Paste some code first.")
        else:
            with st.spinner("Reviewing..."):
                st.session_state.code_review = assistant.check_code(code, language)
    if st.session_state.get("code_review"):
        st.markdown(st.session_state.code_review)


def render_research():
    engine = st.session_state.research_engine
    st.markdown("### Research")
    topic = st.text_input("Topic", placeholder="retrieval augmented generation")
    claim = st.text_input("Claim to fact-check", placeholder="Dropout always improves accuracy")
    col1, col2 = st.columns(2)
    with col1:
        run_research = st.button("Research topic", type="primary")
    with col2:
        run_check = st.button("Fact-check claim")
    if run_research:
        if not topic.strip():
            st.warning("Enter a topic.")
        else:
            with st.spinner("Searching and writing a briefing..."):
                st.session_state.research_result = engine.deep_research(topic.strip())
    if run_check:
        if not claim.strip():
            st.warning("Enter a claim.")
        else:
            with st.spinner("Checking the claim..."):
                st.session_state.fact_result = engine.fact_check(claim.strip())

    research = st.session_state.get("research_result")
    if research:
        st.markdown(f"### {research['topic']}")
        st.markdown(research.get("key_takeaways") or "")
        st.markdown(research.get("report") or "")
        sources = research.get("sources") or []
        if sources:
            st.markdown("**Sources**")
            for source in sources:
                st.markdown(f"- [{source['title']}]({source['url']})")
    fact = st.session_state.get("fact_result")
    if fact:
        st.markdown(f"**Verdict:** {fact.get('verdict', 'unverifiable')}")
        st.markdown(fact.get("explanation", ""))


def render_sidebar():
    guest = st.session_state.get("is_guest")
    name = st.session_state.get("full_name") or st.session_state.get("username") or "User"
    st.markdown(f"### {name}")
    if guest:
        st.caption("Guest mode. Chat history is not saved.")
    mode = st.radio(
        "Section",
        ["Chat", "Interview", "Assignments", "Code", "Research"],
        label_visibility="collapsed",
    )
    st.session_state.app_mode = mode
    if mode == "Chat" and not guest:
        if st.button("New chat", use_container_width=True):
            new_id = create_chat_session(st.session_state.user_id, "New Chat")
            st.session_state.current_session_id = new_id
            st.session_state.messages = []
            st.rerun()
        st.markdown("**History**")
        for sess in get_user_sessions(st.session_state.user_id):
            label = sess["title"] or "New Chat"
            col_a, col_b = st.columns([4, 1])
            with col_a:
                if st.button(label[:28], key=f"session_{sess['id']}", use_container_width=True):
                    st.session_state.current_session_id = sess["id"]
                    st.session_state.messages = get_session_messages(sess["id"])
                    st.rerun()
            with col_b:
                if st.button("×", key=f"del_{sess['id']}"):
                    delete_session(sess["id"])
                    remaining = get_user_sessions(st.session_state.user_id)
                    if remaining:
                        st.session_state.current_session_id = remaining[0]["id"]
                        st.session_state.messages = get_session_messages(remaining[0]["id"])
                    else:
                        st.session_state.current_session_id = create_chat_session(st.session_state.user_id)
                        st.session_state.messages = []
                    st.rerun()
    elif mode == "Chat" and guest:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    if st.button("Log out", use_container_width=True):
        for key in ["user_id", "username", "full_name", "messages", "current_session_id", "is_guest"]:
            st.session_state.pop(key, None)
        st.rerun()


if "user_id" not in st.session_state:
    render_login()
    st.stop()

ensure_core()
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    render_sidebar()

if st.session_state.model_manager.init_error:
    st.error(st.session_state.model_manager.init_error)

mode = st.session_state.get("app_mode", "Chat")
if mode == "Interview":
    render_interview()
elif mode == "Assignments":
    render_assignments()
elif mode == "Code":
    render_code()
elif mode == "Research":
    render_research()
else:
    render_chat()

st.caption("Made by Himanshu")
