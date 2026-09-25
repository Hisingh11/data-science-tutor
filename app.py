import os
import re
import uuid

import streamlit as st

from utils.assignment import AssignmentManager
from utils.code_assistant import CodeAssistant
from utils.deep_research import DeepResearchEngine
from utils.image_recognition import analyze_image
from utils.interview import InterviewSystem
from utils.model_manager import get_model_manager
from utils.rag_engine import RAGEngine

INTERVIEW_TOPICS = {
    "data science": "Data Science Fundamentals",
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "generative": "Generative AI",
    "gen ai": "Generative AI",
    "agent": "Agentic AI",
    "python": "Python & Coding",
}
ASSIGNMENT_TOPICS = {
    "data science": "Data Science Fundamentals",
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "generative": "Generative AI",
    "gen ai": "Generative AI",
    "agent": "Agentic AI",
    "python": "Python",
}

st.set_page_config(
    page_title="Data Scientist BOT",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    #MainMenu, footer, [data-testid="stSidebar"],
    [data-testid="collapsedControl"], .stDeployButton { display: none; }
    header[data-testid="stHeader"] { background: transparent; height: 0; }
    .stApp { background: #f4f0fb; }
    h3, [data-testid="stHeading"] { color: #3d3470; }
    .main .block-container {
        max-width: 740px;
        padding-top: 1.6rem;
        padding-bottom: 6rem;
    }
    .stChatMessage { background: transparent; }
    [data-testid="stChatMessageContent"] {
        background: #ffffff;
        border: 1px solid #e3dcf3;
        border-left: 4px solid #5b4d8a;
        border-radius: 16px;
        padding: 0.85rem 1rem;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        background: #efeaf8;
        border-left-color: #1f8a84;
    }
    .stChatInput textarea {
        background: #ffffff !important;
        color: #241f33 !important;
        border: 1px solid #d9d0ee !important;
        border-radius: 14px !important;
    }
    .stChatInput textarea:focus {
        border-color: #5b4d8a !important;
        box-shadow: 0 0 0 1px #5b4d8a !important;
    }
    div.stButton > button {
        background: #ffffff;
        color: #3d3470;
        border: 1px solid #d9d0ee;
        border-radius: 999px;
        font-weight: 500;
    }
    div.stButton > button:hover {
        background: #5b4d8a;
        border-color: #5b4d8a;
        color: #ffffff;
    }
    button[aria-label="Upload files"] svg { display: none; }
    button[aria-label="Upload files"]::after {
        content: "📎";
        font-size: 1.05rem;
        line-height: 1;
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


def pick_topic(text, table, default):
    lowered = text.lower()
    for key, topic in table.items():
        if key in lowered:
            return topic
    return default


def pick_difficulty(text):
    lowered = text.lower()
    for level in ("beginner", "intermediate", "advanced"):
        if level in lowered:
            return level
    return "intermediate"


def interview_active():
    interview = st.session_state.interview_system
    return bool(interview.current_topic) and interview.questions_asked < len(interview.questions or [])


def format_interview_result(evaluation, nxt):
    strengths = evaluation.get("strengths") or []
    improvements = evaluation.get("improvements") or []
    lines = [
        f"**Score: {evaluation.get('score', 0)} / 10**",
        "",
        evaluation.get("feedback") or "",
    ]
    if strengths:
        lines += ["", "**What worked:** " + "; ".join(strengths)]
    if improvements:
        lines += ["", "**Improve:** " + "; ".join(improvements)]
    if evaluation.get("model_answer"):
        lines += ["", "**Model answer**", evaluation["model_answer"]]
    if nxt:
        asked = st.session_state.interview_system.questions_asked + 1
        total = len(st.session_state.interview_system.questions)
        lines += ["", f"**Question {asked} of {total}**", nxt]
    else:
        summary, _avg, _pct = st.session_state.interview_system.get_summary()
        lines += ["", summary]
        st.session_state.interview_system.current_topic = None
    return "\n".join(lines)


def run_interview(prompt):
    interview = st.session_state.interview_system
    if interview_active() and not prompt.lower().startswith("interview"):
        evaluation = interview.evaluate_answer(prompt.strip()) or {}
        nxt = interview.get_next_question()
        return format_interview_result(evaluation, nxt)
    topic = pick_topic(prompt, INTERVIEW_TOPICS, "Machine Learning")
    difficulty = pick_difficulty(prompt)
    interview.start_interview(topic, difficulty)
    question = interview.get_next_question()
    total = len(interview.questions)
    return (
        f"**{topic}** · {difficulty}\n\n"
        f"Five questions, scored out of 10. Say **stop interview** to end it.\n\n"
        f"**Question 1 of {total}**\n\n{question}"
    )


def run_assignment(prompt):
    manager = st.session_state.assignment_manager
    lowered = prompt.lower()
    if lowered.startswith("grade"):
        assignment = st.session_state.get("current_assignment") or {}
        assignment_id = assignment.get("assignment_id", "")
        body = re.sub(r"(?i)^grade(?:\s+this)?\s*:?\s*", "", prompt).strip()
        result = manager.grade_submission(assignment_id, body)
        if result.get("error"):
            return result["error"]
        strengths = "; ".join(result.get("strengths") or [])
        weak = "; ".join(result.get("weak_areas") or [])
        score = result.get("percentage")
        score_line = (
            f"**{result['earned_points']} / {result['total_points']} ({score}%)**"
            if score is not None
            else "**Score unavailable**"
        )
        return "\n\n".join(
            part for part in (
                score_line,
                result.get("overall_feedback") or "",
                f"**Strengths:** {strengths}" if strengths else "",
                f"**Work on:** {weak}" if weak else "",
            ) if part
        )
    topic = pick_topic(prompt, ASSIGNMENT_TOPICS, "Machine Learning")
    difficulty = pick_difficulty(prompt)
    count_match = re.search(r"(\d+)\s+questions", lowered)
    count = int(count_match.group(1)) if count_match else 6
    count = max(6, min(15, count))
    assignment, pdf_bytes = manager.generate_assignment(topic, difficulty, count, student_id="student")
    st.session_state.current_assignment = assignment
    st.session_state.download = {
        "name": f"{assignment['assignment_id']}.pdf" if pdf_bytes.startswith(b"%PDF") else f"{assignment['assignment_id']}.txt",
        "data": pdf_bytes,
        "mime": "application/pdf" if pdf_bytes.startswith(b"%PDF") else "text/plain",
    }
    lines = [
        f"**{assignment['topic']}** · {assignment['difficulty']}",
        f"{assignment['total_questions']} questions · {assignment['total_points']} points",
        "The PDF is ready below the chat. To grade it, send `grade this:` followed by your answers.",
        "",
    ]
    for question in assignment["questions"]:
        lines.append(f"**{question['id']}. {question['question']}**")
        lines.append(f"{question['type']} · {question['points']} points")
        lines.append("")
    return "\n".join(lines).strip()


def conversation_history():
    history = []
    for msg in st.session_state.messages:
        if msg.get("role") not in ("user", "assistant"):
            continue
        content = (msg.get("model_content") or msg.get("content") or "").strip()
        if content:
            history.append({"role": msg["role"], "content": content})
    return history


def with_history(prompt):
    prior = conversation_history()[-12:]
    if not prior:
        return prompt
    transcript = "\n\n".join(f"{msg['role']}: {msg['content'][:2500]}" for msg in prior)
    return f"Previous conversation:\n{transcript}\n\nCurrent request:\n{prompt}"


def run_code(prompt):
    prompt = with_history(prompt)
    assistant = st.session_state.code_assistant
    language = "sql" if " sql" in prompt.lower() else "r" if re.search(r"\br\b", prompt.lower()) else "python"
    if "```" in prompt or re.search(r"(?i)review|debug|fix this code", prompt):
        code = prompt
        fenced = re.search(r"```(?:\w+)?\n(.*?)```", prompt, re.DOTALL)
        if fenced:
            code = fenced.group(1)
        return assistant.check_code(code, language)
    generated = assistant.generate_code(prompt, language)
    return generated.get("full_response") or generated.get("code") or "I could not write that."


def run_research(prompt):
    prompt = with_history(prompt)
    engine = st.session_state.research_engine
    lowered = prompt.lower()
    if "fact-check" in lowered or "fact check" in lowered:
        claim = re.sub(r"(?i)fact-?check\s*:?\s*", "", prompt).strip()
        result = engine.fact_check(claim or prompt)
        return f"**Verdict:** {result.get('verdict', 'unverifiable')}\n\n{result.get('explanation', '')}"
    topic = re.sub(r"(?i)^(research|look up)\s*:?\s*", "", prompt).strip() or prompt
    result = engine.deep_research(topic)
    sources = result.get("sources") or []
    source_lines = "\n".join(f"- [{item['title']}]({item['url']})" for item in sources)
    parts = [f"**{result['topic']}**", result.get("key_takeaways") or "", result.get("report") or ""]
    if source_lines:
        parts.append("**Sources**\n" + source_lines)
    return "\n\n".join(part for part in parts if part)


def route(prompt):
    lowered = prompt.lower().strip()
    if lowered in {"stop interview", "end interview"}:
        st.session_state.interview_system.current_topic = None
        return "Interview ended. Ask anything else."
    if interview_active() and not re.match(r"(?i)(interview|assignment|research|look up|fact-?check|write code|write a function)", lowered):
        return run_interview(prompt)
    if re.search(r"(?i)interview|quiz me|mock interview", lowered):
        return run_interview(prompt)
    if re.search(r"(?i)assignment|practice questions|^grade\b", lowered):
        return run_assignment(prompt)
    if re.search(r"(?i)fact-?check|^research\b|^look up\b", lowered):
        return run_research(prompt)
    if "```" in prompt or re.search(r"(?i)write code|write a function|write python|generate code|review this code|debug this", lowered):
        return run_code(prompt)
    return None


def read_upload(uploaded):
    folder = "./data/uploads"
    os.makedirs(folder, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{os.path.basename(uploaded.name)}"
    path = os.path.join(folder, safe_name)
    with open(path, "wb") as handle:
        handle.write(uploaded.getbuffer())
    note = ""
    extra = ""
    limit = 24000
    name = safe_name.lower()
    if (uploaded.type or "").startswith("image/"):
        note = analyze_image(path, "Describe this image. Extract any code, diagram, or text you can see.")
        extra = f"\n\n[Image analysis]: {note}"
    elif name.endswith((".txt", ".py", ".md", ".csv", ".json", ".sql", ".r")):
        text = uploaded.getvalue().decode("utf-8", errors="replace")
        clipped = text[:limit]
        if len(text) > limit:
            clipped += "\n\n[File truncated for the model. The full file is saved.]"
        extra = f"\n\n[Attached file {uploaded.name}]:\n{clipped}"
    elif name.endswith(".pdf"):
        text = st.session_state.assignment_manager.extract_text_from_file(uploaded)
        extra = f"\n\n[Attached PDF {uploaded.name}]:\n{text[:limit]}"
    return path, note, extra


def answer_chat(prompt, user_content, image_note):
    history = conversation_history()
    retrieval_query = prompt
    if image_note and len(prompt.split()) < 6:
        retrieval_query = f"{prompt}\n{image_note[:400]}"
    retrieval = st.session_state.rag_engine.corrective_retrieve(
        retrieval_query,
        model=st.session_state.model_manager,
    )
    return st.write_stream(
        st.session_state.model_manager.stream_answer(
            user_content,
            history=history,
            context=retrieval["context"],
            temperature=0.6,
        )
    )


def render_header():
    title, action = st.columns([5, 1])
    with title:
        st.markdown("### Data Scientist BOT")
        st.caption("Ask, practice, or research. Files up to 50 MB. Made by Himanshu.")
    with action:
        if st.button("Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.interview_system.current_topic = None
            st.session_state.pop("download", None)
            st.session_state.pop("current_assignment", None)
            st.rerun()


def render_empty():
    st.markdown("Ask a question, or start from one of these.")
    suggestions = [
        "Explain the bias-variance tradeoff",
        "Interview me on machine learning, beginner",
        "Research retrieval-augmented generation",
        "Write a Python function that fills missing values",
    ]
    left, right = st.columns(2)
    for index, label in enumerate(suggestions):
        column = left if index % 2 == 0 else right
        with column:
            if st.button(label, use_container_width=True, key=f"suggest_{index}"):
                st.session_state.queued = label
                st.rerun()


def render_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for attachment in msg.get("attachments") or []:
                if str(attachment).lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")) and os.path.exists(attachment):
                    st.image(attachment, width=280)


ensure_core()
if "messages" not in st.session_state:
    st.session_state.messages = []

render_header()
if st.session_state.model_manager.init_error:
    st.error(st.session_state.model_manager.init_error)

queued = st.session_state.pop("queued", None)
if not st.session_state.messages and not queued:
    render_empty()
render_messages()

download = st.session_state.get("download")
if download:
    st.download_button(
        "Download assignment",
        data=download["data"],
        file_name=download["name"],
        mime=download["mime"],
    )

st.caption("Use the paperclip in the message box. Attachments up to 50 MB.")

submission = st.chat_input(
    "Message Data Scientist BOT",
    accept_file="multiple",
    max_upload_size=50,
    file_type=["png", "jpg", "jpeg", "gif", "webp", "pdf", "txt", "py", "md", "csv", "json", "sql", "r"],
)
uploads = []
if queued:
    prompt = queued
elif submission is not None:
    prompt = (submission.text or "").strip()
    uploads = list(submission.files or [])
else:
    prompt = ""
if not prompt and not uploads:
    st.stop()
if not prompt:
    prompt = "Look at the attached file and explain what it contains."

user_content = prompt
attachments = []
image_note = ""
for uploaded in uploads:
    with st.spinner("Reading the file..."):
        path, image_note, extra = read_upload(uploaded)
    attachments.append(path)
    user_content += extra

with st.chat_message("user"):
    st.markdown(prompt)
    for uploaded in uploads:
        st.caption(uploaded.name)
with st.chat_message("assistant"):
    special = None
    with st.spinner(""):
        special = route(user_content)
    if special is None:
        reply = answer_chat(prompt, user_content, image_note)
    else:
        reply = special
        st.markdown(reply)

st.session_state.messages.append({
    "role": "user",
    "content": prompt,
    "model_content": user_content,
    "attachments": attachments,
})
st.session_state.messages.append({
    "role": "assistant",
    "content": reply,
    "model_content": reply,
    "attachments": [],
})
st.rerun()
