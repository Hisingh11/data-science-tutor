import inspect
import os
import re
import threading
import uuid

import streamlit as st

from utils.assignment import AssignmentManager
from utils.chat_history import clear_messages, load_messages, save_messages
from utils.code_assistant import CodeAssistant
from utils.deep_research import DeepResearchEngine
from utils.image_recognition import analyze_image
from utils.interview import InterviewSystem
from utils.model_manager import get_model_manager, trim_history
from utils.rag_engine import RAGEngine
from utils.self_rag_graph import SelfRAGEngine

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
        padding-right: 5.4rem !important;
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
    /* The pencil is a sibling of the chat input in the bottom bar.
       Anchor it to that bar so it sits inside the box at any width. */
    [data-testid="stBottomBlockContainer"] [data-testid="stVerticalBlock"] {
        position: relative;
    }
    .st-key-composer_edit {
        position: absolute !important;
        z-index: 5;
        right: 3.2rem;
        top: 50%;
        transform: translateY(-50%);
        width: 2.2rem !important;
        height: 2.2rem !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .st-key-composer_edit button {
        width: 2.2rem;
        height: 2.2rem;
        min-height: 2.2rem !important;
        padding: 0 !important;
        border: none !important;
        border-radius: 8px !important;
        background: transparent !important;
        color: #5b4d8a !important;
        font-size: 1.05rem;
    }
    .st-key-composer_edit button:hover {
        background: #efeaf8 !important;
        border: none !important;
        color: #3d3470 !important;
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
    st.session_state.self_rag = SelfRAGEngine(st.session_state.rag_engine, st.session_state.model_manager)
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


def leaves_interview(text):
    return bool(re.match(
        r"(?i)(interview|assignment|^grade\b|research|look up|fact-?check|write code|write a function|write python|generate code)",
        text.strip(),
    ))


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
        summary, _, _ = st.session_state.interview_system.get_summary()
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
        f"Five questions, scored out of 10. Click **Stop interview** or send `stop interview` to end early.\n\n"
        f"**Question 1 of {total}**\n\n{question}"
    )


def run_assignment(prompt, user_content):
    manager = st.session_state.assignment_manager
    lowered = prompt.lower()
    if lowered.startswith("grade"):
        assignment = st.session_state.get("current_assignment") or {}
        assignment_id = assignment.get("assignment_id", "")
        body = re.sub(r"(?i)^grade(?:\s+this)?\s*:?\s*", "", user_content).strip()
        result = manager.grade_submission(assignment_id, body, assignment=assignment)
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
    assignment, pdf_bytes = manager.generate_assignment(topic, difficulty, count)
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
    prior = trim_history(conversation_history(), total_chars=4000, message_chars=1000)
    if not prior:
        return prompt
    transcript = "\n\n".join(f"{msg['role']}: {msg['content']}" for msg in prior)
    return f"Previous conversation:\n{transcript}\n\nCurrent request:\n{prompt}"


def pick_language(text):
    lowered = text.lower()
    if re.search(r"\bsql\b", lowered):
        return "sql"
    if re.search(r"(?<![a-z])r(?![a-z])", lowered) and "python" not in lowered:
        return "r"
    return "python"


def run_code(prompt, user_content):
    assistant = st.session_state.code_assistant
    language = pick_language(prompt)
    reviewing = "```" in prompt or bool(re.search(r"(?i)review|debug|fix this code", prompt))
    if reviewing:
        code = user_content
        fenced = re.search(r"```(?:\w+)?\n(.*?)```", user_content, re.DOTALL)
        if fenced:
            code = fenced.group(1)
        return assistant.iter_check_code(code, language)
    return assistant.iter_generate_code(with_history(user_content), language)


def run_research(prompt, user_content):
    engine = st.session_state.research_engine
    lowered = prompt.lower()
    if "fact-check" in lowered or "fact check" in lowered:
        claim = re.sub(r"(?i)^fact-?check\s*:?\s*", "", user_content).strip()
        result = engine.fact_check(claim or user_content)
        return f"**Verdict:** {result.get('verdict', 'unverifiable')}\n\n{result.get('explanation', '')}"
    topic = re.sub(r"(?i)^(research|look up)\s*:?\s*", "", prompt).strip() or prompt
    context = with_history(user_content)

    def start(cancel):
        prepared = engine._prepare_research(topic, context, cancel)
        if cancel.is_set():
            return
        yield from engine.iter_prepared(prepared)

    return start


def route(prompt, user_content):
    lowered = prompt.lower().strip()
    if lowered in {"stop interview", "end interview"}:
        st.session_state.interview_system.current_topic = None
        return "Interview ended. Ask anything else."
    if interview_active() and not leaves_interview(lowered):
        return run_interview(user_content)
    if re.search(r"(?i)interview|quiz me|mock interview", lowered):
        return run_interview(prompt)
    if re.search(r"(?i)assignment|practice questions|^grade\b", lowered):
        return run_assignment(prompt, user_content)
    if re.search(r"(?i)fact-?check|^research\b|^look up\b", lowered):
        return run_research(prompt, user_content)
    if "```" in prompt or re.search(r"(?i)write code|write a function|write python|generate code|review this code|debug this", lowered):
        return run_code(prompt, user_content)
    return None


# Total attachment text sent to the model per message. Groq's free tier
# allows about 8,000 tokens per request, so files are shared out of this.
ATTACHMENT_CHARS = 10000


def read_upload(uploaded, limit=ATTACHMENT_CHARS):
    folder = "./data/uploads"
    os.makedirs(folder, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{os.path.basename(uploaded.name)}"
    path = os.path.join(folder, safe_name)
    with open(path, "wb") as handle:
        handle.write(uploaded.getbuffer())
    note = ""
    extra = ""
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
        clipped = text[:limit]
        if len(text) > limit:
            clipped += "\n\n[File truncated for the model. The full file is saved.]"
        extra = f"\n\n[Attached PDF {uploaded.name}]:\n{clipped}"
    return path, note, extra


def make_chat_stream(prompt, user_content, image_note):
    history = conversation_history()
    self_rag = st.session_state.self_rag
    retrieval_query = prompt
    if image_note and len(prompt.split()) < 6:
        retrieval_query = f"{prompt}\n{image_note[:400]}"

    def start(cancel):
        # The graph may self-correct up to five times before returning.
        yield from self_rag.iter_answer(retrieval_query, user_content, history, cancel)

    return start


def open_reply(special, cancel, prompt, user_content, image_note):
    if special is None:
        return make_chat_stream(prompt, user_content, image_note)(cancel)
    if isinstance(special, str):
        def once():
            yield special
        return once()
    if inspect.isgenerator(special):
        return special
    return special(cancel)


def track(generator, inflight):
    """Record each piece in session state so a stopped run keeps what was written."""
    for piece in generator:
        if not piece:
            continue
        text = piece if isinstance(piece, str) else str(piece)
        inflight["parts"].append(text)
        yield text


def commit_reply(user_record, reply, self_rag_log=None):
    message = {
        "role": "assistant",
        "content": reply,
        "model_content": reply,
        "attachments": [],
    }
    if self_rag_log:
        message["self_rag_log"] = self_rag_log
    st.session_state.messages.append(user_record)
    st.session_state.messages.append(message)
    save_messages(st.session_state.messages)


def recover_stopped():
    """A run that was stopped never reaches its commit; keep its partial reply."""
    inflight = st.session_state.pop("inflight", None)
    if not inflight:
        return
    text = "".join(inflight["parts"]).strip()
    reply = f"{text}\n\n*(Stopped.)*" if text else "*(Stopped.)*"
    commit_reply(inflight["user"], reply)


def stop_interview():
    interview = st.session_state.interview_system
    if not interview_active():
        return

    answered = interview.questions_asked
    total = len(interview.questions)
    progress = f"Progress: {answered} of {total} questions answered."
    if answered:
        progress += f" Average score: {interview.score / answered:.1f}/10."
    reply = f"Interview stopped. {progress}"
    interview.current_topic = None
    interview.current_difficulty = None
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "model_content": reply,
        "attachments": [],
    })
    save_messages(st.session_state.messages)


def render_header():
    title, action = st.columns([5, 1])
    with title:
        st.markdown("### Data Scientist BOT")
        st.caption("Ask, practice, or research. Files up to 50 MB. Made by Himanshu.")
    with action:
        if st.button("Clear", use_container_width=True):
            st.session_state.edit_index = None
            st.session_state.messages = []
            clear_messages()
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


def apply_edit(message, draft):
    old = message.get("content") or ""
    model = message.get("model_content") or old
    extra = model[len(old):] if model.startswith(old) else ""
    message["content"] = draft
    message["model_content"] = draft + extra


def latest_reply_index():
    messages = st.session_state.messages
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            return index
    return None


def render_self_rag_log(log):
    revised = "revised after self-check" if log.get("revised") else "kept as drafted"
    lines = [
        f"Self-correction retries: {log.get('retries_used', 0)} of {log.get('max_retries', 5)}",
        f"LLM checks used: {log.get('llm_checks', 0)}",
    ]
    if log.get("retrieved"):
        topics = ", ".join(item.get("topic", "note") for item in log["retrieved"][:5])
        lines.append(f"Notes used: {topics}")
    if log.get("web"):
        lines.append(f"Web notes used: {log['web']}")
    lines.append(f"Grounding: {log.get('support', 'not checked')}")
    lines.append(f"Usefulness: {log.get('usefulness', 'not checked')}")
    with st.expander(f"🧠 Self-RAG · {revised}"):
        st.markdown("\n".join(f"- {line}" for line in lines))


def render_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            log = msg.get("self_rag_log")
            if isinstance(log, dict):
                render_self_rag_log(log)
            for attachment in msg.get("attachments") or []:
                if str(attachment).lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")) and os.path.exists(attachment):
                    st.image(attachment, width=280)


def render_editor():
    messages = st.session_state.messages
    index = st.session_state.get("edit_index")
    if isinstance(index, int) and 0 <= index < len(messages):
        msg = messages[index]
        draft = st.text_area(
            "Edit the latest reply",
            value=msg.get("content") or "",
            key=f"draft_{index}",
            height=140,
        )
        save, again, cancel = st.columns(3)
        with save:
            if st.button("Save", key=f"save_{index}", use_container_width=True):
                text = draft.strip()
                if text:
                    apply_edit(msg, text)
                    save_messages(messages)
                st.session_state.edit_index = None
                st.rerun()
        with again:
            if msg.get("role") == "user" and st.button("Send again", key=f"again_{index}", use_container_width=True):
                text = draft.strip()
                if text:
                    st.session_state.messages = messages[:index]
                    save_messages(st.session_state.messages)
                    st.session_state.edit_index = None
                    st.session_state.queued = text
                    st.rerun()
        with cancel:
            if st.button("Cancel", key=f"cancel_{index}", use_container_width=True):
                st.session_state.edit_index = None
                st.rerun()


def render_pencil():
    """Pencil inside the message bar, beside the send button."""
    if st.session_state.get("edit_index") is not None:
        return
    target = latest_reply_index()
    if target is None:
        return
    bottom = getattr(st, "_bottom", None)
    if bottom is None:
        return
    with bottom:
        if st.button("✎", key="composer_edit", help="Edit the latest reply"):
            st.session_state.edit_index = target
            st.rerun()


def render_stop_interview_button():
    active_interview = interview_active()
    if not active_interview:
        return
    _, action = st.columns([4, 2])
    with action:
        if st.button(
            "Stop interview",
            key="stop_interview",
            help="End the active interview and save your progress.",
            use_container_width=True,
        ):
            stop_interview()
            st.rerun()


ensure_core()
if "messages" not in st.session_state:
    st.session_state.messages = load_messages()
recover_stopped()

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

render_editor()
render_stop_interview_button()

# "stop" turns the send arrow into a stop button while a reply is running.
submission = st.chat_input(
    "Message Data Scientist BOT",
    accept_file="multiple",
    max_upload_size=50,
    file_type=["png", "jpg", "jpeg", "gif", "webp", "pdf", "txt", "py", "md", "csv", "json", "sql", "r"],
    submit_mode="stop",
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
    render_pencil()
    st.stop()
if not prompt:
    prompt = "Look at the attached file and explain what it contains."

st.session_state.edit_index = None
user_record = {
    "role": "user",
    "content": prompt,
    "model_content": prompt,
    "attachments": [],
}
inflight = {"user": user_record, "parts": []}
st.session_state.inflight = inflight

with st.chat_message("user"):
    st.markdown(prompt)
    for uploaded in uploads:
        st.caption(uploaded.name)

user_content = prompt
image_notes = []
per_file_limit = max(1500, ATTACHMENT_CHARS // max(1, len(uploads)))
for uploaded in uploads:
    with st.spinner("Reading the file..."):
        path, image_note, extra = read_upload(uploaded, per_file_limit)
    user_record["attachments"].append(path)
    if image_note:
        image_notes.append(image_note)
    user_content += extra
user_record["model_content"] = user_content
image_note = "\n".join(image_notes)

with st.chat_message("assistant"):
    try:
        with st.spinner(""):
            special = route(prompt, user_content)
        generator = open_reply(special, threading.Event(), prompt, user_content, image_note)
        streamed = st.write_stream(track(generator, inflight))
        reply = (streamed if isinstance(streamed, str) else "".join(inflight["parts"])).strip()
        reply = reply or "I could not write that."
    except Exception as exc:
        reply = f"That request failed: {exc}\n\nTry again, or rephrase it."
        st.markdown(reply)

st.session_state.pop("inflight", None)
self_rag_log = st.session_state.self_rag.last_log
st.session_state.self_rag.last_log = None
commit_reply(user_record, reply, self_rag_log)
st.rerun()
