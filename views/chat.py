"""Tutor Chat: Self-RAG answers, inline code help, research and fact-checks."""

import os
import re
import time

import streamlit as st

from ui.state import core, log_feedback, save_upload, stream_into
from ui.theme import badge, hero
from utils.image_recognition import analyze_image
from utils.model_manager import trim_history
from utils.router import (ASSIGNMENT_TOPICS, INTERVIEW_TOPICS, classify_intent, pick_difficulty,
                          pick_language, pick_topic)

tools = core()

# Total attachment text sent to the model per message (Groq free tier ≈ 8k tokens/request).
ATTACHMENT_CHARS = 10000
TEXT_TYPES = (".txt", ".py", ".md", ".csv", ".json", ".sql", ".r", ".ipynb")
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".gif", ".webp")

SUGGESTIONS = [
    (":material/balance:", "Bias–variance", "Explain the bias-variance tradeoff with an example"),
    (":material/query_stats:", "p-values", "What does a p-value actually tell me?"),
    (":material/hub:", "RAG", "How does retrieval-augmented generation work, and when does it fail?"),
    (":material/code:", "Code", "Write a Python function that imputes missing values per column type"),
    (":material/travel_explore:", "Research", "Research mixture of experts models"),
    (":material/fact_check:", "Fact-check", "Fact-check: accuracy is a good metric for fraud detection"),
]

if "messages" not in st.session_state:
    st.session_state.messages = []
messages = st.session_state.messages


# ----------------------------------------------------------------- helpers
def history_for_model():
    return [
        {"role": m["role"], "content": (m.get("model_content") or m.get("content") or "").strip()}
        for m in messages if m.get("role") in ("user", "assistant")
    ]


def with_history(prompt):
    prior = trim_history(history_for_model(), total_chars=4000, message_chars=1000)
    if not prior:
        return prompt
    transcript = "\n\n".join(f"{m['role']}: {m['content']}" for m in prior)
    return f"Previous conversation:\n{transcript}\n\nCurrent request:\n{prompt}"


def read_upload(uploaded, limit):
    path = save_upload(uploaded)
    name = uploaded.name.lower()
    note, extra = "", ""
    if (uploaded.type or "").startswith("image/") or name.endswith(IMAGE_TYPES):
        note = analyze_image(path, "Describe this image. Extract any code, diagram, chart values, or text you can see.")
        extra = f"\n\n[Image analysis of {uploaded.name}]: {note}"
    elif name.endswith(TEXT_TYPES):
        text = uploaded.getvalue().decode("utf-8", errors="replace")
        clipped = text[:limit] + ("\n\n[File truncated for the model.]" if len(text) > limit else "")
        extra = f"\n\n[Attached file {uploaded.name}]:\n{clipped}"
    elif name.endswith(".pdf"):
        text = tools.assignments.extract_text_from_file(uploaded)
        clipped = text[:limit] + ("\n\n[File truncated for the model.]" if len(text) > limit else "")
        extra = f"\n\n[Attached PDF {uploaded.name}]:\n{clipped}"
    return path, note, extra


def render_meta(msg):
    log = msg.get("self_rag_log")
    meta = msg.get("meta") or {}
    chips = []
    if meta.get("intent") and meta["intent"] != "tutor":
        chips.append(badge(meta["intent"].replace("_", " "), "brand"))
    if isinstance(log, dict):
        support = str(log.get("support", "not checked"))
        kind = "ok" if support == "claims grounded" else "warn" if "unsupported" in support else "neutral"
        chips.append(badge(f"Grounding: {support}", kind))
        useful = str(log.get("usefulness", "not checked"))
        chips.append(badge(f"Usefulness: {useful}", "ok" if useful.startswith("high") else "neutral"))
        if log.get("revised"):
            chips.append(badge("Self-corrected", "brand"))
        notes = [item.get("topic", "note") for item in (log.get("retrieved") or [])]
        if notes:
            chips.append(badge(f"{len(notes)} note(s): " + ", ".join(notes[:3]), "neutral"))
    if meta.get("seconds"):
        chips.append(badge(f"{meta['seconds']:.1f}s", "neutral"))
    if chips:
        st.markdown("".join(chips), unsafe_allow_html=True)


def render_messages():
    for index, msg in enumerate(messages):
        avatar = ":material/person:" if msg["role"] == "user" else ":material/school:"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            for attachment in msg.get("attachments") or []:
                if str(attachment).lower().endswith(IMAGE_TYPES) and os.path.exists(attachment):
                    st.image(attachment, width=280)
                elif attachment:
                    st.caption(f":material/attach_file: {os.path.basename(str(attachment))[9:]}")
            if msg["role"] == "assistant":
                render_meta(msg)
                for link in (msg.get("meta") or {}).get("links", []):
                    st.page_link(link["page"], label=link["label"], icon=link.get("icon"))
                rating = st.feedback("thumbs", key=f"fb_{index}_{len(msg['content'])}")
                if rating is not None and msg.get("rating") != rating:
                    msg["rating"] = rating
                    question = messages[index - 1]["content"] if index else ""
                    log_feedback({"question": question, "answer": msg["content"][:4000],
                                  "rating": "up" if rating == 1 else "down",
                                  "intent": (msg.get("meta") or {}).get("intent", "tutor")})


def render_empty():
    hero("What do you want to learn today?",
         "Ask any data science, statistics, ML, GenAI or Python question. Answers are grounded in "
         "a curated knowledge base and self-checked before you see them.",
         icon="🎓", eyebrow="Tutor chat")
    box = st.container(key="suggestions")
    cols = box.columns(3)
    for i, (icon, title, prompt) in enumerate(SUGGESTIONS):
        with cols[i % 3]:
            if st.button(f"**{title}**  \n{prompt}", key=f"sg_{i}", icon=icon, width="stretch"):
                st.session_state.queued = prompt
                st.rerun()


# ------------------------------------------------------------------ intents
def respond(prompt, user_content, image_note, status):
    """Return (generator_or_text, meta)."""
    intent = classify_intent(prompt)
    meta = {"intent": intent}

    if intent == "interview":
        st.session_state.iv_prefill = {"topic": pick_topic(prompt, INTERVIEW_TOPICS, "Machine Learning"),
                                       "difficulty": pick_difficulty(prompt)}
        pre = st.session_state.iv_prefill
        meta["links"] = [{"page": "views/interview.py", "label": "Open Mock Interview", "icon": ":material/record_voice_over:"}]
        return (f"Mock interviews have their own page with scoring, model answers and a final report. "
                f"I've pre-selected **{pre['topic']}** at **{pre['difficulty']}** level for you."), meta

    if intent in ("assignment", "grade"):
        st.session_state.asg_prefill = {"topic": pick_topic(prompt, ASSIGNMENT_TOPICS, "Machine Learning"),
                                        "difficulty": pick_difficulty(prompt)}
        meta["links"] = [{"page": "views/assignments.py", "label": "Open Assignments", "icon": ":material/assignment:"}]
        what = "grade your submission" if intent == "grade" else "generate a fresh assignment with a PDF"
        return f"The Assignments page can {what}, with per-question scores and feedback.", meta

    if intent == "fact_check":
        claim = re.sub(r"(?i)^(fact[- ]?check|verify( this| the claim)?)\s*:?\s*", "", user_content).strip()
        status.update(label="Searching the web for evidence…")
        result = tools.research.fact_check(claim or user_content)
        icon = {"supported": "✅", "contradicted": "❌", "mixed": "⚖️"}.get(result["verdict"], "❔")
        text = (f"{icon} **Verdict: {result['verdict'].title()}** · confidence {result['confidence']}%\n\n"
                f"{result.get('explanation', '')}")
        if result.get("sources"):
            text += "\n\n**Sources**\n" + "\n".join(f"- [{s['title']}]({s['url']})" for s in result["sources"])
        return text, meta

    if intent == "research":
        topic = re.sub(r"(?i)^(research|look up|deep dive( into| on)?|find sources( on| about)?)\s*:?\s*", "", prompt).strip() or prompt
        status.update(label=f"Running 5 parallel web searches on “{topic[:50]}”…")
        prepared = tools.research._prepare_research(topic, with_history(user_content))
        status.update(label="Writing the briefing…")
        return tools.research.iter_prepared(prepared), meta

    if intent == "code_review":
        language = pick_language(prompt)
        fenced = re.search(r"```(?:\w+)?\n(.*?)```", user_content, re.DOTALL)
        status.update(label="Reviewing your code…")
        return tools.code.iter_check_code(fenced.group(1) if fenced else user_content, language), meta

    if intent == "code_generate":
        status.update(label="Writing code…")
        return tools.code.iter_generate_code(with_history(user_content), pick_language(prompt)), meta

    # Default: LangGraph Self-RAG tutor.
    query = prompt if not (image_note and len(prompt.split()) < 6) else f"{prompt}\n{image_note[:400]}"
    tools.self_rag.on_step = lambda label: status.update(label=f"{label}…")
    answer = "".join(tools.self_rag.iter_answer(query, user_content, history_for_model()))
    tools.self_rag.on_step = None
    return answer, meta


# --------------------------------------------------------------------- page
queued = st.session_state.pop("queued", None)
if not messages and not queued:
    render_empty()
else:
    st.markdown("#### :material/forum: Tutor Chat")
render_messages()

if messages and messages[-1]["role"] == "assistant":
    left, _ = st.columns([1, 5])
    with left:
        if st.button("Regenerate", icon=":material/refresh:", key="regen"):
            last_user = messages[-2] if len(messages) >= 2 else None
            del messages[-2:]
            if last_user:
                st.session_state.queued = last_user["content"]
            st.rerun()

submission = st.chat_input(
    "Ask a question, paste code, or attach a file…",
    accept_file="multiple",
    max_upload_size=50,
    file_type=["png", "jpg", "jpeg", "gif", "webp", "pdf", "txt", "py", "md", "csv", "json", "sql", "r", "ipynb"],
)

uploads = []
if queued:
    prompt = queued
elif submission is not None:
    prompt = (submission.text or "").strip()
    uploads = list(submission.files or [])
else:
    st.stop()
if not prompt and not uploads:
    st.stop()
prompt = prompt or "Look at the attached file and explain what it contains."

user_record = {"role": "user", "content": prompt, "model_content": prompt, "attachments": []}
with st.chat_message("user", avatar=":material/person:"):
    st.markdown(prompt)
    for uploaded in uploads:
        st.caption(f":material/attach_file: {uploaded.name}")

started = time.perf_counter()
with st.chat_message("assistant", avatar=":material/school:"):
    user_content, notes = prompt, []
    per_file = max(1500, ATTACHMENT_CHARS // max(1, len(uploads)))
    try:
        with st.status("Thinking…", expanded=False) as status:
            for uploaded in uploads:
                status.update(label=f"Reading {uploaded.name}…")
                path, note, extra = read_upload(uploaded, per_file)
                user_record["attachments"].append(path)
                if note:
                    notes.append(note)
                user_content += extra
            user_record["model_content"] = user_content
            result, meta = respond(prompt, user_content, "\n".join(notes), status)
            status.update(label="Done", state="complete")
        if isinstance(result, str):
            st.markdown(result)
            reply = result
        else:
            parts = []
            streamed = st.write_stream(stream_into(result, parts))
            reply = (streamed if isinstance(streamed, str) else "".join(parts)).strip()
        reply = reply or "I could not write that. Try rephrasing."
    except Exception as exc:  # keep the chat alive on any tool failure
        meta = {"intent": "error"}
        reply = f"That request failed: {exc}\n\nTry again, or rephrase it."
        st.error(reply)

meta["seconds"] = time.perf_counter() - started
assistant = {"role": "assistant", "content": reply, "model_content": reply, "attachments": [], "meta": meta}
if meta.get("intent") == "tutor" and tools.self_rag.last_log:
    assistant["self_rag_log"] = tools.self_rag.last_log
tools.self_rag.last_log = None
messages.extend([user_record, assistant])
st.rerun()
