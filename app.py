"""Data Scientist BOT — entry point.

Run with:  streamlit run app.py
Each feature lives on its own page in views/. Shared tools are in ui/state.py.
"""

import streamlit as st

st.set_page_config(
    page_title="Data Scientist BOT",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.state import core, reset_core  # noqa: E402  (set_page_config must run first)
from ui.theme import esc, inject_css  # noqa: E402


def _short_model_name(name: str) -> str:
    return esc((name or "").split("/")[-1])


def _status_html(model, ok: bool) -> str:
    dot = "#2dd4bf" if ok else "#f87171"
    rows = (
        f'<div class="sb-row"><span><span class="sb-dot" style="background:{dot}"></span>'
        f'Groq API</span><span>{"connected" if ok else "not set"}</span></div>'
        f'<div class="sb-row"><span>Tutor model</span><span>{_short_model_name(model.models["reasoning"])}</span></div>'
        f'<div class="sb-row"><span>Checker model</span><span>{_short_model_name(model.models["fast"])}</span></div>'
    )
    return f'<div class="sb-section">Status</div>{rows}'


def _usage_html(usage: dict) -> str:
    if not usage.get("calls"):
        return ""
    return (
        f'<div class="sb-row"><span>LLM calls</span><span>{usage["calls"]}</span></div>'
        f'<div class="sb-row"><span>Avg latency</span><span>{usage["avg_latency_s"]}s</span></div>'
        f'<div class="sb-row"><span>Tokens</span><span>{usage["prompt_tokens"] + usage["completion_tokens"]:,}</span></div>'
    )


inject_css()
tools = core()

pages = {
    "Learn": [
        st.Page("views/chat.py", title="Tutor Chat", icon=":material/forum:", default=True),
        st.Page("views/research.py", title="Research & Fact-check", icon=":material/travel_explore:"),
        st.Page("views/code_lab.py", title="Code Lab", icon=":material/code:"),
    ],
    "Practice": [
        st.Page("views/interview.py", title="Mock Interview", icon=":material/record_voice_over:"),
        st.Page("views/assignments.py", title="Assignments", icon=":material/assignment:"),
    ],
}
nav = st.navigation(pages, position="sidebar")

st.logo("assets/logo.svg", size="large", icon_image="assets/icon.svg")

with st.sidebar:
    model = tools.model
    ok = model.ready
    st.markdown(_status_html(model, ok), unsafe_allow_html=True)
    usage = model.usage_summary()
    usage_html = _usage_html(usage)
    if usage_html:
        st.markdown(usage_html, unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Session</div>', unsafe_allow_html=True)
    if st.button("New chat", icon=":material/add_comment:", width="stretch"):
        st.session_state.messages = []
        st.switch_page("views/chat.py")
    messages = st.session_state.get("messages") or []
    if messages:
        transcript = "\n\n".join(f"**{m['role'].title()}:** {m['content']}" for m in messages)
        st.download_button("Export chat", transcript, file_name="tutor_chat.md",
                           mime="text/markdown", icon=":material/download:", width="stretch")
    if not ok and st.button("Reload API key", icon=":material/refresh:", width="stretch"):
        reset_core()
        st.rerun()
    st.markdown('<div class="sb-section">About</div><div class="sb-sub">Made by Himanshu · Groq + LangGraph Self-RAG</div>',
                unsafe_allow_html=True)

if tools.model.init_error:
    st.warning(tools.model.init_error, icon=":material/key:")

nav.run()
