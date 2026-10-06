"""Code Lab: generate data-science code from a spec, or get a review of your own."""

import streamlit as st

from ui.state import core, stream_into
from ui.theme import hero

tools = core()
LANGS = {"Python": "python", "SQL": "sql", "R": "r"}

hero("Code Lab", "Describe what you need and get a plan plus production-style code, or paste your code "
     "for a bug-hunting review with a corrected version.", icon="💻", eyebrow="Learn")

tab_gen, tab_review = st.tabs([":material/auto_awesome: Generate", ":material/bug_report: Review & debug"])

with tab_gen:
    with st.form("gen", border=False):
        spec = st.text_area("What should the code do?", height=130,
                            placeholder="e.g. A function that takes a DataFrame, imputes numeric columns with the "
                                        "median and categoricals with the mode, and returns a report of what changed.")
        c1, c2 = st.columns([1, 3])
        lang = c1.selectbox("Language", list(LANGS))
        go = st.form_submit_button("Generate code", type="primary", icon=":material/play_arrow:")
    if go and spec.strip():
        parts = []
        with st.container(border=True):
            st.write_stream(stream_into(tools.code.iter_generate_code(spec.strip(), LANGS[lang]), parts))
        st.session_state.code_gen_last = "".join(parts)
    elif st.session_state.get("code_gen_last"):
        with st.container(border=True):
            st.markdown(st.session_state.code_gen_last)

with tab_review:
    with st.form("review", border=False):
        code = st.text_area("Paste your code", height=260, placeholder="def mean(xs):\n    return sum(xs) / len(xs)")
        c1, _ = st.columns([1, 3])
        lang_r = c1.selectbox("Language", list(LANGS), key="lang_r")
        go_r = st.form_submit_button("Review code", type="primary", icon=":material/bug_report:")
    if go_r and code.strip():
        parts = []
        with st.container(border=True):
            st.write_stream(stream_into(tools.code.iter_check_code(code, LANGS[lang_r]), parts))
        st.session_state.code_review_last = "".join(parts)
    elif st.session_state.get("code_review_last"):
        with st.container(border=True):
            st.markdown(st.session_state.code_review_last)
