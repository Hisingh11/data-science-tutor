"""Mock Interview: five questions, rubric scoring, model answers, final report."""

import streamlit as st

from ui.state import core
from ui.theme import badges, esc, hero, score_ring, section, tiles
from utils.interview import InterviewSystem

tools = core()
iv: InterviewSystem = tools.interview
TOPICS = list(InterviewSystem.QUESTION_BANK.keys())
ICONS = {"Data Science Fundamentals": "📊", "Machine Learning": "🤖", "Generative AI": "✨",
         "Agentic AI": "🕹️", "Python & Coding": "🐍"}

hero("Mock Interview", "Five fresh questions per session, scored 0–10 against a rubric, with a model "
     "answer after every response and a report at the end.", icon="🎤", eyebrow="Practice")


def setup():
    pre = st.session_state.pop("iv_prefill", None) or {}
    st.markdown("##### Choose a track")
    default_topic = pre.get("topic") if pre.get("topic") in TOPICS else "Machine Learning"
    topic = st.radio("Topic", TOPICS, index=TOPICS.index(default_topic), horizontal=True,
                     format_func=lambda t: f"{ICONS.get(t, '')} {t}", label_visibility="collapsed")
    level = st.segmented_control("Difficulty", ["beginner", "intermediate", "advanced"],
                                 default=pre.get("difficulty", "intermediate"),
                                 format_func=str.title)
    st.write("")
    if st.button("Start interview", type="primary", icon=":material/play_arrow:"):
        with st.spinner("Preparing five fresh questions…"):
            iv.start_interview(topic, level or "intermediate")
            iv.get_next_question()
        st.session_state.iv_last = None
        st.rerun()

    history = st.session_state.get("iv_reports") or []
    if history:
        section("Previous sessions")
        st.dataframe(history, hide_index=True, width="stretch")


def feedback_panel(evaluation, question_no):
    section(f"Feedback on question {question_no}")
    left, right = st.columns([1, 4])
    with left:
        score_ring(evaluation.get("score", 0))
    with right:
        st.markdown(evaluation.get("feedback") or "")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**What worked**")
        for item in evaluation.get("strengths") or ["—"]:
            st.markdown(f"- {item}")
    with c2:
        st.markdown("**To improve**")
        for item in evaluation.get("improvements") or ["—"]:
            st.markdown(f"- {item}")
    if evaluation.get("model_answer"):
        with st.expander("Model answer", icon=":material/lightbulb:"):
            st.markdown(evaluation["model_answer"])


def active():
    total = iv.total
    current = iv.questions_asked + 1
    badges((iv.current_topic, "brand"), (iv.current_difficulty.title(), "neutral"),
           (f"Running score {iv.score}/{iv.questions_asked * 10}" if iv.questions_asked else "Not scored yet", "ok"))
    st.progress(iv.questions_asked / total, text=f"Question {current} of {total}")

    last = st.session_state.get("iv_last")
    if last:
        feedback_panel(last["evaluation"], last["number"])
        st.divider()

    question = iv.get_next_question()
    st.markdown(f'<div class="bigq">{esc(question)}</div>', unsafe_allow_html=True)
    st.write("")
    with st.form(f"answer_{iv.current_topic}_{iv.questions_asked}", clear_on_submit=True, border=False):
        answer = st.text_area("Your answer", height=200, placeholder="Answer as you would in a real interview. "
                              "Definitions, an example, and a trade-off score highest.")
        c1, c2, c3 = st.columns([2, 1.2, 1.2])
        submit = c1.form_submit_button("Submit answer", type="primary", icon=":material/send:", width="stretch")
        skip = c2.form_submit_button("Skip", icon=":material/skip_next:", width="stretch")
        end = c3.form_submit_button("End interview", icon=":material/stop:", width="stretch")
    if end:
        finish(stopped=True)
        st.rerun()
    if submit or skip:
        if submit and not answer.strip():
            st.warning("Write an answer first, or press Skip.")
            st.stop()
        with st.spinner("Scoring your answer…"):
            number = iv.questions_asked + 1
            evaluation = iv.evaluate_answer(answer.strip() if submit else "(skipped)") or iv._default_evaluation()
        st.session_state.iv_last = {"evaluation": evaluation, "number": number}
        if iv.questions_asked >= iv.total:
            finish()
        st.rerun()


def finish(stopped=False):
    answered = iv.questions_asked
    report = {
        "topic": iv.current_topic,
        "difficulty": iv.current_difficulty,
        "answered": answered,
        "score": iv.score,
        "average": round(iv.score / answered, 1) if answered else 0,
        "stopped_early": stopped,
    }
    st.session_state.iv_report = {**report, "items": [h for h in iv.interview_history if h.get("score") is not None]}
    st.session_state.setdefault("iv_reports", []).insert(0, report)
    iv.stop()


def report_view():
    rep = st.session_state.iv_report
    last = st.session_state.get("iv_last")
    if last and not rep.get("stopped_early"):
        feedback_panel(last["evaluation"], last["number"])
        st.divider()
    section("Interview report")
    pct = rep["score"] / (rep["answered"] * 10) * 100 if rep["answered"] else 0
    verdict = "Strong hire" if pct >= 80 else "Hire" if pct >= 65 else "Borderline" if pct >= 50 else "Keep practising"
    tiles([("Answered", f"{rep['answered']}/5", rep["topic"]),
           ("Average", f"{rep['average']}/10", rep["difficulty"].title()),
           ("Overall", f"{pct:.0f}%", verdict)])
    items = rep.get("items") or []
    if items:
        st.write("")
        st.bar_chart([{"Question": f"Q{n}", "Score": i["score"]} for n, i in enumerate(items, 1)],
                     x="Question", y="Score", y_label="Score (0-10)", color="#5b4bdb", height=220)
        for n, item in enumerate(items, 1):
            with st.expander(f"Q{n} · {item['score']}/10 · {item['question'][:80]}"):
                st.markdown(f"**Your answer:** {item.get('user_answer') or '—'}")
                st.markdown(f"**Feedback:** {item.get('ai_feedback') or '—'}")
                st.markdown(f"**Model answer:** {item.get('model_answer') or '—'}")
    if st.button("Start a new interview", type="primary", icon=":material/restart_alt:"):
        st.session_state.pop("iv_report", None)
        st.session_state.iv_last = None
        st.rerun()


if not tools.model.ready:
    st.info("Add GROQ_API_KEY to .env to run interviews.", icon=":material/key:")
if iv.active:
    active()
elif st.session_state.get("iv_report"):
    report_view()
else:
    setup()
