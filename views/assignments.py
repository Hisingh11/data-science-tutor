"""Assignments: generate a set with a PDF, answer per question, get graded."""

import glob
import json
import os

import streamlit as st

from ui.state import ASSIGNMENTS_DIR, core
from ui.theme import badges, esc, hero, score_ring, section, tiles

tools = core()
TOPICS = ["Data Science Fundamentals", "Machine Learning", "Generative AI", "Agentic AI", "Python"]

hero("Assignments", "Generate a mixed set of conceptual, applied and coding questions, download it as a "
     "PDF, then submit answers for per-question grading.", icon="📝", eyebrow="Practice")


def saved_assignments():
    rows = []
    for path in sorted(glob.glob(os.path.join(ASSIGNMENTS_DIR, "*.json")), key=os.path.getmtime, reverse=True):
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("questions") and data.get("assignment_id"):
                # Older files lack the totals; derive them so every row has one shape.
                data.setdefault("total_questions", len(data["questions"]))
                data.setdefault("total_points", sum(q.get("points", 10) for q in data["questions"]))
                data.setdefault("difficulty", "intermediate")
                data.setdefault("topic", "Assignment")
                for q in data["questions"]:
                    q.setdefault("type", "conceptual")
                    q.setdefault("points", 10)
                rows.append(data)
        except (OSError, ValueError):
            continue
    return rows


def question_cards(assignment):
    for q in assignment["questions"]:
        st.markdown(
            f'<div class="qcard {esc(q["type"])}"><div class="meta">Q{q["id"]} · {esc(q["type"])} · {q["points"]} pts</div>'
            f'<div class="q">{esc(q["question"])}</div></div>',
            unsafe_allow_html=True,
        )


tab_new, tab_submit, tab_history = st.tabs([":material/add_circle: Generate", ":material/grading: Submit & grade",
                                            ":material/history: Saved"])

with tab_new:
    pre = st.session_state.pop("asg_prefill", None) or {}
    c1, c2, c3 = st.columns([2, 1.4, 1.2])
    topic = c1.selectbox("Topic", TOPICS, index=TOPICS.index(pre["topic"]) if pre.get("topic") in TOPICS else 1)
    level = c2.selectbox("Difficulty", ["beginner", "intermediate", "advanced"],
                         index=["beginner", "intermediate", "advanced"].index(pre.get("difficulty", "intermediate")),
                         format_func=str.title)
    count = c3.number_input("Questions", min_value=3, max_value=15, value=6)
    if st.button("Generate assignment", type="primary", icon=":material/auto_awesome:"):
        assignment, pdf = tools.assignments.generate_assignment(topic, level, int(count))
        st.session_state.asg_current = assignment
        st.session_state.asg_pdf = pdf
        st.session_state.pop("asg_result", None)

    assignment = st.session_state.get("asg_current")
    if assignment:
        st.write("")
        tiles([("Topic", assignment["topic"], assignment["difficulty"].title()),
               ("Questions", assignment["total_questions"], "conceptual · applied · coding"),
               ("Points", assignment["total_points"], f"ID {assignment['assignment_id']}")])
        pdf = st.session_state.get("asg_pdf") or b""
        is_pdf = pdf.startswith(b"%PDF")
        st.write("")
        st.download_button("Download PDF" if is_pdf else "Download text", pdf,
                           file_name=f"{assignment['assignment_id']}.{'pdf' if is_pdf else 'txt'}",
                           mime="application/pdf" if is_pdf else "text/plain", icon=":material/download:")
        section("Questions")
        question_cards(assignment)
        st.info("When you're ready, open **Submit & grade** to answer each question.", icon=":material/arrow_forward:")

with tab_submit:
    options = saved_assignments()
    current = st.session_state.get("asg_current")
    if current and all(a["assignment_id"] != current["assignment_id"] for a in options):
        options.insert(0, current)
    if not options:
        st.info("Generate an assignment first.", icon=":material/info:")
    else:
        labels = {a["assignment_id"]: f"{a['topic']} · {a['difficulty']} · {a['total_questions']} Qs · {a['assignment_id']}"
                  for a in options}
        default = current["assignment_id"] if current else options[0]["assignment_id"]
        chosen_id = st.selectbox("Assignment", list(labels), index=list(labels).index(default),
                                 format_func=labels.get)
        chosen = next(a for a in options if a["assignment_id"] == chosen_id)
        mode = st.segmented_control("How do you want to submit?", ["Answer per question", "Upload a file"],
                                    default="Answer per question")
        answers = {}
        upload = None
        with st.form(f"submit_{chosen_id}", border=False):
            if mode == "Upload a file":
                upload = st.file_uploader("Answers (.txt or .pdf)", type=["txt", "pdf"])
            else:
                for q in chosen["questions"]:
                    answers[q["id"]] = st.text_area(f"Q{q['id']} · {q['question']}  ({q['points']} pts)",
                                                    key=f"ans_{chosen_id}_{q['id']}", height=110)
            graded = st.form_submit_button("Grade my submission", type="primary", icon=":material/grading:")
        if graded:
            if upload is not None:
                body = tools.assignments.extract_text_from_file(upload)
            else:
                body = "\n\n".join(f"Q{qid}: {text.strip()}" for qid, text in answers.items() if text.strip())
            if not body.strip():
                st.warning("Write at least one answer, or upload a file.")
            else:
                with st.spinner("Grading each question…"):
                    st.session_state.asg_result = tools.assignments.grade_submission(chosen_id, body, assignment=chosen)
                    st.session_state.asg_result["assignment_id"] = chosen_id

        result = st.session_state.get("asg_result")
        if result and result.get("assignment_id") == chosen_id:
            if result.get("error"):
                st.error(result["error"])
            elif result.get("earned_points") is None:
                st.warning(result.get("overall_feedback"))
            else:
                section("Result")
                left, right = st.columns([1, 4])
                with left:
                    score_ring(result["earned_points"], result["total_points"])
                with right:
                    st.markdown(f"#### {result['earned_points']} / {result['total_points']} points")
                    st.markdown(result.get("overall_feedback") or "")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Strengths**")
                    for s in result.get("strengths") or ["—"]:
                        st.markdown(f"- {s}")
                with c2:
                    st.markdown("**Work on**")
                    for s in result.get("weak_areas") or ["—"]:
                        st.markdown(f"- {s}")
                if result.get("per_question"):
                    section("Per question")
                    st.dataframe(
                        [{"Q": f"Q{r['id']}", "Score": f"{r['earned']}/{r['points']}",
                          "%": round(100 * r["earned"] / r["points"]) if r["points"] else 0,
                          "Feedback": r["feedback"]} for r in result["per_question"]],
                        hide_index=True, width="stretch",
                        column_config={"%": st.column_config.ProgressColumn("%", min_value=0, max_value=100, format="%d%%")},
                    )

with tab_history:
    rows = saved_assignments()
    if not rows:
        st.caption("No saved assignments yet.")
    for a in rows[:30]:
        with st.expander(f"{a['topic']} · {a['difficulty']} · {a['total_questions']} questions · {a.get('created_date', '')[:10]}"):
            badges((f"{a['total_points']} pts", "brand"), (a["assignment_id"], "neutral"))
            question_cards(a)
            pdf_path = os.path.join(ASSIGNMENTS_DIR, f"{a['assignment_id']}.pdf")
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as handle:
                    st.download_button("Download PDF", handle.read(), file_name=os.path.basename(pdf_path),
                                       mime="application/pdf", key=f"dl_{a['assignment_id']}")
