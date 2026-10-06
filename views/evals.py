"""LLM Evals dashboard: run the suites and inspect the latest results."""

import json
import os

import streamlit as st

from evals.run_evals import RESULTS, run, save, to_markdown
from evals.suites import ALL, LIVE, OFFLINE
from ui.state import FEEDBACK_PATH, core
from ui.theme import badge, badges, esc, hero, section

tools = core()
LATEST = os.path.join(RESULTS, "latest.json")
LABELS = {"routing": "Intent routing", "retrieval": "Retrieval (BM25)", "json_parsing": "JSON robustness",
          "tutor_qa": "Tutor answers", "interview_grading": "Interview grader", "code_gen": "Code generation",
          "fact_check": "Fact-check"}

hero("LLM Evals", "Golden datasets and automatic scoring for every part of the tutor: routing, retrieval, "
     "answer quality (LLM-as-judge), grader calibration, code pass@1 and fact-checking.",
     icon="📈", eyebrow="Quality")


def load_latest():
    try:
        with open(LATEST, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------ runner
with st.expander("Run evals", icon=":material/play_circle:", expanded=load_latest() is None):
    c1, c2 = st.columns([3, 1])
    chosen = c1.multiselect("Suites", list(ALL), default=list(OFFLINE), format_func=lambda s: LABELS.get(s, s))
    limit = c2.number_input("Max cases / suite", 0, 100, 0, help="0 = all cases")
    live = [s for s in chosen if s in LIVE]
    if live:
        st.caption(f":material/info: {', '.join(LABELS[s] for s in live)} call Groq and use tokens. "
                   "tutor_qa makes ~3 calls per case (answer, self-check, judge).")
        if not tools.model.ready:
            st.warning("Live suites need GROQ_API_KEY.", icon=":material/key:")
    if st.button("Run selected suites", type="primary", icon=":material/play_arrow:", disabled=not chosen
                 or (bool(live) and not tools.model.ready)):
        bar = st.progress(0.0, text="Starting…")
        lines = []

        def on_progress(name, i, n, case_id):
            bar.progress((i + 1) / n, text=f"{LABELS.get(name, name)} · case {i + 1}/{n} ({case_id})")

        with st.status("Running evals…", expanded=True) as status:
            report = run(chosen, int(limit), progress=on_progress, log=lambda m: lines.append(m))
            save(report)
            status.update(label=f"Finished in {report['seconds']}s", state="complete", expanded=False)
        st.rerun()

report = load_latest()
if not report:
    st.info("No results yet. Run the offline suites above (instant, no API key), or from a terminal: "
            "`python -m evals.run_evals`.", icon=":material/info:")
    st.stop()

# ----------------------------------------------------------------- summary
mode_kind = {"live": "ok", "offline": "brand", "mock": "warn"}.get(report["mode"], "neutral")
badges((f"{report['mode'].upper()} run", mode_kind), (report["run_at"], "neutral"),
       *([(f"judge: {report['judge_model']}", "neutral")] if report.get("judge_model") else []))
if report["mode"] == "mock":
    st.warning("This is a MOCK run (fake model) used to test the harness. Its scores are meaningless.")

suites = report["suites"]
section("Scorecard")
cols = st.columns(min(4, len(suites)) or 1)
for i, (name, res) in enumerate(suites.items()):
    head = res.get("headline") or ["error", "—", ""]
    status = res.get("status", "error")
    color = {"pass": "#0f9d8a", "fail": "#d1435b"}.get(status, "#c9840a")
    with cols[i % len(cols)]:
        st.markdown(
            f'<div class="tile" style="border-top:3px solid {color};margin-bottom:.8rem">'
            f'<div class="label">{esc(LABELS.get(name, name))}</div>'
            f'<div class="value">{esc(head[1])}{esc(head[2])}</div>'
            f'<div class="hint">{esc(head[0])} · gate {esc(res.get("threshold", "—"))} · '
            f'<b style="color:{color}">{esc(status.upper())}</b></div></div>',
            unsafe_allow_html=True,
        )

if "tutor_qa" in suites and suites["tutor_qa"].get("metrics"):
    m = suites["tutor_qa"]["metrics"]
    section("Tutor answer quality (LLM-as-judge, 1–5)")
    dims = ["correctness_1to5", "faithfulness_1to5", "relevance_1to5", "clarity_1to5"]
    st.bar_chart({"score": {d.replace("_1to5", ""): m.get(d) or 0 for d in dims}}, horizontal=True,
                 color="#5b4bdb", height=200)
    st.caption(f"Key-point recall {m.get('keypoint_recall')}% · injection resisted {m.get('injection_resisted')}% · "
               f"self-corrected {m.get('self_corrected_share')}% · latency p50 {m.get('latency_p50_s')}s / "
               f"p90 {m.get('latency_p90_s')}s")

# ---------------------------------------------------------------- drilldown
section("Drill down")
name = st.selectbox("Suite", list(suites), format_func=lambda s: LABELS.get(s, s))
res = suites[name]
if res.get("error"):
    st.error(res["error"])
metrics = res.get("metrics") or {}
if metrics:
    st.markdown("".join(badge(f"{k}: {v}", "neutral") for k, v in metrics.items()), unsafe_allow_html=True)
if res.get("confusions"):
    st.caption("Confusions: " + ", ".join(f"{k} ×{v}" for k, v in res["confusions"].items()))
cases = res.get("cases") or []
only_fail = st.toggle("Show failures only", value=bool([c for c in cases if not c.get("pass")]))
shown = [c for c in cases if not (only_fail and c.get("pass"))]
if shown:
    table = [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
              for k, v in c.items() if k not in ("answer", "log")} for c in shown]
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"pass": st.column_config.CheckboxColumn("pass")})
    detail = [c for c in shown if c.get("answer") or c.get("log")]
    if detail:
        pick = st.selectbox("Inspect case", [c["id"] for c in detail])
        case = next(c for c in detail if c["id"] == pick)
        with st.container(border=True):
            st.markdown(f"**Input:** {case.get('input', '')}")
            if case.get("answer"):
                st.markdown(case["answer"])
            if case.get("rationale"):
                st.caption(f"Judge: {case['rationale']}")
            if case.get("log"):
                st.code(case["log"])
else:
    st.success("No failures in this suite.")

# ------------------------------------------------------------ human signal
if os.path.exists(FEEDBACK_PATH):
    with open(FEEDBACK_PATH, encoding="utf-8") as handle:
        votes = [json.loads(line) for line in handle if line.strip()]
    if votes:
        up = sum(v.get("rating") == "up" for v in votes)
        section("Human feedback from chat (👍/👎)")
        st.markdown(f"{up} of {len(votes)} rated answers were helpful (**{100 * up / len(votes):.0f}%**). "
                    "Thumbs-down answers are good candidates for new golden cases.")
        with st.expander("Recent thumbs-down"):
            for v in [v for v in votes if v.get("rating") == "down"][-10:]:
                st.markdown(f"- **{v.get('question', '')[:120]}**")

st.download_button("Download report (.md)", to_markdown(report), file_name="eval_report.md",
                   icon=":material/download:")
