"""Research & Fact-check: web-grounded briefings and claim verification."""

import streamlit as st

from ui.state import core, stream_into
from ui.theme import badge, hero, section, sources

tools = core()

hero("Research & Fact-check", "Deep research runs five parallel web searches and writes a cited briefing. "
     "Fact-check weighs a claim against live search evidence.", icon="🔎", eyebrow="Learn")

tab_research, tab_fact = st.tabs([":material/travel_explore: Deep research", ":material/fact_check: Fact-check"])

with tab_research:
    with st.form("research", border=False):
        topic = st.text_input("Topic", placeholder="e.g. Mixture of Experts, CUPED for A/B tests, SHAP values")
        context = st.text_area("Optional angle or context", height=80,
                               placeholder="e.g. I'm preparing for an ML engineer interview; focus on trade-offs.")
        go = st.form_submit_button("Research", type="primary", icon=":material/search:")
    if go and topic.strip():
        with st.status(f"Searching the web for “{topic.strip()}”…", expanded=False) as status:
            prepared = tools.research._prepare_research(topic.strip(), context.strip())
            status.update(label=f"Found {len(prepared['sources'])} sources · writing briefing…", state="complete")
        parts = []
        with st.container(border=True):
            st.write_stream(stream_into(tools.research.iter_prepared({**prepared, "sources": []}), parts))
        st.session_state.research_last = {"topic": topic.strip(), "text": "".join(parts), "sources": prepared["sources"]}
        st.rerun()
    last = st.session_state.get("research_last")
    if last:
        with st.container(border=True):
            st.markdown(last["text"])
        if last["sources"]:
            section(f"Sources ({len(last['sources'])})")
            sources(last["sources"])
        st.download_button("Download briefing (.md)",
                           last["text"] + "\n\n## Sources\n" + "\n".join(f"- [{s['title']}]({s['url']})" for s in last["sources"]),
                           file_name=f"research_{last['topic'][:30].replace(' ', '_')}.md", icon=":material/download:")

with tab_fact:
    with st.form("fact", border=False):
        claim = st.text_area("Claim to check", height=90,
                             placeholder="e.g. Random forests cannot overfit because they average many trees.")
        check = st.form_submit_button("Check claim", type="primary", icon=":material/fact_check:")
    if check and claim.strip():
        with st.spinner("Gathering evidence…"):
            st.session_state.fact_last = tools.research.fact_check(claim.strip())
    result = st.session_state.get("fact_last")
    if result:
        kind = {"supported": "ok", "contradicted": "bad", "mixed": "warn"}.get(result["verdict"], "neutral")
        st.markdown(badge(f"Verdict: {result['verdict'].title()}", kind), unsafe_allow_html=True)
        st.progress(result["confidence"] / 100, text=f"Confidence {result['confidence']}%")
        st.markdown(result.get("explanation") or "")
        if result.get("sources"):
            section("Evidence")
            sources(result["sources"])
