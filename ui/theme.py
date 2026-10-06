"""Visual layer: one stylesheet plus a few small HTML helpers used by every page."""

import html

import streamlit as st

BRAND = "Data Scientist BOT"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --ink: #1d1b2e;
  --ink-2: #4a4766;
  --muted: #7b7896;
  --line: #e6e3f3;
  --surface: #ffffff;
  --canvas: #f7f6fc;
  --brand: #5b4bdb;
  --brand-2: #8b5cf6;
  --brand-soft: #efecff;
  --teal: #0f9d8a;
  --amber: #c9840a;
  --rose: #d1435b;
  --radius: 14px;
  --shadow: 0 1px 2px rgba(29,27,46,.04), 0 4px 16px rgba(29,27,46,.06);
}

html, body, [class*="css"], .stMarkdown, .stTextInput, .stTextArea, button, input, textarea, select {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
code, pre, .stCode, textarea.mono { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }

#MainMenu, footer, .stDeployButton, [data-testid="stAppDeployButton"] { display: none !important; }
[data-testid="stSidebarHeader"] img, [data-testid="stLogo"] { height: 2.6rem !important; max-width: 100%; }
.st-key-suggestions button { justify-content: flex-start; text-align: left; min-height: 4.4rem; padding: .7rem .9rem; }
.st-key-suggestions button p { white-space: normal; text-align: left; }
.st-key-suggestions button:hover { border-color: var(--brand); background: #fbfaff; }
header[data-testid="stHeader"] { background: transparent; }
.stApp { background: var(--canvas); }
.main .block-container, [data-testid="stMainBlockContainer"] {
  max-width: 980px; padding-top: 2rem; padding-bottom: 6rem;
}
h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.015em; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #17142e 0%, #221c46 100%);
  border-right: none;
}
[data-testid="stSidebar"] * { color: #d9d6f2; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #fff; }
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] {
  border-radius: 10px; margin: 1px 0; padding: .35rem .6rem;
}
[data-testid="stSidebarNav"] a:hover, [data-testid="stSidebarNavLink"]:hover { background: rgba(255,255,255,.07); }
[data-testid="stSidebarNav"] a[aria-current="page"], [data-testid="stSidebarNavLink"][aria-current="page"] {
  background: rgba(139,92,246,.28);
}
[data-testid="stSidebarNav"] a[aria-current="page"] span, [data-testid="stSidebarNavLink"][aria-current="page"] span { color: #fff; font-weight: 600; }
[data-testid="stSidebar"] .stButton > button, [data-testid="stSidebar"] .stDownloadButton > button {
  background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.14); color: #eceaff;
  border-radius: 10px; font-weight: 500;
}
[data-testid="stSidebar"] .stButton > button:hover, [data-testid="stSidebar"] .stDownloadButton > button:hover {
  background: rgba(255,255,255,.14); border-color: rgba(255,255,255,.3); color: #fff;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.1); }
.sb-brand { display:flex; align-items:center; gap:.7rem; padding:.2rem 0 .9rem; }
.sb-logo {
  width: 40px; height: 40px; border-radius: 12px; display:grid; place-items:center;
  background: linear-gradient(135deg, var(--brand), var(--brand-2)); font-size: 1.25rem;
  box-shadow: 0 6px 18px rgba(91,75,219,.45);
}
.sb-title { font-weight: 800; font-size: 1.02rem; color: #fff !important; line-height: 1.1; }
.sb-sub { font-size: .74rem; color: #a7a3cc !important; }
.sb-section { font-size: .68rem; text-transform: uppercase; letter-spacing: .09em; color: #8d89b8 !important; margin: .9rem 0 .35rem; font-weight: 600; }
.sb-row { display:flex; justify-content:space-between; font-size:.8rem; padding:.18rem 0; }
.sb-row span:last-child { color:#fff !important; font-weight:600; font-family:'JetBrains Mono', monospace; font-size:.74rem; }
.sb-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:.4rem; }

/* ---------- Buttons & inputs ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius: 10px; font-weight: 600; border: 1px solid var(--line);
  transition: all .15s ease;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] {
  background: linear-gradient(135deg, var(--brand), var(--brand-2)); border: none; color: #fff;
  box-shadow: 0 4px 14px rgba(91,75,219,.3);
}
button[data-testid^="stBaseButton-primary"]:hover, .stButton > button[kind="primary"]:hover { filter: brightness(1.06); transform: translateY(-1px); }
.stButton > button[kind="secondary"]:hover { border-color: var(--brand); color: var(--brand); }
.stTextArea textarea, .stTextInput input {
  border-radius: 12px !important; background: var(--surface) !important;
}
[data-testid="stChatInput"] { border-radius: 16px !important; box-shadow: var(--shadow); }
[data-testid="stTabs"] button[role="tab"] p { font-weight: 600; }

/* ---------- Chat ---------- */
[data-testid="stChatMessage"] { background: transparent; padding: .35rem 0; gap: .75rem; }
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
  background: var(--surface); border: 1px solid var(--line); border-radius: 16px;
  padding: .9rem 1.1rem; box-shadow: var(--shadow);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
  background: var(--brand-soft); border-color: #dcd6ff;
}

/* ---------- Custom blocks ---------- */
.hero {
  position: relative; overflow: hidden; border-radius: 20px; padding: 1.6rem 1.8rem;
  background: radial-gradient(1200px 300px at 100% 0%, rgba(139,92,246,.35), transparent 60%),
              linear-gradient(135deg, #1d1940 0%, #2c2470 60%, #3b2f97 100%);
  color: #fff; margin-bottom: 1.3rem; box-shadow: 0 10px 30px rgba(43,34,110,.25);
}
.hero .eyebrow { font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color:#c5bdff; font-weight:700; }
.hero h1 { color:#fff; font-size:1.75rem; margin:.25rem 0 .35rem; font-weight:800; padding:0; }
.hero p { color:#d9d4ff; margin:0; font-size:.96rem; max-width: 640px; }
.hero .icon { position:absolute; right:1.6rem; top:50%; transform:translateY(-50%); font-size:3.6rem; opacity:.9; }

.card {
  background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 1rem 1.15rem; box-shadow: var(--shadow); height: 100%;
}
.card h4 { margin: 0 0 .3rem; font-size: .98rem; }
.card p { margin: 0; color: var(--ink-2); font-size: .88rem; }
.card .k { font-size: 1.35rem; margin-bottom: .35rem; }

.tile { background: var(--surface); border:1px solid var(--line); border-radius: var(--radius); padding:.85rem 1rem; box-shadow: var(--shadow); }
.tile .label { font-size:.72rem; text-transform:uppercase; letter-spacing:.07em; color:var(--muted); font-weight:600; }
.tile .value { font-size:1.6rem; font-weight:800; color:var(--ink); line-height:1.2; margin-top:.15rem; }
.tile .hint { font-size:.78rem; color:var(--muted); }

.badge { display:inline-flex; align-items:center; gap:.3rem; padding:.18rem .6rem; border-radius:999px;
  font-size:.75rem; font-weight:600; border:1px solid transparent; margin: 0 .3rem .3rem 0; }
.badge.brand { background: var(--brand-soft); color: var(--brand); border-color:#dcd6ff; }
.badge.ok { background:#e6f6f3; color:var(--teal); border-color:#bfe8e0; }
.badge.warn { background:#fdf3e1; color:var(--amber); border-color:#f3dcae; }
.badge.bad { background:#fdebee; color:var(--rose); border-color:#f6c9d1; }
.badge.neutral { background:#f1f0f7; color:var(--ink-2); border-color:var(--line); }

.qcard { background: var(--surface); border:1px solid var(--line); border-left: 4px solid var(--brand);
  border-radius: 12px; padding: .9rem 1.1rem; margin-bottom: .6rem; box-shadow: var(--shadow); }
.qcard .meta { font-size:.75rem; color: var(--muted); font-weight:600; text-transform: uppercase; letter-spacing:.06em; }
.qcard .q { font-size: 1rem; color: var(--ink); font-weight: 600; margin-top: .25rem; }
.qcard.conceptual { border-left-color: var(--brand); }
.qcard.application { border-left-color: var(--teal); }
.qcard.coding { border-left-color: var(--amber); }

.bigq { background: var(--surface); border:1px solid var(--line); border-radius: 18px; padding: 1.4rem 1.5rem;
  box-shadow: var(--shadow); font-size: 1.18rem; font-weight: 600; color: var(--ink); line-height:1.5; }
.score-ring { width: 92px; height: 92px; border-radius: 50%; display:grid; place-items:center; margin:auto;
  font-weight: 800; font-size: 1.5rem; color: var(--ink); }
.score-ring span { background: var(--surface); width: 72px; height: 72px; border-radius:50%; display:grid; place-items:center; }
.section-title { font-size:.78rem; text-transform:uppercase; letter-spacing:.09em; color:var(--muted); font-weight:700; margin: 1.1rem 0 .5rem; }
.src { display:block; padding:.55rem .8rem; border:1px solid var(--line); border-radius:10px; background:var(--surface);
  margin-bottom:.4rem; text-decoration:none !important; font-size:.86rem; color: var(--ink) !important; }
.src:hover { border-color: var(--brand); }
.src small { display:block; color: var(--muted); font-size:.74rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.muted { color: var(--muted); font-size: .86rem; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


def hero(title: str, subtitle: str, icon: str = "", eyebrow: str = BRAND):
    st.markdown(
        f"""<div class="hero"><div class="eyebrow">{esc(eyebrow)}</div>
        <h1>{esc(title)}</h1><p>{esc(subtitle)}</p>
        {f'<div class="icon">{icon}</div>' if icon else ''}</div>""",
        unsafe_allow_html=True,
    )


def tiles(items):
    """items: list of (label, value, hint)."""
    cols = st.columns(len(items))
    for col, (label, value, hint) in zip(cols, items):
        with col:
            st.markdown(
                f'<div class="tile"><div class="label">{esc(label)}</div>'
                f'<div class="value">{esc(value)}</div><div class="hint">{esc(hint)}</div></div>',
                unsafe_allow_html=True,
            )


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="badge {kind}">{esc(text)}</span>'


def badges(*items):
    st.markdown("".join(badge(t, k) for t, k in items), unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="section-title">{esc(title)}</div>', unsafe_allow_html=True)


def score_ring(score: float, out_of: float = 10):
    pct = 0 if not out_of else max(0.0, min(1.0, score / out_of))
    color = "#0f9d8a" if pct >= 0.7 else "#c9840a" if pct >= 0.45 else "#d1435b"
    label = f"{score:g}" if out_of == 10 else f"{pct * 100:.0f}%"
    st.markdown(
        f'<div class="score-ring" style="background: conic-gradient({color} {pct * 360:.0f}deg, #ebe9f5 0deg)">'
        f"<span>{esc(label)}</span></div>",
        unsafe_allow_html=True,
    )


def sources(items):
    if not items:
        return
    st.markdown(
        "".join(
            f'<a class="src" href="{esc(item.get("url", ""))}" target="_blank" rel="noopener">'
            f'{esc(item.get("title") or item.get("url"))}<small>{esc(item.get("url", ""))}</small></a>'
            for item in items if item.get("url")
        ),
        unsafe_allow_html=True,
    )
