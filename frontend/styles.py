"""Global CSS styling for the Streamlit frontend.
"""

GLOBAL_CSS = """
<style>
/* ── Base colours (works on both light and dark Streamlit themes) ── */
:root {
    --primary:    #4F8EF7;
    --accent:     #7C5CBF;
    --success:    #2ECC71;
    --warning:    #F39C12;
    --error:      #E74C3C;
    --surface:    rgba(255,255,255,0.04);
    --border:     rgba(255,255,255,0.10);
    --text-muted: rgba(255,255,255,0.50);
}

/* ── Chat messages ────────────────────────────────────────────── */
.stChatMessage {
    border-radius: 12px;
    padding: 4px 0;
    margin-bottom: 4px;
}

/* User bubble: right-aligned feel via background tint */
[data-testid="stChatMessageContent"]:has(.user-bubble) {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 10px 14px;
}

/* ── SQL code block ───────────────────────────────────────────── */
.stCode {
    border-radius: 8px;
    font-size: 0.82rem;
}

/* ── Metrics row ──────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 10px;
}

[data-testid="stMetricLabel"] {
    font-size: 0.70rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

[data-testid="stMetricValue"] {
    font-size: 1.1rem;
    font-weight: 600;
}

/* ── Sidebar ──────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.25);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] .stButton button {
    background: transparent;
    border: none;
    text-align: left;
    color: inherit;
    padding: 4px 6px;
    border-radius: 6px;
    font-size: 0.84rem;
}

section[data-testid="stSidebar"] .stButton button:hover {
    background: var(--surface);
}

/* ── Download buttons ─────────────────────────────────────────── */
.stDownloadButton button {
    font-size: 0.78rem;
    padding: 4px 10px;
    border-radius: 6px;
    background: var(--surface);
    border: 1px solid var(--border);
}

/* ── Chat input ───────────────────────────────────────────────── */
.stChatInputContainer {
    border-top: 1px solid var(--border);
    padding-top: 8px;
}

/* ── Expander (SQL viewer) ────────────────────────────────────── */
.streamlit-expanderHeader {
    font-size: 0.82rem;
    color: var(--text-muted);
}

/* ── Error / warning boxes ────────────────────────────────────── */
.stAlert {
    border-radius: 8px;
}

/* ── Dataframe ────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

/* ── Thinking spinner ─────────────────────────────────────────── */
.thinking-text {
    color: var(--text-muted);
    font-style: italic;
    font-size: 0.88rem;
    animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}
</style>
"""

def inject_styles() -> None:
    """Inject global CSS. Call once at the top of the main Streamlit app."""
    import streamlit as st
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
