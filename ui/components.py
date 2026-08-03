"""
Shared UI primitives: CSS, status chips, metric tiles, source cards.

Kept as small, pure-HTML helpers so the rest of the UI stays declarative.
"""
from __future__ import annotations

import streamlit as st


# ── Global CSS ───────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .main-header { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px rgba(48, 43, 99, 0.4); border: 1px solid rgba(255,255,255,0.08); }
    .main-header h1 { color: #fff; font-weight: 800; font-size: 1.7rem; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.65); font-size: 0.9rem; margin: 0.3rem 0 0 0; }
    .status-row { display: flex; gap: 10px; margin-top: 0.8rem; }
    .status-badge { display:inline-block; padding:0.25rem 0.8rem; border-radius:20px; font-size:0.72rem; font-weight:600; }
    .status-ok { background: rgba(34,197,94,0.15); color:#22c55e; border:1px solid rgba(34,197,94,0.3); }
    .status-warn { background: rgba(234,179,8,0.15); color:#eab308; border:1px solid rgba(234,179,8,0.3); }
    .status-err { background: rgba(239,68,68,0.15); color:#ef4444; border:1px solid rgba(239,68,68,0.3); }
    .metric-row { display: flex; gap: 12px; margin-bottom: 1rem; }
    .metric-card { background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 1rem; flex: 1; text-align: center; }
    .metric-value { font-size: 1.6rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
    .metric-label { font-size: 0.72rem; color: rgba(255,255,255,0.5); text-transform: uppercase; margin: 0.3rem 0 0 0;}
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%); }
    .sidebar-section { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 1rem; margin: 0.8rem 0; }
    .sidebar-section h3 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;
        color: rgba(255,255,255,0.4); margin: 0 0 0.6rem 0; }
    .source-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px; padding: 0.7rem 0.9rem; margin: 0.4rem 0; font-size: 0.85rem; }
    .source-card .src-head { display: flex; justify-content: space-between; gap: 0.5rem; align-items: baseline; }
    .source-card .src-name { font-weight: 600; }
    .source-card .src-meta { color: rgba(255,255,255,0.5); font-size: 0.75rem; }
    .source-card .src-score { background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; }
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Status chips ─────────────────────────────────────────────────────
def status_chip(label: str, state: str = "ok") -> str:
    """One status pill (Gemini / Cohere / DB)."""
    return f'<span class="status-badge status-{state}">{label}</span>'


def render_status_row(gemini: bool, cohere: bool, db_ok: bool) -> None:
    chips = [
        status_chip("Gemini ✓" if gemini else "Gemini ✗",
                    "ok" if gemini else "err"),
        status_chip("Cohere ✓" if cohere else "Cohere ✗",
                    "ok" if cohere else "err"),
        status_chip("DB ✓" if db_ok else "DB ✗",
                    "ok" if db_ok else "err"),
    ]
    st.markdown('<div class="status-row">' + "".join(chips) + "</div>",
                unsafe_allow_html=True)


# ── Header + metric tiles ────────────────────────────────────────────
def render_header() -> None:
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Multi-Model RAG Chatbot</h1>
        <p>Hybrid retrieval • Rerank-Lite • Token-optimized context • Cited answers</p>
    </div>""", unsafe_allow_html=True)


def render_metric_tiles(queries: int, total_tokens: int) -> None:
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card"><p class="metric-value">{queries}</p><p class="metric-label">Queries</p></div>
        <div class="metric-card"><p class="metric-value">{total_tokens:,}</p><p class="metric-label">Total Tokens</p></div>
        <div class="metric-card"><p class="metric-value">Supabase</p><p class="metric-label">Vector DB</p></div>
    </div>""", unsafe_allow_html=True)


# ── Source cards (rendered alongside a fallback message) ─────────────
def render_source_cards(rows) -> None:
    if not rows:
        return
    seen = set()
    for r in rows:
        key = (r.get("file_name"), r.get("page_number"))
        if key in seen:
            continue
        seen.add(key)
        score = r.get("rerank_score", r.get("similarity", 0.0))
        loc = ""
        if r.get("page_number") is not None:
            kind = "Slide" if r.get("document_type") == "pptx" else "Page"
            loc = f" — {kind} {r['page_number']}"
        st.markdown(
            f'<div class="source-card">'
            f'<div class="src-head">'
            f'  <span class="src-name">{r.get("file_name", "unknown")}{loc}</span>'
            f'  <span class="src-score">{score:.2f}</span>'
            f'</div>'
            f'<div class="src-meta">{r.get("document_type", "")}'
            f'{" · " + r["source_url"] if r.get("source_url") else ""}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
