"""
Multi-Model Advanced RAG Chatbot — Streamlit entry point.

The actual logic lives in:
    pipeline.py        retrieval, indexing, generation, history persistence
    retrieval.py       hybrid retrieve + rerank-lite
    context_builder.py dedup, knapsack, citations, TOON metadata
    chunking.py        document parsing + cleaning
    guardrails.py      input/output safety checks
    ui/                Streamlit widgets (sidebar, master settings, chat)
    eval/              retrieval + LLM-as-judge eval harness
"""
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

import config
from clients import get_gemini_client, get_cohere_client
from pipeline import schema_is_ready
from ui import chat as chat_ui
from ui import master_settings as master_ui
from ui import sidebar as sidebar_ui
from ui.components import (
    inject_css, render_header, render_metric_tiles, render_status_row,
)

load_dotenv()
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ── Page config + CSS ────────────────────────────────────────────────
st.set_page_config(page_title="Multi-Model RAG", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")
inject_css()


# ── Session state ────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "messages": [],
        "total_input_tokens": 0, "total_output_tokens": 0,
        "total_queries": 0,
        "embedding_model": "Gemini",
        "current_session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "summary": "", "owner_id": "", "dev_mode": config.DEV_MODE,
        "threshold": config.SIMILARITY_THRESHOLD,
        "answer_feedback": {}, "feedback_guidance": [],
        "embedding_errors": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init_state()


# ── Sidebar (operator flows: upload, crawl, history, identity, eval) ─
sidebar_ui.render()


# ── Main column ──────────────────────────────────────────────────────
render_header()
render_status_row(
    gemini=get_gemini_client() is not None,
    cohere=get_cohere_client() is not None,
    db_ok=schema_is_ready(),
)
master_ui.render()
render_metric_tiles(
    queries=st.session_state.total_queries,
    total_tokens=st.session_state.total_input_tokens + st.session_state.total_output_tokens,
)

if not schema_is_ready():
    st.error(
        "🗄️ **Database not initialized.** Your Supabase schema is missing the V3 "
        "tables/columns, so indexing and search won't work yet. Open the sidebar → "
        "**🔧 Database → Initialize Database** (one-time setup)."
    )

# ── Chat surface ─────────────────────────────────────────────────────
chat_ui.render_history()
chat_ui.render_feedback_summary()
chat_ui.render_input_and_run()
