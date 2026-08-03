"""
Master Settings panel — all the tunables in one place (your explicit request).

Tabs (rendered as a modal-like expander):
  • Embedding model  — Gemini / Cohere (the old sidebar radio lives here now)
  • Multimodal mode   — auto / keyword / off
  • Retrieval         — similarity threshold, rerank top-k
  • Uploads           — max upload MB
  • Diagnostics       — dev mode toggle, faithfulness-check toggle

The sidebar keeps only operator flows (upload, crawl, history, identity).
"""
from __future__ import annotations

import streamlit as st

import config


def render() -> None:
    with st.expander("⚙️ Master Settings", expanded=False):
        tab_model, tab_mm, tab_retr, tab_up, tab_diag = st.tabs([
            "Embedding model", "Multimodal", "Retrieval", "Uploads", "Diagnostics",
        ])
        with tab_model:
            _render_model_tab()
        with tab_mm:
            _render_multimodal_tab()
        with tab_retr:
            _render_retrieval_tab()
        with tab_up:
            _render_uploads_tab()
        with tab_diag:
            _render_diagnostics_tab()


# ── Embedding model ──────────────────────────────────────────────────
def _render_model_tab() -> None:
    model_choice = st.radio(
        "Embedding model",
        ["Gemini", "Cohere"],
        index=0 if st.session_state.embedding_model == "Gemini" else 1,
        key="ms_model_radio",
        help="Both models are populated at index time, so retrieval works "
             "no matter which you pick here. Switching is instant.",
    )
    if model_choice != st.session_state.embedding_model:
        st.session_state.embedding_model = model_choice
        st.rerun()
    st.caption(
        f"Active: **{config.GEMINI_EMBED_MODEL if model_choice == 'Gemini' else config.COHERE_EMBED_MODEL}**"
    )
    st.caption(f"Generation model: **{config.GENERATION_MODEL}**")


# ── Multimodal mode ──────────────────────────────────────────────────
def _render_multimodal_tab() -> None:
    mode = st.radio(
        "Multimodal mode",
        ["auto", "keyword", "off"],
        index=["auto", "keyword", "off"].index(
            getattr(config, "MULTIMODAL_MODE", "auto")),
        key="ms_mm_radio",
        help=(
            "**auto** — always pull the top image by cross-modal similarity, "
            "even when the query doesn't ask for one. (default; recommended)\n\n"
            "**keyword** — legacy: only return image chunks when the query "
            "asks for an image (e.g. contains 'diagram', 'photo', 'show me').\n\n"
            "**off** — text-only retrieval."
        ),
    )
    config.MULTIMODAL_MODE = mode
    st.session_state.multimodal_mode = mode
    top_k = st.slider(
        "Auto image top-k", 0, 3,
        value=int(getattr(config, "AUTO_IMAGE_TOP_K", 1)),
        key="ms_mm_topk",
    )
    config.AUTO_IMAGE_TOP_K = top_k
    st.session_state.auto_image_top_k = top_k


# ── Retrieval ────────────────────────────────────────────────────────
def _render_retrieval_tab() -> None:
    # `value=` doubles as both the default and the setter for new sessions.
    new_threshold = st.slider(
        "Similarity threshold",
        0.0, 1.0,
        value=float(st.session_state.threshold),
        step=0.05, key="ms_threshold",
        help="Min cosine similarity for a chunk to count. Cohere/Gemini "
             "embeddings run low — raise this if too many irrelevant chunks "
             "pass; lower it if you get too many 'not found' replies.",
    )
    st.session_state.threshold = new_threshold

    new_topk = st.slider(
        "Rerank top-k",
        1, 12, value=int(config.RERANK_TOP_K), key="ms_topk",
        help="How many candidates to pass to the LLM after rerank-lite.",
    )
    config.RERANK_TOP_K = new_topk

    new_min_rerank = st.number_input(
        "Min rerank score (fallback gate)",
        0.0, 1.0,
        value=float(config.MIN_RERANK_SCORE), step=0.01, key="ms_min_rerank",
        help="Strengthened fallback: a chunk must also clear this composite "
             "score floor independently of the vector threshold.",
    )
    config.MIN_RERANK_SCORE = new_min_rerank

    new_max_tokens = st.number_input(
        "Max context tokens",
        100, 8000,
        value=int(config.MAX_CONTEXT_TOKENS), step=100, key="ms_max_tokens",
    )
    config.MAX_CONTEXT_TOKENS = new_max_tokens


# ── Uploads ──────────────────────────────────────────────────────────
def _render_uploads_tab() -> None:
    new_max_upload = st.number_input(
        "Max upload MB", 1, 200,
        value=int(config.MAX_UPLOAD_MB), step=1, key="ms_max_upload",
    )
    config.MAX_UPLOAD_MB = new_max_upload
    st.caption(f"Allowed file types: {', '.join(config.ALLOWED_EXTS)}")


# ── Diagnostics ─────────────────────────────────────────────────────
def _render_diagnostics_tab() -> None:
    st.session_state.dev_mode = st.toggle(
        "Dev / admin metrics",
        value=st.session_state.dev_mode, key="ms_dev_toggle",
    )
    new_faith = st.toggle(
        "LLM-as-judge output faithfulness check",
        value=bool(config.OUTPUT_FAITHFULNESS_CHECK), key="ms_faith_toggle",
        help="Adds ~500-1500ms to each response while the judge runs. "
             "Off by default; flip on for sensitive demos or to surface "
             "ungrounded answers in the dev panel.",
    )
    config.OUTPUT_FAITHFULNESS_CHECK = new_faith
    new_anon = st.toggle(
        "Use anon reads (RLS-enforced read path)",
        value=bool(config.USE_ANON_READS), key="ms_anon_toggle",
        help="Routes read queries through the anon key (requires the RLS "
             "migration 0005 and explicit anon SELECT grants).",
    )
    config.USE_ANON_READS = new_anon
