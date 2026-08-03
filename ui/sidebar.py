"""
Sidebar: identity, upload, crawl, history, dev-mode toggle.

The Master Settings panel (model switch, multimodal mode, similarity
threshold, etc.) lives in `ui.master_settings` — this sidebar stays focused
on the operator flows (upload / crawl / history).
"""
from __future__ import annotations

import time
from datetime import datetime

import streamlit as st

import config
import rag_db
from pipeline import (
    SCHEMA_HELP, index_uploaded_files, is_schema_error, schema_is_ready, cached_retrieve,
)
from eval import run_eval, ablation  # noqa: F401  (eval entry-points exposed in the sidebar)


def render() -> None:
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:1rem 0;">
            <span style="font-size:2.5rem;">🧠</span>
            <h2 style="margin:0; font-weight:700; background:linear-gradient(135deg,#667eea,#764ba2);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;">Multi-Model RAG</h2>
        </div>""", unsafe_allow_html=True)

        _render_identity()
        _render_database()
        _render_upload()
        _render_crawl()
        _render_history()
        _render_eval_panel()
        _render_dev_toggle()


# ──────────────────────────────────────────────────────────────────────
#  Identity
# ──────────────────────────────────────────────────────────────────────
def _render_identity() -> None:
    st.markdown("<div class='sidebar-section'><h3>👤 Identity</h3></div>",
                unsafe_allow_html=True)
    st.session_state.owner_id = st.text_input(
        "User ID (optional document isolation)",
        value=st.session_state.owner_id,
        help="Set a user id to keep your uploads private to you. Leave blank to share.",
    )


# ──────────────────────────────────────────────────────────────────────
#  Database bootstrap
# ──────────────────────────────────────────────────────────────────────
def _render_database() -> None:
    st.markdown("<div class='sidebar-section'><h3>🔧 Database</h3></div>",
                unsafe_allow_html=True)
    if config.SUPABASE_DB_URL:
        if st.button("🔧 Initialize Database", use_container_width=True):
            import migrate
            with st.spinner("Applying schema…"):
                try:
                    migrate.run_migrations()
                    schema_is_ready.clear()
                    cached_retrieve.clear()
                    st.success("Schema ready ✅")
                    time.sleep(1.0)
                    st.rerun()
                except Exception as e:                      # noqa: BLE001
                    st.error(f"Initialization failed: {e}\n\n"
                             "Check that `supabase_db_url` is the **Session Pooler** "
                             "URI (port 5432) with the correct password.")
    else:
        st.caption(
            "To enable one-click setup, add a **`supabase_db_url`** secret — the "
            "**Session Pooler** URI from Supabase → Project Settings → Database → "
            "Connection string → *Session pooler* (looks like "
            "`postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres`). "
            "Or run `db/migrations/RUN_THIS_IN_SUPABASE.sql` manually."
        )


# ──────────────────────────────────────────────────────────────────────
#  Upload
# ──────────────────────────────────────────────────────────────────────
def _render_upload() -> None:
    st.markdown("<div class='sidebar-section'><h3>📁 Upload</h3></div>",
                unsafe_allow_html=True)
    files = st.file_uploader(
        f"Index using {st.session_state.embedding_model} (≤ {config.MAX_UPLOAD_MB} MB each)",
        type=list(config.ALLOWED_EXTS), accept_multiple_files=True,
    )
    if st.button("📥 Index Uploads", use_container_width=True) and files:
        prog = st.progress(0, "Starting…")
        status = st.empty()
        try:
            count, rejected = index_uploaded_files(
                files, st.session_state.embedding_model, prog, status)
            if count > 0:
                cached_retrieve.clear()
                st.success(f"Indexed {count} chunk(s) using {st.session_state.embedding_model}!")
            if rejected:
                st.error("Some files were rejected:\n\n" + "\n\n".join(rejected))
            time.sleep(1.5)
            st.rerun()
        except Exception as e:                              # noqa: BLE001
            if is_schema_error(e):
                st.error(SCHEMA_HELP)
            else:
                st.error(f"Indexing failed: {e}")


# ──────────────────────────────────────────────────────────────────────
#  Web crawl
# ──────────────────────────────────────────────────────────────────────
def _render_crawl() -> None:
    import crawl
    import chunking
    from embeddings import embed_text_all

    st.markdown("<div class='sidebar-section'><h3>🌐 Web Crawl</h3></div>",
                unsafe_allow_html=True)
    crawl_url = st.text_input("Website URL", placeholder="https://docs.example.com")
    crawl_mode = st.radio("Crawl mode", ["Single page", "Entire website"],
                          horizontal=True, key="crawl_mode",
                          help="Entire website stays on the same domain, respects "
                               "robots.txt and stops at the page limit.")
    crawl_max_pages = config.CRAWL_MAX_PAGES_DEFAULT
    if crawl_mode == "Entire website":
        crawl_max_pages = int(st.number_input(
            "Max pages", min_value=1, max_value=config.CRAWL_MAX_PAGES_LIMIT,
            value=config.CRAWL_MAX_PAGES_DEFAULT))
    if config.CRAWL_ALLOW_ALL:
        st.caption("Allowed domains: **any** (crawl_allow_all is on; robots.txt still respected)")
    else:
        allowed = ", ".join(config.CRAWL_ALLOWED_DOMAINS) or \
            "**none yet** — set `crawl_allowed_domains` or `crawl_allow_all=true`"
        st.caption(f"Allowed domains: {allowed}")
    if st.button("🕸️ Crawl & Index", use_container_width=True) and crawl_url:
        prog = st.progress(0.0, text="Starting crawl…")
        try:
            mode = "site" if crawl_mode == "Entire website" else "single"

            def _cb(done, total, url):
                prog.progress(min(done / max(total, 1), 0.99),
                              text=f"Crawling page {done + 1}/{total}: {url[:60]}")

            pages, skipped = crawl.crawl_pages(crawl_url, mode=mode,
                                               max_pages=crawl_max_pages,
                                               progress_cb=_cb)
            owner = st.session_state.owner_id or None
            n_chunks = 0
            for i, pg in enumerate(pages):
                prog.progress((i + 1) / max(len(pages), 1),
                              text=f"Indexing {i + 1}/{len(pages)}: {pg['title'][:50]}")
                page_meta = {"page_title": pg["title"], "domain": pg["domain"],
                             "crawl_timestamp": pg["crawled_at"], "source_type": "web"}
                web_chunks = chunking.chunk_web_text(pg["url"], pg["title"] or pg["url"],
                                                     pg["markdown"], extra_metadata=page_meta)
                for ch in web_chunks:
                    vecs = embed_text_all(ch["content"])
                    if rag_db.upsert_chunk_multi(ch, vecs, owner_id=owner) in ("inserted", "updated"):
                        n_chunks += 1
            cached_retrieve.clear()
            prog.empty()
            if pages:
                st.success(f"Crawled {len(pages)} page(s) → indexed {n_chunks} chunk(s).")
            else:
                st.warning("Crawl finished, but no page yielded readable content.")
            if skipped:
                with st.expander(f"⚠️ {len(skipped)} page(s) skipped"):
                    for u, reason in skipped[:20]:
                        st.caption(f"{u} — {reason}")
        except crawl.CrawlError as e:
            prog.empty()
            st.error(f"Crawl blocked: {e}")
        except Exception as e:                              # noqa: BLE001
            prog.empty()
            st.error(SCHEMA_HELP if is_schema_error(e) else f"Crawl failed: {e}")


# ──────────────────────────────────────────────────────────────────────
#  History
# ──────────────────────────────────────────────────────────────────────
def _render_history() -> None:
    st.markdown("<div class='sidebar-section'><h3>🕐 History</h3></div>",
                unsafe_allow_html=True)
    try:
        history = rag_db.load_sessions()
    except Exception:                                       # noqa: BLE001
        history = []
    for h in history:
        sid = h["session_id"]
        ts = h.get("timestamp", "")[:16].replace("T", " ")
        if st.button(
            f"{'🟢 ' if sid == st.session_state.current_session_id else ''}{ts} "
            f"({h.get('embedding_model')})",
            key=f"hs_{sid}", use_container_width=True,
        ):
            if sid != st.session_state.current_session_id:
                st.session_state.messages = h.get("messages", [])
                st.session_state.total_input_tokens = h.get("total_input_tokens", 0)
                st.session_state.total_output_tokens = h.get("total_output_tokens", 0)
                st.session_state.total_queries = h.get("total_queries", 0)
                st.session_state.current_session_id = sid
                st.session_state.summary = ""
                st.rerun()

    if st.button("🆕 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.summary = ""
        st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.rerun()


# ──────────────────────────────────────────────────────────────────────
#  Eval panel (CLI entry-points surfaced in the UI for portfolio demos)
# ──────────────────────────────────────────────────────────────────────
def _render_eval_panel() -> None:
    st.markdown("<div class='sidebar-section'><h3>📊 Eval</h3></div>",
                unsafe_allow_html=True)
    st.caption("Run the eval harness. See `eval/README.md` for the full guide.")
    c1, c2 = st.columns(2)
    if c1.button("Run eval", use_container_width=True):
        with st.spinner("Running eval…"):
            import io as _io
            from contextlib import redirect_stdout
            buf = _io.StringIO()
            try:
                with redirect_stdout(buf):
                    run_eval.main()
            except SystemExit:
                pass
            st.code(buf.getvalue() or "(no output)")
    if c2.button("Run ablation", use_container_width=True):
        with st.spinner("Running ablation…"):
            import io as _io
            from contextlib import redirect_stdout
            buf = _io.StringIO()
            try:
                with redirect_stdout(buf):
                    ablation.main()
            except SystemExit:
                pass
            st.code(buf.getvalue() or "(no output)")


# ──────────────────────────────────────────────────────────────────────
#  Dev toggle
# ──────────────────────────────────────────────────────────────────────
def _render_dev_toggle() -> None:
    st.session_state.dev_mode = st.toggle("🛠️ Dev / Admin metrics",
                                          value=st.session_state.dev_mode)
