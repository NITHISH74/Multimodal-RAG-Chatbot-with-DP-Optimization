"""
╔══════════════════════════════════════════════════════════════════════╗
║  Multi-Model RAG Chatbot — Streamlit Web Application                  ║
║  Storage: Supabase (pgvector + Storage)                               ║
║  Embeddings: Gemini & Cohere   •   Generation: Gemini 2.5 Flash       ║
║                                                                        ║
║  Pipeline: route -> hybrid retrieve (vector+keyword) -> rerank-lite   ║
║            -> semantic dedup -> token-budget knapsack -> generate     ║
║            -> cite sources. Safe fallback when nothing is relevant.   ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import os
import io
import time
import base64
import concurrent.futures
from datetime import datetime

import requests
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from st_copy_to_clipboard import st_copy_to_clipboard

import config
import chunking
import rag_db
import retrieval
import context_builder
import routing
import conversation
import crawl
import migrate
from clients import get_gemini_client, get_supabase_client
from embeddings import embed_text_all, embed_image_all

load_dotenv()
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Multi-Model RAG", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .main-header { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 1.8rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(48, 43, 99, 0.4); border: 1px solid rgba(255,255,255,0.08); }
    .main-header h1 { color: #fff; font-weight: 800; font-size: 1.8rem; margin: 0; }
    .main-header p { color: rgba(255,255,255,0.65); font-size: 0.9rem; margin: 0.3rem 0 0 0; }
    .metric-row { display: flex; gap: 12px; margin-bottom: 1rem; }
    .metric-card { background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 1rem; flex: 1; text-align: center; }
    .metric-value { font-size: 1.6rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
    .metric-label { font-size: 0.75rem; color: rgba(255,255,255,0.5); text-transform: uppercase; margin: 0.3rem 0 0 0;}
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%); }
    .sidebar-section { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 1rem; margin: 0.8rem 0; }
    .sidebar-section h3 { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;
        color: rgba(255,255,255,0.4); margin: 0 0 0.6rem 0; }
    .status-badge { display:inline-block; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.72rem; font-weight:600; }
    .status-gemini { background: rgba(66,133,244,0.15); color:#4285f4; border:1px solid rgba(66,133,244,0.3); }
    .status-cohere { background: rgba(168,85,247,0.15); color:#a855f7; border:1px solid rgba(168,85,247,0.3); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "messages": [], "total_input_tokens": 0, "total_output_tokens": 0,
        "total_queries": 0, "embedding_model": "Gemini",
        "current_session_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "summary": "", "owner_id": "", "dev_mode": config.DEV_MODE,
        "threshold": config.SIMILARITY_THRESHOLD,
        "answer_feedback": {},
        "feedback_guidance": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_state()


# ══════════════════════════════════════════════════════════════════════
#  UPLOAD VALIDATION (Phase 9 / 15)
# ══════════════════════════════════════════════════════════════════════
SCHEMA_HELP = (
    "🗄️ **Database not migrated yet.** Your Supabase `documents` table is missing "
    "the V3 columns (e.g. `content_hash`). Open **Supabase → SQL Editor** and run "
    "**`db/migrations/RUN_THIS_IN_SUPABASE.sql`** (or migrations 0000–0005 in order), "
    "then try again. This is a one-time database setup step."
)


@st.cache_data(ttl=60, show_spinner=False)
def schema_is_ready():
    """Cached check: is the V3 DB schema applied? (Cleared after Initialize.)"""
    return migrate.schema_ready(get_supabase_client())


def is_schema_error(exc):
    """Detect the 'column/function/relation does not exist' class of errors
    that mean the DB migrations haven't been applied yet."""
    msg = str(getattr(exc, "message", "") or exc).lower()
    return ("does not exist" in msg or "42703" in msg or "42p01" in msg
            or "pgrst" in msg or "schema cache" in msg or "could not find" in msg)


def validate_upload(uploaded_file):
    """Return (ok, error_message). Enforces extension + size limits."""
    name = uploaded_file.name.lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    if ext not in config.ALLOWED_EXTS:
        return False, f"`{uploaded_file.name}`: unsupported type .{ext} (allowed: {', '.join(config.ALLOWED_EXTS)})"
    size_mb = (uploaded_file.size or 0) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_MB:
        return False, f"`{uploaded_file.name}`: {size_mb:.1f} MB exceeds the {config.MAX_UPLOAD_MB} MB limit."
    return True, ""


# ══════════════════════════════════════════════════════════════════════
#  INDEXING (Phase 2 chunking, Phase 8 storage, Phase 9 staged status)
# ══════════════════════════════════════════════════════════════════════
def _embed_task(item):
    """Thread worker: embed one chunk (text) or image under ALL index models.
    Returns (item, {model: vector}, err)."""
    try:
        if item["kind"] == "image":
            vecs = embed_image_all(item["pil"])
        else:
            vecs = embed_text_all(item["chunk"]["content"])
        if not vecs:
            return item, {}, "no embedding model available (check API keys)"
        return item, vecs, None
    except Exception as e:
        return item, {}, str(e)


def index_uploaded_files(uploaded_files, model_name, progress_bar, status_text):
    owner_id = st.session_state.owner_id or None
    tasks = []            # embedding jobs
    rejected = []

    # ── Stage 1: validate + extract/chunk (uploaded -> extracting) ──
    for f in uploaded_files:
        ok, err = validate_upload(f)
        if not ok:
            rejected.append(err)
            status_text.markdown(f"❌ {err}")
            continue

        status_text.markdown(f"📤 `{f.name}` uploaded → extracting…")
        file_bytes = f.read()
        ext = f.name.rsplit(".", 1)[-1].lower()

        if ext in config.ALLOWED_IMG_EXTS:
            # Phase 8: push image to Storage, keep only the URL.
            ctype = f"image/{'jpeg' if ext in ('jpg','jpeg') else ext}"
            image_url = rag_db.upload_image(file_bytes, f.name, content_type=ctype)
            pil = Image.open(io.BytesIO(file_bytes))
            chunk = chunking.image_chunk(f.name)
            tasks.append({"kind": "image", "chunk": chunk, "pil": pil,
                          "image_url": image_url, "file": f.name})
        else:
            doc_chunks = chunking.chunk_document(f.name, file_bytes)
            if not doc_chunks:
                status_text.markdown(f"⚠️ `{f.name}`: no usable text extracted.")
                continue
            for ch in doc_chunks:
                tasks.append({"kind": "text", "chunk": ch, "file": f.name})

    if not tasks:
        return 0, rejected

    # ── Stage 2: embed in parallel (extracting -> embedding) ────────
    progress_bar.progress(0.05, text=f"Embedding {len(tasks)} chunk(s)…")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.EMBED_MAX_WORKERS) as ex:
        futures = [ex.submit(_embed_task, t) for t in tasks]
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            progress_bar.progress(done / len(tasks), text=f"Embedded {done}/{len(tasks)} chunks…")
            results.append(fut.result())

    # ── Stage 3: upsert with dedup (embedding -> completed) ─────────
    files_done, inserted = set(), 0
    for item, vecs, err in results:
        if err or not vecs:
            status_text.markdown(f"⚠️ `{item['file']}` chunk failed: {err or 'no vector'}")
            continue
        image_path = item.get("image_url") if item["kind"] == "image" else None
        outcome = rag_db.upsert_chunk_multi(item["chunk"], vecs,
                                            image_path=image_path, owner_id=owner_id)
        if outcome in ("inserted", "updated"):
            inserted += 1
        files_done.add(item["file"])

    for fn in files_done:
        status_text.markdown(f"✅ `{fn}` → completed.")
    return inserted, rejected


# ══════════════════════════════════════════════════════════════════════
#  RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=config.QUERY_CACHE_TTL, show_spinner=False)
def cached_retrieve(query, model_name, threshold, filter_type, owner_id):
    """Cached hybrid retrieval (Phase 1). Cleared after indexing."""
    return retrieval.hybrid_retrieve(query, model_name, threshold=threshold,
                                     filter_type=filter_type, owner_id=owner_id)


def _load_image_from_url(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))
    except Exception:
        return None


def build_feedback_guidance():
    notes = [n.strip() for n in st.session_state.feedback_guidance if n.strip()]
    if not notes:
        return ""
    recent = "\n".join(f"- {n}" for n in notes[-3:])
    return (
        "User feedback from this session:\n"
        f"{recent}\n"
        "Apply this style/quality feedback when it is relevant, but keep answers grounded in retrieved sources."
    )


def run_rag_pipeline(user_query, model_name):
    threshold = float(st.session_state.threshold)
    meta = {"retrieval_time": 0.0, "generation_time": 0.0, "context_chunks": 0,
            "input_tokens": 0, "output_tokens": 0, "intent": "document",
            "threshold": threshold, "scores": [], "fallback": False}
    owner_id = st.session_state.owner_id or None
    # Exclude the just-appended current user message from the history block
    # (it is added explicitly as the Question below).
    prior_messages = st.session_state.messages[:-1] if st.session_state.messages else []
    history_block = conversation.build_history_block(prior_messages, st.session_state.summary)
    feedback_block = build_feedback_guidance()

    # ── Phase 11: intent routing ────────────────────────────────────
    intent = routing.classify_intent(user_query)
    meta["intent"] = intent

    # ── General query: skip RAG entirely (Phase 11) ─────────────────
    if intent == "general":
        t1 = time.time()
        parts = [p for p in [history_block, feedback_block, user_query] if p]
        prompt = "\n\n".join(parts)
        response_text = _generate(prompt, [], meta)
        meta["generation_time"] = time.time() - t1
        return response_text, meta, []

    # ── Retrieve (Phase 3/4/5) ──────────────────────────────────────
    filter_type = routing.intent_to_filter(intent)
    t0 = time.time()
    try:
        rows = cached_retrieve(user_query, model_name, threshold, filter_type, owner_id)
    except Exception as e:
        st.warning(SCHEMA_HELP if is_schema_error(e) else f"Retrieval error: {e}")
        rows = []
    meta["retrieval_time"] = time.time() - t0
    meta["scores"] = [round(r.get("rerank_score", r.get("similarity", 0)), 3) for r in rows]

    # ── Phase 12: safe fallback when nothing is relevant ────────────
    if not rows:
        meta["fallback"] = True
        return config.FALLBACK_MESSAGE, meta, []

    # ── Phase 6: dedup -> token-budget knapsack -> context ──────────
    rows = context_builder.semantic_dedup(rows)
    chosen = context_builder.knapsack_select(rows)
    meta["context_chunks"] = len(chosen)
    context_str, image_rows, used = context_builder.build_context(chosen)

    # Load any cited images (Phase 8: from Storage URL, not base64).
    image_parts = []
    for r in image_rows:
        if r.get("image_path"):
            img = _load_image_from_url(r["image_path"])
            if img is not None:
                image_parts.append(img)

    # ── Generate with history + cited context (Phase 13 TOON metadata) ──
    toon = context_builder.toon_metadata(chosen)
    prompt_parts = []
    if history_block:
        prompt_parts.append(history_block)
    if feedback_block:
        prompt_parts.append(feedback_block)
    prompt_parts.append(
        "Use ONLY the context below to answer. Cite sources by file name. "
        "If the context is insufficient, say so.\n\n"
        f"Context metadata ({toon}):\n---\n{context_str}\n---\n\nQuestion: {user_query}"
    )
    t1 = time.time()
    response_text = _generate("\n\n".join(prompt_parts), image_parts, meta)
    meta["generation_time"] = time.time() - t1

    # ── Phase 7: append Sources section ─────────────────────────────
    sources = context_builder.format_sources(used)
    if sources:
        response_text = f"{response_text}\n{sources}"
    return response_text, meta, used


def _generate(prompt, image_parts, meta):
    try:
        client = get_gemini_client()
        if client is None:
            return "⚠️ Gemini API key not configured."
        res = client.models.generate_content(
            model=config.GENERATION_MODEL, contents=[prompt] + image_parts)
        if hasattr(res, "usage_metadata") and res.usage_metadata:
            meta["input_tokens"] = res.usage_metadata.prompt_token_count or 0
            meta["output_tokens"] = res.usage_metadata.candidates_token_count or 0
        return res.text
    except Exception as e:
        return f"Generation error: {e}"


# ══════════════════════════════════════════════════════════════════════
#  HISTORY PERSISTENCE
# ══════════════════════════════════════════════════════════════════════
def save_chat_history():
    sid = st.session_state.current_session_id
    try:
        rag_db.save_session(sid, st.session_state.embedding_model,
                            st.session_state.total_input_tokens,
                            st.session_state.total_output_tokens,
                            st.session_state.total_queries,
                            st.session_state.summary)
        rag_db.save_messages(sid, st.session_state.messages)
    except Exception as e:
        st.warning(f"Could not save history: {e}")


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <span style="font-size:2.5rem;">🧠</span>
        <h2 style="margin:0; font-weight:700; background:linear-gradient(135deg,#667eea,#764ba2);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;">Multi-Model RAG</h2>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'><h3>⚙️ Settings</h3></div>", unsafe_allow_html=True)
    model_choice = st.radio("Embedding Model", ["Gemini", "Cohere"],
                            index=0 if st.session_state.embedding_model == "Gemini" else 1)
    if model_choice != st.session_state.embedding_model:
        st.session_state.embedding_model = model_choice
        st.rerun()

    st.session_state.owner_id = st.text_input(
        "User ID (optional document isolation)", value=st.session_state.owner_id,
        help="Set a user id to keep your uploads private to you. Leave blank to share.")

    st.session_state.threshold = st.slider(
        "Similarity threshold", 0.0, 1.0, value=float(st.session_state.threshold), step=0.05,
        help="Min similarity for a chunk to count. Cohere/Gemini scores run low — "
             "lower this if you get too many 'not found' replies; raise it to be stricter.")

    st.caption(f"Model: {config.GEMINI_EMBED_MODEL if model_choice=='Gemini' else config.COHERE_EMBED_MODEL} · "
               f"threshold {st.session_state.threshold:.2f}")

    # ── Database setup ──────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'><h3>🔧 Database</h3></div>", unsafe_allow_html=True)
    if config.SUPABASE_DB_URL:
        if st.button("🔧 Initialize Database", use_container_width=True):
            with st.spinner("Applying schema…"):
                try:
                    migrate.run_migrations()
                    schema_is_ready.clear()
                    cached_retrieve.clear()
                    st.success("Schema ready ✅")
                    time.sleep(1.0); st.rerun()
                except Exception as e:
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

    # ── Upload (Phase 9) ────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'><h3>📁 Upload</h3></div>", unsafe_allow_html=True)
    files = st.file_uploader(f"Index using {st.session_state.embedding_model} (≤ {config.MAX_UPLOAD_MB} MB each)",
                             type=list(config.ALLOWED_EXTS), accept_multiple_files=True)
    if st.button("📥 Index Uploads", use_container_width=True) and files:
        prog = st.progress(0, "Starting…")
        status = st.empty()
        try:
            count, rejected = index_uploaded_files(files, st.session_state.embedding_model, prog, status)
            if count > 0:
                cached_retrieve.clear()
                st.success(f"Indexed {count} chunk(s) using {st.session_state.embedding_model}!")
            if rejected:
                st.error("Some files were rejected:\n\n" + "\n\n".join(rejected))
            time.sleep(1.5); st.rerun()
        except Exception as e:
            if is_schema_error(e):
                st.error(SCHEMA_HELP)
            else:
                st.error(f"Indexing failed: {e}")

    # ── Web crawl (Phase 10 — Crawl4AI) ─────────────────────────────
    st.markdown("<div class='sidebar-section'><h3>🌐 Web Crawl</h3></div>", unsafe_allow_html=True)
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
        allowed = ", ".join(config.CRAWL_ALLOWED_DOMAINS) or "**none yet** — set `crawl_allowed_domains` or `crawl_allow_all=true`"
        st.caption(f"Allowed domains: {allowed}")
    if st.button("🕸️ Crawl & Index", use_container_width=True) and crawl_url:
        prog = st.progress(0.0, text="Starting crawl…")
        try:
            mode = "site" if crawl_mode == "Entire website" else "single"

            def _crawl_cb(done, total, url):
                prog.progress(min(done / max(total, 1), 0.99),
                              text=f"Crawling page {done + 1}/{total}: {url[:60]}")

            pages, skipped = crawl.crawl_pages(crawl_url, mode=mode,
                                               max_pages=crawl_max_pages,
                                               progress_cb=_crawl_cb)
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
        except Exception as e:
            prog.empty()
            st.error(SCHEMA_HELP if is_schema_error(e) else f"Crawl failed: {e}")

    # ── History ─────────────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'><h3>🕐 History</h3></div>", unsafe_allow_html=True)
    try:
        history = rag_db.load_sessions()
    except Exception:
        history = []
    for h in history:
        sid = h["session_id"]
        ts = h.get("timestamp", "")[:16].replace("T", " ")
        if st.button(f"{'🟢 ' if sid==st.session_state.current_session_id else ''}{ts} ({h.get('embedding_model')})",
                     key=f"hs_{sid}", use_container_width=True):
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

    # Dev mode toggle (Phase 14)
    st.session_state.dev_mode = st.toggle("🛠️ Dev / Admin metrics", value=st.session_state.dev_mode)


# ══════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🧠 Multi-Model RAG Chatbot</h1>
    <p>Hybrid retrieval • Rerank-Lite • Token-optimized context • Cited answers</p>
</div>""", unsafe_allow_html=True)

if not schema_is_ready():
    st.error(
        "🗄️ **Database not initialized.** Your Supabase schema is missing the V3 "
        "tables/columns, so indexing and search won't work yet. Open the sidebar → "
        "**🔧 Database → Initialize Database** (one-time setup)."
    )

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card"><p class="metric-value">{st.session_state.total_queries}</p><p class="metric-label">Queries</p></div>
    <div class="metric-card"><p class="metric-value">{st.session_state.total_input_tokens + st.session_state.total_output_tokens:,}</p><p class="metric-label">Total Tokens</p></div>
    <div class="metric-card"><p class="metric-value">Supabase</p><p class="metric-label">Vector DB</p></div>
</div>""", unsafe_allow_html=True)


def render_dev_panel(meta):
    """Phase 14: per-query diagnostics, dev-mode only."""
    if not st.session_state.dev_mode:
        return
    with st.expander("🛠️ Query diagnostics", expanded=False):
        total = meta.get("retrieval_time", 0) + meta.get("generation_time", 0)
        c = st.columns(4)
        c[0].metric("Retrieval", f"{meta.get('retrieval_time',0)*1000:.0f} ms")
        c[1].metric("Generation", f"{meta.get('generation_time',0)*1000:.0f} ms")
        c[2].metric("Total", f"{total*1000:.0f} ms")
        c[3].metric("Chunks", meta.get("context_chunks", 0))
        c2 = st.columns(3)
        c2[0].metric("Input tokens", meta.get("input_tokens", 0))
        c2[1].metric("Output tokens", meta.get("output_tokens", 0))
        c2[2].metric("Intent", meta.get("intent", "—"))
        st.caption(f"Threshold: {meta.get('threshold')} · Fallback: {meta.get('fallback')} · "
                   f"Similarity/rerank scores: {meta.get('scores')}")


def _previous_user_question(message_index):
    for j in range(message_index - 1, -1, -1):
        msg = st.session_state.messages[j]
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def submit_feedback(message_index, rating, comment=""):
    msg = st.session_state.messages[message_index]
    st.session_state.answer_feedback[message_index] = {
        "rating": rating,
        "comment": comment or "",
    }
    if rating == "down" and comment:
        st.session_state.feedback_guidance.append(comment)
    try:
        rag_db.save_feedback(
            st.session_state.current_session_id,
            message_index,
            rating,
            _previous_user_question(message_index),
            msg.get("content", ""),
            comment=comment,
            meta=msg.get("meta", {}),
            owner_id=st.session_state.owner_id or None,
        )
        st.toast("Feedback saved. Thank you.")
    except Exception as e:
        st.warning(SCHEMA_HELP if is_schema_error(e) else f"Could not save feedback: {e}")


def render_feedback_controls(message_index):
    saved = st.session_state.answer_feedback.get(message_index)
    if saved:
        label = "Helpful" if saved.get("rating") == "up" else "Needs improvement"
        st.caption(f"Feedback: {label}")
        return

    cols = st.columns([1, 1, 4])
    if cols[0].button("Helpful", key=f"fb_up_{message_index}", use_container_width=True):
        submit_feedback(message_index, "up")
        st.rerun()
    if cols[1].button("Improve", key=f"fb_down_{message_index}", use_container_width=True):
        st.session_state[f"fb_open_{message_index}"] = True

    if st.session_state.get(f"fb_open_{message_index}"):
        with st.form(f"fb_form_{message_index}", clear_on_submit=True):
            comment = st.text_area(
                "What should be improved?",
                placeholder="Example: answer more directly, include page references, avoid extra explanation...",
                key=f"fb_comment_{message_index}",
            )
            submitted = st.form_submit_button("Save feedback")
            if submitted:
                submit_feedback(message_index, "down", comment)
                st.session_state[f"fb_open_{message_index}"] = False
                st.rerun()


def render_feedback_summary():
    if not st.session_state.dev_mode:
        return
    try:
        stats = rag_db.load_feedback_summary()
    except Exception:
        return
    if not stats["total"]:
        return
    with st.expander("Answer feedback summary", expanded=False):
        cols = st.columns(4)
        cols[0].metric("Feedback", stats["total"])
        cols[1].metric("Helpful", stats["positive"])
        cols[2].metric("Improve", stats["negative"])
        cols[3].metric("Helpful rate", f"{stats['positive_rate'] * 100:.0f}%")


for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(msg["content"])
            m = msg.get("meta") or {}
            if m:
                cols = st.columns([1, 1, 1, 2])
                cols[0].caption(f"⏱️ {m.get('retrieval_time',0):.2f}s")
                cols[1].caption(f"⚡ {m.get('generation_time',0):.2f}s")
                cols[2].caption(f"📎 {m.get('context_chunks',0)}")
                with cols[3]:
                    st_copy_to_clipboard(msg["content"], key=f"copy_hist_{i}")
                render_dev_panel(m)
                render_feedback_controls(i)

render_feedback_summary()

if user_input := st.chat_input("Ask about your documents…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner(f"Searching with {st.session_state.embedding_model}…"):
            response_text, meta, used = run_rag_pipeline(user_input, st.session_state.embedding_model)
        st.markdown(response_text)
        cols = st.columns([1, 1, 1, 2])
        cols[0].caption(f"⏱️ {meta['retrieval_time']:.2f}s")
        cols[1].caption(f"⚡ {meta['generation_time']:.2f}s")
        cols[2].caption(f"📎 {meta['context_chunks']}")
        with cols[3]:
            st_copy_to_clipboard(response_text, key=f"copy_live_{len(st.session_state.messages)}")
        render_dev_panel(meta)

    st.session_state.messages.append({"role": "assistant", "content": response_text, "meta": meta})
    st.session_state.total_input_tokens += meta["input_tokens"]
    st.session_state.total_output_tokens += meta["output_tokens"]
    st.session_state.total_queries += 1

    # Phase 13: refresh running summary, then persist.
    st.session_state.summary = conversation.maybe_update_summary(
        st.session_state.messages, st.session_state.summary)
    save_chat_history()
    time.sleep(0.3)
    st.rerun()
