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
from clients import get_gemini_client, get_supabase_client
from embeddings import embed_text, embed_image

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
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

init_state()


# ══════════════════════════════════════════════════════════════════════
#  UPLOAD VALIDATION (Phase 9 / 15)
# ══════════════════════════════════════════════════════════════════════
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
def _embed_task(item, model_name):
    """Thread worker: embed one chunk (text) or image. Returns (item, vector, err)."""
    try:
        if item["kind"] == "image":
            vec = embed_image(item["pil"], model_name)
        else:
            vec = embed_text(item["chunk"]["content"], model_name)
        return item, vec, None
    except Exception as e:
        return item, None, str(e)


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
        futures = [ex.submit(_embed_task, t, model_name) for t in tasks]
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            progress_bar.progress(done / len(tasks), text=f"Embedded {done}/{len(tasks)} chunks…")
            results.append(fut.result())

    # ── Stage 3: upsert with dedup (embedding -> completed) ─────────
    files_done, inserted = set(), 0
    for item, vec, err in results:
        if err or not vec:
            status_text.markdown(f"⚠️ `{item['file']}` chunk failed: {err or 'no vector'}")
            continue
        image_path = item.get("image_url") if item["kind"] == "image" else None
        outcome = rag_db.upsert_chunk(item["chunk"], vec, model_name,
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


def run_rag_pipeline(user_query, model_name):
    meta = {"retrieval_time": 0.0, "generation_time": 0.0, "context_chunks": 0,
            "input_tokens": 0, "output_tokens": 0, "intent": "document",
            "threshold": config.SIMILARITY_THRESHOLD, "scores": [], "fallback": False}
    owner_id = st.session_state.owner_id or None
    # Exclude the just-appended current user message from the history block
    # (it is added explicitly as the Question below).
    prior_messages = st.session_state.messages[:-1] if st.session_state.messages else []
    history_block = conversation.build_history_block(prior_messages, st.session_state.summary)

    # ── Phase 11: intent routing ────────────────────────────────────
    intent = routing.classify_intent(user_query)
    meta["intent"] = intent

    # ── General query: skip RAG entirely (Phase 11) ─────────────────
    if intent == "general":
        t1 = time.time()
        prompt = (f"{history_block}\n\n" if history_block else "") + user_query
        response_text = _generate(prompt, [], meta)
        meta["generation_time"] = time.time() - t1
        return response_text, meta, []

    # ── Retrieve (Phase 3/4/5) ──────────────────────────────────────
    filter_type = routing.intent_to_filter(intent)
    t0 = time.time()
    try:
        rows = cached_retrieve(user_query, model_name, config.SIMILARITY_THRESHOLD, filter_type, owner_id)
    except Exception as e:
        st.warning(f"Retrieval error: {e}")
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

    st.caption(f"Model: {config.GEMINI_EMBED_MODEL if model_choice=='Gemini' else config.COHERE_EMBED_MODEL} · "
               f"threshold {config.SIMILARITY_THRESHOLD}")

    # ── Upload (Phase 9) ────────────────────────────────────────────
    st.markdown("<div class='sidebar-section'><h3>📁 Upload</h3></div>", unsafe_allow_html=True)
    files = st.file_uploader(f"Index using {st.session_state.embedding_model} (≤ {config.MAX_UPLOAD_MB} MB each)",
                             type=list(config.ALLOWED_EXTS), accept_multiple_files=True)
    if st.button("📥 Index Uploads", use_container_width=True) and files:
        prog = st.progress(0, "Starting…")
        status = st.empty()
        count, rejected = index_uploaded_files(files, st.session_state.embedding_model, prog, status)
        if count > 0:
            cached_retrieve.clear()
            st.success(f"Indexed {count} chunk(s) using {st.session_state.embedding_model}!")
        if rejected:
            st.error("Some files were rejected:\n\n" + "\n\n".join(rejected))
        time.sleep(1.5); st.rerun()

    # ── Web crawl (Phase 10) ────────────────────────────────────────
    st.markdown("<div class='sidebar-section'><h3>🌐 Web Crawl</h3></div>", unsafe_allow_html=True)
    crawl_url = st.text_input("Single URL to crawl", placeholder="https://example.com/page")
    if config.CRAWL_ALLOW_ALL:
        st.caption("Allowed domains: **any** (crawl_allow_all is on; robots.txt still respected)")
    else:
        allowed = ", ".join(config.CRAWL_ALLOWED_DOMAINS) or "**none yet** — set `crawl_allowed_domains` or `crawl_allow_all=true`"
        st.caption(f"Allowed domains: {allowed}")
    if st.button("🕸️ Crawl & Index", use_container_width=True) and crawl_url:
        with st.spinner("Crawling…"):
            try:
                title, text = crawl.fetch_url(crawl_url)
                web_chunks = chunking.chunk_web_text(crawl_url, title or crawl_url, text)
                owner = st.session_state.owner_id or None
                n = 0
                for ch in web_chunks:
                    vec = embed_text(ch["content"], st.session_state.embedding_model)
                    if rag_db.upsert_chunk(ch, vec, st.session_state.embedding_model, owner_id=owner) in ("inserted", "updated"):
                        n += 1
                cached_retrieve.clear()
                st.success(f"Crawled '{title or crawl_url}' → indexed {n} chunk(s).")
            except crawl.CrawlError as e:
                st.error(f"Crawl blocked: {e}")
            except Exception as e:
                st.error(f"Crawl failed: {e}")

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


for msg in st.session_state.messages:
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
                    st_copy_to_clipboard(msg["content"])
                render_dev_panel(m)

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
            st_copy_to_clipboard(response_text)
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
