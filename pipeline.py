"""
RAG pipeline logic, decoupled from the Streamlit UI.

`app.py` is a thin entry point that wires this module to UI components.
Everything in here is plain Python (with a few `import streamlit as st`
calls for caching + warning helpers) so it can also be exercised from the
eval harness and from CLI smoke tests.
"""
from __future__ import annotations

import concurrent.futures
import io
import time
from typing import Any, Callable, Optional

import requests
import streamlit as st
from PIL import Image

import chunking
import config
import context_builder
import conversation
import embeddings
import guardrails
import migrate
import rag_db
import retrieval
import routing
from clients import get_gemini_client, get_supabase_client


# ──────────────────────────────────────────────────────────────────────
#  Schema / migration helpers (cached for the UI)
# ──────────────────────────────────────────────────────────────────────
SCHEMA_HELP = (
    "🗄️ **Database not migrated yet.** Your Supabase `documents` table is missing "
    "the V3 columns (e.g. `content_hash`). Open **Supabase → SQL Editor** and run "
    "**`db/migrations/RUN_THIS_IN_SUPABASE.sql`** (or migrations 0000–0005 in order), "
    "then try again. This is a one-time database setup step."
)


@st.cache_data(ttl=60, show_spinner=False)
def schema_is_ready():
    return migrate.schema_ready(get_supabase_client())


def is_schema_error(exc):
    msg = str(getattr(exc, "message", "") or exc).lower()
    return ("does not exist" in msg or "42703" in msg or "42p01" in msg
            or "pgrst" in msg or "schema cache" in msg or "could not find" in msg)


# ──────────────────────────────────────────────────────────────────────
#  Upload validation
# ──────────────────────────────────────────────────────────────────────
def validate_upload(uploaded_file) -> tuple[bool, str]:
    name = uploaded_file.name.lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    if ext not in config.ALLOWED_EXTS:
        return False, f"`{uploaded_file.name}`: unsupported type .{ext} (allowed: {', '.join(config.ALLOWED_EXTS)})"
    size_mb = (uploaded_file.size or 0) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_MB:
        return False, f"`{uploaded_file.name}`: {size_mb:.1f} MB exceeds the {config.MAX_UPLOAD_MB} MB limit."
    return True, ""


# ──────────────────────────────────────────────────────────────────────
#  Indexing
# ──────────────────────────────────────────────────────────────────────
def _embed_task(item, on_error):
    """Thread worker: embed one chunk (text) or image under ALL index models.
    Returns (item, {model: vector}, err_or_None)."""
    try:
        if item["kind"] == "image":
            vecs = embeddings.embed_image_all(item["pil"], on_error=on_error)
        else:
            vecs = embeddings.embed_text_all(item["chunk"]["content"], on_error=on_error)
        if not vecs:
            return item, {}, "no embedding model available (check API keys / retry log)"
        return item, vecs, None
    except Exception as e:                                  # noqa: BLE001
        return item, {}, str(e)


def index_uploaded_files(uploaded_files, model_name, progress_bar, status_text):
    """Parse, embed, and upsert a batch of uploaded files.

    Returns (inserted_count, rejected_messages). Embedding failures are
    accumulated into st.session_state.embedding_errors so they're visible
    to the operator (instead of silently passing through as a quiet
    partial success, the old behaviour).
    """
    owner_id = st.session_state.owner_id or None
    tasks: list[dict] = []
    rejected: list[str] = []
    errors: list[str] = []

    def _record_error(label, exc):
        msg = f"{label}: {type(exc).__name__}: {exc}"
        errors.append(msg)
        if "embedding_errors" not in st.session_state:
            st.session_state.embedding_errors = []
        st.session_state.embedding_errors.append(msg)

    # ── Stage 1: validate + extract/chunk ───────────────────────────
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
            ctype = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
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

    # ── Stage 2: embed in parallel ──────────────────────────────────
    progress_bar.progress(0.05, text=f"Embedding {len(tasks)} chunk(s)…")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.EMBED_MAX_WORKERS) as ex:
        futures = [ex.submit(_embed_task, t, _record_error) for t in tasks]
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            progress_bar.progress(done / len(tasks),
                                  text=f"Embedded {done}/{len(tasks)} chunks…")
            results.append(fut.result())

    # ── Stage 3: upsert with dedup ──────────────────────────────────
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
    if errors:
        with status_text.expander(f"⚠️ {len(errors)} embedding warning(s)", expanded=False):
            for msg in errors[:25]:
                st.caption(msg)
    return inserted, rejected


# ──────────────────────────────────────────────────────────────────────
#  Retrieval (cached) + image fetch
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=config.QUERY_CACHE_TTL, show_spinner=False)
def cached_retrieve(query, model_name, threshold, filter_type, owner_id):
    """Cached hybrid retrieval. Cleared after indexing."""
    return retrieval.hybrid_retrieve(query, model_name, threshold=threshold,
                                     filter_type=filter_type, owner_id=owner_id)


def _load_image_from_url(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))
    except Exception:
        return None


def preload_image_parts(image_rows: list[dict]) -> list[Image.Image]:
    """Fetch all cited images concurrently (Phase 3.3: off the critical path).

    Uses a small thread pool so the total wait is bounded by the slowest
    image rather than the sum. Returns only successfully loaded PIL images;
    the caller treats an empty list as "no image" — generation still works.
    """
    urls = [r.get("image_path") for r in image_rows if r.get("image_path")]
    if not urls:
        return []
    out: list[Image.Image] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(urls))) as ex:
        for img in ex.map(_load_image_from_url, urls):
            if img is not None:
                out.append(img)
    return out


# ──────────────────────────────────────────────────────────────────────
#  Feedback guidance (per-session)
# ──────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
#  Generation
# ──────────────────────────────────────────────────────────────────────
def _generate(prompt, image_parts, meta):
    try:
        client = get_gemini_client()
        if client is None:
            return "⚠️ Gemini API key not configured.", 0, 0
        res = client.models.generate_content(
            model=config.GENERATION_MODEL, contents=[prompt] + image_parts)
        in_tok = out_tok = 0
        if hasattr(res, "usage_metadata") and res.usage_metadata:
            in_tok = res.usage_metadata.prompt_token_count or 0
            out_tok = res.usage_metadata.candidates_token_count or 0
        meta["input_tokens"] = in_tok
        meta["output_tokens"] = out_tok
        return (getattr(res, "text", "") or ""), in_tok, out_tok
    except Exception as e:                                  # noqa: BLE001
        return f"Generation error: {e}", 0, 0


# ──────────────────────────────────────────────────────────────────────
#  The full pipeline
# ──────────────────────────────────────────────────────────────────────
def run_rag_pipeline(user_query: str, model_name: str) -> tuple[str, dict, list[dict]]:
    """Route → retrieve → rerank-gate → dedup → knapsack → generate.

    Returns (response_text, meta, used_rows). ``meta`` carries the per-query
    diagnostics surfaced by the dev panel; ``used_rows`` are the chunk dicts
    that actually made it into the LLM context.
    """
    threshold = float(st.session_state.threshold)
    meta: dict[str, Any] = {
        "retrieval_time": 0.0, "generation_time": 0.0,
        "context_chunks": 0, "input_tokens": 0, "output_tokens": 0,
        "intent": "document", "threshold": threshold,
        "scores": [], "fallback": False, "guardrail_blocked": False,
    }
    owner_id = st.session_state.owner_id or None

    # ── Input guard (Phase 2.2) ──────────────────────────────────────
    ok, reason = guardrails.input_guard(user_query, user_key=owner_id or "anon")
    if not ok:
        meta["guardrail_blocked"] = reason
        return _guardrail_message(reason), meta, []

    # ── History + feedback context ──────────────────────────────────
    prior_messages = st.session_state.messages[:-1] if st.session_state.messages else []
    history_block = conversation.build_history_block(prior_messages, st.session_state.summary)
    feedback_block = build_feedback_guidance()

    # ── Intent routing ──────────────────────────────────────────────
    intent = routing.classify_intent(user_query)
    meta["intent"] = intent

    # ── General: skip RAG, answer directly ──────────────────────────
    if intent == "general":
        t1 = time.time()
        parts = [p for p in [history_block, feedback_block, user_query] if p]
        prompt = "\n\n".join(parts)
        text, in_tok, out_tok = _generate(prompt, [], meta)
        meta["generation_time"] = time.time() - t1
        return text, meta, []

    # ── Retrieve (parallel vector + keyword) ────────────────────────
    filter_type = routing.intent_to_filter(intent)
    t0 = time.time()
    try:
        rows = cached_retrieve(user_query, model_name, threshold, filter_type, owner_id)
    except Exception as e:                                  # noqa: BLE001
        st.warning(SCHEMA_HELP if is_schema_error(e) else f"Retrieval error: {e}")
        rows = []
    meta["retrieval_time"] = time.time() - t0
    meta["scores"] = [round(r.get("rerank_score", r.get("similarity", 0)), 3) for r in rows]

    # ── Safe fallback when nothing is relevant ─────────────────────
    if not rows:
        meta["fallback"] = True
        return config.FALLBACK_MESSAGE, meta, []

    # ── Dedup → knapsack → context ─────────────────────────────────
    rows = context_builder.semantic_dedup(rows)
    chosen = context_builder.knapsack_select(rows)
    meta["context_chunks"] = len(chosen)
    context_str, image_rows, used = context_builder.build_context(chosen)

    # ── Preload cited images concurrently (Phase 3.3) ──────────────
    image_parts = preload_image_parts(image_rows)

    # ── Build prompt + generate ────────────────────────────────────
    toon = context_builder.toon_metadata(chosen)
    prompt_parts: list[str] = []
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
    response_text, _in, _out = _generate("\n\n".join(prompt_parts), image_parts, meta)
    meta["generation_time"] = time.time() - t1

    # ── Append Sources section + optional faithfulness check ───────
    sources = context_builder.format_sources(used)
    if sources:
        response_text = f"{response_text}\n{sources}"

    # Optional LLM-as-judge output faithfulness check (Phase 2.2). Off by
    # default for latency; flipped on by `OUTPUT_FAITHFULNESS_CHECK=true`.
    cited_sources = [f"{r.get('file_name','')}: {(r.get('content','') or '')[:300]}" for r in used]
    verdict = guardrails.output_faithfulness_check(response_text, cited_sources)
    if verdict and isinstance(verdict, dict) and verdict.get("score", 1.0) < 0.4:
        meta["faithfulness_warning"] = verdict.get("reason", "low faithfulness")

    return response_text, meta, used


def _guardrail_message(reason: str) -> str:
    if reason == "empty_query":
        return "Please type a question."
    if reason == "query_too_long":
        return f"That question is over the {config.GUARDRAIL_MAX_QUERY_CHARS}-character limit. Try a shorter version."
    if reason == "injection_suspected":
        return "I can't follow that instruction. Please rephrase your question about the uploaded documents."
    if reason == "rate_limited":
        return "You've sent a lot of questions very quickly — please slow down a little."
    return f"Request blocked by safety guardrail ({reason})."


# ──────────────────────────────────────────────────────────────────────
#  History persistence
# ──────────────────────────────────────────────────────────────────────
def save_chat_history():
    sid = st.session_state.current_session_id
    try:
        rag_db.save_session(sid, st.session_state.embedding_model,
                            st.session_state.total_input_tokens,
                            st.session_state.total_output_tokens,
                            st.session_state.total_queries,
                            st.session_state.summary)
        rag_db.save_messages(sid, st.session_state.messages)
    except Exception as e:                                  # noqa: BLE001
        st.warning(f"Could not save history: {e}")
