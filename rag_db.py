"""
Supabase data-access layer: chunk upsert (with dedup), image storage,
vector + keyword search RPCs, and chat history.

All privileged DB access lives here and runs server-side only (Phase 15).
"""
import io
import uuid

import config
from clients import get_supabase_client


# ──────────────────────────────────────────────────────────────────────
#  Indexing
# ──────────────────────────────────────────────────────────────────────
def chunk_hash_exists(content_hash):
    """Return existing row id for this content hash, else None (Phase 2.3)."""
    sb = get_supabase_client()
    res = sb.table("documents").select("id").eq("content_hash", content_hash).limit(1).execute()
    return res.data[0]["id"] if res.data else None


def upsert_chunk(chunk, vector, model_name, image_path=None, owner_id=None):
    """Insert a chunk row, or fill in the missing model embedding if the
    same content hash already exists. Returns ("inserted"|"updated"|"skipped").
    """
    sb = get_supabase_client()
    embed_col = "embedding_gemini" if model_name == "Gemini" else "embedding_cohere"

    existing_id = chunk_hash_exists(chunk["content_hash"])
    if existing_id is not None:
        # Row exists (likely from the other embedding model). Only set our
        # embedding column if it is not already populated.
        row = sb.table("documents").select(embed_col).eq("id", existing_id).limit(1).execute()
        if row.data and row.data[0].get(embed_col) is not None:
            return "skipped"
        update = {embed_col: vector}
        if image_path:
            update["image_path"] = image_path
        sb.table("documents").update(update).eq("id", existing_id).execute()
        return "updated"

    record = {
        "content": chunk["content"] or "Visual content",
        "file_name": chunk["file_name"],
        "document_type": chunk["document_type"],
        "page_number": chunk.get("page_number"),
        "chunk_index": chunk.get("chunk_index", 0),
        "content_hash": chunk["content_hash"],
        "source_url": chunk.get("source_url"),
        "image_path": image_path,
        "owner_id": owner_id,
        # Keep a lightweight metadata blob for backward compatibility.
        "metadata": {
            "file": chunk["file_name"],
            "type": chunk["type"],
            "page_number": chunk.get("page_number"),
            "chunk_index": chunk.get("chunk_index", 0),
        },
        embed_col: vector,
    }
    sb.table("documents").insert(record).execute()
    return "inserted"


def file_already_indexed(file_name, model_name, owner_id=None):
    """True if any chunk of this file already has this model's embedding."""
    sb = get_supabase_client()
    embed_col = "embedding_gemini" if model_name == "Gemini" else "embedding_cohere"
    q = sb.table("documents").select(f"id,{embed_col}").eq("file_name", file_name)
    if owner_id:
        q = q.eq("owner_id", owner_id)
    res = q.execute()
    return any(r.get(embed_col) is not None for r in res.data)


# ──────────────────────────────────────────────────────────────────────
#  Image storage (Phase 8)
# ──────────────────────────────────────────────────────────────────────
def upload_image(file_bytes, file_name, content_type="image/png"):
    """Upload image bytes to the Supabase Storage bucket and return its
    public URL. Falls back to None on failure (caller can degrade)."""
    sb = get_supabase_client()
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "png"
    object_path = f"{uuid.uuid4().hex}.{ext}"
    try:
        sb.storage.from_(config.SUPABASE_IMAGE_BUCKET).upload(
            object_path, file_bytes,
            {"content-type": content_type, "upsert": "true"},
        )
        return sb.storage.from_(config.SUPABASE_IMAGE_BUCKET).get_public_url(object_path)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────
#  Retrieval RPCs (Phase 3 / 4)
# ──────────────────────────────────────────────────────────────────────
def vector_search(query_vec, model_name, threshold, match_count,
                  filter_type=None, owner_id=None):
    sb = get_supabase_client()
    rpc = "match_documents_gemini" if model_name == "Gemini" else "match_documents_cohere"
    res = sb.rpc(rpc, {
        "query_embedding": query_vec,
        "match_threshold": threshold,
        "match_count": match_count,
        "filter_type": filter_type,
        "filter_owner": owner_id,
    }).execute()
    return res.data or []


def keyword_search(query_text, match_count, filter_type=None, owner_id=None):
    sb = get_supabase_client()
    res = sb.rpc("keyword_search", {
        "query_text": query_text,
        "match_count": match_count,
        "filter_type": filter_type,
        "filter_owner": owner_id,
    }).execute()
    return res.data or []


# ──────────────────────────────────────────────────────────────────────
#  Chat history
# ──────────────────────────────────────────────────────────────────────
def save_session(session_id, embedding_model, in_tok, out_tok, queries, summary=""):
    sb = get_supabase_client()
    sb.table("chat_sessions").upsert({
        "session_id": session_id,
        "embedding_model": embedding_model,
        "total_input_tokens": in_tok,
        "total_output_tokens": out_tok,
        "total_queries": queries,
    }).execute()


def save_messages(session_id, messages):
    sb = get_supabase_client()
    sb.table("chat_messages").delete().eq("session_id", session_id).execute()
    rows = [{
        "session_id": session_id, "role": m["role"],
        "content": m["content"], "meta": m.get("meta", {}),
    } for m in messages]
    if rows:
        sb.table("chat_messages").insert(rows).execute()


def load_sessions(limit=20):
    sb = get_supabase_client()
    sessions = sb.table("chat_sessions").select("*").order(
        "created_at", desc=True).limit(limit).execute().data
    for s in sessions:
        msgs = sb.table("chat_messages").select("*").eq(
            "session_id", s["session_id"]).order("id").execute().data
        s["messages"] = msgs
        s["timestamp"] = s["created_at"]
    return sessions
