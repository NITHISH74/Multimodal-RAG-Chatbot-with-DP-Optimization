"""
Central configuration for the Multi-Model RAG chatbot.

Every tunable lives here so behaviour can be changed without hunting through
app.py. Values fall back to sensible defaults but can be overridden via
environment variables / Streamlit Cloud secrets.
"""
import os


def _get(name, default):
    val = os.getenv(name)
    return val if val not in (None, "") else default


def _int(name, default):
    try:
        return int(_get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name, default):
    try:
        return float(_get(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name, default):
    val = _get(name, None)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


# ── Models ───────────────────────────────────────────────────────────
GEMINI_EMBED_MODEL = _get("gemini_embed_model", "gemini-embedding-2-preview")
COHERE_EMBED_MODEL = _get("cohere_embed_model", "embed-v4.0")
GENERATION_MODEL = _get("generation_model", "gemini-2.5-flash")
SUMMARY_MODEL = _get("summary_model", "gemini-2.5-flash")

GEMINI_DIM = 3072
COHERE_DIM = 1536

# ── Retrieval / performance (Phase 1, 3, 4, 5) ───────────────────────
RETRIEVAL_MATCH_COUNT = _int("retrieval_match_count", 10)      # candidates from vector search
KEYWORD_MATCH_COUNT = _int("keyword_match_count", 10)          # candidates from full-text search
SIMILARITY_THRESHOLD = _float("similarity_threshold", 0.70)    # Phase 3 default
QUERY_CACHE_TTL = _int("query_cache_ttl", 600)                 # seconds
EMBED_MAX_WORKERS = _int("embed_max_workers", 5)
RERANK_TOP_K = _int("rerank_top_k", 6)                         # 5-7 chunks to the LLM (Phase 5)

# ── Chunking (Phase 2) ───────────────────────────────────────────────
CHUNK_TARGET_CHARS = _int("chunk_target_chars", 1200)          # ~300 tokens
CHUNK_OVERLAP_CHARS = _int("chunk_overlap_chars", 150)
MIN_CHUNK_CHARS = _int("min_chunk_chars", 20)                  # garbage filter floor

# ── Context building (Phase 6) ───────────────────────────────────────
MAX_CONTEXT_CHARS = _int("max_context_chars", 5000)            # DP knapsack capacity
MAX_CONTEXT_TOKENS = _int("max_context_tokens", 1500)          # token budget for context
CHARS_PER_TOKEN = 4                                            # rough estimate
SEMANTIC_DEDUP_THRESHOLD = _float("semantic_dedup_threshold", 0.92)  # cosine for near-dupes

# ── Uploads (Phase 9) ────────────────────────────────────────────────
MAX_UPLOAD_MB = _int("max_upload_mb", 20)
ALLOWED_DOC_EXTS = ("pdf", "docx", "pptx", "txt", "md")
ALLOWED_IMG_EXTS = ("png", "jpg", "jpeg", "webp")
ALLOWED_EXTS = ALLOWED_DOC_EXTS + ALLOWED_IMG_EXTS

# ── Image storage (Phase 8) ──────────────────────────────────────────
SUPABASE_IMAGE_BUCKET = _get("supabase_image_bucket", "rag-images")

# ── Web crawl (Phase 10) ─────────────────────────────────────────────
# Comma-separated allowlist of domains permitted for manual crawling.
# Empty list => nothing allowed until the user configures it.
CRAWL_ALLOWED_DOMAINS = tuple(
    d.strip().lower() for d in _get("crawl_allowed_domains", "").split(",") if d.strip()
)
CRAWL_TIMEOUT = _int("crawl_timeout", 15)
CRAWL_USER_AGENT = _get("crawl_user_agent", "MultiModalRAGBot/1.0 (+manual-crawl)")
CRAWL_MAX_BYTES = _int("crawl_max_bytes", 5_000_000)

# ── History / token optimization (Phase 13) ──────────────────────────
HISTORY_SUMMARY_TRIGGER = _int("history_summary_trigger", 6)   # turns before summarizing
HISTORY_RECENT_TURNS = _int("history_recent_turns", 2)         # verbatim recent turns kept

# ── Anti-hallucination fallback (Phase 12) ───────────────────────────
FALLBACK_MESSAGE = _get(
    "fallback_message",
    "Uploaded documentsல் இதற்கான clear information கிடைக்கவில்லை. "
    "(Clear information for this was not found in the uploaded documents.)",
)

# ── Dev / admin (Phase 14) ───────────────────────────────────────────
DEV_MODE = _bool("dev_mode", False)
