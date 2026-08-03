"""
Central configuration for the Multi-Model RAG chatbot.

Every tunable lives here so behaviour can be changed without hunting through
app.py. Values fall back to sensible defaults but can be overridden via
environment variables / Streamlit Cloud secrets.
"""
import os

try:
    import streamlit as st
except Exception:  # streamlit absent (e.g. tests / CLI tools)
    st = None


def _from_secrets(name):
    """Read a value from Streamlit secrets, if available.

    Accessing st.secrets triggers Streamlit's lazy parse, which also mirrors
    string/int/float secrets into os.environ. We read it directly so that:
      * booleans (e.g. crawl_allow_all = true) are honoured — Streamlit never
        mirrors bool secrets into os.environ, so os.getenv would miss them;
      * values are available at import time even before any other st.secrets
        access has triggered the mirror.
    Guarded broadly because st.secrets raises when no secrets file exists.
    """
    if st is None:
        return None
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return None


def _get(name, default):
    # Real env vars win (local dev / Docker), then Streamlit secrets, then default.
    val = os.getenv(name)
    if val not in (None, ""):
        return val
    sval = _from_secrets(name)
    if sval not in (None, ""):
        return sval
    return default


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
JUDGE_MODEL = _get("judge_model", "gemini-2.5-flash")

GEMINI_DIM = 3072
COHERE_DIM = 1536

# ── Retrieval (hybrid + rerank) ─────────────────────────────────────
RETRIEVAL_MATCH_COUNT = _int("retrieval_match_count", 10)      # candidates from vector search
KEYWORD_MATCH_COUNT = _int("keyword_match_count", 10)          # candidates from full-text search
# Min cosine similarity for a chunk to count. NOTE: Cohere embed-v4 / Gemini
# embeddings produce LOW cosine values — measured relevant chunks score ~0.15-0.35
# (verified: self-similarity is 1.0, but query↔doc relevance lands low). 0.70
# filters out everything. 0.15 is a practical default; tune via the Master
# Settings panel.
SIMILARITY_THRESHOLD = _float("similarity_threshold", 0.15)
QUERY_CACHE_TTL = _int("query_cache_ttl", 600)                 # seconds
EMBED_MAX_WORKERS = _int("embed_max_workers", 5)
RERANK_TOP_K = _int("rerank_top_k", 6)                         # chunks to the LLM

# ── Relevance-gated fallback (strengthened: independent thresholds) ─
# A chunk must clear the SIMILARITY_THRESHOLD AND a separate floor on its
# Rerank-Lite score. This fixes the "one shared keyword beats a threshold-
# passing chunk" bug — vector threshold and composite score are now gated
# independently.
MIN_RERANK_SCORE = _float("min_rerank_score", 0.10)
# `keyword_search` returns ts_rank — drop trivially-low keyword hits.
MIN_KEYWORD_RANK = _float("min_keyword_rank", 0.05)

# ── Multimodal mode ─────────────────────────────────────────────────
# auto   — always pull top-1 image by cross-modal similarity alongside text
# keyword — legacy: only return image chunks when the query asks for one
# off    — text-only retrieval
MULTIMODAL_MODE = _get("multimodal_mode", "auto")  # auto|keyword|off
AUTO_IMAGE_TOP_K = _int("auto_image_top_k", 1)     # how many images in auto mode

# ── Chunking ────────────────────────────────────────────────────────
CHUNK_TARGET_CHARS = _int("chunk_target_chars", 1200)          # ~300 tokens
CHUNK_OVERLAP_CHARS = _int("chunk_overlap_chars", 150)
MIN_CHUNK_CHARS = _int("min_chunk_chars", 20)                  # garbage filter floor
# Min fraction of "information-dense" characters in a chunk. Drops chunks
# that are mostly whitespace / punctuation / boilerplate.
MIN_INFORMATION_DENSITY = _float("min_information_density", 0.35)

# ── Context building (knapsack) ─────────────────────────────────────
MAX_CONTEXT_TOKENS = _int("max_context_tokens", 1500)          # token budget for context
CHARS_PER_TOKEN = 4                                            # rough estimate
SEMANTIC_DEDUP_THRESHOLD = _float("semantic_dedup_threshold", 0.92)  # cosine for near-dupes

# ── Uploads ────────────────────────────────────────────────────────
MAX_UPLOAD_MB = _int("max_upload_mb", 50)
ALLOWED_DOC_EXTS = ("pdf", "docx", "pptx", "txt", "md")
ALLOWED_IMG_EXTS = ("png", "jpg", "jpeg", "webp")
ALLOWED_EXTS = ALLOWED_DOC_EXTS + ALLOWED_IMG_EXTS

# ── Image storage ──────────────────────────────────────────────────
SUPABASE_IMAGE_BUCKET = _get("supabase_image_bucket", "rag-images")

# ── DB bootstrap (in-app "Initialize Database" button) ─────────────
# Supabase Session Pooler connection URI (IPv4-compatible, supports DDL):
#   postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres
# Sensitive — keep in secrets only, never log it.
SUPABASE_DB_URL = _get("supabase_db_url", "")

# Opt-in: route read queries through the anon key (least-privilege). Only enable
# AFTER applying RLS read policies (migration 0005) and granting anon SELECT —
# otherwise anon returns 0 rows and search silently breaks. Off by default so the
# server-side service client handles reads.
USE_ANON_READS = _bool("use_anon_reads", False)

# ── Web crawl ──────────────────────────────────────────────────────
# Comma-separated allowlist of domains permitted for manual crawling.
# Empty list => nothing allowed until the user configures it.
CRAWL_ALLOWED_DOMAINS = tuple(
    d.strip().lower() for d in _get("crawl_allowed_domains", "").split(",") if d.strip()
)
# Escape hatch: set crawl_allow_all=true to permit ANY domain (robots.txt is
# still respected). Use with care — disables the domain allowlist safeguard.
CRAWL_ALLOW_ALL = _bool("crawl_allow_all", False)
CRAWL_TIMEOUT = _int("crawl_timeout", 30)
CRAWL_USER_AGENT = _get("crawl_user_agent", "MultiModalRAGBot/1.0 (+manual-crawl)")
CRAWL_MAX_BYTES = _int("crawl_max_bytes", 5_000_000)
# Multi-page ("Entire website") crawl bounds — same-domain BFS via Crawl4AI.
CRAWL_MAX_PAGES_DEFAULT = _int("crawl_max_pages", 25)
CRAWL_MAX_PAGES_LIMIT = _int("crawl_max_pages_limit", 100)

# ── History (running summary) ──────────────────────────────────────
HISTORY_SUMMARY_TRIGGER = _int("history_summary_trigger", 6)   # turns before summarizing
HISTORY_RECENT_TURNS = _int("history_recent_turns", 2)         # verbatim recent turns kept

# ── Anti-hallucination fallback ────────────────────────────────────
# Clean English by default — the Tamil/English code-switched version reads
# as unfinished to most reviewers.
FALLBACK_MESSAGE = _get(
    "fallback_message",
    "I couldn't find this in the uploaded documents. "
    "Try rephrasing the question, or check that the relevant file has been indexed.",
)

# ── Guardrails ─────────────────────────────────────────────────────
GUARDRAIL_MAX_QUERY_CHARS = _int("guardrail_max_query_chars", 4000)
GUARDRAIL_RATE_LIMIT_PER_MIN = _int("guardrail_rate_limit_per_min", 30)
# Toggle the LLM-as-judge output faithfulness check (off by default to keep
# p50 latency low; flip on for portfolio demos / sensitive domains).
OUTPUT_FAITHFULNESS_CHECK = _bool("output_faithfulness_check", False)

# ── Embedding retry / backoff ─────────────────────────────────────
EMBED_RETRY_MAX = _int("embed_retry_max", 2)
EMBED_RETRY_BASE_DELAY = _float("embed_retry_base_delay", 0.6)   # seconds

# ── Dev / admin ────────────────────────────────────────────────────
DEV_MODE = _bool("dev_mode", False)
