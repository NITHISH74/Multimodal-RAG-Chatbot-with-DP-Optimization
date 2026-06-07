"""
Lightweight intent routing (Phase 11) — no agents, no autonomous loops.

Classifies a query into one of:
    "image"    -> restrict retrieval to image chunks
    "web"      -> restrict retrieval to web-crawled chunks
    "general"  -> skip RAG entirely, answer directly from the LLM
    "document" -> default: search document chunks (also used for "all")

Keyword rules first (fast, free, deterministic). An optional LLM classifier
is available but off by default.
"""
import re

_URL_RE = re.compile(r"https?://", re.I)

_IMAGE_KW = {"image", "images", "picture", "photo", "photograph", "diagram",
             "figure", "screenshot", "visual", "chart", "graph", "logo", "icon"}
_WEB_KW = {"website", "webpage", "web page", "url", "link", "crawl", "crawled",
           "site"}
_GENERAL_PATTERNS = [
    r"^\s*(hi|hello|hey|yo|hola|vanakkam)\b",
    r"\b(how are you|who are you|what can you do|your name)\b",
    r"\b(thanks|thank you|thx|bye|goodbye)\b",
    r"^\s*(translate|write a poem|tell me a joke|what is \d)",
    r"\b(answer (this )?(generally|directly|without (the )?documents))\b",
]
_GENERAL_RE = [re.compile(p, re.I) for p in _GENERAL_PATTERNS]


def classify_intent(query):
    q = (query or "").strip().lower()
    if not q:
        return "general"

    for rx in _GENERAL_RE:
        if rx.search(q):
            return "general"

    if _URL_RE.search(query) or any(k in q for k in _WEB_KW):
        return "web"

    words = set(re.findall(r"[a-z]+", q))
    if words & _IMAGE_KW:
        return "image"

    return "document"


def intent_to_filter(intent):
    """Map an intent to a document_type DB filter (None = no filter)."""
    return {
        "image": "image",
        "web": "web",
        "document": None,   # search across all document chunks
        "general": None,
    }.get(intent)
