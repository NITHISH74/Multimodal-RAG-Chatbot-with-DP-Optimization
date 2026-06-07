"""
Hybrid retrieval (Phase 4) + Rerank-Lite (Phase 5).

Pipeline:
    embed query -> vector search (threshold-filtered) + keyword search
    -> merge & dedup by chunk id -> rerank-lite -> top-K chunks.

The merge and rerank functions are pure (no I/O) so they can be unit-tested
without a database or API keys.
"""
import re
from datetime import datetime, timezone

import config
import rag_db
from embeddings import embed_query

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Rerank-lite signal weights (Phase 5).
W_SIMILARITY = 0.60   # semantic vector score
W_KEYWORD = 0.25      # query/chunk keyword overlap
W_RECENCY = 0.15      # recently uploaded files rank higher


def _tokens(text):
    return set(t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1)


def merge_results(vector_rows, keyword_rows):
    """Merge vector + keyword hits, dedup by chunk id (Phase 4.3).

    Each merged row carries both ``similarity`` (0 if keyword-only) and
    ``keyword_rank`` (0 if vector-only).
    """
    merged = {}
    for row in vector_rows:
        r = dict(row)
        r.setdefault("similarity", 0.0)
        r.setdefault("keyword_rank", 0.0)
        merged[r["id"]] = r
    for row in keyword_rows:
        rid = row["id"]
        if rid in merged:
            merged[rid]["keyword_rank"] = row.get("keyword_rank", 0.0)
        else:
            r = dict(row)
            r.setdefault("similarity", 0.0)
            r.setdefault("keyword_rank", row.get("keyword_rank", 0.0))
            merged[rid] = r
    return list(merged.values())


def _recency_scores(rows):
    """Map each row id -> 0..1 recency score (newest = 1)."""
    times = {}
    for r in rows:
        ts = r.get("upload_date")
        dt = None
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                dt = None
        if dt is not None:
            times[r["id"]] = dt.timestamp()
    if not times:
        return {r["id"]: 0.0 for r in rows}
    lo, hi = min(times.values()), max(times.values())
    span = (hi - lo) or 1.0
    return {r["id"]: ((times[r["id"]] - lo) / span if r["id"] in times else 0.0) for r in rows}


def rerank_lite(query, rows, top_k=None):
    """Score and sort candidates by combined signals; return top_k (Phase 5)."""
    top_k = top_k or config.RERANK_TOP_K
    if not rows:
        return []

    q_tokens = _tokens(query)
    max_kw_rank = max((r.get("keyword_rank", 0.0) for r in rows), default=0.0) or 1.0
    recency = _recency_scores(rows)

    scored = []
    for r in rows:
        sim = max(0.0, min(1.0, r.get("similarity", 0.0)))
        overlap = (len(q_tokens & _tokens(r.get("content", ""))) / len(q_tokens)) if q_tokens else 0.0
        kw = (r.get("keyword_rank", 0.0) / max_kw_rank)
        keyword_signal = max(overlap, kw)  # explicit overlap OR DB ts_rank
        score = (W_SIMILARITY * sim
                 + W_KEYWORD * keyword_signal
                 + W_RECENCY * recency.get(r["id"], 0.0))
        out = dict(r)
        out["rerank_score"] = score
        out["keyword_overlap"] = overlap
        scored.append(out)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]


def hybrid_retrieve(query, model_name, threshold=None, filter_type=None, owner_id=None):
    """Full hybrid retrieval. Returns reranked top-K chunk rows.

    Each returned row has: id, content, file_name, document_type,
    page_number, source_url, image_path, similarity, keyword_rank,
    rerank_score.
    """
    threshold = config.SIMILARITY_THRESHOLD if threshold is None else threshold
    query_vec = embed_query(query, model_name)

    vector_rows = rag_db.vector_search(
        query_vec, model_name, threshold, config.RETRIEVAL_MATCH_COUNT,
        filter_type=filter_type, owner_id=owner_id)
    keyword_rows = rag_db.keyword_search(
        query, config.KEYWORD_MATCH_COUNT, filter_type=filter_type, owner_id=owner_id)

    merged = merge_results(vector_rows, keyword_rows)
    return rerank_lite(query, merged)
