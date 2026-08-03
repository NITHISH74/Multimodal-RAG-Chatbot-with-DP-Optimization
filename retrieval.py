"""
Hybrid retrieval + Rerank-Lite.

Pipeline:
    embed query
      ├─ (parallel)  vector_search  (threshold-filtered)
      └─ (parallel)  keyword_search  (rank-floor filtered)
    → merge & dedup by chunk id
    → rerank-lite
    → top-K chunks.

The merge and rerank functions are pure (no I/O) so they can be unit-tested
without a database or API keys.
"""
from __future__ import annotations

import concurrent.futures
import re
from datetime import datetime, timezone

import config
import rag_db
from embeddings import embed_query

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Rerank-lite signal weights
W_SIMILARITY = 0.60   # semantic vector score
W_KEYWORD = 0.25      # query/chunk keyword overlap
W_RECENCY = 0.15      # recently uploaded files rank higher


def _tokens(text):
    return set(t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1)


def merge_results(vector_rows, keyword_rows):
    """Merge vector + keyword hits, dedup by chunk id.

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
    """Score and sort candidates by combined signals; return top_k."""
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


def _passes_gating(row, min_rerank=None, min_keyword=None):
    """Strengthened fallback gate. A chunk survives if:

    * it has a *vector similarity* above the configured threshold (or is
      keyword-only with a ts_rank above the floor), AND
    * its rerank-lite score is above the separate composite floor.
    """
    sim = float(row.get("similarity") or 0.0)
    rank = float(row.get("keyword_rank") or 0.0)
    min_rerank = config.MIN_RERANK_SCORE if min_rerank is None else min_rerank
    min_keyword = config.MIN_KEYWORD_RANK if min_keyword is None else min_keyword

    # Vector-only: must clear both the vector threshold AND the rerank floor.
    if sim > 0.0 and rank <= 0.0:
        return sim >= config.SIMILARITY_THRESHOLD and (
            row.get("rerank_score", sim) >= min_rerank
        )
    # Keyword-only: must clear the keyword rank floor.
    if sim <= 0.0 and rank > 0.0:
        return rank >= min_keyword
    # Both: vector similarity AND rerank floor.
    return sim >= config.SIMILARITY_THRESHOLD and (
        row.get("rerank_score", sim) >= min_rerank
    )


def _drop_below_gate(rows):
    """Apply the gating floor to a reranked list. Rows without rerank_score
    (e.g. pre-rerank) get a neutral default so the rerank step still wins."""
    for r in rows:
        if "rerank_score" not in r:
            r["rerank_score"] = max(
                float(r.get("similarity") or 0.0),
                float(r.get("keyword_rank") or 0.0),
            )
    return [r for r in rows if _passes_gating(r)]


def _vector_search_job(args):
    query_vec, model_name, threshold, match_count, filter_type, owner_id = args
    return rag_db.vector_search(
        query_vec, model_name, threshold, match_count,
        filter_type=filter_type, owner_id=owner_id,
    )


def _keyword_search_job(args):
    query_text, match_count, filter_type, owner_id = args
    return rag_db.keyword_search(
        query_text, match_count, filter_type=filter_type, owner_id=owner_id,
    )


def _fan_out_search(query_vec, query_text, model_name, threshold,
                    filter_type=None, owner_id=None):
    """Run vector + keyword searches in parallel (Phase 3.1).

    Roughly halves retrieval wall time vs the old sequential version. Uses
    a small thread pool (2 workers — there's nothing else to parallelize
    within a single retrieve call).
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        v_fut = ex.submit(_vector_search_job,
                          (query_vec, model_name, threshold,
                           config.RETRIEVAL_MATCH_COUNT, filter_type, owner_id))
        k_fut = ex.submit(_keyword_search_job,
                          (query_text, config.KEYWORD_MATCH_COUNT,
                           filter_type, owner_id))
        return v_fut.result(), k_fut.result()


def _maybe_attach_image(rows, query, model_name, owner_id=None):
    """Multimodal auto mode: if the user *didn't* explicitly ask for an image,
    still pull the top image chunk by cross-modal similarity (Phase 2.3).

    Returns a new list with the image chunk appended (or inserted near the
    top of the rerank order). No-op when the mode is keyword/off or when
    an image is already in the candidate set.
    """
    mode = (getattr(config, "MULTIMODAL_MODE", "auto") or "auto").lower()
    if mode == "off":
        return rows
    if any(r.get("document_type") == "image" or r.get("type") == "image" for r in rows):
        return rows  # already pulled an image (image-intent)
    if mode == "keyword":
        return rows  # legacy behavior — don't proactively pull images

    # auto: pull top-K image chunks by cross-modal similarity
    try:
        from embeddings import embed_query
        qv = embed_query(query, model_name)
        image_rows = rag_db.vector_search(
            qv, model_name,
            config.SIMILARITY_THRESHOLD,
            max(1, config.AUTO_IMAGE_TOP_K) * 3,  # over-fetch, take top-K after rerank
            filter_type="image", owner_id=owner_id,
        )
    except Exception:
        return rows
    if not image_rows:
        return rows
    image_rows = image_rows[: max(1, config.AUTO_IMAGE_TOP_K)]
    return list(rows) + image_rows


def hybrid_retrieve(query, model_name, threshold=None, filter_type=None,
                    owner_id=None, return_ranked=False):
    """Full hybrid retrieval. Returns reranked top-K chunk rows.

    Each returned row has: id, content, file_name, document_type,
    page_number, source_url, image_path, similarity, keyword_rank,
    rerank_score.

    When ``return_ranked=True`` the function also returns the full reranked
    list (pre-knapsack) so callers like the eval harness can compute their
    own metrics on the candidate ranking.
    """
    threshold = config.SIMILARITY_THRESHOLD if threshold is None else threshold
    query_vec = embed_query(query, model_name)

    vector_rows, keyword_rows = _fan_out_search(
        query_vec, query, model_name, threshold,
        filter_type=filter_type, owner_id=owner_id,
    )

    merged = merge_results(vector_rows, keyword_rows)
    ranked = rerank_lite(query, merged)
    gated = _drop_below_gate(ranked)
    with_image = _maybe_attach_image(gated, query, model_name, owner_id=owner_id)

    if return_ranked:
        return with_image, ranked
    return with_image
