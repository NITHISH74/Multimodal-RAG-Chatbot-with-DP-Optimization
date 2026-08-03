"""
Rank-aware retrieval metrics — pure Python, zero dependencies.

Every metric takes a ranked list of retrieved item ids and a set (or list) of
ground-truth relevant ids, and returns a float. All functions are unit-tested
against known vectors in `tests/test_metrics.py`.

Conventions
-----------
* A *higher* score is always better.
* Ties in the ranked list are broken by the order the caller supplies — these
  metrics do not re-sort. (If you want to be conservative, dedupe + sort first.)
* For ranking metrics the *position* of the first hit matters; for Recall@k
  only the membership in the top-k window matters.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


# ──────────────────────────────────────────────────────────────────────
#  Set / list helpers
# ──────────────────────────────────────────────────────────────────────
def _hits(ranked: Sequence, relevant: set) -> list[int]:
    """1/0 indicator list aligned to the ranked order."""
    return [1 if x in relevant else 0 for x in ranked]


# ──────────────────────────────────────────────────────────────────────
#  Hit Rate @ k — is at least one relevant item in the top k?
# ──────────────────────────────────────────────────────────────────────
def hit_rate_at_k(ranked: Sequence, relevant: Iterable, k: int) -> float:
    """1.0 if any relevant item appears in the top k, else 0.0."""
    rel = set(relevant)
    if not rel:
        return 0.0
    return 1.0 if any(x in rel for x in ranked[:k]) else 0.0


# ──────────────────────────────────────────────────────────────────────
#  Recall @ k — fraction of the relevant set that appears in the top k
# ──────────────────────────────────────────────────────────────────────
def recall_at_k(ranked: Sequence, relevant: Iterable, k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    top = set(ranked[:k])
    return len(top & rel) / len(rel)


# ──────────────────────────────────────────────────────────────────────
#  MRR — Mean Reciprocal Rank (1 / rank_of_first_relevant)
# ──────────────────────────────────────────────────────────────────────
def reciprocal_rank(ranked: Sequence, relevant: Iterable) -> float:
    """Reciprocal rank of the *first* relevant item. 0.0 if none."""
    rel = set(relevant)
    for i, x in enumerate(ranked, start=1):
        if x in rel:
            return 1.0 / i
    return 0.0


def mrr(queries: Iterable[Sequence], relevants: Iterable[Iterable]) -> float:
    """Mean of reciprocal_rank across a batch of queries."""
    rr = (reciprocal_rank(r, rel) for r, rel in zip(queries, relevants))
    vals = list(rr)
    return sum(vals) / len(vals) if vals else 0.0


# ──────────────────────────────────────────────────────────────────────
#  NDCG @ k — graded relevance (binary here: relevant = 1, else 0)
# ──────────────────────────────────────────────────────────────────────
def _dcg(rels: Sequence[int], k: int) -> float:
    s = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        # Standard log2 discount, with the +2 so i=1 has discount 1.
        s += rel / (1.0 if i == 1 else math.log2(i + 1))
    return s


def ndcg_at_k(ranked: Sequence, relevant: Iterable, k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    gains = _hits(ranked, rel)[:k]
    dcg = _dcg(gains, k)
    ideal = sorted(gains, reverse=True)
    idcg = _dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────
#  Aggregator
# ──────────────────────────────────────────────────────────────────────
def summarize(per_query: list[dict]) -> dict:
    """Average the common retrieval metrics across a list of per-query dicts.

    Each dict is expected to carry the values produced by `run_eval.py`:
        {"hit_rate@5": ..., "recall@5": ..., "mrr": ..., "ndcg@5": ...}
    """
    if not per_query:
        return {"queries": 0}
    keys = per_query[0].keys()
    out = {"queries": len(per_query)}
    for k in keys:
        vals = [q[k] for q in per_query if k in q]
        if vals:
            out[k] = round(sum(vals) / len(vals), 4)
    return out

