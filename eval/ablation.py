"""
Pipeline ablation: run the golden core through 4 variants of the RAG
pipeline and emit a comparison table.

This is what makes the README's "Benchmark Comparison" table real instead of
aspirational — every cell is a measured number, not a claim.

Variants
--------
A) vector-only          pgvector top-k, no keyword, no rerank, no knapsack
B) hybrid               vector + keyword, no rerank, no knapsack
C) hybrid + rerank      adds Rerank-Lite scoring (still no knapsack)
D) hybrid + rerank + dp full pipeline (the production default)

Usage:
    python -m eval.ablation
    python -m eval.ablation --model cohere
    python -m eval.ablation --limit 10 --json ab.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "eval"

from eval import dataset, metrics  # noqa: E402
import config  # noqa: E402
import context_builder  # noqa: E402
import rag_db  # noqa: E402
import retrieval  # noqa: E402
from embeddings import embed_query  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
#  Variant implementations — each returns a *selected* list of chunk rows
# ──────────────────────────────────────────────────────────────────────
def variant_a_vector(query: str, model: str, k: int, **_) -> list[dict]:
    """A) vector-only. Threshold-filtered pgvector top-k. No keyword, no rerank,
    no knapsack — just take the first k."""
    vec = embed_query(query, model)
    rows = rag_db.vector_search(
        vec, model, config.SIMILARITY_THRESHOLD, k, filter_type=_.get("filter_type"))
    return rows[:k]


def variant_b_hybrid(query: str, model: str, k: int, **_) -> list[dict]:
    """B) hybrid, no rerank. Merge vector + keyword hits, dedup, take first k."""
    vec = embed_query(query, model)
    v = rag_db.vector_search(vec, model, config.SIMILARITY_THRESHOLD,
                             config.RETRIEVAL_MATCH_COUNT, filter_type=_.get("filter_type"))
    kw = rag_db.keyword_search(query, config.KEYWORD_MATCH_COUNT,
                               filter_type=_.get("filter_type"))
    merged = retrieval.merge_results(v, kw)
    return merged[:k]


def variant_c_hybrid_rerank(query: str, model: str, k: int, **_) -> list[dict]:
    """C) hybrid + Rerank-Lite. Take the top k after rerank scoring."""
    rows = retrieval.hybrid_retrieve(query, model, filter_type=_.get("filter_type"))
    return rows[:k]


def variant_d_full(query: str, model: str, k: int, **_) -> list[dict]:
    """D) the production pipeline: hybrid + rerank + dedup + knapsack."""
    rows = retrieval.hybrid_retrieve(query, model, filter_type=_.get("filter_type"))
    rows = rows[:max(k, config.RERANK_TOP_K)]
    rows = context_builder.semantic_dedup(rows)
    chosen = context_builder.knapsack_select(rows)
    return chosen


VARIANTS = {
    "A · vector-only":         variant_a_vector,
    "B · hybrid":              variant_b_hybrid,
    "C · hybrid + rerank":     variant_c_hybrid_rerank,
    "D · hybrid + rerank + DP": variant_d_full,
}


# ──────────────────────────────────────────────────────────────────────
#  Per-query evaluation across all variants
# ──────────────────────────────────────────────────────────────────────
def _file_hits(chosen: list[dict], expected_files: list[str]) -> list[str]:
    if not expected_files:
        return []
    expected = {f.lower() for f in expected_files}
    return [r.get("id") for r in chosen if (r.get("file_name") or "").lower() in expected]


def run_variant(variant_fn, items: list[dict], model: str, k: int) -> list[dict]:
    out = []
    for item in items:
        q = item["query"]
        expected_files = item.get("expected_files", [])
        try:
            t0 = time.time()
            chosen = variant_fn(q, model, k=k)
            ms = (time.time() - t0) * 1000
        except Exception as e:
            out.append({"id": item.get("id"), "error": str(e)})
            continue
        ids = [r.get("id") for r in chosen]
        rel = _file_hits(chosen, expected_files)
        out.append({
            "id": item.get("id"),
            "category": item.get("category"),
            f"hit_rate@{k}": metrics.hit_rate_at_k(ids, rel, k),
            f"recall@{k}": metrics.recall_at_k(ids, rel, k),
            "mrr": metrics.reciprocal_rank(ids, rel),
            f"ndcg@{k}": metrics.ndcg_at_k(ids, rel, k),
            "retrieval_ms": round(ms, 1),
            "chunks": len(chosen),
        })
    return out


def _avg(vals: list) -> float:
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _print_table(rows: list[list[Any]]) -> None:
    widths = [max(len(str(c)) for c in col) for col in zip(*rows)]
    for i, r in enumerate(rows):
        line = " │ ".join(str(c).ljust(w) for c, w in zip(r, widths))
        print(f" {line}")
        if i == 0:
            print(" " + "─┼─".join("─" * w for w in widths))


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description="Pipeline ablation: 4 variants on the golden core")
    p.add_argument("--model", default="Gemini", choices=["Gemini", "Cohere"])
    p.add_argument("--limit", type=int, default=0, help="cap queries (smoke test)")
    p.add_argument("--json", default="", help="path to dump the raw ablation report")
    args = p.parse_args()

    items = dataset.load_golden()
    if args.limit:
        items = items[: args.limit]
    print(f"Ablating {len(items)} golden query(ies) × {len(VARIANTS)} variants (model={args.model})\n")

    all_results: dict[str, list[dict]] = {}
    summary_rows: list[list[Any]] = [["variant", "hit@5", "recall@5", "mrr", "ndcg@5", "ms", "chunks"]]
    for name, fn in VARIANTS.items():
        print(f"  • {name} ...")
        per_q = run_variant(fn, items, args.model, k=5)
        all_results[name] = per_q
        summary_rows.append([
            name,
            f"{_avg([r.get('hit_rate@5', 0) for r in per_q]):.3f}",
            f"{_avg([r.get('recall@5', 0) for r in per_q]):.3f}",
            f"{_avg([r.get('mrr', 0) for r in per_q]):.3f}",
            f"{_avg([r.get('ndcg@5', 0) for r in per_q]):.3f}",
            f"{_avg([r.get('retrieval_ms', 0) for r in per_q]):.1f}",
            f"{_avg([r.get('chunks', 0) for r in per_q]):.1f}",
        ])

    print("\nAblation summary:\n")
    _print_table(summary_rows)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"model": args.model, "summary": summary_rows,
                       "per_query": all_results}, f, indent=2, default=str)
        print(f"\nWrote ablation report to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
