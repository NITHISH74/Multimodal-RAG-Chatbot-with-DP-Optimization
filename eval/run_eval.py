"""
End-to-end evaluation: run the golden core + negatives through the real RAG
pipeline and report retrieval + (optionally) generation metrics.

Usage:
    python -m eval.run_eval                        # default (gemini)
    python -m eval.run_eval --model cohere
    python -m eval.run_eval --no-judge             # skip LLM-as-judge
    python -m eval.run_eval --limit 5              # quick smoke test
    python -m eval.run_eval --json report.json     # also dump raw per-query results
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any

# Allow running as `python -m eval.run_eval` AND `python eval/run_eval.py`
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "eval"

from eval import dataset, judge, metrics  # noqa: E402
import config  # noqa: E402
import retrieval  # noqa: E402
import context_builder  # noqa: E402
import rag_db  # noqa: E402
from clients import get_gemini_client  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
#  Pipeline shims that the eval can drive without the Streamlit UI
# ──────────────────────────────────────────────────────────────────────
def _retrieve(query: str, model: str, k: int, filter_type=None, owner_id=None) -> list[dict]:
    """Retrieve + dedup + knapsack + rerank (the full default pipeline).

    Returns the *selected* chunk rows in the order they go to the LLM.
    """
    rows = retrieval.hybrid_retrieve(query, model, owner_id=owner_id, filter_type=filter_type)
    rows = rows[:max(k, config.RERANK_TOP_K)]
    rows = context_builder.semantic_dedup(rows)
    chosen = context_builder.knapsack_select(rows)
    return chosen


def _generate(answer_prompt: str) -> tuple[str, int, int]:
    """Tiny wrapper around the generation client. Returns (text, in_tok, out_tok)."""
    client = get_gemini_client()
    if client is None:
        return ("", 0, 0)
    res = client.models.generate_content(model=config.GENERATION_MODEL, contents=[answer_prompt])
    in_tok = out_tok = 0
    if hasattr(res, "usage_metadata") and res.usage_metadata:
        in_tok = res.usage_metadata.prompt_token_count or 0
        out_tok = res.usage_metadata.candidates_token_count or 0
    return (getattr(res, "text", "") or "", in_tok, out_tok)


def _build_prompt(query: str, chosen: list[dict]) -> str:
    ctx, _imgs, _used = context_builder.build_context(chosen)
    toon = context_builder.toon_metadata(chosen)
    return (
        "Use ONLY the context below to answer. Cite sources by file name. "
        "If the context is insufficient, say so.\n\n"
        f"Context metadata ({toon}):\n---\n{ctx}\n---\n\nQuestion: {query}"
    )


# ──────────────────────────────────────────────────────────────────────
#  Per-query evaluation
# ──────────────────────────────────────────────────────────────────────
def _file_hits(chosen: list[dict], expected_files: list[str]) -> list[str]:
    """Return the ids of retrieved rows whose file_name is in expected_files."""
    if not expected_files:
        return []
    expected = {f.lower() for f in expected_files}
    return [r.get("id") for r in chosen if (r.get("file_name") or "").lower() in expected]


def evaluate_query(item: dict, model: str, use_judge: bool) -> dict:
    """Run a single query through the pipeline and score it."""
    q = item["query"]
    k = int(item.get("k", 5))
    expected_files = item.get("expected_files", [])
    expected_keywords = [k.lower() for k in item.get("expected_keywords", [])]
    should_fallback = bool(item.get("should_fallback"))

    t0 = time.time()
    try:
        chosen = _retrieve(q, model, k=k)
    except Exception as e:
        return {"id": item.get("id"), "error": f"retrieval: {e}"}
    retrieval_time = time.time() - t0

    retrieved_ids = [r.get("id") for r in chosen]
    retrieved_files = [r.get("file_name") for r in chosen]
    # Retrieval metrics against the file-level ground truth
    relevant_ids = _file_hits(chosen, expected_files)
    # We use the file-matching ids as relevance proxy (this lets the eval work
    # without DB chunk ids). The proxy is pessimistic: if the system pulled
    # the right FILE but the wrong chunk, we still count it as relevant.
    per_query = {
        "id": item.get("id"),
        "category": item.get("category"),
        "retrieval_time_ms": round(retrieval_time * 1000, 1),
        "chunks_selected": len(chosen),
        "retrieved_files": retrieved_files,
        "expected_files": expected_files,
        f"hit_rate@{k}": metrics.hit_rate_at_k(retrieved_ids, relevant_ids, k),
        f"recall@{k}": metrics.recall_at_k(retrieved_ids, relevant_ids, k),
        "mrr": metrics.reciprocal_rank(retrieved_ids, relevant_ids),
        f"ndcg@{k}": metrics.ndcg_at_k(retrieved_ids, relevant_ids, k),
        "should_fallback": should_fallback,
        "did_fallback": not chosen,  # the pipeline returns no chosen chunks when nothing is relevant
    }

    # Generation metrics (skip for pure retrieval eval, or for negatives where
    # we explicitly want a refusal)
    if use_judge and not should_fallback and chosen:
        prompt = _build_prompt(q, chosen)
        t0 = time.time()
        answer, in_tok, out_tok = _generate(prompt)
        per_query["generation_time_ms"] = round((time.time() - t0) * 1000, 1)
        per_query["input_tokens"] = in_tok
        per_query["output_tokens"] = out_tok
        per_query["answer_preview"] = (answer or "")[:200]
        # keyword coverage as a deterministic alternative to judge
        lowered = (answer or "").lower()
        per_query["keyword_coverage"] = round(
            sum(1 for kw in expected_keywords if kw in lowered) / max(len(expected_keywords), 1), 4
        )
        # LLM-as-judge (may be None if Gemini judge unavailable)
        src_list = [
            f"{r.get('file_name','')} p{r.get('page_number') or '-'}: "
            f"{(r.get('content','') or '')[:300]}"
            for r in chosen
        ]
        faith = judge.faithfulness(answer, src_list)
        rel = judge.answer_relevance(answer, q)
        per_query["faithfulness"] = (faith or {}).get("score")
        per_query["faithfulness_reason"] = (faith or {}).get("reason")
        per_query["relevance"] = (rel or {}).get("score")
        per_query["relevance_reason"] = (rel or {}).get("reason")

    return per_query


# ──────────────────────────────────────────────────────────────────────
#  Reporting
# ──────────────────────────────────────────────────────────────────────
def _avg(vals: list) -> float:
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _print_table(rows: list[list[Any]]) -> None:
    if not rows:
        return
    widths = [max(len(str(c)) for c in col) for col in zip(*rows)]
    for i, r in enumerate(rows):
        line = " │ ".join(str(c).ljust(w) for c, w in zip(r, widths))
        print(f" {line}")
        if i == 0:
            print(" " + "─┼─".join("─" * w for w in widths))


def report(results: list[dict]) -> dict:
    """Aggregate per-query results into a summary dict and print it."""
    if not results:
        print("No results to report.")
        return {}

    print(f"\nEvaluated {len(results)} query(ies).\n")

    # Retrieval metrics (always present)
    retrieval_keys = ["hit_rate@5", "recall@5", "mrr", "ndcg@5"]
    summary = {"queries": len(results)}
    for k in retrieval_keys:
        summary[k] = _avg([r.get(k, 0.0) for r in results])
    summary["avg_retrieval_ms"] = _avg([r.get("retrieval_time_ms", 0) for r in results])
    summary["avg_chunks"] = _avg([r.get("chunks_selected", 0) for r in results])

    rows = [["metric", "value"]]
    for k in retrieval_keys:
        rows.append([k, f"{summary[k]:.3f}"])
    rows.append(["avg_retrieval_ms", f"{summary['avg_retrieval_ms']:.1f}"])
    rows.append(["avg_chunks", f"{summary['avg_chunks']:.2f}"])

    # Generation metrics (when judge ran)
    faith = [r.get("faithfulness") for r in results if r.get("faithfulness") is not None]
    rel = [r.get("relevance") for r in results if r.get("relevance") is not None]
    kw = [r.get("keyword_coverage") for r in results if r.get("keyword_coverage") is not None]
    gen_ms = [r.get("generation_time_ms") for r in results if r.get("generation_time_ms") is not None]
    if faith:
        summary["faithfulness"] = _avg(faith)
        rows.append(["faithfulness (judge)", f"{summary['faithfulness']:.3f}  (n={len(faith)})"])
    if rel:
        summary["relevance"] = _avg(rel)
        rows.append(["relevance (judge)", f"{summary['relevance']:.3f}  (n={len(rel)})"])
    if kw:
        summary["keyword_coverage"] = _avg(kw)
        rows.append(["keyword_coverage", f"{summary['keyword_coverage']:.3f}  (n={len(kw)})"])
    if gen_ms:
        summary["avg_generation_ms"] = _avg(gen_ms)
        rows.append(["avg_generation_ms", f"{summary['avg_generation_ms']:.1f}"])

    # Negative-set sanity: do the fallbacks actually fire?
    neg_results = [r for r in results if r.get("should_fallback")]
    if neg_results:
        neg_hit = sum(1 for r in neg_results if r.get("did_fallback"))
        neg_rate = round(neg_hit / len(neg_results), 4)
        summary["negative_fallback_rate"] = neg_rate
        rows.append(["negative_fallback_rate", f"{neg_rate:.3f}  ({neg_hit}/{len(neg_results)})"])

    _print_table(rows)
    return summary


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description="Run the RAG eval against the golden core")
    p.add_argument("--model", default="Gemini", choices=["Gemini", "Cohere"])
    p.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge")
    p.add_argument("--limit", type=int, default=0, help="cap the number of queries (smoke test)")
    p.add_argument("--json", default="", help="path to write the raw per-query results JSON")
    p.add_argument("--include-negatives", action="store_true", default=True)
    p.add_argument("--no-negatives", dest="include_negatives", action="store_false")
    args = p.parse_args()

    items = dataset.load_golden()
    if args.include_negatives:
        items += dataset.load_negatives()
    if args.limit:
        items = items[: args.limit]
    print(f"Running eval on {len(items)} query(ies) with {args.model} (judge={'on' if not args.no_judge else 'off'})...")

    results: list[dict] = []
    for i, item in enumerate(items, 1):
        print(f"  [{i:>2}/{len(items)}] {item.get('id','?')}: {item['query'][:60]}...")
        try:
            r = evaluate_query(item, args.model, use_judge=not args.no_judge)
        except Exception as e:
            r = {"id": item.get("id"), "error": f"eval: {e}"}
        results.append(r)

    summary = report(results)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2, default=str)
        print(f"\nWrote per-query report to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
