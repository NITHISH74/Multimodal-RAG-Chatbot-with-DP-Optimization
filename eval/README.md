# Evaluation Harness

Pure-Python + Gemini-as-judge evaluation for the RAG pipeline. Zero new
production dependencies — uses the same `google-genai` client the app already
depends on.

## Layout

```
eval/
├── README.md            (this file)
├── __init__.py
├── sim.py               shared cosine similarity
├── metrics.py           Recall@k, HitRate@k, MRR, NDCG@k
├── judge.py             Gemini LLM-as-judge (faithfulness, relevance)
├── dataset.py           JSONL loaders for golden_core + negatives
├── dataset/
│   ├── golden_core.jsonl   30 curated queries (deterministic, CI-safe)
│   └── negatives.jsonl     8 should-not-answer queries
├── run_eval.py          end-to-end: runs the pipeline, scores, reports
├── ablation.py          A/B/C/D pipeline variant comparison
└── generate_dataset.py  one-off LLM-based dataset expansion
```

## Quick start

```bash
# Smoke test — 5 queries, no judge (fastest, no API calls except retrieval)
python -m eval.run_eval --limit 5 --no-judge

# Full golden core + LLM-as-judge
python -m eval.run_eval

# Full ablation: 4 pipeline variants × 30 queries
python -m eval.ablation

# Expand the dataset with synthesized queries (human-review before merging)
python -m eval.generate_dataset --per-chunk 3
```

## Metrics reported

| Metric | What it measures | Where |
|--------|------------------|-------|
| `hit_rate@5` | At least one relevant chunk in top-5 | `run_eval`, `ablation` |
| `recall@5` | Fraction of expected files recovered in top-5 | `run_eval`, `ablation` |
| `mrr` | Mean reciprocal rank of the first relevant chunk | `run_eval`, `ablation` |
| `ndcg@5` | Normalized discounted cumulative gain, top-5 | `run_eval`, `ablation` |
| `faithfulness` | Judge: every claim traces to a cited source | `run_eval` |
| `relevance` | Judge: answer addresses the question | `run_eval` |
| `keyword_coverage` | Deterministic: fraction of expected keywords present | `run_eval` |
| `negative_fallback_rate` | Sanity: do negatives trigger the safe refusal? | `run_eval` |

## Determinism

The golden core is hand-curated and is what CI runs. Generated queries
(`eval/dataset/generated.jsonl`) carry a `needs_review: true` flag — review
and promote into `golden_core.jsonl` before relying on the numbers.

## Why file-level ground truth?

Chunk ids are assigned at index time and differ per environment, so the
golden core uses **file_name** as the relevance signal. This is the standard
trade-off for a portfolio-grade eval: it works without a stable DB, and it
is pessimistic in a useful direction (the right *file* but wrong *chunk*
still counts as a hit).
