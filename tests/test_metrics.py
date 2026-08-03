"""
Unit tests for the pure-Python eval metrics.

Runs without any DB / API keys.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import metrics, sim


# ── cosine ────────────────────────────────────────────────────────────
def test_cosine_identical_is_one():
    assert sim.cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0


def test_cosine_orthogonal_is_zero():
    assert sim.cosine_similarity([1, 0], [0, 1]) == 0.0


def test_cosine_opposite_is_negative_one():
    assert sim.cosine_similarity([1, 0], [-1, 0]) == -1.0


def test_cosine_mismatched_dims_safe():
    # Stops at the shorter vector; should still return a finite number.
    v = sim.cosine_similarity([1, 0, 0], [1])
    assert -1.0 <= v <= 1.0


def test_cosine_zero_vector_is_zero():
    assert sim.cosine_similarity([0, 0], [1, 2]) == 0.0


# ── hit rate @ k ─────────────────────────────────────────────────────
def test_hit_rate_at_k_finds_match():
    assert metrics.hit_rate_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0
    assert metrics.hit_rate_at_k(["a", "b", "c"], {"z"}, k=3) == 0.0
    assert metrics.hit_rate_at_k(["a", "b"], {"a"}, k=5) == 1.0   # k > list size


# ── recall @ k ───────────────────────────────────────────────────────
def test_recall_at_k_fraction():
    # 2 of 3 relevant in top 2 -> 2/3
    assert metrics.recall_at_k(["r1", "r2", "x"], {"r1", "r2", "r3"}, k=2) == 2 / 3


def test_recall_at_k_perfect():
    assert metrics.recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_recall_at_k_empty_relevant():
    assert metrics.recall_at_k(["a"], set(), k=5) == 0.0


# ── reciprocal rank / MRR ───────────────────────────────────────────
def test_rr_first_position():
    assert metrics.reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_rr_third_position():
    assert metrics.reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3


def test_rr_no_match_is_zero():
    assert metrics.reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_mrr_batch():
    queries = [["a", "b"], ["x", "y", "a"], ["a"]]
    relevants = [{"a"}, {"a"}, {"a"}]
    # Reciprocal ranks: 1/1, 1/3, 1/1 -> mean
    expected = (1.0 + 1 / 3 + 1.0) / 3
    assert abs(metrics.mrr(queries, relevants) - expected) < 1e-9


# ── NDCG @ k ────────────────────────────────────────────────────────
def test_ndcg_perfect_ranking_is_one():
    assert metrics.ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_ndcg_reversed_lower():
    # With only 1 of 2 items relevant, the optimal vs. reversed ranking give
    # measurably different NDCG (1.0 vs. 1/log2(3) ≈ 0.631).
    perfect = metrics.ndcg_at_k(["a", "b"], {"a"}, k=2)
    reversed_ = metrics.ndcg_at_k(["b", "a"], {"a"}, k=2)
    assert perfect == 1.0
    assert reversed_ < perfect
    assert reversed_ < 0.7   # loose sanity bound


def test_ndcg_empty_relevant():
    assert metrics.ndcg_at_k(["a", "b"], set(), k=2) == 0.0


# ── summarizer ──────────────────────────────────────────────────────
def test_summarize_averages():
    per_q = [
        {"hit_rate@5": 1.0, "recall@5": 0.5, "mrr": 1.0, "ndcg@5": 0.8},
        {"hit_rate@5": 0.0, "recall@5": 0.0, "mrr": 0.0, "ndcg@5": 0.0},
    ]
    out = metrics.summarize(per_q)
    assert out["queries"] == 2
    assert out["hit_rate@5"] == 0.5
    assert out["recall@5"] == 0.25
    assert out["ndcg@5"] == 0.4
