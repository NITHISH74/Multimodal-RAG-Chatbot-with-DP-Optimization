"""
Shared similarity utilities (cosine / dot / L2).

Folded from the legacy `embedding_comparison.py` (now removed). One canonical
implementation so the eval harness and any future cross-modal comparison share
the same math.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

Number = float
Vector = Sequence[Number]


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 if either vector is zero-norm.

    Works on plain Python sequences — no NumPy required, so the metrics
    module stays usable in any environment that can import stdlib.
    """
    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    # zip stops at the shorter vector on purpose — embeddings from different
    # models can have different dimensionalities; mismatched pairs should
    # score ~0 rather than crash.
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def l2(a: Vector, b: Vector) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def pairwise_cosine(query: Vector, matrix: Iterable[Vector]) -> list[float]:
    """Convenience: cosine of a single query against every row of a matrix."""
    return [cosine_similarity(query, row) for row in matrix]
