"""
JSONL dataset loader for the eval harness.

Each row is a JSON object with at least:
    id, query, expected_files, expected_keywords, k, category, ...
A row may additionally carry `should_fallback: true` (negatives).
"""
from __future__ import annotations

import json
import os
from typing import Iterator

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


def _load(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_golden() -> list[dict]:
    """Curated ~30-query deterministic core (runs in CI)."""
    return _load(os.path.join(DATASET_DIR, "golden_core.jsonl"))


def load_negatives() -> list[dict]:
    """~10 'should-not-answer' queries used to verify the strengthened fallback."""
    return _load(os.path.join(DATASET_DIR, "negatives.jsonl"))


def load_all() -> list[dict]:
    """Convenience: golden + negatives (the two shipped-by-default sets)."""
    return load_golden() + load_negatives()


def iter_jsonl(path: str) -> Iterator[dict]:
    """Stream rows from any jsonl file (used by `generate_dataset.py` and CI)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
