"""
One-off dataset expansion script.

Uses Gemini to synthesize plausible user queries for a corpus, one per chunk.
Output: `eval/dataset/generated.jsonl` — designed to be hand-spot-checked
before being promoted into `golden_core.jsonl`.

Run:
    python -m eval.generate_dataset                    # generate from default data/demo.txt
    python -m eval.generate_dataset --input data/      # whole data/ directory
    python -m eval.generate_dataset --per-chunk 3      # 3 queries per chunk
    python -m eval.generate_dataset --out my_set.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "eval"

import chunking  # noqa: E402
from clients import get_gemini_client  # noqa: E402
import config  # noqa: E402


PROMPT = """You are generating evaluation queries for a RAG system.

Given a single document chunk below, produce {n} diverse user queries that:
  1. Are answerable from the chunk (the chunk contains the answer).
  2. Span DIFFERENT phrasings: a literal keyword query, a paraphrased
     semantic query, and a question that requires combining two facts.
  3. Are short (1-2 sentences), natural, and unambiguous.

Chunk:
\"\"\"
{chunk}
\"\"\"

Return ONLY a JSON array (no markdown) of {n} strings. Example:
["query one", "query two", "query three"]"""


def _extract_chunks(input_path: str) -> list[dict]:
    """Walk a file or directory and yield chunk dicts via the real chunker."""
    if os.path.isdir(input_path):
        chunks: list[dict] = []
        for fn in sorted(os.listdir(input_path)):
            full = os.path.join(input_path, fn)
            if not os.path.isfile(full):
                continue
            with open(full, "rb") as f:
                chunks.extend(chunking.chunk_document(fn, f.read()))
        return chunks
    with open(input_path, "rb") as f:
        return chunking.chunk_document(os.path.basename(input_path), f.read())


def _call_llm(prompt: str) -> str:
    client = get_gemini_client()
    if client is None:
        raise RuntimeError("Gemini client unavailable — set the `gemini_api_key` env var.")
    res = client.models.generate_content(model=config.GENERATION_MODEL, contents=[prompt])
    return getattr(res, "text", "") or ""


def _parse_queries(raw: str, n: int) -> list[str]:
    """Lenient parse: strip code fences, find the first JSON array, truncate."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    m = re.search(r"\[.*\]", s, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(arr, list):
        return []
    return [str(q) for q in arr[:n] if str(q).strip()]


def main() -> int:
    p = argparse.ArgumentParser(description="Generate a RAG eval dataset with Gemini.")
    p.add_argument("--input", default="data/demo.txt", help="file or directory to chunk")
    p.add_argument("--out", default="eval/dataset/generated.jsonl")
    p.add_argument("--per-chunk", type=int, default=3)
    p.add_argument("--limit", type=int, default=0, help="cap chunks (smoke test)")
    args = p.parse_args()

    chunks = _extract_chunks(args.input)
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"Synthesizing {args.per_chunk} query(ies) for each of {len(chunks)} chunk(s)...")

    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, ch in enumerate(chunks, 1):
            text = (ch.get("content") or "").strip()
            if not text:
                continue
            print(f"  [{i:>2}/{len(chunks)}] {ch.get('file_name')} chunk {ch.get('chunk_index', 0)}")
            try:
                raw = _call_llm(PROMPT.format(n=args.per_chunk, chunk=text[:1500]))
            except Exception as e:
                print(f"      LLM error: {e}")
                continue
            queries = _parse_queries(raw, args.per_chunk)
            for q in queries:
                row = {
                    "id": f"gen-{i:02d}-{written:03d}",
                    "category": "generated",
                    "query": q,
                    "expected_files": [ch.get("file_name")],
                    "expected_keywords": [],
                    "k": 5,
                    "needs_review": True,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    print(f"\nWrote {written} candidate query(ies) to {args.out}")
    print("Spot-check the file, then promote reviewed rows into golden_core.jsonl.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
