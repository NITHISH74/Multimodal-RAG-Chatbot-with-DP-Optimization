"""
LLM-as-judge for answer quality.

Uses Gemini (no extra dependencies) to score, on a 0..1 scale, two axes:

    faithfulness      — every claim in the answer traces to a cited source
    answer_relevance  — the answer addresses the user's question

Both calls fail soft: if Gemini is unavailable the judge returns None and the
eval pipeline simply skips generation metrics (retrieval metrics still report).
"""
from __future__ import annotations

import json
import re
from typing import Optional

from clients import get_gemini_client
import config


# ── Prompt templates ──────────────────────────────────────────────────
_FAITHFULNESS_PROMPT = """You are an evaluation judge. Score the answer's
*faithfulness* (every claim is supported by the cited sources) on a 0..1 scale.

If the answer says it cannot find the information in the sources, return
{{"score": 1.0, "reason": "appropriate refusal"}} (a clean refusal is fully
faithful).

Sources:
{sources}

Answer:
{answer}

Respond with ONLY a JSON object on one line, no markdown:
{{"score": <float 0..1>, "reason": "<one short sentence>"}}
"""


_RELEVANCE_PROMPT = """You are an evaluation judge. Score the answer's
*relevance* to the user's question on a 0..1 scale. 0.0 = completely off
topic, 1.0 = directly and fully answers the question.

Question:
{query}

Answer:
{answer}

Respond with ONLY a JSON object on one line, no markdown:
{{"score": <float 0..1>, "reason": "<one short sentence>"}}
"""


def _parse_score(text: str) -> Optional[dict]:
    """Pull the first JSON object from the model response. Lenient about
    accidental markdown fences, since some Gemini runs still wrap output."""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # last-ditch: find the first {...} block
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _call(prompt: str) -> Optional[dict]:
    client = get_gemini_client()
    if client is None:
        return None
    try:
        res = client.models.generate_content(
            model=config.GENERATION_MODEL, contents=[prompt])
        return _parse_score(getattr(res, "text", "") or "")
    except Exception:
        return None


def faithfulness(answer: str, sources: list[str]) -> Optional[dict]:
    """Score the answer against the cited sources. None if judge unavailable."""
    src_block = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(sources)) or "(no sources cited)"
    return _call(_FAITHFULNESS_PROMPT.format(sources=src_block, answer=answer))


def answer_relevance(answer: str, query: str) -> Optional[dict]:
    """Score the answer's relevance to the user query. None if judge unavailable."""
    return _call(_RELEVANCE_PROMPT.format(query=query, answer=answer))
