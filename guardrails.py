"""
Input + output guardrails for the RAG pipeline.

A small, deterministic guard layer that fails soft — every check returns
either an "ok" verdict or a refusal reason, never raises. The pipeline uses
it at the very start (on the user's query) and optionally on the generated
answer (LLM-as-judge faithfulness check).

Components
----------
input_guard(query)         length cap + obvious prompt-injection heuristics.
redact_secrets(text)       strip API keys / credit cards / emails from text
                           that will be embedded (crawled pages, pasted text).
output_faithfulness_check  thin wrapper around eval.judge for live answers.

All knobs live in `config.GUARDRAIL_*` so they can be tuned without code
changes.
"""
from __future__ import annotations

import re
import time
from collections import deque
from typing import Optional

import config


# ──────────────────────────────────────────────────────────────────────
#  Input guard
# ──────────────────────────────────────────────────────────────────────
# Common prompt-injection patterns. Deliberately conservative — we flag
# *obvious* attempts only, not every mention of "ignore".
_INJECTION_PATTERNS = [
    re.compile(r"\bignore (all|the) (previous|prior|above) (instructions|prompts?)\b", re.I),
    re.compile(r"\bdisregard (everything|all) (above|before)\b", re.I),
    re.compile(r"\byou are now\b.{0,40}\b(developer mode|dan|jailbroken|unfiltered)\b", re.I),
    re.compile(r"\bsystem\s*:\s*you are\b", re.I),
    re.compile(r"<\s*\|?\s*(system|assistant)\s*\|?\s*>", re.I),
    re.compile(r"\bforget (your|all) (rules|guidelines|restrictions)\b", re.I),
]

# Rate limiter (sliding window per-process, per-user). Best-effort — Streamlit
# spawns a worker per session, so this is per-session, not global.
_rate_window: dict[str, deque[float]] = {}


def _rate_limit_ok(key: str) -> bool:
    """True if the given key has not exceeded the per-minute limit."""
    now = time.monotonic()
    limit = config.GUARDRAIL_RATE_LIMIT_PER_MIN
    if limit <= 0:
        return True
    bucket = _rate_window.setdefault(key, deque())
    # drop entries older than 60 s
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= limit:
        return False
    bucket.append(now)
    return True


def input_guard(query: str, *, user_key: str = "default") -> tuple[bool, str]:
    """Return (ok, reason).  reason is empty when ok=True.

    Refuses the query if:
      * it's empty / whitespace-only
      * it exceeds the configured max length
      * it matches a known prompt-injection pattern
      * the per-user rate limit is exceeded
    """
    if query is None or not str(query).strip():
        return False, "empty_query"
    if len(query) > config.GUARDRAIL_MAX_QUERY_CHARS:
        return False, "query_too_long"
    for pat in _INJECTION_PATTERNS:
        if pat.search(query):
            return False, "injection_suspected"
    if not _rate_limit_ok(user_key):
        return False, "rate_limited"
    return True, ""


# ──────────────────────────────────────────────────────────────────────
#  Secret redaction (for ingested text)
# ──────────────────────────────────────────────────────────────────────
_REDACT_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{32,}\b"), "[REDACTED_API_KEY]"),
    # Credit-card: 13-19 digits not surrounded by other digits (avoids eating
    # 20-char run-on digit sequences from IDs / timestamps).
    (re.compile(r"(?<!\d)\d{13,19}(?!\d)"), "[REDACTED_CARD]"),
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"), "[REDACTED_TOKEN]"),
]


def redact_secrets(text: str) -> str:
    """Replace API keys / card numbers / emails with [REDACTED_*] tokens.

    Apply to crawled pages, paste buffers, and any user-supplied free text
    BEFORE embedding — keeps secrets out of the vector store and out of
    prompt contexts.
    """
    if not text:
        return text
    out = text
    for pat, repl in _REDACT_RULES:
        out = pat.sub(repl, out)
    return out


# ──────────────────────────────────────────────────────────────────────
#  Output faithfulness check (optional, opt-in via config)
# ──────────────────────────────────────────────────────────────────────
def output_faithfulness_check(answer: str, sources: list[str]) -> Optional[dict]:
    """LLM-as-judge for answer grounding. None if the judge is unavailable.

    Disabled by default; enable via `OUTPUT_FAITHFULNESS_CHECK=true` in
    secrets. The judge adds ~500-1500ms to the response, so we leave it
    opt-in for latency-sensitive deployments.
    """
    if not config.OUTPUT_FAITHFULNESS_CHECK:
        return None
    try:
        from eval.judge import faithfulness as _faith
    except Exception:
        return None
    return _faith(answer, sources or [])
