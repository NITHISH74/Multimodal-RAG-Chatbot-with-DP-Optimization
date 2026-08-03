"""
Context assembly (Phase 6), citations (Phase 7) and TOON metadata (Phase 13).

- Token estimation (~4 chars/token) drives the budget instead of raw chars.
- Near-identical chunks from the same file are de-duplicated before inclusion.
- The 0/1 Knapsack DP (the project's signature optimiser) now maximises
  relevance under a TOKEN budget.
- Each context block carries citation metadata so the model can attribute
  sources, and a clean "Sources" section is appended to the answer.
"""
import re

import config

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def estimate_tokens(text):
    """Rough token estimate: ~4 characters per token (Phase 6.1)."""
    if not text:
        return 0
    return max(1, len(text) // config.CHARS_PER_TOKEN)


def _tokens(text):
    return set(t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1)


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def semantic_dedup(rows, threshold=None):
    """Drop near-identical chunks from the SAME file (Phase 6.2).

    Uses token-set Jaccard overlap as a lightweight proxy for semantic
    similarity (avoids extra embedding calls). Keeps the first occurrence,
    which — because input is rerank-ordered — is the higher-scoring chunk.
    """
    threshold = config.SEMANTIC_DEDUP_THRESHOLD if threshold is None else threshold
    kept = []
    kept_tokens = []
    for r in rows:
        toks = _tokens(r.get("content", ""))
        dup = False
        for prev, prev_toks in kept_tokens:
            if prev.get("file_name") == r.get("file_name") and _jaccard(toks, prev_toks) >= threshold:
                dup = True
                break
        if not dup:
            kept.append(r)
            kept_tokens.append((r, toks))
    return kept


def knapsack_select(candidates, token_budget=None):
    """0/1 Knapsack DP: pick the chunk subset maximising total relevance
    (rerank_score / similarity) without exceeding the TOKEN budget.

    Fast-path: when every candidate fits inside the budget, the optimal
    solution is "take everything" — we skip the O(n*W) table entirely.
    """
    token_budget = token_budget or config.MAX_CONTEXT_TOKENS
    n = len(candidates)
    if n == 0:
        return []

    weights = [max(1, estimate_tokens(c.get("content", ""))) for c in candidates]
    if sum(weights) <= token_budget:
        # Fast path: greedy-is-optimal when everything fits. Preserves the
        # rerank order so the LLM sees the strongest chunks first.
        return list(candidates)

    # Scale tokens down to keep the DP table small.
    SCALE = 5
    W = max(1, token_budget // SCALE)
    scaled_weights = [max(1, w // SCALE) for w in weights]
    values = [c.get("rerank_score", c.get("similarity", 0.0)) for c in candidates]

    dp = [[0.0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        wi, vi = scaled_weights[i - 1], values[i - 1]
        for w in range(1, W + 1):
            dp[i][w] = dp[i - 1][w]
            if wi <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - wi] + vi)

    chosen, w = [], W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen.append(candidates[i - 1])
            w -= scaled_weights[i - 1]
    chosen.reverse()
    return chosen


def _cite_label(row):
    loc = ""
    if row.get("page_number") is not None:
        kind = "Slide" if row.get("document_type") == "pptx" else "Page"
        loc = f", {kind} {row['page_number']}"
    return f"{row.get('file_name', 'unknown')}{loc}"


def build_context(rows):
    """Build the LLM context block + image list from selected chunks.

    Returns (context_str, image_rows, used_rows). Each text block embeds
    citation metadata (file, page/slide, score) so the model can reference
    sources (Phase 6.3)."""
    contexts, image_rows, used = [], [], []
    for r in rows:
        used.append(r)
        if r.get("document_type") == "image" or r.get("type") == "image":
            image_rows.append(r)
            contexts.append(f"[Source: {_cite_label(r)} (image attached)]")
            continue
        score = r.get("rerank_score", r.get("similarity", 0.0))
        contexts.append(
            f"[Source: {_cite_label(r)} | relevance {score:.2f}]\n{r.get('content', '')}"
        )
    context_str = "\n---\n".join(contexts)
    return context_str, image_rows, used


def format_sources(rows):
    """Markdown 'Sources' section appended to answers (Phase 7)."""
    if not rows:
        return ""
    lines = ["", "---", "**Sources**"]
    seen = set()
    for r in rows:
        key = (r.get("file_name"), r.get("page_number"))
        if key in seen:
            continue
        seen.add(key)
        score = r.get("rerank_score", r.get("similarity", 0.0))
        loc = ""
        if r.get("page_number") is not None:
            kind = "Slide" if r.get("document_type") == "pptx" else "Page"
            loc = f" — {kind} {r['page_number']}"
        src = f" ({r['source_url']})" if r.get("source_url") else ""
        lines.append(f"- `{r.get('file_name', 'unknown')}`{loc}{src} · similarity {score:.2f}")
    return "\n".join(lines)


def toon_metadata(rows):
    """Compact TOON-style table of chunk metadata (Phase 13.3).

    Applied ONLY to metadata / tool output — never to raw document text
    (Phase 13.4). Far cheaper in tokens than repeating JSON keys per row::

        sources[2]{file,page,score}:
        report.pdf,3,0.87
        notes.docx,-,0.81
    """
    if not rows:
        return "sources[0]{file,page,score}:"
    header = f"sources[{len(rows)}]{{file,page,score}}:"
    out = [header]
    for r in rows:
        page = r.get("page_number")
        page = "-" if page is None else page
        score = r.get("rerank_score", r.get("similarity", 0.0))
        out.append(f"{r.get('file_name', 'unknown')},{page},{score:.2f}")
    return "\n".join(out)
