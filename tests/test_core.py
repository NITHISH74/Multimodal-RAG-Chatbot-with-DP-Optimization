"""
Unit tests for the pure-Python RAG logic (no DB / API keys required).
Run with:  python -m pytest tests/ -q     (or)   python tests/test_core.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chunking
import routing
import retrieval
import context_builder
import conversation
import crawl
import config


# ── chunking / cleaning (Phase 2) ─────────────────────────────────────
def test_normalize_and_meaningful():
    assert chunking.normalize_text("a\r\n\r\n\r\nb   c") == "a\n\nb c"
    assert not chunking.is_meaningful("    ")
    assert not chunking.is_meaningful("!!!???...")           # gibberish/punct
    assert not chunking.is_meaningful("ab")                  # too short
    assert chunking.is_meaningful("This is a real sentence with words.")


def test_split_text_respects_target():
    text = ". ".join(f"Sentence number {i} has some words" for i in range(200))
    chunks = chunking.split_text(text, target_chars=200, overlap_chars=20)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)                # target + overlap slack


def test_content_hash_stable_and_filename_scoped():
    h1 = chunking.content_hash("a.pdf", "hello world")
    h2 = chunking.content_hash("a.pdf", "hello   world")     # whitespace-normalized
    h3 = chunking.content_hash("b.pdf", "hello world")
    assert h1 == h2
    assert h1 != h3


def test_chunk_document_dedups_within_doc():
    txt = ("Para one is unique content here.\n\n"
           "Repeated block of meaningful text.\n\n"
           "Repeated block of meaningful text.\n\n").encode("utf-8")
    chunks = chunking.chunk_document("notes.txt", txt)
    contents = [c["content"] for c in chunks]
    assert len(contents) == len(set(contents))               # no dupes
    assert all(c["document_type"] == "text" for c in chunks)
    assert chunks[0]["chunk_index"] == 0


# ── routing (Phase 11) ────────────────────────────────────────────────
def test_routing_intents():
    assert routing.classify_intent("hello there") == "general"
    assert routing.classify_intent("show me the diagram of the architecture") == "image"
    assert routing.classify_intent("crawl https://example.com") == "web"
    assert routing.classify_intent("what does the contract say about payment") == "document"
    assert routing.intent_to_filter("image") == "image"
    assert routing.intent_to_filter("document") is None


# ── merge + rerank (Phase 4 / 5) ──────────────────────────────────────
def test_merge_dedups_by_id():
    v = [{"id": 1, "content": "alpha beta", "similarity": 0.9}]
    k = [{"id": 1, "content": "alpha beta", "keyword_rank": 0.5},
         {"id": 2, "content": "gamma", "keyword_rank": 0.2}]
    merged = retrieval.merge_results(v, k)
    ids = sorted(r["id"] for r in merged)
    assert ids == [1, 2]
    row1 = next(r for r in merged if r["id"] == 1)
    assert row1["similarity"] == 0.9 and row1["keyword_rank"] == 0.5


def test_rerank_orders_and_limits():
    rows = [
        {"id": 1, "content": "machine learning models", "similarity": 0.95, "keyword_rank": 0.0},
        {"id": 2, "content": "unrelated cooking recipe", "similarity": 0.40, "keyword_rank": 0.0},
        {"id": 3, "content": "learning rate tuning", "similarity": 0.50, "keyword_rank": 0.9},
    ]
    out = retrieval.rerank_lite("machine learning", rows, top_k=2)
    assert len(out) == 2
    assert out[0]["id"] == 1                                  # highest combined signal
    assert all("rerank_score" in r for r in out)


# ── context: tokens, knapsack, dedup, citations, TOON (Phase 6/7/13) ──
def test_token_estimate():
    assert context_builder.estimate_tokens("a" * 40) == 10
    assert context_builder.estimate_tokens("") == 0


def test_semantic_dedup_same_file():
    rows = [
        {"id": 1, "file_name": "a.pdf", "content": "the quick brown fox jumps"},
        {"id": 2, "file_name": "a.pdf", "content": "the quick brown fox jumps over"},  # near-dup
        {"id": 3, "file_name": "b.pdf", "content": "completely different subject matter"},
    ]
    kept = context_builder.semantic_dedup(rows, threshold=0.7)
    ids = sorted(r["id"] for r in kept)
    assert 1 in ids and 3 in ids and 2 not in ids


def test_knapsack_respects_budget():
    rows = [
        {"id": 1, "content": "x" * 400, "rerank_score": 0.9},
        {"id": 2, "content": "x" * 400, "rerank_score": 0.8},
        {"id": 3, "content": "x" * 400, "rerank_score": 0.1},
    ]
    chosen = context_builder.knapsack_select(rows, token_budget=200)  # ~200 tokens
    assert chosen and 1 in [c["id"] for c in chosen]
    total_tokens = sum(context_builder.estimate_tokens(c["content"]) for c in chosen)
    assert total_tokens <= 220


def test_build_context_and_sources():
    rows = [{"id": 1, "file_name": "r.pdf", "document_type": "pdf", "page_number": 3,
             "content": "Revenue grew 20%.", "rerank_score": 0.87}]
    ctx, imgs, used = context_builder.build_context(rows)
    assert "r.pdf" in ctx and "Page 3" in ctx and "0.87" in ctx
    assert imgs == [] and len(used) == 1
    src = context_builder.format_sources(rows)
    assert "r.pdf" in src and "Page 3" in src and "0.87" in src


def test_toon_metadata_format():
    rows = [{"file_name": "a.pdf", "page_number": 2, "rerank_score": 0.81},
            {"file_name": "b.pptx", "page_number": None, "rerank_score": 0.7}]
    toon = context_builder.toon_metadata(rows)
    assert toon.startswith("sources[2]{file,page,score}:")
    assert "a.pdf,2,0.81" in toon and "b.pptx,-,0.70" in toon


# ── conversation history block (Phase 13) ─────────────────────────────
def test_history_block():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    block = conversation.build_history_block(msgs, "prior summary")
    assert "prior summary" in block and "hello" in block
    assert conversation.build_history_block([], "") == ""


# ── crawl safety (Phase 10) ───────────────────────────────────────────
def test_crawl_allowlist(monkeypatch):
    monkeypatch.setattr(config, "CRAWL_ALLOWED_DOMAINS", ("example.com",))
    assert crawl.is_allowed_domain("https://example.com/page")
    assert crawl.is_allowed_domain("https://docs.example.com/x")
    assert not crawl.is_allowed_domain("https://evil.com")
    monkeypatch.setattr(config, "CRAWL_ALLOWED_DOMAINS", ())
    assert not crawl.is_allowed_domain("https://example.com")  # empty list => deny all


def test_tidy_markdown_dedupes_and_collapses():
    md = "# Title\n\nAccept cookies\nAccept cookies\nAccept cookies\n\n\n\nReal paragraph."
    out = crawl.tidy_markdown(md)
    assert out.count("Accept cookies") == 1          # consecutive dupes dropped
    assert "\n\n\n" not in out                        # blank runs collapsed
    assert "# Title" in out and "Real paragraph." in out


def test_normalize_url():
    assert crawl.normalize_url("https://Example.com/docs/#intro") == "https://example.com/docs"
    assert crawl.normalize_url("https://example.com") == "https://example.com/"
    assert (crawl.normalize_url("https://example.com/a?x=1") ==
            crawl.normalize_url("https://example.com/a/?x=1#frag"))


def test_crawl_pages_rejects_bad_and_disallowed(monkeypatch):
    monkeypatch.setattr(config, "CRAWL_ALLOW_ALL", False)
    monkeypatch.setattr(config, "CRAWL_ALLOWED_DOMAINS", ("example.com",))
    try:
        crawl.crawl_pages("not-a-url")
        assert False, "expected CrawlError"
    except crawl.CrawlError as e:
        assert "Invalid URL" in str(e)
    try:
        crawl.crawl_pages("https://evil.com/x")
        assert False, "expected CrawlError"
    except crawl.CrawlError as e:
        assert "not allowed" in str(e)


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    passed = failed = 0

    class _MP:  # minimal monkeypatch shim for direct (non-pytest) runs
        def __init__(self): self._undo = []
        def setattr(self, obj, name, val):
            self._undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, val)
        def undo(self):
            for obj, name, val in reversed(self._undo): setattr(obj, name, val)

    for name, fn in fns:
        mp = _MP()
        try:
            fn(mp) if fn.__code__.co_argcount else fn()
            print(f"PASS {name}"); passed += 1
        except Exception:
            print(f"FAIL {name}"); traceback.print_exc(); failed += 1
        finally:
            mp.undo()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
