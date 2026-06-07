# 🧠 Multi-Model Advanced RAG Chatbot — **V3**

> A production-minded, **Retrieval-Augmented Generation** system on a 100% **Supabase** backend (pgvector + Storage), deployable for free on **Streamlit Cloud**. V3 is a ground-up overhaul: hybrid retrieval, a local reranker, token-optimized context, cited answers, anti-hallucination fallback, manual web crawl, query routing, and security hardening.

**🔴 Live Demo:** [multimodal-rag-chatbot-with-dp-optimization-2341qf.streamlit.app](https://multimodal-rag-chatbot-with-dp-optimization-2341qf.streamlit.app/)

**Stack:** Gemini 2.5 Flash (generation) • Gemini Embedding 2 / Cohere v4.0 (dual embeddings) • Supabase pgvector + Storage • Streamlit

---

## 📑 Table of Contents
- [Why V3 (Advantages)](#-why-v3-advantages)
- [Feature Overview](#-feature-overview)
- [Techniques Used](#-techniques-used)
- [The Query Pipeline](#-the-query-pipeline)
- [Architecture & Modules](#-architecture--modules)
- [Installation & Setup](#️-installation--setup)
- [Database Migrations](#-database-migrations)
- [Configuration Reference](#-configuration-reference)
- [Security](#-security)
- [Testing](#-testing)
- [Deployment (Streamlit Cloud)](#-deployment-streamlit-cloud)

---

## 🚀 Why V3 (Advantages)

| Advantage | How V3 delivers it |
|-----------|--------------------|
| **More accurate retrieval** | Hybrid search (semantic **+** keyword) catches both meaning *and* exact terms/IDs/names, then a local reranker reorders by combined signals. |
| **No hallucinated answers** | A real similarity **threshold** filters weak matches; if nothing is relevant the bot returns a configurable "not found" message instead of guessing. |
| **Faster & cheaper** | HNSW ANN indexes, query-level caching, parallel embedding, token-budgeted context, running-summary memory, and TOON-compressed metadata cut both latency and token spend. |
| **Trustworthy** | Every answer ends with a **Sources** section (file · page/slide · similarity score). |
| **Better data quality** | Sentence/slide-aware chunking, content-hash de-duplication, and garbage-text filtering keep the vector DB clean. |
| **Multi-format & multimodal** | PDF, DOCX, PPTX, TXT/MD, and images — images stored in Supabase Storage (not bloating the DB as base64). |
| **Extensible knowledge** | Manual, policy-respecting single-URL web crawl feeds the same pipeline. |
| **Observable** | A dev-only diagnostics panel surfaces latency, token counts, and similarity scores per query. |
| **Hardened** | Secrets stay server-side, Row-Level Security makes the anon key read-only, uploads are type/size-validated, optional per-user isolation. |

---

## ✨ Feature Overview

### Retrieval & Generation
- **Dual embedding models** — switch between **Gemini** and **Cohere** at runtime.
- **Hybrid search** — pgvector cosine similarity **+** Postgres full-text (`tsvector`) keyword search, merged and de-duplicated by chunk id.
- **Rerank-Lite** — a free, local reranker scoring each candidate by *similarity + keyword overlap + recency*; only the top 5–7 chunks reach the LLM.
- **Similarity threshold** — weak matches are excluded in SQL (default `0.70`, configurable).
- **Safe fallback** — no relevant chunks ⇒ a configurable message, never an empty-context guess.
- **Citations** — answers append a clean **Sources** list.
- **Query routing** — intent classifier sends each query to *document / image / web* retrieval, or answers *general* chit-chat directly (skipping RAG).

### Ingestion & Data Quality
- **Smart chunking** — respects page (PDF) / slide (PPTX) boundaries, then splits paragraph- and sentence-aware with overlap.
- **De-duplication** — `sha256(file + content)` hash prevents re-storing identical chunks.
- **Garbage filtering** — drops empty, too-short (< 20 chars), and gibberish chunks.
- **Per-chunk metadata** — `file_name`, `page_number`/`slide_number`, `chunk_index`, `upload_date`, `document_type`, `content_hash`, `source_url`.
- **Image storage** — images go to a Supabase **Storage** bucket; the DB keeps only the URL + metadata.
- **Upload guardrails** — type allowlist + 20 MB limit, staged status (`uploaded → extracting → embedding → completed/failed`), parallel non-blocking embedding.

### Efficiency & Memory
- **HNSW ANN indexes** — Cohere (1536-d) directly; Gemini (3072-d) via a `halfvec` index (pgvector's HNSW caps at 2000 dims).
- **Query cache** — repeated queries reuse the embedding + DB lookup.
- **Token-budgeted context** — token estimation (≈4 chars/token) + a **0/1 Knapsack DP** optimizer pick the most relevant chunks under a token budget; semantic dedup removes near-identical chunks.
- **Running-summary memory** — instead of resending the whole chat, a rolling summary + the last few turns are sent.
- **TOON metadata** — compact Token-Oriented format for chunk metadata only (never for raw document text).

### Ops & Security
- **Dev/Admin panel** — per-query retrieval/generation/total latency, chunk count, input/output tokens, similarity scores (hidden from end users).
- **RLS** — anon key is read-only; service key (server-side only) handles writes.
- **Optional per-user isolation** via `owner_id`.

---

## 🧪 Techniques Used

| Technique | Where | What it does |
|-----------|-------|--------------|
| **HNSW (Hierarchical Navigable Small World)** | `0001_phase1_hnsw.sql` | Approximate-nearest-neighbour index for fast vector search. |
| **`halfvec` half-precision indexing** | `0001` | Lets the 3072-d Gemini vectors be HNSW-indexed despite pgvector's 2000-d limit. |
| **Hybrid (dense + sparse) retrieval** | `retrieval.py`, `keyword_search` RPC | Combines vector similarity with `tsvector`/`ts_rank` keyword matching. |
| **Learning-to-rank style Rerank-Lite** | `retrieval.py` | Weighted linear fusion of similarity, keyword overlap, recency. |
| **Cosine-distance threshold filtering** | `0003_*.sql` | `WHERE 1-(embedding <=> query) >= threshold`. |
| **0/1 Knapsack Dynamic Programming** | `context_builder.py` | Maximizes total relevance under a token budget (the project's signature optimizer). |
| **Token estimation & budgeting** | `context_builder.py` | ~4 chars/token heuristic to bound context size. |
| **Semantic / near-duplicate dedup** | `context_builder.py` | Token-set Jaccard removes redundant same-file chunks. |
| **Sentence/paragraph & page/slide-aware chunking** | `chunking.py` | Structure-preserving splitting with overlap. |
| **Content-hash de-duplication** | `chunking.py`, `rag_db.py` | SHA-256 idempotent inserts. |
| **Conversation summarization** | `conversation.py` | Running summary to cap prompt growth. |
| **TOON (Token-Oriented Object Notation)** | `context_builder.py` | Compact tabular metadata encoding. |
| **Intent routing** | `routing.py` | Lightweight keyword classifier (no agents/loops). |
| **robots.txt + domain-allowlist crawling** | `crawl.py` | Polite, scoped single-page extraction with HTML cleaning. |
| **Row-Level Security (RLS)** | `0005_phase15_rls.sql` | Least-privilege DB access. |
| **Parallel embedding (ThreadPool)** | `app.py` | Non-blocking, concurrent network-bound embeds. |

---

## 🔁 The Query Pipeline

```
User query
   │
   ▼
[Route]  intent = document | image | web | general
   │                                   └── general ─▶ answer directly (skip RAG)
   ▼
[Embed query]  (Gemini or Cohere)  ── cached ──┐
   ▼                                            │
[Hybrid retrieve]  vector (threshold-filtered) + keyword
   ▼
[Merge + dedup]  by chunk id
   ▼
[Rerank-Lite]  similarity + keyword overlap + recency  ─▶ top 5–7
   ▼
[Semantic dedup] ─▶ [Knapsack DP under token budget]
   │
   ├── no chunks ─▶ Safe fallback message
   ▼
[Build context]  citations + TOON metadata  (+ history summary)
   ▼
[Generate]  Gemini 2.5 Flash  (+ images from Storage)
   ▼
Answer  +  📚 Sources (file · page/slide · score)
```

---

## 🧩 Architecture & Modules

| Module | Responsibility |
|--------|----------------|
| [`app.py`](app.py) | Streamlit UI + pipeline orchestration, upload/crawl, dev panel |
| [`config.py`](config.py) | All tunables, fallback message, allowlist, dev flag (env-overridable) |
| [`clients.py`](clients.py) | Cached Gemini / Cohere / Supabase clients (service + anon) |
| [`chunking.py`](chunking.py) | Parsing + structure-aware chunking + cleaning + hashing |
| [`embeddings.py`](embeddings.py) | Text / image / query embedding (both models) |
| [`rag_db.py`](rag_db.py) | Dedup-aware upsert, image→Storage, vector/keyword RPCs, history |
| [`retrieval.py`](retrieval.py) | Hybrid merge + Rerank-Lite |
| [`context_builder.py`](context_builder.py) | Token budget, knapsack, dedup, citations, TOON |
| [`routing.py`](routing.py) | Intent classification |
| [`crawl.py`](crawl.py) | Manual single-URL crawl (robots + allowlist + cleaning) |
| [`conversation.py`](conversation.py) | Running-summary memory |
| [`tests/test_core.py`](tests/test_core.py) | Unit tests for the pure logic (15 passing) |
| [`db/migrations/`](db/migrations/) | Ordered SQL migrations (source of truth) |

> `rag_chatbot.py` (local FAISS CLI) is **legacy** and not used by the web app.

---

## 🛠️ Installation & Setup

### 1. Clone & create a virtual environment
```bash
py -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Create your `.env`
Copy [`.env.example`](.env.example) to `.env` and fill in your keys. **Never commit `.env`** (it is gitignored).
```ini
gemini_api_key  = YOUR_GEMINI_API_KEY
cohere_api_key  = YOUR_COHERE_API_KEY
project_url     = https://YOUR_PROJECT.supabase.co
service_key     = YOUR_SUPABASE_SERVICE_ROLE_KEY   # server-side only
anon_key        = YOUR_SUPABASE_ANON_KEY           # optional, read-only path
```

### 3. Run
```bash
streamlit run app.py
```

---

## 🗄️ Database Setup

**The database schema must be applied once before indexing/search will work.** Choose one:

### Option A — One-click in-app setup (easiest)
1. In Supabase → **Project Settings → Database → Connection string → Session pooler**, copy the URI
   (`postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres`) and put your DB password in it.
   > Use the **Session pooler** URI — Streamlit Cloud is IPv4-only and Supabase's direct connection is IPv6-only.
2. Add it as the **`supabase_db_url`** secret (Streamlit Cloud → Manage app → Secrets, or your `.env`).
3. In the app sidebar → **🔧 Database → Initialize Database**. You should see **"Schema ready ✅"**. Done.

### Option B — Manual SQL (no extra secret)
Run [`db/migrations/RUN_THIS_IN_SUPABASE.sql`](db/migrations/RUN_THIS_IN_SUPABASE.sql) in the Supabase **SQL Editor** (one paste),
or apply the individual migrations in [`db/migrations/`](db/migrations/) **in numeric order** (or `supabase db push`):

| File | What it does |
|------|--------------|
| `0000_baseline.sql` | Base tables + vector search RPCs |
| `0001_phase1_hnsw.sql` | HNSW ANN indexes (Cohere direct, Gemini via `halfvec`) |
| `0002_phase2_chunk_metadata.sql` | Per-chunk metadata columns, content-hash dedup, full-text `tsvector` |
| `0003_phase3_threshold_and_hybrid.sql` | Threshold filtering + `keyword_search` for hybrid retrieval |
| `0004_phase8_storage.sql` | `rag-images` Storage bucket + public-read policy |
| `0005_phase15_rls.sql` | Row-Level Security: anon key is read-only (see [SECURITY.md](SECURITY.md)) |

> Requires **pgvector ≥ 0.7** (for `halfvec`) — Supabase ships 0.8+. Verify:
> `select extversion from pg_extension where extname='vector';`

---

## ⚙️ Configuration Reference

All values live in [`config.py`](config.py) and can be overridden by env vars / Streamlit secrets.

| Variable | Default | Purpose |
|----------|---------|---------|
| `similarity_threshold` | `0.70` | Min cosine similarity for a chunk to count |
| `retrieval_match_count` | `10` | Vector candidates fetched per query |
| `keyword_match_count` | `10` | Keyword candidates fetched per query |
| `rerank_top_k` | `6` | Chunks sent to the LLM (5–7) |
| `max_context_tokens` | `1500` | Token budget for the knapsack |
| `chunk_target_chars` | `1200` | Target chunk size (~300 tokens) |
| `max_upload_mb` | `20` | Per-file upload limit |
| `supabase_image_bucket` | `rag-images` | Storage bucket for images |
| `crawl_allowed_domains` | *(empty)* | Comma-separated allowlist; **empty = crawl disabled** |
| `dev_mode` | `false` | Show the per-query diagnostics panel |
| `fallback_message` | *(built-in)* | Override the "not found" message |

---

## 🔐 Security

See [SECURITY.md](SECURITY.md) for the full model. In short:
- All Python runs **server-side** (Streamlit) — secrets never reach the browser.
- `.env` is gitignored; no secrets are hardcoded.
- **RLS** (migration `0005`) makes the anon key **read-only**; writes use the server-side service key.
- Uploads are **type + size validated**.
- Optional **per-user isolation** via `owner_id` (full enforcement needs Supabase Auth — policy provided).

---

## ✅ Testing

```bash
python tests/test_core.py        # 15 pure-logic tests, no DB/keys needed
```
Covers chunking, cleaning, hashing, routing, merge/rerank, token budget, knapsack, dedup, citations, TOON, history, and crawl safety.

---

## 🌐 Deployment (Streamlit Cloud)

1. Apply all migrations and create the `rag-images` Storage bucket.
2. Push to GitHub, then on [Streamlit Community Cloud](https://share.streamlit.io) → **Create app** → point to `app.py`.
3. Add your secrets under **Advanced settings → Secrets** (same keys as `.env`).
4. Deploy. The architecture is stateless — chat history and documents live in Supabase, so server resets lose nothing.

---

<sub>Built with the **0/1 Knapsack DP context optimizer** at its core — now wrapped in a full hybrid-retrieval, reranking, cited, and hardened RAG pipeline.</sub>
