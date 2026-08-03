# Security Model

This document describes how secrets, database access, uploads, and the
prompt-injection attack surface are handled in the Multi-Model RAG Chatbot.

## 1. Secrets never reach the browser

The app runs on **Streamlit** — all Python executes server-side, and only
rendered HTML is sent to the user's browser. The Supabase `service_role` key,
Gemini key, and Cohere key live in environment variables / Streamlit Cloud
**Secrets** and are read via `os.getenv` in [`clients.py`](clients.py). They
are never embedded in client-side HTML or JavaScript.

- `.env` is in `.gitignore` and is **not** tracked by git.
- No secrets are hardcoded anywhere (all access goes through `os.getenv`).
- A template lives in [`.env.example`](.env.example) — copy it to `.env`.

> **Never** paste the `service_role` key into client-side code, a public repo,
> or a browser. On Streamlit Cloud, put it under **Settings → Secrets**.

## 2. Least-privilege database access (RLS)

Migration [`db/migrations/0005_phase15_rls.sql`](db/migrations/0005_phase15_rls.sql)
enables **Row-Level Security** on `documents`, `chat_sessions`, and
`chat_messages`.

| Role | Capability |
|------|------------|
| `service_role` (the app's writes) | Full access — `service_role` bypasses RLS by design. |
| `anon` / `authenticated` | **Read-only.** No insert/update/delete policy exists, so writes are denied. |

The app uses the `service_role` key for privileged work (indexing, storage,
history). If you also set an **`anon_key`** (see `.env.example`), read queries
(`vector_search`, `keyword_search`, `load_sessions`) automatically use it via
`rag_db._read_client()` — so the read path runs with least privilege. If the
anon key were ever exposed, the blast radius is limited to reading public
chunks; it cannot modify or delete data.

The `owner_id` column on `documents` lets a session scope its own retrieval
(turn the "User ID" input in the sidebar on). Enforcing it at the DB layer
requires swapping to Supabase Auth (login) and activating the owner-scoped
RLS policy commented at the bottom of migration `0005` — that is documented
as a future step.

## 3. Prompt-injection guardrails

[`guardrails.py`](guardrails.py) wraps the RAG pipeline with:

- **Input guard** — length cap, prompt-injection pattern blocklist
  ("ignore previous instructions", role-override phrasing, system/assistant
  tag injection), per-user rate limit.
- **Secret redaction** — `redact_secrets(text)` strips API keys, credit-card
  numbers, emails, and GitHub tokens from any text before it's embedded
  (crawled pages, paste buffers, free-form input). Keeps secrets out of the
  vector store.
- **Optional output faithfulness check** — when `OUTPUT_FAITHFULNESS_CHECK=true`
  in secrets, the generated answer is graded by Gemini against the cited
  sources; low-faithfulness answers surface a warning in the dev panel.
  Off by default (adds ~500-1500ms per response).

## 4. Upload validation

[`pipeline.py`](pipeline.py) `validate_upload()` enforces, before any processing:

- **File type:** only `pdf, docx, pptx, txt, md` and images (`png, jpg, jpeg, webp`).
- **Size:** rejects files over `max_upload_mb` (default 50 MB) with a clear error.

## 5. Web crawl safety

The manual crawl feature ([`crawl.py`](crawl.py)) only fetches a URL if: it is
well-formed http(s), its domain is in the configurable `crawl_allowed_domains`
allowlist (**empty = everything denied**), and `robots.txt` permits it. The
multi-page BFS mode stays on the start URL's host and stops at the page limit.
