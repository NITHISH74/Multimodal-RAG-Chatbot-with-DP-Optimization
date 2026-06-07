# Security Model (Phase 15)

This document describes how secrets, database access, and uploads are
protected in the Multi-Model RAG Chatbot.

## 1. Secrets never reach the browser

The app runs on **Streamlit** — all Python executes **server-side**, and only
rendered HTML is sent to the user's browser. The Supabase `service_role` key,
Gemini key, and Cohere key live in environment variables / Streamlit Cloud
**Secrets** and are read via `os.getenv` in [`clients.py`](clients.py). They are
never embedded in client-side HTML or JavaScript.

- ✅ `.env` is in `.gitignore` and is **not** tracked by git.
- ✅ No secrets are hardcoded anywhere (all access goes through `os.getenv`).
- ✅ A template lives in [`.env.example`](.env.example) — copy it to `.env`.

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

## 3. Upload validation

[`app.py`](app.py) `validate_upload()` enforces, before any processing:

- **File type:** only `pdf, docx, pptx, txt, md` and images (`png, jpg, jpeg, webp`).
- **Size:** rejects files over `max_upload_mb` (default 20 MB) with a clear error.

## 4. Document isolation (optional, multi-user)

Each chunk row has an `owner_id` column. Setting the sidebar **User ID** tags
your uploads and filters retrieval to your own documents (app-level isolation).

**For true enforcement** (a user *cannot* read another user's docs even with the
anon key), add **Supabase Auth** (login) and switch to the owner-scoped RLS
policy commented at the bottom of migration `0005`:

```sql
create policy documents_owner_read on documents for select
  to authenticated using (owner_id = auth.uid()::text);
```

This requires wiring a login flow (`supabase.auth`) into the UI, which is not
enabled by default.

## 5. Web crawl safety

The manual crawl feature ([`crawl.py`](crawl.py)) only fetches a URL if: it is
well-formed http(s), its domain is in the configurable `crawl_allowed_domains`
allowlist (**empty = everything denied**), and `robots.txt` permits it. Exactly
one page is fetched — no autonomous multi-page crawling.
