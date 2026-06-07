-- ════════════════════════════════════════════════════════════════════
--  0002_phase2_chunk_metadata.sql  —  Phase 2: per-chunk schema
--
--  Moves from "one row per document" to "one row per chunk" by adding
--  first-class metadata columns, a content-hash dedup key, and a
--  full-text-search column (used by Phase 4 hybrid search).
--
--  Existing rows keep working: the new columns are nullable and are
--  backfilled from the old metadata jsonb at the bottom of this file.
-- ════════════════════════════════════════════════════════════════════

alter table documents add column if not exists file_name     text;
alter table documents add column if not exists document_type text;       -- pdf|docx|pptx|text|image|web
alter table documents add column if not exists page_number   int;        -- page (pdf) / slide (pptx)
alter table documents add column if not exists chunk_index    int;
alter table documents add column if not exists content_hash  text;
alter table documents add column if not exists source_url     text;       -- Phase 10 (web)
alter table documents add column if not exists image_path     text;       -- Phase 8 (Storage path/URL)
alter table documents add column if not exists owner_id       text;       -- Phase 15 (optional isolation)
alter table documents add column if not exists upload_date   timestamptz default timezone('utc'::text, now());

-- ── Dedup key (Phase 2.3): same file + same content => one row ───────
create unique index if not exists documents_content_hash_uidx
  on documents (content_hash) where content_hash is not null;

-- ── Full-text search column + GIN index (Phase 4) ───────────────────
alter table documents add column if not exists fts tsvector
  generated always as (to_tsvector('english', coalesce(content, ''))) stored;
create index if not exists documents_fts_gin on documents using gin (fts);

-- ── Helpful filter indexes ──────────────────────────────────────────
create index if not exists documents_document_type_idx on documents (document_type);
create index if not exists documents_owner_id_idx       on documents (owner_id);

-- ── Backfill legacy rows from the old metadata jsonb ────────────────
update documents set
  file_name     = coalesce(file_name, metadata->>'file'),
  document_type = coalesce(document_type, metadata->>'type'),
  chunk_index   = coalesce(chunk_index, 0)
where file_name is null and metadata is not null;
