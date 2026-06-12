-- ════════════════════════════════════════════════════════════════════
--  RUN_THIS_IN_SUPABASE.sql  —  ONE-SHOT SETUP (copy-paste convenience)
--
--  This is the combined, idempotent equivalent of migrations 0000–0005.
--  If you'd rather not open six files, just paste THIS whole file into the
--  Supabase SQL Editor and click "Run". Safe to run multiple times.
--
--  HOW:  Supabase Dashboard  ->  SQL Editor  ->  New query  ->
--        paste everything below  ->  Run.
--
--  After it succeeds, "Index Uploads" in the app will work.
-- ════════════════════════════════════════════════════════════════════

create extension if not exists vector;

-- ── Tables ──────────────────────────────────────────────────────────
create table if not exists documents (
  id bigserial primary key,
  content text,
  metadata jsonb,
  embedding_gemini vector(3072),
  embedding_cohere vector(1536)
);

create table if not exists chat_sessions (
  session_id text primary key,
  created_at timestamp with time zone default timezone('utc'::text, now()),
  embedding_model text,
  total_input_tokens integer default 0,
  total_output_tokens integer default 0,
  total_queries integer default 0
);

create table if not exists chat_messages (
  id bigserial primary key,
  session_id text references chat_sessions(session_id) on delete cascade,
  role text, content text, meta jsonb,
  created_at timestamp with time zone default timezone('utc'::text, now())
);

create table if not exists answer_feedback (
  id bigserial primary key,
  session_id text references chat_sessions(session_id) on delete cascade,
  message_index integer not null,
  rating text not null check (rating in ('up', 'down')),
  question text,
  answer text,
  comment text,
  meta jsonb default '{}'::jsonb,
  owner_id text,
  created_at timestamp with time zone default timezone('utc'::text, now()),
  unique (session_id, message_index)
);

-- ── Per-chunk columns (THIS fixes "content_hash does not exist") ────
alter table documents add column if not exists file_name     text;
alter table documents add column if not exists document_type text;
alter table documents add column if not exists page_number   int;
alter table documents add column if not exists chunk_index    int;
alter table documents add column if not exists content_hash  text;
alter table documents add column if not exists source_url     text;
alter table documents add column if not exists image_path     text;
alter table documents add column if not exists owner_id       text;
alter table documents add column if not exists upload_date   timestamptz default timezone('utc'::text, now());

create unique index if not exists documents_content_hash_uidx
  on documents (content_hash) where content_hash is not null;

alter table documents add column if not exists fts tsvector
  generated always as (to_tsvector('english', coalesce(content, ''))) stored;
create index if not exists documents_fts_gin on documents using gin (fts);
create index if not exists documents_document_type_idx on documents (document_type);
create index if not exists documents_owner_id_idx on documents (owner_id);
create index if not exists answer_feedback_session_idx on answer_feedback (session_id);
create index if not exists answer_feedback_rating_idx on answer_feedback (rating);

-- ── ANN indexes (Cohere direct; Gemini via halfvec, needs pgvector >=0.7) ──
create index if not exists documents_embedding_cohere_hnsw
  on documents using hnsw (embedding_cohere vector_cosine_ops) with (m = 16, ef_construction = 64);
create index if not exists documents_embedding_gemini_hnsw
  on documents using hnsw ((embedding_gemini::halfvec(3072)) halfvec_cosine_ops) with (m = 16, ef_construction = 64);

-- ── Search RPCs (threshold-filtered + hybrid) ──────────────────────
drop function if exists match_documents_gemini(vector, float, int);
drop function if exists match_documents_cohere(vector, float, int);
drop function if exists match_documents_gemini(vector, float, int, text, text);
drop function if exists match_documents_cohere(vector, float, int, text, text);

create or replace function match_documents_gemini (
  query_embedding vector(3072), match_threshold float, match_count int,
  filter_type text default null, filter_owner text default null
) returns table ( id bigint, content text, file_name text, document_type text,
  page_number int, chunk_index int, source_url text, image_path text,
  upload_date timestamptz, similarity float )
language sql stable as $$
  select id, content, file_name, document_type, page_number, chunk_index,
         source_url, image_path, upload_date,
         1 - (embedding_gemini <=> query_embedding) as similarity
  from documents
  where embedding_gemini is not null
    and (filter_type is null or document_type = filter_type)
    and (filter_owner is null or owner_id = filter_owner)
    and (1 - (embedding_gemini <=> query_embedding)) >= match_threshold
  order by embedding_gemini::halfvec(3072) <=> (query_embedding::halfvec(3072))
  limit match_count;
$$;

create or replace function match_documents_cohere (
  query_embedding vector(1536), match_threshold float, match_count int,
  filter_type text default null, filter_owner text default null
) returns table ( id bigint, content text, file_name text, document_type text,
  page_number int, chunk_index int, source_url text, image_path text,
  upload_date timestamptz, similarity float )
language sql stable as $$
  select id, content, file_name, document_type, page_number, chunk_index,
         source_url, image_path, upload_date,
         1 - (embedding_cohere <=> query_embedding) as similarity
  from documents
  where embedding_cohere is not null
    and (filter_type is null or document_type = filter_type)
    and (filter_owner is null or owner_id = filter_owner)
    and (1 - (embedding_cohere <=> query_embedding)) >= match_threshold
  order by embedding_cohere <=> query_embedding
  limit match_count;
$$;

create or replace function keyword_search (
  query_text text, match_count int,
  filter_type text default null, filter_owner text default null
) returns table ( id bigint, content text, file_name text, document_type text,
  page_number int, chunk_index int, source_url text, image_path text,
  upload_date timestamptz, keyword_rank float )
language sql stable as $$
  select id, content, file_name, document_type, page_number, chunk_index,
         source_url, image_path, upload_date,
         ts_rank(fts, websearch_to_tsquery('english', query_text)) as keyword_rank
  from documents
  where fts @@ websearch_to_tsquery('english', query_text)
    and (filter_type is null or document_type = filter_type)
    and (filter_owner is null or owner_id = filter_owner)
  order by keyword_rank desc
  limit match_count;
$$;

-- ── Image Storage bucket ────────────────────────────────────────────
insert into storage.buckets (id, name, public)
values ('rag-images', 'rag-images', true)
on conflict (id) do nothing;

-- ── Tell PostgREST (the API layer supabase-py uses) to reload its schema
--    cache, so the new functions/columns are visible immediately. Without
--    this, RPC calls can return empty results for a minute or two after a
--    direct-connection migration.
notify pgrst, 'reload schema';

-- ════════════════════════════════════════════════════════════════════
--  Done. Re-run "Index Uploads" in the app — it will work now.
--  (RLS from migration 0005 is optional; add it later for hardening.)
-- ════════════════════════════════════════════════════════════════════
