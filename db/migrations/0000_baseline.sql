-- ════════════════════════════════════════════════════════════════════
--  0000_baseline.sql
--  Baseline schema for the Multi-Model RAG chatbot (Supabase pgvector).
--  This captures the schema that previously lived only in README.md so
--  the repository has a tracked source of truth.
--
--  Apply in the Supabase SQL Editor (or `supabase db push`) in order.
-- ════════════════════════════════════════════════════════════════════

create extension if not exists vector;

-- ── Document chunks / items ──────────────────────────────────────────
create table if not exists documents (
  id               bigserial primary key,
  content          text,
  metadata         jsonb,
  embedding_gemini vector(3072),
  embedding_cohere vector(1536)
);

-- ── Chat session bookkeeping ─────────────────────────────────────────
create table if not exists chat_sessions (
  session_id          text primary key,
  created_at          timestamp with time zone default timezone('utc'::text, now()),
  embedding_model     text,
  total_input_tokens  integer default 0,
  total_output_tokens integer default 0,
  total_queries       integer default 0
);

create table if not exists chat_messages (
  id          bigserial primary key,
  session_id  text references chat_sessions(session_id) on delete cascade,
  role        text,
  content     text,
  meta        jsonb,
  created_at  timestamp with time zone default timezone('utc'::text, now())
);

-- ── Vector search RPCs ───────────────────────────────────────────────
-- NOTE: match_threshold is accepted but NOT yet used for filtering here.
--       This is fixed in 0002_phase3_threshold.sql.

create or replace function match_documents_gemini (
  query_embedding vector(3072),
  match_threshold float,
  match_count int
) returns table ( id bigint, content text, metadata jsonb, similarity float )
language sql stable as $$
  select id, content, metadata, 1 - (embedding_gemini <=> query_embedding) as similarity
  from documents where embedding_gemini is not null
  order by embedding_gemini <=> query_embedding limit match_count;
$$;

create or replace function match_documents_cohere (
  query_embedding vector(1536),
  match_threshold float,
  match_count int
) returns table ( id bigint, content text, metadata jsonb, similarity float )
language sql stable as $$
  select id, content, metadata, 1 - (embedding_cohere <=> query_embedding) as similarity
  from documents where embedding_cohere is not null
  order by embedding_cohere <=> query_embedding limit match_count;
$$;
