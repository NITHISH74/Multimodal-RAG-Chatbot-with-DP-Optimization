-- ════════════════════════════════════════════════════════════════════
--  0003_phase3_threshold_and_hybrid.sql
--  Phase 3: enforce match_threshold in the WHERE clause (it was ignored).
--  Phase 4: add keyword_search() for full-text / hybrid retrieval.
--
--  All functions now return the first-class chunk metadata columns so the
--  app can build citations (Phase 7) without re-reading the jsonb.
-- ════════════════════════════════════════════════════════════════════

-- Drop old signatures (return type changes => must drop first).
drop function if exists match_documents_gemini(vector, float, int);
drop function if exists match_documents_cohere(vector, float, int);

-- ── Gemini vector search (3072 dims, halfvec-indexed) ───────────────
create or replace function match_documents_gemini (
  query_embedding vector(3072),
  match_threshold float,
  match_count int,
  filter_type text default null,
  filter_owner text default null
) returns table (
  id bigint, content text, file_name text, document_type text,
  page_number int, chunk_index int, source_url text, image_path text,
  upload_date timestamptz, similarity float
)
language sql stable as $$
  select id, content, file_name, document_type, page_number, chunk_index,
         source_url, image_path, upload_date,
         1 - (embedding_gemini <=> query_embedding) as similarity
  from documents
  where embedding_gemini is not null
    and (filter_type is null or document_type = filter_type)
    and (filter_owner is null or owner_id = filter_owner)
    and (1 - (embedding_gemini <=> query_embedding)) >= match_threshold   -- Phase 3 fix
  order by embedding_gemini::halfvec(3072) <=> (query_embedding::halfvec(3072))
  limit match_count;
$$;

-- ── Cohere vector search (1536 dims) ────────────────────────────────
create or replace function match_documents_cohere (
  query_embedding vector(1536),
  match_threshold float,
  match_count int,
  filter_type text default null,
  filter_owner text default null
) returns table (
  id bigint, content text, file_name text, document_type text,
  page_number int, chunk_index int, source_url text, image_path text,
  upload_date timestamptz, similarity float
)
language sql stable as $$
  select id, content, file_name, document_type, page_number, chunk_index,
         source_url, image_path, upload_date,
         1 - (embedding_cohere <=> query_embedding) as similarity
  from documents
  where embedding_cohere is not null
    and (filter_type is null or document_type = filter_type)
    and (filter_owner is null or owner_id = filter_owner)
    and (1 - (embedding_cohere <=> query_embedding)) >= match_threshold   -- Phase 3 fix
  order by embedding_cohere <=> query_embedding
  limit match_count;
$$;

-- ── Full-text keyword search (Phase 4) — model-agnostic ─────────────
create or replace function keyword_search (
  query_text text,
  match_count int,
  filter_type text default null,
  filter_owner text default null
) returns table (
  id bigint, content text, file_name text, document_type text,
  page_number int, chunk_index int, source_url text, image_path text,
  upload_date timestamptz, keyword_rank float
)
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
