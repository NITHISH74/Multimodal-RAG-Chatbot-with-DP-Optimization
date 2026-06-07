-- ════════════════════════════════════════════════════════════════════
--  0001_phase1_hnsw.sql  —  Phase 1: ANN indexes for faster retrieval
--
--  WHY THIS IS NOT A PLAIN HNSW INDEX ON BOTH COLUMNS:
--    pgvector's HNSW (and IVFFlat) indexes support a maximum of 2000
--    dimensions for the `vector` type.
--      * embedding_cohere = vector(1536)  -> indexes directly.            ✅
--      * embedding_gemini = vector(3072)  -> EXCEEDS the 2000 limit.      ❌
--    For the 3072-dim Gemini column we build the HNSW index on a
--    half-precision (`halfvec`) cast, which supports up to 4000 dims.
--    Requires pgvector >= 0.7.0 (Supabase ships 0.8.x — verify with
--    `select extversion from pg_extension where extname = 'vector';`).
--
--  These indexes accelerate the `<=>` (cosine distance) ordering used by
--  match_documents_gemini / match_documents_cohere. Build can take a
--  while on large tables; it is safe to run while the app is live.
-- ════════════════════════════════════════════════════════════════════

-- ── Cohere (1536 dims): standard HNSW, cosine ───────────────────────
create index if not exists documents_embedding_cohere_hnsw
  on documents using hnsw (embedding_cohere vector_cosine_ops)
  with (m = 16, ef_construction = 64);

-- ── Gemini (3072 dims): HNSW over a halfvec cast, cosine ────────────
create index if not exists documents_embedding_gemini_hnsw
  on documents using hnsw ((embedding_gemini::halfvec(3072)) halfvec_cosine_ops)
  with (m = 16, ef_construction = 64);

-- The Gemini index above is only used by the planner when the query
-- ORDERs BY the SAME halfvec expression. Re-create the function so its
-- ordering matches the index. Similarity is still reported in full
-- precision; only the ANN ordering uses halfvec. (Behaviour preserved;
-- threshold filtering is added in Phase 3.)
create or replace function match_documents_gemini (
  query_embedding vector(3072),
  match_threshold float,
  match_count int
) returns table ( id bigint, content text, metadata jsonb, similarity float )
language sql stable as $$
  select id, content, metadata, 1 - (embedding_gemini <=> query_embedding) as similarity
  from documents
  where embedding_gemini is not null
  order by embedding_gemini::halfvec(3072) <=> (query_embedding::halfvec(3072))
  limit match_count;
$$;
