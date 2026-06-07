-- ════════════════════════════════════════════════════════════════════
--  0004_phase8_storage.sql  —  Phase 8: image storage bucket
--
--  Images are no longer stored as base64 inside the documents table.
--  They are uploaded to Supabase Storage and only the path/URL is kept
--  (documents.image_path, added in 0002).
--
--  Create the storage bucket. This can also be done from the Supabase
--  dashboard (Storage -> New bucket -> name "rag-images"). Run via SQL:
-- ════════════════════════════════════════════════════════════════════

insert into storage.buckets (id, name, public)
values ('rag-images', 'rag-images', true)
on conflict (id) do nothing;

-- Public read policy so retrieved image URLs render in the UI.
-- (Adjust to authenticated-only if you add per-user isolation in Phase 15.)
do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'storage' and tablename = 'objects'
      and policyname = 'rag_images_public_read'
  ) then
    create policy "rag_images_public_read"
      on storage.objects for select
      using (bucket_id = 'rag-images');
  end if;
end $$;
