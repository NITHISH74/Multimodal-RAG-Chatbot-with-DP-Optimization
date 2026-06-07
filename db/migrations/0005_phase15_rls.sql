-- ════════════════════════════════════════════════════════════════════
--  0005_phase15_rls.sql  —  Phase 15: Row-Level Security (defense-in-depth)
--
--  SECURITY MODEL
--    * The app performs all privileged work (inserts, updates, storage,
--      history) server-side using the SERVICE_ROLE key. service_role has
--      BYPASSRLS, so these policies do NOT affect the app's normal flow.
--    * These policies only constrain the ANON key. If you ever use the
--      anon key (e.g. a read-only path, or it leaks), the blast radius is
--      limited to READ-ONLY access — no writes, no deletes.
--    * Reads are public by default to keep the shared-knowledge-base UX.
--      To make documents private per user you need Supabase Auth; see the
--      commented owner-scoped policy at the bottom.
--
--  Safe to run multiple times (policies are dropped/recreated).
-- ════════════════════════════════════════════════════════════════════

alter table documents     enable row level security;
alter table chat_sessions enable row level security;
alter table chat_messages enable row level security;

-- ── documents: anon may READ, never write ───────────────────────────
drop policy if exists documents_anon_read on documents;
create policy documents_anon_read
  on documents for select
  to anon, authenticated
  using (true);
-- (No insert/update/delete policy => denied for anon/authenticated.
--  service_role bypasses RLS and continues to write normally.)

-- ── chat tables: anon may READ, never write ─────────────────────────
drop policy if exists chat_sessions_anon_read on chat_sessions;
create policy chat_sessions_anon_read
  on chat_sessions for select
  to anon, authenticated
  using (true);

drop policy if exists chat_messages_anon_read on chat_messages;
create policy chat_messages_anon_read
  on chat_messages for select
  to anon, authenticated
  using (true);

-- ════════════════════════════════════════════════════════════════════
--  OPTIONAL — true per-user isolation (requires Supabase Auth / login).
--  If you add auth and set documents.owner_id = auth.uid() on insert,
--  replace the public read policy above with the owner-scoped one below:
--
--    drop policy if exists documents_anon_read on documents;
--    create policy documents_owner_read
--      on documents for select
--      to authenticated
--      using (owner_id = auth.uid()::text);
-- ════════════════════════════════════════════════════════════════════
