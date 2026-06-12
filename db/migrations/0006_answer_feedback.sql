-- 0006_answer_feedback.sql - User feedback for answer quality analytics.
-- Safe to run multiple times.

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

create index if not exists answer_feedback_session_idx on answer_feedback (session_id);
create index if not exists answer_feedback_rating_idx on answer_feedback (rating);

alter table answer_feedback enable row level security;

drop policy if exists answer_feedback_anon_read on answer_feedback;
create policy answer_feedback_anon_read
  on answer_feedback for select
  to anon, authenticated
  using (true);

notify pgrst, 'reload schema';
