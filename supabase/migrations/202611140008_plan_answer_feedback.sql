create table if not exists public.plan_answer_feedback (
  id uuid primary key default gen_random_uuid(),
  plan_id uuid not null references public.plans(id) on delete cascade,
  question text not null,
  answer text not null,
  feedback text not null check (feedback in ('helpful', 'incorrect', 'unclear')),
  reason text,
  evidence_ids jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists plan_answer_feedback_plan_id_created_at_idx
  on public.plan_answer_feedback(plan_id, created_at);

alter table public.plan_answer_feedback enable row level security;
-- Feedback is accessed only through the backend service role.
