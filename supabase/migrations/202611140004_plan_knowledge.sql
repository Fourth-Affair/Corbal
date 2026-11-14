-- Persist the machine-readable analysis so later questions can reuse it without
-- downloading and analyzing the source image again.
create table public.plan_knowledge (
  plan_id uuid primary key references public.plans(id) on delete cascade,
  graph jsonb not null,
  updated_at timestamptz not null default now()
);

alter table public.plan_knowledge enable row level security;

-- Only the backend service role reads and writes analysis knowledge. No anon
-- policies or grants are created for this table.
