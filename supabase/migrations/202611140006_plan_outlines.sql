create table if not exists public.plan_outlines (
  plan_id uuid primary key references public.plans(id) on delete cascade,
  outlines jsonb not null,
  updated_at timestamptz not null default now()
);

alter table public.plan_outlines enable row level security;
