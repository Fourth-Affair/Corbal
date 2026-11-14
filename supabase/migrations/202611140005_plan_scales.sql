create table if not exists public.plan_scales (
  plan_id uuid primary key references public.plans(id) on delete cascade,
  points jsonb not null,
  distance double precision not null check (distance > 0),
  unit text not null check (unit in ('m', 'cm', 'ft', 'in')),
  updated_at timestamptz not null default now()
);

alter table public.plan_scales enable row level security;
