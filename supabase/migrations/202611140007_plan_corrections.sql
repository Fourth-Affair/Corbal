create table if not exists public.plan_corrections (
  id uuid primary key default gen_random_uuid(),
  plan_id uuid not null references public.plans(id) on delete cascade,
  action text not null check (action in ('add', 'update', 'delete')),
  item_id text not null,
  original_item jsonb,
  corrected_item jsonb,
  created_at timestamptz not null default now(),
  check (
    (action = 'add' and original_item is null and corrected_item is not null) or
    (action = 'update' and original_item is not null and corrected_item is not null) or
    (action = 'delete' and original_item is not null and corrected_item is null)
  )
);

create index if not exists plan_corrections_plan_id_created_at_idx
  on public.plan_corrections(plan_id, created_at);

alter table public.plan_corrections enable row level security;
-- Corrections are accessed only through the backend service role.
