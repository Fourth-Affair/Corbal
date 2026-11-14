create table public.plans (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb
);

create table public.plan_images (
  id uuid primary key default gen_random_uuid(),
  plan_id uuid not null references public.plans(id) on delete cascade,
  bucket_id text not null,
  object_path text not null,
  role text not null default 'source',
  created_at timestamptz not null default now(),
  unique (bucket_id, object_path)
);

create index plan_images_plan_id_idx on public.plan_images(plan_id);

alter table public.plans enable row level security;
alter table public.plan_images enable row level security;

-- Browser clients may create plan records after a successful Storage upload.
-- Reads remain restricted; the backend uses its service role key to look up plans.
create policy "Allow anonymous plan creation"
  on public.plans for insert to anon
  with check (true);

create policy "Allow anonymous plan image linking"
  on public.plan_images for insert to anon
  with check (bucket_id <> '' and object_path <> '');

grant insert on public.plans to anon;
grant insert on public.plan_images to anon;
