-- Rooms (polygons + the text read inside each room), walls and openings of an analyzed plan, for the room overlay
-- and the answering agent. One row per plan; `geometry` is the backend's PlanGeometry JSON (app/models.py).
-- Coordinates are pixels of the plan image (PDF plans: points of page 1), origin top-left.
create table if not exists public.plan_geometry (
  plan_id uuid primary key references public.plans(id) on delete cascade,
  geometry jsonb not null,
  model text not null,
  labeled_rooms integer not null default 0,
  rough_rooms integer not null default 0,
  updated_at timestamptz not null default now()
);

alter table public.plan_geometry enable row level security;

-- Only the backend service role reads and writes plan geometry. No anon policies or grants are created.
