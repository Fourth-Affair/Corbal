-- Rooms (polygons, type from combined evidence: OCR labels, fixtures, the 2019 net; strong / weak / unknown tier),
-- wall segments with lengths and the rooms on each side, walls and openings of an analyzed plan, for the overlays
-- and the answering agent. One row per plan; `geometry` is the backend's PlanGeometry JSON (app/models.py).
-- Coordinates are pixels of the plan image (PDF plans: points of page 1), origin top-left.
create table if not exists public.plan_geometry (
  plan_id uuid primary key references public.plans(id) on delete cascade,
  geometry jsonb not null,
  model text not null,
  strong_rooms integer not null default 0,
  weak_rooms integer not null default 0,
  updated_at timestamptz not null default now()
);

alter table public.plan_geometry enable row level security;

-- Only the backend service role reads and writes plan geometry. No anon policies or grants are created.
