alter table public.plan_scales
  alter column points drop not null,
  alter column distance drop not null,
  alter column unit drop not null,
  add column if not exists accepted_m_per_px double precision check (accepted_m_per_px > 0),
  add column if not exists accepted_source text check (accepted_source in ('dimensions', 'written_areas'));
