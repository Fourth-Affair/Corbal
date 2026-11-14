create or replace function public.create_plan_with_image(
  p_plan_id uuid,
  p_bucket_id text,
  p_object_path text,
  p_role text default 'source',
  p_metadata jsonb default '{}'::jsonb
)
returns uuid
language plpgsql
security invoker
set search_path = public
as $$
begin
  if p_bucket_id is null or p_bucket_id = '' then
    raise exception 'bucket id is required';
  end if;
  if p_object_path is null or p_object_path not like p_plan_id::text || '/%' then
    raise exception 'object path must be under the plan id folder';
  end if;

  insert into public.plans (id, metadata) values (p_plan_id, coalesce(p_metadata, '{}'::jsonb));
  insert into public.plan_images (
    plan_id, bucket_id, object_path, role, upload_status
  ) values (
    p_plan_id, p_bucket_id, p_object_path, coalesce(nullif(p_role, ''), 'source'), 'pending'
  );

  return p_plan_id;
end;
$$;

revoke all on function public.create_plan_with_image(uuid, text, text, text, jsonb) from public;
grant execute on function public.create_plan_with_image(uuid, text, text, text, jsonb) to anon;
