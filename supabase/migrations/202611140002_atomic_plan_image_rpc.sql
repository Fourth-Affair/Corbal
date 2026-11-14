alter table public.plan_images
  add column upload_status text not null default 'uploaded'
  check (upload_status in ('pending', 'uploaded', 'failed'));

create or replace function public.create_plan_with_image(
  p_plan_id uuid,
  p_bucket_id text,
  p_object_path text,
  p_role text default 'source'
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

  insert into public.plans (id) values (p_plan_id);
  insert into public.plan_images (
    plan_id, bucket_id, object_path, role, upload_status
  ) values (
    p_plan_id, p_bucket_id, p_object_path, coalesce(nullif(p_role, ''), 'source'), 'pending'
  );

  return p_plan_id;
end;
$$;

create or replace function public.set_plan_image_upload_status(
  p_plan_id uuid,
  p_object_path text,
  p_status text
)
returns void
language plpgsql
security definer
set search_path = public, pg_temp
as $$
begin
  if p_status not in ('uploaded', 'failed') then
    raise exception 'status must be uploaded or failed';
  end if;

  update public.plan_images
  set upload_status = p_status
  where plan_id = p_plan_id
    and object_path = p_object_path
    and upload_status = 'pending';

  if not found then
    raise exception 'pending plan image not found';
  end if;
end;
$$;

revoke all on function public.create_plan_with_image(uuid, text, text, text) from public;
revoke all on function public.set_plan_image_upload_status(uuid, text, text) from public;
grant execute on function public.create_plan_with_image(uuid, text, text, text) to anon;
grant execute on function public.set_plan_image_upload_status(uuid, text, text) to anon;
