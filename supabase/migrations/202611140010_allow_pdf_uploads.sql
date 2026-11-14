-- Browser uploads to the `plans` bucket: allow PDFs as well as images. The backend parses vector PDFs
-- (app/pdfplan); the original policy (README, "Store uploaded images") only allowed image extensions, so every PDF
-- upload failed with "new row violates row-level security policy".
drop policy if exists "Allow browser image uploads to plans" on storage.objects;
drop policy if exists "Allow browser plan uploads to plans" on storage.objects;

create policy "Allow browser plan uploads to plans"
on storage.objects
for insert
to anon
with check (
  bucket_id = 'plans'
  and lower(storage.extension(name)) in ('png', 'jpg', 'jpeg', 'webp', 'gif', 'pdf')
);
