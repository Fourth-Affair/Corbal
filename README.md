# Next.js + FastAPI starter

This repository runs a Next.js frontend and a FastAPI backend with Docker Compose.

## Start the app

```sh
docker compose up --build
```

Open [http://localhost:3000](http://localhost:3000) for the frontend. The API is available at [http://localhost:8000](http://localhost:8000), with interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

The frontend and backend source folders are mounted into their containers for development. Both services reload when their source changes. Rebuild the services after changing their dependencies:

```sh
docker compose up --build
```

Stop the stack with `Ctrl+C`, or run `docker compose down` in another terminal.

## Store uploaded images in Supabase Storage

Image uploads are sent directly from the browser to a Supabase Storage bucket. PDFs continue to be previewed locally and are not uploaded. The frontend needs the Supabase project URL and the public anon/publishable key; never put a `service_role` or secret key in frontend environment variables.

1. In the Supabase dashboard, open **Storage** and create a bucket named `plans` (or choose another name). Keep it private unless you specifically want uploaded images to be publicly readable.
2. Open **Project Settings → API** and copy the project URL and anon/publishable key.
3. Create `frontend/.env.local` using `frontend/.env.example`, and fill in the values. For Docker Compose, also copy those same three variables into a root `.env` file (next to `compose.yaml`), since Compose passes them to the frontend container.
4. Allow browser uploads by adding this policy in **SQL Editor**. It allows anonymous users to add objects to this bucket; access to read or delete objects remains restricted.

   ```sql
   create policy "Allow browser image uploads to plans"
   on storage.objects
   for insert
   to anon
   with check (
     bucket_id = 'plans'
     and (storage.extension(name) = 'png'
       or storage.extension(name) = 'jpg'
       or storage.extension(name) = 'jpeg'
       or storage.extension(name) = 'webp'
       or storage.extension(name) = 'gif')
   );
   ```

   If you use a different bucket name, change `bucket_id` in the policy and set `NEXT_PUBLIC_SUPABASE_BUCKET` to that name in both env files.
5. Restart the frontend (`docker compose up --build` for Docker, or restart `npm run dev` if running locally). Select or drop an image; the UI reports whether the upload succeeded. Objects appear under **Storage → plans**.

This setup uses the anon key, which is intended to be exposed in browser apps. The Storage policy controls what that key can do. For a production app with user accounts, restrict inserts to authenticated users and their own folder, and set appropriate bucket file-size and MIME-type limits in Supabase.

## Plan records and image associations

The migration in `supabase/migrations/` creates `plans` and `plan_images`. Each uploaded image gets a plan UUID; `plan_images` stores the Supabase bucket and object path, and the plan has a JSONB `metadata` field for future data. The frontend displays the plan UUID after it saves the association.

Apply the migration with the Supabase CLI from the repository root:

```sh
supabase login
supabase link --project-ref your-project-ref
supabase db push
```

Or paste the migration SQL into Supabase **SQL Editor** and run it. Keep the migration file in source control either way.

Set `SUPABASE_SERVICE_ROLE_KEY` in the root `.env` for Docker Compose, using the service role secret from **Project Settings → API**. This key is only passed to FastAPI and must never use a `NEXT_PUBLIC_` prefix. The backend lookup endpoint is `GET /api/plans/{plan_id}`; it returns the plan and linked image metadata, including each `bucket_id` and `object_path`. It uses the service key to read the RLS-protected tables.
