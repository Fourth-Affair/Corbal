# Hosting on a single server (Verda CPU instance)

```
internet ──443/80──> caddy (HTTPS, per-IP rate limits) ──> frontend:3000 (next start) ──> backend:8000 (uvicorn)
                                                                                              │
                                                       Supabase (plans, analyses, corrections, uploaded files)
```

Only Caddy is published. Frontend and backend live on the internal compose network. The backend's dataset endpoints
(`/api/ask`, `/api/render`, ...) are not reachable from outside: the browser only gets the Next.js `/api/plans/*` routes.

## 1. Create the instance (Verda dashboard)

- CPU instance, **4+ vCPU, 16 GB RAM (8 GB works with `MAX_ANALYSIS_PIXELS=10000000`), 40+ GB disk**, Ubuntu image,
  your SSH key. Analysis peaks at about 1.7 GB + 0.27 GB per megapixel of the plan (measured: 2.3 GB at 2 MP, 4.9 GB
  at 12 MP). Larger uploads are downscaled to `MAX_ANALYSIS_PIXELS` first.
- Note the public IP, e.g. `65.108.1.2`. The site address is then `65-108-1-2.sslip.io` (sslip.io resolves it to
  that IP, so Let's Encrypt can issue a certificate without buying a domain).

## 2. Supabase

1. **Storage**: create a private bucket named `plans` (or set `NEXT_PUBLIC_SUPABASE_BUCKET`). If you set
   *Allowed MIME types* on it, include `image/png`, `image/jpeg`, `image/webp` and `application/pdf`.
2. **Migrations**: apply everything in `supabase/migrations/` in timestamp order, either with the CLI
   (`supabase link --project-ref <ref> && supabase db push`) or by pasting each file into the SQL Editor. They create
   the tables (`plans`, `plan_images`, `plan_knowledge`, `plan_scales`, `plan_outlines`, `plan_corrections`,
   `plan_answer_feedback`, `plan_geometry`, ...), the RPCs the upload uses, and the Storage policy that lets the anon
   key upload images and PDFs to `plans`. A missing migration shows up as a 503 "needs administrator attention" or
   a 502 from the matching feature. PDF uploads failing with "new row violates row-level security policy" means
   `202611140010_allow_pdf_uploads.sql` is missing.
3. **Keys** (*Project Settings → API*) in `.env`: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY` (public,
   goes into the browser bundle) and `SUPABASE_SERVICE_ROLE_KEY` (backend only; never give it a `NEXT_PUBLIC_` prefix).

`NEXT_PUBLIC_*` values are baked in by `next build`: after changing them, rebuild the frontend (`up` does).

## 3. Deploy with the script (from your laptop, repo root)

```sh
deploy/deploy.sh <ip> all        # first time: setup + push + up
```

Put `DEPLOY_HOST=<ip>` in `.env` to leave out the IP. `SITE_ADDRESS` defaults to `<ip-with-dashes>.sslip.io`.

| Command | What it does |
|---|---|
| `setup` | Runs `deploy/bootstrap.sh` on the server: Docker, ufw (22/80/443 only), fail2ban. Once per server. |
| `push` | Copies the code to `/srv/stoa`: tracked files of your working tree, uncommitted edits included, untracked files not (it warns). Also uploads the 208 MB CubiCasa weights if missing (from `WEIGHTS_SRC`, default `dataset/cubicasa-pretrained/model_best_val_loss_var.pkl`), `.env` (without `VERDA_*` / `DEPLOY_HOST`, plus `SITE_ADDRESS`) and the Gemini answer cache. Keeps the server's `.env`, `dataset/` and weights. |
| `up` | `docker compose -f compose.prod.yaml up -d --build`, then checks that the site answers. The first build takes ~5–10 min. |
| `update` | `push` + `up`: deploys a new version. |
| `status` / `logs [service]` / `down` / `ssh` | Health and URL / follow logs / stop / shell in `/srv/stoa`. |

The code goes from your laptop over SSH (`git archive`), not with `git clone`, because the GitHub repo is private. The
dataset is not needed: the web app works on uploaded plans (stored in Supabase). Caddy fetches the certificate on
the first request, which takes a few seconds.

**`.env` changes are not deployed by CI.** After editing a value, run `deploy/deploy.sh push` and then `up`, or edit
`/srv/stoa/.env` on the server and run `docker compose -f compose.prod.yaml up -d`.

## Settings, cost and abuse protection

| Setting (`.env`)        | Default | Effect |
|-------------------------|---------|--------|
| `GEMINI_API_KEY`        | (empty) | Enables the Gemini agent. Without it, the template answerer replies (counts only). |
| `GEMINI_MODEL`          | `gemini-3.8-flash` | Agent model. If Google retires the id, every question silently falls back to the template answerer (logged as `Gemini agent failed`): set a current model here. |
| `GEMINI_MAX_PER_HOUR`   | 300     | Uncached Gemini questions per rolling hour; beyond it the template answerer replies (no Gemini cost). Cached answers don't count. Counted in the backend process, so it resets on restart. |
| `RATE_ASK_PER_MIN`      | 20      | Questions per client IP per minute (Caddy returns 429 beyond). |
| `RATE_ANALYZE_PER_MIN`  | 6       | Uploads + analyses per client IP per minute (CPU-heavy). |
| `MAX_CONCURRENT_ANALYSES` | 1     | Heavy jobs (analysis, room / wall geometry) running at once in the backend. Each takes the memory above: raise it only with RAM to spare. |
| `ANALYSIS_QUEUE_TIMEOUT` | 45     | Seconds a heavy job waits for a slot before returning a retryable 503 "busy". |
| `MAX_ANALYSIS_PIXELS`   | 16000000 | Uploads above this are downscaled before analysis, and PDF pages are rendered at most this large. Citations still line up with the original image. |

Also set a **budget alert / spending cap** for the Gemini key in Google AI Studio / Cloud Billing.

## Things that can go wrong

- **Slow first analysis.** Loading the models plus analysing a large plan takes 1–3 minutes on a 4 vCPU server
  (85 s for a 12 MP plan on a laptop). Node's `fetch` from the Next.js proxy gives up after 300 s and the user sees
  "Could not reach the analysis backend". The backend still finishes and saves the result, so opening the plan
  again loads it.
- **Busy server.** With `MAX_CONCURRENT_ANALYSES=1`, a second upload waits up to `ANALYSIS_QUEUE_TIMEOUT` and then
  gets a retryable 503. The page shows the message.
- **Disk.** `dataset/cache/` on the server keeps every uploaded plan's PNG / PDF, features (~0.2 MB per plan),
  analysis graphs and cached Gemini answers, without any eviction. Everything in it can be rebuilt from Supabase,
  so pruning old files is safe. Container logs are rotated (5 × 20 MB per service).
- **Supabase outage.** Analyses are also saved locally (`dataset/cache/plan-graphs/`), and questions keep working
  from there. Corrections, scale, outlines and new uploads need Supabase.
- **Backend replicas.** The analysis lock, the job limit and the Gemini cap live in one process. Add a shared queue,
  lock and cache before running more than one backend container or uvicorn worker.
- **No accounts.** Anyone with a plan's URL can view, ask, correct and recalibrate it. Plan ids are random UUIDs, but
  links are shareable. Add Supabase Auth and per-user policies before handling private plans.
- **Public IP.** It lives as long as the instance. If it changes, update `SITE_ADDRESS` in `.env`, the
  `DEPLOY_HOST` secret and the environment URL in `.github/workflows/deploy.yml`, then run
  `docker compose -f compose.prod.yaml up -d`.
- **Firewall.** Docker-published ports bypass ufw. Don't add `ports:` to `frontend` or `backend`.
- **Licence.** The CubiCasa net weights are CC BY-NC: fine for a public demo, not for a commercial service.

## Automatic deployment (GitHub Actions)

`.github/workflows/deploy.yml`: every push to `main` runs the backend tests, then `deploy/deploy.sh ci` (code only:
`.env`, weights and cache stay on the server), then checks that the site answers. A failed image build leaves the
running version in place. Manual run: Actions → Deploy → Run workflow. Without the three secrets below, the deploy
job fails at once (it prints the script's usage).

One-time setup (the server must have been deployed once with `deploy.sh <ip> all`). The repo is
`Fourth-Affair/Junc-2026-FA` (the `origin` remote); if it moved from another owner, check that the secrets moved with it:

```sh
ssh-keygen -t ed25519 -N "" -C github-actions-deploy -f /tmp/stoa_deploy_ci      # CI-only key, no passphrase
ssh root@<ip> 'cat >> ~/.ssh/authorized_keys' < /tmp/stoa_deploy_ci.pub
gh secret set DEPLOY_SSH_KEY     -R Fourth-Affair/Junc-2026-FA < /tmp/stoa_deploy_ci
ssh-keyscan -t ed25519 <ip> | gh secret set DEPLOY_KNOWN_HOSTS -R Fourth-Affair/Junc-2026-FA
printf '<ip>' | gh secret set DEPLOY_HOST -R Fourth-Affair/Junc-2026-FA
rm /tmp/stoa_deploy_ci /tmp/stoa_deploy_ci.pub
```
