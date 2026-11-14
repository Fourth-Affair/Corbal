# Hosting on a single server (Verda CPU instance)

```
internet ──443/80──> caddy (HTTPS, per-IP rate limits) ──> frontend:3000 (next start) ──> backend:8000 (uvicorn)
```

Only Caddy is published. Frontend and backend live on the internal compose network.

## 1. Create the instance (Verda dashboard)

- CPU instance, **4+ vCPU, 8–16 GB RAM, 40+ GB disk**, Ubuntu image, your SSH key.
- Note the public IP, e.g. `65.108.1.2`. The site address is then `65-108-1-2.sslip.io` (sslip.io resolves it to
  that IP, so Let's Encrypt can issue a certificate without buying a domain).

## 2. Deploy with the script (from your laptop, repo root)

```sh
deploy/deploy.sh <ip> all        # first time: setup + push + up
```

Put `DEPLOY_HOST=<ip>` in `.env` to leave out the IP. `SITE_ADDRESS` defaults to `<ip-with-dashes>.sslip.io`.

| Command | What it does |
|---|---|
| `setup` | Runs `deploy/bootstrap.sh` on the server: Docker, ufw (22/80/443 only), fail2ban. Once per server. |
| `push` | Copies the code to `/srv/stoa`: tracked files of your working tree, uncommitted edits included, untracked files not (it warns). Also uploads the 208 MB CubiCasa weights if missing, `.env` (without `VERDA_*`, plus `SITE_ADDRESS`) and the warm answer cache. Keeps the server's `.env`, `dataset/` and weights. |
| `up` | `docker compose -f compose.prod.yaml up -d --build`, then checks that the site answers. The first build takes ~5–10 min. |
| `update` | `push` + `up`: deploys a new version. |
| `status` / `logs [service]` / `down` / `ssh` | Health and URL / follow logs / stop / shell in `/srv/stoa`. |

The GitHub repo is private, so the code goes from your laptop over SSH (`git archive`), not with `git clone`. The
dataset is not needed: the web app works on uploaded plans (stored in Supabase). Caddy fetches the certificate on
the first request, which takes a few seconds.

## Cost and abuse protection

| Setting (`.env`)        | Default | Effect |
|-------------------------|---------|--------|
| `GEMINI_MAX_PER_HOUR`   | 300     | Uncached Gemini questions per rolling hour; beyond it the template answerer replies (no Gemini cost). Cached answers don't count. |
| `RATE_ASK_PER_MIN`      | 20      | Questions per client IP per minute (Caddy returns 429 beyond). |
| `RATE_ANALYZE_PER_MIN`  | 6       | Uploads + analyses per client IP per minute (CPU-heavy). |

Also set a **budget alert / spending cap** for the Gemini key in Google AI Studio / Cloud Billing.

## Notes

- The public IP lives as long as the instance. If it changes, update `SITE_ADDRESS` in `.env` and run
  `docker compose -f compose.prod.yaml up -d`.
- Docker-published ports bypass ufw. Don't add `ports:` to `frontend` or `backend`.
- The CubiCasa net weights are CC BY-NC: fine for a public demo, not for a commercial service.
