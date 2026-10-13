# Deployment

Target: Cloudflare Workers + R2 + D1, free tier. All commands run from `worker/` unless noted.

## Prerequisites
| Requirement | Notes |
|---|---|
| Node ≥ 20, npm | `npm install` pulls a pinned `wrangler` |
| Cloudflare account | R2 enabled in the dashboard (**R2 Object Storage → Enable**). A payment method is required, but the free tier costs $0. Without it, API calls fail with `code: 10042` |
| `localhost` resolvable | `wrangler login` runs its OAuth callback on `localhost:8976`, and `wrangler dev` uses loopback too. `/etc/hosts` must contain `127.0.0.1 localhost` (check with `getent hosts localhost`) |

## Provision
```bash
cp wrangler.toml.example wrangler.toml      # gitignored; holds account-specific ids
npm install
npx wrangler login
npx wrangler r2 bucket create ml-data-bucket
npx wrangler d1 create ml-data-bucket       # → set database_id in wrangler.toml
npm run db:init:remote                      # applies schema.sql (idempotent)
openssl rand -hex 32 | npx wrangler secret put API_KEY
npm run deploy                              # prints BASE_URL: https://ml-data-bucket.<subdomain>.workers.dev
```
A new account is asked to register a `workers.dev` subdomain on its first deploy. `workers_dev = true` in `wrangler.toml` keeps that route enabled.

Keep a copy of the API key, because `wrangler secret put` can't read it back. If you lose it, rotate it (see below).

## Verify
```bash
B=https://ml-data-bucket.<subdomain>.workers.dev
curl -s -o /dev/null -w '%{http_code}\n' $B/v1/projects                         # 401 → auth enforced
curl -s -H "Authorization: Bearer $API_KEY" $B/v1/projects                      # {"projects":[]}
```
`500 internal error` on the second call means the schema was not applied: run `npm run db:init:remote`.

Run a smoke test from the repo root with `.env` set:
```bash
set -a; . ./.env; set +a
.venv/bin/python device/example_yolo_loop.py --project smoke-test --synthetic 5 --keep-all
(cd tools && ../.venv/bin/python export_raw.py --project smoke-test --labels yolo --delete-after)
```

## Configuration
| Name | Kind | Where | Purpose |
|---|---|---|---|
| `API_KEY` | secret | `wrangler secret put` | Bearer token for every route |
| `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` | secret | `wrangler secret put` | Enable presigned uploads (optional) |
| `R2_ACCOUNT_ID` | var | `wrangler.toml [vars]` | Enable presigned uploads (optional) |
| `R2_BUCKET_NAME` | var | `wrangler.toml [vars]` | Bucket used in presigned URLs |
| `DB`, `BUCKET` | bindings | `wrangler.toml` | D1 / R2 |

Clients (device and exporter) read `BASE_URL` and `API_KEY` from the environment. See `.env.example`.

## Upload modes
| Mode | Enabled when | Path | Cost per image |
|---|---|---|---|
| `direct` (default) | presign config absent | device → Worker → R2 (≤ 25 MB) | 3 Worker requests |
| `presigned` | all three `R2_*` values set | device → R2 (5-min SigV4 URL) | 2 Worker requests, no Worker bandwidth |

To switch to presigned: create an R2 API token (**R2 → Manage API tokens**, Object Read & Write on the bucket), then:
```bash
npx wrangler secret put R2_ACCESS_KEY_ID
npx wrangler secret put R2_SECRET_ACCESS_KEY
# set R2_ACCOUNT_ID in wrangler.toml
npm run deploy
```
Devices need egress to `*.r2.cloudflarestorage.com`. No client change is needed, because the mode is negotiated per request.

## Free-tier budget
| Resource | Limit | Practical ceiling |
|---|---|---|
| Workers | 100k req/day | ~30k images/day (direct), ~50k (presigned) |
| R2 storage | 10 GB | ~60k JPEGs at ~150 KB. Use `export_raw.py --delete-after` to free space |
| R2 ops | 1M Class A, 10M Class B / month | Not a constraint at this volume |
| D1 | 5 GB, 100k writes/day | Rows are ~1 KB; 2 writes per image |

Usage is shown in the dashboard under **Workers & Pages → ml-data-bucket → Metrics** and **R2 → ml-data-bucket → Metrics**. Billing notifications are under **Manage Account → Billing**.

## Operations
```bash
npx wrangler tail                                   # live request logs
npx wrangler deployments list                       # version history
npx wrangler rollback [<version-id>]                # revert the Worker (does not touch data)
npx wrangler d1 execute ml-data-bucket --remote \
  --command "SELECT project, COUNT(*), SUM(uploaded) FROM samples GROUP BY project"
npx wrangler d1 export ml-data-bucket --remote --output backup.sql   # metadata backup
```

**Key rotation:** run `wrangler secret put API_KEY`. It takes effect immediately with no redeploy. Update `API_KEY` on every device afterwards. Their queues keep the data and retry once the key matches, but a 401 parks items as `failed`. Re-queue them with `collector.py --retry-failed` or `Collector.retry_failed()`.

**Schema changes:** add `ALTER TABLE` statements as a new `.sql` file and apply it with `wrangler d1 execute ml-data-bucket --remote --file <file>`. `schema.sql` only uses `IF NOT EXISTS`, so re-running it is safe but it won't migrate existing tables.

**Teardown:**
```bash
npx wrangler delete
npx wrangler d1 delete ml-data-bucket
npx wrangler r2 bucket delete ml-data-bucket   # the bucket must be empty first
```

## Local development
```bash
cp -n wrangler.toml.example wrangler.toml
echo 'API_KEY=dev-secret-key' > .dev.vars          # gitignored
npm run db:init:local
npm run dev                                         # http://127.0.0.1:8787, R2 + D1 emulated in .wrangler/state
npm run typecheck && (cd ../tools && ../.venv/bin/pytest -q)
```
Local dev always uses `direct` mode, because presigned URLs point at real R2.

## Files not committed
`worker/wrangler.toml`, `worker/.dev.vars`, `.env`, `exports/`. Templates are provided: `wrangler.toml.example` and `.env.example`.
