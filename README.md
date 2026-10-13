# ml-data-bucket

Zero-cost collector for computer-vision inference data. Edge devices upload frames (and optionally their detections) per **project**, for example `plastic-bottle` or `people-counting`. You export each project as raw files for re-evaluation and retraining, optionally with YOLO/COCO pre-labels so you correct boxes instead of labelling from scratch.

```
Device ──────POST /v1/{project}/presign {meta?}──▶ Cloudflare Worker ──▶ D1 (metadata)
     └──PUT image ─────────────────────────────────▶ R2  {project}/{yyyy-mm-dd}/{id}.jpg
     └──POST /v1/{project}/confirm/{id}
```

| Piece | Free tier |
|---|---|
| R2 (images) | 10 GB storage, no egress fees |
| D1 (metadata) | 5 GB |
| Workers (API) | 100k requests/day |

Cloudflare asks for a payment method before enabling R2. Usage within the free tier costs $0.

## Layout
```
worker/   Cloudflare Worker (TypeScript): auth, presign, metadata, export API
device/   collector.py (offline-safe upload queue) + example_yolo_loop.py
tools/    export_raw.py (raw / YOLO / COCO export) + bbox.py conversions + tests
```

## 1. Deploy the Worker (one time)
See [DEPLOYMENT.md](DEPLOYMENT.md) for the full reference (verification, quotas, operations, key rotation, teardown).

```bash
cd worker
cp wrangler.toml.example wrangler.toml       # wrangler.toml is gitignored (holds your ids)
npm install
npx wrangler login
npx wrangler r2 bucket create ml-data-bucket
npx wrangler d1 create ml-data-bucket        # paste the printed database_id into wrangler.toml
npm run db:init:remote
npx wrangler secret put API_KEY              # e.g. output of: openssl rand -hex 32
npm run deploy                               # prints https://ml-data-bucket.<you>.workers.dev
```

### Optional: direct-to-R2 uploads (recommended)
Without this, images are uploaded *through* the Worker (`upload_mode: "direct"`, max 25 MB). That works, but every image costs a Worker request. To make devices upload straight to R2 with presigned URLs:

1. In the dashboard, go to R2 → Manage API tokens → create a token with **Object Read & Write** on `ml-data-bucket`.
2. Set `R2_ACCOUNT_ID` in `wrangler.toml` to your account id.
3. Run `npx wrangler secret put R2_ACCESS_KEY_ID` and `npx wrangler secret put R2_SECRET_ACCESS_KEY`, then `npm run deploy`.

Devices then need outbound access to `*.r2.cloudflarestorage.com`.

## 2. Collect on the device
```bash
pip install -r device/requirements.txt
export BASE_URL=https://ml-data-bucket.<you>.workers.dev API_KEY=<key>
```
```python
from collector import Collector, should_keep

col = Collector(BASE_URL, API_KEY, project="plastic-bottle", device_id="pi-01", model_version="v3")

col.submit("frame.jpg")                                    # image only, no metadata
col.submit(jpeg_bytes,                                     # with optional detections
           detections=[{"class": "bottle", "class_id": 0, "confidence": 0.41,
                        "bbox": [0.51, 0.40, 0.12, 0.30]}],
           bbox_format="yolo", image_width=1280, image_height=720,
           tags={"reason": "low_conf"})
col.close()  # waits (bounded) for the queue to drain
```
- `project` is any slug matching `^[a-z0-9][a-z0-9-]{1,62}$`. There's no registration step; check `GET /v1/projects` to spot typos.
- `submit()` never blocks on the network. Images are written to `~/.cache/ml-data-bucket` and uploaded by a background thread with retry and backoff. Anything still queued is uploaded on the next start, even after a reboot.
- Images are uploaded byte-for-byte. Supported types are JPEG, PNG and WebP.
- `should_keep(detections, threshold, sample_rate, known_classes)` is an optional filter that returns `low_conf`, `new_class`, `random` or `None`.
- See `device/example_yolo_loop.py` for an Ultralytics camera loop. Run it with `--synthetic N` to test without a camera.

### Metadata (everything optional)
| Field | Notes |
|---|---|
| `detections[]` | each item may have any of `class`, `class_id`, `confidence` (0–1), `bbox` |
| `bbox_format` | `yolo` = `[cx, cy, w, h]` normalized (default) · `coco` = `[x, y, w, h]` px · `xyxy` = `[x1, y1, x2, y2]` px |
| `image_width`, `image_height` | lets the exporter convert pixel ↔ normalized boxes (otherwise it reads the image header) |
| `device_id`, `model_version`, `captured_at`, `tags` | free-form bookkeeping (`tags` is a JSON object ≤ 8 KB) |

Boxes are stored exactly as sent. Conversion only happens at export time.

## 3. Export a project
```bash
pip install -r tools/requirements.txt
cd tools
python export_raw.py --project plastic-bottle                       # raw: images + metadata.jsonl
python export_raw.py --project plastic-bottle --labels yolo --min-conf 0.25
python export_raw.py --project people-counting --labels coco --since 2026-10-01
python export_raw.py --project plastic-bottle --delete-after         # free bucket space after export
```
```
exports/<project>/
  images/<yyyy-mm-dd>/<id>.<ext>   original bytes
  metadata.jsonl                   one sample per line; detections as sent (null if none)
  labels/<yyyy-mm-dd>/<id>.txt     --labels yolo  (empty file = no objects)
  classes.txt                      --labels yolo  (line N = class_id N)
  annotations.json                 --labels coco  (confidence kept as "score")
```
- Exports are incremental: metadata is merged by id and existing images are skipped.
- **Class ids:** a `class_id` sent by the device is kept. Classes sent only by name get the lowest free ids in alphabetical order, consistently across YOLO and COCO.
- `--min-conf` only filters the label files. `metadata.jsonl` always keeps every detection.
- `--delete-after` removes exported images from R2. The metadata rows stay, marked `archived`.
- Downloads go through the Worker by default. Set `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` to download from R2 directly with boto3.

## API reference
All routes need `Authorization: Bearer <API_KEY>`.

| Method | Path | |
|---|---|---|
| POST | `/v1/{project}/presign` | body: metadata (optional, may include a client `id` UUID for idempotent retries) → `{id, image_key, upload_mode, upload_url, upload_headers}` |
| PUT | `/v1/{project}/upload/{id}` | direct-mode upload (when presigning is not configured) |
| POST | `/v1/{project}/confirm/{id}` | marks the sample uploaded once the object exists |
| GET | `/v1/projects` | slugs with total / uploaded / archived counts |
| GET | `/v1/{project}/samples?since=&cursor=&limit=` | paginated uploaded samples (`uploaded=all` to include pending) |
| GET / DELETE | `/v1/{project}/samples/{id}/image` | download / archive one image |

## Development
```bash
cd worker
cp -n wrangler.toml.example wrangler.toml    # placeholder ids are fine for local dev
echo 'API_KEY=dev-secret-key' > .dev.vars
npm run db:init:local
npm run dev                         # http://127.0.0.1:8787, local R2 + D1
npm run typecheck
cd ../tools && pytest -q            # bbox conversion tests
```
