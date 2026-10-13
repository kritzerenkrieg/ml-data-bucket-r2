"""Device-side collector: spools images + optional detections locally, uploads in the background.

    from collector import Collector
    col = Collector("https://ml-data-bucket.<you>.workers.dev", api_key, project="plastic-bottle")
    col.submit("frame.jpg")                                   # image only
    col.submit(jpeg_bytes, detections=[{"class": "bottle", "confidence": 0.41,
                                        "bbox": [0.5, 0.4, 0.1, 0.3]}],
               bbox_format="yolo", image_width=1280, image_height=720)
    col.close()   # waits for the queue to drain (bounded by timeout)

Uploads survive restarts and network loss: every submit() is written to a spool
directory + SQLite queue first, then a background thread does presign -> PUT -> confirm.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger("collector")

EXT_TO_TYPE = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
TYPE_TO_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
MAX_BACKOFF_SECONDS = 300
# Errors that will never succeed on retry; the item is parked as 'failed' instead.
PERMANENT_STATUSES = {400, 401, 403, 404, 413}


class UploadError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _sniff_content_type(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("unsupported image type (expected JPEG, PNG or WebP bytes)")


def should_keep(
    detections: list[dict] | None,
    threshold: float = 0.5,
    sample_rate: float = 0.01,
    known_classes: set[str] | None = None,
) -> str | None:
    """Common "is this frame worth uploading?" rules. Returns a reason string or None.

    - low_conf:  any detection below `threshold`
    - new_class: a class not yet in `known_classes` (the set is updated in place)
    - random:    unbiased control sample at `sample_rate`
    """
    detections = detections or []
    if any(d.get("confidence") is not None and d["confidence"] < threshold for d in detections):
        return "low_conf"
    if known_classes is not None:
        new = {d["class"] for d in detections if d.get("class")} - known_classes
        if new:
            known_classes.update(new)
            return "new_class"
    if random.random() < sample_rate:
        return "random"
    return None


class Collector:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        project: str,
        spool_dir: str | os.PathLike = "~/.cache/ml-data-bucket",
        device_id: str | None = None,
        model_version: str | None = None,
        timeout: float = 30.0,
        start: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.project = project
        self.device_id = device_id
        self.model_version = model_version
        self.timeout = timeout
        self.spool = Path(spool_dir).expanduser()
        (self.spool / "images").mkdir(parents=True, exist_ok=True)

        self._db = sqlite3.connect(self.spool / "queue.db", check_same_thread=False)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS queue (
                 id TEXT PRIMARY KEY,
                 project TEXT NOT NULL,
                 path TEXT NOT NULL,
                 meta TEXT NOT NULL,
                 state TEXT NOT NULL DEFAULT 'pending',   -- pending | failed
                 attempts INTEGER NOT NULL DEFAULT 0,
                 next_try REAL NOT NULL DEFAULT 0,
                 last_error TEXT,
                 created_at REAL NOT NULL)"""
        )
        # A fresh process usually means the device rebooted or reconnected: retry right away.
        self._db.execute("UPDATE queue SET next_try = 0 WHERE state = 'pending'")
        self._db.commit()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._http = requests.Session()
        self._thread: threading.Thread | None = None
        if start:
            self.start()

    # ---------- public API ----------

    def submit(
        self,
        image: bytes | str | os.PathLike,
        detections: list[dict] | None = None,
        bbox_format: str | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
        captured_at: datetime | str | None = None,
        tags: dict | None = None,
        project: str | None = None,
    ) -> str:
        """Queue an image for upload. Returns the sample id. Never blocks on the network."""
        if isinstance(image, (str, os.PathLike)):
            data = Path(image).read_bytes()
        else:
            data = bytes(image)
        content_type = _sniff_content_type(data)

        sample_id = str(uuid.uuid4())
        if captured_at is None:
            captured_at = datetime.now(timezone.utc)
        if isinstance(captured_at, datetime):
            captured_at = captured_at.isoformat()

        meta = {
            "id": sample_id,
            "content_type": content_type,
            "device_id": self.device_id,
            "model_version": self.model_version,
            "captured_at": captured_at,
            "image_width": image_width,
            "image_height": image_height,
            "bbox_format": bbox_format,
            "detections": detections,
            "tags": tags,
        }
        meta = {k: v for k, v in meta.items() if v is not None}

        path = self.spool / "images" / f"{sample_id}{TYPE_TO_EXT[content_type]}"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)  # atomic: the queue never points at a half-written file
        with self._lock:
            self._db.execute(
                "INSERT INTO queue (id, project, path, meta, created_at) VALUES (?, ?, ?, ?, ?)",
                (sample_id, project or self.project, str(path), json.dumps(meta), time.time()),
            )
            self._db.commit()
        self._wake.set()
        return sample_id

    def pending(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM queue WHERE state = 'pending'").fetchone()[0]

    def failed(self) -> list[tuple[str, str]]:
        with self._lock:
            return self._db.execute("SELECT id, last_error FROM queue WHERE state = 'failed'").fetchall()

    def retry_failed(self) -> None:
        with self._lock:
            self._db.execute("UPDATE queue SET state = 'pending', attempts = 0, next_try = 0 WHERE state = 'failed'")
            self._db.commit()
        self._wake.set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="collector-uploader", daemon=True)
        self._thread.start()

    def flush(self, timeout: float | None = None) -> bool:
        """Block until the queue is empty (or timeout). Returns True if drained."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.pending():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            self._wake.set()
            time.sleep(0.2)
        return True

    def close(self, timeout: float | None = 30.0) -> bool:
        drained = self.flush(timeout)
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)
        with self._lock:
            self._db.close()
        return drained

    # ---------- uploader ----------

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._next_due()
            if item is None:
                self._wake.wait(timeout=self._seconds_until_next())
                self._wake.clear()
                continue
            sample_id, project, path, meta_json, attempts = item
            try:
                self._upload(project, Path(path), json.loads(meta_json))
            except Exception as exc:  # noqa: BLE001 - every failure is recorded and retried/parked
                self._record_failure(sample_id, attempts, exc)
            else:
                self._record_success(sample_id, Path(path))

    def _next_due(self):
        with self._lock:
            return self._db.execute(
                """SELECT id, project, path, meta, attempts FROM queue
                   WHERE state = 'pending' AND next_try <= ? ORDER BY created_at LIMIT 1""",
                (time.time(),),
            ).fetchone()

    def _seconds_until_next(self) -> float:
        with self._lock:
            row = self._db.execute("SELECT MIN(next_try) FROM queue WHERE state = 'pending'").fetchone()
        if row[0] is None:
            return 60.0
        return max(0.05, min(60.0, row[0] - time.time()))

    def _api(self, method: str, path: str, **kwargs) -> requests.Response:
        headers = {"authorization": f"Bearer {self.api_key}", **kwargs.pop("headers", {})}
        resp = self._http.request(method, f"{self.base_url}{path}", headers=headers, timeout=self.timeout, **kwargs)
        if resp.status_code >= 400:
            try:
                msg = resp.json().get("error", resp.text)
            except ValueError:
                msg = resp.text
            raise UploadError(f"{method} {path} -> {resp.status_code}: {msg}", resp.status_code)
        return resp

    def _upload(self, project: str, path: Path, meta: dict) -> None:
        sample_id = meta["id"]
        try:
            ticket = self._api("POST", f"/v1/{project}/presign", json=meta).json()
        except UploadError as exc:
            # 409 "already uploaded" means an earlier attempt succeeded but we crashed before
            # recording it; confirming again is harmless and closes the loop.
            if exc.status != 409 or "already uploaded" not in str(exc):
                raise
            self._api("POST", f"/v1/{project}/confirm/{sample_id}")
            return

        headers = dict(ticket.get("upload_headers", {}))
        if ticket["upload_mode"] == "direct":
            headers["authorization"] = f"Bearer {self.api_key}"
        with path.open("rb") as fh:
            resp = self._http.put(ticket["upload_url"], data=fh, headers=headers, timeout=self.timeout)
        if resp.status_code >= 400:
            raise UploadError(f"PUT image -> {resp.status_code}: {resp.text[:200]}", resp.status_code)
        self._api("POST", f"/v1/{project}/confirm/{sample_id}")

    def _record_success(self, sample_id: str, path: Path) -> None:
        with self._lock:
            self._db.execute("DELETE FROM queue WHERE id = ?", (sample_id,))
            self._db.commit()
        path.unlink(missing_ok=True)
        log.debug("uploaded %s", sample_id)

    def _record_failure(self, sample_id: str, attempts: int, exc: Exception) -> None:
        attempts += 1
        status = getattr(exc, "status", None)
        permanent = status in PERMANENT_STATUSES
        delay = min(MAX_BACKOFF_SECONDS, 2 ** min(attempts, 10)) * random.uniform(0.8, 1.2)
        with self._lock:
            self._db.execute(
                "UPDATE queue SET attempts = ?, next_try = ?, last_error = ?, state = ? WHERE id = ?",
                (attempts, time.time() + delay, str(exc)[:500], "failed" if permanent else "pending", sample_id),
            )
            self._db.commit()
        if permanent:
            log.error("upload of %s failed permanently (kept in spool): %s", sample_id, exc)
        else:
            log.warning("upload of %s failed (attempt %d, retry in %.0fs): %s", sample_id, attempts, delay, exc)


def main() -> None:
    p = argparse.ArgumentParser(description="Queue images for upload and drain the queue.")
    p.add_argument("images", nargs="*", help="image files to submit")
    p.add_argument("--project", required=True)
    p.add_argument("--base-url", default=os.environ.get("BASE_URL"))
    p.add_argument("--api-key", default=os.environ.get("API_KEY"))
    p.add_argument("--detections", help="JSON file with a detections array applied to every image")
    p.add_argument("--bbox-format", choices=["yolo", "coco", "xyxy"])
    p.add_argument("--device-id")
    p.add_argument("--model-version")
    p.add_argument("--spool-dir", default="~/.cache/ml-data-bucket")
    p.add_argument("--retry-failed", action="store_true", help="re-queue items parked as failed")
    p.add_argument("--timeout", type=float, default=120, help="seconds to wait for the queue to drain")
    args = p.parse_args()
    if not args.base_url or not args.api_key:
        p.error("--base-url/--api-key (or BASE_URL/API_KEY env vars) are required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    detections = json.loads(Path(args.detections).read_text()) if args.detections else None
    col = Collector(args.base_url, args.api_key, args.project, spool_dir=args.spool_dir,
                    device_id=args.device_id, model_version=args.model_version)
    if args.retry_failed:
        col.retry_failed()
    for img in args.images:
        sid = col.submit(img, detections=detections, bbox_format=args.bbox_format)
        log.info("queued %s as %s", img, sid)
    drained = col.flush(timeout=args.timeout)
    failed = col.failed()
    col.close(timeout=0)
    for sid, err in failed:
        log.error("failed: %s: %s", sid, err)
    raise SystemExit(0 if drained and not failed else 1)


if __name__ == "__main__":
    main()
