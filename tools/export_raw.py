"""Export one project's collected data as raw files (optionally with YOLO/COCO pre-labels).

    python export_raw.py --project plastic-bottle
    python export_raw.py --project plastic-bottle --since 2026-10-01 --labels yolo --min-conf 0.25
    python export_raw.py --project people-counting --labels coco --delete-after

Output (exports/<project>/):
    images/<yyyy-mm-dd>/<id>.<ext>   original bytes, never re-encoded
    metadata.jsonl                   one sample per line, detections exactly as sent (may be null)
    labels/<yyyy-mm-dd>/<id>.txt     --labels yolo
    classes.txt                      --labels yolo (line N = name of class_id N)
    annotations.json                 --labels coco

Runs are incremental: metadata is merged by id and already-downloaded images are skipped.
Images are fetched from R2 directly when R2_ACCOUNT_ID/R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY
are set, otherwise through the Worker.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import requests

from bbox import convert, needs_size

log = logging.getLogger("export")


class Source:
    def __init__(self, base_url: str, api_key: str, project: str):
        self.base_url = base_url.rstrip("/")
        self.project = project
        self.http = requests.Session()
        self.http.headers["authorization"] = f"Bearer {api_key}"
        self.s3 = None
        account = os.environ.get("R2_ACCOUNT_ID")
        if account and os.environ.get("R2_ACCESS_KEY_ID") and os.environ.get("R2_SECRET_ACCESS_KEY"):
            import boto3

            self.s3 = boto3.client(
                "s3",
                endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
                aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                region_name="auto",
            )
            self.bucket = os.environ.get("R2_BUCKET_NAME", "ml-data-bucket")

    def _check(self, resp: requests.Response) -> requests.Response:
        if resp.status_code >= 400:
            raise RuntimeError(f"{resp.request.method} {resp.url} -> {resp.status_code}: {resp.text[:300]}")
        return resp

    def samples(self, since: str | None):
        params = {"limit": 1000}
        if since:
            params["since"] = since
        while True:
            data = self._check(
                self.http.get(f"{self.base_url}/v1/{self.project}/samples", params=params, timeout=60)
            ).json()
            yield from data["samples"]
            if not data["next_cursor"]:
                return
            params["cursor"] = data["next_cursor"]

    def download(self, row: dict, dest: Path) -> None:
        tmp = dest.with_suffix(dest.suffix + ".part")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if self.s3:
            self.s3.download_file(self.bucket, row["image_key"], str(tmp))
        else:
            url = f"{self.base_url}/v1/{self.project}/samples/{row['id']}/image"
            with self._check(self.http.get(url, stream=True, timeout=120)) as resp, tmp.open("wb") as fh:
                for chunk in resp.iter_content(1 << 16):
                    fh.write(chunk)
        tmp.replace(dest)

    def archive(self, row: dict) -> None:
        url = f"{self.base_url}/v1/{self.project}/samples/{row['id']}/image"
        self._check(self.http.delete(url, timeout=60))


def local_file(row: dict, project: str) -> str:
    """images/<yyyy-mm-dd>/<id>.<ext>, derived from the R2 key '<project>/<date>/<id>.<ext>'."""
    key = row["image_key"]
    prefix = f"{project}/"
    return "images/" + (key[len(prefix):] if key.startswith(prefix) else key)


def image_size(row: dict, out: Path) -> tuple[int, int] | None:
    if row.get("image_width") and row.get("image_height"):
        return row["image_width"], row["image_height"]
    path = out / row["file"]
    if not path.exists():
        return None
    try:
        from PIL import Image
    except ImportError:
        log.warning("pillow not installed; cannot read size of %s (pip install pillow)", path)
        return None
    with Image.open(path) as im:
        return im.size


def build_class_map(rows: list[dict]) -> dict[int, str]:
    """class_id -> name. Ids sent by the device win; unnamed ids become 'class_<id>';
    names sent without an id get the lowest free ids in alphabetical order."""
    name_to_id: dict[str, int] = {}
    id_to_name: dict[int, str] = {}
    unassigned: set[str] = set()
    for row in rows:
        for d in row.get("detections") or []:
            name, cid = d.get("class"), d.get("class_id")
            if cid is not None and name:
                if name in name_to_id and name_to_id[name] != cid:
                    log.warning("class %r sent with ids %d and %d; using %d", name, name_to_id[name], cid, name_to_id[name])
                    continue
                name_to_id.setdefault(name, cid)
                id_to_name.setdefault(cid, name)
            elif cid is not None:
                id_to_name.setdefault(cid, f"class_{cid}")
            elif name:
                unassigned.add(name)
    next_id = 0
    for name in sorted(unassigned - name_to_id.keys()):
        while next_id in id_to_name:
            next_id += 1
        id_to_name[next_id] = name
        name_to_id[name] = next_id
    return id_to_name


def labelled_boxes(row, out, name_to_id, dst_fmt, min_conf, stats):
    """Yields (class_id, box_in_dst_fmt, confidence) for one sample; also returns the image size."""
    size = image_size(row, out)
    src_fmt = row.get("bbox_format") or "yolo"
    boxes = []
    for d in row.get("detections") or []:
        if d.get("bbox") is None:
            stats["no_bbox"] += 1
            continue
        if min_conf is not None and d.get("confidence") is not None and d["confidence"] < min_conf:
            stats["below_min_conf"] += 1
            continue
        cid = d.get("class_id")
        if cid is None and d.get("class"):
            cid = name_to_id.get(d["class"])
        if cid is None:
            stats["no_class"] += 1
            continue
        if needs_size(src_fmt, dst_fmt) and size is None:
            stats["no_size"] += 1
            continue
        w, h = size if size else (None, None)
        boxes.append((cid, convert(d["bbox"], src_fmt, dst_fmt, w, h), d.get("confidence")))
    return size, boxes


def write_yolo(rows: list[dict], out: Path, min_conf: float | None) -> dict:
    id_to_name = build_class_map(rows)
    name_to_id = {v: k for k, v in id_to_name.items()}
    stats = {"images": 0, "boxes": 0, "no_bbox": 0, "no_class": 0, "no_size": 0, "below_min_conf": 0}
    for row in rows:
        _, boxes = labelled_boxes(row, out, name_to_id, "yolo", min_conf, stats)
        label = out / "labels" / Path(row["file"]).relative_to("images").with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        # Empty file = background image (no objects), which YOLO trainers expect.
        label.write_text("".join(f"{cid} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n" for cid, b, _ in boxes))
        stats["images"] += 1
        stats["boxes"] += len(boxes)
    n = max(id_to_name, default=-1) + 1
    (out / "classes.txt").write_text("".join(f"{id_to_name.get(i, f'class_{i}')}\n" for i in range(n)))
    return stats


def write_coco(rows: list[dict], out: Path, min_conf: float | None) -> dict:
    id_to_name = build_class_map(rows)
    name_to_id = {v: k for k, v in id_to_name.items()}
    stats = {"images": 0, "boxes": 0, "no_bbox": 0, "no_class": 0, "no_size": 0, "below_min_conf": 0}
    images, annotations = [], []
    for image_id, row in enumerate(rows, start=1):
        size, boxes = labelled_boxes(row, out, name_to_id, "coco", min_conf, stats)
        if size is None:
            stats["images_skipped_no_size"] = stats.get("images_skipped_no_size", 0) + 1
            continue
        images.append({"id": image_id, "file_name": row["file"], "width": size[0], "height": size[1],
                       "sample_id": row["id"]})
        for cid, b, conf in boxes:
            ann = {"id": len(annotations) + 1, "image_id": image_id, "category_id": cid,
                   "bbox": [round(v, 2) for v in b], "area": round(b[2] * b[3], 2), "iscrowd": 0}
            if conf is not None:
                ann["score"] = conf
            annotations.append(ann)
        stats["images"] += 1
    stats["boxes"] = len(annotations)
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for i, n in sorted(id_to_name.items())],
    }
    (out / "annotations.json").write_text(json.dumps(coco, indent=1))
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", required=True)
    p.add_argument("--base-url", default=os.environ.get("BASE_URL"))
    p.add_argument("--api-key", default=os.environ.get("API_KEY"))
    p.add_argument("--since", help="only fetch samples received on/after this date (YYYY-MM-DD or ISO time)")
    p.add_argument("--out", default="exports")
    p.add_argument("--labels", choices=["yolo", "coco"], help="also write pre-labels from the stored detections")
    p.add_argument("--min-conf", type=float, help="drop boxes below this confidence from label files only")
    p.add_argument("--delete-after", action="store_true",
                   help="delete exported images from R2 (metadata stays) to free bucket space")
    args = p.parse_args()
    if not args.base_url or not args.api_key:
        p.error("--base-url/--api-key (or BASE_URL/API_KEY env vars) are required")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out = Path(args.out) / args.project
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.jsonl"
    rows: dict[str, dict] = {}
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r

    src = Source(args.base_url, args.api_key, args.project)
    fetched = downloaded = missing = 0
    for row in src.samples(args.since):
        row["file"] = local_file(row, args.project)
        rows[row["id"]] = row
        fetched += 1
        dest = out / row["file"]
        if dest.exists():
            continue
        if row.get("archived"):
            log.warning("%s was deleted from the bucket by an earlier export and is not in %s", row["id"], out)
            missing += 1
            continue
        try:
            src.download(row, dest)
            downloaded += 1
        except Exception as exc:  # noqa: BLE001 - keep exporting the rest
            log.error("download of %s failed: %s", row["id"], exc)
            missing += 1

    ordered = sorted(rows.values(), key=lambda r: (r["received_at"], r["id"]))
    tmp = meta_path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
    tmp.replace(meta_path)
    log.info("%s: fetched %d rows, downloaded %d images, %d missing, %d rows total in %s",
             args.project, fetched, downloaded, missing, len(ordered), meta_path)

    present = [r for r in ordered if (out / r["file"]).exists()]
    if args.labels == "yolo":
        log.info("yolo labels: %s", write_yolo(present, out, args.min_conf))
    elif args.labels == "coco":
        log.info("coco annotations: %s", write_coco(present, out, args.min_conf))

    if args.delete_after:
        archived = 0
        for r in present:
            if not r.get("archived"):
                src.archive(r)
                r["archived"] = True
                archived += 1
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
        tmp.replace(meta_path)
        log.info("deleted %d exported images from the bucket", archived)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
