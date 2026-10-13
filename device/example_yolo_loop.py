"""Example inference loop feeding the collector.

Real camera + Ultralytics YOLO:
    pip install ultralytics opencv-python
    python example_yolo_loop.py --project plastic-bottle --weights best.pt --source 0

No model / camera (generates synthetic frames, useful to test the pipeline):
    pip install pillow
    python example_yolo_loop.py --project plastic-bottle --synthetic 20
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import random

from collector import Collector, should_keep


def yolo_frames(weights: str, source: str):
    """Yields (jpeg_bytes, width, height, detections) from a real camera/video."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(weights)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model(frame, verbose=False)[0]
        h, w = frame.shape[:2]
        detections = [
            {
                "class": result.names[int(c)],
                "class_id": int(c),
                "confidence": round(float(s), 4),
                "bbox": [round(float(v), 6) for v in box],  # normalized cx, cy, w, h
            }
            for box, c, s in zip(result.boxes.xywhn.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
        ]
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        yield jpeg.tobytes(), w, h, detections
    cap.release()


def synthetic_frames(n: int):
    """Yields frames with a drawn rectangle and a matching fake detection."""
    from PIL import Image, ImageDraw

    w, h = 640, 480
    for i in range(n):
        img = Image.new("RGB", (w, h), (30, 30, 30))
        x1, y1 = random.randint(0, w - 200), random.randint(0, h - 200)
        x2, y2 = x1 + random.randint(40, 200), y1 + random.randint(40, 200)
        ImageDraw.Draw(img).rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        # Every 3rd frame has no detections at all (metadata is optional).
        detections = None if i % 3 == 2 else [{
            "class": "bottle",
            "class_id": 0,
            "confidence": round(random.uniform(0.2, 0.95), 3),
            "bbox": [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h],
        }]
        yield buf.getvalue(), w, h, detections


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:8787"))
    p.add_argument("--api-key", default=os.environ.get("API_KEY"))
    p.add_argument("--weights", default="yolo11n.pt")
    p.add_argument("--source", default="0", help="camera index or video path")
    p.add_argument("--synthetic", type=int, metavar="N", help="use N synthetic frames instead of a camera")
    p.add_argument("--keep-all", action="store_true", help="upload every frame (skip should_keep)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--spool-dir", default="~/.cache/ml-data-bucket")
    args = p.parse_args()
    if not args.api_key:
        p.error("--api-key or API_KEY env var is required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    col = Collector(args.base_url, args.api_key, args.project, spool_dir=args.spool_dir,
                    device_id="example", model_version="demo")
    known: set[str] = set()
    frames = synthetic_frames(args.synthetic) if args.synthetic else yolo_frames(args.weights, args.source)
    kept = 0
    for jpeg, w, h, detections in frames:
        reason = "all" if args.keep_all else should_keep(detections, args.threshold, known_classes=known)
        if reason is None:
            continue
        col.submit(jpeg, detections=detections, bbox_format="yolo" if detections else None,
                   image_width=w, image_height=h, tags={"reason": reason})
        kept += 1
    logging.info("queued %d frames, waiting for upload...", kept)
    drained = col.close(timeout=120)
    logging.info("done (queue drained: %s)", drained)


if __name__ == "__main__":
    main()
