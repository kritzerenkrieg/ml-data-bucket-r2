"""Bounding-box conversion between the formats the collector accepts.

yolo: [cx, cy, w, h] normalized to 0-1
coco: [x_min, y_min, w, h] in pixels
xyxy: [x1, y1, x2, y2] in pixels
"""

from __future__ import annotations

FORMATS = ("yolo", "coco", "xyxy")
Box = tuple[float, float, float, float]


def needs_size(src: str, dst: str) -> bool:
    """Converting between normalized and pixel boxes needs the image size."""
    return src != dst and "yolo" in (src, dst)


def _to_xyxy(b: Box, fmt: str, width: float | None, height: float | None) -> Box:
    if fmt == "xyxy":
        return b
    if fmt == "coco":
        x, y, w, h = b
        return (x, y, x + w, y + h)
    if fmt == "yolo":
        cx, cy, w, h = b
        return (
            (cx - w / 2) * width,
            (cy - h / 2) * height,
            (cx + w / 2) * width,
            (cy + h / 2) * height,
        )
    raise ValueError(f"unknown bbox format: {fmt}")


def _from_xyxy(b: Box, fmt: str, width: float | None, height: float | None) -> Box:
    x1, y1, x2, y2 = b
    if fmt == "xyxy":
        return b
    if fmt == "coco":
        return (x1, y1, x2 - x1, y2 - y1)
    if fmt == "yolo":
        clip = lambda v: min(max(v, 0.0), 1.0)
        return (
            clip((x1 + x2) / 2 / width),
            clip((y1 + y2) / 2 / height),
            clip((x2 - x1) / width),
            clip((y2 - y1) / height),
        )
    raise ValueError(f"unknown bbox format: {fmt}")


def convert(
    bbox: list[float] | Box,
    src: str,
    dst: str,
    width: float | None = None,
    height: float | None = None,
) -> Box:
    for fmt in (src, dst):
        if fmt not in FORMATS:
            raise ValueError(f"unknown bbox format: {fmt}")
    b = tuple(float(v) for v in bbox)
    if len(b) != 4:
        raise ValueError("bbox must have 4 values")
    if src == dst:
        return b  # type: ignore[return-value]
    if needs_size(src, dst) and not (width and height):
        raise ValueError(f"image width/height required to convert {src} -> {dst}")
    return _from_xyxy(_to_xyxy(b, src, width, height), dst, width, height)  # type: ignore[arg-type]
