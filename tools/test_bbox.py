import itertools

import pytest

from bbox import FORMATS, convert

W, H = 1280, 720
# The same box in every format: x 512..704, y 216..432 on a 1280x720 image.
SAME_BOX = {
    "xyxy": (512.0, 216.0, 704.0, 432.0),
    "coco": (512.0, 216.0, 192.0, 216.0),
    "yolo": (0.475, 0.45, 0.15, 0.3),
}


@pytest.mark.parametrize("src,dst", list(itertools.product(FORMATS, FORMATS)))
def test_known_box_converts_between_all_formats(src, dst):
    assert convert(SAME_BOX[src], src, dst, W, H) == pytest.approx(SAME_BOX[dst])


@pytest.mark.parametrize("src,dst", list(itertools.permutations(FORMATS, 2)))
def test_round_trip(src, dst):
    there = convert(SAME_BOX[src], src, dst, W, H)
    back = convert(there, dst, src, W, H)
    assert back == pytest.approx(SAME_BOX[src])


def test_pixel_formats_do_not_need_size():
    assert convert(SAME_BOX["coco"], "coco", "xyxy") == pytest.approx(SAME_BOX["xyxy"])


@pytest.mark.parametrize("src,dst", [("yolo", "coco"), ("xyxy", "yolo")])
def test_normalized_conversion_requires_size(src, dst):
    with pytest.raises(ValueError, match="width/height"):
        convert(SAME_BOX[src], src, dst)


def test_yolo_output_is_clipped_to_image():
    # Box hanging off the right edge.
    cx, cy, w, h = convert((1200, 0, 1400, 100), "xyxy", "yolo", W, H)
    assert 0 <= cx <= 1 and 0 <= w <= 1


def test_rejects_unknown_format():
    with pytest.raises(ValueError):
        convert((0, 0, 1, 1), "voc", "yolo", W, H)
