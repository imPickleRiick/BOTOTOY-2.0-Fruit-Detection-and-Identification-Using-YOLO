"""Tests for the dataset validator and shared image helpers (synthetic data, no dataset needed)."""
import numpy as np
import pytest

from src.data.dataset_validator import find_near_duplicate_pairs, parse_label_file, source_id_from_name
from src.utils.image_utils import dhash, yolo_to_xyxy

NC = 6


def _labels(tmp_path, text):
    path = tmp_path / "img.txt"
    path.write_text(text, encoding="utf-8")
    return parse_label_file(path, NC)


def test_valid_labels_are_parsed(tmp_path):
    boxes, issues = _labels(tmp_path, "0 0.5 0.5 0.2 0.2\n5 0.1 0.1 0.1 0.1\n")
    assert issues == []
    assert boxes == [(0, 0.5, 0.5, 0.2, 0.2), (5, 0.1, 0.1, 0.1, 0.1)]


@pytest.mark.parametrize("line, code", [
    ("6 0.5 0.5 0.2 0.2", "invalid_class"),         # class id must be 0..5
    ("-1 0.5 0.5 0.2 0.2", "invalid_class"),
    ("0 0.5 0.5 0.2", "malformed_line"),            # only 4 values
    ("0 0.5 0.5 0.2 0.2 0.3 0.3", "malformed_line"),  # polygon, not a box
    ("a 0.5 0.5 0.2 0.2", "malformed_line"),
    ("1.5 0.5 0.5 0.2 0.2", "malformed_line"),       # non-integer class
    ("0 1.5 0.5 0.2 0.2", "out_of_range_coords"),    # not normalized
    ("0 0.5 0.5 0 0.2", "non_positive_size"),
])
def test_invalid_lines_are_flagged(tmp_path, line, code):
    boxes, issues = _labels(tmp_path, line + "\n")
    assert boxes == []
    assert issues == [code]


def test_box_past_border_is_a_warning_but_kept(tmp_path):
    boxes, issues = _labels(tmp_path, "0 0.95 0.5 0.2 0.2\n")
    assert issues == ["box_exceeds_image"]
    assert len(boxes) == 1


def test_duplicate_and_empty(tmp_path):
    boxes, issues = _labels(tmp_path, "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n")
    assert len(boxes) == 1 and issues == ["duplicate_box"]
    boxes, issues = _labels(tmp_path, "\n  \n")
    assert boxes == [] and issues == ["empty_label"]


def test_source_id_from_roboflow_name():
    assert source_id_from_name("000aee0af66d4237_jpg.rf.6b99256606e5b7a1ede8b833a87196a7") == "000aee0af66d4237"
    assert source_id_from_name("my_photo") == "my_photo"


def test_dhash_detects_near_duplicates():
    rng = np.random.default_rng(0)
    image = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    brighter = np.clip(image.astype(int) + 10, 0, 255).astype(np.uint8)  # same picture, brighter
    other = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    h = np.array([dhash(image), dhash(brighter), dhash(other)], dtype=np.uint64)
    pairs = find_near_duplicate_pairs(h, max_distance=4)
    assert (0, 1) in {(i, j) for i, j, _ in pairs}
    assert all(2 not in (i, j) for i, j, _ in pairs)


def test_yolo_to_xyxy_converts_and_clips():
    assert yolo_to_xyxy(0.5, 0.5, 0.5, 0.5, 640, 640) == (160, 160, 480, 480)
    assert yolo_to_xyxy(0.0, 0.0, 0.2, 0.2, 100, 100) == (0, 0, 10, 10)  # clipped at 0
