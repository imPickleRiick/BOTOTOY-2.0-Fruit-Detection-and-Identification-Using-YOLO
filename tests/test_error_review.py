"""Tests for the automatic error categories of the YOLO error review."""
import numpy as np

from src.evaluation.error_review import categorize
from src.utils.metrics import ImageEval

AREA = 640.0 * 640.0


def img(preds, gts):
    """preds: [(cls, score, box)], gts: [(cls, box)]"""
    return ImageEval(np.array([p[0] for p in preds], int), np.array([p[1] for p in preds], float),
                     np.array([p[2] for p in preds], float).reshape(-1, 4),
                     np.array([g[0] for g in gts], int), np.array([g[1] for g in gts], float).reshape(-1, 4))


def test_correct_detection_is_not_an_error():
    fps, fns = categorize(img([(0, 0.9, (0, 0, 100, 100))], [(0, (0, 0, 100, 100))]), AREA)
    assert fps == [] and fns == []


def test_wrong_class_on_both_sides():
    fps, fns = categorize(img([(1, 0.9, (0, 0, 100, 100))], [(0, (0, 0, 100, 100))]), AREA)
    assert fps == [(0, "wrong_class")]
    assert fns == [(0, "wrong_class")]


def test_group_box_and_covered_fruits():
    # One prediction around three labeled apples: an FP group box, three misses covered by it.
    gts = [(0, (0, 0, 100, 100)), (0, (100, 0, 200, 100)), (0, (200, 0, 300, 100))]
    fps, fns = categorize(img([(0, 0.8, (0, 0, 300, 100))], gts), AREA)
    assert fps == [(0, "group_box")]
    assert sorted(c for _, c in fns) == ["covered_by_group"] * 3


def test_single_fruits_inside_a_group_label():
    # Ground truth labels the whole pile; the model finds two single fruits inside it.
    preds = [(0, 0.8, (0, 0, 100, 100)), (0, 0.7, (150, 0, 250, 100))]
    fps, fns = categorize(img(preds, [(0, (0, 0, 300, 100))]), AREA)
    assert [c for _, c in fps] == ["inside_group_label"] * 2
    assert fns == [(0, "poor_box")]  # the group box is overlapped (IoU 0.33), just not matched


def test_background_poor_box_duplicate_small():
    gts = [(0, (0, 0, 100, 100)), (1, (400, 400, 420, 420))]  # second fruit < 1% of the image
    preds = [(0, 0.9, (0, 0, 100, 100)),      # correct
             (0, 0.8, (5, 5, 100, 100)),      # duplicate of the same apple
             (0, 0.7, (60, 0, 200, 100)),     # poor box (IoU 0.29 with the apple)
             (1, 0.6, (500, 500, 600, 600))]  # background
    fps, fns = categorize(img(preds, gts), AREA)
    assert dict(fps) == {1: "duplicate", 2: "poor_box", 3: "background"}
    assert fns == [(1, "small")]


def test_empty_image():
    assert categorize(img([], []), AREA) == ([], [])
    fps, _ = categorize(img([(0, 0.5, (0, 0, 50, 50))], []), AREA)
    assert fps == [(0, "background")]
