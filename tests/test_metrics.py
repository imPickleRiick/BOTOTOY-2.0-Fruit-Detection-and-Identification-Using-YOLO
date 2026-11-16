"""Tests for detection metrics and shared detection helpers."""
import numpy as np
import pytest

from src.utils.detections import Detection, box_iou, count_by_class, non_max_suppression
from src.utils.metrics import ImageEval, count_errors, evaluate, iou_matrix, localization_recall

CLASSES = ["Apple", "Banana"]


def img(preds, gts):
    """preds: [(cls, score, box)], gts: [(cls, box)]"""
    return ImageEval(np.array([p[0] for p in preds], int), np.array([p[1] for p in preds], float),
                     np.array([p[2] for p in preds], float).reshape(-1, 4),
                     np.array([g[0] for g in gts], int), np.array([g[1] for g in gts], float).reshape(-1, 4))


def test_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou_matrix(np.array([[0, 0, 10, 10]], float), np.array([[5, 0, 15, 10]], float))[0, 0] == pytest.approx(1 / 3)


# Ultralytics' 101-point AP formula tops out at 0.995 (the curve is closed with precision 0 at
# recall 1.0, losing half of the last 0.01 recall step). We use the identical formula on purpose.
MAX_AP = 0.995


def test_perfect_predictions_give_max_ap():
    data = [img([(0, 0.9, (0, 0, 10, 10)), (1, 0.8, (20, 20, 40, 40))], [(0, (0, 0, 10, 10)), (1, (20, 20, 40, 40))])]
    m = evaluate(data, CLASSES)
    assert m["mean"]["mAP50"] == pytest.approx(MAX_AP)
    assert m["mean"]["mAP50-95"] == pytest.approx(MAX_AP)
    assert m["per_class"]["Apple"]["precision@0.5"] == 1.0


def test_wrong_class_is_a_false_positive_but_localized():
    data = [img([(1, 0.9, (0, 0, 10, 10))], [(0, (0, 0, 10, 10))])]
    m = evaluate(data, CLASSES)
    assert m["per_class"]["Apple"]["recall@0.5"] == 0.0
    assert m["per_class"]["Banana"]["precision@0.5"] == 0.0
    assert localization_recall(data) == 1.0


def test_duplicate_detection_counts_once():
    data = [img([(0, 0.9, (0, 0, 10, 10)), (0, 0.8, (0, 0, 10, 10))], [(0, (0, 0, 10, 10))])]
    a = evaluate(data, CLASSES)["per_class"]["Apple"]
    assert a["tp@0.5"] == 1 and a["precision@0.5"] == 0.5 and a["recall@0.5"] == 1.0
    assert a["AP50"] == pytest.approx(MAX_AP)  # the extra FP is ranked below the TP


def test_count_errors():
    data = [img([(0, 0.9, (0, 0, 1, 1))], [(0, (0, 0, 1, 1)), (0, (2, 2, 3, 3))]), img([], [])]
    c = count_errors(data)
    assert c["mean_absolute_count_error"] == 0.5 and c["mean_signed_count_error"] == -0.5 and c["exact_count_rate"] == 0.5


def test_nms_and_counts():
    dets = [Detection("Apple", 0.9, (0, 0, 10, 10)), Detection("Banana", 0.5, (1, 1, 10, 10)),
            Detection("Apple", 0.7, (50, 50, 60, 60))]
    kept = non_max_suppression(dets, 0.5)
    assert [d.confidence for d in kept] == [0.9, 0.7]
    assert count_by_class(kept, ["Apple", "Banana", "Grape"]) == {"Apple": 2, "Banana": 0, "Grape": 0}
    assert len(non_max_suppression(dets, 0.5, class_agnostic=False)) == 3
