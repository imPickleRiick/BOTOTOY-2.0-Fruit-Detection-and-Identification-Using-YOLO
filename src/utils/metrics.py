"""Object-detection metrics shared by the baseline and YOLO evaluations.

Definitions
-----------
- A prediction is a TRUE POSITIVE (TP) at IoU threshold t if it overlaps a not-yet-matched
  ground-truth box of the SAME class with IoU >= t. Otherwise it is a FALSE POSITIVE (FP).
  Unmatched ground-truth boxes are FALSE NEGATIVES (FN).
- Precision = TP / (TP + FP)   ("of what we detected, how much was right")
- Recall    = TP / (TP + FN)   ("of the real fruits, how many did we find")
- AP (Average Precision) = area under the precision-recall curve obtained by ranking
  predictions by confidence. We use the same 101-point interpolation as Ultralytics,
  so baseline and YOLO numbers are computed the same way.
- mAP@0.50 = mean AP over classes at IoU 0.50; mAP@0.50:0.95 = mean over IoU 0.50..0.95.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)


@dataclass
class ImageEval:
    """Predictions and ground truth for one image. Boxes are (x1, y1, x2, y2) in pixels."""
    pred_classes: np.ndarray   # (P,)
    pred_scores: np.ndarray    # (P,)
    pred_boxes: np.ndarray     # (P, 4)
    gt_classes: np.ndarray     # (G,)
    gt_boxes: np.ndarray       # (G, 4)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and every box in b (M,4) -> (N, M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match_predictions(img: ImageEval, iou_thresholds=IOU_THRESHOLDS, class_aware: bool = True) -> np.ndarray:
    """Greedy matching (highest score first). Returns a (P, T) boolean TP matrix."""
    n_pred = len(img.pred_scores)
    tp = np.zeros((n_pred, len(iou_thresholds)), dtype=bool)
    if n_pred == 0 or len(img.gt_classes) == 0:
        return tp
    ious = iou_matrix(img.pred_boxes, img.gt_boxes)
    if class_aware:
        ious = ious * (img.pred_classes[:, None] == img.gt_classes[None, :])
    order = np.argsort(-img.pred_scores, kind="stable")
    for t, thr in enumerate(iou_thresholds):
        taken = np.zeros(len(img.gt_classes), dtype=bool)
        for p in order:
            candidates = np.where(~taken & (ious[p] >= thr))[0]
            if len(candidates):
                g = candidates[np.argmax(ious[p, candidates])]
                taken[g] = True
                tp[p, t] = True
    return tp


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP (same method as Ultralytics/COCO)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))  # precision envelope
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def evaluate(images: list[ImageEval], classes: list[str]) -> dict:
    """Per-class AP@0.5, AP@0.5:0.95, precision, recall and F1 plus their means."""
    tps, scores, pclasses, gt_counts = [], [], [], np.zeros(len(classes), dtype=int)
    for img in images:
        tps.append(match_predictions(img))
        scores.append(img.pred_scores)
        pclasses.append(img.pred_classes)
        gt_counts += np.bincount(img.gt_classes.astype(int), minlength=len(classes))
    tp = np.concatenate(tps) if tps else np.zeros((0, len(IOU_THRESHOLDS)), bool)
    sc = np.concatenate(scores) if scores else np.zeros(0)
    pc = np.concatenate(pclasses) if pclasses else np.zeros(0)

    per_class = {}
    for c, name in enumerate(classes):
        sel = pc == c
        order = np.argsort(-sc[sel], kind="stable")
        tp_c = tp[sel][order]
        n_gt = int(gt_counts[c])
        n_pred = int(sel.sum())
        aps = []
        for t in range(len(IOU_THRESHOLDS)):
            cum_tp = np.cumsum(tp_c[:, t])
            cum_fp = np.cumsum(~tp_c[:, t])
            recall = cum_tp / max(n_gt, 1)
            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
            aps.append(average_precision(recall, precision) if n_gt else 0.0)
        tp50 = int(tp_c[:, 0].sum())
        p = tp50 / n_pred if n_pred else 0.0
        r = tp50 / n_gt if n_gt else 0.0
        per_class[name] = {
            "ground_truth": n_gt, "predictions": n_pred, "tp@0.5": tp50,
            "precision@0.5": round(p, 4), "recall@0.5": round(r, 4),
            "f1@0.5": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "AP50": round(aps[0], 4), "AP50-95": round(float(np.mean(aps)), 4),
        }
    present = [c for c in classes if per_class[c]["ground_truth"] > 0]
    mean = lambda key: round(float(np.mean([per_class[c][key] for c in present])), 4) if present else 0.0
    return {
        "per_class": per_class,
        "mean": {"precision@0.5": mean("precision@0.5"), "recall@0.5": mean("recall@0.5"),
                 "f1@0.5": mean("f1@0.5"), "mAP50": mean("AP50"), "mAP50-95": mean("AP50-95")},
    }


def localization_recall(images: list[ImageEval], iou_threshold: float = 0.5) -> float:
    """Class-agnostic recall: share of real fruits covered by ANY predicted box (label ignored)."""
    found = total = 0
    for img in images:
        total += len(img.gt_classes)
        tp = match_predictions(img, [iou_threshold], class_aware=False)
        found += int(tp.sum())
    return round(found / total, 4) if total else 0.0


def count_errors(images: list[ImageEval]) -> dict:
    """How far the predicted fruit count is from the true count, per image."""
    diff = np.array([len(i.pred_scores) - len(i.gt_classes) for i in images], dtype=float)
    return {"mean_absolute_count_error": round(float(np.abs(diff).mean()), 3) if len(diff) else 0.0,
            "mean_signed_count_error": round(float(diff.mean()), 3) if len(diff) else 0.0,
            "exact_count_rate": round(float((diff == 0).mean()), 4) if len(diff) else 0.0}
