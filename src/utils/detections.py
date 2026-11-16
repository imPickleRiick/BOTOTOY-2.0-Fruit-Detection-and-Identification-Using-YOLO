"""Common detection data type used by BOTH detectors (OpenCV baseline and YOLO11n).

Using one shared format means visualization, counting, evaluation and the user
interface work the same way no matter which detector produced the result.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from src.utils.image_utils import CLASS_COLORS_BGR, DEFAULT_COLOR_BGR, draw_box


@dataclass
class Detection:
    class_name: str
    confidence: float                     # 0..1 (for the baseline: a rule score, not a probability)
    box: tuple[int, int, int, int]        # x1, y1, x2, y2 in pixels of the ORIGINAL image
    class_id: int | None = None
    extra: dict = field(default_factory=dict)  # detector-specific details (e.g. features, rule)

    def as_dict(self) -> dict:
        return {"class_name": self.class_name, "class_id": self.class_id,
                "confidence": round(float(self.confidence), 4), "box": [int(v) for v in self.box]}


def box_iou(a: tuple | np.ndarray, b: tuple | np.ndarray) -> float:
    """Intersection over Union of two boxes (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(dets: list[Detection], iou_threshold: float, class_agnostic: bool = True) -> list[Detection]:
    """Greedy NMS: keep the highest-scoring box, drop boxes overlapping it by more than iou_threshold."""
    kept: list[Detection] = []
    for det in sorted(dets, key=lambda d: d.confidence, reverse=True):
        if all((not class_agnostic and k.class_name != det.class_name) or box_iou(k.box, det.box) <= iou_threshold
               for k in kept):
            kept.append(det)
    return kept


def count_by_class(dets: list[Detection], classes: list[str]) -> dict[str, int]:
    """Number of detections per class (every class listed, zero if absent)."""
    counts = Counter(d.class_name for d in dets)
    return {c: int(counts.get(c, 0)) for c in classes}


def draw_detections(image: np.ndarray, dets: list[Detection], show_confidence: bool = True,
                    thickness: int | None = None) -> np.ndarray:
    """Return a copy of the image with labeled boxes drawn on it."""
    out = image.copy()
    th = thickness or max(2, round(max(out.shape[:2]) / 400))
    for d in dets:
        label = f"{d.class_name} {d.confidence:.2f}" if show_confidence else d.class_name
        draw_box(out, d.box, label, CLASS_COLORS_BGR.get(d.class_name, DEFAULT_COLOR_BGR), th)
    return out
