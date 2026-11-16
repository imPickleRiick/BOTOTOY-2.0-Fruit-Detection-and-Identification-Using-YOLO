"""Turn detection results into things people can look at: tables, image bytes, labels.

Nothing here depends on Streamlit, so the same helpers serve the app, scripts and tests.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from src.baseline.pipeline import GROUP_COLORS
from src.utils.detections import Detection

# Friendly names for the timing keys produced by FruitDetectionEngine.run().
TIMING_LABELS = {
    "decode": "Decode image",
    "preprocess": "Preprocess",
    "inference": "Model inference",
    "postprocess": "Postprocess + NMS",
    "segmentation": "HSV masks + morphology",
    "contours_features_rules": "Contours, features, rules",
    "nms": "NMS",
    "detector_total": "Detector total",
    "end_to_end": "End-to-end",
}

# Titles and explanations for the OpenCV baseline's processing stages.
BASELINE_STAGES = {
    "1_preprocessed": ("1. Preprocessed", "Resized so the longest side is 640 px, then Gaussian-blurred to reduce noise."),
    "2_color_masks": ("2. HSV color masks", "Pixels whose hue/saturation/value fall in each fruit-color range."),
    "3_after_morphology": ("3. After morphology", "Opening removes specks; closing fills small holes inside fruits."),
    "4_contours": ("4. Contours", "Outlines of the connected regions that are large enough to be fruit."),
    "5_detections": ("5. Detections", "Regions accepted by the shape/color rules, after non-maximum suppression."),
}


def bgr_to_hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02X}{g:02X}{b:02X}"


COLOR_GROUP_LEGEND = {group: bgr_to_hex(c) for group, c in GROUP_COLORS.items()}


def to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def encode_image(image_bgr: np.ndarray, ext: str = ".jpg", quality: int = 92) -> bytes:
    """Encode an image for download (JPG by default)."""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, image_bgr, params)
    if not ok:
        raise ValueError(f"Could not encode image as {ext}")
    return buf.tobytes()


def fit_within(image: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale (never upscale) so the longest side is at most max_side; for previews."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def safe_stem(filename: str) -> str:
    """File name without extension, reduced to characters that are safe in any file system."""
    base = re.split(r"[\\/]", filename)[-1]         # drop any folder part
    stem, dot, ext = base.rpartition(".")
    if not (dot and stem and ext.isalnum() and len(ext) <= 5):  # no real extension
        stem = base
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem[:80] or "image"


def detection_rows(dets: list[Detection], include_rule: bool = False) -> list[dict]:
    """One row per detected fruit: class, confidence and box in pixels of the original image."""
    rows = []
    for i, d in enumerate(sorted(dets, key=lambda d: -d.confidence), 1):
        x1, y1, x2, y2 = d.box
        row = {"#": i, "Fruit": d.class_name, "Confidence": round(float(d.confidence), 3),
               "Box (x1, y1, x2, y2)": f"{x1}, {y1}, {x2}, {y2}", "Width": x2 - x1, "Height": y2 - y1}
        if include_rule:
            row["Color group"] = d.extra.get("group", "")
            row["Rule"] = d.extra.get("reason", "")
        rows.append(row)
    return rows


def count_rows(counts: dict[str, int]) -> list[dict]:
    return [{"Fruit": c, "Count": int(n)} for c, n in counts.items()]


def timing_rows(timing_ms: dict[str, float]) -> list[dict]:
    return [{"Step": TIMING_LABELS.get(k, k), "Time (ms)": round(float(v), 1)} for k, v in timing_ms.items()]


def region_rows(regions: list[dict]) -> list[dict]:
    """Every candidate region the baseline considered, with the rule decision (accepted or why rejected)."""
    rows = []
    for r in regions:
        rows.append({"Color group": r["group"], "Decision": r["class"] or "rejected", "Reason": r["reason"],
                     "Area %": round(100 * r["area_fraction"], 2), "Mean hue": round(r["mean_hue"], 1),
                     "Saturation": round(r["mean_saturation"], 1), "Elongation": round(r["elongation"], 2),
                     "Circularity": round(r["circularity"], 2), "Solidity": round(r["solidity"], 2),
                     "Extent": round(r["extent"], 2)})
    return rows
