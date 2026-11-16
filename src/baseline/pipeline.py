"""OpenCV baseline detector: runs the full traditional computer-vision pipeline.

    image -> preprocess (resize, blur, BGR->HSV)
          -> one HSV mask per color group
          -> morphology (opening, closing)
          -> external contours -> filter by area -> (optional) split touching fruits
          -> features (hue, saturation, area, aspect ratio, extent, circularity, ...)
          -> rule-based classification
          -> bounding boxes + non-maximum suppression -> counts

Usage (from the project root):
    python -m src.baseline.pipeline path/to/image.jpg [more images or folders] --out outputs/baseline/demo
    python -m src.baseline.pipeline photo.jpg --stages      # also save the processing-stage panel
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.baseline.feature_extraction import extract_features
from src.baseline.fruit_classifier import classify_region, rule_score
from src.baseline.preprocessing import preprocess
from src.baseline.segmentation import build_color_masks, clean_mask, find_contours, split_touching_objects
from src.utils.detections import Detection, count_by_class, draw_detections, non_max_suppression
from src.utils.helpers import ensure_dir, list_images, load_config, save_json
from src.utils.image_utils import make_grid, read_image, write_image

# Display colors (BGR) for each color-group mask in the stage visualization.
GROUP_COLORS = {"red": (60, 60, 230), "warm": (0, 190, 255), "green": (80, 200, 80), "purple": (200, 80, 160)}


@dataclass
class BaselineResult:
    detections: list[Detection]
    counts: dict[str, int]
    total: int
    regions: list[dict] = field(default_factory=list)       # every candidate region + decision
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    timing_ms: dict[str, float] = field(default_factory=dict)


class BaselineDetector:
    """Traditional OpenCV fruit detector configured from config.yaml (section 'baseline')."""

    def __init__(self, config: dict | None = None):
        cfg = config or load_config()
        self.classes: list[str] = cfg["classes"]
        self.p: dict = cfg["baseline"]

    def detect(self, image: np.ndarray, keep_stages: bool = False) -> BaselineResult:
        p = self.p
        t0 = time.perf_counter()
        pre = preprocess(image, p["resize_max_side"], p["blur_kernel"])
        t1 = time.perf_counter()

        h, w = pre.hsv.shape[:2]
        img_area = float(h * w)
        raw_masks = build_color_masks(pre.hsv, p["color_ranges"])
        m = p["morphology"]
        clean_masks = {g: clean_mask(mk, m["open_kernel"], m["close_kernel"], m["open_iterations"], m["close_iterations"])
                       for g, mk in raw_masks.items()}
        t2 = time.perf_counter()

        candidates: list[Detection] = []
        regions: list[dict] = []
        min_area, max_area = p["min_area_fraction"] * img_area, p["max_area_fraction"] * img_area
        split = p["split_touching"]
        for group, mask in clean_masks.items():
            for contour in find_contours(mask, min_area, max_area):
                pieces = [contour]
                # Watershed only for groups whose fruits are round and often touch (e.g. oranges).
                if split["enabled"] and group in split["groups"]:
                    f0 = extract_features(contour, pre.hsv)
                    if f0["solidity"] < split["max_solidity"]:
                        region_mask = np.zeros((h, w), np.uint8)
                        cv2.drawContours(region_mask, [contour], -1, 255, cv2.FILLED)
                        pieces = [c for c in split_touching_objects(region_mask, split["min_distance_ratio"])
                                  if cv2.contourArea(c) >= min_area] or [contour]
                for piece in pieces:
                    f = extract_features(piece, pre.hsv)
                    cls, reason = classify_region(group, f, p["rules"])
                    region = {"group": group, "class": cls, "reason": reason, **f}
                    regions.append(region)
                    if cls is None:
                        continue
                    x1, y1, x2, y2 = f["bbox"]
                    s = pre.scale
                    box = (int(x1 / s), int(y1 / s), int(x2 / s), int(y2 / s))  # back to original size
                    score = rule_score(cls, f, p["base_confidence"], p.get("size_reference_fraction"))
                    candidates.append(Detection(cls, score, box,
                                                self.classes.index(cls), {"group": group, "reason": reason, "features": f}))
        t3 = time.perf_counter()

        dets = non_max_suppression(candidates, p["nms_iou"], class_agnostic=True)
        dets = [d for d in dets if d.confidence >= p["min_confidence"]][: p["max_detections"]]
        t4 = time.perf_counter()

        result = BaselineResult(dets, count_by_class(dets, self.classes), len(dets), regions)
        result.timing_ms = {"preprocess": (t1 - t0) * 1e3, "segmentation": (t2 - t1) * 1e3,
                            "contours_features_rules": (t3 - t2) * 1e3, "nms": (t4 - t3) * 1e3,
                            "total": (t4 - t0) * 1e3}
        if keep_stages:
            result.stages = self._stage_images(image, pre.bgr, raw_masks, clean_masks, regions, dets)
        return result

    def _stage_images(self, original, processed, raw_masks, clean_masks, regions, dets) -> dict[str, np.ndarray]:
        """Images of each processing stage, for explanation and the user interface."""
        h, w = processed.shape[:2]

        def overlay(masks: dict[str, np.ndarray]) -> np.ndarray:
            canvas = np.zeros((h, w, 3), np.uint8)
            for g, mk in masks.items():
                canvas[mk > 0] = GROUP_COLORS.get(g, (255, 255, 255))
            return canvas

        contour_img = processed.copy()
        for g, mk in clean_masks.items():
            cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_img, cnts, -1, GROUP_COLORS.get(g, (255, 255, 255)), 2)
        return {
            "1_preprocessed": processed,
            "2_color_masks": overlay(raw_masks),
            "3_after_morphology": overlay(clean_masks),
            "4_contours": contour_img,
            "5_detections": draw_detections(original, dets),
        }


def stage_panel(stages: dict[str, np.ndarray], cell: int = 360) -> np.ndarray:
    """Side-by-side panel of all stages with titles."""
    tiles = []
    for name, img in stages.items():
        tile = cv2.resize(img, (cell, cell))
        cv2.rectangle(tile, (0, 0), (cell, 26), (255, 255, 255), -1)
        cv2.putText(tile, name.split("_", 1)[1].replace("_", " "), (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    return make_grid(tiles, cols=len(tiles), cell_size=cell)


def _collect_inputs(paths: list[str], extensions: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        files.extend(list_images(p, extensions) if p.is_dir() else [p])
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="image files and/or folders")
    parser.add_argument("--out", default="outputs/baseline/demo", help="output folder")
    parser.add_argument("--stages", action="store_true", help="also save the processing-stage panel")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    detector = BaselineDetector(cfg)
    out = ensure_dir(args.out)
    files = _collect_inputs(args.inputs, cfg["dataset"]["image_extensions"])
    if not files:
        print("ERROR: no images found", file=sys.stderr)
        return 1

    summary = []
    for path in files:
        if not path.is_file():
            print(f"SKIP {path}: file not found")
            summary.append({"image": path.name, "error": "file not found"})
            continue
        image = read_image(path)
        if image is None:
            print(f"SKIP {path}: not a readable image")
            summary.append({"image": path.name, "error": "unreadable image"})
            continue
        res = detector.detect(image, keep_stages=args.stages)
        write_image(out / f"{path.stem}_baseline.jpg", draw_detections(image, res.detections))
        if args.stages:
            write_image(out / f"{path.stem}_stages.jpg", stage_panel(res.stages))
        found = {k: v for k, v in res.counts.items() if v}
        print(f"{path.name}: {res.total} fruit(s) {found}  [{res.timing_ms['total']:.0f} ms]")
        summary.append({"image": path.name, "total": res.total, "counts": res.counts,
                        "time_ms": round(res.timing_ms["total"], 1),
                        "detections": [d.as_dict() | {"reason": d.extra["reason"]} for d in res.detections]})
    save_json(summary, out / "results.json")
    print(f"Saved annotated images and results.json to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
