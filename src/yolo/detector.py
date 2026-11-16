"""YOLO11n fruit detector: loads trained weights and turns Ultralytics output into our
shared Detection format (same as the OpenCV baseline).

What happens inside model.predict():
  1. Preprocess: letterbox-resize to imgsz (keeps aspect ratio, pads the rest), BGR->RGB,
     scale pixels to 0..1.
  2. Inference: one forward pass of the network predicts, for every grid location, box
     coordinates and a score for each of the 6 classes.
  3. Postprocess: drop boxes below the CONFIDENCE threshold, then NON-MAXIMUM SUPPRESSION
     removes boxes that overlap a higher-scoring box of the same class by more than the IoU
     threshold, so each fruit is reported once. Boxes are scaled back to the original image.

Usage:
    python -m src.yolo.detector photo.jpg [more images or folders] --out outputs/yolo/demo
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.utils.detections import Detection, count_by_class, draw_detections
from src.utils.helpers import ensure_dir, list_images, load_config, resolve_path, save_json
from src.utils.image_utils import read_image, write_image


@dataclass
class YoloResult:
    detections: list[Detection]
    counts: dict[str, int]
    total: int
    timing_ms: dict[str, float] = field(default_factory=dict)


class YoloDetector:
    def __init__(self, weights: str | Path | None = None, config: dict | None = None, **overrides):
        from ultralytics import YOLO

        cfg = config or load_config()
        self.classes: list[str] = cfg["classes"]
        self.params = {**cfg["yolo"]["inference"], **{k: v for k, v in overrides.items() if v is not None}}
        self.weights = resolve_path(weights or cfg["yolo"]["trained_weights"])
        if not self.weights.is_file():
            raise FileNotFoundError(f"Trained weights not found: {self.weights}. "
                                    "Train the model (see README) or copy best.pt there.")
        self.model = YOLO(str(self.weights))
        names = [self.model.names[i] for i in sorted(self.model.names)]
        if names != self.classes:
            raise ValueError(f"Model classes {names} do not match config classes {self.classes}")

    def detect(self, image_bgr: np.ndarray, conf: float | None = None, iou: float | None = None,
               imgsz: int | None = None) -> YoloResult:
        """Detect fruits in one BGR image (numpy array, OpenCV convention)."""
        p = self.params
        t0 = time.perf_counter()
        result = self.model.predict(image_bgr, conf=conf if conf is not None else p["conf"],
                                    iou=iou if iou is not None else p["iou"], imgsz=imgsz or p["imgsz"],
                                    max_det=p["max_det"], device=p["device"], verbose=False)[0]
        total_ms = (time.perf_counter() - t0) * 1e3

        dets = []
        boxes = result.boxes
        for xyxy, score, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
            cid = int(cls)
            dets.append(Detection(self.classes[cid], float(score), tuple(int(round(v)) for v in xyxy), cid))
        # result.speed holds Ultralytics' own per-stage timing in milliseconds.
        timing = {k: round(v, 2) for k, v in result.speed.items()} | {"total": round(total_ms, 2)}
        return YoloResult(dets, count_by_class(dets, self.classes), len(dets), timing)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="image files and/or folders")
    parser.add_argument("--weights", default=None, help="default: yolo.trained_weights in config.yaml")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--out", default="outputs/yolo/demo")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    try:
        detector = YoloDetector(args.weights, cfg, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    out = ensure_dir(args.out)
    files: list[Path] = []
    for p in map(Path, args.inputs):
        files.extend(list_images(p, cfg["dataset"]["image_extensions"]) if p.is_dir() else [p])

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
        res = detector.detect(image)
        write_image(out / f"{path.stem}_yolo.jpg", draw_detections(image, res.detections))
        found = {k: v for k, v in res.counts.items() if v}
        print(f"{path.name}: {res.total} fruit(s) {found}  [{res.timing_ms['total']:.0f} ms]")
        summary.append({"image": path.name, "total": res.total, "counts": res.counts, "timing_ms": res.timing_ms,
                        "detections": [d.as_dict() for d in res.detections]})
    save_json(summary, out / "results.json")
    print(f"Saved annotated images and results.json to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
