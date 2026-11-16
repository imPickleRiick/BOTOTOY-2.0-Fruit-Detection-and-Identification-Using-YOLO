"""Compare the OpenCV baseline with YOLO11n on the SAME images using the SAME metrics.

For every image both detectors run on this computer (CPU); the predictions are scored with
src/utils/metrics.py (identical matching and AP formula), on two subsets:
  - all images of the split (cluttered, real-world scenes)
  - plain-background images only (controlled scenes, where the baseline is designed to work)

Usage:
    python -m src.evaluation.compare_methods                    # validation split
    python -m src.evaluation.compare_methods --split test --final   # final, one-time comparison
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from src.baseline.evaluate_baseline import gt_detections, is_plain_background, load_ground_truth
from src.baseline.pipeline import BaselineDetector
from src.utils.detections import Detection, draw_detections
from src.utils.helpers import ensure_dir, list_images, load_config, resolve_path, save_json, system_info
from src.utils.image_utils import make_grid, read_image, write_image
from src.utils.metrics import ImageEval, count_errors, evaluate, localization_recall

AP_CONF = 0.001


def _eval(dets: list[Detection], gt_c, gt_b) -> ImageEval:
    return ImageEval(np.array([d.class_id for d in dets], int), np.array([d.confidence for d in dets], float),
                     np.array([d.box for d in dets], float).reshape(-1, 4), gt_c, gt_b)


def _summarize(ap_evals: list[ImageEval], op_evals: list[ImageEval], times: list[float], classes) -> dict:
    ap, op = evaluate(ap_evals, classes), evaluate(op_evals, classes)
    return {"mAP50": ap["mean"]["mAP50"], "mAP50-95": ap["mean"]["mAP50-95"],
            "precision": op["mean"]["precision@0.5"], "recall": op["mean"]["recall@0.5"], "f1": op["mean"]["f1@0.5"],
            "AP50_per_class": {c: v["AP50"] for c, v in ap["per_class"].items()},
            "recall_per_class": {c: v["recall@0.5"] for c, v in op["per_class"].items()},
            "localization_recall": localization_recall(op_evals),
            **count_errors(op_evals),
            "end_to_end_ms_mean": round(float(np.mean(times)), 1) if times else None}


def _labelled(image: np.ndarray, title: str, size: int = 420) -> np.ndarray:
    tile = cv2.resize(image, (size, size))
    cv2.rectangle(tile, (0, 0), (size, 28), (255, 255, 255), -1)
    cv2.putText(tile, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return tile


def compare(cfg: dict, split: str, limit: int | None, n_examples: int) -> dict:
    from src.yolo.detector import YoloDetector

    classes = cfg["classes"]
    split_dir = resolve_path(cfg["dataset"]["processed_dir"]) / split
    images = list_images(split_dir / "images", cfg["dataset"]["image_extensions"])
    if not images:
        raise FileNotFoundError(f"No images in {split_dir}")
    images = images[:limit] if limit else images

    baseline, yolo = BaselineDetector(cfg), YoloDetector(None, cfg)
    op_conf = yolo.params["conf"]
    yolo.detect(read_image(images[0]))  # warm-up

    rec = {"baseline": {"ap": [], "op": [], "t": []}, "yolo": {"ap": [], "op": [], "t": []}}
    plain_flags, per_image = [], []
    for i, path in enumerate(images, 1):
        image = read_image(path)
        h, w = image.shape[:2]
        gt_c, gt_b = load_ground_truth(split_dir / "labels" / f"{path.stem}.txt", w, h, len(classes))
        plain_flags.append(is_plain_background(image, gt_b, cfg["baseline"]["plain_background"]))

        t0 = time.perf_counter()
        b = baseline.detect(image)
        b_img = draw_detections(image, b.detections)
        rec["baseline"]["t"].append((time.perf_counter() - t0) * 1e3)
        b_eval = _eval(b.detections, gt_c, gt_b)
        rec["baseline"]["ap"].append(b_eval)   # the baseline has no confidence threshold to vary
        rec["baseline"]["op"].append(b_eval)

        t0 = time.perf_counter()
        y = yolo.detect(image, conf=op_conf)
        y_img = draw_detections(image, y.detections)
        rec["yolo"]["t"].append((time.perf_counter() - t0) * 1e3)
        rec["yolo"]["op"].append(_eval(y.detections, gt_c, gt_b))
        rec["yolo"]["ap"].append(_eval(yolo.detect(image, conf=AP_CONF).detections, gt_c, gt_b))

        per_image.append({"path": path, "gt": len(gt_c), "plain": plain_flags[-1],
                          "baseline": b.total, "yolo": y.total, "gt_img": None})
        if i % 100 == 0 or i == len(images):
            print(f"  {i}/{len(images)} images", flush=True)

    results = {}
    for subset, mask in (("all", [True] * len(images)), ("plain_background", plain_flags)):
        idx = [k for k, keep in enumerate(mask) if keep]
        results[subset] = {"images": len(idx)}
        for method, r in rec.items():
            results[subset][method] = _summarize([r["ap"][k] for k in idx], [r["op"][k] for k in idx],
                                                 [r["t"][k] for k in idx], classes)

    # --- side-by-side examples: controlled scenes first, then the most crowded scenes ---
    out_examples = []
    plain = [r for r in per_image if r["plain"] and r["gt"] >= 1][: n_examples // 2]
    crowded = sorted([r for r in per_image if not r["plain"]], key=lambda r: -r["gt"])[: n_examples - len(plain)]
    for r in plain + crowded:
        image = read_image(r["path"])
        h, w = image.shape[:2]
        gt_c, gt_b = load_ground_truth(split_dir / "labels" / f"{r['path'].stem}.txt", w, h, len(classes))
        panel = make_grid([
            _labelled(draw_detections(image, gt_detections(gt_c, gt_b, classes), show_confidence=False),
                      f"Ground truth ({r['gt']})"),
            _labelled(draw_detections(image, baseline.detect(image).detections), f"OpenCV baseline ({r['baseline']})"),
            _labelled(draw_detections(image, yolo.detect(image, conf=op_conf).detections), f"YOLO11n ({r['yolo']})"),
        ], cols=3, cell_size=420)
        out_examples.append((("plain" if r["plain"] else "crowded") + "_" + r["path"].stem[:16], panel))
    return {"split": split, "images": len(images), "yolo_operating_confidence": op_conf,
            "results": results, "system": system_info(), "examples": out_examples}


def write_report(report: dict, out_dir) -> None:
    lines = [f"# OpenCV baseline vs YOLO11n — `{report['split']}` split\n",
             f"Same images, same metrics (`src/utils/metrics.py`), same computer ({report['system']['cpu']}, CPU). "
             f"AP uses all predictions; precision/recall/counts use the app's operating point "
             f"(YOLO confidence ≥ {report['yolo_operating_confidence']}; the baseline has no threshold).\n"]
    for subset, title in (("all", "All images (real-world scenes)"), ("plain_background", "Plain-background images (controlled scenes)")):
        r = report["results"][subset]
        b, y = r["baseline"], r["yolo"]
        lines += [f"## {title} — {r['images']} images\n", "| Metric | OpenCV baseline | YOLO11n |", "|---|---|---|"]
        for key, label in (("mAP50", "mAP@0.50"), ("mAP50-95", "mAP@0.50:0.95"), ("precision", "Precision"),
                           ("recall", "Recall"), ("f1", "F1"), ("localization_recall", "Localization recall (label ignored)"),
                           ("mean_absolute_count_error", "Mean absolute count error (fruits/image)"),
                           ("exact_count_rate", "Exact count rate"), ("end_to_end_ms_mean", "End-to-end time (ms/image)")):
            fmt = (lambda v: f"{v:.1%}") if key == "exact_count_rate" else (lambda v: f"{v:.3f}" if isinstance(v, float) and v < 10 else f"{v}")
            lines.append(f"| {label} | {fmt(b[key])} | {fmt(y[key])} |")
        lines += ["", "| Class | Baseline AP50 | YOLO11n AP50 | Baseline recall | YOLO11n recall |", "|---|---|---|---|---|"]
        for c in b["AP50_per_class"]:
            lines.append(f"| {c} | {b['AP50_per_class'][c]:.3f} | {y['AP50_per_class'][c]:.3f} | "
                         f"{b['recall_per_class'][c]:.3f} | {y['recall_per_class'][c]:.3f} |")
        lines.append("")
    lines.append("Side-by-side examples (ground truth | baseline | YOLO11n, count in brackets): `examples/`.")
    (out_dir / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--final", action="store_true", help="required for the locked test split")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--out", default=None, help="default outputs/comparisons/<split>")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        print("ERROR: the test split is locked. Use --split test --final only for the final evaluation.", file=sys.stderr)
        return 1
    cfg = load_config(args.config)
    out_dir = ensure_dir(args.out or f"outputs/comparisons/{args.split}")
    try:
        report = compare(cfg, args.split, args.limit, args.examples)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    ex_dir = ensure_dir(out_dir / "examples")
    for old in ex_dir.glob("*.jpg"):
        old.unlink()
    for name, panel in report.pop("examples"):
        write_image(ex_dir / f"{name}.jpg", panel)
    save_json(report, out_dir / "comparison.json")
    write_report(report, out_dir)
    print((out_dir / "COMPARISON.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
