"""Evaluate the OpenCV baseline against ground-truth labels.

Default split is VALIDATION. The test split is locked: it can only be used with
--split test --final (done once, in Phase 4, for the YOLO-vs-baseline comparison).

Measures: per-class precision / recall / F1 / AP50 / AP50-95 (same formulas as for YOLO),
class-agnostic localization recall, fruit-count error, and processing time.
Optional --brightness simulates darker/brighter lighting to test robustness.

Usage:
    python -m src.baseline.evaluate_baseline                      # full val split
    python -m src.baseline.evaluate_baseline --limit 100          # quick check
    python -m src.baseline.evaluate_baseline --brightness 0.6     # darker images
    python -m src.baseline.evaluate_baseline --plain-background   # controlled-scene subset only
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from src.baseline.pipeline import BaselineDetector
from src.data.dataset_validator import parse_label_file
from src.utils.detections import Detection, draw_detections
from src.utils.helpers import ensure_dir, list_images, load_config, resolve_path, save_json, system_info
from src.utils.image_utils import make_grid, read_image, write_image, yolo_to_xyxy
from src.utils.metrics import ImageEval, count_errors, evaluate, localization_recall


def load_ground_truth(label_path, width: int, height: int, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    boxes, _ = parse_label_file(label_path, num_classes) if label_path.is_file() else ([], [])
    classes = np.array([b[0] for b in boxes], dtype=int)
    xyxy = np.array([yolo_to_xyxy(*b[1:], width, height) for b in boxes], dtype=float).reshape(-1, 4)
    return classes, xyxy


def to_eval(dets: list[Detection], gt_classes, gt_boxes) -> ImageEval:
    return ImageEval(
        pred_classes=np.array([d.class_id for d in dets], dtype=int),
        pred_scores=np.array([d.confidence for d in dets], dtype=float),
        pred_boxes=np.array([d.box for d in dets], dtype=float).reshape(-1, 4),
        gt_classes=gt_classes, gt_boxes=gt_boxes)


def is_plain_background(image: np.ndarray, gt_boxes: np.ndarray, rules: dict) -> bool:
    """True if the image area outside all ground-truth boxes is mostly gray/white (a 'controlled' scene)."""
    h, w = image.shape[:2]
    outside = np.ones((h, w), bool)
    for x1, y1, x2, y2 in gt_boxes.astype(int):
        outside[y1:y2, x1:x2] = False
    if outside.mean() < rules["min_background_fraction"]:
        return False
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1][outside]
    return bool((saturation < rules["max_saturation"]).mean() >= rules["min_plain_fraction"])


def gt_detections(gt_classes, gt_boxes, classes) -> list[Detection]:
    return [Detection(classes[c], 1.0, tuple(int(v) for v in b), int(c)) for c, b in zip(gt_classes, gt_boxes)]


def run(split: str, cfg: dict, limit: int | None, brightness: float, n_examples: int,
        plain_background: bool = False) -> dict:
    classes = cfg["classes"]
    split_dir = resolve_path(cfg["dataset"]["processed_dir"]) / split
    images = list_images(split_dir / "images", cfg["dataset"]["image_extensions"])
    if not images:
        raise FileNotFoundError(f"No images in {split_dir}. Run the Phase 2 dataset tools first.")
    if limit:
        images = images[:limit]

    detector = BaselineDetector(cfg)
    evals, det_ms, e2e_ms, examples = [], [], [], []
    # Spread saved examples over the split (a filtered subset is small, so take them in order).
    example_step = 1 if plain_background else max(1, len(images) // max(1, n_examples))
    for i, path in enumerate(images, 1):
        t_read = time.perf_counter()
        image = read_image(path)
        read_ms = (time.perf_counter() - t_read) * 1e3
        if image is None:
            continue
        h, w = image.shape[:2]
        gt_c, gt_b = load_ground_truth(split_dir / "labels" / f"{path.stem}.txt", w, h, len(classes))
        if plain_background and not is_plain_background(image, gt_b, cfg["baseline"]["plain_background"]):
            continue
        t0 = time.perf_counter()  # restart the timer: selection is not part of the detector
        if brightness != 1.0:
            image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        result = detector.detect(image)
        annotated = draw_detections(image, result.detections)
        e2e_ms.append(read_ms + (time.perf_counter() - t0) * 1e3)  # read + detect + draw
        det_ms.append(result.timing_ms["total"])

        evals.append(to_eval(result.detections, gt_c, gt_b))
        if len(examples) < n_examples and len(evals) % example_step == 0:
            gt_img = draw_detections(image, gt_detections(gt_c, gt_b, classes), show_confidence=False)
            examples.append((path.stem, gt_img, annotated))
        if i % 100 == 0 or i == len(images):
            print(f"  {i}/{len(images)} images", flush=True)

    metrics = evaluate(evals, classes)
    metrics.update({
        "split": split, "images": len(evals), "brightness_factor": brightness,
        "subset": "plain background only" if plain_background else "all images",
        "localization_recall@0.5": localization_recall(evals),
        "counting": count_errors(evals),
        "timing_ms": {"detector_mean": round(float(np.mean(det_ms)), 1),
                      "detector_median": round(float(np.median(det_ms)), 1),
                      "end_to_end_mean": round(float(np.mean(e2e_ms)), 1),
                      "note": "end-to-end = image read + detection + drawing boxes; single CPU process"},
        "hardware": system_info() | {"opencv": cv2.__version__},
    })
    return {"metrics": metrics, "examples": examples}


def write_report(metrics: dict, out_dir) -> None:
    lines = [f"# OpenCV baseline - evaluation on `{metrics['split']}` "
             f"({metrics['images']} images, {metrics['subset']}, brightness x{metrics['brightness_factor']})\n",
             "| Class | GT boxes | Predictions | Precision@0.5 | Recall@0.5 | F1@0.5 | AP50 | AP50-95 |",
             "|---|---|---|---|---|---|---|---|"]
    for c, m in metrics["per_class"].items():
        lines.append(f"| {c} | {m['ground_truth']} | {m['predictions']} | {m['precision@0.5']:.3f} | "
                     f"{m['recall@0.5']:.3f} | {m['f1@0.5']:.3f} | {m['AP50']:.3f} | {m['AP50-95']:.3f} |")
    mm = metrics["mean"]
    lines.append(f"| **Mean** | | | **{mm['precision@0.5']:.3f}** | **{mm['recall@0.5']:.3f}** | "
                 f"**{mm['f1@0.5']:.3f}** | **{mm['mAP50']:.3f}** | **{mm['mAP50-95']:.3f}** |")
    c, t = metrics["counting"], metrics["timing_ms"]
    lines += ["", f"- Localization recall (any fruit found at IoU 0.5, label ignored): "
                  f"**{metrics['localization_recall@0.5']:.3f}**",
              f"- Count error per image: mean absolute {c['mean_absolute_count_error']}, mean signed "
              f"{c['mean_signed_count_error']} (negative = under-count), exact count {c['exact_count_rate']:.1%}",
              f"- Time per image: detector {t['detector_mean']} ms (median {t['detector_median']}), "
              f"end-to-end {t['end_to_end_mean']} ms ({metrics['hardware']['cpu']})",
              "", "Example images: `examples/` (left = ground truth, right = baseline prediction)."]
    (out_dir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--final", action="store_true", help="required to evaluate on the locked test split")
    parser.add_argument("--limit", type=int, default=None, help="only the first N images")
    parser.add_argument("--brightness", type=float, default=1.0, help="multiply pixel values (lighting test)")
    parser.add_argument("--plain-background", action="store_true",
                        help="only images with a plain (gray/white) background, i.e. controlled scenes")
    parser.add_argument("--examples", type=int, default=8, help="number of GT-vs-prediction examples to save")
    parser.add_argument("--out", default=None, help="output folder (default outputs/baseline/eval_<split>)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        print("ERROR: the test split is locked. Use --split test --final only for the final evaluation.",
              file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    suffix = "" if args.brightness == 1.0 else f"_brightness{args.brightness}"
    suffix += "_plain_background" if args.plain_background else ""
    out = ensure_dir(args.out or f"outputs/baseline/eval_{args.split}{suffix}")
    try:
        res = run(args.split, cfg, args.limit, args.brightness, args.examples, args.plain_background)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    metrics = res["metrics"]
    save_json(metrics, out / "metrics.json")
    write_report(metrics, out)
    ex_dir = ensure_dir(out / "examples")
    for old in ex_dir.glob("*.jpg"):
        old.unlink()
    for name, gt_img, pred_img in res["examples"]:
        write_image(ex_dir / f"{name[:16]}.jpg", make_grid([gt_img, pred_img], cols=2, cell_size=400))
    print((out / "metrics.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
