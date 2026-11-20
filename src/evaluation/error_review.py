"""Error review for YOLO11n: every false positive and every missed fruit, with its likely cause.

The Phase 4 evaluation saved whole example images. This tool looks at errors one at a time,
as the proposal asks ("a confusion/error review with at least 20 false positives or missed objects"):

1. Run YOLO11n at the app's operating point (confidence 0.25, NMS IoU 0.7) on a split.
2. Match predictions to ground truth (IoU >= 0.5, same class), exactly like src/utils/metrics.py.
3. Give every error an automatic category from its geometry:

   False positives (a box with no matching ground-truth fruit)
   - wrong_class        overlaps (IoU >= 0.5) a ground-truth fruit of another class
   - group_box          the box contains the centers of >= 2 ground-truth fruits of its class
                        (the model boxed a whole group; ground truth has single fruits)
   - duplicate          overlaps a same-class fruit that another prediction already matched
   - inside_group_label the box lies inside a ground-truth box of its class that covers a group
                        (ground truth labeled a group; the model found >= 2 smaller single fruits in it)
   - poor_box           overlaps a same-class fruit, but with IoU between 0.1 and 0.5
   - background         no ground-truth fruit of any class nearby (IoU < 0.1)

   Missed fruits (false negatives)
   - wrong_class        a prediction of another class covers it (IoU >= 0.5)
   - covered_by_group   its center lies inside a predicted box of its class that covers a group
   - poor_box           a same-class prediction overlaps it with IoU between 0.1 and 0.5
   - small              not detected at all, and smaller than 1% of the image area
   - not_detected       not detected at all

4. Save the counts, a CSV of every error, and two numbered contact sheets with a fixed random
   sample of errors (seed 42) for the manual review in ERROR_REVIEW.md.

The test split was already evaluated once in Phase 4; this review only describes those errors.
Nothing here changes the model, the thresholds or the reported metrics.

Usage:
    python -m src.evaluation.error_review --split test --final
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.baseline.evaluate_baseline import load_ground_truth
from src.utils.helpers import ensure_dir, list_images, load_config, resolve_path, save_json
from src.utils.image_utils import CLASS_COLORS_BGR, draw_box, read_image, write_image
from src.utils.metrics import ImageEval, iou_matrix, match_predictions

FP_CATEGORIES = ["wrong_class", "group_box", "inside_group_label", "duplicate", "poor_box", "background"]
FN_CATEGORIES = ["wrong_class", "covered_by_group", "poor_box", "small", "not_detected"]
SMALL_AREA = 0.01  # share of the image area


def _centers(boxes: np.ndarray) -> np.ndarray:
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


def _inside(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    return (points[:, 0] >= box[0]) & (points[:, 0] <= box[2]) & (points[:, 1] >= box[1]) & (points[:, 1] <= box[3])


def _box_inside(inner: np.ndarray, outer: np.ndarray, tol: float = 0.1) -> bool:
    """True if `inner` lies inside `outer` (with a tolerance of 10% of the outer box size)."""
    w, h = outer[2] - outer[0], outer[3] - outer[1]
    return (inner[0] >= outer[0] - tol * w and inner[1] >= outer[1] - tol * h
            and inner[2] <= outer[2] + tol * w and inner[3] <= outer[3] + tol * h)


def matched_pairs(ev: ImageEval) -> tuple[np.ndarray, dict[int, int]]:
    """Same greedy matching as metrics.match_predictions at IoU 0.5, also returning pred -> gt."""
    tp = match_predictions(ev, [0.5])[:, 0]
    pairs: dict[int, int] = {}
    if len(ev.pred_scores) and len(ev.gt_classes):
        ious = iou_matrix(ev.pred_boxes, ev.gt_boxes) * (ev.pred_classes[:, None] == ev.gt_classes[None, :])
        taken = np.zeros(len(ev.gt_classes), bool)
        for p in np.argsort(-ev.pred_scores, kind="stable"):
            cand = np.where(~taken & (ious[p] >= 0.5))[0]
            if len(cand):
                g = cand[np.argmax(ious[p, cand])]
                taken[g] = True
                pairs[int(p)] = int(g)
    return tp, pairs


def categorize(ev: ImageEval, img_area: float) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Return [(pred index, category)] for false positives and [(gt index, category)] for misses."""
    tp, pairs = matched_pairs(ev)
    matched_gt = set(pairs.values())
    n_p, n_g = len(ev.pred_scores), len(ev.gt_classes)
    ious = iou_matrix(ev.pred_boxes, ev.gt_boxes) if n_p and n_g else np.zeros((n_p, n_g))
    same = (ev.pred_classes[:, None] == ev.gt_classes[None, :]) if n_p and n_g else np.zeros((n_p, n_g), bool)
    gt_centers = _centers(ev.gt_boxes) if n_g else np.zeros((0, 2))
    pred_centers = _centers(ev.pred_boxes) if n_p else np.zeros((0, 2))

    # Group boxes: a predicted box holding the centers of >= 2 same-class GT fruits, and a GT box
    # holding the centers of >= 2 same-class predictions.
    pred_is_group = np.zeros(n_p, bool)
    for p in range(n_p):
        pred_is_group[p] = int((_inside(gt_centers, ev.pred_boxes[p]) & (ev.gt_classes == ev.pred_classes[p])).sum()) >= 2 if n_g else False
    # Only predictions clearly smaller than the GT box count (a duplicate of the GT box is not a "single fruit").
    pred_area = (ev.pred_boxes[:, 2] - ev.pred_boxes[:, 0]) * (ev.pred_boxes[:, 3] - ev.pred_boxes[:, 1]) if n_p else np.zeros(0)
    gt_area = (ev.gt_boxes[:, 2] - ev.gt_boxes[:, 0]) * (ev.gt_boxes[:, 3] - ev.gt_boxes[:, 1]) if n_g else np.zeros(0)
    gt_is_group = np.zeros(n_g, bool)
    for g in range(n_g):
        smaller = (ev.pred_classes == ev.gt_classes[g]) & (pred_area < 0.5 * gt_area[g]) if n_p else np.zeros(0, bool)
        gt_is_group[g] = int((_inside(pred_centers, ev.gt_boxes[g]) & smaller).sum()) >= 2 if n_p else False

    fps = []
    for p in range(n_p):
        if tp[p]:
            continue
        other = ious[p] * ~same[p] if n_g else np.zeros(0)
        own = ious[p] * same[p] if n_g else np.zeros(0)
        if n_g and other.max() >= 0.5:
            cat = "wrong_class"
        elif pred_is_group[p]:
            cat = "group_box"
        elif n_g and own.max() >= 0.5:
            cat = "duplicate"
        elif n_g and any(gt_is_group[g] and same[p, g] and _box_inside(ev.pred_boxes[p], ev.gt_boxes[g]) for g in range(n_g)):
            cat = "inside_group_label"
        elif n_g and own.max() >= 0.1:
            cat = "poor_box"
        else:
            cat = "background"
        fps.append((p, cat))

    fns = []
    for g in range(n_g):
        if g in matched_gt:
            continue
        other = ious[:, g] * ~same[:, g] if n_p else np.zeros(0)
        own = ious[:, g] * same[:, g] if n_p else np.zeros(0)
        covering = [p for p in range(n_p) if same[p, g] and pred_is_group[p] and _inside(gt_centers[g:g + 1], ev.pred_boxes[p])[0]]
        box = ev.gt_boxes[g]
        area = (box[2] - box[0]) * (box[3] - box[1]) / img_area
        if n_p and other.max() >= 0.5:
            cat = "wrong_class"
        elif covering:
            cat = "covered_by_group"
        elif n_p and own.max() >= 0.1:
            cat = "poor_box"
        elif area < SMALL_AREA:
            cat = "small"
        else:
            cat = "not_detected"
        fns.append((g, cat))
    return fps, fns


def crop_with_context(image: np.ndarray, ev: ImageEval, box: np.ndarray, classes: list[str],
                      highlight: tuple[int, int, int], label: str, cell: int = 300) -> np.ndarray:
    """Crop around one error (2x its size, at least 160 px), all boxes drawn thin, the error box thick."""
    h, w = image.shape[:2]
    out = image.copy()
    for c, b in zip(ev.gt_classes, ev.gt_boxes):  # ground truth: thin white
        cv2.rectangle(out, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (255, 255, 255), 1)
    for c, b in zip(ev.pred_classes, ev.pred_boxes):  # predictions: thin class color
        cv2.rectangle(out, tuple(map(int, b[:2])), tuple(map(int, b[2:])), CLASS_COLORS_BGR[classes[c]], 1)
    draw_box(out, tuple(box), "", highlight, 3)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = max(box[2] - box[0], box[3] - box[1], 80)
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(w, cx + half)), int(min(h, cy + half))
    crop = out[y1:y2, x1:x2]
    scale = cell / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))))
    tile = np.full((cell + 28, cell, 3), 255, np.uint8)
    tile[28:28 + crop.shape[0], :crop.shape[1]] = crop
    cv2.putText(tile, label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return tile


def contact_sheet(tiles: list[np.ndarray], cols: int = 6) -> np.ndarray:
    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * th, cols * tw, 3), 230, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th:r * th + t.shape[0], c * tw:c * tw + t.shape[1]] = t
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--final", action="store_true", help="required for the test split")
    parser.add_argument("--sample", type=int, default=24, help="errors per contact sheet (FP and FN)")
    parser.add_argument("--out", default=None, help="default outputs/yolo/eval_<split>/error_review")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        print("ERROR: the test split is locked. Use --split test --final (review of the final evaluation only).",
              file=sys.stderr)
        return 1

    from src.yolo.detector import YoloDetector

    cfg = load_config(args.config)
    classes = cfg["classes"]
    try:
        detector = YoloDetector(None, cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    split_dir = resolve_path(cfg["dataset"]["processed_dir"]) / args.split
    images = list_images(split_dir / "images", cfg["dataset"]["image_extensions"])
    if not images:
        print(f"ERROR: no images in {split_dir}. Prepare the dataset first (see README).", file=sys.stderr)
        return 1
    out_dir = ensure_dir(args.out or f"outputs/yolo/eval_{args.split}/error_review")

    rows = []
    totals = {"tp": 0, "gt": 0, "pred": 0}
    for i, path in enumerate(images, 1):
        image = read_image(path)
        h, w = image.shape[:2]
        dets = detector.detect(image).detections  # operating point from config (conf 0.25)
        gt_c, gt_b = load_ground_truth(split_dir / "labels" / f"{path.stem}.txt", w, h, len(classes))
        ev = ImageEval(np.array([d.class_id for d in dets], int), np.array([d.confidence for d in dets], float),
                       np.array([d.box for d in dets], float).reshape(-1, 4), gt_c, gt_b)
        fps, fns = categorize(ev, float(w * h))
        totals["tp"] += len(ev.pred_scores) - len(fps)
        totals["gt"] += len(gt_c)
        totals["pred"] += len(ev.pred_scores)
        for p, cat in fps:
            rows.append({"image": path.name, "kind": "FP", "class": classes[ev.pred_classes[p]],
                         "confidence": round(float(ev.pred_scores[p]), 3), "category": cat,
                         "box": [int(v) for v in ev.pred_boxes[p]], "gt_in_image": len(gt_c)})
        for g, cat in fns:
            b = ev.gt_boxes[g]
            rows.append({"image": path.name, "kind": "FN", "class": classes[ev.gt_classes[g]], "confidence": "",
                         "category": cat, "box": [int(v) for v in b], "gt_in_image": len(gt_c),
                         "area_pct": round(100 * (b[2] - b[0]) * (b[3] - b[1]) / (w * h), 2)})
        if i % 100 == 0 or i == len(images):
            print(f"  {i}/{len(images)} images", flush=True)

    fp_rows = [r for r in rows if r["kind"] == "FP"]
    fn_rows = [r for r in rows if r["kind"] == "FN"]
    by_cat = {"FP": Counter(r["category"] for r in fp_rows), "FN": Counter(r["category"] for r in fn_rows)}
    by_class: dict[str, dict[str, Counter]] = {"FP": defaultdict(Counter), "FN": defaultdict(Counter)}
    for r in rows:
        by_class[r["kind"]][r["class"]][r["category"]] += 1
    imgs_fn = Counter(r["image"] for r in fn_rows)
    top_share = sum(n for _, n in imgs_fn.most_common(20)) / max(1, len(fn_rows))

    # Fixed random sample for the manual review (numbered tiles = row ids in sample.csv).
    rng = random.Random(cfg["project"]["seed"])
    samples = {"FP": rng.sample(fp_rows, min(args.sample, len(fp_rows))),
               "FN": rng.sample(fn_rows, min(args.sample, len(fn_rows)))}
    sample_rows = []
    cache: dict[str, tuple[np.ndarray, ImageEval]] = {}
    for kind, picked in samples.items():
        tiles = []
        for n, r in enumerate(picked, 1):
            if r["image"] not in cache:
                img = read_image(split_dir / "images" / r["image"])
                h, w = img.shape[:2]
                dets = detector.detect(img).detections
                gt_c, gt_b = load_ground_truth(split_dir / "labels" / f"{Path(r['image']).stem}.txt", w, h, len(classes))
                cache[r["image"]] = (img, ImageEval(np.array([d.class_id for d in dets], int),
                                                    np.array([d.confidence for d in dets], float),
                                                    np.array([d.box for d in dets], float).reshape(-1, 4), gt_c, gt_b))
            img, ev = cache[r["image"]]
            color = (0, 0, 255) if kind == "FP" else (255, 0, 255)
            conf = f" {r['confidence']:.2f}" if kind == "FP" else ""
            tiles.append(crop_with_context(img, ev, np.array(r["box"], float), classes, color,
                                           f"{kind}{n:02d} {r['class']}{conf} [{r['category']}]"))
            sample_rows.append({"id": f"{kind}{n:02d}", **{k: r.get(k, "") for k in
                                ("image", "kind", "class", "confidence", "category", "box", "area_pct")}})
        write_image(out_dir / f"sample_{kind.lower()}.jpg", contact_sheet(tiles), quality=90)

    with open(out_dir / "all_errors.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["image", "kind", "class", "confidence", "category", "box", "area_pct", "gt_in_image"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in fields} for r in rows)
    with open(out_dir / "sample.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)

    summary = {
        "split": args.split, "images": len(images), "operating_point": {"conf": detector.params["conf"],
                                                                         "iou_nms": detector.params["iou"]},
        "ground_truth": totals["gt"], "predictions": totals["pred"], "true_positives": totals["tp"],
        "false_positives": len(fp_rows), "missed": len(fn_rows),
        "fp_by_category": {c: by_cat["FP"].get(c, 0) for c in FP_CATEGORIES},
        "fn_by_category": {c: by_cat["FN"].get(c, 0) for c in FN_CATEGORIES},
        "fp_by_class": {c: dict(by_class["FP"][c]) for c in classes},
        "fn_by_class": {c: dict(by_class["FN"][c]) for c in classes},
        "images_with_misses": len(imgs_fn),
        "share_of_misses_in_20_worst_images": round(top_share, 3),
        "sample_size": {k: len(v) for k, v in samples.items()},
    }
    save_json(summary, out_dir / "summary.json")
    print(f"Saved to {out_dir}")
    print(f"FP {len(fp_rows)}: {summary['fp_by_category']}")
    print(f"FN {len(fn_rows)}: {summary['fn_by_category']}")
    print(f"Misses in the 20 worst images: {100 * top_share:.1f}% of all misses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
