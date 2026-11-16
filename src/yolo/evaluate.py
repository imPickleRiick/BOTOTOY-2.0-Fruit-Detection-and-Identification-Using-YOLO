"""Evaluate trained YOLO11n weights on the validation split or (once, at the end) the locked test split.

Two kinds of results:
1. Ultralytics' official validation (model.val): precision, recall, mAP@0.50,
   mAP@0.50:0.95, per-class AP, confusion matrix, precision-recall curves.
2. Our shared evaluation (src/utils/metrics.py, identical to the OpenCV baseline's):
   AP from all predictions (conf >= 0.001), plus precision / recall / F1, fruit-count
   error and localization recall at the operating confidence threshold (config, 0.25),
   end-to-end timing on this computer, and TP / FP / FN example images.

Usage:
    python -m src.yolo.evaluate                          # validation split
    python -m src.yolo.evaluate --split test --final     # final, one-time test evaluation
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from src.baseline.evaluate_baseline import load_ground_truth
from src.utils.detections import Detection, draw_detections
from src.utils.helpers import ensure_dir, list_images, load_config, resolve_path, save_json, system_info
from src.utils.image_utils import draw_box, read_image, write_image
from src.utils.metrics import ImageEval, count_errors, evaluate, localization_recall, match_predictions

TARGETS = {"mAP50": 0.80, "mAP50-95": 0.50, "min_class_recall": 0.70, "max_seconds_per_image": 1.0}
AP_CONF = 0.001  # keep (almost) all predictions to compute AP, like Ultralytics validation


# ----------------------------------------------------------------------------- Ultralytics validation
def ultralytics_validation(detector, cfg: dict, split: str, out_dir: Path) -> dict:
    """Run model.val() and return its metrics; plots are copied into out_dir/ultralytics/."""
    p = detector.params
    tmp_project = ensure_dir(out_dir / "_ultralytics_tmp")
    m = detector.model.val(data=str(resolve_path(cfg["dataset"]["data_yaml"])), split=split, imgsz=p["imgsz"],
                           batch=8, conf=AP_CONF, iou=p["iou"], device=p["device"], plots=True,
                           project=str(tmp_project), name="val", exist_ok=True, verbose=False, workers=0)
    plots_dir = ensure_dir(out_dir / "ultralytics")
    for f in Path(m.save_dir).glob("*"):
        if f.suffix in (".png", ".jpg"):
            shutil.copy2(f, plots_dir / f.name)
    shutil.rmtree(tmp_project, ignore_errors=True)

    box = m.box
    per_class = {}
    for i, cid in enumerate(box.ap_class_index):
        name = cfg["classes"][int(cid)]
        per_class[name] = {"precision": round(float(box.p[i]), 4), "recall": round(float(box.r[i]), 4),
                           "AP50": round(float(box.ap50[i]), 4), "AP50-95": round(float(box.ap[i]), 4)}
    return {"precision": round(float(box.mp), 4), "recall": round(float(box.mr), 4),
            "mAP50": round(float(box.map50), 4), "mAP50-95": round(float(box.map), 4),
            "per_class": per_class, "speed_ms_per_image": {k: round(v, 2) for k, v in m.speed.items()},
            "note": "precision/recall are at the confidence that maximizes mean F1 (Ultralytics convention)"}


# ----------------------------------------------------------------------------- our evaluation
def _to_eval(dets: list[Detection], gt_c, gt_b) -> ImageEval:
    return ImageEval(np.array([d.class_id for d in dets], int), np.array([d.confidence for d in dets], float),
                     np.array([d.box for d in dets], float).reshape(-1, 4), gt_c, gt_b)


def draw_errors(image, ev: ImageEval, classes: list[str]) -> np.ndarray:
    """TP boxes in the class color, FP boxes in red, missed ground truth (FN) in magenta."""
    out = image.copy()
    tp = match_predictions(ev, [0.5])[:, 0]
    matched_gt = set()
    if len(ev.pred_scores) and len(ev.gt_classes):  # find which GT boxes were matched
        from src.utils.metrics import iou_matrix
        ious = iou_matrix(ev.pred_boxes, ev.gt_boxes) * (ev.pred_classes[:, None] == ev.gt_classes[None, :])
        for p in np.argsort(-ev.pred_scores):
            if tp[p]:
                cand = [g for g in np.argsort(-ious[p]) if g not in matched_gt and ious[p, g] >= 0.5]
                if cand:
                    matched_gt.add(cand[0])
    for k, (c, s, b) in enumerate(zip(ev.pred_classes, ev.pred_scores, ev.pred_boxes)):
        if tp[k]:
            from src.utils.image_utils import CLASS_COLORS_BGR
            draw_box(out, tuple(b), f"TP {classes[c]} {s:.2f}", CLASS_COLORS_BGR[classes[c]], 2)
        else:
            draw_box(out, tuple(b), f"FP {classes[c]} {s:.2f}", (0, 0, 255), 2)
    for g, (c, b) in enumerate(zip(ev.gt_classes, ev.gt_boxes)):
        if g not in matched_gt:
            draw_box(out, tuple(b), f"FN {classes[c]}", (255, 0, 255), 3)
    return out


def shared_evaluation(detector, cfg: dict, split: str, out_dir: Path, limit: int | None, n_examples: int) -> dict:
    classes = cfg["classes"]
    split_dir = resolve_path(cfg["dataset"]["processed_dir"]) / split
    images = list_images(split_dir / "images", cfg["dataset"]["image_extensions"])
    if not images:
        raise FileNotFoundError(f"No images in {split_dir}")
    if limit:
        images = images[:limit]
    op_conf = detector.params["conf"]

    detector.detect(read_image(images[0]))  # warm-up (first call loads/compiles; not timed)
    all_evals, op_evals, e2e_ms, speed = [], [], [], []
    per_image = []
    for i, path in enumerate(images, 1):
        t0 = time.perf_counter()
        image = read_image(path)
        result = detector.detect(image, conf=op_conf)              # what the app does
        draw_detections(image, result.detections)
        e2e_ms.append((time.perf_counter() - t0) * 1e3)          # read + preprocess + infer + NMS + draw
        speed.append(result.timing_ms)

        h, w = image.shape[:2]
        gt_c, gt_b = load_ground_truth(split_dir / "labels" / f"{path.stem}.txt", w, h, len(classes))
        op_eval = _to_eval(result.detections, gt_c, gt_b)
        op_evals.append(op_eval)
        all_evals.append(_to_eval(detector.detect(image, conf=AP_CONF).detections, gt_c, gt_b))
        tp = int(match_predictions(op_eval, [0.5])[:, 0].sum())
        per_image.append({"path": path, "tp": tp, "fp": len(op_eval.pred_scores) - tp,
                          "fn": len(gt_c) - tp, "gt": len(gt_c)})
        if i % 100 == 0 or i == len(images):
            print(f"  {i}/{len(images)} images", flush=True)

    ap_metrics = evaluate(all_evals, classes)
    op_metrics = evaluate(op_evals, classes)
    stage = {k: round(float(np.mean([s[k] for s in speed])), 2) for k in ("preprocess", "inference", "postprocess")}

    # --- error-analysis examples ---
    ex_dir = ensure_dir(out_dir / "examples")
    for old in ex_dir.glob("*.jpg"):
        old.unlink()
    groups = {
        "tp": sorted([r for r in per_image if r["fp"] == 0 and r["fn"] == 0 and r["gt"] >= 2], key=lambda r: -r["gt"]),
        "fp": sorted([r for r in per_image if r["fp"] > 0], key=lambda r: -r["fp"]),
        "fn": sorted([r for r in per_image if r["fn"] > 0], key=lambda r: -r["fn"]),
    }
    index = {r["path"]: k for k, r in enumerate(per_image)}
    for kind, rows in groups.items():
        for n, r in enumerate(rows[:n_examples], 1):
            img = read_image(r["path"])
            write_image(ex_dir / f"{kind}_{n:02d}_{r['path'].stem[:16]}.jpg",
                        draw_errors(img, op_evals[index[r['path']]], classes))

    return {
        "images": len(images),
        "ap_from_all_predictions": {"mean": ap_metrics["mean"],
                                    "per_class": {c: {k: v[k] for k in ("AP50", "AP50-95")}
                                                  for c, v in ap_metrics["per_class"].items()}},
        "operating_point": {"confidence": op_conf, "iou_nms": detector.params["iou"],
                            "mean": {k: op_metrics["mean"][k] for k in ("precision@0.5", "recall@0.5", "f1@0.5")},
                            "per_class": {c: {k: v[k] for k in ("ground_truth", "predictions", "precision@0.5",
                                                                  "recall@0.5", "f1@0.5")}
                                          for c, v in op_metrics["per_class"].items()},
                            "localization_recall@0.5": localization_recall(op_evals),
                            "counting": count_errors(op_evals)},
        "timing_ms": {"model_stages_mean": stage,
                      "end_to_end_mean": round(float(np.mean(e2e_ms)), 1),
                      "end_to_end_median": round(float(np.median(e2e_ms)), 1),
                      "end_to_end_p95": round(float(np.percentile(e2e_ms, 95)), 1),
                      "note": "end-to-end = read image + preprocess + inference + NMS + draw boxes, "
                              "batch size 1, CPU, after one warm-up image"},
        "error_examples": {k: min(len(v), n_examples) for k, v in groups.items()},
    }


# ----------------------------------------------------------------------------- report
def target_checks(ultra: dict, shared: dict) -> dict:
    recalls = {c: v["recall"] for c, v in ultra["per_class"].items()}
    e2e_s = shared["timing_ms"]["end_to_end_mean"] / 1000
    return {
        "mAP50 >= 0.80": {"value": ultra["mAP50"], "met": ultra["mAP50"] >= TARGETS["mAP50"]},
        "mAP50-95 >= 0.50": {"value": ultra["mAP50-95"], "met": ultra["mAP50-95"] >= TARGETS["mAP50-95"]},
        "recall >= 0.70 for every class": {"value": recalls,
                                           "met": all(r >= TARGETS["min_class_recall"] for r in recalls.values())},
        "end-to-end <= 1 s per image": {"value_seconds": round(e2e_s, 3), "met": e2e_s <= TARGETS["max_seconds_per_image"]},
    }


def write_markdown(report: dict, out_dir: Path) -> None:
    u, s = report["ultralytics"], report["shared"]
    lines = [f"# YOLO11n evaluation - `{report['split']}` split ({s['images']} images)\n",
             f"Weights: `{report['weights']}` (sha256 `{report['weights_sha256'][:12]}…`)  ",
             f"Hardware: {report['system']['cpu']}, {report['system'].get('ram_gb', '?')} GB RAM, CPU inference\n",
             "## Ultralytics validation metrics\n",
             "| Class | Precision | Recall | AP50 | AP50-95 |", "|---|---|---|---|---|"]
    for c, v in u["per_class"].items():
        lines.append(f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['AP50']:.3f} | {v['AP50-95']:.3f} |")
    lines.append(f"| **All** | **{u['precision']:.3f}** | **{u['recall']:.3f}** | **{u['mAP50']:.3f}** | **{u['mAP50-95']:.3f}** |")
    lines += ["", f"_{u['note']}._ Plots: `ultralytics/` (confusion matrix, PR / F1 / P / R curves).\n",
              "## Proposal targets\n", "| Target | Result | Met? |", "|---|---|---|"]
    for name, t in report["targets"].items():
        value = t.get("value", t.get("value_seconds"))
        if isinstance(value, dict):
            value = ", ".join(f"{k} {v:.2f}" for k, v in value.items())
        lines.append(f"| {name} | {value} | {'✅ yes' if t['met'] else '❌ no'} |")
    op = s["operating_point"]
    lines += ["", f"## At the app's operating point (confidence ≥ {op['confidence']}, NMS IoU {op['iou_nms']})\n",
              "Computed with the same code as the OpenCV baseline (`src/utils/metrics.py`).\n",
              "| Class | GT boxes | Predictions | Precision@0.5 | Recall@0.5 | F1@0.5 |", "|---|---|---|---|---|---|"]
    for c, v in op["per_class"].items():
        lines.append(f"| {c} | {v['ground_truth']} | {v['predictions']} | {v['precision@0.5']:.3f} | "
                     f"{v['recall@0.5']:.3f} | {v['f1@0.5']:.3f} |")
    m, cnt, t = op["mean"], op["counting"], s["timing_ms"]
    lines += [f"| **Mean** | | | **{m['precision@0.5']:.3f}** | **{m['recall@0.5']:.3f}** | **{m['f1@0.5']:.3f}** |", "",
              f"- AP with our implementation (all predictions): mAP50 **{s['ap_from_all_predictions']['mean']['mAP50']:.3f}**, "
              f"mAP50-95 **{s['ap_from_all_predictions']['mean']['mAP50-95']:.3f}**",
              f"- Localization recall (label ignored): {op['localization_recall@0.5']:.3f}",
              f"- Counting: mean absolute error {cnt['mean_absolute_count_error']} fruits/image, mean signed "
              f"{cnt['mean_signed_count_error']}, exact count {cnt['exact_count_rate']:.1%}",
              "", "## Processing time (this computer, CPU)\n",
              f"- Model stages (mean ms): preprocess {t['model_stages_mean']['preprocess']}, inference "
              f"{t['model_stages_mean']['inference']}, postprocess/NMS {t['model_stages_mean']['postprocess']}",
              f"- End-to-end per image: mean **{t['end_to_end_mean']} ms**, median {t['end_to_end_median']} ms, "
              f"95th percentile {t['end_to_end_p95']} ms",
              f"- {t['note']}", "",
              "## Error analysis\n",
              "`examples/`: `tp_*` = every fruit found with no false alarms; `fp_*` = images with the most false "
              "positives (red boxes); `fn_*` = images with the most missed fruits (magenta boxes).",
              f" Saved: {s['error_examples']}."]
    (out_dir / "EVALUATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--final", action="store_true", help="required for the locked test split")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--limit", type=int, default=None, help="our evaluation on the first N images only (quick check)")
    parser.add_argument("--examples", type=int, default=6, help="examples per category (tp/fp/fn)")
    parser.add_argument("--out", default=None, help="default outputs/yolo/eval_<split>")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        print("ERROR: the test split is locked. Use --split test --final only for the final evaluation.", file=sys.stderr)
        return 1

    from src.yolo.detector import YoloDetector
    from src.yolo.train import sha256, software_info

    cfg = load_config(args.config)
    try:
        detector = YoloDetector(args.weights, cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    out_dir = ensure_dir(args.out or f"outputs/yolo/eval_{args.split}")

    print(f"[1/2] Ultralytics validation on '{args.split}' ...")
    ultra = ultralytics_validation(detector, cfg, args.split, out_dir)
    print(f"[2/2] Shared evaluation, timing and error examples on '{args.split}' ...")
    shared = shared_evaluation(detector, cfg, args.split, out_dir, args.limit, args.examples)

    report = {"split": args.split, "weights": detector.weights.name, "weights_sha256": sha256(detector.weights),
              "system": system_info() | software_info(), "inference_settings": detector.params,
              "ultralytics": ultra, "shared": shared, "targets": target_checks(ultra, shared)}
    save_json(report, out_dir / "metrics.json")
    write_markdown(report, out_dir)
    print((out_dir / "EVALUATION.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
