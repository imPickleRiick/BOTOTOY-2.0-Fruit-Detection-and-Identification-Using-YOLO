"""Baseline improvement log (ablation study) on the VALIDATION split.

Each step adds ONE change on top of the previous steps and is evaluated on the full
validation split with the same metrics as YOLO. This documents *why* each rule or
threshold in config.yaml exists, and shows that tuning used validation data only
(the test split stays locked).

Usage:
    python -m src.baseline.ablation            # ~1 minute per step on a laptop CPU
"""
from __future__ import annotations

import copy
import sys

from src.baseline.evaluate_baseline import run
from src.utils.helpers import ensure_dir, load_config, save_json

# (name, explanation, overrides of config["baseline"] applied cumulatively)
STEPS = [
    ("A. Initial rules", "color masks + shape rules, every region >= 0.2% of the image", {}),
    ("B. Minimum region size 1%", "regions under 1% of the image were ~98% false positives",
     {"min_area_fraction": 0.01}),
    ("C. Reject irregular green regions", "green 'grape' regions were mostly leaves/grass (0.4% correct)",
     {"rules": {"green_grapes_enabled": False}}),
    ("D. Size-aware rule score", "rank larger regions higher: they are far more often real fruits",
     {"size_reference_fraction": 0.05}),
    ("E. Watershed split of touching fruits", "split blobs of touching round fruits (warm/red)",
     {"split_touching": {"enabled": True}}),
]


def deep_update(base: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def main() -> int:
    base_cfg = load_config()
    cfg = copy.deepcopy(base_cfg)
    rows = []
    for name, why, overrides in STEPS:
        deep_update(cfg["baseline"], overrides)
        print(f"\n=== {name} ===")
        try:
            m = run("val", cfg, limit=None, brightness=1.0, n_examples=0)["metrics"]
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        rows.append({"step": name, "why": why, "overrides": overrides, **m["mean"],
                     "localization_recall": m["localization_recall@0.5"],
                     "mean_abs_count_error": m["counting"]["mean_absolute_count_error"],
                     "predictions": sum(c["predictions"] for c in m["per_class"].values()),
                     "detector_ms": m["timing_ms"]["detector_mean"]})
        print({k: rows[-1][k] for k in ("mAP50", "precision@0.5", "recall@0.5", "localization_recall")})

    out = ensure_dir("outputs/baseline")
    save_json(rows, out / "ablation_val.json")
    lines = ["# OpenCV baseline - improvement log (validation split, 686 images)\n",
             "Each step keeps all previous changes. Metrics use IoU 0.5 and the same AP formula as YOLO.\n",
             "| Step | Why | Predictions | Precision | Recall | mAP50 | mAP50-95 | Localization recall | Count MAE | ms/img |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['step']} | {r['why']} | {r['predictions']} | {r['precision@0.5']:.3f} | {r['recall@0.5']:.3f} | "
                     f"{r['mAP50']:.3f} | {r['mAP50-95']:.3f} | {r['localization_recall']:.3f} | "
                     f"{r['mean_abs_count_error']} | {r['detector_ms']} |")
    (out / "ablation_val.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
