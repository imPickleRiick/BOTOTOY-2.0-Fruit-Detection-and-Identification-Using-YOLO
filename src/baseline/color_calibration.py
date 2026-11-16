"""Measure the real color (HSV) and shape statistics of each fruit class on TRAINING images.

Why: the OpenCV baseline classifies fruits with hand-written HSV thresholds and shape rules.
Instead of guessing those numbers, we measure them from the ground-truth boxes of the
training split. Only the training split is used, so validation/test stay unbiased.

How: for up to N boxes per class, take the central 50% of the box (mostly fruit, little
background), convert to HSV, keep "colorful" pixels (S >= 50, V >= 40), and collect
hue/saturation/value histograms plus box aspect ratios.

Outputs (outputs/baseline/calibration/):
- color_profiles.json      hue/saturation/value percentiles per class
- hue_histograms.png       hue distribution per class (OpenCV hue scale 0-179)

Usage:
    python -m src.baseline.color_calibration
"""
from __future__ import annotations

import argparse
import random
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset_splitter import read_manifests
from src.utils.helpers import ensure_dir, load_config, resolve_path, save_json
from src.utils.image_utils import read_image, yolo_to_xyxy

# Hue is circular (red sits at both 0 and 179). Shifting by +15 makes red one continuous
# block (0..25) so percentiles are meaningful; we shift back when reporting.
HUE_SHIFT = 15


def _hue_percentiles(hues: np.ndarray) -> dict:
    shifted = (hues.astype(int) + HUE_SHIFT) % 180
    q = np.percentile(shifted, [5, 25, 50, 75, 95])
    return {f"p{p}": int((v - HUE_SHIFT) % 180) for p, v in zip((5, 25, 50, 75, 95), q)}


def calibrate(cfg: dict, boxes_per_class: int = 400, min_area: float = 0.01) -> dict:
    classes = cfg["classes"]
    rng = random.Random(cfg["project"]["seed"])
    raw_dir = resolve_path(cfg["dataset"]["raw_dir"])
    train = set(read_manifests(resolve_path(cfg["dataset"]["splits_dir"]))["train"])
    boxes = pd.read_csv(resolve_path(cfg["dataset"]["audit_dir"]) / "boxes.csv")
    boxes = boxes[boxes["image"].isin(train) & (boxes["area"] >= min_area)]

    profiles, hue_hist = {}, {}
    for cls in classes:
        rows = boxes[boxes.class_name == cls]
        rows = rows.iloc[rng.sample(range(len(rows)), min(boxes_per_class, len(rows)))]
        h_all, s_all, v_all, colorful_frac, aspect = [], [], [], [], []
        for r in rows.itertuples():
            img = read_image(raw_dir / r.image)
            if img is None:
                continue
            H, W = img.shape[:2]
            # central 50% of the box
            x1, y1, x2, y2 = yolo_to_xyxy(r.x_center, r.y_center, r.width * 0.5, r.height * 0.5, W, H)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            colorful = (hsv[:, 1] >= 50) & (hsv[:, 2] >= 40)
            colorful_frac.append(colorful.mean())
            h_all.append(hsv[colorful, 0])
            s_all.append(hsv[colorful, 1])
            v_all.append(hsv[colorful, 2])
            aspect.append(r.width / r.height)
        h, s, v = (np.concatenate(a) if a else np.array([]) for a in (h_all, s_all, v_all))
        hue_hist[cls] = np.bincount(h, minlength=180) / max(1, len(h))
        profiles[cls] = {
            "boxes_sampled": len(colorful_frac),
            "colorful_pixel_fraction_mean": round(float(np.mean(colorful_frac)), 3),
            "hue_percentiles": _hue_percentiles(h),
            "saturation_percentiles": {f"p{p}": int(x) for p, x in zip((5, 25, 50, 75, 95), np.percentile(s, [5, 25, 50, 75, 95]))},
            "value_percentiles": {f"p{p}": int(x) for p, x in zip((5, 25, 50, 75, 95), np.percentile(v, [5, 25, 50, 75, 95]))},
            "box_aspect_w_over_h_percentiles": {f"p{p}": round(float(x), 2) for p, x in zip((5, 25, 50, 75, 95), np.percentile(aspect, [5, 25, 50, 75, 95]))},
        }
        print(f"{cls:>10}: hue {profiles[cls]['hue_percentiles']}  sat p50={profiles[cls]['saturation_percentiles']['p50']}")
    return {"profiles": profiles, "hue_hist": hue_hist}


def plot_hue_histograms(hue_hist: dict, out_path) -> None:
    classes = list(hue_hist)
    fig, axes = plt.subplots(len(classes), 1, figsize=(9, 1.5 * len(classes)), sharex=True)
    # Color strip: what each OpenCV hue value looks like.
    strip = cv2.cvtColor(np.array([[[h, 255, 255] for h in range(180)]], dtype=np.uint8), cv2.COLOR_HSV2RGB)[0] / 255
    for ax, cls in zip(axes, classes):
        ax.bar(range(180), hue_hist[cls] * 100, width=1.0, color=strip)
        ax.set_ylabel(cls, rotation=0, ha="right", va="center", fontsize=9, color="#0b0b0b")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8, colors="#52514e")
    axes[-1].set_xlabel("OpenCV hue (0-179); bar color = that hue", color="#52514e")
    fig.suptitle("Hue of colorful pixels inside training boxes (% of pixels)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--boxes-per-class", type=int, default=400)
    args = parser.parse_args()
    cfg = load_config(args.config)
    try:
        result = calibrate(cfg, args.boxes_per_class)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}. Run the Phase 2 dataset tools first.", file=sys.stderr)
        return 1
    out = ensure_dir("outputs/baseline/calibration")
    save_json(result["profiles"], out / "color_profiles.json")
    plot_hue_histograms(result["hue_hist"], out / "hue_histograms.png")
    print(f"Saved calibration to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
