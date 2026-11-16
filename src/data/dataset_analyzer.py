"""Dataset analyzer: statistics, charts, annotated samples and the final dataset audit report.

Runs AFTER dataset_validator.py and dataset_splitter.py. Uses only the final split
(data/splits/*.txt). Annotated samples are drawn from the TRAINING split only, so the
test set stays "locked" (nobody looks at it while developing the model).

Outputs (outputs/reports/dataset/):
- class_instances_by_split.png, class_share_by_split.png,
  boxes_per_image.png, box_size_distribution.png
- samples/sample_XX_<class>.jpg (24 annotated training images) and samples_grid.jpg
- dataset_statistics.json
- DATASET_AUDIT.md  (the human-readable audit report)

Usage (from the project root):
    python -m src.data.dataset_analyzer
    python -m src.data.dataset_analyzer --skip-yolo-check   # skip the Ultralytics loader test
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset_splitter import read_manifests
from src.utils.helpers import ensure_dir, load_config, load_json, resolve_path, save_json
from src.utils.image_utils import CLASS_COLORS_BGR, draw_box, make_grid, read_image, write_image, yolo_to_xyxy

SPLITS = ("train", "val", "test")
# Chart colors (validated categorical palette, slots 1-3) and neutral ink.
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
SINGLE_SERIES = "#2a78d6"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


# ----------------------------------------------------------------------------- statistics
def load_tables(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    audit_dir = resolve_path(cfg["dataset"]["audit_dir"])
    images = pd.read_csv(audit_dir / "images.csv")
    boxes = pd.read_csv(audit_dir / "boxes.csv")
    manifests = read_manifests(resolve_path(cfg["dataset"]["splits_dir"]))
    split_of = {img: s for s, imgs in manifests.items() for img in imgs}
    images["final_split"] = images["image"].map(split_of)
    boxes["final_split"] = boxes["image"].map(split_of)
    # Keep only images that are part of the final split.
    return images[images["final_split"].notna()].copy(), boxes[boxes["final_split"].notna()].copy(), manifests


def compute_statistics(images: pd.DataFrame, boxes: pd.DataFrame, classes: list[str]) -> dict:
    stats: dict = {"per_split": {}, "per_class": {}}
    for s in SPLITS:
        im, bx = images[images.final_split == s], boxes[boxes.final_split == s]
        per_img = bx.groupby("image").size().reindex(im["image"], fill_value=0)
        stats["per_split"][s] = {
            "images": int(len(im)),
            "unique_sources": int(im["source_id"].nunique()),
            "boxes": int(len(bx)),
            "boxes_per_image_mean": round(float(per_img.mean()), 2),
            "boxes_per_image_median": float(per_img.median()),
            "boxes_per_image_max": int(per_img.max()),
            "instances_per_class": {c: int((bx.class_name == c).sum()) for c in classes},
            "images_per_class": {c: int(bx.loc[bx.class_name == c, "image"].nunique()) for c in classes},
        }
    total = len(boxes)
    for c in classes:
        bx = boxes[boxes.class_name == c]
        side = np.sqrt(bx["area"])  # relative box side length (fraction of image side)
        stats["per_class"][c] = {
            "instances": int(len(bx)),
            "instance_share": round(len(bx) / total, 4) if total else 0,
            "images": int(bx["image"].nunique()),
            "median_relative_side": round(float(side.median()), 3),
            "small_boxes_pct": round(float((bx["area"] < 0.01).mean() * 100), 1),  # < 1% of image area
        }
    multi = boxes.groupby("image")["class_id"].nunique()
    stats["images_with_multiple_classes"] = int((multi > 1).sum())
    stats["total_images"] = int(len(images))
    stats["total_boxes"] = int(total)
    counts = [stats["per_class"][c]["instances"] for c in classes]
    stats["imbalance_ratio_max_over_min"] = round(max(counts) / max(1, min(counts)), 2)
    return stats


# ----------------------------------------------------------------------------- charts
def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> str:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path.name


def plot_class_instances(stats: dict, classes: list[str], out: Path) -> str:
    """Small multiples: one panel per split (different scales, so no shared axis)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, s in zip(axes, SPLITS):
        values = [stats["per_split"][s]["instances_per_class"][c] for c in classes]
        bars = ax.bar(classes, values, color=SINGLE_SERIES, width=0.6)
        ax.bar_label(bars, labels=[f"{v:,}" for v in values], fontsize=8, color=MUTED, padding=2)
        ax.set_title(f"{s}  ({stats['per_split'][s]['images']:,} images)", fontsize=11, color=INK, loc="left")
        _style_axes(ax)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Bounding-box instances", color=MUTED)
    fig.suptitle("Instances per class in each split", color=INK, fontsize=12, x=0.01, ha="left")
    return _save(fig, out / "class_instances_by_split.png")


def plot_class_share(stats: dict, classes: list[str], out: Path) -> str:
    """Share of instances per class, per split - shows whether the splits are balanced alike."""
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(classes))
    width = 0.26
    for k, s in enumerate(SPLITS):
        counts = np.array([stats["per_split"][s]["instances_per_class"][c] for c in classes], dtype=float)
        share = counts / counts.sum() * 100
        ax.bar(x + (k - 1) * (width + 0.01), share, width, color=SPLIT_COLORS[s], label=s)
    ax.set_xticks(x, classes)
    ax.set_ylabel("Share of split's instances (%)", color=MUTED)
    ax.set_title("Class mix per split (instances, %)", color=INK, fontsize=12, loc="left")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncols=3, loc="upper left")
    return _save(fig, out / "class_share_by_split.png")


def plot_boxes_per_image(images: pd.DataFrame, out: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    counts = images["num_boxes"].clip(upper=30)
    ax.hist(counts, bins=np.arange(0.5, 31.5, 1), color=SINGLE_SERIES, rwidth=0.85)
    ax.set_xlabel("Boxes per image (30 = 30 or more)", color=MUTED)
    ax.set_ylabel("Images", color=MUTED)
    ax.set_title("Objects per image (all final splits)", color=INK, fontsize=12, loc="left")
    _style_axes(ax)
    return _save(fig, out / "boxes_per_image.png")


def plot_box_sizes(boxes: pd.DataFrame, classes: list[str], out: Path) -> str:
    """Relative box size per class: sqrt(area) = side length as a fraction of the image side."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    data = [np.sqrt(boxes.loc[boxes.class_name == c, "area"]) for c in classes]
    parts = ax.boxplot(data, tick_labels=classes, showfliers=False, widths=0.5, patch_artist=True)
    for patch in parts["boxes"]:
        patch.set(facecolor="#cde2fb", edgecolor=SINGLE_SERIES)
    for key in ("whiskers", "caps"):
        for line in parts[key]:
            line.set_color(MUTED)
    for line in parts["medians"]:
        line.set(color=SINGLE_SERIES, linewidth=2)
    ax.set_ylabel("sqrt(box area / image area)", color=MUTED)
    ax.set_title("Relative object size per class (outliers hidden)", color=INK, fontsize=12, loc="left")
    _style_axes(ax)
    return _save(fig, out / "box_size_distribution.png")


# ----------------------------------------------------------------------------- samples
def save_annotated_samples(images: pd.DataFrame, boxes: pd.DataFrame, cfg: dict, out: Path) -> list[str]:
    """Draw ground-truth boxes on training images: an equal number per class, fixed seed."""
    classes = cfg["classes"]
    n_total = cfg["outputs"]["num_sample_visualizations"]
    per_class = max(1, n_total // len(classes))
    rng = random.Random(cfg["project"]["seed"])
    raw_dir = resolve_path(cfg["dataset"]["raw_dir"])
    sample_dir = ensure_dir(out / "samples")
    for old in sample_dir.glob("*.jpg"):
        old.unlink()

    train_boxes = boxes[boxes.final_split == "train"]
    chosen: list[tuple[str, str]] = []
    used: set[str] = set()
    for c in classes:
        candidates = sorted(set(train_boxes.loc[train_boxes.class_name == c, "image"]) - used)
        for img in rng.sample(candidates, min(per_class, len(candidates))):
            chosen.append((c, img))
            used.add(img)

    tiles, names = [], []
    for k, (cls, rel) in enumerate(chosen, 1):
        image = read_image(raw_dir / rel)
        if image is None:
            continue
        h, w = image.shape[:2]
        for b in train_boxes[train_boxes.image == rel].itertuples():
            xyxy = yolo_to_xyxy(b.x_center, b.y_center, b.width, b.height, w, h)
            draw_box(image, xyxy, b.class_name, CLASS_COLORS_BGR.get(b.class_name, (200, 200, 200)))
        name = f"sample_{k:02d}_{cls.lower()}.jpg"
        write_image(sample_dir / name, image)
        tiles.append(image)
        names.append(name)
    if tiles:
        write_image(out / "samples_grid.jpg", make_grid(tiles, cols=6, cell_size=300))
    return names


# ----------------------------------------------------------------------------- YOLO loader check
def check_ultralytics_loading(cfg: dict) -> dict:
    """Confirm Ultralytics can parse data.yaml and load every split's labels without errors."""
    result: dict = {"ok": False}
    try:
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.data.utils import check_det_dataset

        data = check_det_dataset(str(resolve_path(cfg["dataset"]["data_yaml"])))
        result["names"] = list(data["names"].values())
        result["splits"] = {}
        for s in SPLITS:
            ds = YOLODataset(img_path=data[s], data=data, task="detect", augment=False, imgsz=640)
            sample = ds[0]  # run the real loading + transform pipeline on one image
            result["splits"][s] = {"images_loaded": len(ds.labels),
                                   "boxes_loaded": int(sum(len(lb["cls"]) for lb in ds.labels)),
                                   "sample_tensor_shape": list(sample["img"].shape)}
        result["ok"] = True
    except Exception as exc:  # report the failure instead of hiding it
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# ----------------------------------------------------------------------------- report
def _table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def write_markdown_report(cfg: dict, validation: dict, split_summary: dict, stats: dict,
                          charts: list[str], samples: list[str], yolo_check: dict, out: Path) -> Path:
    classes = cfg["classes"]
    ds = cfg["dataset"]
    dup = validation["duplicates"]
    md: list[str] = []
    md.append("# Dataset Audit Report - BOTOTOY 2.0\n")
    md.append(f"_Generated by `src/data/dataset_analyzer.py` on {date.today().isoformat()}. "
              "All numbers below are computed from the downloaded files._\n")

    md.append("## 1. Source and license\n")
    md.append(f"- **Source used:** GitHub mirror `{ds['github_repo']}` at commit `{ds['github_commit']}`")
    md.append(f"- **Original source:** Kaggle - {ds['kaggle_url']} (by Lakshay Tyagi)")
    md.append(f"- **License:** {ds['license']}. The Kaggle page's own license field should be "
              "cross-checked by the team when citing.")
    md.append("- **Pre-processing applied by the dataset author (Roboflow export):** auto-orientation, "
              "resize to 640x640 (stretch). The training split also contains Roboflow-augmented copies.\n")

    md.append("## 2. Original structure and data.yaml\n")
    y = validation["data_yaml"]
    md.append(f"- data.yaml: `nc={y.get('nc')}`, names = {y.get('names')} -> "
              f"**{'matches' if y['ok'] else 'DOES NOT match'}** config.yaml")
    md.append(f"- Class ids: " + ", ".join(f"{i} = {c}" for i, c in enumerate(classes)))
    rows = [[s, v["images"], v["boxes"], dup["unique_sources_per_split"][s]] for s, v in validation["splits"].items()]
    md.append("\n" + _table(["Original split", "Images", "Boxes", "Unique source photos"], rows) + "\n")

    md.append("## 3. Integrity checks\n")
    t = validation["totals"]
    p = validation["image_properties"]
    md.append(_table(["Check", "Result"], [
        ["Images checked / boxes parsed", f"{t['images']} / {t['boxes']}"],
        ["Images without a label file", sum(v["issues_images_affected"].get("missing_label", 0) for v in validation["splits"].values())],
        ["Label files without an image", t["orphan_label_files"]],
        ["Unreadable / corrupt images", sum(v["issues_images_affected"].get("unreadable_image", 0) for v in validation["splits"].values())],
        ["Empty label files (background images)", sum(v["issues_images_affected"].get("empty_label", 0) for v in validation["splits"].values())],
        ["Invalid class ids", sum(v["issues_images_affected"].get("invalid_class", 0) for v in validation["splits"].values())],
        ["Malformed label lines", sum(v["issues_images_affected"].get("malformed_line", 0) for v in validation["splits"].values())],
        ["Coordinates outside [0, 1]", sum(v["issues_images_affected"].get("out_of_range_coords", 0) for v in validation["splits"].values())],
        ["Boxes extending past the image border", sum(v["issues_images_affected"].get("box_exceeds_image", 0) for v in validation["splits"].values())],
        ["Images with duplicate box lines (warning)", sum(v["issues_images_affected"].get("duplicate_box", 0) for v in validation["splits"].values())],
        ["Images with a critical issue", t["critical_images"]],
        ["Image formats", p["formats"]],
        ["Image sizes", p["sizes"]],
        ["Color channels", p["channels"]],
    ]))
    md.append("\nDuplicate box lines are harmless: Ultralytics removes duplicate labels automatically.\n")

    md.append("## 4. Duplicates and data leakage\n")
    md.append(f"- Unique source photos (from Roboflow file names): **{dup['unique_source_images']}**; "
              f"source photos appearing in more than one original split: **{dup['sources_in_multiple_splits']}**")
    md.append(f"- Exact duplicate files (MD5): {dup['exact_duplicate_groups']} groups, "
              f"{dup['exact_duplicate_groups_cross_split']} across splits (all within-split groups are "
              "identical augmented copies of the same training photo)")
    md.append(f"- Near-duplicates (64-bit dHash, Hamming distance <= {dup['near_duplicate_max_hamming']}): "
              f"{dup['near_duplicate_pairs_total']} pairs, of which {dup['near_duplicate_pairs_same_source']} are "
              f"augmented copies of the same photo and **{dup['near_duplicate_pairs_cross_split']} cross original "
              f"splits** {dup['near_duplicate_pairs_cross_split_by_splits']}")
    md.append("- Cross-split pairs were inspected visually (`near_duplicate_pairs/`): at least one is the same "
              "photo with different color processing (true leakage); others are different photos from the same "
              "photo series. **All validation images involved were excluded** from the final val/test sets:\n")
    for e in split_summary.get("excluded", []):
        md.append(f"  - `{e['image']}` - {e['reason']}")
    md.append("")

    md.append("## 5. Final train / validation / test split (Option A)\n")
    md.append("The original training split is kept unchanged (it contains every augmented copy, so copies can never "
              "leak into evaluation). The un-augmented original valid + test images were pooled, leaking images "
              "removed, near-duplicates grouped, and groups assigned ~50/50 to val and test, stratified by each "
              f"group's dominant class (seed {split_summary.get('seed')}).\n")
    rows = [[s, v["images"], f"{v['image_share']:.1%}", v["unique_sources"], f"{v['unique_source_share']:.1%}", v["boxes"]]
            for s, v in split_summary["splits"].items()]
    md.append(_table(["Split", "Images", "Image share", "Unique photos", "Unique-photo share", "Boxes"], rows))
    v = split_summary["verification"]
    md.append(f"\n**Leakage verification on the final split:** file overlap {v['file_overlap']}, "
              f"MD5 overlap {v['md5_overlap']}, source-photo overlap {v['source_overlap']}, "
              f"near-duplicate pairs across splits = {v['near_duplicate_pairs_across_splits']} -> "
              f"**{'LEAKAGE-FREE' if v['leakage_free'] else 'LEAKAGE FOUND'}**\n")
    md.append("The **test split is locked**: it is used only once, for the final evaluation in Phase 4. Model "
              "selection and threshold tuning use the val split.\n")

    md.append("## 6. Class distribution\n")
    rows = []
    for c in classes:
        pc = stats["per_class"][c]
        rows.append([c] + [stats["per_split"][s]["instances_per_class"][c] for s in SPLITS]
                    + [stats["per_split"][s]["images_per_class"][c] for s in SPLITS]
                    + [f"{pc['instance_share']:.1%}", pc["median_relative_side"], f"{pc['small_boxes_pct']}%"])
    md.append(_table(["Class", "Inst. train", "Inst. val", "Inst. test", "Img. train", "Img. val", "Img. test",
                      "Share of all inst.", "Median rel. side", "Boxes < 1% area"], rows))
    md.append(f"\n- Imbalance ratio (most / least frequent class instances): **{stats['imbalance_ratio_max_over_min']}x**")
    md.append(f"- Images containing more than one fruit class: {stats['images_with_multiple_classes']}")
    rows = [[s, d["boxes_per_image_mean"], d["boxes_per_image_median"], d["boxes_per_image_max"]]
            for s, d in stats["per_split"].items()]
    md.append("\n" + _table(["Split", "Mean boxes/image", "Median", "Max"], rows) + "\n")

    md.append("## 7. Figures\n")
    for name in charts:
        md.append(f"![{name}]({name})\n")
    md.append(f"### Annotated training samples ({len(samples)} images, {len(samples) // len(classes)} per class)\n")
    md.append("![samples](samples_grid.jpg)\n")
    md.append("Individual files: `samples/`. Samples come from the training split only (test set stays locked).\n")

    md.append("## 8. Ultralytics loader check\n")
    if yolo_check.get("ok"):
        rows = [[s, d["images_loaded"], d["boxes_loaded"], d["sample_tensor_shape"]] for s, d in yolo_check["splits"].items()]
        md.append("Ultralytics successfully parsed `data/data.yaml` and loaded every split:\n")
        md.append(_table(["Split", "Images loaded", "Boxes loaded", "Sample tensor (C,H,W)"], rows) + "\n")
    else:
        md.append(f"Not run or failed: `{yolo_check.get('error', 'skipped')}`\n")

    md.append("## 9. Characteristics and limitations\n")
    pc = stats["per_class"]
    most = max(classes, key=lambda c: pc[c]["instances"])
    least = min(classes, key=lambda c: pc[c]["instances"])
    md += [
        f"- **Class imbalance:** {most} has {pc[most]['instances']:,} instances vs {least} with "
        f"{pc[least]['instances']:,} ({stats['imbalance_ratio_max_over_min']}x). Expect lower recall on rare classes; "
        "report per-class AP, not only the mean.",
        "- **Augmented training copies:** 7,108 training images come from only 3,200 photos, so the effective "
        "training-set diversity is smaller than the image count suggests.",
        "- **Stretched images:** all images were resized to 640x640 without keeping the aspect ratio, so fruit "
        "shapes can be distorted. This matters for the OpenCV baseline's shape features (aspect ratio, circularity).",
        "- **Dense, small objects:** grapes and oranges often appear in clusters of many small boxes; overlapping "
        "fruits are hard for contour-based methods and for counting.",
        "- **Near-duplicate detection is limited:** dHash catches re-encoded or re-colored copies of the same image, "
        "not different shots of the same scene, so some scene-level similarity between splits may remain.",
        "- **Label quality:** automated checks found no invalid labels, but box accuracy and missing annotations "
        "were only spot-checked through the sample visualizations, not exhaustively. The spot check of the "
        "24 samples found:",
        "  - **Group boxes:** some boxes cover a whole group of fruits rather than one fruit (e.g. "
        "`samples/sample_05_banana.jpg`: one box spans the entire banana display), and some fruits in a group are "
        "left unlabeled. Consequently, *fruit count = number of detected boxes*, which can under-count dense piles. "
        "This must be stated when presenting counting results.",
        "  - **Varied image types:** paintings, illustrations and framed photos appear alongside real photos "
        "(e.g. samples 04, 09, 12).",
        "  - **Possible class noise:** yellow citrus that may be lemons is labeled Orange "
        "(`samples/sample_16_orange.jpg`).",
        "",
        "## 10. How to reproduce\n",
        "```bash",
        "python -m src.data.download_dataset      # download pinned dataset to data/raw/",
        "python -m src.data.dataset_validator     # integrity + duplicate checks",
        "python -m src.data.dataset_splitter --from-manifest   # rebuild the committed split",
        "python -m src.data.dataset_analyzer      # statistics, figures, this report",
        "```",
    ]
    path = out / "DATASET_AUDIT.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--skip-yolo-check", action="store_true", help="skip the Ultralytics loader test")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg["outputs"]["dataset_report_dir"])
    try:
        validation = load_json(out / "validation_report.json")
        split_summary = load_json(resolve_path(cfg["dataset"]["splits_dir"]) / "split_summary.json")
        images, boxes, _ = load_tables(cfg)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}\nRun dataset_validator and dataset_splitter first.", file=sys.stderr)
        return 1

    classes = cfg["classes"]
    stats = compute_statistics(images, boxes, classes)
    charts = [plot_class_instances(stats, classes, out), plot_class_share(stats, classes, out),
              plot_boxes_per_image(images, out), plot_box_sizes(boxes, classes, out)]
    samples = save_annotated_samples(images, boxes, cfg, out)
    print(f"Saved {len(charts)} charts and {len(samples)} annotated samples to {out}")

    yolo_check = {"error": "skipped (--skip-yolo-check)"}
    if not args.skip_yolo_check:
        print("Checking that Ultralytics can load the dataset ...")
        yolo_check = check_ultralytics_loading(cfg)
        print(f"  Ultralytics loader: {'OK' if yolo_check.get('ok') else yolo_check.get('error')}")
    stats["ultralytics_loader_check"] = yolo_check
    save_json(stats, out / "dataset_statistics.json")
    report = write_markdown_report(cfg, validation, split_summary, stats, charts, samples, yolo_check, out)
    print(f"Audit report: {report}")
    return 0 if (yolo_check.get("ok") or args.skip_yolo_check) else 1


if __name__ == "__main__":
    sys.exit(main())
