"""Dataset validator: checks every image and YOLO label file in the raw dataset.

What it checks
--------------
1. data.yaml exists and its class names/ids match config.yaml.
2. Every image has a label file and every label file has an image.
3. Every image can be decoded; records its width, height and channels.
4. Every label line has the YOLO format  "class_id x_center y_center width height":
   - class_id is an integer in [0, number_of_classes)
   - coordinates are numbers in [0, 1] (normalized), width/height > 0
   - flags boxes whose edges extend past the image border
   - flags duplicate boxes and empty label files (background images)
5. Exact duplicate images (same MD5 file hash) inside and across splits.
6. Near-duplicate images (perceptual dHash, small Hamming distance) across splits,
   which would mean data leakage between train / validation / test.
7. Source-image overlap across splits (Roboflow names augmented copies
   "<source>_jpg.rf.<hash>.jpg", so copies of one photo share a <source> id).

Outputs
-------
- data/processed/audit/images.csv          one row per image (cached for the other tools)
- data/processed/audit/boxes.csv           one row per bounding box
- data/processed/audit/near_duplicates.csv near-duplicate image pairs
- outputs/reports/dataset/validation_report.json
- outputs/reports/dataset/near_duplicate_pairs/*.jpg (side-by-side examples to inspect)

Usage (from the project root):
    python -m src.data.dataset_validator
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.helpers import ensure_dir, list_images, load_config, relpath_posix, resolve_path, save_json
from src.utils.image_utils import dhash, make_grid, read_image, write_image

RAW_SPLITS = ("train", "valid", "test")
COORD_TOLERANCE = 1e-6

# Issue codes. "Critical" issues make an image unusable for training/evaluation
# (Ultralytics itself would skip such images); warnings are reported only.
CRITICAL_ISSUES = {
    "unreadable_image", "missing_label", "malformed_line", "invalid_class",
    "non_positive_size", "out_of_range_coords",
}
WARNING_ISSUES = {"empty_label", "box_exceeds_image", "duplicate_box"}

_ROBOFLOW_NAME = re.compile(r"^(?P<source>.+?)_(jpg|jpeg|png|bmp)\.rf\.[0-9a-f]+$", re.IGNORECASE)


def source_id_from_name(stem: str) -> str:
    """Original photo id: 'abc123_jpg.rf.9f8e...' -> 'abc123'. Other names are returned unchanged."""
    match = _ROBOFLOW_NAME.match(stem)
    return match.group("source") if match else stem


def parse_label_file(label_path: Path, num_classes: int) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    """Parse one YOLO label file.

    Returns (boxes, issues). `boxes` holds every syntactically valid box
    (class_id, xc, yc, w, h); `issues` holds issue codes such as 'invalid_class'.
    """
    boxes: list[tuple[int, float, float, float, float]] = []
    issues: list[str] = []
    seen_lines: set[tuple] = set()

    text = label_path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return boxes, ["empty_label"]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:  # e.g. segmentation polygons or truncated lines
            issues.append("malformed_line")
            continue
        try:
            cls_float = float(parts[0])
            xc, yc, w, h = (float(v) for v in parts[1:])
        except ValueError:
            issues.append("malformed_line")
            continue
        if not cls_float.is_integer():
            issues.append("malformed_line")
            continue
        cls = int(cls_float)
        if not 0 <= cls < num_classes:
            issues.append("invalid_class")
            continue
        if w <= 0 or h <= 0:
            issues.append("non_positive_size")
            continue
        if any(v < -COORD_TOLERANCE or v > 1 + COORD_TOLERANCE for v in (xc, yc, w, h)):
            issues.append("out_of_range_coords")
            continue
        if (xc - w / 2 < -COORD_TOLERANCE or yc - h / 2 < -COORD_TOLERANCE
                or xc + w / 2 > 1 + COORD_TOLERANCE or yc + h / 2 > 1 + COORD_TOLERANCE):
            issues.append("box_exceeds_image")  # usable: training clips it to the border
        key = (cls, round(xc, 6), round(yc, 6), round(w, 6), round(h, 6))
        if key in seen_lines:
            issues.append("duplicate_box")
            continue
        seen_lines.add(key)
        boxes.append((cls, xc, yc, w, h))
    return boxes, issues


def check_data_yaml(raw_dir: Path, expected_classes: list[str]) -> dict:
    """Compare the dataset's data.yaml with the class list in config.yaml."""
    yaml_path = raw_dir / "data.yaml"
    result = {"path": f"{raw_dir.name}/data.yaml", "exists": yaml_path.is_file(), "ok": False}
    if not result["exists"]:
        result["error"] = "data.yaml not found"
        return result
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):  # YOLO also allows {0: name, 1: name}
        names = [names[k] for k in sorted(names)]
    result.update({"nc": data.get("nc"), "names": names,
                   "splits": {k: data.get(k) for k in ("train", "val", "test")}})
    result["ok"] = names == expected_classes and data.get("nc") == len(expected_classes)
    if not result["ok"]:
        result["error"] = f"class mismatch: dataset={names} config={expected_classes}"
    return result


def scan_split(raw_dir: Path, split: str, classes: list[str], extensions: list[str]) -> tuple[list[dict], list[dict], list[str]]:
    """Validate every image/label pair in one split.

    Returns (image_rows, box_rows, orphan_label_files).
    """
    img_dir, lbl_dir = raw_dir / split / "images", raw_dir / split / "labels"
    images = list_images(img_dir, extensions)
    label_files = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.is_dir() else {}
    image_stems = {p.stem for p in images}
    orphan_labels = sorted(relpath_posix(p, raw_dir) for s, p in label_files.items() if s not in image_stems)

    image_rows, box_rows = [], []
    for i, img_path in enumerate(images, 1):
        issues: list[str] = []
        image = read_image(img_path)
        file_bytes = img_path.read_bytes()
        row = {
            "split": split,
            "image": relpath_posix(img_path, raw_dir),
            "source_id": source_id_from_name(img_path.stem),
            "md5": hashlib.md5(file_bytes).hexdigest(),
            "file_kb": round(len(file_bytes) / 1024, 1),
            "width": None, "height": None, "channels": None, "dhash": None,
        }
        if image is None:
            issues.append("unreadable_image")
        else:
            row["height"], row["width"] = image.shape[:2]
            row["channels"] = image.shape[2] if image.ndim == 3 else 1
            row["dhash"] = f"{dhash(image):016x}"

        label_path = label_files.get(img_path.stem)
        boxes: list = []
        if label_path is None:
            issues.append("missing_label")
            row["label"] = None
        else:
            row["label"] = relpath_posix(label_path, raw_dir)
            boxes, label_issues = parse_label_file(label_path, len(classes))
            issues.extend(label_issues)

        row["num_boxes"] = len(boxes)
        row["issues"] = ";".join(sorted(set(issues)))
        row["critical"] = any(code in CRITICAL_ISSUES for code in issues)
        image_rows.append(row)

        for cls, xc, yc, w, h in boxes:
            box_rows.append({"split": split, "image": row["image"], "class_id": cls,
                             "class_name": classes[cls], "x_center": xc, "y_center": yc,
                             "width": w, "height": h, "area": w * h})
        if i % 1000 == 0 or i == len(images):
            print(f"  {split}: {i}/{len(images)} images checked", flush=True)
    return image_rows, box_rows, orphan_labels


def find_near_duplicate_pairs(hashes: np.ndarray, max_distance: int, chunk: int = 512) -> list[tuple[int, int, int]]:
    """All index pairs (i, j, distance) with i < j whose 64-bit hashes differ in <= max_distance bits.

    Uses XOR + popcount, computed in chunks to keep memory small.
    """
    pairs: list[tuple[int, int, int]] = []
    n = len(hashes)
    for start in range(0, n, chunk):
        block = hashes[start:start + chunk]
        distances = np.bitwise_count(block[:, None] ^ hashes[None, :])  # (chunk, n)
        rows, cols = np.nonzero(distances <= max_distance)
        for r, c in zip(rows, cols):
            i = start + int(r)
            if i < c:
                pairs.append((i, int(c), int(distances[r, c])))
    return pairs


def analyze_duplicates(images: pd.DataFrame, max_hamming: int) -> tuple[dict, pd.DataFrame]:
    """Find exact duplicates (MD5), near duplicates (dHash) and source overlap across splits."""
    summary: dict = {}

    # --- Exact duplicates (identical files) ---
    groups = images.groupby("md5")["split"].agg(list)
    dup_groups = groups[groups.map(len) > 1]
    summary["exact_duplicate_groups"] = int(len(dup_groups))
    summary["exact_duplicate_groups_cross_split"] = int(sum(len(set(s)) > 1 for s in dup_groups))

    # --- Source photo overlap (augmented copies must stay in one split) ---
    src_splits = images.groupby("source_id")["split"].agg(lambda s: sorted(set(s)))
    overlap = src_splits[src_splits.map(len) > 1]
    summary["unique_source_images"] = int(len(src_splits))
    summary["sources_in_multiple_splits"] = int(len(overlap))
    summary["unique_sources_per_split"] = {s: int(images.loc[images.split == s, "source_id"].nunique())
                                           for s in RAW_SPLITS}

    # --- Near duplicates (perceptual hash) ---
    valid = images[images["dhash"].notna()].reset_index(drop=True)
    hashes = np.array([int(h, 16) for h in valid["dhash"]], dtype=np.uint64)
    pairs = find_near_duplicate_pairs(hashes, max_hamming)
    rows = []
    for i, j, dist in pairs:
        a, b = valid.iloc[i], valid.iloc[j]
        rows.append({"image_a": a["image"], "split_a": a["split"], "image_b": b["image"], "split_b": b["split"],
                     "hamming": dist, "same_source": a["source_id"] == b["source_id"],
                     "cross_split": a["split"] != b["split"]})
    near = pd.DataFrame(rows, columns=["image_a", "split_a", "image_b", "split_b", "hamming", "same_source", "cross_split"])
    summary["near_duplicate_max_hamming"] = max_hamming
    summary["near_duplicate_pairs_total"] = int(len(near))
    summary["near_duplicate_pairs_same_source"] = int(near["same_source"].sum()) if len(near) else 0
    cross = near[near["cross_split"]] if len(near) else near
    summary["near_duplicate_pairs_cross_split"] = int(len(cross))
    summary["near_duplicate_pairs_cross_split_by_splits"] = (
        dict(Counter("-".join(sorted((r.split_a, r.split_b))) for r in cross.itertuples())) if len(cross) else {})
    return summary, near


def save_pair_examples(near: pd.DataFrame, raw_dir: Path, out_dir: Path, limit: int = 12) -> list[str]:
    """Save side-by-side images of cross-split near-duplicate pairs so a human can confirm them."""
    out_dir = ensure_dir(out_dir)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    cross = near[near["cross_split"]].sort_values("hamming") if len(near) else near
    saved = []
    for k, row in enumerate(cross.head(limit).itertuples(), 1):
        a, b = read_image(raw_dir / row.image_a), read_image(raw_dir / row.image_b)
        if a is None or b is None:
            continue
        path = out_dir / f"pair_{k:02d}_{row.split_a}-{row.split_b}_d{row.hamming}.jpg"
        write_image(path, make_grid([a, b], cols=2, cell_size=320))
        saved.append(path.name)
    return saved


def validate_dataset(config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    ds = cfg["dataset"]
    classes: list[str] = cfg["classes"]
    raw_dir = resolve_path(ds["raw_dir"])
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found at {raw_dir}. Run: python -m src.data.download_dataset")

    started = time.time()
    # Store the project-relative path so committed reports contain no machine-specific paths.
    report: dict = {"raw_dir": ds["raw_dir"], "classes": classes}
    report["data_yaml"] = check_data_yaml(raw_dir, classes)
    print(f"data.yaml check: {'OK' if report['data_yaml']['ok'] else report['data_yaml'].get('error')}")

    all_images, all_boxes, orphans = [], [], []
    for split in RAW_SPLITS:
        imgs, boxes, orph = scan_split(raw_dir, split, classes, ds["image_extensions"])
        all_images += imgs
        all_boxes += boxes
        orphans += orph
    images = pd.DataFrame(all_images)
    boxes = pd.DataFrame(all_boxes)

    # --- Per-split integrity summary ---
    issue_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for split, grp in images.groupby("split"):
        counter = Counter(code for s in grp["issues"] if s for code in s.split(";"))
        issue_counts[split] = dict(sorted(counter.items()))
    report["splits"] = {
        s: {"images": int((images.split == s).sum()),
            "boxes": int((boxes.split == s).sum()) if len(boxes) else 0,
            "critical_images": int(images.loc[images.split == s, "critical"].sum()),
            "issues_images_affected": issue_counts.get(s, {})}
        for s in RAW_SPLITS
    }
    report["totals"] = {"images": int(len(images)), "boxes": int(len(boxes)),
                        "critical_images": int(images["critical"].sum()),
                        "orphan_label_files": len(orphans)}
    report["orphan_label_files"] = orphans[:50]
    report["critical_examples"] = images.loc[images["critical"], ["image", "issues"]].head(30).to_dict("records")

    # --- Image properties ---
    readable = images[images["width"].notna()]
    report["image_properties"] = {
        "formats": dict(Counter(Path(p).suffix.lower() for p in images["image"])),
        "sizes": {f"{int(w)}x{int(h)}": int(n) for (w, h), n in readable.groupby(["width", "height"]).size().items()},
        "channels": {str(int(k)): int(v) for k, v in readable["channels"].value_counts().items()},
        "file_kb_mean": round(float(images["file_kb"].mean()), 1),
    }

    # --- Duplicates and leakage ---
    max_ham = ds["validation"]["near_duplicate_max_hamming"]
    print(f"Checking duplicates and near-duplicates (dHash, Hamming <= {max_ham}) ...")
    report["duplicates"], near = analyze_duplicates(images, max_ham)

    # --- Save outputs ---
    audit_dir = ensure_dir(ds["audit_dir"])
    images.to_csv(audit_dir / "images.csv", index=False)
    boxes.to_csv(audit_dir / "boxes.csv", index=False)
    near.to_csv(audit_dir / "near_duplicates.csv", index=False)
    report_dir = ensure_dir(cfg["outputs"]["dataset_report_dir"])
    report["near_duplicate_examples"] = save_pair_examples(near, raw_dir, report_dir / "near_duplicate_pairs")
    report["elapsed_seconds"] = round(time.time() - started, 1)
    save_json(report, report_dir / "validation_report.json")
    return report


def print_summary(report: dict) -> None:
    print("\n=== VALIDATION SUMMARY ===")
    for split, s in report["splits"].items():
        print(f"{split:>6}: {s['images']:5d} images, {s['boxes']:6d} boxes, "
              f"{s['critical_images']} critical, issues={s['issues_images_affected']}")
    print(f"Totals: {report['totals']}")
    print(f"Image sizes: {report['image_properties']['sizes']}")
    d = report["duplicates"]
    print(f"Unique source photos: {d['unique_source_images']} {d['unique_sources_per_split']}, "
          f"sources in >1 split: {d['sources_in_multiple_splits']}")
    print(f"Exact duplicate groups: {d['exact_duplicate_groups']} "
          f"(cross-split: {d['exact_duplicate_groups_cross_split']})")
    print(f"Near-duplicate pairs: {d['near_duplicate_pairs_total']} "
          f"(same source: {d['near_duplicate_pairs_same_source']}, cross-split: "
          f"{d['near_duplicate_pairs_cross_split']} {d['near_duplicate_pairs_cross_split_by_splits']})")
    print(f"Elapsed: {report['elapsed_seconds']} s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()
    try:
        report = validate_dataset(args.config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print_summary(report)
    return 0 if report["data_yaml"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
