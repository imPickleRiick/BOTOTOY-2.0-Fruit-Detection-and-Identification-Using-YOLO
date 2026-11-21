"""Prepare the 30-class tropical fruit dataset (v2) for training, with leakage removed.

Source: Roboflow Universe "Fruit" by trucly, version 1 (CC BY 4.0), exported in YOLOv8 format:
https://universe.roboflow.com/trucly/fruit-5ix6x/dataset/1  -> unzip into data/raw/trucly_fruit_v1/

What this does
--------------
1. Reads every image/label pair of the original train / valid / test splits.
2. Fixes labels: coordinates are clipped to [0, 1]; boxes with no area are dropped.
3. Removes leakage, keeping the training split whole:
   - valid/test images whose Roboflow source photo (`<source>_jpg.rf.<hash>`) also appears in train,
   - valid/test images that are near-duplicates (64-bit dHash, Hamming <= 4) of a training image,
   - test images whose source or near-duplicate is in valid.
4. Writes the split lists to data/splits_fruit30/ (committed, so Colab rebuilds the exact same split),
   builds data/processed_fruit30/{train,val,test}/ and data/fruit30.yaml for Ultralytics.

Usage (from the project root):
    python -m src.data.prepare_fruit30                    # build the split from scratch
    python -m src.data.prepare_fruit30 --from-manifest    # rebuild folders from the committed lists
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from src.data.dataset_splitter import _link_or_copy
from src.data.dataset_validator import find_near_duplicate_pairs, source_id_from_name
from src.utils.image_utils import dhash, read_image

RAW = Path("data/raw/trucly_fruit_v1")
OUT = Path("data/processed_fruit30")
SPLITS = Path("data/splits_fruit30")
YAML_OUT = Path("data/fruit30.yaml")
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}
MAX_HAMMING = 4


def read_raw() -> tuple[list[str], list[dict]]:
    names = yaml.safe_load((RAW / "data.yaml").read_text(encoding="utf-8"))["names"]
    items = []
    for raw_split, split in SPLIT_MAP.items():
        for img in sorted((RAW / raw_split / "images").glob("*")):
            label = RAW / raw_split / "labels" / f"{img.stem}.txt"
            items.append({"rel": f"{raw_split}/images/{img.name}", "split": split, "stem": img.stem,
                          "label": label, "source": source_id_from_name(img.stem)})
    return names, items


def clean_label(path: Path) -> tuple[list[str], int]:
    """Clip coordinates to [0, 1] and drop empty boxes. Returns (lines, number of boxes changed)."""
    out, changed = [], 0
    if not path.is_file():
        return out, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        c = int(p[0])
        xc, yc, w, h = map(float, p[1:])
        x1, y1 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
        x2, y2 = min(1.0, xc + w / 2), min(1.0, yc + h / 2)
        if x2 - x1 <= 1e-4 or y2 - y1 <= 1e-4:
            changed += 1
            continue
        new = (c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        if any(abs(a - b) > 1e-6 for a, b in zip(new[1:], (xc, yc, w, h))):
            changed += 1
        out.append(f"{c} {new[1]:.6f} {new[2]:.6f} {new[3]:.6f} {new[4]:.6f}")
    return out, changed


def build_split(items: list[dict]) -> tuple[dict[str, list[str]], dict]:
    print("Hashing images for near-duplicate detection ...", flush=True)
    hashes = np.array([dhash(read_image(RAW / it["rel"])) for it in items], dtype=np.uint64)
    pairs = find_near_duplicate_pairs(hashes, MAX_HAMMING)

    sources_in = {s: {it["source"] for it in items if it["split"] == s} for s in ("train", "val")}
    removed: dict[str, str] = {}
    for it in items:  # same source photo in an earlier split
        if it["split"] == "val" and it["source"] in sources_in["train"]:
            removed[it["rel"]] = "source photo also in train"
        elif it["split"] == "test" and it["source"] in sources_in["train"]:
            removed[it["rel"]] = "source photo also in train"
        elif it["split"] == "test" and it["source"] in sources_in["val"]:
            removed[it["rel"]] = "source photo also in val"
    order = {"train": 0, "val": 1, "test": 2}
    cross = 0
    for i, j, d in pairs:  # near-duplicates: drop the image from the later split
        a, b = items[i], items[j]
        if a["split"] == b["split"]:
            continue
        cross += 1
        later = a if order[a["split"]] > order[b["split"]] else b
        removed.setdefault(later["rel"], f"near-duplicate (Hamming {d}) of a {min(a['split'], b['split'], key=order.get)} image")

    lists = {s: [it["rel"] for it in items if it["split"] == s and it["rel"] not in removed] for s in ("train", "val", "test")}
    info = {"near_duplicate_pairs_total": len(pairs), "near_duplicate_pairs_across_splits": cross,
            "removed": len(removed), "removed_by_reason": dict(Counter(r.split(" (")[0] for r in removed.values())),
            "removed_by_split": dict(Counter(next(it["split"] for it in items if it["rel"] == r) for r in removed)),
            "removed_images": removed}
    return lists, info


def materialize(lists: dict[str, list[str]], names: list[str]) -> dict:
    if OUT.exists():
        shutil.rmtree(OUT)
    stats = {"changed_boxes": 0, "per_split": {}}
    for split, rels in lists.items():
        (OUT / split / "images").mkdir(parents=True)
        (OUT / split / "labels").mkdir(parents=True)
        inst, n_img = Counter(), 0
        for rel in rels:
            src = RAW / rel
            _link_or_copy(src, OUT / split / "images" / src.name)
            lines, changed = clean_label(RAW / (rel.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"))
            stats["changed_boxes"] += changed
            (OUT / split / "labels" / f"{src.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            inst.update(names[int(l.split()[0])] for l in lines)
            n_img += 1
        stats["per_split"][split] = {"images": n_img, "boxes": sum(inst.values()),
                                     "boxes_per_class": {n: inst.get(n, 0) for n in names}}
    root = OUT.resolve().as_posix()
    YAML_OUT.write_text(yaml.safe_dump({"path": root, "train": "train/images", "val": "val/images",
                                        "test": "test/images", "nc": len(names), "names": names},
                                       sort_keys=False), encoding="utf-8")
    return stats


def verify(lists: dict[str, list[str]]) -> dict:
    src = {s: {source_id_from_name(Path(r).stem) for r in rels} for s, rels in lists.items()}
    files = {s: set(rels) for s, rels in lists.items()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    return {"file_overlap": {f"{a}-{b}": len(files[a] & files[b]) for a, b in pairs},
            "source_overlap": {f"{a}-{b}": len(src[a] & src[b]) for a, b in pairs}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-manifest", action="store_true", help="use the committed lists in data/splits_fruit30/")
    args = parser.parse_args()
    if not (RAW / "data.yaml").is_file():
        print(f"ERROR: {RAW}/data.yaml not found. Unzip the Roboflow export (Fruit.v1i.yolov8.zip) into {RAW}/.",
              file=sys.stderr)
        return 1
    names, items = read_raw()
    if args.from_manifest:
        lists = {s: [l for l in (SPLITS / f"{s}.txt").read_text(encoding="utf-8").splitlines() if l]
                 for s in ("train", "val", "test")}
        info = json.loads((SPLITS / "summary.json").read_text(encoding="utf-8"))["leakage_removal"]
    else:
        lists, info = build_split(items)
        SPLITS.mkdir(parents=True, exist_ok=True)
        for s, rels in lists.items():
            (SPLITS / f"{s}.txt").write_text("\n".join(rels) + "\n", encoding="utf-8")
    stats = materialize(lists, names)
    check = verify(lists)
    summary = {"source": "https://universe.roboflow.com/trucly/fruit-5ix6x/dataset/1", "license": "CC BY 4.0",
               "classes": names, "original_images": len(items), "leakage_removal": info,
               "label_fixes": {"boxes_clipped_or_dropped": stats["changed_boxes"]},
               "final": stats["per_split"], "verification": check}
    if not args.from_manifest:
        (SPLITS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Original images: {len(items)}; removed for leakage: {info['removed']} {info.get('removed_by_reason', {})}")
    for s, v in stats["per_split"].items():
        print(f"  {s:5s} {v['images']:5d} images {v['boxes']:6d} boxes")
    print(f"Boxes clipped/dropped: {stats['changed_boxes']}; overlap after cleaning: {check}")
    print(f"Wrote {OUT}/ and {YAML_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
