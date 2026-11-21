"""Prepare the COMBINED 31-fruit dataset: v1 (6 fruits) + v2 (30 tropical fruits), leakage-free.

Why: the 30-fruit dataset has no apples or grapes, which are common in Philippine markets, and the
30-fruit model called apples "star apple" or "guava". Combining both datasets gives one model that knows
both groups.

Class list (31): the 30 tropical classes with "tangerines" merged into "orange" (the v1 dataset labels
mandarins as Orange, so keeping two citrus classes would teach contradictory labels), plus "apple" and
"grape" from v1. v1's Banana / Orange / Pineapple / Watermelon map to the same-named v2 classes.

Splits: each dataset keeps its own committed, leakage-free split (data/splits/ and data/splits_fruit30/).
Because both come from public web collections, the same photo could appear in BOTH datasets, so every
image is hashed again (64-bit dHash) and any validation/test image that is a near-duplicate (Hamming <= 4)
of a training image from either dataset is removed (and test images that duplicate a validation image).

Inputs : data/raw/dataset_fruits_detection/  (python -m src.data.download_dataset)
         data/raw/trucly_fruit_v1/           (Roboflow "Fruit" by trucly v1, YOLOv8 export, unzipped)
Outputs: data/splits_fruit31/ (committed lists + summary), data/processed_fruit31/, data/fruit31.yaml

Usage:
    python -m src.data.prepare_fruit31                  # build the split from scratch
    python -m src.data.prepare_fruit31 --from-manifest  # rebuild folders from the committed lists (Colab)
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
from src.data.dataset_validator import find_near_duplicate_pairs
from src.utils.image_utils import dhash, read_image

SOURCES = {
    "v1": {"raw": Path("data/raw/dataset_fruits_detection"), "splits": Path("data/splits"),
           "classes": ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon"]},
    "v2": {"raw": Path("data/raw/trucly_fruit_v1"), "splits": Path("data/splits_fruit30"), "classes": None},
}
RENAME = {"Apple": "apple", "Banana": "banana", "Grape": "grape", "Orange": "orange", "Pineapple": "pineapple",
          "Watermelon": "watermelon", "tangerines": "orange"}
OUT = Path("data/processed_fruit31")
SPLITS = Path("data/splits_fruit31")
YAML_OUT = Path("data/fruit31.yaml")
MAX_HAMMING = 4


def class_list() -> list[str]:
    v2 = yaml.safe_load((SOURCES["v2"]["raw"] / "data.yaml").read_text(encoding="utf-8"))["names"]
    return [c for c in v2 if c != "tangerines"] + ["apple", "grape"]


def source_classes(src: str) -> list[str]:
    s = SOURCES[src]
    return s["classes"] or yaml.safe_load((s["raw"] / "data.yaml").read_text(encoding="utf-8"))["names"]


def read_items() -> list[dict]:
    items = []
    for src, s in SOURCES.items():
        for split in ("train", "val", "test"):
            for rel in (s["splits"] / f"{split}.txt").read_text(encoding="utf-8").splitlines():
                if rel.strip():
                    items.append({"id": f"{src}/{rel}", "src": src, "rel": rel, "split": split})
    return items


def label_path(item: dict) -> Path:
    raw = SOURCES[item["src"]]["raw"]
    return raw / (item["rel"].replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")


def convert_label(item: dict, names: list[str]) -> tuple[list[str], int]:
    """Re-number classes into the combined list, clip boxes to the image. Returns (lines, boxes changed)."""
    src_names = source_classes(item["src"])
    index = {n: i for i, n in enumerate(names)}
    out, changed = [], 0
    path = label_path(item)
    if not path.is_file():
        return out, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cls = RENAME.get(src_names[int(p[0])], src_names[int(p[0])])
        xc, yc, w, h = map(float, p[1:])
        x1, y1, x2, y2 = max(0.0, xc - w / 2), max(0.0, yc - h / 2), min(1.0, xc + w / 2), min(1.0, yc + h / 2)
        if x2 - x1 <= 1e-4 or y2 - y1 <= 1e-4:
            changed += 1
            continue
        if abs(x2 - x1 - w) > 1e-6 or abs(y2 - y1 - h) > 1e-6:
            changed += 1
        out.append(f"{index[cls]} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return out, changed


def remove_cross_dataset_leaks(items: list[dict]) -> tuple[list[dict], dict]:
    print(f"Hashing {len(items)} images from both datasets (a few minutes) ...", flush=True)
    hashes = np.array([dhash(read_image(SOURCES[it["src"]]["raw"] / it["rel"])) for it in items], dtype=np.uint64)
    order = {"train": 0, "val": 1, "test": 2}
    removed: dict[str, str] = {}
    cross_dataset = 0
    for i, j, d in find_near_duplicate_pairs(hashes, MAX_HAMMING):
        a, b = items[i], items[j]
        if a["split"] == b["split"]:
            continue
        cross_dataset += a["src"] != b["src"]
        later, earlier = (a, b) if order[a["split"]] > order[b["split"]] else (b, a)
        removed.setdefault(later["id"], f"near-duplicate (Hamming {d}) of {earlier['src']} {earlier['split']} image")
    kept = [it for it in items if it["id"] not in removed]
    info = {"removed": len(removed), "cross_dataset_pairs_across_splits": cross_dataset,
            "removed_by_split": dict(Counter(next(it["split"] for it in items if it["id"] == r) for r in removed)),
            "removed_images": removed}
    return kept, info


def materialize(items: list[dict], names: list[str]) -> dict:
    if OUT.exists():
        shutil.rmtree(OUT)
    stats: dict = {"changed_boxes": 0, "per_split": {}}
    for split in ("train", "val", "test"):
        (OUT / split / "images").mkdir(parents=True)
        (OUT / split / "labels").mkdir(parents=True)
        inst, by_src = Counter(), Counter()
        for it in (x for x in items if x["split"] == split):
            src_img = SOURCES[it["src"]]["raw"] / it["rel"]
            stem = f"{it['src']}_{src_img.stem}"   # prefix: the two datasets could reuse a file name
            _link_or_copy(src_img, OUT / split / "images" / f"{stem}{src_img.suffix}")
            lines, changed = convert_label(it, names)
            stats["changed_boxes"] += changed
            (OUT / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                                 encoding="utf-8")
            inst.update(names[int(l.split()[0])] for l in lines)
            by_src[it["src"]] += 1
        stats["per_split"][split] = {"images": sum(by_src.values()), "images_by_source": dict(by_src),
                                     "boxes": sum(inst.values()),
                                     "boxes_per_class": {n: inst.get(n, 0) for n in names}}
    YAML_OUT.write_text(yaml.safe_dump({"path": OUT.resolve().as_posix(), "train": "train/images",
                                        "val": "val/images", "test": "test/images", "nc": len(names),
                                        "names": names}, sort_keys=False), encoding="utf-8")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-manifest", action="store_true", help="use the committed lists in data/splits_fruit31/")
    args = parser.parse_args()
    for src, s in SOURCES.items():
        if not (s["raw"] / "data.yaml").is_file():
            hint = ("python -m src.data.download_dataset" if src == "v1"
                    else "unzip Fruit.v1i.yolov8.zip into data/raw/trucly_fruit_v1/")
            print(f"ERROR: {s['raw']}/data.yaml not found. Run: {hint}", file=sys.stderr)
            return 1
    names = class_list()
    if args.from_manifest:
        wanted = {s: set((SPLITS / f"{s}.txt").read_text(encoding="utf-8").split()) for s in ("train", "val", "test")}
        items = [dict(it, split=s) for it in read_items() for s in wanted if it["id"] in wanted[s]]
        info = json.loads((SPLITS / "summary.json").read_text(encoding="utf-8"))["leakage_removal"]
    else:
        items, info = remove_cross_dataset_leaks(read_items())
        SPLITS.mkdir(parents=True, exist_ok=True)
        for s in ("train", "val", "test"):
            (SPLITS / f"{s}.txt").write_text("\n".join(it["id"] for it in items if it["split"] == s) + "\n",
                                             encoding="utf-8")
    stats = materialize(items, names)
    summary = {"sources": {"v1": "lightly-ai/dataset_fruits_detection @ e626559 (CC0-1.0)",
                           "v2": "Roboflow Universe trucly/fruit-5ix6x v1 (CC BY 4.0)"},
               "classes": names, "renamed": RENAME, "leakage_removal": info,
               "label_fixes": {"boxes_clipped_or_dropped": stats["changed_boxes"]}, "final": stats["per_split"]}
    if not args.from_manifest:
        (SPLITS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Classes: {len(names)}; removed for cross-split duplicates: {info['removed']} "
          f"({info['cross_dataset_pairs_across_splits']} pairs across the two datasets)")
    for s, v in stats["per_split"].items():
        print(f"  {s:5s} {v['images']:6d} images {v['boxes']:6d} boxes {v['images_by_source']}")
    print(f"Wrote {OUT}/ and {YAML_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
