"""Dataset splitter: builds the final train / val / test sets (approved Option A).

Why we re-split
---------------
The original dataset's training split is Roboflow-augmented (each source photo has 1-3
augmented copies), while its valid/test splits are not. Counting unique source photos the
original split is 3200 / 914 / 457 = 70 / 20 / 10. Our proposal targets 70 / 15 / 15.

Option A (approved by the team)
-------------------------------
1. Keep the original training split unchanged (all augmented copies of a photo stay there).
2. Pool the un-augmented original valid + test images (the "holdout pool").
3. Remove holdout images that leak training content (exact MD5 duplicates of, or perceptual
   near-duplicates of, a training image - found by dataset_validator.py).
4. Group holdout images that are near-duplicates of each other (they must share a split).
5. Stratify by each group's dominant fruit class and assign groups ~50/50 to val and test
   with a fixed random seed, so both sets have a similar class mix.

Outputs
-------
- data/splits/{train,val,test}.txt  file lists relative to the raw dataset (committed to Git,
                                     so everybody - including the Kaggle notebook - gets the
                                     exact same split)
- data/splits/split_summary.json    counts, exclusions and leakage verification
- data/processed/{train,val,test}/{images,labels}  hard links to the raw files (no extra disk)
- data/data.yaml                    dataset config consumed by Ultralytics YOLO

Usage (from the project root, after running dataset_validator):
    python -m src.data.dataset_splitter              # create split + materialize folders
    python -m src.data.dataset_splitter --from-manifest   # rebuild folders from committed lists
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

from src.utils.helpers import ensure_dir, load_config, resolve_path, save_json

FINAL_SPLITS = ("train", "val", "test")


class UnionFind:
    """Tiny union-find (disjoint set) used to group near-duplicate images together."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # deterministic root


def find_leaking_holdout_images(images: pd.DataFrame, near: pd.DataFrame) -> dict[str, str]:
    """Holdout images that duplicate a training image -> {image: reason}."""
    train_md5 = set(images.loc[images.split == "train", "md5"])
    leaks: dict[str, str] = {}
    for row in images[images.split != "train"].itertuples():
        if row.md5 in train_md5:
            leaks[row.image] = "exact duplicate of a training image"
    for row in near.itertuples():
        if row.split_a == "train" and row.split_b != "train":
            leaks.setdefault(row.image_b, f"near-duplicate of training image (Hamming {row.hamming})")
        elif row.split_b == "train" and row.split_a != "train":
            leaks.setdefault(row.image_a, f"near-duplicate of training image (Hamming {row.hamming})")
    return leaks


def dominant_class(class_ids: list[int], classes: list[str]) -> str:
    """Most frequent class in a group (ties -> lowest class id); 'background' if no boxes."""
    if not class_ids:
        return "background"
    counts = Counter(class_ids)
    best = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    return classes[best]


def assign_val_test(groups: dict[str, list[str]], strata: dict[str, str],
                    val_fraction: float, seed: int) -> dict[str, str]:
    """Stratified, grouped assignment of holdout groups to 'val' / 'test'.

    Within each stratum (dominant class) the groups are shuffled, then each group goes to
    whichever split is currently furthest below its target share. Returns {image: split}.
    """
    rng = random.Random(seed)
    targets = {"val": val_fraction, "test": 1.0 - val_fraction}
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for gid in sorted(groups):
        by_stratum[strata[gid]].append(gid)

    assignment: dict[str, str] = {}
    for stratum in sorted(by_stratum):
        gids = by_stratum[stratum]
        rng.shuffle(gids)
        counts = {"val": 0, "test": 0}
        for gid in gids:
            total = counts["val"] + counts["test"] + len(groups[gid])
            # pick the split whose share would stay lowest relative to its target
            split = min(("val", "test"), key=lambda s: (counts[s] + len(groups[gid])) / (targets[s] * total))
            counts[split] += len(groups[gid])
            for image in groups[gid]:
                assignment[image] = split
    return assignment


def build_split(cfg: dict) -> tuple[dict[str, list[str]], dict]:
    """Compute the final split from the validator's audit tables."""
    ds = cfg["dataset"]
    classes = cfg["classes"]
    audit_dir = resolve_path(ds["audit_dir"])
    if not (audit_dir / "images.csv").is_file():
        raise FileNotFoundError("Audit tables missing. Run: python -m src.data.dataset_validator")
    images = pd.read_csv(audit_dir / "images.csv")
    boxes = pd.read_csv(audit_dir / "boxes.csv")
    near = pd.read_csv(audit_dir / "near_duplicates.csv")

    critical = images.loc[images["critical"], "image"].tolist()
    usable = images[~images["critical"]]

    train = sorted(usable.loc[usable.split == "train", "image"])
    holdout = usable[usable.split.isin(["valid", "test"])]

    excluded: dict[str, str] = {img: "critical validation issue" for img in critical}
    if ds["split"]["exclude_holdout_duplicates_of_train"]:
        leaks = find_leaking_holdout_images(usable, near)
        excluded.update(leaks)
        holdout = holdout[~holdout["image"].isin(leaks)]

    # Group holdout images: same source photo, or near-duplicates of each other.
    uf = UnionFind()
    holdout_set = set(holdout["image"])
    by_source = holdout.groupby("source_id")["image"].agg(list)
    for imgs in by_source:
        for img in imgs:
            uf.union(imgs[0], img)
    for row in near.itertuples():
        if row.image_a in holdout_set and row.image_b in holdout_set:
            uf.union(row.image_a, row.image_b)
    groups: dict[str, list[str]] = defaultdict(list)
    for img in sorted(holdout_set):
        groups[uf.find(img)].append(img)

    class_ids_per_image = boxes.groupby("image")["class_id"].agg(list).to_dict()
    strata = {gid: dominant_class([c for img in imgs for c in class_ids_per_image.get(img, [])], classes)
              for gid, imgs in groups.items()}

    assignment = assign_val_test(groups, strata, ds["split"]["holdout_val_fraction"], cfg["project"]["seed"])
    split_lists = {
        "train": train,
        "val": sorted(i for i, s in assignment.items() if s == "val"),
        "test": sorted(i for i, s in assignment.items() if s == "test"),
    }
    info = {
        "strategy": "Option A: keep original train; stratified grouped 50/50 re-split of original valid+test",
        "seed": cfg["project"]["seed"],
        "holdout_groups": len(groups),
        "holdout_groups_with_multiple_images": sum(len(v) > 1 for v in groups.values()),
        "excluded": [{"image": k, "reason": v} for k, v in sorted(excluded.items())],
    }
    return split_lists, info


def summarize_and_verify(split_lists: dict[str, list[str]], cfg: dict) -> dict:
    """Counts per split/class plus leakage checks on the FINAL split."""
    audit_dir = resolve_path(cfg["dataset"]["audit_dir"])
    images = pd.read_csv(audit_dir / "images.csv").set_index("image")
    boxes = pd.read_csv(audit_dir / "boxes.csv")
    near = pd.read_csv(audit_dir / "near_duplicates.csv")
    split_of = {img: s for s, imgs in split_lists.items() for img in imgs}

    total_imgs = sum(len(v) for v in split_lists.values())
    sources = {s: set(images.loc[imgs, "source_id"]) for s, imgs in split_lists.items()}
    total_sources = sum(len(v) for v in sources.values())
    summary: dict = {"splits": {}}
    for s, imgs in split_lists.items():
        b = boxes[boxes["image"].isin(set(imgs))]
        summary["splits"][s] = {
            "images": len(imgs),
            "image_share": round(len(imgs) / total_imgs, 4),
            "unique_sources": len(sources[s]),
            "unique_source_share": round(len(sources[s]) / total_sources, 4),
            "boxes": int(len(b)),
            "instances_per_class": {c: int((b["class_name"] == c).sum()) for c in cfg["classes"]},
            "images_per_class": {c: int(b.loc[b["class_name"] == c, "image"].nunique()) for c in cfg["classes"]},
        }

    md5 = {s: set(images.loc[imgs, "md5"]) for s, imgs in split_lists.items()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    near_cross = 0
    for row in near.itertuples():
        sa, sb = split_of.get(row.image_a), split_of.get(row.image_b)
        if sa and sb and sa != sb:
            near_cross += 1
    checks = {
        "file_overlap": {f"{a}-{b}": len(set(split_lists[a]) & set(split_lists[b])) for a, b in pairs},
        "md5_overlap": {f"{a}-{b}": len(md5[a] & md5[b]) for a, b in pairs},
        "source_overlap": {f"{a}-{b}": len(sources[a] & sources[b]) for a, b in pairs},
        "near_duplicate_pairs_across_splits": near_cross,
    }
    checks["leakage_free"] = (all(v == 0 for k in ("file_overlap", "md5_overlap", "source_overlap")
                                  for v in checks[k].values()) and near_cross == 0)
    summary["verification"] = checks
    return summary


def write_manifests(split_lists: dict[str, list[str]], splits_dir: Path) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for s, imgs in split_lists.items():
        (splits_dir / f"{s}.txt").write_text("\n".join(imgs) + "\n", encoding="utf-8")


def read_manifests(splits_dir: Path) -> dict[str, list[str]]:
    lists = {}
    for s in FINAL_SPLITS:
        path = splits_dir / f"{s}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split manifest {path}")
        lists[s] = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lists


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hard link (no extra disk space) with a copy fallback (e.g. across drives)."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize(split_lists: dict[str, list[str]], cfg: dict) -> Path:
    """Create data/processed/{split}/images|labels and data/data.yaml for Ultralytics."""
    ds = cfg["dataset"]
    raw_dir = resolve_path(ds["raw_dir"])
    processed = ensure_dir(ds["processed_dir"])
    for s, imgs in split_lists.items():
        split_dir = processed / s
        if split_dir.exists():
            shutil.rmtree(split_dir)  # only our own generated folder
        (split_dir / "images").mkdir(parents=True)
        (split_dir / "labels").mkdir(parents=True)
        for rel in imgs:
            img = raw_dir / rel
            label = img.parent.parent / "labels" / (img.stem + ".txt")
            if not img.is_file() or not label.is_file():
                raise FileNotFoundError(f"Missing raw file for {rel}")
            _link_or_copy(img, split_dir / "images" / img.name)
            _link_or_copy(label, split_dir / "labels" / label.name)
        print(f"  {s}: {len(imgs)} images linked into {split_dir}")

    data_yaml = resolve_path(ds["data_yaml"])
    content = {
        # Absolute path is written at generation time so Ultralytics finds the data on any machine.
        "path": processed.as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(cfg["classes"]),
        "names": {i: n for i, n in enumerate(cfg["classes"])},
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write("# Generated by src/data/dataset_splitter.py - do not edit by hand.\n")
        yaml.safe_dump(content, f, sort_keys=False)
    print(f"  wrote {data_yaml}")
    return data_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--from-manifest", action="store_true",
                        help="reuse the committed data/splits/*.txt instead of computing a new split")
    args = parser.parse_args()
    cfg = load_config(args.config)
    splits_dir = resolve_path(cfg["dataset"]["splits_dir"])
    try:
        if args.from_manifest:
            split_lists, info = read_manifests(splits_dir), {"strategy": "loaded from committed manifests"}
        else:
            split_lists, info = build_split(cfg)
            write_manifests(split_lists, splits_dir)
        summary = summarize_and_verify(split_lists, cfg)
        summary.update(info)
        save_json(summary, splits_dir / "split_summary.json")
        print("Materializing split folders ...")
        materialize(split_lists, cfg)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\n=== FINAL SPLIT ===")
    for s, d in summary["splits"].items():
        print(f"{s:>5}: {d['images']:5d} images ({d['image_share']:.1%}), {d['unique_sources']:5d} unique sources "
              f"({d['unique_source_share']:.1%}), {d['boxes']} boxes")
    print(f"Excluded images: {len(info.get('excluded', []))}")
    print(f"Verification: {summary['verification']}")
    return 0 if summary["verification"]["leakage_free"] else 1


if __name__ == "__main__":
    sys.exit(main())
