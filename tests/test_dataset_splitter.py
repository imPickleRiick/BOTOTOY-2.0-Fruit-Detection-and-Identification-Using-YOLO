"""Tests for the dataset splitter logic and the committed split manifests."""
import json

import pandas as pd
import pytest

from src.data.dataset_splitter import (UnionFind, assign_val_test, dominant_class,
                                       find_leaking_holdout_images, read_manifests)
from src.utils.helpers import PROJECT_ROOT

CLASSES = ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon"]


def test_dominant_class():
    assert dominant_class([3, 3, 1], CLASSES) == "Orange"
    assert dominant_class([1, 0], CLASSES) == "Apple"  # tie -> lowest id
    assert dominant_class([], CLASSES) == "background"


def test_union_find_groups():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    assert uf.find("a") == uf.find("c")
    assert uf.find("d") != uf.find("a")


def test_assign_val_test_is_balanced_grouped_and_deterministic():
    groups = {f"g{i}": [f"g{i}_img{k}" for k in range(1 + i % 2)] for i in range(200)}
    strata = {g: CLASSES[i % 6] for i, g in enumerate(groups)}
    first = assign_val_test(groups, strata, 0.5, seed=42)
    assert first == assign_val_test(groups, strata, 0.5, seed=42)  # reproducible
    for imgs in groups.values():  # a group never straddles val and test
        assert len({first[i] for i in imgs}) == 1
    val_share = sum(s == "val" for s in first.values()) / len(first)
    assert 0.45 <= val_share <= 0.55


def test_leaking_holdout_images_are_found():
    images = pd.DataFrame({"image": ["t1", "t2", "v1", "v2", "v3"],
                           "split": ["train", "train", "valid", "valid", "test"],
                           "md5": ["x", "y", "x", "z", "w"]})
    near = pd.DataFrame([{"image_a": "t2", "split_a": "train", "image_b": "v3", "split_b": "test", "hamming": 2}])
    leaks = find_leaking_holdout_images(images, near)
    assert set(leaks) == {"v1", "v3"}  # v1 = exact copy of t1, v3 = near-duplicate of t2


SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


@pytest.mark.skipif(not (SPLITS_DIR / "train.txt").is_file(), reason="split manifests not generated")
def test_committed_manifests_have_no_overlap_and_match_summary():
    lists = read_manifests(SPLITS_DIR)
    sets = {k: set(v) for k, v in lists.items()}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
    summary = json.loads((SPLITS_DIR / "split_summary.json").read_text(encoding="utf-8"))
    assert summary["verification"]["leakage_free"] is True
    for split, imgs in lists.items():
        assert summary["splits"][split]["images"] == len(imgs)
    # Option A: training images come only from the original train folder.
    assert all(p.startswith("train/") for p in lists["train"])
    assert not any(p.startswith("train/") for p in lists["val"] + lists["test"])


def test_from_manifest_keeps_build_info(tmp_path):
    """Regression: --from-manifest used to drop the seed and the excluded-image list."""
    from src.data.dataset_splitter import load_build_info
    summary = {"strategy": "Option A", "seed": 42, "excluded": [{"image": "x.jpg", "reason": "leak"}],
               "splits": {"train": {}}, "verification": {"leakage_free": True}}
    path = tmp_path / "split_summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    info = load_build_info(path)
    assert info == {"strategy": "Option A", "seed": 42, "excluded": [{"image": "x.jpg", "reason": "leak"}]}
    assert "strategy" in load_build_info(tmp_path / "missing.json")


@pytest.mark.skipif(not (SPLITS_DIR / "split_summary.json").is_file(), reason="split summary not generated")
def test_committed_summary_records_seed_and_exclusions():
    summary = json.loads((SPLITS_DIR / "split_summary.json").read_text(encoding="utf-8"))
    assert summary["seed"] == 42 and len(summary["excluded"]) == 3
