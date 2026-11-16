"""Tests for the YOLO modules that do not need a trained model."""
import zipfile

import pytest

from src.utils.helpers import PROJECT_ROOT, load_config
from src.yolo.evaluate import target_checks
from src.yolo.train import best_epoch_from_results, resolve_device

PRETRAINED = PROJECT_ROOT / "models" / "pretrained" / "yolo11n.pt"


def test_resolve_device_passthrough_and_auto():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device(0) == 0
    import torch
    assert resolve_device("auto") == (0 if torch.cuda.is_available() else "cpu")


def test_best_epoch_is_selected_by_validation_map50_95(tmp_path):
    csv = tmp_path / "results.csv"
    csv.write_text(
        "epoch,time,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,60,0.5,0.4,0.50,0.30\n"
        "2,120,0.6,0.5,0.62,0.41\n"   # best mAP50-95
        "3,180,0.7,0.5,0.65,0.40\n", encoding="utf-8")
    best = best_epoch_from_results(csv)
    assert best["best_epoch"] == 2 and best["epochs_completed"] == 3
    assert best["best_val_mAP50-95"] == 0.41 and best["logged_training_minutes"] == 3.0


def test_target_checks():
    ultra = {"mAP50": 0.85, "mAP50-95": 0.49, "per_class": {"Apple": {"recall": 0.8}, "Grape": {"recall": 0.65}}}
    shared = {"timing_ms": {"end_to_end_mean": 140.0}}
    t = target_checks(ultra, shared)
    assert t["mAP50 >= 0.80"]["met"] and not t["mAP50-95 >= 0.50"]["met"]
    assert not t["recall >= 0.70 for every class"]["met"]
    assert t["end-to-end <= 1 s per image"]["met"]


def test_detector_requires_existing_weights(tmp_path):
    from src.yolo.detector import YoloDetector
    with pytest.raises(FileNotFoundError):
        YoloDetector(tmp_path / "missing.pt", load_config())


@pytest.mark.skipif(not PRETRAINED.is_file(), reason="COCO checkpoint not downloaded")
def test_detector_rejects_model_with_other_classes():
    """The COCO model has 80 classes, not our 6 fruits: it must not be used by mistake."""
    from src.yolo.detector import YoloDetector
    with pytest.raises(ValueError):
        YoloDetector(PRETRAINED, load_config())


def test_training_report_rejects_zip_without_results(tmp_path):
    from src.yolo.training_report import extract_run
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("run/readme.txt", "no results here")
    with pytest.raises(FileNotFoundError):
        extract_run(bad, tmp_path / "out")
