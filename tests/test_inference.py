"""Tests for src/inference: image decoding, the detection engine (incl. fallback) and batch processing."""
import csv
import io
import json
import zipfile

import cv2
import numpy as np
import pytest

from src.inference.batch_processor import MAX_BATCH_IMAGES, process_batch, save_batch
from src.inference.image_processor import (BASELINE, YOLO, FruitDetectionEngine, InvalidImageError,
                                           decode_image, save_result)
from src.inference.visualization import detection_rows, fit_within, safe_stem
from src.utils.detections import Detection
from src.utils.helpers import PROJECT_ROOT

TRAINED = PROJECT_ROOT / "models" / "trained" / "bototoy_yolo11n.pt"
SAMPLE = PROJECT_ROOT / "assets" / "samples" / "oranges_in_bowls.jpg"
needs_model = pytest.mark.skipif(not TRAINED.is_file(), reason="trained YOLO weights not installed")


def encode(image: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, image)
    assert ok
    return buf.tobytes()


# ----------------------------------------------------------------------------- decoding
@pytest.mark.parametrize("data", [b"", b"this is not an image", b"\xff\xd8\xff\xe0 truncated jpeg"])
def test_decode_rejects_invalid_files(data):
    with pytest.raises(InvalidImageError):
        decode_image(data, "bad.jpg")


def test_decode_rejects_tiny_images():
    with pytest.raises(InvalidImageError, match="too small"):
        decode_image(encode(np.zeros((10, 10, 3), np.uint8)), "tiny.png")


def test_decode_grayscale_becomes_three_channels():
    img = decode_image(encode(np.full((64, 80), 128, np.uint8)), "gray.png")
    assert img.shape == (64, 80, 3) and img.dtype == np.uint8


def test_decode_transparent_png_is_placed_on_white():
    rgba = np.zeros((64, 64, 4), np.uint8)
    rgba[:, :32] = (0, 0, 255, 255)      # left half: opaque red
    rgba[:, 32:] = (0, 0, 0, 0)          # right half: fully transparent
    img = decode_image(encode(rgba), "transparent.png")
    assert img.shape == (64, 64, 3)
    assert tuple(img[10, 10]) == (0, 0, 255)
    assert tuple(img[10, 50]) == (255, 255, 255)


def test_decode_applies_exif_rotation():
    """Phone photos store 'rotate 90°' in EXIF; the decoded image must be upright."""
    PIL = pytest.importorskip("PIL.Image")
    img = PIL.new("RGB", (120, 60), (200, 50, 50))
    exif = img.getexif()
    exif[0x0112] = 6  # orientation: rotate 90° clockwise to display
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    decoded = decode_image(buf.getvalue(), "phone.jpg")
    assert decoded.shape[:2] == (120, 60)  # height and width swapped


# ----------------------------------------------------------------------------- helpers
def test_safe_stem_and_fit_within():
    assert safe_stem("My Photo (1).JPG") == "My_Photo_1"
    assert safe_stem("../../etc/passwd") == "passwd"
    assert safe_stem(r"C:\Users\x\fruit.v2.png") == "fruit.v2"
    assert safe_stem("???.jpg") == "image"
    assert fit_within(np.zeros((100, 4000, 3), np.uint8), 1000).shape[:2] == (25, 1000)
    small = np.zeros((10, 20, 3), np.uint8)
    assert fit_within(small, 1000) is small  # never upscaled


def test_detection_rows_sorted_by_confidence():
    rows = detection_rows([Detection("Apple", 0.4, (0, 0, 10, 20)), Detection("Banana", 0.9, (5, 5, 15, 15))])
    assert [r["Fruit"] for r in rows] == ["Banana", "Apple"]
    assert rows[1]["Width"] == 10 and rows[1]["Height"] == 20
    assert rows[1]["Box (x1, y1, x2, y2)"] == "0, 0, 10, 20"


# ----------------------------------------------------------------------------- engine
@pytest.fixture(scope="module")
def engine():
    return FruitDetectionEngine()


@pytest.fixture(scope="module")
def engine_without_yolo(tmp_path_factory):
    return FruitDetectionEngine(weights=tmp_path_factory.mktemp("w") / "missing.pt")


def test_engine_falls_back_to_baseline_without_weights(engine_without_yolo):
    eng = engine_without_yolo
    assert not eng.yolo_available and "not found" in eng.yolo_error
    res = eng.run_bytes(SAMPLE.read_bytes(), SAMPLE.name, method=YOLO)
    assert res.method == BASELINE and "OpenCV baseline was used" in res.notice
    assert res.total == len(res.detections) == sum(res.counts.values())
    assert eng.model_info()["yolo"]["available"] is False


def test_engine_rejects_unknown_method(engine_without_yolo):
    with pytest.raises(ValueError):
        engine_without_yolo.run(np.zeros((64, 64, 3), np.uint8), method="sift")


def test_baseline_run_keeps_stages(engine_without_yolo):
    res = engine_without_yolo.run_bytes(SAMPLE.read_bytes(), method=BASELINE, keep_stages=True)
    assert len(res.stages) == 5 and res.regions
    assert res.annotated.shape == res.image.shape
    assert {"decode", "detector_total", "end_to_end"} <= set(res.timing_ms)
    assert res.timing_ms["end_to_end"] >= res.timing_ms["detector_total"]


@needs_model
def test_yolo_run_and_settings(engine):
    data = SAMPLE.read_bytes()
    res = engine.run_bytes(data, SAMPLE.name, method=YOLO, conf=0.25, iou=0.7, imgsz=640)
    assert res.method == YOLO and res.notice is None
    assert {"preprocess", "inference", "postprocess", "end_to_end"} <= set(res.timing_ms)
    assert res.params["confidence"] == 0.25 and res.params["image_size"] == 640
    assert all(d.confidence >= 0.25 for d in res.detections)
    assert set(res.counts) == set(engine.classes)
    low = engine.run_bytes(data, method=YOLO, conf=0.05)
    assert low.total >= res.total  # a lower threshold never keeps fewer boxes
    s = res.summary("x.jpg")
    json.dumps(s)  # serializable
    assert s["total"] == res.total and s["method"] == "YOLO11n"


@needs_model
def test_model_info_reports_trained_weights(engine):
    info = engine.model_info()["yolo"]
    assert info["available"] and info["classes"] == engine.classes
    assert info["size_mb"] > 1 and len(info["sha256"]) == 64 and info["parameters"] > 1_000_000


def test_save_result_writes_image_and_json(engine_without_yolo, tmp_path):
    res = engine_without_yolo.run_bytes(SAMPLE.read_bytes(), method=BASELINE)
    paths = save_result(res, tmp_path, "oranges")
    assert paths["image"].is_file() and paths["image"].name == "oranges_baseline.jpg"
    assert json.loads(paths["json"].read_text())["total"] == res.total


# ----------------------------------------------------------------------------- batch
def test_batch_handles_invalid_files_and_summarizes(engine_without_yolo, tmp_path):
    good = SAMPLE.read_bytes()
    files = [("a.jpg", good), ("broken.jpg", b"garbage"), ("a.jpg", good), ("empty.png", b"")]
    seen = []
    batch = process_batch(files, engine_without_yolo, BASELINE, progress=lambda k, n, name: seen.append(k))
    assert seen == [1, 2, 3, 4]
    assert len(batch.succeeded) == 2 and len(batch.failed) == 2
    assert all("broken.jpg" in i.error or "empty.png" in i.error for i in batch.failed)
    assert [i.output_stem for i in batch.items[::2]] == ["a", "a_2"]  # duplicate names do not overwrite
    combined = batch.combined_counts()
    assert sum(combined.values()) == batch.total_fruits == sum(i.total for i in batch.succeeded)

    rows = list(csv.DictReader(io.StringIO(batch.to_csv())))
    assert [r["Status"] for r in rows] == ["ok", "error", "ok", "error"]
    with zipfile.ZipFile(io.BytesIO(batch.to_zip())) as z:
        names = set(z.namelist())
        assert {"results.csv", "summary.json", "annotated/a_baseline.jpg", "annotated/a_2_baseline.jpg"} <= names
        assert json.loads(z.read("summary.json"))["failed"] == 2

    out = save_batch(batch, tmp_path / "batch")
    assert len(list((out / "annotated").glob("*.jpg"))) == 2 and (out / "results.csv").is_file()


def test_batch_limits(engine_without_yolo):
    with pytest.raises(ValueError):
        process_batch([], engine_without_yolo, BASELINE)
    with pytest.raises(ValueError, match="limit"):
        process_batch([("x.jpg", b"")] * (MAX_BATCH_IMAGES + 1), engine_without_yolo, BASELINE)
