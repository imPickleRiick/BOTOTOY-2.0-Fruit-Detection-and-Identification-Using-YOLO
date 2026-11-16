"""Tests for the OpenCV baseline using synthetic images with known answers."""
import math

import cv2
import numpy as np
import pytest

from src.baseline.feature_extraction import circular_mean_hue, extract_features
from src.baseline.pipeline import BaselineDetector
from src.baseline.preprocessing import preprocess, resize_max_side
from src.baseline.segmentation import clean_mask, color_mask, split_touching_objects
from src.utils.helpers import load_config


def hsv_to_bgr(h, s, v):
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


def white_canvas(size=640):
    return np.full((size, size, 3), 255, np.uint8)


@pytest.fixture(scope="module")
def detector():
    return BaselineDetector(load_config())


# ---------------------------------------------------------------- preprocessing / segmentation
def test_resize_keeps_aspect_and_never_enlarges():
    img = np.zeros((1000, 500, 3), np.uint8)
    out, scale = resize_max_side(img, 640)
    assert out.shape[:2] == (640, 320) and scale == pytest.approx(0.64)
    small, s2 = resize_max_side(np.zeros((100, 100, 3), np.uint8), 640)
    assert small.shape[:2] == (100, 100) and s2 == 1.0


def test_preprocess_rejects_non_color_images():
    with pytest.raises(ValueError):
        preprocess(np.zeros((10, 10), np.uint8))


def test_red_mask_handles_hue_wraparound():
    hsv = np.array([[[2, 200, 200], [176, 200, 200], [60, 200, 200]]], np.uint8)
    mask = color_mask(hsv, [[[0, 90, 50], [7, 255, 255]], [[165, 90, 50], [179, 255, 255]]])
    assert mask.tolist() == [[255, 255, 0]]  # both reds selected, green not


def test_opening_removes_specks_and_closing_fills_holes():
    mask = np.zeros((100, 100), np.uint8)
    mask[10, 10] = 255                                    # isolated speck
    cv2.circle(mask, (60, 60), 25, 255, -1)
    mask[58:62, 58:62] = 0                                # small hole
    cleaned = clean_mask(mask, open_kernel=5, close_kernel=9)
    assert cleaned[10, 10] == 0
    assert cleaned[60, 60] == 255


def test_watershed_splits_two_touching_circles():
    mask = np.zeros((200, 300), np.uint8)
    cv2.circle(mask, (100, 100), 60, 255, -1)
    cv2.circle(mask, (200, 100), 60, 255, -1)
    assert len(split_touching_objects(mask, 0.6)) == 2


# ---------------------------------------------------------------- features
def test_shape_features_of_circle_and_triangle():
    hsv = np.zeros((400, 400, 3), np.uint8)
    circle = np.zeros((400, 400), np.uint8)
    cv2.circle(circle, (200, 200), 100, 255, -1)
    c, _ = cv2.findContours(circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    f = extract_features(c[0], hsv)
    assert f["circularity"] > 0.85 and f["extent"] == pytest.approx(math.pi / 4, abs=0.03)
    assert f["elongation"] == pytest.approx(1.0, abs=0.05)

    tri = np.array([[[50, 350]], [[350, 350]], [[200, 90]]], np.int32)
    f = extract_features(tri, hsv)
    assert f["extent"] == pytest.approx(0.5, abs=0.02) and f["solidity"] == pytest.approx(1.0, abs=0.01)


def test_circular_mean_hue_for_red():
    mean = circular_mean_hue(np.array([178, 2]))
    assert min(mean, 180 - mean) < 0.5   # ~0, not the naive average 90


# ---------------------------------------------------------------- full pipeline
def test_blank_image_has_no_detections(detector):
    assert detector.detect(white_canvas()).total == 0


@pytest.mark.parametrize("name, draw", [
    ("Orange", lambda im: cv2.circle(im, (320, 320), 90, hsv_to_bgr(15, 230, 230), -1)),
    ("Apple", lambda im: cv2.circle(im, (320, 320), 90, hsv_to_bgr(0, 220, 200), -1)),
    ("Banana", lambda im: cv2.ellipse(im, (320, 320), (220, 45), 20, 0, 360, hsv_to_bgr(27, 220, 230), -1)),
    ("Grape", lambda im: cv2.circle(im, (320, 320), 90, hsv_to_bgr(130, 150, 120), -1)),
])
def test_single_synthetic_fruit_is_classified(detector, name, draw):
    image = white_canvas()
    draw(image)
    result = detector.detect(image, keep_stages=True)
    assert result.total == 1
    assert result.detections[0].class_name == name
    assert result.counts[name] == 1
    assert set(result.stages) == {"1_preprocessed", "2_color_masks", "3_after_morphology", "4_contours", "5_detections"}


def test_boxes_are_mapped_back_to_original_resolution(detector):
    image = np.full((1280, 1280, 3), 255, np.uint8)          # processed at 640, i.e. scale 0.5
    cv2.circle(image, (640, 640), 180, hsv_to_bgr(15, 230, 230), -1)
    x1, y1, x2, y2 = detector.detect(image).detections[0].box
    assert abs(x1 - 460) <= 6 and abs(x2 - 820) <= 6 and abs(y1 - 460) <= 6 and abs(y2 - 820) <= 6
