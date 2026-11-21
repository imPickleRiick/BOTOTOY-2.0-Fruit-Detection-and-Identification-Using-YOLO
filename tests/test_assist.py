"""Tests for the assistive reader: spoken descriptions, fruit names and the photo-quality check."""
import cv2
import numpy as np
import pytest

from src.assist.describe import describe, full_message
from src.assist.fruit_names import FRUIT_NAMES, display_name, names_for
from src.assist.quality import check_quality
from src.utils.detections import Detection
from src.utils.helpers import load_config
from src.utils.image_utils import read_image

W = 600


def det(cls, conf, x):
    return Detection(cls, conf, (x, 10, x + 40, 50), 0)


def test_single_fruit_with_filipino_name():
    assert describe([det("mango", 0.9, 10)], W) == "I see 1 mango (mangga), on the left."


def test_counts_positions_and_plural():
    text = describe([det("mango", 0.9, 10), det("mango", 0.8, 60), det("mango", 0.7, 400)], W)
    assert text == "I see 3 mangoes (mangga): 2 on the left and 1 on the right."


def test_most_confident_type_first_and_certainty_words():
    text = describe([det("banana", 0.5, 250), det("guava", 0.95, 10), det("lychee", 0.3, 500)], W)
    assert text.index("guava") < text.index("banana") < text.index("lychee")
    assert "I probably see 1 banana" in text
    assert "lychee, on the right, but I am not sure" in text


def test_large_groups_are_many():
    text = describe([det("orange", 0.9, x) for x in range(0, 560, 70)], W)
    assert text == "I see many oranges (dalandan), across the photo."


def test_no_filipino_option_and_nothing_found():
    assert describe([det("mango", 0.9, 10)], W, filipino=False) == "I see 1 mango, on the left."
    assert full_message([], W).startswith("I don't see any fruit")
    assert full_message([], W, "The photo is too dark.").startswith("The photo is too dark.")


@pytest.mark.parametrize("config", ["config/config.yaml", "config/config_fruit30.yaml"])
def test_every_class_has_a_spoken_name(config):
    for cls in load_config(config)["classes"]:
        assert cls in FRUIT_NAMES, cls
        assert names_for(cls)[0]


def test_display_name():
    assert display_name("sapodilla") == "Sapodilla (chico)"
    assert display_name("durian") == "Durian"


@pytest.fixture(scope="module")
def photo():
    return read_image("assets/unseen/mandarins_on_plate.jpg")


def test_quality_ok_on_a_normal_photo(photo):
    assert check_quality(photo).ok


def test_quality_flags_dark_photo(photo):
    q = check_quality(cv2.convertScaleAbs(photo, alpha=0.3))
    assert q.issues[0] == "dark" and "too dark" in q.message


def test_quality_flags_blur_and_motion_blur(photo):
    assert "blurry" in check_quality(cv2.GaussianBlur(photo, (31, 31), 0)).issues
    k = np.zeros((31, 31), np.float32)
    k[15, :] = 1 / 31
    assert "motion_blur" in check_quality(cv2.filter2D(photo, -1, k)).issues


def test_quality_flags_overexposed_photo(photo):
    assert "overexposed" in check_quality(cv2.convertScaleAbs(photo, alpha=2.5)).issues


def test_phone_qr_code_decodes_to_the_url():
    from src.assist.phone import phone_url, qr_image
    url = "http://192.168.1.23:8501"
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(qr_image(url))
    assert text == url
    found = phone_url(8501)
    assert found is None or (found.startswith("http://") and found.endswith(":8501"))
