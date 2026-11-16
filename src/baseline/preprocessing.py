"""Step 1-3 of the OpenCV baseline: load, resize, denoise and convert BGR -> HSV.

Why HSV? In BGR, a color's brightness is mixed into all three channels, so the same
orange looks very different in shade and in sunlight. HSV separates
  H (hue)        - the "color name" (red, orange, yellow, green ...), 0-179 in OpenCV
  S (saturation) - how vivid the color is (0 = gray)
  V (value)      - brightness
so we can select a fruit's color with a hue range while tolerating lighting changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessedImage:
    bgr: np.ndarray     # resized + blurred BGR image used for processing
    hsv: np.ndarray     # same image in HSV
    scale: float        # resized size / original size (to map boxes back)
    original_shape: tuple[int, int]


def resize_max_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Shrink so the longer side is at most `max_side` (keeps aspect ratio; never enlarges)."""
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return image, scale


def preprocess(image: np.ndarray, max_side: int = 640, blur_kernel: int = 5) -> PreprocessedImage:
    """Resize -> Gaussian blur (removes pixel noise / fine texture) -> HSV conversion."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("preprocess() expects a BGR color image")
    resized, scale = resize_max_side(image, max_side)
    if blur_kernel and blur_kernel > 1:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1  # kernel size must be odd
        resized = cv2.GaussianBlur(resized, (k, k), 0)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    return PreprocessedImage(bgr=resized, hsv=hsv, scale=scale, original_shape=image.shape[:2])
