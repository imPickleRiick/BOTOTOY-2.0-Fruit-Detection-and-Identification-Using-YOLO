"""Step 8 of the OpenCV baseline: color and shape features of one region (contour).

Color features (measured only on the pixels inside the contour):
  mean_hue        circular mean of hue (hue wraps around: 178 and 2 are both red)
  mean_saturation average S (vividness)
  mean_value      average V (brightness)
Shape features:
  area            contour area in pixels; area_fraction = area / image area
  aspect_ratio    bounding-box width / height
  height_ratio    bounding-box height / width (> 1 means taller than wide)
  elongation      long side / short side of the minimum-area ROTATED rectangle
                  (rotation-independent: a tilted banana is still elongated)
  extent          area / bounding-box area (a circle fills ~0.785, a triangle ~0.5)
  circularity     4*pi*area / perimeter^2 (1.0 = perfect circle, lower = irregular/elongated)
  solidity        area / convex-hull area (1.0 = convex; bumpy grape clusters are lower)
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def circular_mean_hue(hues: np.ndarray) -> float:
    """Average OpenCV hue (0-179) treating it as an angle, so red (0 and 179) averages correctly."""
    if hues.size == 0:
        return 0.0
    angles = hues.astype(np.float64) * (2 * math.pi / 180.0)
    mean = math.atan2(np.sin(angles).mean(), np.cos(angles).mean())
    return (mean * 180.0 / (2 * math.pi)) % 180.0


def extract_features(contour: np.ndarray, hsv: np.ndarray) -> dict:
    h_img, w_img = hsv.shape[:2]
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))

    region = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.drawContours(region, [contour], -1, 255, thickness=cv2.FILLED)
    pixels = hsv[region == 255]

    return {
        "bbox": (x, y, x + w, y + h),
        "area": area,
        "area_fraction": area / float(h_img * w_img),
        "mean_hue": round(circular_mean_hue(pixels[:, 0]), 1) if len(pixels) else 0.0,
        "mean_saturation": round(float(pixels[:, 1].mean()), 1) if len(pixels) else 0.0,
        "mean_value": round(float(pixels[:, 2].mean()), 1) if len(pixels) else 0.0,
        "aspect_ratio": round(w / h, 3) if h else 0.0,
        "height_ratio": round(h / w, 3) if w else 0.0,
        "elongation": round(max(rw, rh) / min(rw, rh), 3) if min(rw, rh) > 0 else 0.0,
        "extent": round(area / (w * h), 3) if w * h else 0.0,
        "circularity": round(4 * math.pi * area / perimeter ** 2, 3) if perimeter else 0.0,
        "solidity": round(area / hull_area, 3) if hull_area else 0.0,
    }
