"""Steps 4-7 of the OpenCV baseline: HSV color masks, morphology, contours, contour filtering.

- Color segmentation: cv2.inRange() keeps pixels whose H, S and V are inside a range,
  producing a binary mask (255 = "this pixel has the fruit color").
- Morphological OPENING (erode then dilate) removes small specks of noise.
- Morphological CLOSING (dilate then erode) fills small holes and gaps inside a fruit.
- External contours = outlines of each white blob in the cleaned mask.
- Contour filtering removes blobs that are too small (noise) or too large (background).
"""
from __future__ import annotations

import cv2
import numpy as np


def color_mask(hsv: np.ndarray, ranges: list[list[list[int]]]) -> np.ndarray:
    """Union of several [lower, upper] HSV ranges (red needs two: near 0 and near 179)."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    return mask


def build_color_masks(hsv: np.ndarray, color_ranges: dict[str, list]) -> dict[str, np.ndarray]:
    """One binary mask per color group (e.g. red, warm, green, purple)."""
    return {name: color_mask(hsv, ranges) for name, ranges in color_ranges.items()}


def clean_mask(mask: np.ndarray, open_kernel: int = 5, close_kernel: int = 9,
               open_iterations: int = 1, close_iterations: int = 2) -> np.ndarray:
    """Morphological opening (remove noise) followed by closing (fill holes)."""
    out = mask
    if open_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k, iterations=open_iterations)
    if close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k, iterations=close_iterations)
    return out


def find_contours(mask: np.ndarray, min_area: float, max_area: float) -> list[np.ndarray]:
    """External contours whose area is within [min_area, max_area] pixels."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]


def split_touching_objects(region_mask: np.ndarray, min_distance_ratio: float = 0.5) -> list[np.ndarray]:
    """Split a blob of touching round fruits using the distance transform + watershed.

    The distance transform gives every white pixel its distance to the nearest black pixel;
    the centers of round fruits become bright peaks. Each peak strong enough
    (>= min_distance_ratio * max distance) becomes a seed ("marker"), and the watershed
    algorithm grows the seeds until they meet, cutting the blob between fruits.
    Returns one contour per separated object (the original contour if no split happened).
    """
    dist = cv2.distanceTransform(region_mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []
    _, peaks = cv2.threshold(dist, min_distance_ratio * dist.max(), 255, cv2.THRESH_BINARY)
    peaks = peaks.astype(np.uint8)
    n_markers, markers = cv2.connectedComponents(peaks)
    if n_markers <= 2:  # background + one peak -> nothing to split
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)
    unknown = cv2.subtract(region_mask, peaks)
    markers = markers + 1            # background becomes 1, peaks 2..n
    markers[unknown == 255] = 0      # 0 = "to be decided by watershed"
    color = cv2.cvtColor(region_mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color, markers)
    pieces = []
    for label in range(2, n_markers + 1):
        piece = np.uint8(markers == label) * 255
        cnts, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pieces.extend(cnts)
    return pieces
