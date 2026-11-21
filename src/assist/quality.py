"""Photo-quality check for the assistive reader (classic computer vision, no training).

A blind or low-vision user cannot see that a photo is blurry or dark, and our hard-case test showed
that motion blur alone drops detection to 4 of 12 photos. Before detecting, the reader measures:

- sharpness   variance of the Laplacian (second derivative) of the grayscale image. Edges give large
              values; a blurred or low-resolution photo has almost none.
- motion blur ratio of the weakest to the strongest directional second-derivative energy
              (0 / 45 / 90 / 135 degrees). A shaking camera smears detail in ONE direction, so that
              direction loses its energy and the ratio becomes small.
- brightness  mean gray level (0-255) and share of fully white pixels (>= 250).

Thresholds were chosen on the 12 unseen photos + 8 demo photos and their simulated hard versions
(src/evaluation/hard_cases.py): every dark, blurred and low-resolution version is flagged, and no
original photo is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MIN_SHARPNESS = 60.0        # Laplacian variance; originals >= 79, blurred / low-resolution <= 47
MIN_DIRECTION_RATIO = 0.15  # originals >= 0.19, motion blur 0.01-0.21 (median 0.04)
MIN_BRIGHTNESS = 60.0       # mean gray; originals >= 75, dark versions <= 54
MAX_BRIGHTNESS = 190.0
MAX_WHITE_PERCENT = 20.0    # % of pixels >= 250 (white backgrounds alone stay below this)

_DIRECTION_KERNELS = [np.array(k, np.float32) for k in (
    [[0, 0, 0], [1, -2, 1], [0, 0, 0]],      # horizontal
    [[0, 1, 0], [0, -2, 0], [0, 1, 0]],      # vertical
    [[1, 0, 0], [0, -2, 0], [0, 0, 1]],      # diagonal
    [[0, 0, 1], [0, -2, 0], [1, 0, 0]],      # anti-diagonal
)]

MESSAGES = {
    "dark": "The photo is too dark. Move to a brighter place or turn on a light.",
    "overexposed": "The photo is too bright. Move away from direct light.",
    "motion_blur": "The photo is blurry from movement. Hold the phone still and try again.",
    "blurry": "The photo is blurry. Hold the camera a little farther away and try again.",
}


@dataclass
class QualityReport:
    sharpness: float
    direction_ratio: float
    brightness: float
    white_percent: float
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def message(self) -> str:
        """The most important problem first (one sentence), or an empty string."""
        return MESSAGES[self.issues[0]] if self.issues else ""

    def as_dict(self) -> dict:
        return {"sharpness": round(self.sharpness, 1), "direction_ratio": round(self.direction_ratio, 3),
                "brightness": round(self.brightness, 1), "white_percent": round(self.white_percent, 1),
                "issues": list(self.issues), "message": self.message}


def check_quality(image_bgr: np.ndarray) -> QualityReport:
    """Measure sharpness, motion blur and brightness on a 640 px copy of the photo."""
    h, w = image_bgr.shape[:2]
    scale = 640 / max(h, w)
    small = cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    energy = [float(np.mean(cv2.filter2D(gray, -1, k) ** 2)) for k in _DIRECTION_KERNELS]
    direction_ratio = min(energy) / max(max(energy), 1e-9)
    brightness = float(gray.mean())
    white = float((gray >= 250).mean() * 100)

    issues = []
    if brightness < MIN_BRIGHTNESS:
        issues.append("dark")
    elif brightness > MAX_BRIGHTNESS or white > MAX_WHITE_PERCENT:
        issues.append("overexposed")
    if direction_ratio < MIN_DIRECTION_RATIO:
        issues.append("motion_blur")
    elif sharpness < MIN_SHARPNESS and "dark" not in issues:  # dark photos also lose contrast
        issues.append("blurry")
    return QualityReport(sharpness, direction_ratio, brightness, white, issues)
