"""Feasibility spike (checkpoint 1): load 5 images and run one computer-vision operation on them.

Operation: HSV color thresholding + morphology + external contours (the core of the OpenCV
baseline), drawn on each image. Output: one panel image with the original (top) and the result
(bottom) for each of the 5 images.

Usage (from the project root):
    python -m src.feasibility_spike
    python -m src.feasibility_spike path/to/folder --out outputs/reports/feasibility_spike
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Pixels that are colorful enough to be fruit (any hue, saturation >= 90, value >= 60).
LOWER, UPPER = np.array([0, 90, 60]), np.array([179, 255, 255])


def spike(image: np.ndarray) -> tuple[np.ndarray, int]:
    """HSV threshold -> opening/closing -> contours >= 1% of the image. Returns (drawn image, region count)."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER, UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep = [c for c in contours if cv2.contourArea(c) >= 0.01 * image.shape[0] * image.shape[1]]
    out = image.copy()
    cv2.drawContours(out, keep, -1, (0, 255, 0), 3)
    for c in keep:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return out, len(keep)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", nargs="?", default="assets/samples")
    parser.add_argument("--out", default="outputs/reports/feasibility_spike")
    args = parser.parse_args()

    paths = sorted(Path(args.folder).glob("*.jpg"))[:5]
    if len(paths) < 5:
        print(f"ERROR: need 5 .jpg images in {args.folder}, found {len(paths)}", file=sys.stderr)
        return 1
    cells_top, cells_bottom = [], []
    for p in paths:
        image = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        result, n = spike(image)
        print(f"{p.name}: {image.shape[1]}x{image.shape[0]}, {n} candidate fruit regions")
        top, bottom = cv2.resize(image, (320, 320)), cv2.resize(result, (320, 320))
        cv2.putText(bottom, f"{n} regions", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cells_top.append(top)
        cells_bottom.append(bottom)
    panel = np.vstack([np.hstack(cells_top), np.hstack(cells_bottom)])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", panel)[1].tofile(str(out / "spike_output.jpg"))
    print(f"Saved {out / 'spike_output.jpg'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
