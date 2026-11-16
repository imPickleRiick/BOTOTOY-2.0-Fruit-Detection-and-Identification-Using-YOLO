"""Image helpers shared by the dataset tools, the OpenCV baseline and YOLO inference."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# One fixed BGR color per class so every visualization in the project is consistent.
CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "Apple": (40, 40, 220),        # red
    "Banana": (0, 215, 255),       # yellow
    "Grape": (150, 40, 140),       # purple
    "Orange": (0, 140, 255),       # orange
    "Pineapple": (200, 150, 0),    # blue
    "Watermelon": (60, 170, 60),   # green
}
DEFAULT_COLOR_BGR = (200, 200, 200)

# Deeper versions of the same hues for charts (bars, legends). The bright box colors above are
# too light on a white background (banana yellow); these pass contrast and color-blindness
# checks in both light and dark themes while staying recognisably the same fruit color.
CLASS_CHART_COLORS_HEX: dict[str, str] = {
    "Apple": "#D63031", "Banana": "#B88A00", "Grape": "#9B45B0",
    "Orange": "#E86A10", "Pineapple": "#0088C2", "Watermelon": "#2F9132",
}


def read_image(path: str | Path) -> np.ndarray | None:
    """Read an image as a BGR array; returns None if the file is missing or corrupt.

    cv2.imread() fails on Windows paths with non-ASCII characters, so we read the
    raw bytes with NumPy and decode them with cv2.imdecode() instead.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: str | Path, image: np.ndarray, quality: int = 92) -> bool:
    """Save an image (unicode-safe on Windows). Returns True on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(str(path))
    return bool(ok)


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convert a normalized YOLO box (center x, center y, width, height) to pixel corners."""
    x1 = int(round((xc - w / 2) * img_w))
    y1 = int(round((yc - h / 2) * img_h))
    x2 = int(round((xc + w / 2) * img_w))
    y2 = int(round((yc + h / 2) * img_h))
    # Clip to the image so drawing never goes out of bounds.
    return max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)


def draw_box(image: np.ndarray, xyxy: tuple[int, int, int, int], label: str,
             color: tuple[int, int, int] = DEFAULT_COLOR_BGR, thickness: int = 2,
             font_scale: float = 0.5) -> np.ndarray:
    """Draw one bounding box with a filled label tag (modifies and returns `image`)."""
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if label:
        font, scale, txt_th = cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, round(font_scale * 2))
        (tw, th), baseline = cv2.getTextSize(label, font, scale, txt_th)
        ty = y1 - 4 if y1 - th - baseline - 4 >= 0 else y1 + th + 4  # put tag inside if no room above
        cv2.rectangle(image, (x1, ty - th - baseline), (x1 + tw + 4, ty + baseline - 2), color, -1)
        # Black text on light colors, white text on dark colors.
        text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
        cv2.putText(image, label, (x1 + 2, ty - 2), font, scale, text_color, txt_th, cv2.LINE_AA)
    return image


def dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """Difference hash: a 64-bit fingerprint that stays the same for visually identical images.

    The image is shrunk to (hash_size+1) x hash_size grayscale pixels and each bit records
    whether a pixel is brighter than its right neighbour. Near-duplicate images (re-encoded,
    slightly resized) have hashes that differ in only a few bits (small Hamming distance).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def make_grid(images: list[np.ndarray], cols: int, cell_size: int = 320) -> np.ndarray:
    """Tile images into one grid image (each resized to cell_size x cell_size)."""
    if not images:
        raise ValueError("make_grid() needs at least one image")
    cells = [cv2.resize(im, (cell_size, cell_size)) for im in images]
    rows = (len(cells) + cols - 1) // cols
    blank = np.full((cell_size, cell_size, 3), 255, dtype=np.uint8)
    cells += [blank] * (rows * cols - len(cells))
    return np.vstack([np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)])
