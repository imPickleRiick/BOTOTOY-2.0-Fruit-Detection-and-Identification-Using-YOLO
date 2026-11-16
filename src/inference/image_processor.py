"""Single-image inference: one entry point for both detectors.

The user interface (and any script) calls `FruitDetectionEngine.run()` / `run_bytes()`, which
  1. decodes the uploaded file into a BGR image (`decode_image`, with clear errors for bad files),
  2. runs YOLO11n (main method) or the OpenCV baseline,
  3. draws the labeled boxes and times every step,
and returns an `InferenceResult`: detections, counts per class, total count, annotated image and
timings. If the YOLO weights are missing or cannot be loaded, the engine falls back to the
OpenCV baseline automatically and explains why in `result.notice`.
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.baseline.pipeline import BaselineDetector
from src.utils.detections import Detection, draw_detections
from src.utils.helpers import PROJECT_ROOT, ensure_dir, load_config, resolve_path, save_json
from src.utils.image_utils import write_image

YOLO, BASELINE = "yolo", "baseline"
METHOD_LABELS = {YOLO: "YOLO11n", BASELINE: "OpenCV baseline"}
SUPPORTED_FORMATS = ("jpg", "jpeg", "png", "bmp", "webp")
SUPPORTED_FORMATS_TEXT = "JPG, PNG, BMP, WEBP"
MAX_PIXELS = 50_000_000   # ~50 megapixels: larger files are rejected instead of exhausting memory
MIN_SIDE = 32             # smaller images carry too little detail to detect anything


class InvalidImageError(ValueError):
    """The file is empty, corrupt, not an image, or outside the supported size range."""


def decode_image(data: bytes, name: str = "image") -> np.ndarray:
    """Decode uploaded file bytes into an 8-bit, 3-channel BGR image (OpenCV convention).

    - grayscale images are expanded to 3 channels,
    - transparent PNG/WEBP images are placed on a white background,
    - JPEG EXIF orientation (phone photos) is applied, so the image is shown upright.
    """
    if not data:
        raise InvalidImageError(f"{name}: the file is empty.")
    buf = np.frombuffer(data, np.uint8)
    raw = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise InvalidImageError(f"{name}: not a readable image. Supported formats: {SUPPORTED_FORMATS_TEXT}.")
    h, w = raw.shape[:2]
    if h * w > MAX_PIXELS:
        raise InvalidImageError(f"{name}: image is too large ({w}x{h}); the limit is {MAX_PIXELS // 1_000_000} megapixels.")
    if min(h, w) < MIN_SIDE:
        raise InvalidImageError(f"{name}: image is too small ({w}x{h}); each side must be at least {MIN_SIDE} px.")
    if raw.ndim == 3 and raw.shape[2] == 4:
        if raw.dtype != np.uint8:                       # 16-bit PNG
            raw = (raw / 257).astype(np.uint8)
        alpha = raw[:, :, 3:4].astype(np.float32) / 255.0
        return (raw[:, :, :3] * alpha + 255.0 * (1.0 - alpha)).round().astype(np.uint8)
    # IMREAD_COLOR: always 8-bit BGR; also applies the EXIF rotation stored by cameras.
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError(f"{name}: not a readable image.")
    return image


@dataclass
class InferenceResult:
    method: str                         # "yolo" or "baseline" (the method actually used)
    image: np.ndarray                   # original image, BGR
    annotated: np.ndarray               # image with labeled boxes, BGR
    detections: list[Detection]
    counts: dict[str, int]              # every class, zero if absent
    total: int
    timing_ms: dict[str, float]         # detector stages + "detector_total" + "end_to_end"
    params: dict                        # settings used (confidence, IoU, image size, ...)
    notice: str | None = None           # e.g. why the baseline was used instead of YOLO
    stages: dict[str, np.ndarray] = field(default_factory=dict)  # baseline step images
    regions: list[dict] = field(default_factory=list)            # baseline candidate regions

    @property
    def method_label(self) -> str:
        return METHOD_LABELS[self.method]

    def summary(self, name: str | None = None) -> dict:
        """JSON-serializable summary (used for downloads and saved results)."""
        h, w = self.image.shape[:2]
        return {"image": name, "method": self.method_label, "image_size": [w, h], "settings": self.params,
                "total": self.total, "counts": self.counts,
                "timing_ms": {k: round(float(v), 2) for k, v in self.timing_ms.items()},
                "notice": self.notice,
                "detections": [d.as_dict() for d in self.detections]}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FruitDetectionEngine:
    """Holds both detectors (loaded once) and runs either of them on an image."""

    def __init__(self, config: dict | None = None, weights: str | Path | None = None, warm_up: bool = True):
        self.config = config or load_config()
        self.classes: list[str] = self.config["classes"]
        self.baseline = BaselineDetector(self.config)
        self.weights_path = resolve_path(weights or self.config["yolo"]["trained_weights"])
        self.yolo = None
        self.yolo_error: str | None = None
        self._lock = threading.Lock()   # the app may serve several browser tabs at once
        try:
            from src.yolo.detector import YoloDetector
            self.yolo = YoloDetector(self.weights_path, self.config)
        except Exception as exc:  # missing/corrupt weights, wrong classes, Ultralytics not installed
            self.yolo_error = str(exc)
        if warm_up and self.yolo is not None:
            # The first prediction is slow (layer fusing, memory allocation); do it now so the
            # first user image is timed like every other one.
            self.yolo.detect(np.zeros((640, 640, 3), np.uint8))

    @property
    def yolo_available(self) -> bool:
        return self.yolo is not None

    @property
    def yolo_defaults(self) -> dict:
        return dict(self.config["yolo"]["inference"])

    def model_info(self) -> dict:
        """Status of both models, for the dashboard."""
        try:
            weights = self.weights_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            weights = str(self.weights_path)
        yolo: dict = {"available": self.yolo_available, "weights": weights, "error": self.yolo_error}
        if self.weights_path.is_file():
            yolo["size_mb"] = round(self.weights_path.stat().st_size / 1e6, 2)
            yolo["sha256"] = file_sha256(self.weights_path)
        if self.yolo is not None:
            net = self.yolo.model.model
            yolo["parameters"] = int(sum(p.numel() for p in net.parameters()))
            yolo["classes"] = list(self.yolo.classes)
            yolo["defaults"] = self.yolo_defaults
        return {"yolo": yolo, "baseline": {"available": True, "classes": self.classes,
                                           "resize_max_side": self.baseline.p["resize_max_side"]}}

    def run(self, image: np.ndarray, method: str = YOLO, conf: float | None = None, iou: float | None = None,
            imgsz: int | None = None, keep_stages: bool = False, show_confidence: bool = True,
            _decode_ms: float = 0.0) -> InferenceResult:
        """Detect fruits in one BGR image. `conf`, `iou`, `imgsz` apply to YOLO only."""
        if method not in METHOD_LABELS:
            raise ValueError(f"Unknown method {method!r}; use {YOLO!r} or {BASELINE!r}")
        notice = None
        if method == YOLO and self.yolo is None:
            method = BASELINE
            notice = f"YOLO11n is not available ({self.yolo_error}). The OpenCV baseline was used instead."

        t0 = time.perf_counter()
        stages, regions = {}, []
        if method == YOLO:
            d = self.yolo_defaults
            params = {"confidence": conf if conf is not None else d["conf"], "iou_nms": iou if iou is not None else d["iou"],
                      "image_size": imgsz or d["imgsz"], "device": d["device"]}
            with self._lock:
                r = self.yolo.detect(image, conf=params["confidence"], iou=params["iou_nms"], imgsz=params["image_size"])
            timing = {"preprocess": r.timing_ms.get("preprocess", 0.0), "inference": r.timing_ms.get("inference", 0.0),
                      "postprocess": r.timing_ms.get("postprocess", 0.0), "detector_total": r.timing_ms["total"]}
        else:
            p = self.baseline.p
            params = {"resize_max_side": p["resize_max_side"], "nms_iou": p["nms_iou"]}
            r = self.baseline.detect(image, keep_stages=keep_stages)
            timing = {k: v for k, v in r.timing_ms.items() if k != "total"} | {"detector_total": r.timing_ms["total"]}
            stages, regions = r.stages, r.regions
        annotated = draw_detections(image, r.detections, show_confidence)
        timing = {k: round(float(v), 2) for k, v in timing.items()}
        if _decode_ms:
            timing = {"decode": round(_decode_ms, 2)} | timing
        timing["end_to_end"] = round((time.perf_counter() - t0) * 1e3 + _decode_ms, 2)
        return InferenceResult(method, image, annotated, r.detections, r.counts, r.total, timing, params,
                               notice, stages, regions)

    def run_bytes(self, data: bytes, name: str = "image", **kwargs) -> InferenceResult:
        """Decode an uploaded file and run detection; the decode time is part of end-to-end time."""
        t0 = time.perf_counter()
        image = decode_image(data, name)
        return self.run(image, _decode_ms=(time.perf_counter() - t0) * 1e3, **kwargs)


def save_result(result: InferenceResult, out_dir: str | Path, stem: str) -> dict[str, Path]:
    """Save the annotated image (JPG) and the detection summary (JSON)."""
    out = ensure_dir(out_dir)
    base = f"{stem}_{result.method}"
    paths = {"image": out / f"{base}.jpg", "json": out / f"{base}.json"}
    write_image(paths["image"], result.annotated)
    save_json(result.summary(stem), paths["json"])
    return paths
