"""The assistive fruit reader: photo -> quality check -> YOLO11n -> one spoken sentence.

Model choice: the 30-class tropical fruit model (config/config_fruit30.yaml) when its trained weights
exist, otherwise the 6-class v1 model (config/config.yaml). Both use the same engine and fallback.

Usage (command line):
    python -m src.assist.reader photo.jpg [more photos ...]
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.assist.describe import full_message
from src.assist.fruit_names import display_name
from src.assist.quality import QualityReport, check_quality
from src.inference.image_processor import YOLO, FruitDetectionEngine, InferenceResult, decode_image
from src.utils.detections import Detection, draw_detections
from src.utils.helpers import PROJECT_ROOT, load_config, resolve_path

FRUIT30_CONFIG = PROJECT_ROOT / "config" / "config_fruit30.yaml"
V1_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


def reader_config() -> tuple[dict, str]:
    """(config, model description): the 30-fruit model if its weights exist, else the 6-fruit model."""
    cfg30 = load_config(FRUIT30_CONFIG)
    if resolve_path(cfg30["yolo"]["trained_weights"]).is_file():
        return cfg30, f"YOLO11n, {len(cfg30['classes'])} tropical fruits"
    cfg = load_config(V1_CONFIG)
    return cfg, f"YOLO11n, {len(cfg['classes'])} fruits (the 30-fruit model is not installed yet)"


@dataclass
class ReaderResult:
    message: str                     # what is spoken
    quality: QualityReport
    detection: InferenceResult
    annotated: np.ndarray            # boxes labeled with readable (English + Filipino) names
    timing_ms: dict = field(default_factory=dict)


class FruitReader:
    def __init__(self, config: dict | None = None, model_note: str | None = None):
        if config is None:
            config, model_note = reader_config()
        self.config = config
        self.model_note = model_note or ""
        self.engine = FruitDetectionEngine(config)

    @property
    def classes(self) -> list[str]:
        return self.config["classes"]

    def read(self, image: np.ndarray, conf: float | None = None, filipino: bool = True) -> ReaderResult:
        t0 = time.perf_counter()
        quality = check_quality(image)
        t1 = time.perf_counter()
        result = self.engine.run(image, method=YOLO, conf=conf)
        t2 = time.perf_counter()
        width = image.shape[1]
        message = full_message(result.detections, width, quality.message, filipino)
        if result.notice:  # YOLO missing -> baseline was used; say it honestly
            message = "Warning: the main model is not available, results may be wrong. " + message
        readable = [Detection(d.class_name, d.confidence, d.box, d.class_id,
                              {"label": display_name(d.class_name, filipino)}) for d in result.detections]
        annotated = draw_detections(image, readable)
        timing = {"quality_check": round((t1 - t0) * 1e3, 1), "detection": round((t2 - t1) * 1e3, 1),
                  "total": round((time.perf_counter() - t0) * 1e3, 1)}
        return ReaderResult(message, quality, result, annotated, timing)

    def read_bytes(self, data: bytes, name: str = "photo", **kwargs) -> ReaderResult:
        return self.read(decode_image(data, name), **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--no-filipino", action="store_true")
    args = parser.parse_args()
    reader = FruitReader()
    print(f"Model: {reader.model_note}")
    for p in args.images:
        r = reader.read_bytes(Path(p).read_bytes(), Path(p).name, filipino=not args.no_filipino)
        print(f"\n{Path(p).name}  ({r.timing_ms['total']:.0f} ms)\n  {r.message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
