"""Batch processing: run detection on many images independently and summarize the results.

Each image is processed on its own; an unreadable or unsupported file is recorded as an error
and the batch continues. To keep memory low on a laptop, only the annotated image is kept,
JPEG-encoded, not the decoded pixel arrays.

Usage (command line):
    python -m src.inference.batch_processor photo1.jpg folder/ --method yolo --out outputs/app/batch_cli
"""
from __future__ import annotations

import argparse
import csv
import io
import statistics
import sys
import time
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.inference.image_processor import (BASELINE, METHOD_LABELS, YOLO, FruitDetectionEngine,
                                           InvalidImageError)
from src.inference.visualization import encode_image, fit_within, safe_stem
from src.utils.helpers import ensure_dir, save_json

MAX_BATCH_IMAGES = 100
ANNOTATED_MAX_SIDE = 2048   # batch outputs are capped at this size to keep memory low (boxes in the
                            # JSON/CSV stay in original-image pixels)


@dataclass
class BatchItem:
    name: str                       # uploaded file name
    output_stem: str                # unique, file-system-safe name used for outputs
    ok: bool
    error: str | None = None
    method: str | None = None       # method actually used
    image_size: tuple[int, int] | None = None
    counts: dict[str, int] = field(default_factory=dict)
    total: int = 0
    detections: list[dict] = field(default_factory=list)
    timing_ms: dict[str, float] = field(default_factory=dict)
    notice: str | None = None
    annotated_jpeg: bytes | None = None

    @property
    def annotated_name(self) -> str:
        return f"{self.output_stem}_{self.method}.jpg"


@dataclass
class BatchResult:
    items: list[BatchItem]
    classes: list[str]
    requested_method: str
    settings: dict
    wall_time_ms: float
    _zip_cache: bytes | None = field(default=None, repr=False)

    @property
    def succeeded(self) -> list[BatchItem]:
        return [i for i in self.items if i.ok]

    @property
    def failed(self) -> list[BatchItem]:
        return [i for i in self.items if not i.ok]

    def combined_counts(self) -> dict[str, int]:
        return {c: sum(i.counts.get(c, 0) for i in self.succeeded) for c in self.classes}

    @property
    def total_fruits(self) -> int:
        return sum(i.total for i in self.succeeded)

    def timing_summary(self) -> dict[str, float | None]:
        times = [i.timing_ms["end_to_end"] for i in self.succeeded]
        if not times:
            return {"mean_ms": None, "median_ms": None, "max_ms": None, "wall_time_ms": round(self.wall_time_ms, 1)}
        return {"mean_ms": round(statistics.mean(times), 1), "median_ms": round(statistics.median(times), 1),
                "max_ms": round(max(times), 1), "wall_time_ms": round(self.wall_time_ms, 1)}

    def rows(self) -> list[dict]:
        """One table row per image (used for the on-screen table and results.csv)."""
        out = []
        for i in self.items:
            row = {"Image": i.name, "Status": "ok" if i.ok else "error",
                   "Method": METHOD_LABELS.get(i.method, "") if i.method else "", "Total": i.total if i.ok else None}
            row |= {c: (i.counts.get(c, 0) if i.ok else None) for c in self.classes}
            row["Time (ms)"] = i.timing_ms.get("end_to_end") if i.ok else None
            row["Error"] = i.error or ""
            out.append(row)
        return out

    def summary(self) -> dict:
        return {"created": datetime.now().isoformat(timespec="seconds"),
                "requested_method": METHOD_LABELS[self.requested_method], "settings": self.settings,
                "images": len(self.items), "succeeded": len(self.succeeded), "failed": len(self.failed),
                "total_fruits": self.total_fruits, "combined_counts": self.combined_counts(),
                "timing": self.timing_summary(),
                "per_image": [{"image": i.name, "ok": i.ok, "error": i.error,
                               "method": METHOD_LABELS.get(i.method) if i.method else None,
                               "image_size": i.image_size, "total": i.total, "counts": i.counts,
                               "timing_ms": i.timing_ms, "notice": i.notice, "detections": i.detections}
                              for i in self.items]}

    def to_csv(self) -> str:
        rows = self.rows()
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else ["Image"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()

    def to_zip(self) -> bytes:
        """annotated/<image>_<method>.jpg for every processed image + results.csv + summary.json.
        Built once and reused (the page asks for it on every refresh)."""
        import json
        if self._zip_cache is not None:
            return self._zip_cache
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for i in self.succeeded:
                z.writestr(f"annotated/{i.annotated_name}", i.annotated_jpeg)
            z.writestr("results.csv", self.to_csv())
            z.writestr("summary.json", json.dumps(self.summary(), indent=2))
        self._zip_cache = buf.getvalue()
        return self._zip_cache


def _unique_stems(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    stems = []
    for n in names:
        s = safe_stem(n)
        k = seen.get(s.lower(), 0)
        seen[s.lower()] = k + 1
        stems.append(s if k == 0 else f"{s}_{k + 1}")
    return stems


def process_batch(files: Iterable[tuple[str, bytes]], engine: FruitDetectionEngine, method: str = YOLO,
                  conf: float | None = None, iou: float | None = None, imgsz: int | None = None,
                  show_confidence: bool = True,
                  progress: Callable[[int, int, str], None] | None = None) -> BatchResult:
    """Run detection on every (file name, file bytes) pair independently."""
    files = list(files)
    if not files:
        raise ValueError("No images to process.")
    if len(files) > MAX_BATCH_IMAGES:
        raise ValueError(f"Too many images ({len(files)}); the limit is {MAX_BATCH_IMAGES} per batch.")
    stems = _unique_stems([n for n, _ in files])
    items: list[BatchItem] = []
    t0 = time.perf_counter()
    for k, ((name, data), stem) in enumerate(zip(files, stems), 1):
        try:
            r = engine.run_bytes(data, name, method=method, conf=conf, iou=iou, imgsz=imgsz,
                                 show_confidence=show_confidence)
            h, w = r.image.shape[:2]
            items.append(BatchItem(name, stem, True, method=r.method, image_size=(w, h), counts=r.counts,
                                   total=r.total, detections=[d.as_dict() for d in r.detections],
                                   timing_ms=r.timing_ms, notice=r.notice,
                                   annotated_jpeg=encode_image(fit_within(r.annotated, ANNOTATED_MAX_SIDE))))
        except InvalidImageError as exc:
            items.append(BatchItem(name, stem, False, error=str(exc)))
        except Exception as exc:  # one bad image must not stop the whole batch
            items.append(BatchItem(name, stem, False, error=f"{name}: processing failed ({type(exc).__name__}: {exc})"))
        if progress:
            progress(k, len(files), name)
    settings = {"method": METHOD_LABELS[method]}
    if method == YOLO:
        d = engine.yolo_defaults
        settings |= {"confidence": conf if conf is not None else d["conf"], "iou_nms": iou if iou is not None else d["iou"],
                     "image_size": imgsz or d["imgsz"]}
    return BatchResult(items, engine.classes, method, settings, (time.perf_counter() - t0) * 1e3)


def save_batch(batch: BatchResult, out_dir: str | Path) -> Path:
    """Write annotated images, results.csv and summary.json to a folder."""
    out = ensure_dir(out_dir)
    ann = ensure_dir(out / "annotated")
    for i in batch.succeeded:
        (ann / i.annotated_name).write_bytes(i.annotated_jpeg)
    (out / "results.csv").write_text(batch.to_csv(), encoding="utf-8")
    save_json(batch.summary(), out / "summary.json")
    return out


def main() -> int:
    from src.utils.helpers import list_images, load_config
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="image files and/or folders")
    parser.add_argument("--method", choices=[YOLO, BASELINE], default=YOLO)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--out", default=None, help="default outputs/app/batch_<timestamp>")
    args = parser.parse_args()

    cfg = load_config()
    paths: list[Path] = []
    for p in map(Path, args.inputs):
        paths.extend(list_images(p, cfg["dataset"]["image_extensions"]) if p.is_dir() else [p])
    files = [(p.name, p.read_bytes() if p.is_file() else b"") for p in paths]
    engine = FruitDetectionEngine(cfg)
    try:
        batch = process_batch(files, engine, args.method, args.conf, args.iou, args.imgsz,
                              progress=lambda k, n, name: print(f"  {k}/{n} {name}", flush=True))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    out = save_batch(batch, args.out or f"outputs/app/batch_{datetime.now():%Y%m%d-%H%M%S}")
    t = batch.timing_summary()
    print(f"{len(batch.succeeded)} processed, {len(batch.failed)} failed, {batch.total_fruits} fruits "
          f"{ {k: v for k, v in batch.combined_counts().items() if v} }; mean {t['mean_ms']} ms/image")
    for i in batch.failed:
        print(f"  ERROR {i.error}")
    print(f"Saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
