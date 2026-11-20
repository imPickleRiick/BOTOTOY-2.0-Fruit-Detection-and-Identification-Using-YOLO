"""Hard-case test: both detectors on the 12 unseen photos under difficult conditions.

The 12 photos in assets/unseen/ come from Wikimedia Commons and were never used for training or
tuning. Each is modified to simulate a hard condition (poor lighting, blur, angle, occlusion,
low quality), then run through YOLO11n and the OpenCV baseline with the app's default settings.

A case PASSES if at least one detection has the expected fruit type (assets/unseen/expected.csv,
decided by looking at the photo). Wrong extra types are listed but do not fail a case, because
counts in crowded photos are not reliable (see the system test report).

Usage (from the project root):
    python -m src.evaluation.hard_cases
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from src.utils.helpers import ensure_dir, save_json
from src.utils.image_utils import read_image, write_image

UNSEEN = Path("assets/unseen")
OUT = Path("outputs/reports/hard_cases")


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), borderValue=(255, 255, 255))


def _occlude(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    out[int(h * 0.3):int(h * 0.7), int(w * 0.3):int(w * 0.7)] = (128, 128, 128)  # gray block, 16% of the image
    return out


def _motion_blur(img: np.ndarray, k: int = 31) -> np.ndarray:
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def _low_res(img: np.ndarray, side: int = 128) -> np.ndarray:
    h, w = img.shape[:2]
    s = side / max(h, w)
    small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _jpeg(img: np.ndarray, quality: int = 5) -> np.ndarray:
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR)


CONDITIONS = {
    "original": ("Original photo", lambda im: im),
    "dark": ("Poor lighting: brightness x0.35", lambda im: cv2.convertScaleAbs(im, alpha=0.35)),
    "overexposed": ("Overexposed: brightness x1.8", lambda im: cv2.convertScaleAbs(im, alpha=1.8)),
    "blur": ("Out of focus: Gaussian blur 25x25", lambda im: cv2.GaussianBlur(im, (25, 25), 0)),
    "motion_blur": ("Motion blur: 31 px horizontal", _motion_blur),
    "tilted_30": ("Camera angle: rotated 30 degrees", lambda im: _rotate(im, 30)),
    "rotated_90": ("Camera angle: rotated 90 degrees", lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)),
    "occluded": ("Occlusion: gray block over the center (16% of the image)", _occlude),
    "low_res": ("Low resolution: 128 px, upscaled", _low_res),
    "jpeg_q5": ("Heavy compression: JPEG quality 5", _jpeg),
    "grayscale": ("No color: grayscale", lambda im: cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)),
}


def main() -> int:
    from src.inference.image_processor import FruitDetectionEngine

    with open(UNSEEN / "expected.csv", encoding="utf-8") as f:
        expected = {r["image"]: set(r["expected_classes"].split(";")) for r in csv.DictReader(f)}
    engine = FruitDetectionEngine()
    ensure_dir(OUT / "examples")

    rows = []
    for name, exp in expected.items():
        original = read_image(UNSEEN / name)
        for cond, (desc, fn) in CONDITIONS.items():
            image = fn(original)
            row = {"image": name, "condition": cond, "description": desc, "expected": ";".join(sorted(exp))}
            for method in ("yolo", "baseline"):
                r = engine.run(image, method=method)
                found = {c for c, n in r.counts.items() if n}
                row[f"{method}_actual"] = ", ".join(f"{c} {n}" for c, n in r.counts.items() if n) or "nothing"
                row[f"{method}_pass"] = bool(exp & found)
                row[f"{method}_wrong_types"] = ", ".join(sorted(found - exp))
                row[f"{method}_ms"] = round(r.timing_ms.get("end_to_end", 0.0), 1)
                if method == "yolo" and name == "mandarins_on_plate.jpg":
                    write_image(OUT / "examples" / f"{cond}.jpg", r.annotated)
            rows.append(row)
        print(f"  {name}: done", flush=True)

    for i, r in enumerate(rows, 1):
        r["id"] = f"HC{i:03d}"
    summary = {}
    for cond in CONDITIONS:
        sel = [r for r in rows if r["condition"] == cond]
        summary[cond] = {"description": CONDITIONS[cond][0], "cases": len(sel),
                         "yolo_pass": sum(r["yolo_pass"] for r in sel),
                         "baseline_pass": sum(r["baseline_pass"] for r in sel),
                         "yolo_cases_with_wrong_types": sum(bool(r["yolo_wrong_types"]) for r in sel),
                         "baseline_cases_with_wrong_types": sum(bool(r["baseline_wrong_types"]) for r in sel)}
    save_json({"summary": summary, "cases": rows}, OUT / "hard_cases.json")

    # Panel: YOLO11n on one photo under every condition.
    tiles = []
    for cond in CONDITIONS:
        t = cv2.resize(read_image(OUT / "examples" / f"{cond}.jpg"), (360, 360))
        cv2.rectangle(t, (0, 0), (360, 30), (255, 255, 255), -1)
        cv2.putText(t, cond, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
        tiles.append(t)
    tiles += [np.full((360, 360, 3), 255, np.uint8)] * (-len(tiles) % 4)
    write_image(OUT / "hard_cases_panel.jpg",
                np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))

    lines = [
        "# Hard-case test log",
        "",
        "Generated by `python -m src.evaluation.hard_cases`. The 12 photos in `assets/unseen/` (Wikimedia Commons, "
        "never used for training or tuning) were modified to simulate hard conditions and run through both "
        "detectors with the app's default settings (YOLO11n confidence 0.25).",
        "",
        "A case **passes** if at least one detection has the expected fruit type (`assets/unseen/expected.csv`). "
        "Wrong extra types are listed separately; they do not fail a case.",
        "",
        "## Summary by condition (12 photos each)",
        "",
        "| Condition | YOLO11n pass | Baseline pass | YOLO11n cases with a wrong extra type | Baseline cases with a wrong extra type |",
        "|---|---|---|---|---|",
    ]
    for cond, s in summary.items():
        lines.append(f"| {s['description']} | **{s['yolo_pass']}/12** | {s['baseline_pass']}/12 | "
                     f"{s['yolo_cases_with_wrong_types']} | {s['baseline_cases_with_wrong_types']} |")
    tot_y = sum(s["yolo_pass"] for s in summary.values())
    tot_b = sum(s["baseline_pass"] for s in summary.values())
    lines += ["", f"**Total: YOLO11n {tot_y}/{len(rows)}, OpenCV baseline {tot_b}/{len(rows)}.**", "",
              "YOLO11n on `mandarins_on_plate.jpg` under every condition:", "",
              "![hard cases](hard_cases_panel.jpg)", "",
              "## Every case", "",
              "| ID | Input | Condition | Expected | YOLO11n actual | YOLO11n | Baseline actual | Baseline |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['image']} | {r['condition']} | {r['expected']} | {r['yolo_actual']} | "
                     f"{'PASS' if r['yolo_pass'] else 'FAIL'} | {r['baseline_actual']} | "
                     f"{'PASS' if r['baseline_pass'] else 'FAIL'} |")
    (OUT / "HARD_CASES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"YOLO11n {tot_y}/{len(rows)}, baseline {tot_b}/{len(rows)} -> {OUT / 'HARD_CASES.md'}")
    for cond, s in summary.items():
        print(f"  {cond:12s} yolo {s['yolo_pass']:2d}/12  baseline {s['baseline_pass']:2d}/12")
    return 0


if __name__ == "__main__":
    sys.exit(main())
