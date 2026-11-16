# BOTOTOY 2.0: Fruit Detection and Identification Using YOLO

**Course:** CSST 106 - Perception and Computer Vision
**Team:** Kyle Angelo D. Mercado · Ian Patrick Z. Mesias · James Paul L. Panal

BOTOTOY 2.0 detects and identifies multiple fruits in a single image. For every fruit it outputs a
bounding box, the fruit class, a confidence score, and counts per class plus an overall count.

- **Main method:** Ultralytics **YOLO11n**, fine-tuned with transfer learning.
- **Baseline / fallback:** a traditional **OpenCV** pipeline (HSV color segmentation, morphology,
  contours, shape/color features, rule-based classification).

**Supported classes (6):** Apple · Banana · Grape · Orange · Pineapple · Watermelon

> Scope: detection and identification only. No ripeness, freshness, disease, weight or nutrition analysis.

---

## Development status

| Phase | Status |
|---|---|
| 1. Inspection and planning | ✅ Done |
| 2. Environment and dataset preparation | ✅ Done ([dataset audit report](outputs/reports/dataset/DATASET_AUDIT.md)) |
| 3. OpenCV baseline | ✅ Done ([baseline report](outputs/baseline/BASELINE_REPORT.md)) |
| 4. YOLO11n training (Kaggle GPU) and evaluation | ⏳ Not started |
| 5. Streamlit user interface | ⏳ Not started |
| 6. Integration and end-to-end testing | ⏳ Not started |
| 7. Final documentation | ⏳ Not started |

This README is updated at the end of each phase and only describes features that already work.

---

## Dataset

| | |
|---|---|
| Source | GitHub mirror [lightly-ai/dataset_fruits_detection](https://github.com/lightly-ai/dataset_fruits_detection) (pinned commit `e626559`), a copy of the Kaggle dataset [lakshaytyagi01/fruit-detection](https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection) |
| License | CC0-1.0 (LICENSE file in the GitHub mirror) |
| Size | 8,479 JPG images, all 640x640, 36,922 bounding boxes, YOLO format |
| Unique source photos | 4,571 (the training split contains Roboflow-augmented copies) |

**Final split (verified leakage-free, see the audit report):**

| Split | Images | Unique photos | Share of unique photos | Boxes |
|---|---|---|---|---|
| train | 7,108 | 3,200 | 70.0% | 32,059 |
| val | 686 | 686 | 15.0% | 2,491 |
| test (locked until final evaluation) | 682 | 682 | 14.9% | 2,369 |

The original training split was kept, and the un-augmented original valid + test images were re-split
50/50, stratified by class. Three validation images that were near-duplicates of training images
were excluded. The split file lists are committed in [data/splits/](data/splits/), so every machine
reproduces the exact same split.

The images themselves are **not** stored in this repository. Download them with the script below.

---

## Setup

Requirements: Python 3.12, Git, about 3 GB of free disk space (virtual environment + dataset).

```bash
git clone https://github.com/imPickleRiick/BOTOTOY-2.0-Fruit-Detection-and-Identification-Using-YOLO.git
cd BOTOTOY-2.0-Fruit-Detection-and-Identification-Using-YOLO

python -m venv .venv
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# Windows (Git Bash):     source .venv/Scripts/activate
# macOS / Linux:          source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset preparation

Run these commands from the project root, with the virtual environment activated:

```bash
# 1. Download the dataset (~500 MB) into data/raw/
python -m src.data.download_dataset

# 2. Validate images and labels, find duplicates and leakage (~2 min)
python -m src.data.dataset_validator

# 3. Build data/processed/{train,val,test} and data/data.yaml from the committed split
python -m src.data.dataset_splitter --from-manifest

# 4. Statistics, charts, 24 annotated samples and the audit report
python -m src.data.dataset_analyzer
```

`dataset_splitter` without `--from-manifest` recomputes the split from scratch. The result is
deterministic (seed 42) and identical to the committed lists.

Outputs:
- `outputs/reports/dataset/DATASET_AUDIT.md`: the full audit report, with figures
- `outputs/reports/dataset/samples/`: annotated ground-truth samples
- `data/data.yaml`: generated dataset config for Ultralytics (machine-specific, not committed)

## OpenCV baseline (traditional computer vision)

Pipeline: resize + Gaussian blur → BGR→HSV → 4 color masks (red, warm, green, purple) →
morphological opening/closing → external contours → area filter → features (mean hue, saturation,
area, aspect ratio, elongation, extent, circularity, solidity) → rule-based classification →
bounding boxes + NMS → counts per class. All thresholds are in `config/config.yaml` (`baseline:`).

```bash
# Detect fruits in your own images or folders (annotated copies + results.json)
python -m src.baseline.pipeline path/to/photo.jpg --out outputs/baseline/demo

# Also save the step-by-step panel (preprocessed / color masks / morphology / contours / detections)
python -m src.baseline.pipeline path/to/photo.jpg --stages

# Measure training-set fruit colors (used to choose the HSV ranges)
python -m src.baseline.color_calibration

# Evaluate on the validation split (precision, recall, AP, counting error, timing)
python -m src.baseline.evaluate_baseline
python -m src.baseline.evaluate_baseline --brightness 0.6   # simulated darker lighting

# Reproduce the improvement log (each design decision measured on validation)
python -m src.baseline.ablation
```

The baseline is intentionally simple and explainable. It works on single fruits against plain
backgrounds, but it is weak on the cluttered real-world images that make up most of this dataset
(validation mAP50 ≈ 0.05). See the [baseline report](outputs/baseline/BASELINE_REPORT.md) for the
results and limitations.

## Tests

```bash
python -m pytest
```

---

## Project structure (current)

```
├── config/config.yaml          # all paths and parameters
├── data/
│   ├── splits/                 # committed split file lists + summary
│   ├── raw/                    # downloaded dataset (git-ignored)
│   └── processed/              # final split folders, hard links (git-ignored)
├── src/
│   ├── data/
│   │   ├── download_dataset.py
│   │   ├── dataset_validator.py
│   │   ├── dataset_splitter.py
│   │   └── dataset_analyzer.py
│   ├── baseline/
│   │   ├── preprocessing.py       # resize, blur, BGR->HSV
│   │   ├── segmentation.py        # HSV masks, morphology, contours, watershed
│   │   ├── feature_extraction.py  # color + shape features
│   │   ├── fruit_classifier.py    # heuristic rules
│   │   ├── pipeline.py            # BaselineDetector + command-line tool
│   │   ├── color_calibration.py   # measures fruit colors on training data
│   │   ├── evaluate_baseline.py   # metrics on val (test locked)
│   │   └── ablation.py            # improvement log
│   └── utils/
│       ├── helpers.py          # config, paths, JSON, system info
│       ├── image_utils.py      # image I/O, box drawing, perceptual hash
│       ├── detections.py       # shared Detection type, NMS, counting, drawing
│       └── metrics.py          # precision, recall, AP (Ultralytics formula), count error
├── outputs/
│   ├── reports/dataset/        # audit report, charts, samples
│   └── baseline/               # calibration, evaluation, demo outputs, report
├── tests/
└── requirements.txt
```

## Known dataset limitations

See section 9 of the [audit report](outputs/reports/dataset/DATASET_AUDIT.md). In short:
- Orange has about 9.6x as many instances as Pineapple.
- Training images are augmented copies of only 3,200 photos.
- Images were stretched to 640x640.
- Some boxes cover a group of fruits rather than a single fruit, so a fruit count means the number of detected boxes.
