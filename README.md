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
| 4. YOLO11n training (Google Colab GPU) and evaluation | ✅ Done ([training](outputs/yolo/training/TRAINING_REPORT.md), [test evaluation](outputs/yolo/eval_test/EVALUATION.md), [baseline vs YOLO](outputs/comparisons/test/COMPARISON.md)) |
| 5. Streamlit user interface | ✅ Done (`streamlit run app.py`) |
| 6. Integration and end-to-end testing | ✅ Done ([system test report](outputs/reports/system_test/SYSTEM_TEST_REPORT.md)) |
| 7. Final documentation | ✅ Done ([final report](docs/FINAL_REPORT.md), [error review](outputs/yolo/eval_test/error_review/ERROR_REVIEW.md), [demo guide](docs/DEMO_GUIDE.md), [defense guide](docs/DEFENSE_GUIDE.md)) |

## Documentation

| Document | For |
|---|---|
| [Final project report](docs/FINAL_REPORT.md) | The whole project in one place: objectives vs results, data, methods, results, error analysis, limitations |
| [Error review](outputs/yolo/eval_test/error_review/ERROR_REVIEW.md) | Every false positive and missed fruit on the test set, sorted by cause; 48 inspected by eye |
| [Presentation and demo guide](docs/DEMO_GUIDE.md) | Presentation outline, live demo script, backup plan |
| [Defense guide](docs/DEFENSE_GUIDE.md) | Each member's module, how to show it, likely questions |
| [Dataset audit](outputs/reports/dataset/DATASET_AUDIT.md) · [Baseline report](outputs/baseline/BASELINE_REPORT.md) · [Training report](outputs/yolo/training/TRAINING_REPORT.md) · [Test evaluation](outputs/yolo/eval_test/EVALUATION.md) · [Comparison](outputs/comparisons/test/COMPARISON.md) · [System test](outputs/reports/system_test/SYSTEM_TEST_REPORT.md) | Detailed phase reports |

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

## Run the app

```bash
streamlit run app.py
# or simply:  python app.py   (also works with the VS Code "Run" button)
```

Your browser opens at http://localhost:8501. Stop the app with Ctrl+C in the terminal. The trained model is already in the repository, so
detection works immediately; the dataset is only needed for training and evaluation. Pages:

| Page | What it does |
|---|---|
| Dashboard | Project summary, the 6 fruit classes, model status, test results, system information |
| Fruit Detection | Upload one photo (or pick a demo image) → boxes, labels, confidences, counts per fruit and total, timing, download annotated image / JSON |
| Batch Detection | Many photos at once → per-image results, combined counts, timing, ZIP download (annotated images + CSV + JSON); unreadable files are reported and skipped |
| YOLO11n Model | How YOLO works, model card, confidence-threshold explorer, same image with the OpenCV baseline |
| OpenCV Baseline | Every processing step (masks, morphology, contours) and the rule decision for every candidate region |
| Model Evaluation | Final test results vs targets, per-class metrics, confusion matrix, PR curve, training curves, baseline vs YOLO, error examples |
| About | Project, course and team |

The sidebar selects the model (YOLO11n or the OpenCV baseline) and the YOLO confidence threshold, NMS IoU
threshold and image size. If the YOLO weights are missing or cannot be loaded, the app automatically uses the
OpenCV baseline and says so. The 8 demo images in `assets/samples/` come from the dataset's test split
(CC0) and are labeled as demonstration data in the app. "Save to outputs/app/" writes results to
`outputs/app/` (git-ignored).

Batch processing also works from the command line:

```bash
python -m src.inference.batch_processor path/to/folder --method yolo --out outputs/app/my_batch
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

## YOLO11n (main method)

**Training runs on Google Colab** because the team's laptops have no NVIDIA GPU. CPU training of 50 epochs
would take more than 100 hours.

1. Open [notebooks/train_yolo_colab.ipynb](notebooks/train_yolo_colab.ipynb) in Google Colab
   (File → Upload notebook, or open it from GitHub).
2. Runtime → Change runtime type → **T4 GPU**, then run the cells in order. The notebook clones this
   repository, downloads the dataset, rebuilds the committed split, and trains YOLO11n. Checkpoints are
   saved to Google Drive, so an interrupted run can be resumed.
3. Download `bototoy_training_results.zip` at the end, then on the laptop:

```bash
# Verify + install the trained weights (models/trained/bototoy_yolo11n.pt) and write the training report
python -m src.yolo.training_report path/to/bototoy_training_results.zip

# Evaluate on the validation split (Ultralytics metrics + our shared metrics, timing, error examples)
python -m src.yolo.evaluate

# Detect fruits in your own images
python -m src.yolo.detector path/to/photo.jpg --out outputs/yolo/demo

# Baseline vs YOLO on the same images with the same metrics
python -m src.evaluation.compare_methods
```

The test split is evaluated **once**, at the end: `python -m src.yolo.evaluate --split test --final`
and `python -m src.evaluation.compare_methods --split test --final`.

Training settings are in `config/config.yaml` under `yolo.train`: YOLO11n pretrained on COCO, image size 640,
up to 50 epochs, batch 16, early-stopping patience 10, SGD with lr 0.01, seed 42. For a quick CPU
pipeline check (not a real model): `python -m src.yolo.train --epochs 1 --fraction 0.02 --imgsz 320 --device cpu --no-copy`.

The trained model is committed as `models/trained/bototoy_yolo11n.pt` (5.5 MB), so the detector works
right after cloning without retraining.

### Results

Trained on a Colab Tesla T4 for 50 epochs (94 min). The best checkpoint (epoch 47) was selected on the
validation split. The test split was evaluated once, after all choices were fixed.

| Target (from the proposal) | Validation | **Test** | Met? |
|---|---|---|---|
| mAP50 ≥ 0.80 | 0.478 | **0.490** | ❌ |
| mAP50-95 ≥ 0.50 | 0.321 | **0.338** | ❌ |
| Recall ≥ 0.70 for every class | 0.34 to 0.54 | **0.32 to 0.68** | ❌ |
| ≤ 1 s per 640 image (end-to-end, laptop CPU) | 67 ms | **69 ms** | ✅ |

**OpenCV baseline vs YOLO11n on the test split** (same images, same metric code):

| | All 682 test images | | 59 plain-background images | |
|---|---|---|---|---|
| | Baseline | YOLO11n | Baseline | YOLO11n |
| mAP50 | 0.049 | **0.486** | 0.167 | **0.750** |
| Precision / recall (conf ≥ 0.25) | 0.072 / 0.075 | **0.668 / 0.410** | 0.160 / 0.155 | **0.776 / 0.668** |
| Exact fruit count | 15.8% | **45.0%** | 35.6% | **59.3%** |
| Time per image | 35 ms | 62 ms | 24 ms | 62 ms |

YOLO11n is about 10 times as accurate as the baseline (mAP50) and fast enough for interactive use on a laptop CPU.
It still misses the accuracy targets. Why:

- **Missed fruits, not wrong labels.** When YOLO finds a fruit it usually names it correctly: 65–88% per class in
  the confusion matrix, with almost no fruit-to-fruit confusion. Most errors are fruits it never detects
  (signed count error −1.7 fruits/image) and false alarms on background.
- **Crowded scenes are labeled inconsistently.** In some images every orange in a crate has its own box; in
  others one box covers the whole pile. The model learns both styles and often draws one group box where
  the ground truth has dozens of single boxes, which counts as many misses at once (see `outputs/yolo/eval_test/examples/fn_*`).
- **The training set is small in practice.** Its 7,108 images are augmented copies of only 3,200 photos. Validation
  mAP levels off after about epoch 30 while training loss keeps falling, so more epochs alone would not close the gap.
- On controlled, plain-background images, the conditions the proposal targets, test mAP50 is 0.750,
  close to the 0.80 target.

The [error review](outputs/yolo/eval_test/error_review/ERROR_REVIEW.md) sorts all 1,977 test errors by cause.
Only 2.3% are a wrong fruit name. 49.8% of the missed fruits lie inside a box the model drew around their whole
group, and 30.0% are smaller than 1% of the image. Reproduce it with
`python -m src.evaluation.error_review --split test --final`.

## Tests

```bash
# Unit and page tests (82 tests, ~35 s)
python -m pytest

# End-to-end system test (~4 min): every part of the system on this computer, the app's inference path
# re-run on the whole test split, 12 unseen photos, timing and memory -> outputs/reports/system_test/
python -m src.evaluation.system_test            # add --quick to skip the test-split re-run
```

Latest result on the team laptop (AMD Ryzen 5 5600U, 5.9 GB RAM, CPU only): **16/16 checks passed**,
and a browser end-to-end run (real uploads and downloads) passed 11/11 steps. YOLO11n takes about 66 ms per
640 px image (95th percentile 77 ms), and named the right fruit in 11 of 12 unseen photos from Wikimedia
Commons (`assets/unseen/`). See the [system test report](outputs/reports/system_test/SYSTEM_TEST_REPORT.md).

**Large photos:** phone photos (12 MP and more) are supported. The app keeps a 1600 px preview plus the
full-resolution annotated JPEG for download, so memory stays low; batch outputs are saved at up to 2048 px
(box coordinates in the CSV/JSON always refer to the original image).

---

## Project structure

```
├── app.py                      # Streamlit app entry point (streamlit run app.py)
├── config/config.yaml          # all paths and parameters
├── docs/                       # final report, demo guide, defense guide
├── assets/samples/             # 8 demo images from the test split (CC0)
├── assets/unseen/              # 12 public-domain photos from outside the dataset (system test)
├── notebooks/train_yolo_colab.ipynb   # YOLO11n training on Google Colab
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
│   ├── yolo/
│   │   ├── train.py               # transfer-learning training (Colab or local)
│   │   ├── detector.py            # YoloDetector + command-line tool
│   │   ├── evaluate.py            # metrics, timing, error analysis (test locked)
│   │   └── training_report.py     # import Colab results, verify + install weights
│   ├── evaluation/
│   │   ├── compare_methods.py     # baseline vs YOLO on the same images
│   │   ├── error_review.py        # every FP / missed fruit, sorted by cause
│   │   └── system_test.py         # end-to-end system test + report
│   ├── inference/
│   │   ├── image_processor.py     # decode upload, run YOLO or baseline (with fallback), timing
│   │   ├── batch_processor.py     # many images, summary, CSV/ZIP, command-line tool
│   │   └── visualization.py       # tables, image encoding, stage titles
│   ├── app/
│   │   ├── common.py              # cached models, sidebar settings, result display, charts
│   │   └── pages/                 # one module per page (dashboard, detection, batch, ...)
│   └── utils/
│       ├── helpers.py          # config, paths, JSON, system info
│       ├── image_utils.py      # image I/O, box drawing, perceptual hash
│       ├── detections.py       # shared Detection type, NMS, counting, drawing
│       └── metrics.py          # precision, recall, AP (Ultralytics formula), count error
├── models/trained/bototoy_yolo11n.pt   # trained YOLO11n weights (5.5 MB)
├── outputs/
│   ├── reports/dataset/        # audit report, charts, samples
│   ├── baseline/               # calibration, evaluation, demo outputs, report
│   ├── yolo/                   # training report + curves, val and test evaluations, error review
│   ├── comparisons/test/       # baseline vs YOLO11n on the test split
│   └── reports/system_test/    # end-to-end test report, unseen-photo results, screenshots
├── tests/
└── requirements.txt
```

## Known dataset limitations

See section 9 of the [audit report](outputs/reports/dataset/DATASET_AUDIT.md). In short:
- Orange has about 9.6x as many instances as Pineapple.
- Training images are augmented copies of only 3,200 photos.
- Images were stretched to 640x640.
- Some boxes cover a group of fruits rather than a single fruit, so a fruit count means the number of detected boxes.
