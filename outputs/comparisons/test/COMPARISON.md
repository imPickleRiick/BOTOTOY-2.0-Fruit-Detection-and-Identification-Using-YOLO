# OpenCV baseline vs YOLO11n — `test` split

Same images, same metrics (`src/utils/metrics.py`), same computer (AMD Ryzen 5 5600U with Radeon Graphics, CPU). AP uses all predictions; precision/recall/counts use the app's operating point (YOLO confidence ≥ 0.25; the baseline has no threshold).

## All images (real-world scenes) — 682 images

| Metric | OpenCV baseline | YOLO11n |
|---|---|---|
| mAP@0.50 | 0.049 | 0.486 |
| mAP@0.50:0.95 | 0.031 | 0.336 |
| Precision | 0.072 | 0.668 |
| Recall | 0.075 | 0.410 |
| F1 | 0.060 | 0.498 |
| Localization recall (label ignored) | 0.152 | 0.350 |
| Mean absolute count error (fruits/image) | 3.468 | 2.213 |
| Exact count rate | 15.8% | 45.0% |
| End-to-end time (ms/image) | 35.2 | 62.1 |

| Class | Baseline AP50 | YOLO11n AP50 | Baseline recall | YOLO11n recall |
|---|---|---|---|---|
| Apple | 0.081 | 0.407 | 0.040 | 0.305 |
| Banana | 0.064 | 0.471 | 0.163 | 0.401 |
| Grape | 0.041 | 0.376 | 0.063 | 0.345 |
| Orange | 0.070 | 0.405 | 0.055 | 0.267 |
| Pineapple | 0.010 | 0.591 | 0.030 | 0.525 |
| Watermelon | 0.028 | 0.665 | 0.101 | 0.620 |

## Plain-background images (controlled scenes) — 59 images

| Metric | OpenCV baseline | YOLO11n |
|---|---|---|
| mAP@0.50 | 0.167 | 0.750 |
| mAP@0.50:0.95 | 0.119 | 0.635 |
| Precision | 0.160 | 0.776 |
| Recall | 0.155 | 0.668 |
| F1 | 0.147 | 0.701 |
| Localization recall (label ignored) | 0.442 | 0.567 |
| Mean absolute count error (fruits/image) | 1.339 | 0.847 |
| Exact count rate | 35.6% | 59.3% |
| End-to-end time (ms/image) | 24.3 | 62.3 |

| Class | Baseline AP50 | YOLO11n AP50 | Baseline recall | YOLO11n recall |
|---|---|---|---|---|
| Apple | 0.235 | 0.783 | 0.125 | 0.719 |
| Banana | 0.387 | 0.520 | 0.391 | 0.391 |
| Grape | 0.033 | 0.248 | 0.032 | 0.226 |
| Orange | 0.344 | 0.958 | 0.385 | 0.923 |
| Pineapple | 0.000 | 0.995 | 0.000 | 0.750 |
| Watermelon | 0.000 | 0.995 | 0.000 | 1.000 |

Side-by-side examples (ground truth | baseline | YOLO11n, count in brackets): `examples/`.
