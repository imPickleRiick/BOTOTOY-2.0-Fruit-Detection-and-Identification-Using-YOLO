# YOLO11n evaluation - `val` split (686 images)

Weights: `bototoy_yolo11n.pt` (sha256 `6ce708595601…`)  
Hardware: AMD Ryzen 5 5600U with Radeon Graphics, 5.9 GB RAM, CPU inference

## Ultralytics validation metrics

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|
| Apple | 0.658 | 0.475 | 0.549 | 0.396 |
| Banana | 0.583 | 0.447 | 0.482 | 0.292 |
| Grape | 0.634 | 0.427 | 0.454 | 0.291 |
| Orange | 0.633 | 0.343 | 0.391 | 0.240 |
| Pineapple | 0.584 | 0.379 | 0.385 | 0.247 |
| Watermelon | 0.633 | 0.543 | 0.607 | 0.458 |
| **All** | **0.621** | **0.436** | **0.478** | **0.321** |

_precision/recall are at the confidence that maximizes mean F1 (Ultralytics convention)._ Plots: `ultralytics/` (confusion matrix, PR / F1 / P / R curves).

## Proposal targets

| Target | Result | Met? |
|---|---|---|
| mAP50 >= 0.80 | 0.478 | ❌ no |
| mAP50-95 >= 0.50 | 0.3205 | ❌ no |
| recall >= 0.70 for every class | Apple 0.48, Banana 0.45, Grape 0.43, Orange 0.34, Pineapple 0.38, Watermelon 0.54 | ❌ no |
| end-to-end <= 1 s per image | 0.067 | ✅ yes |

## At the app's operating point (confidence ≥ 0.25, NMS IoU 0.7)

Computed with the same code as the OpenCV baseline (`src/utils/metrics.py`).

| Class | GT boxes | Predictions | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---|---|---|---|---|
| Apple | 448 | 261 | 0.713 | 0.415 | 0.525 |
| Banana | 284 | 182 | 0.665 | 0.426 | 0.519 |
| Grape | 572 | 331 | 0.689 | 0.399 | 0.505 |
| Orange | 883 | 354 | 0.701 | 0.281 | 0.401 |
| Pineapple | 140 | 62 | 0.629 | 0.279 | 0.386 |
| Watermelon | 164 | 132 | 0.682 | 0.549 | 0.608 |
| **Mean** | | | **0.680** | **0.391** | **0.491** |

- AP with our implementation (all predictions): mAP50 **0.479**, mAP50-95 **0.327**
- Localization recall (label ignored): 0.372
- Counting: mean absolute error 2.168 fruits/image, mean signed -1.704, exact count 41.1%

## Processing time (this computer, CPU)

- Model stages (mean ms): preprocess 1.94, inference 58.2, postprocess/NMS 1.26
- End-to-end per image: mean **66.7 ms**, median 64.0 ms, 95th percentile 83.1 ms
- end-to-end = read image + preprocess + inference + NMS + draw boxes, batch size 1, CPU, after one warm-up image

## Error analysis

`examples/`: `tp_*` = every fruit found with no false alarms; `fp_*` = images with the most false positives (red boxes); `fn_*` = images with the most missed fruits (magenta boxes).
 Saved: {'tp': 6, 'fp': 6, 'fn': 6}.
