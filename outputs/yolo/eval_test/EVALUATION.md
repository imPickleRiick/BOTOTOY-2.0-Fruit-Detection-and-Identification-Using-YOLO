# YOLO11n evaluation - `test` split (682 images)

Weights: `bototoy_yolo11n.pt` (sha256 `6ce708595601…`)  
Hardware: AMD Ryzen 5 5600U with Radeon Graphics, 5.9 GB RAM, CPU inference

## Ultralytics validation metrics

| Class | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---|---|
| Apple | 0.618 | 0.332 | 0.436 | 0.309 |
| Banana | 0.610 | 0.444 | 0.485 | 0.283 |
| Grape | 0.597 | 0.366 | 0.387 | 0.273 |
| Orange | 0.592 | 0.324 | 0.389 | 0.251 |
| Pineapple | 0.627 | 0.564 | 0.547 | 0.364 |
| Watermelon | 0.641 | 0.682 | 0.697 | 0.547 |
| **All** | **0.614** | **0.452** | **0.490** | **0.338** |

_precision/recall are at the confidence that maximizes mean F1 (Ultralytics convention)._ Plots: `ultralytics/` (confusion matrix, PR / F1 / P / R curves).

## Proposal targets

| Target | Result | Met? |
|---|---|---|
| mAP50 >= 0.80 | 0.4903 | ❌ no |
| mAP50-95 >= 0.50 | 0.3377 | ❌ no |
| recall >= 0.70 for every class | Apple 0.33, Banana 0.44, Grape 0.37, Orange 0.32, Pineapple 0.56, Watermelon 0.68 | ❌ no |
| end-to-end <= 1 s per image | 0.069 | ✅ yes |

## At the app's operating point (confidence ≥ 0.25, NMS IoU 0.7)

Computed with the same code as the OpenCV baseline (`src/utils/metrics.py`).

| Class | GT boxes | Predictions | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---|---|---|---|---|
| Apple | 531 | 241 | 0.672 | 0.305 | 0.420 |
| Banana | 277 | 164 | 0.677 | 0.401 | 0.503 |
| Grape | 603 | 323 | 0.644 | 0.345 | 0.449 |
| Orange | 728 | 299 | 0.649 | 0.267 | 0.378 |
| Pineapple | 101 | 74 | 0.716 | 0.525 | 0.606 |
| Watermelon | 129 | 123 | 0.650 | 0.620 | 0.635 |
| **Mean** | | | **0.668** | **0.410** | **0.498** |

- AP with our implementation (all predictions): mAP50 **0.486**, mAP50-95 **0.336**
- Localization recall (label ignored): 0.350
- Counting: mean absolute error 2.213 fruits/image, mean signed -1.679, exact count 45.0%

## Processing time (this computer, CPU)

- Model stages (mean ms): preprocess 1.87, inference 60.18, postprocess/NMS 1.28
- End-to-end per image: mean **69.1 ms**, median 65.6 ms, 95th percentile 92.5 ms
- end-to-end = read image + preprocess + inference + NMS + draw boxes, batch size 1, CPU, after one warm-up image

## Error analysis

`examples/`: `tp_*` = every fruit found with no false alarms; `fp_*` = images with the most false positives (red boxes); `fn_*` = images with the most missed fruits (magenta boxes).
 Saved: {'tp': 6, 'fp': 6, 'fn': 6}.
