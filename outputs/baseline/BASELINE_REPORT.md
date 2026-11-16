# OpenCV Baseline Report - BOTOTOY 2.0

The baseline is the project's traditional computer-vision method. It is the reference point for
YOLO11n and the fallback if YOLO is unavailable. Every number in this report was measured on the
**validation split** (686 images). The test split stays locked until the final YOLO-vs-baseline
comparison in Phase 4.

Hardware: AMD Ryzen 5 5600U (12 threads), 5.9 GB RAM, Windows 11, Python 3.12.7, OpenCV 5.0.0, CPU only.

## 1. Pipeline

| Step | Operation | Module |
|---|---|---|
| 1 | Load image (unicode-safe), resize so the longer side is ≤ 640 px | `preprocessing.py` |
| 2 | Gaussian blur 5×5 (removes noise and fine texture) | `preprocessing.py` |
| 3 | BGR → HSV (separates color from brightness) | `preprocessing.py` |
| 4 | Four color-group masks with `cv2.inRange`: **red**, **warm** (orange/yellow), **green**, **purple** | `segmentation.py` |
| 5 | Morphological **opening** (5×5 ellipse, removes specks), then **closing** (9×9 ellipse ×2, fills holes) | `segmentation.py` |
| 6 | External contours; keep regions between 1% and 90% of the image area | `segmentation.py` |
| 7 | Features: mean hue (circular mean), saturation, value, area, aspect ratio, elongation, extent, circularity, solidity | `feature_extraction.py` |
| 8 | Rule-based classification (decision rules per color group) | `fruit_classifier.py` |
| 9 | Bounding boxes mapped back to the original resolution; NMS (IoU 0.5) merges duplicates from different masks | `pipeline.py` |
| 10 | Count per class + total; save the annotated image | `pipeline.py` |

**How the thresholds were chosen.** The HSV ranges come from measured colors of 400 ground-truth boxes
per class in the **training** split (`calibration/hue_histograms.png`, `calibration/color_profiles.json`).
Key findings:
- Orange (median hue 16), Banana (24) and Pineapple (22) share almost the same hue band, so the
  rules must separate them by shape and saturation.
- Dark grapes appear **blue** in OpenCV hue (≈ 105-120). Green grapes look like yellow-green.
- Watermelon is a mix of red flesh (hue 0-8 / 170-179) and green rind.

## 2. Improvement log (ablation on validation)

Each step keeps the previous changes (`python -m src.baseline.ablation`):

| Step | Why | Predictions | Precision | Recall | mAP50 | mAP50-95 | Count MAE |
|---|---|---|---|---|---|---|---|
| A. Initial rules | — | 6,658 | 0.035 | 0.090 | 0.026 | 0.015 | 7.61 |
| B. Minimum region size 1% | regions under 1% were ~98% false positives | 2,899 | 0.075 | 0.074 | 0.047 | 0.029 | 3.66 |
| C. Reject irregular green regions | green "grape" regions were mostly leaves (0.4% correct) | 2,337 | 0.084 | 0.072 | 0.052 | 0.032 | 3.32 |
| **D. Size-aware rule score (final)** | larger regions are more often real fruits | 2,337 | **0.084** | **0.072** | **0.053** | **0.032** | **3.32** |
| E. Watershed split of touching fruits | split blobs of touching round fruits | 2,815 | 0.074 | 0.075 | 0.049 | 0.029 | 3.80 |

Step E (distance transform + watershed) was **rejected**. It lowered mAP50 and nearly doubled the time
per image (56 → 102 ms). The code is kept and can be switched on in `config.yaml`
(`split_touching.enabled`).

## 3. Final results (validation, 686 images)

| Class | GT boxes | Predictions | Precision@0.5 | Recall@0.5 | AP50 | AP50-95 |
|---|---|---|---|---|---|---|
| Apple | 448 | 165 | 0.200 | 0.074 | 0.115 | 0.076 |
| Banana | 284 | 580 | 0.060 | 0.123 | 0.041 | 0.021 |
| Grape | 572 | 480 | 0.117 | 0.098 | 0.071 | 0.042 |
| Orange | 883 | 445 | 0.101 | 0.051 | 0.064 | 0.040 |
| Pineapple | 140 | 144 | 0.000 | 0.000 | 0.000 | 0.000 |
| Watermelon | 164 | 523 | 0.027 | 0.085 | 0.030 | 0.013 |
| **Mean** | | | **0.084** | **0.072** | **0.053** | **0.032** |

- Localization recall (a fruit is covered by any box at IoU ≥ 0.5, label ignored): **0.157**
- Counting: mean absolute count error **3.32** fruits per image; exact count in **14.7%** of images
- Speed: **57.7 ms** detector, **66.3 ms** end-to-end (read + detect + draw) per image, CPU only

AP uses the same 101-point interpolation as Ultralytics, so these numbers are directly comparable
with YOLO11n's validation metrics. (With this formula even a perfect detector scores 0.995.)

## 4. Controlled scenes vs. real-world scenes

"Plain background" subset: 76 validation images where ≥ 70% of the pixels outside the fruit boxes
are gray or white (saturation < 50) (`--plain-background`).

| Subset | Images | Precision | Recall | mAP50 | Localization recall | Exact count | ms/image |
|---|---|---|---|---|---|---|---|
| All validation images | 686 | 0.084 | 0.072 | 0.053 | 0.157 | 14.7% | 57.7 |
| Plain background only | 76 | 0.141 | 0.119 | **0.108** | **0.392** | **38.2%** | 41.4 |

On plain backgrounds the baseline finds **2.5× more fruits** and gets the exact count **2.6× more
often**. This supports using it in **controlled environments**, such as a single fruit on a table
against a plain background. The subset also contains black-and-white photos, where no color-based
method can work.

## 5. Lighting robustness (simulated)

All validation pixel values were multiplied by a brightness factor:

| Brightness | mAP50 | Precision | Recall | Localization recall |
|---|---|---|---|---|
| ×0.6 (darker) | 0.057 | 0.088 | 0.078 | 0.155 |
| ×1.0 (original) | 0.053 | 0.084 | 0.072 | 0.157 |
| ×1.4 (brighter) | 0.061 | 0.094 | 0.081 | 0.152 |

Global brightness changes have little effect. HSV keeps hue separate from brightness, and the
saturation/value limits are broad. The small differences are within noise. This test does **not**
cover colored light, strong shadows or overexposure, which change hue and saturation.

## 6. Demo outputs (`demo/`)

Annotated images, `*_stages.jpg` panels (preprocessed → color masks → morphology → contours →
detections) and `results.json`, which includes each rule's reason.

| Image | Scene | Result |
|---|---|---|
| 028ee4d4… | one red apple, plain table | ✅ 1 Apple ("red + compact") |
| 12d334ca… | apple next to a computer mouse, white | ✅ 1 Apple |
| 11a50eb1… | one orange, plain background | ✅ Orange, plus one false "Watermelon" on its shadowed lower part |
| 100353af… | bunch of 3 bananas, white | ⚠️ 1 Banana: the touching bananas merge into one blob (ground truth has 2 boxes) |
| ae2b9b0d… | orange tree, 24 oranges | ❌ 3 "Banana": small oranges fall under the 1% size filter; touching oranges merge into tall chains that the elongation rule calls Banana |
| 72bc33e9… | 11 grapes | ❌ nothing detected |
| e47fc8e7… | 17 watermelon pieces | ❌ 5 detections with mixed labels |
| 5444b98b… | 21 apples | ❌ 2 detections |

## 7. Limitations (why YOLO is the main method)

1. **Cluttered backgrounds.** Wood, skin, cardboard, clothing and foliage share fruit colors.
   Blue clothing falls in the "dark grape" hue range, and leaves resemble green grapes.
2. **Similar fruit colors.** Orange, banana and pineapple overlap in hue. Shape rules separate them
   only when the fruit is isolated. **Pineapple was never correctly detected** (AP 0.000): its
   brown-yellow skin and green crown split across masks, and its color matches oranges and bananas.
3. **Overlapping and touching fruits** merge into one region, which produces one box (under-counting)
   or a wrong shape-based class. Watershed splitting did not help on this dataset (ablation step E).
4. **Small fruits.** Regions under 1% of the image are discarded to suppress noise, so distant fruits
   are missed. This is a deliberate precision/recall trade-off chosen on validation.
5. **Shadows and highlights** change saturation and value. Shadowed parts of a fruit can fall into a
   different mask (e.g. the false "Watermelon" on the orange's shadowed lower part).
6. **Stretched images.** The dataset resized every image to 640×640 without keeping the aspect ratio,
   which distorts aspect ratio and circularity.
7. **Grayscale and artwork images** contain no usable color information.
8. **The rule score is not a probability.** It only ranks detections.

## 8. Reproduce

```bash
python -m src.baseline.color_calibration
python -m src.baseline.ablation
python -m src.baseline.evaluate_baseline
python -m src.baseline.evaluate_baseline --brightness 0.6
python -m src.baseline.evaluate_baseline --brightness 1.4
python -m src.baseline.evaluate_baseline --plain-background --examples 12
python -m src.baseline.pipeline <images...> --stages --out outputs/baseline/demo
```
