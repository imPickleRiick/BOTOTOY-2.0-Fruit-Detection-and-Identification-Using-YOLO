# Checkpoint 4 of 5: 80% progress check

**BOTOTOY 2.0: Fruit Detection and Identification Using YOLO** · CSST 106 · BSCS 4A

Team: Kyle Angelo D. Mercado · Ian Patrick Z. Mesias · James Paul L. Panal

> This document was compiled on 20 Nov 2026 from the repository. It maps each item of the checkpoint hand-in
> list to its evidence. The Git history shows when each part was committed.

## Hand-in checklist

| # | Required | Evidence | Status |
|---|---|---|---|
| 1 | Nearly complete application, main features integrated | Streamlit app `python app.py`: single and batch detection, both detectors, automatic fallback, downloads | ✅ (complete) |
| 2 | Interface with image upload and clear output (boxes, labels, confidence, counts) | 7 pages; screenshots in [`outputs/reports/system_test/screenshots/`](../../outputs/reports/system_test/screenshots/) | ✅ |
| 3 | Test-set metrics produced by `src/evaluate.py`, speed on the demo laptop, change since checkpoint 3 | `python -m src.evaluate --split test --final` (runs the YOLO evaluation, the baseline comparison and the error review); results below | ✅ |
| 4 | Performance analysis and limitations | Below; [final report](../FINAL_REPORT.md) sections 8–10 | ✅ |
| 5 | ≥ 10 test cases (ID, input, condition, expected, actual, pass/fail), ≥ 3 hard | 20 cases below, 11 of them hard; 132 more hard cases in [HARD_CASES.md](../../outputs/reports/hard_cases/HARD_CASES.md); 16 system checks in the [system test](../../outputs/reports/system_test/SYSTEM_TEST_REPORT.md) | ✅ |
| 6 | Draft report (introduction, data, method, experiments) and draft slides (8+) | Report: [`docs/FINAL_REPORT.md`](../FINAL_REPORT.md) (complete). Slides: outline in [`docs/DEMO_GUIDE.md`](../DEMO_GUIDE.md) | ⚠️ slides not yet made |
| 7 | Demo recording ≤ 5 minutes (backup) | Script: [`docs/DEMO_GUIDE.md`](../DEMO_GUIDE.md) section 3 | ❌ not yet recorded |
| — | Tag `cp4-80` | — | ❌ not created |

## Test-set results (682 held-out images, evaluated once)

| Metric | Validation (selection) | **Test (final)** | Target | Met? |
|---|---|---|---|---|
| mAP@0.50 | 0.478 | **0.490** | ≥ 0.80 | ❌ |
| mAP@0.50:0.95 | 0.321 | **0.338** | ≥ 0.50 | ❌ |
| Recall per class | 0.34 to 0.54 | **0.32 to 0.68** | ≥ 0.70 | ❌ |
| End-to-end time per image | 67 ms | **69 ms** | ≤ 1 s | ✅ |

**Change since checkpoint 3:**
- The validation and test scores are close (0.478 vs 0.490), so the model was not tuned to the test set.
- The test set was evaluated once, after the model and settings were frozen.
- The baseline improved from the first rules to the final rules on validation: mAP@0.50 0.026 → 0.053
  ([ablation](../../outputs/baseline/ablation_val.md)).
- YOLO11n is about 10 times as accurate as the baseline (test mAP@0.50 0.486 vs 0.049).

**Speed on the demo laptop** (AMD Ryzen 5 5600U, 5.9 GB RAM, CPU only, batch size 1):

| Configuration | Mean | 95th percentile |
|---|---|---|
| YOLO11n, 640 px | 66 ms (≈ 15 FPS) | 77 ms |
| YOLO11n, 320 px | 35 ms | 47 ms |
| OpenCV baseline | 27 ms | 38 ms |

## Test cases

A YOLO11n case **passes** if the expected fruit type is detected; for error cases, if the expected message
appears. Hard cases are marked ★.

| ID | Input | Condition | Expected | Actual | Result |
|---|---|---|---|---|---|
| TC01 | `mandarins_on_plate.jpg` (unseen) | normal photo | Orange | Orange ×3 | PASS |
| TC02 | `three_bananas.jpg` (unseen) | normal photo | Banana | Banana ×1 (one box around the bunch) | PASS |
| TC03 | `single_pineapple.jpg` (unseen) | normal photo | Pineapple | Pineapple ×1 | PASS |
| TC04 | `apples_on_tree.jpg` (demo) | fruit on a tree | Apple | Apple ×4 | PASS |
| TC05 | `green_unripe_orange.jpg` (unseen) | ★ unusual color | Orange | Apple ×1 | **FAIL** |
| TC06 | `mandarins_on_plate.jpg` | ★ poor lighting (×0.35) | Orange | Orange ×3 | PASS |
| TC07 | `mandarins_on_plate.jpg` | ★ overexposed (×1.8) | Orange | Orange ×3 | PASS |
| TC08 | `mandarins_on_plate.jpg` | ★ Gaussian blur | Orange | Orange ×3 | PASS |
| TC09 | `mandarins_on_plate.jpg` | ★ motion blur | Orange | nothing | **FAIL** |
| TC10 | `mandarins_on_plate.jpg` | ★ rotated 30° | Orange | Orange ×4 | PASS |
| TC11 | `mandarins_on_plate.jpg` | ★ occlusion (center block) | Orange | Orange ×4 | PASS |
| TC12 | `mandarins_on_plate.jpg` | ★ grayscale | Orange | Orange ×1 | PASS |
| TC13 | `single_pineapple.jpg` | ★ occlusion (center block) | Pineapple | nothing | **FAIL** |
| TC14 | `apples_bananas_grayscale.jpg` (demo) | ★ grayscale original | Apple, Banana | Apple ×2, Banana ×1 | PASS |
| TC15 | `apples_supermarket_display.jpg` (unseen) | ★ clutter, dozens of apples | Apple | Apple ×1 (one group box) | PASS (count wrong) |
| TC16 | Same photo as PNG, BMP and WEBP | file formats | same detections as the JPG | identical for PNG and BMP | PASS |
| TC17 | Phone photo with EXIF rotation | orientation metadata | shown upright, detected | upright, 3 fruits | PASS |
| TC18 | 12-megapixel photo (4000×3000) | very large input | detected in < 1 s | 2 fruits, 235 ms | PASS |
| TC19 | Empty file, text file renamed `.jpg`, truncated JPEG, 10×10 image | invalid input | rejected with a clear message | all rejected with a message | PASS |
| TC20 | YOLO weights file missing | model unavailable | falls back to the OpenCV baseline with a notice | baseline used, notice shown | PASS |

**Result: 17 of 20 pass.** The 3 failures are real limitations of the model, not bugs: motion blur, a green
orange, and a pineapple with its body hidden. TC01–TC15 come from
[HARD_CASES.md](../../outputs/reports/hard_cases/HARD_CASES.md) and the demo images. TC16–TC20 come from the
[system test](../../outputs/reports/system_test/SYSTEM_TEST_REPORT.md) (checks 3, 8 and 14).

## Performance analysis and limitations

**Where it works:**
- Single fruits and small groups.
- Plain backgrounds: test mAP@0.50 0.750.
- Dark, overexposed, slightly blurred and rotated photos.
- Grayscale photos (where the baseline fails completely).
- Fruit names are rarely wrong: only 2.3% of test errors are a wrong class.

**Where it fails, and why:**
- **Crowded piles:** the model often draws one box around the pile. The dataset labels such scenes
  inconsistently, which causes 49.8% of the misses. Counts in piles are too low.
- **Small or distant fruits** (< 1% of the image): 30.0% of the misses.
- **Motion blur and heavy compression:** 4 of 12 photos pass under each condition.
- **Unusual colors:** green oranges and yellow watermelons are called Apple, because the training set has few
  such examples.

**Not designed for:**
- Video or webcam streams.
- Fruits outside the six classes (they may get the closest class).
- Ripeness, quality and weight.
- Exact counting in large piles.
