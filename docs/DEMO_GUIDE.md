# Final presentation and live demo guide

A script for the final presentation (about 12–15 minutes plus questions), with a step-by-step live demo and a
backup plan if anything fails. Every result quoted below was produced by the current model on the team laptop.

## 1. Before the presentation (the day before, then again 15 minutes before)

- [ ] `git pull`, then activate the virtual environment (`.venv\Scripts\Activate.ps1`).
- [ ] `python -m pytest -q`: expect **82 passed**.
- [ ] `streamlit run app.py`. Check that the Dashboard shows **YOLO11n: Ready** and test mAP@0.50 **0.490**.
- [ ] Run one detection, so the first (slower) run happens before the audience is watching.
- [ ] Turn off Wi-Fi once and repeat a detection. The app needs no internet, so it should still work.
- [ ] Close other heavy programs (the laptop has 6 GB of RAM). Plug in the charger. Set the power mode to "Best
      performance".
- [ ] Open these folders in File Explorer, ready to drag files from:
  - `assets/unseen/` (photos never seen in training)
  - `outputs/reports/system_test/screenshots/` (backup)
- [ ] Prepare a fake image for the error-handling demo, e.g. rename any `.txt` file to `not_an_image.jpg`.
- [ ] Set the browser zoom so the projector shows the sidebar and the image side by side (about 90%).

## 2. Presentation outline

Each member presents their own module. This matches the proposal and prepares everyone for the individual
defense.

| # | Part | Speaker | Time | Content and evidence |
|---|---|---|---|---|
| 1 | Problem and objectives | any | 1 min | Manual fruit counting is slow. Scope: 6 classes, still images. The four measurable targets (report §1–2). |
| 2 | Dataset and audit | Panal | 2 min | 8,479 images, CC0. The audit found 5 leaking near-duplicate pairs; 3 images removed. Split 70/15/15 by unique photo, leakage-free. Class imbalance 9.6×. Show `class_instances_by_split.png` and `samples_grid.jpg`. |
| 3 | OpenCV baseline | Mercado | 2 min | HSV masks → morphology → contours → features → rules. Show a `*_stages.jpg` panel from `outputs/baseline/demo/`. Validation ablation table. Why it fails in clutter. |
| 4 | YOLO11n training | Mesias | 2 min | Transfer learning from COCO, Colab T4, 50 epochs, best epoch 47. Show `validation_metrics.png`: validation mAP flat after epoch 30. |
| 5 | **Live demo** | Mesias (others narrate) | 4 min | Section 3 below |
| 6 | Results | Mesias | 2 min | Test table vs targets. Be direct: accuracy targets not met; speed met. Baseline vs YOLO: 0.049 → 0.486 mAP@0.50. Plain backgrounds: 0.750. |
| 7 | Error analysis and limitations | any / all | 1.5 min | Only 2.3% of errors are a wrong class. Half the misses come from group boxes in inconsistent labels. Show `sample_fn.jpg`. |
| 8 | Conclusion and future work | any | 0.5 min | Working, fast, tested system. Next step: consistent relabeling, then a larger image size or model. |

**How to present the missed targets.** State them plainly, then explain them with evidence:
- *"Our model reached mAP@0.50 of 0.49 against a target of 0.80."*
- *"Our error review shows that only 2.3% of errors are wrong fruit names. Most come from crowded scenes that the
  dataset labels in two different ways."*
- *"On plain backgrounds, where the labels are consistent, we reach 0.75."*

Do not hide the numbers or round them up.

## 3. Live demo script (about 4 minutes)

Settings for the whole demo: YOLO11n, confidence 0.25 (the defaults in the sidebar).

| Step | Page / action | Image | What the audience should see | What to say |
|---|---|---|---|---|
| 1 | **Dashboard** | — | Model ready, the 6 classes, test results, laptop CPU | "Everything runs locally on this laptop's CPU, with no GPU and no internet." |
| 2 | **Fruit Detection** → upload with the file picker | `assets/unseen/mandarins_on_plate.jpg` | **3 Orange**, boxes with confidence scores, count chart, time in ms | "This photo is from Wikimedia Commons; the model has never seen it. Each box has a label and a confidence score, and the counts are on the right." |
| 3 | Same page → demo image | `apples_on_tree.jpg` | **4 Apple** on a tree | Point out the per-class count and the total. |
| 4 | Same page → demo image | `apples_bananas_grayscale.jpg` | **2 Apple + 1 Banana** in a black-and-white photo | "There is no color here at all. YOLO uses shape and texture." |
| 5 | Sidebar → switch the model to **OpenCV baseline**, same image | `apples_bananas_grayscale.jpg` | **0 fruits** | "The baseline works only with color, so it finds nothing." |
| 6 | Baseline, demo image | `watermelon_slices.jpg` | **11 detections with mixed labels** (Watermelon, Banana, Apple, Pineapple); YOLO finds 2 Watermelon | "Red and yellow regions trigger the color rules. This is why YOLO is the main method." Optionally open the **OpenCV Baseline** page to show the masks and the rule for each region. |
| 7 | Switch back to **YOLO11n**; download the annotated image and JSON | — | Two files saved | "Results can be saved and reused." |
| 8 | **Batch Detection** → upload all 12 files in `assets/unseen/` **plus** `not_an_image.jpg` | 13 files | 12 processed, **1 error reported**, combined counts, ZIP download | "One bad file doesn't stop the batch. It is reported, and the rest are processed." |
| 9 | **Model Evaluation** | — | Targets table, per-class metrics, confusion matrix | Link to the results part of the talk. |
| 10 | (Honest failure, optional) Fruit Detection | `assets/unseen/green_unripe_orange.jpg` | **Apple** (wrong) | "A known limitation: almost all training oranges are orange-colored, so a green orange looks like an apple to the model." |

Also known, if someone asks about counting: `apples_supermarket_display.jpg` gives **1 Apple** box around the
whole display. This is the group-box behavior described in the error review.

### Optional: show the automatic fallback

The system test proves it (check 3). To show it live, rename `models/trained/bototoy_yolo11n.pt` to
`bototoy_yolo11n.pt.bak` and restart the app. The app shows a notice and uses the OpenCV baseline. **Rename it
back afterwards.** Only do this if there is time; restarting takes about 10 seconds.

## 4. Backup plan

| Problem | What to do |
|---|---|
| The app does not start or the laptop freezes | Show the screenshots in `outputs/reports/system_test/screenshots/` (`detection_upload.png`, `batch.png`, `evaluation.png`), then the annotated results below. |
| Detection is slow | It should take about 0.1–0.3 s per photo. If it is slow, close other programs, or set the image size to 320 in the sidebar (about 35 ms per image). |
| No projector / browser problem | Use the command line: `python -m src.yolo.detector assets/unseen/mandarins_on_plate.jpg --out outputs/app/cli_demo`, then open the annotated JPG. |
| Questions about a specific result | All reports are Markdown files in `outputs/`; open them in VS Code (Ctrl+Shift+V for the preview). |

**Backup annotated images** (already in the repository, all generated by the final model):

| What | Folder |
|---|---|
| YOLO11n on the 12 unseen photos | `outputs/reports/system_test/unseen/` |
| Ground truth vs baseline vs YOLO11n, side by side | `outputs/comparisons/test/examples/` |
| Baseline step-by-step panels (masks, morphology, contours) | `outputs/baseline/demo/*_stages.jpg` |
| Correct detections, false positives and misses on the test set | `outputs/yolo/eval_test/examples/` (`tp_*`, `fp_*`, `fn_*`) |
| Error review contact sheets | `outputs/yolo/eval_test/error_review/sample_fp.jpg`, `sample_fn.jpg` |

## 5. Numbers to know by heart

| | |
|---|---|
| Dataset | 8,479 images, 4,571 unique photos, 6 classes, CC0 |
| Split (images) | 7,108 / 686 / 682, which is 70/15/15 by unique photo |
| Test mAP@0.50 / mAP@0.50:0.95 | **0.490 / 0.338** (targets 0.80 / 0.50) |
| Recall per class | 0.32 (Orange) to 0.68 (Watermelon) (target 0.70) |
| Baseline test mAP@0.50 | 0.049 |
| Plain-background mAP@0.50 (YOLO) | 0.750 |
| Speed | 69 ms per image on a laptop CPU (target ≤ 1 s) |
| Unseen photos | 11 of 12 correct fruit type (baseline 7 of 12) |
| Wrong-class errors | 2.3% of all errors |
| Misses caused by group boxes | 49.8% |
| Tests | 82 unit tests, 16/16 system checks, 11/11 browser steps |
