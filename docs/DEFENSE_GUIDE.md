# Individual technical defense: study guide

The proposal says each member must "prepare to explain their module during the individual technical defense".
This guide gives each member their files, a short explanation to practice saying aloud, a way to show the module
working, and likely questions with answers. All answers are based on the actual code and reports. If an answer
here ever disagrees with the code, the code is right.

**Everyone should read Section 4 (shared questions).** Panels often ask any member about the whole system.

---

## 1. James Paul L. Panal: dataset pipeline

**Files:**
- `src/data/download_dataset.py`
- `src/data/dataset_validator.py`
- `src/data/dataset_splitter.py`
- `src/data/dataset_analyzer.py`
- `src/utils/image_utils.py` (`dhash`, `yolo_to_xyxy`)

**Evidence:**
- [Dataset audit report](../outputs/reports/dataset/DATASET_AUDIT.md)
- `data/splits/*.txt` (the committed split lists)
- `outputs/reports/dataset/samples_grid.jpg`

**Explain it in one minute.**
1. The downloader fetches a *pinned commit* of the GitHub mirror, so everyone gets identical files.
2. The validator checks all 8,479 images and 36,922 labels: every image has a label, class IDs are 0–5, and
   coordinates are in [0, 1].
3. It then looks for duplicates: exact copies by MD5 file hash, and near-duplicates by a 64-bit perceptual hash
   (dHash).
4. Roboflow names augmented copies `<source>_jpg.rf.<hash>.jpg`, so copies of one photo can be grouped by their
   source ID.
5. Five near-duplicate pairs crossed train and valid. We inspected them and removed the 3 validation images.
6. The splitter keeps the original training split whole, so all augmented copies stay in training. It pools the
   original valid and test images, groups near-duplicates, and assigns the groups 50/50 to val and test,
   stratified by dominant class with seed 42.
7. The result is 70/15/15 by unique photo, with zero overlap by file, hash, source photo or near-duplicate.

**Show it:**
- `python -m src.data.dataset_splitter --from-manifest` rebuilds the exact split from the committed lists.
- Then open `DATASET_AUDIT.md`, section 5 (leakage verification).

**Likely questions:**

| Question | Answer |
|---|---|
| What is data leakage, and why does it matter? | When the same (or almost the same) image is in training and testing, the test measures memory, not generalization. Scores look better than real-world performance. |
| How does dHash work? | Shrink the image to 9×8 grayscale and compare each pixel with its right neighbor: brighter = 1, else 0. That gives 64 bits. Similar images give hashes that differ in only a few bits (the Hamming distance). We used a distance ≤ 4. |
| Why not just use MD5? | MD5 only finds byte-identical files. A re-encoded or re-colored copy has a different MD5 but almost the same dHash. |
| Why is the test set only 682 images, not ~1,272 as planned? | The training split contains 1–3 augmented copies of each photo. Moving any of them would leak, so the original training split had to stay whole. Counted by unique photo, the split is still 70/15/15. |
| What does "stratified" mean here? | Within each dominant-class group, groups are shuffled and then each is given to whichever split is furthest below its target share. That way val and test get a similar class mix (`assign_val_test`). |
| What does a YOLO label line mean? | `class x_center y_center width height`, all relative to the image size (0–1). `yolo_to_xyxy` converts it to pixel corners. |
| What limitations did you find? | Orange has 9.6 times as many boxes as Pineapple. Images are stretched to 640×640. Some boxes cover a whole group of fruits. Some lemons are labeled Orange. The training set is augmented copies of only 3,200 photos. |
| Why commit the split lists? | So every member, and Colab, gets exactly the same split. Results are reproducible. |

---

## 2. Kyle Angelo D. Mercado: OpenCV baseline

**Files:**
- `src/baseline/preprocessing.py`
- `src/baseline/segmentation.py`
- `src/baseline/feature_extraction.py`
- `src/baseline/fruit_classifier.py`
- `src/baseline/pipeline.py`
- `src/baseline/color_calibration.py`
- `src/baseline/evaluate_baseline.py`
- `src/baseline/ablation.py`
- Thresholds: `config/config.yaml` → `baseline:`

**Evidence:**
- [Baseline report](../outputs/baseline/BASELINE_REPORT.md)
- `outputs/baseline/demo/*_stages.jpg`
- `outputs/baseline/calibration/hue_histograms.png`

**Explain it in one minute.**
1. Resize to 640 px, apply a 5×5 Gaussian blur, and convert BGR to HSV. HSV separates color (hue) from brightness
   (value).
2. Build four color masks with `cv2.inRange`: red, warm (orange/yellow), green and purple. Red needs two ranges,
   because hue wraps around at 0/179.
3. Clean each mask with opening, which removes specks, then closing, which fills holes.
4. Find external contours and keep regions between 1% and 90% of the image.
5. Measure features: circular mean hue, saturation, elongation (from the rotated rectangle), extent, circularity
   (4πA/P²) and solidity.
6. Decide the class with rules per color group. Then apply NMS, count, and draw.
7. The HSV ranges come from measured colors of 400 training boxes per class. Each change was tested on
   validation (the ablation table).

**Show it:**
- `python -m src.baseline.pipeline assets/unseen/mandarins_on_plate.jpg --stages --out outputs/app/baseline_demo`
- Then open the `_stages.jpg` panel. The app's **OpenCV Baseline** page shows the same steps, with the reason
  each rule fired.

**Likely questions:**

| Question | Answer |
|---|---|
| Why HSV and not RGB? | In HSV, the fruit's color (hue) is one channel, mostly independent of lighting (value). In RGB, a shadow changes all three channels. The ×0.6 and ×1.4 brightness test barely changed the results. |
| Opening vs closing? | Opening = erode then dilate, which removes small white noise. Closing = dilate then erode, which fills small black holes and gaps inside a fruit. |
| What is circularity / solidity / extent? | Circularity = 4π·area / perimeter², which is 1 for a circle. Solidity = area / convex-hull area; bumpy shapes are lower. Extent = area / bounding-box area; a circle gives about 0.785, a triangle (watermelon slice) about 0.5. |
| How are fruits classified? | By color group, then shape. Red: low extent → Watermelon slice, otherwise Apple. Warm: elongation ≥ 2.2 → Banana; orange hue + vivid + round → Orange; tall + dull → Pineapple; yellow → Banana. Green: large and round → Watermelon; round and convex → Apple; irregular → rejected (leaves). Purple → Grape. |
| How did you choose the thresholds? | From measured training-set colors (`color_calibration.py`). Then each change was kept only if it improved validation results. Example: the 1% minimum size doubled precision, from 0.035 to 0.075. |
| Why did you reject watershed? | It splits touching fruits, but on validation it lowered mAP@0.50 (0.053 → 0.049) and nearly doubled the time (56 → 102 ms). It is still in the code, switched off in the config. |
| Why does it never detect pineapple? | Its brown-yellow skin has the same hue as orange and banana, and its green crown falls into a different mask, so the fruit splits into parts. |
| What is the "confidence" of the baseline? | A rule score: rule reliability × compactness × size. It is only used to rank detections. It is not a probability. |
| Then why keep the baseline? | It is the fallback if YOLO cannot load, it needs no training or GPU, and every decision can be explained. It also gives YOLO a fair comparison: same images, same metric code. On plain backgrounds, it finds the exact count in 36% of test images. |

---

## 3. Ian Patrick Z. Mesias: YOLO11n, inference and application

**Files:**
- `src/yolo/train.py`
- `src/yolo/detector.py`
- `src/yolo/evaluate.py`
- `src/yolo/training_report.py`
- `notebooks/train_yolo_colab.ipynb`
- `src/inference/` (engine, batch, visualization)
- `src/app/` (Streamlit)
- `src/evaluation/` (comparison, system test, error review)
- `src/utils/metrics.py`

**Evidence:**
- [Training report](../outputs/yolo/training/TRAINING_REPORT.md)
- [Test evaluation](../outputs/yolo/eval_test/EVALUATION.md)
- [Comparison](../outputs/comparisons/test/COMPARISON.md)
- [System test](../outputs/reports/system_test/SYSTEM_TEST_REPORT.md)
- [Error review](../outputs/yolo/eval_test/error_review/ERROR_REVIEW.md)

**Explain it in one minute.**
1. YOLO11n is a one-stage detector. One CNN forward pass predicts boxes, classes and scores at three scales, and
   NMS removes duplicates.
2. We used transfer learning from COCO weights: Ultralytics replaces the head with a 6-class head and fine-tunes
   all layers.
3. Training ran on a Colab T4: 640 px, batch 16, SGD with lr 0.01, 50 epochs with patience 10, seed 42, 94
   minutes.
4. `best.pt` is the epoch with the best validation fitness (epoch 47). The test set was run once at the end.
5. The engine decodes the upload (EXIF rotation, format and size checks) and runs YOLO on the CPU. If the weights
   are missing, it falls back to the baseline.
6. Results: 0.490 mAP@0.50 on test, 69 ms per image, 16/16 system checks passed.

**Show it:**
- `streamlit run app.py`, following the [demo guide](DEMO_GUIDE.md).
- `python -m pytest -q` → 82 passed.

**Likely questions:**

| Question | Answer |
|---|---|
| What is IoU? mAP@0.50 vs mAP@0.50:0.95? | IoU = overlap area / union area of two boxes. A detection counts as correct at IoU ≥ 0.5 with the right class. AP = area under the precision-recall curve for one class; mAP = the mean over classes. mAP@0.50:0.95 averages over IoU 0.50, 0.55, …, 0.95, so it also rewards precise boxes. |
| What does NMS do? | It sorts boxes by confidence, keeps the best one, and deletes other boxes of the same class that overlap it with IoU above 0.7. It repeats until no boxes are left. |
| Why transfer learning? | 3,200 real photos are too few to train from scratch. COCO features (edges, textures, even apple/banana/orange) give a head start: validation mAP@0.50 was already 0.32 after one epoch. |
| Why YOLO11n and not a bigger model? | The proposal chose the nano model for a student workload and for CPU inference. 69 ms per image leaves room, so a larger model is a reasonable next step. |
| Why train on Colab? | The team's laptops have no NVIDIA GPU. 50 epochs on the CPU would take over 100 hours, compared with 94 minutes on a T4. |
| How do you know it isn't overfitting to the test set? | The test set was never used for training, checkpoint selection or thresholds. It was evaluated once. Validation and test scores are close (0.478 vs 0.490). |
| Why didn't you reach 0.80? | The error review shows only 2.3% of errors are wrong classes. 49.8% of misses come from group boxes in inconsistent labels, and 30% are tiny fruits. On consistently labeled plain-background images, the model reaches 0.750. Training longer would not help: validation mAP was flat after epoch 30. |
| Why is recall lower than precision? | The model misses more than it invents. The count error is −1.7 fruits per image on average. Crowded piles and small fruits are the misses (see above). |
| What does the confidence threshold change? | Lower (e.g. 0.05) gives more boxes and higher recall, but more false alarms. Higher gives fewer, more certain boxes. 0.25 is the default; the YOLO11n page in the app shows the trade-off live. |
| How is speed measured? | End-to-end: decode + preprocess + inference + NMS + drawing, batch size 1, after one warm-up image, on the Ryzen 5 5600U CPU. Mean 66–69 ms; 95th percentile 77–92 ms. |
| What happens with a bad file? | `decode_image` rejects empty, corrupt, non-image, < 32 px and > 50 MP files with a clear message. In a batch, the file is reported and the others continue. |
| Why are YOLO and baseline numbers comparable? | Both return the same `Detection` type and are scored by the same `src/utils/metrics.py`, using the same 101-point AP formula as Ultralytics. |

---

## 4. Shared questions (every member)

| Question | Answer |
|---|---|
| What does the system do, in one sentence? | It finds and names six fruit types in a photo, drawing a box, label and confidence for each, and counts them per type and in total. |
| What is out of scope? | Video, ripeness, freshness, disease, weight and grading. |
| Traditional CV vs deep learning, in your project? | The baseline uses hand-designed color and shape rules: fast and explainable, but it breaks in clutter (test mAP@0.50 0.049). YOLO learns features from data and handles clutter and grayscale far better (0.486), at the cost of training and less explainability. |
| Precision vs recall? | Precision = how many detections are correct (0.668). Recall = how many real fruits were found (0.410). |
| How is a fruit counted? | One box = one fruit. If the model boxes a whole pile, the pile counts as one. This is a stated limitation. |
| What would you do with two more weeks? | Relabel crowded validation and test images in one consistent style, then train at 960 px or with YOLO11s. |
| Where did you use course topics? | Image enhancement (blur, HSV), morphology (opening, closing), segmentation (thresholding, contours, watershed), feature extraction (shape descriptors), classification (rule-based decisions), CNN-based detection (YOLO), and evaluation (precision, recall, AP, confusion matrix). |
| How do I run it from a fresh clone? | `pip install -r requirements.txt`, then `streamlit run app.py`. The trained weights are in the repository. |
