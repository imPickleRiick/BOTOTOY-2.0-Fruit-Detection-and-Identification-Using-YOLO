# Screenshots and sample outputs

## Application screenshots

Captured on 20 Nov 2026 from the running app (`python app.py`) on the team laptop (CPU only). The inputs were real
uploads of photos from `assets/unseen/` (Wikimedia Commons; the model never saw them in training).

| File | Page | What it shows |
|---|---|---|
| [`01_dashboard.png`](01_dashboard.png) | Dashboard | Model status (YOLO11n ready, test mAP@0.50 0.490, 69 ms per image), the 6 classes, system information |
| [`02_fruit_detection.png`](02_fruit_detection.png) | Fruit Detection | `mandarins_on_plate.jpg`: 3 Oranges with boxes and confidence scores, counts, timing, downloads |
| [`03_batch_detection.png`](03_batch_detection.png) | Batch Detection | 4 photos at once: per-image table, combined counts, annotated images, ZIP/CSV download |
| [`04_yolo_model.png`](04_yolo_model.png) | YOLO11n Model | How the model works, model card, confidence-threshold explorer |
| [`05_opencv_baseline.png`](05_opencv_baseline.png) | OpenCV Baseline | `single_pineapple.jpg` with the baseline, and every processing step |
| [`06_model_evaluation.png`](06_model_evaluation.png) | Model Evaluation | Test results vs targets, per-class metrics, confusion matrix, PR curve |
| [`07_about.png`](07_about.png) | About | Project, course and team |

## Sample outputs (elsewhere in the repository)

| Folder | Contents |
|---|---|
| [`outputs/reports/system_test/unseen/`](../../outputs/reports/system_test/unseen/) | YOLO11n on the 12 unseen photos |
| [`outputs/reports/hard_cases/`](../../outputs/reports/hard_cases/) | One photo under 11 hard conditions (dark, blur, rotation, occlusion, …) |
| [`outputs/comparisons/test/examples/`](../../outputs/comparisons/test/examples/) | Ground truth vs baseline vs YOLO11n, side by side |
| [`outputs/baseline/demo/`](../../outputs/baseline/demo/) | Baseline outputs and step-by-step panels |
| [`outputs/yolo/eval_test/examples/`](../../outputs/yolo/eval_test/examples/) | Correct detections, false positives and misses on the test set |
| [`outputs/yolo/eval_test/error_review/`](../../outputs/yolo/eval_test/error_review/) | Error galleries (48 errors, labeled by cause) |
| [`outputs/reports/system_test/screenshots/`](../../outputs/reports/system_test/screenshots/) | Screenshots from the automated browser test (17 Nov) |

## v2: assistive Fruit Reader

Captured on 24 Nov 2026 with the 31-fruit model and the minimal design, in Chrome on the team laptop. The
live-reader screenshot uses Chrome's simulated phone screen and a fake camera that plays a video of
`mandarins_on_plate.jpg`.

| File | What it shows |
|---|---|
| [`v2_01_reader_photo.png`](v2_01_reader_photo.png) | Photo mode with an uploaded photo: the spoken sentence ("I see 3 oranges (dalandan): 1 on the left, 1 in the middle and 1 on the right."), the Read again button and the labeled boxes for a sighted helper |
| [`v2_02_quality_warning.png`](v2_02_quality_warning.png) | A dark photo: the reader first says what is wrong with the photo, then still names the banana |
| [`v2_03_phone_panel.png`](v2_03_phone_panel.png) | The Settings button: speech options, minimum confidence and the QR code to open the reader on a phone |
| [`v2_04_live_reader_phone.png`](v2_04_live_reader_phone.png) | Live mode on a phone screen: thin boxes over the camera view, the sentence it spoke, Stop and Repeat |
| [`v2_05_dashboard.png`](v2_05_dashboard.png) | Dashboard: model status, main results, how the reader works and the per-fruit naming chart |
