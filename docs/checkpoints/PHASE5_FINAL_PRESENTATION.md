# Checkpoint 5 of 5: Final presentation and live demonstration

**BOTOTOY 2.0: Fruit Detection and Identification Using YOLO** · CSST 106 · BSCS 4A

Team: Kyle Angelo D. Mercado · Ian Patrick Z. Mesias · James Paul L. Panal

> This document was compiled on 20 Nov 2026 from the repository. It maps each item of the checkpoint hand-in
> list to its evidence.

## Hand-in checklist

| # | Required | Evidence | Status |
|---|---|---|---|
| 1 | Final working application, tagged `v1.0` | Tag [`v1.0`](https://github.com/imPickleRiick/BOTOTOY-2.0-Fruit-Detection-and-Identification-Using-YOLO/tree/v1.0) on GitHub. Run with `python app.py` (or `streamlit run app.py`). | ✅ |
| 2 | Complete source code + README: setup, one-command run, how to reproduce the results, where to get the weights, sample inputs | [README](../../README.md). The weights are in the repository (`models/trained/bototoy_yolo11n.pt`, 5.5 MB). Sample inputs are in `assets/samples/` (8 dataset test images) and `assets/unseen/` (12 outside photos). All evaluations reproduce with `python -m src.evaluate --split test --final`. | ✅ |
| 3 | Final report PDF: problem, objectives, methodology, algorithms, dataset, experiments, results, limitations, conclusion, references, contribution table | Content: [`docs/FINAL_REPORT.md`](../FINAL_REPORT.md) | ⚠️ PDF export and a separate Conclusion section still needed |
| 4 | Screenshots and sample outputs: a folder in the repository + an appendix in the report | Folders: [`outputs/reports/system_test/screenshots/`](../../outputs/reports/system_test/screenshots/), [`outputs/reports/system_test/unseen/`](../../outputs/reports/system_test/unseen/), [`outputs/reports/hard_cases/`](../../outputs/reports/hard_cases/), [`outputs/baseline/demo/`](../../outputs/baseline/demo/), [`outputs/comparisons/test/examples/`](../../outputs/comparisons/test/examples/) | ⚠️ the report has an evidence index, but no image appendix yet |
| 5 | Final presentation slides (PDF) | Outline and talking points: [`docs/DEMO_GUIDE.md`](../DEMO_GUIDE.md) section 2 | ❌ not yet made |
| 6 | Demo video ≤ 5 minutes (backup for the live demo) | Script: [`docs/DEMO_GUIDE.md`](../DEMO_GUIDE.md) section 3 | ❌ not yet recorded |

## Final results at a glance

| | Result | Target |
|---|---|---|
| Test mAP@0.50 / mAP@0.50:0.95 (682 held-out images) | 0.490 / 0.338 | 0.80 / 0.50 ❌ |
| Recall per class | 0.32 to 0.68 | ≥ 0.70 ❌ |
| Plain-background test mAP@0.50 | 0.750 | — |
| Baseline test mAP@0.50 | 0.049 | — |
| Time per image, laptop CPU | 69 ms | ≤ 1 s ✅ |
| Unseen photos (expected fruit type found) | 11 of 12 | — |
| Hard-case tests (132 cases) | YOLO11n 100, baseline 68 | — |
| System checks / unit tests | 16 of 16 / 82 of 82 | — |

## Supporting documents

| Document | Purpose |
|---|---|
| [Final report](../FINAL_REPORT.md) | The full technical report |
| [Error review](../../outputs/yolo/eval_test/error_review/ERROR_REVIEW.md) | All 1,977 test errors by cause, 48 inspected |
| [Hard-case test log](../../outputs/reports/hard_cases/HARD_CASES.md) | 132 cases under 11 conditions |
| [Demo guide](../DEMO_GUIDE.md) | Presentation outline, live demo script, backup plan |
| [Defense guide](../DEFENSE_GUIDE.md) | Each member's module and likely questions |
| [System test](../../outputs/reports/system_test/SYSTEM_TEST_REPORT.md) | 16 end-to-end checks, speed, memory, browser test |

## Contribution table

Module ownership, as assigned in the proposal:

| Member | Module | Main files |
|---|---|---|
| James Paul L. Panal | Dataset download, validation, duplicate/leakage checks, split, audit report | `src/data/` |
| Kyle Angelo D. Mercado | OpenCV baseline: segmentation, morphology, features, rules, evaluation | `src/baseline/` |
| Ian Patrick Z. Mesias | YOLO11n training and evaluation; inference engine, batch processing, Streamlit app, system test | `src/yolo/`, `src/inference/`, `src/app/`, `src/evaluation/` |

## Still to do before submitting

1. Export the final report to PDF, adding a Conclusion section and a screenshots appendix.
2. Make the presentation slides (8+), following the outline in the demo guide, and export them to PDF.
3. Record the demo video (≤ 5 minutes) following the demo guide, e.g. with Win + Alt + R.
4. Have each member push at least one commit from their own GitHub account.
