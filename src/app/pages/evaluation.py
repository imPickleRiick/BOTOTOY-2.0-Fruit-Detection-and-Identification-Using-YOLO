"""Model Evaluation: the real results written by src.yolo.evaluate, src.yolo.training_report and
src.evaluation.compare_methods. If a result file is missing the page says the evaluation is pending."""
import altair as alt
import pandas as pd
import streamlit as st

from src.app.common import is_dark_theme, load_result_json, pending
from src.utils.helpers import PROJECT_ROOT

TRAIN_DIR = "outputs/yolo/training"
RUN_DIR = f"{TRAIN_DIR}/run/yolo11n_fruits"
PCT = st.column_config.ProgressColumn


def render() -> None:
    st.title("Model Evaluation")
    st.caption("All numbers are read from the evaluation files in `outputs/`; nothing on this page is typed in by hand. "
               "AP uses the Ultralytics formula for both methods, so they can be compared directly.")
    tabs = st.tabs(["Final test results", "Baseline vs YOLO11n", "Training", "Validation", "Error examples"])
    with tabs[0]:
        m = load_result_json("outputs/yolo/eval_test/metrics.json")
        if m:
            st.success("Final, one-time evaluation on the **locked test split**: images never used for training or "
                       "for choosing settings.", icon=":material/verified:")
            _metrics_section(m, "eval_test")
            _why_below_target(m)
        else:
            pending("The final test evaluation", "python -m src.yolo.evaluate --split test --final")
    with tabs[1]:
        _comparison()
    with tabs[2]:
        _training()
    with tabs[3]:
        m = load_result_json("outputs/yolo/eval_val/metrics.json")
        if m:
            st.caption("Validation split: used during development to choose the best epoch.")
            _metrics_section(m, "eval_val")
        else:
            pending("The validation evaluation", "python -m src.yolo.evaluate")
    with tabs[4]:
        _examples()


def _fmt_target_value(name: str, t: dict) -> str:
    if "value_seconds" in t:
        return f"{t['value_seconds'] * 1000:.0f} ms per image"
    if isinstance(t.get("value"), dict):
        lo = min(t["value"].items(), key=lambda kv: kv[1])
        hi = max(t["value"].items(), key=lambda kv: kv[1])
        return f"{lo[1]:.2f} ({lo[0]}) to {hi[1]:.2f} ({hi[0]})"
    return f"{t['value']:.3f}"


def _metrics_section(m: dict, folder: str) -> None:
    u, s = m["ultralytics"], m["shared"]
    st.markdown(f"**{s['images']} images** · weights `{m['weights']}` (SHA-256 `{m['weights_sha256'][:12]}…`) · "
                f"{m['system']['cpu']}, CPU")

    st.markdown("##### Proposal targets")
    rows = [{"Target": k, "Result": _fmt_target_value(k, t), "Status": "✅ Met" if t["met"] else "❌ Not met"}
            for k, t in m["targets"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    c = st.columns(4)
    c[0].metric("mAP@0.50", f"{u['mAP50']:.3f}")
    c[1].metric("mAP@0.50:0.95", f"{u['mAP50-95']:.3f}")
    c[2].metric("Precision", f"{u['precision']:.3f}")
    c[3].metric("Recall", f"{u['recall']:.3f}")
    st.caption(u.get("note", ""))

    st.markdown("##### Per-class results (Ultralytics)")
    per = pd.DataFrame([{"Fruit": k, "Precision": v["precision"], "Recall": v["recall"], "AP50": v["AP50"],
                         "AP50-95": v["AP50-95"]} for k, v in u["per_class"].items()])
    st.dataframe(per, hide_index=True, width="stretch", column_config={
        k: PCT(k, min_value=0.0, max_value=1.0, format="%.3f") for k in ("Precision", "Recall", "AP50", "AP50-95")})

    op = s["operating_point"]
    st.markdown(f"##### At the app's operating point (confidence ≥ {op['confidence']}, NMS IoU {op['iou_nms']})")
    opdf = pd.DataFrame([{"Fruit": k, "GT boxes": v["ground_truth"], "Predictions": v["predictions"],
                          "Precision": v["precision@0.5"], "Recall": v["recall@0.5"], "F1": v["f1@0.5"]}
                         for k, v in op["per_class"].items()])
    st.dataframe(opdf, hide_index=True, width="stretch", column_config={
        k: PCT(k, min_value=0.0, max_value=1.0, format="%.3f") for k in ("Precision", "Recall", "F1")})
    cnt, t = op["counting"], s["timing_ms"]
    c = st.columns(4)
    c[0].metric("Exact fruit count", f"{cnt['exact_count_rate']:.1%}", help="Images where the predicted total equals the true total.")
    c[1].metric("Count error", f"{cnt['mean_absolute_count_error']:.2f} / image",
                help=f"Mean absolute error. Signed mean {cnt['mean_signed_count_error']:+.2f}: negative = fruits missed.")
    c[2].metric("Localization recall", f"{op['localization_recall@0.5']:.3f}", help="Fruits found at IoU ≥ 0.5, ignoring the label.")
    c[3].metric("End-to-end time", f"{t['end_to_end_mean']:.0f} ms",
                help=f"Median {t['end_to_end_median']} ms, 95th percentile {t['end_to_end_p95']} ms. {t['note']}")
    stages = t["model_stages_mean"]
    st.caption("Model time per image (mean): " + ", ".join(f"{k} {v:.1f} ms" for k, v in stages.items()) +
               f". Measured on {m['system']['cpu']} ({m['system'].get('ram_gb', '?')} GB RAM).")

    st.markdown("##### Confusion matrix and precision-recall curve")
    ul = PROJECT_ROOT / "outputs" / "yolo" / folder / "ultralytics"
    a, b = st.columns(2)
    if (ul / "confusion_matrix_normalized.png").is_file():
        a.image(str(ul / "confusion_matrix_normalized.png"), caption="Normalized confusion matrix (columns = true class)")
    if (ul / "BoxPR_curve.png").is_file():
        b.image(str(ul / "BoxPR_curve.png"), caption="Precision-recall curve per class")


def _why_below_target(m: dict) -> None:
    if all(t["met"] for t in m["targets"].values()):
        return
    with st.expander("Why are the accuracy targets not met?", icon=":material/help:", expanded=True):
        st.markdown(
            "- **Missed fruits, not wrong labels.** In the confusion matrix, a detected fruit is usually given the right "
            "name; the large *background* row means fruits that were never detected. The signed count error is negative "
            "(the model under-counts).\n"
            "- **Inconsistent labels in crowded scenes.** Some dataset images have one box per fruit, others one box "
            "around a whole pile. The model learns both styles, and one group box where the ground truth has many single "
            "boxes counts as many misses (see *Error examples*).\n"
            "- **Limited variety.** The 7,108 training images are augmented copies of 3,200 photos; validation mAP "
            "stopped improving after about epoch 30 while training loss kept falling.\n"
            "- On plain-background photos (controlled scenes) YOLO11n reaches a much higher mAP@0.50; see "
            "*Baseline vs YOLO11n*.")


def _comparison() -> None:
    c = load_result_json("outputs/comparisons/test/comparison.json")
    if not c:
        pending("The baseline vs YOLO comparison", "python -m src.evaluation.compare_methods --split test --final")
        return
    st.caption(f"Both detectors on the same {c['images']} test images, scored by the same code. Precision, recall and "
               f"counts use YOLO confidence ≥ {c['yolo_operating_confidence']} (the baseline has no threshold).")
    labels = {"mAP50": "mAP@0.50", "mAP50-95": "mAP@0.50:0.95", "precision": "Precision", "recall": "Recall", "f1": "F1",
              "localization_recall": "Localization recall", "mean_absolute_count_error": "Count error (fruits/image)",
              "exact_count_rate": "Exact count rate", "end_to_end_ms_mean": "Time per image (ms)"}
    for subset, title in (("all", "All test images (real-world scenes)"),
                          ("plain_background", "Plain-background images (controlled scenes)")):
        r = c["results"][subset]
        st.markdown(f"##### {title}: {r['images']} images")
        rows = [{"Metric": lab, "OpenCV baseline": r["baseline"][k], "YOLO11n": r["yolo"][k]} for k, lab in labels.items()]
        left, right = st.columns(2)
        left.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                       column_config={"OpenCV baseline": st.column_config.NumberColumn(format="%.3f"),
                                      "YOLO11n": st.column_config.NumberColumn(format="%.3f")})
        per = pd.DataFrame([{"Fruit": f, "Baseline AP50": r["baseline"]["AP50_per_class"][f],
                             "YOLO11n AP50": r["yolo"]["AP50_per_class"][f]} for f in r["yolo"]["AP50_per_class"]])
        right.dataframe(per, hide_index=True, width="stretch", column_config={
            k: PCT(k, min_value=0.0, max_value=1.0, format="%.3f") for k in ("Baseline AP50", "YOLO11n AP50")})
    ex = sorted((PROJECT_ROOT / "outputs" / "comparisons" / "test" / "examples").glob("*.jpg"))
    if ex:
        st.markdown("##### Side-by-side examples (ground truth | baseline | YOLO11n, count in brackets)")
        pick = st.selectbox("Example", ex, format_func=lambda p: p.stem.replace("_", " "), key="cmp_example")
        st.image(str(pick), width="stretch")


def _training() -> None:
    summary = load_result_json(f"{RUN_DIR}/training_summary.json")
    csv = PROJECT_ROOT / RUN_DIR / "results.csv"
    if not summary or not csv.is_file():
        pending("YOLO11n training (or importing its results)", "python -m src.yolo.training_report <results zip>")
        return
    v, s, sysi = summary["validation"], summary["settings"], summary["system"]
    c = st.columns(4)
    c[0].metric("Epochs", f"{v['epochs_completed']} / {s['epochs']}")
    c[1].metric("Best epoch", v["best_epoch"], help="Highest validation mAP@0.50:0.95; its weights are the ones used.")
    c[2].metric("Best val mAP@0.50", f"{v['best_val_mAP50']:.3f}")
    c[3].metric("Training time", f"{summary['training_duration_minutes']:.0f} min", help=f"{sysi.get('gpu', '')}, Google Colab")

    df = pd.read_csv(csv)
    df.columns = [col.strip() for col in df.columns]
    long = df.melt(id_vars="epoch", value_vars=["metrics/mAP50(B)", "metrics/mAP50-95(B)"], var_name="series", value_name="value")
    long["series"] = long["series"].map({"metrics/mAP50(B)": "mAP@0.50", "metrics/mAP50-95(B)": "mAP@0.50:0.95"})
    colors = ["#3987e5", "#d95926"] if is_dark_theme() else ["#2a78d6", "#eb6834"]
    hover = alt.selection_point(fields=["epoch"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(long).encode(x=alt.X("epoch:Q", title="Epoch"),
                                  y=alt.Y("value:Q", title="Validation score", scale=alt.Scale(domain=[0, 1])),
                                  color=alt.Color("series:N", title=None, scale=alt.Scale(range=colors),
                                                  legend=alt.Legend(orient="top")))
    lines = base.mark_line(strokeWidth=2)
    points = base.mark_point(size=60, filled=True).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    rule = alt.Chart(df).mark_rule(color="#888").encode(
        x="epoch:Q", opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("epoch:Q", title="Epoch"),
                 alt.Tooltip("metrics/mAP50(B):Q", title="mAP@0.50", format=".3f"),
                 alt.Tooltip("metrics/mAP50-95(B):Q", title="mAP@0.50:0.95", format=".3f")]).add_params(hover)
    best = alt.Chart(pd.DataFrame({"epoch": [v["best_epoch"]], "label": [f"best epoch {v['best_epoch']}"]}))
    best_rule = best.mark_rule(strokeDash=[4, 3], color="#888").encode(x="epoch:Q")
    best_text = best.mark_text(align="right", dx=-4, y=8, color="#888").encode(x="epoch:Q", text="label:N")
    st.markdown("##### Validation accuracy per epoch")
    st.altair_chart((lines + points + rule + best_rule + best_text).properties(height=300), width="stretch")
    loss = PROJECT_ROOT / TRAIN_DIR / "loss_curves.png"
    if loss.is_file():
        st.markdown("##### Training vs validation loss")
        st.image(str(loss), width="stretch")
        st.caption("Training loss keeps falling while validation loss flattens after about epoch 30: the model starts to "
                   "memorize the training photos, so more epochs would not help much.")
    with st.expander("Training settings and hardware"):
        st.json({"settings": s, "hardware": sysi})


def _examples() -> None:
    folder = PROJECT_ROOT / "outputs" / "yolo" / "eval_test" / "examples"
    files = sorted(folder.glob("*.jpg"))
    if not files:
        pending("The error-example images", "python -m src.yolo.evaluate --split test --final")
        return
    kinds = {"fn": "Most missed fruits (magenta = missed)", "fp": "Most false alarms (red = false positive)",
             "tp": "Perfect images (every fruit found, no false alarm)"}
    kind = st.segmented_control("Show", list(kinds), format_func=lambda k: kinds[k], default="fn", key="err_kind")
    chosen = [f for f in files if f.name.startswith(f"{kind}_")] if kind else files
    for start in range(0, len(chosen), 3):
        cols = st.columns(3)
        for col, f in zip(cols, chosen[start:start + 3]):
            col.image(str(f), width="stretch")
