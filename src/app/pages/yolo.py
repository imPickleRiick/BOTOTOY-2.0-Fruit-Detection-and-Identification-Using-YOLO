"""YOLO11n Model: model card, how it works, a confidence-threshold explorer and a baseline comparison."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.app.common import (PREVIEW_MAX_SIDE, counts_chart, current_settings, detect_cached, get_engine,
                            image_input, load_result_json)
from src.inference.image_processor import BASELINE, YOLO, InvalidImageError
from src.inference.visualization import detection_rows, fit_within, scale_detections, timing_rows, to_rgb
from src.utils.detections import count_by_class, draw_detections

EXPLORER_MIN_CONF = 0.05


def render() -> None:
    engine = get_engine()
    st.title("YOLO11n Model")
    if not engine.yolo_available:
        st.error(f"The trained YOLO11n model could not be loaded: {engine.yolo_error}", icon=":material/error:")
        st.markdown("The rest of the app keeps working with the OpenCV baseline. See the README (*YOLO11n* "
                    "section) to train the model or install the trained weights.")
        return
    _model_card(engine)
    st.divider()
    _explorer(engine)


def _model_card(engine) -> None:
    info = engine.model_info()["yolo"]
    train = load_result_json("outputs/yolo/training/run/yolo11n_fruits/training_summary.json")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            "**How it works.** YOLO (*You Only Look Once*) looks at the whole image in **one forward pass** of a "
            "convolutional neural network:\n"
            "1. **Preprocess:** the image is letterboxed to the input size (aspect ratio kept, borders padded) "
            "and scaled to 0–1.\n"
            "2. **Inference:** the network predicts, at every location of three feature-map scales, a box and a "
            "score for each of the 6 fruits.\n"
            "3. **Postprocess:** boxes below the **confidence threshold** are dropped; **non-maximum suppression** "
            "(NMS) removes boxes overlapping a better box of the same class by more than the **IoU threshold**.\n\n"
            "**Transfer learning.** We started from YOLO11n pretrained on COCO (80 everyday object classes, including "
            "apple, banana and orange) and fine-tuned all layers on our 6 fruit classes.")
    with c2, st.container(border=True):
        lines = [f"- **Architecture:** YOLO11n (nano), {info['parameters'] / 1e6:.2f} M parameters",
                 f"- **Weights:** `{info['weights']}` ({info['size_mb']} MB)"]
        if train:
            s, v, sysi = train["settings"], train["validation"], train["system"]
            lines += [f"- **Training:** {v['epochs_completed']} epochs, best epoch {v['best_epoch']} "
                      f"(chosen on validation mAP50-95)",
                      f"- **Hardware:** {sysi.get('gpu', 'GPU')} on Google Colab, {train['training_duration_minutes']:.0f} min",
                      f"- **Settings:** {s['imgsz']} px, batch {s['batch']}, {s['optimizer']} lr {s['lr0']}, seed {s['seed']}"]
        lines.append(f"- **Inference here:** CPU, default confidence {engine.yolo_defaults['conf']}, "
                     f"IoU {engine.yolo_defaults['iou']}")
        st.markdown("\n".join(lines))


def _threshold_curve(confidences: np.ndarray, chosen: float) -> alt.Chart:
    grid = np.round(np.arange(EXPLORER_MIN_CONF, 0.951, 0.01), 2)
    df = pd.DataFrame({"Confidence threshold": grid, "Fruits kept": [(confidences >= g).sum() for g in grid]})
    line = alt.Chart(df).mark_line(interpolate="step-after", strokeWidth=2, color="#2a78d6").encode(
        x=alt.X("Confidence threshold:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Fruits kept:Q", axis=alt.Axis(tickMinStep=1)),
        tooltip=["Confidence threshold:Q", "Fruits kept:Q"])
    rule = alt.Chart(pd.DataFrame({"Confidence threshold": [chosen]})).mark_rule(
        strokeDash=[4, 3], color="#888").encode(x="Confidence threshold:Q")
    return (line + rule).properties(height=220)


def _explorer(engine) -> None:
    settings = current_settings()
    st.subheader("Confidence threshold explorer")
    st.caption(f"The model runs once with confidence ≥ {EXPLORER_MIN_CONF}; moving the slider then shows which "
               f"boxes a higher threshold keeps (identical to re-running, because NMS keeps boxes in confidence "
               f"order). NMS IoU {settings['iou']:.2f} and image size {settings['imgsz']} px come from the sidebar.")
    picked = image_input("yolo")
    if picked is None:
        return
    name, data = picked
    try:
        with st.spinner("Running YOLO11n ..."):
            res = detect_cached(data, name, YOLO, EXPLORER_MIN_CONF, settings["iou"], settings["imgsz"], True, False)
    except InvalidImageError as exc:
        st.error(str(exc))
        return

    conf = st.slider("Confidence threshold", EXPLORER_MIN_CONF, 0.95, float(settings["conf"]), 0.01, key="yolo_explorer_conf")
    kept = [d for d in res.detections if d.confidence >= conf]
    counts = count_by_class(kept, engine.classes)
    left, right = st.columns([3, 2])
    left.image(to_rgb(fit_within(draw_detections(res.image, scale_detections(kept, res.preview_scale)), PREVIEW_MAX_SIDE)),
               caption=f"{len(kept)} fruit(s) with confidence ≥ {conf:.2f}", width="stretch")
    with right:
        st.markdown("**Fruits kept vs threshold**")
        st.altair_chart(_threshold_curve(np.array([d.confidence for d in res.detections]), conf), width="stretch")
        st.caption("Lower threshold: more fruits found (higher recall) but more false alarms (lower precision).")
        st.markdown("**Count per fruit**")
        st.altair_chart(counts_chart(counts, 28), width="stretch")

    with st.expander(f"Detections at confidence ≥ {conf:.2f} and YOLO timing"):
        rows = detection_rows(kept)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(timing_rows(res.timing_ms)), hide_index=True)

    st.subheader("Same image, OpenCV baseline")
    base = detect_cached(data, name, BASELINE, None, None, None, True, False)
    a, b = st.columns(2)
    a.image(to_rgb(fit_within(draw_detections(res.image, scale_detections(kept, res.preview_scale)), 900)),
            caption=f"YOLO11n: {len(kept)} fruit(s), {res.timing_ms['detector_total']:.0f} ms", width="stretch")
    b.image(to_rgb(fit_within(base.annotated, 900)),
            caption=f"OpenCV baseline: {base.total} fruit(s), {base.timing_ms['detector_total']:.0f} ms", width="stretch")
