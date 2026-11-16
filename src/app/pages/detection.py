"""Fruit Detection: one image -> boxes, labels, confidences, counts, downloads."""
import streamlit as st

from src.app.common import current_settings, image_input, run_detection, show_result
from src.inference.image_processor import METHOD_LABELS, YOLO


def render() -> None:
    settings = current_settings()
    st.title("Fruit Detection")
    method = settings["method"]
    note = (f"confidence ≥ {settings['conf']:.2f}, NMS IoU {settings['iou']:.2f}, {settings['imgsz']} px"
            if method == YOLO else "fixed rules from config.yaml")
    st.caption(f"Model: **{METHOD_LABELS[method]}** ({note}). Change the model and thresholds in the sidebar.")

    picked = image_input("detect")
    if picked is None:
        st.info("Upload a photo containing apples, bananas, grapes, oranges, pineapples or watermelons, "
                "or choose a demo image.", icon=":material/upload:")
        return
    name, data = picked
    result = run_detection(data, name, method, settings)
    if result is not None:
        show_result(result, name, key="detect")
