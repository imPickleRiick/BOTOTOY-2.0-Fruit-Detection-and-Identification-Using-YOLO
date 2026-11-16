"""Dashboard: what the project does, the six classes, model status and system information."""
import platform

import streamlit as st

from src.app.common import class_chip_html, get_engine, load_result_json, page
from src.inference.image_processor import METHOD_LABELS, YOLO
from src.utils.helpers import system_info

FRUIT_NOTES = {
    "Apple": "round, red/green", "Banana": "elongated, yellow", "Grape": "small berries in bunches",
    "Orange": "round, orange", "Pineapple": "tall, spiky crown", "Watermelon": "large, green rind / red flesh",
}


@st.cache_data(show_spinner=False)
def _software_versions() -> dict:
    import cv2
    import torch
    import ultralytics
    return {"Python": platform.python_version(), "PyTorch": torch.__version__, "Ultralytics": ultralytics.__version__,
            "OpenCV": cv2.__version__, "Streamlit": st.__version__,
            "GPU (CUDA)": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not available, CPU inference"}


@st.cache_data(show_spinner=False)
def _system() -> dict:
    return system_info()


def render() -> None:
    engine = get_engine()
    info = engine.model_info()
    st.title("BOTOTOY 2.0")
    st.markdown("#### Fruit Detection and Identification Using YOLO")
    st.markdown(
        "Upload a photo and BOTOTOY finds every fruit in it: a **bounding box**, the **fruit type**, a "
        "**confidence score**, and the **number of fruits per type** plus the overall total. "
        "The main detector is a **YOLO11n** deep-learning model fine-tuned on 7,108 labeled photos; a traditional "
        "**OpenCV** pipeline (color segmentation + shape rules) serves as the baseline and fallback.")

    st.markdown("**Supported fruits (6)**")
    st.markdown("".join(class_chip_html(c) for c in engine.classes), unsafe_allow_html=True)
    st.caption(" · ".join(f"{c}: {FRUIT_NOTES.get(c, '')}" for c in engine.classes))
    st.caption("Scope: detection, identification and counting only. No ripeness, freshness, disease, weight or "
               "nutrition analysis.")

    st.divider()
    st.subheader("Model status")
    test = load_result_json("outputs/yolo/eval_test/metrics.json")
    active = st.session_state.get("method", YOLO)
    m = st.columns(4)
    m[0].metric("Active model", METHOD_LABELS[active], help="Change it in the sidebar.")
    m[1].metric("YOLO11n", "Ready" if info["yolo"]["available"] else "Unavailable")
    if test:
        m[2].metric("Test mAP@0.50", f"{test['ultralytics']['mAP50']:.3f}",
                    help="Final one-time evaluation on 682 unseen test images. Target: 0.80.")
        m[3].metric("Time per image (CPU)", f"{test['shared']['timing_ms']['end_to_end_mean']:.0f} ms",
                    help=f"Mean end-to-end time on {test['system']['cpu']}. Target: 1000 ms or less.")
    else:
        m[2].metric("Test mAP@0.50", "Pending")
        m[3].metric("Time per image", "Pending")

    y, b = st.columns(2)
    with y.container(border=True):
        st.markdown("**YOLO11n (main method)**")
        yi = info["yolo"]
        if yi["available"]:
            st.markdown(f"- Weights: `{yi['weights']}` ({yi['size_mb']} MB)\n"
                        f"- SHA-256: `{yi['sha256'][:16]}…`\n"
                        f"- {yi['parameters']:,} parameters, transfer learning from COCO\n"
                        f"- Classes: {', '.join(yi['classes'])}")
            st.page_link(page("yolo"), label="Model details and threshold explorer", icon=":material/arrow_forward:")
        else:
            st.error(f"Not loaded: {yi['error']}", icon=":material/error:")
            st.markdown("The app falls back to the OpenCV baseline. To fix it, train the model (README, "
                        "*YOLO11n* section) or copy the trained `best.pt` to "
                        f"`{yi['weights']}`.")
    with b.container(border=True):
        st.markdown("**OpenCV baseline (fallback)**")
        st.markdown(f"- Always available, no training needed\n"
                    f"- HSV color masks → morphology → contours → shape/color features → rules\n"
                    f"- Works on images resized to {info['baseline']['resize_max_side']} px\n"
                    f"- Best on single fruits against plain backgrounds")
        st.page_link(page("baseline"), label="See every processing step", icon=":material/arrow_forward:")

    st.divider()
    st.subheader("System information")
    s, v = st.columns(2)
    sysinfo = _system()
    s.markdown("\n".join([f"- **OS:** {sysinfo['os']}", f"- **CPU:** {sysinfo['cpu']} ({sysinfo['cpu_threads']} threads)",
                          f"- **RAM:** {sysinfo.get('ram_gb', '?')} GB",
                          f"- **Inference device:** {engine.yolo_defaults['device'].upper()}"]))
    v.markdown("\n".join(f"- **{k}:** {val}" for k, val in _software_versions().items()))

