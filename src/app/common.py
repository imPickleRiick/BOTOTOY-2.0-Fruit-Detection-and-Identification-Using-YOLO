"""Shared pieces of the Streamlit app: model loading, sidebar settings, image input, result display.

Every number shown in the app comes from the real detectors (src/inference) or from the
evaluation files written in Phases 3-4 (outputs/). Nothing is hard-coded.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.inference.image_processor import (BASELINE, METHOD_LABELS, SUPPORTED_FORMATS, SUPPORTED_FORMATS_TEXT,
                                           YOLO, FruitDetectionEngine, InferenceResult, InvalidImageError,
                                           save_result)
from src.inference.visualization import (count_rows, detection_rows, encode_image, fit_within, safe_stem,
                                         timing_rows, to_rgb)
from src.utils.helpers import PROJECT_ROOT, load_json
from src.utils.image_utils import CLASS_CHART_COLORS_HEX

SAMPLES_DIR = PROJECT_ROOT / "assets" / "samples"
DEFAULT_DEMO = "mixed_fruit_bowl"
APP_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "app"
PREVIEW_MAX_SIDE = 1600
IMAGE_SIZES = [320, 416, 512, 640, 800, 960]


# ----------------------------------------------------------------------------- navigation
PAGES: dict = {}   # url_path -> st.Page, filled by app.py so pages can link to each other


def page(url_path: str):
    return PAGES[url_path]


# ----------------------------------------------------------------------------- models
@st.cache_resource(show_spinner="Loading YOLO11n and the OpenCV baseline ...")
def get_engine() -> FruitDetectionEngine:
    """Loaded once per app process and shared by all pages and browser tabs."""
    return FruitDetectionEngine()


@st.cache_data(max_entries=24, show_spinner=False)
def detect_cached(data: bytes, name: str, method: str, conf: float | None, iou: float | None, imgsz: int | None,
                  show_confidence: bool, keep_stages: bool) -> InferenceResult:
    """Run detection once per (image, settings); re-showing a result does not re-run the model.

    The timings shown are the ones measured when the detection actually ran.
    """
    return get_engine().run_bytes(data, name, method=method, conf=conf, iou=iou, imgsz=imgsz,
                                  show_confidence=show_confidence, keep_stages=keep_stages)


def run_detection(data: bytes, name: str, method: str, settings: dict, keep_stages: bool = False) -> InferenceResult | None:
    """Detection with a spinner and friendly errors; returns None (and shows the error) on failure."""
    yolo = method == YOLO
    try:
        with st.spinner(f"Running {METHOD_LABELS[method]} ..."):
            return detect_cached(data, name, method, settings["conf"] if yolo else None,
                                 settings["iou"] if yolo else None, settings["imgsz"] if yolo else None,
                                 settings["show_conf"], keep_stages)
    except InvalidImageError as exc:
        st.error(f"{exc}")
    return None


# ----------------------------------------------------------------------------- sidebar
def render_sidebar(engine: FruitDetectionEngine) -> None:
    """Global detection settings, shown on every page (kept in st.session_state)."""
    sb = st.sidebar
    sb.markdown("### Detection settings")
    methods = [YOLO, BASELINE] if engine.yolo_available else [BASELINE]
    if not engine.yolo_available:
        sb.warning("YOLO11n is unavailable, so the OpenCV baseline is used (fallback).", icon=":material/warning:")
    sb.radio("Detection model", methods, format_func=lambda m: METHOD_LABELS[m] + (" (main)" if m == YOLO else " (fallback)"),
             key="method", help="Used on the Fruit Detection and Batch Detection pages.")
    yolo_on = st.session_state.get("method", methods[0]) == YOLO
    d = engine.yolo_defaults
    sb.slider("Confidence threshold", 0.05, 0.95, float(d["conf"]), 0.05, key="conf", disabled=not yolo_on,
              help="YOLO only. Boxes with a lower confidence are discarded. Lower = more fruits found but more false alarms.")
    sb.slider("NMS IoU threshold", 0.30, 0.90, float(d["iou"]), 0.05, key="iou", disabled=not yolo_on,
              help="YOLO only. Two boxes of the same fruit class overlapping more than this are merged into one.")
    sb.select_slider("Image size (px)", IMAGE_SIZES, value=int(d["imgsz"]), key="imgsz", disabled=not yolo_on,
                     help="YOLO only. The model was trained at 640 px; larger finds smaller fruits but runs slower.")
    sb.toggle("Show confidence on boxes", value=True, key="show_conf")
    if not yolo_on:
        sb.caption("The OpenCV baseline uses the fixed rules in config.yaml; the YOLO sliders do not apply.")
    sb.caption(f"Evaluation settings: confidence {d['conf']}, IoU {d['iou']}, {d['imgsz']} px, CPU.")


def current_settings() -> dict:
    ss = st.session_state
    return {"method": ss.get("method", YOLO), "conf": ss.get("conf", 0.25), "iou": ss.get("iou", 0.7),
            "imgsz": ss.get("imgsz", 640), "show_conf": ss.get("show_conf", True)}


# ----------------------------------------------------------------------------- inputs
def demo_images() -> list[Path]:
    return sorted(SAMPLES_DIR.glob("*.jpg"))


def pretty_name(path: Path) -> str:
    return path.stem.replace("_", " ").capitalize()


def image_input(key: str) -> tuple[str, bytes] | None:
    """Upload a photo, or pick one of the labeled demonstration images."""
    source = st.segmented_control("Image source", ["Upload a photo", "Demo image"], default="Upload a photo",
                                  key=f"{key}_source", label_visibility="collapsed")
    if source == "Demo image":
        samples = demo_images()
        if not samples:
            st.info("No demonstration images found in assets/samples/.")
            return None
        default = next((k for k, p in enumerate(samples) if p.stem == DEFAULT_DEMO), 0)
        choice = st.selectbox("Demonstration image", samples, index=default, format_func=pretty_name, key=f"{key}_demo")
        st.caption("Demonstration data: a photo from the dataset's test split (CC0 license), "
                   "not seen by the model during training.")
        return choice.name, choice.read_bytes()
    file = st.file_uploader(f"Upload a photo ({SUPPORTED_FORMATS_TEXT})", type=list(SUPPORTED_FORMATS),
                            key=f"{key}_upload")
    if file is None:
        return None
    return file.name, file.getvalue()


# ----------------------------------------------------------------------------- charts & tables
def counts_chart(counts: dict[str, int], height_per_bar: int = 34) -> alt.Chart:
    """Horizontal bar per fruit class (every class shown, zero included), labeled with its count."""
    order = list(counts)
    df = pd.DataFrame({"Fruit": order, "Count": [counts[c] for c in order]})
    base = alt.Chart(df).encode(
        y=alt.Y("Fruit:N", sort=order, title=None, axis=alt.Axis(labelFontSize=13, ticks=False, domain=False)),
        x=alt.X("Count:Q", title=None, axis=None, scale=alt.Scale(domainMin=0, nice=True)))
    bars = base.mark_bar(cornerRadiusEnd=4, size=18).encode(
        color=alt.Color("Fruit:N", scale=alt.Scale(domain=order, range=[CLASS_CHART_COLORS_HEX[c] for c in order]),
                        legend=None),
        tooltip=[alt.Tooltip("Fruit:N"), alt.Tooltip("Count:Q")])
    labels = base.mark_text(align="left", dx=6, fontSize=13, color=text_color()).encode(text="Count:Q")
    return (bars + labels).properties(height=height_per_bar * len(order))


def is_dark_theme() -> bool:
    theme = getattr(st.context, "theme", None)
    return getattr(theme, "type", "light") == "dark"


def text_color() -> str:
    """Chart text uses the page's text color, never a series color (readable in both themes)."""
    return "#f0efec" if is_dark_theme() else "#0b0b0b"


def class_chip_html(name: str) -> str:
    color = CLASS_CHART_COLORS_HEX.get(name, "#888888")
    return (f"<span style='display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0;font-weight:600'>"
            f"<span style='width:12px;height:12px;border-radius:3px;background:{color};display:inline-block'></span>"
            f"{name}</span>")


# ----------------------------------------------------------------------------- result display
def show_result(result: InferenceResult, name: str, key: str, allow_save: bool = True) -> None:
    """Original vs annotated image, counts, detections table, timings, downloads and save."""
    if result.notice:
        st.warning(result.notice, icon=":material/warning:")
    left, right = st.columns(2)
    left.image(to_rgb(fit_within(result.image, PREVIEW_MAX_SIDE)), caption="Original image", width="stretch")
    right.image(to_rgb(fit_within(result.annotated, PREVIEW_MAX_SIDE)),
                caption=f"{result.method_label}: {result.total} fruit(s) detected", width="stretch")

    found = sum(1 for v in result.counts.values() if v)
    m = st.columns(4)
    m[0].metric("Total fruits", result.total)
    m[1].metric("Fruit types found", f"{found} of {len(result.counts)}")
    m[2].metric("Detector time", f"{result.timing_ms['detector_total']:.0f} ms")
    m[3].metric("End-to-end time", f"{result.timing_ms['end_to_end']:.0f} ms",
                help="Decode the file + detect + draw the boxes, measured on this computer. The first image "
                     "of a new browser session is usually slower (one-time setup in a new thread).")

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Count per fruit**")
        st.altair_chart(counts_chart(result.counts), width="stretch")
    with c2:
        st.markdown("**Detected fruits**")
        rows = detection_rows(result.detections)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(38 + 35 * len(rows), 320),
                         column_config={"Confidence": st.column_config.ProgressColumn(
                             "Confidence", min_value=0.0, max_value=1.0, format="%.2f")})
        else:
            st.info("No fruit was detected in this image." +
                    (" Try lowering the confidence threshold in the sidebar." if result.method == YOLO else ""))
        st.caption("Box corners in pixels of the original image: (x1, y1) top-left, (x2, y2) bottom-right.")

    with st.expander("Processing time per step and settings used"):
        t1, t2 = st.columns(2)
        t1.dataframe(pd.DataFrame(timing_rows(result.timing_ms)), hide_index=True, width="stretch")
        t2.json(result.params)

    stem = safe_stem(name)
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download annotated image", encode_image(result.annotated), f"{stem}_{result.method}.jpg",
                       "image/jpeg", icon=":material/download:", key=f"{key}_dl_img", width="stretch")
    import json
    d2.download_button("Download results (JSON)", json.dumps(result.summary(name), indent=2),
                       f"{stem}_{result.method}.json", "application/json", icon=":material/data_object:",
                       key=f"{key}_dl_json", width="stretch")
    if allow_save and d3.button("Save to outputs/app/", icon=":material/save:", key=f"{key}_save", width="stretch"):
        paths = save_result(result, APP_OUTPUT_DIR / "single", stem)
        st.success(f"Saved {paths['image'].relative_to(PROJECT_ROOT).as_posix()} and "
                   f"{paths['json'].name}", icon=":material/check_circle:")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# ----------------------------------------------------------------------------- saved evaluation results
def load_result_json(rel_path: str) -> dict | None:
    """Evaluation file written by Phases 3-4, or None if that evaluation has not been run yet."""
    path = PROJECT_ROOT / rel_path
    return load_json(path) if path.is_file() else None


def pending(what: str, command: str) -> None:
    st.info(f"**Pending:** {what} has not been run yet on this machine. Run `{command}` to produce it.",
            icon=":material/hourglass_empty:")
