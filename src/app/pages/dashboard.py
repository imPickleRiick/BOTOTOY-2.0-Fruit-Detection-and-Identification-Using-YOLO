"""Dashboard: the 31-fruit model behind the Fruit Reader - status, results, fruits and data.

Every number comes from the files written by the evaluation scripts (nothing is hard-coded):
- outputs/yolo_fruit31/eval_test/metrics.json     locked test set (python -m src.yolo.evaluate ... --final)
- outputs/reports/reader_test/reader_test.json    what the reader says (python -m src.evaluation.reader_test)
- data/splits_fruit31/summary.json                the merged 31-fruit dataset (python -m src.data.prepare_fruit31)
"""
import platform

import altair as alt
import pandas as pd
import streamlit as st

from src.app.common import load_result_json, page, pending
from src.app.pages.reader import get_reader
from src.assist.fruit_names import display_name
from src.utils.helpers import system_info

ACCENT = "#1D6B3E"
TEST_METRICS = "outputs/yolo_fruit31/eval_test/metrics.json"
READER_TEST = "outputs/reports/reader_test/reader_test.json"
SPLIT_SUMMARY = "data/splits_fruit31/summary.json"

STYLE = """
<style>
.block-container {max-width: 980px; padding-top: 2.5rem;}
h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0;}
.dash-sub {color: #5f6b64; font-size: 1.1rem; margin: 0.2rem 0 1.4rem;}
.chips {display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 8px;}
.chip {background: #F1F0EA; border: 1px solid #E2DFD3; border-radius: 999px; padding: 4px 12px; font-size: 0.95rem;}
</style>
"""


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


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def naming_chart(per_class: dict) -> None:
    """One bar per fruit: share of test photos in which the reader says the fruit's name."""
    df = pd.DataFrame([{"Fruit": display_name(c), "Named (%)": round(100 * r["named"], 1),
                        "Right when said (%)": round(100 * r["correct_when_said"], 1), "Test photos": r["images"]}
                       for c, r in per_class.items()]).sort_values("Named (%)", ascending=False)
    chart = (alt.Chart(df)
             .mark_bar(size=14, cornerRadiusEnd=4, color=ACCENT)
             .encode(x=alt.X("Named (%):Q", scale=alt.Scale(domain=[0, 100]),
                             title="Test photos in which the reader names the fruit (%)"),
                     y=alt.Y("Fruit:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
                     tooltip=["Fruit", "Named (%)", "Right when said (%)", "Test photos"])
             .properties(height=len(df) * 24))
    st.altair_chart(chart, use_container_width=True)
    with st.expander("Table view"):
        st.dataframe(df, hide_index=True, use_container_width=True)


def render() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)
    reader = get_reader()
    info = reader.engine.model_info()["yolo"]
    test = load_result_json(TEST_METRICS)
    rtest = load_result_json(READER_TEST)
    split = load_result_json(SPLIT_SUMMARY)

    st.title("Dashboard")
    st.markdown(f'<p class="dash-sub">The model behind the Fruit Reader: {reader.model_note}.</p>',
                unsafe_allow_html=True)

    # --- headline numbers
    m = st.columns(4)
    m[0].metric("Model", "Ready" if info["available"] else "Unavailable",
                help="YOLO11n fine-tuned on 31 fruits; runs on this computer's CPU.")
    if test:
        m[1].metric("Test mAP@0.50", f"{test['ultralytics']['mAP50']:.3f}",
                    help=f"Locked test set, {test['shared']['images']:,} images never used in training.")
        m[3].metric("Time per photo", f"{test['shared']['timing_ms']['end_to_end_mean']:.0f} ms",
                    help=f"Mean end-to-end detection time on {test['system']['cpu']}.")
    if rtest:
        m[2].metric("Fruit named correctly", _pct(rtest["naming"]["at_least_one_right"]),
                    help="Share of test photos in which the reader names at least one fruit that is really there.")
    if not info["available"]:
        st.error(f"The 31-fruit model is not loaded: {info['error']}", icon=":material/error:")

    # --- how it reads a fruit
    st.subheader("How the reader works")
    steps = st.columns(4)
    for col, (title, text) in zip(steps, [
            ("1. Photo check", "Dark, too bright or blurry? Say what to fix."),
            ("2. YOLO11n", "Find each fruit: box, name and confidence."),
            ("3. Sentence", "Name (English + Filipino), count, left / middle / right."),
            ("4. Speech", "Spoken on the phone when 3 frames agree.")]):
        with col.container(border=True):
            st.markdown(f"**{title}**  \n{text}")

    # --- results
    st.subheader("Results on the locked test set")
    if not (test and rtest):
        pending("The 31-fruit test evaluation or the reader test",
                "python -m src.evaluation.reader_test")
    else:
        hard = rtest["hard_conditions"]["summary"].values()
        passed = sum(s["fruit31_pass"] for s in hard)
        cases = sum(s["cases"] for s in hard)
        r = st.columns(4)
        r[0].metric("mAP@0.50:0.95", f"{test['ultralytics']['mAP50-95']:.3f}",
                    help="Box accuracy averaged over strict overlap thresholds (0.50 to 0.95).")
        r[1].metric("Exactly the right fruits", _pct(rtest["naming"]["exact_set"]),
                    help="Nothing missing and nothing extra in the spoken fruit names.")
        r[2].metric("Hard photos read right", f"{passed}/{cases}",
                    help="12 unseen photos under 11 conditions: dark, blurred, rotated, compressed, ...")
        r[3].metric("Needless photo warnings", _pct(rtest["quality_false_alarms"]["rate"]),
                    help="Normal test photos that still got photo-quality advice (mostly white backgrounds).")
        st.markdown("**Per fruit**")
        naming_chart(rtest["naming"]["per_class"])

    # --- fruits
    st.subheader(f"Fruits it knows ({len(reader.classes)})")
    names = sorted(display_name(c) for c in reader.classes)
    st.markdown('<div class="chips">' + "".join(f'<span class="chip">{n}</span>' for n in names) + "</div>",
                unsafe_allow_html=True)
    st.caption("Filipino names are said too, where a common one exists. Not a ripeness or freshness checker.")

    # --- data
    st.subheader("Training data")
    if split:
        final = split["final"]
        total = sum(s["images"] for s in final.values())
        boxes = sum(s["boxes"] for s in final.values())
        st.markdown(f"**{total:,} images** and **{boxes:,} labeled fruits** from two public datasets, split with no "
                    "photo shared between training and testing.")
        st.dataframe(pd.DataFrame([{"Split": k, "Images": f"{v['images']:,}", "Fruit boxes": f"{v['boxes']:,}"}
                                   for k, v in final.items()]), hide_index=True)
        st.caption(" · ".join(f"{k}: {v}" for k, v in split["sources"].items()))
    st.page_link(page("about"), label="About the project", icon=":material/arrow_forward:")

    # --- system
    with st.expander("System information"):
        s, v = st.columns(2)
        sysinfo = _system()
        s.markdown("\n".join([f"- **OS:** {sysinfo['os']}",
                              f"- **CPU:** {sysinfo['cpu']} ({sysinfo['cpu_threads']} threads)",
                              f"- **RAM:** {sysinfo.get('ram_gb', '?')} GB",
                              f"- **Weights:** `{info['weights']}` ({info.get('size_mb', '?')} MB)",
                              f"- **Parameters:** {info.get('parameters', 0):,}"]))
        v.markdown("\n".join(f"- **{k}:** {val}" for k, val in _software_versions().items()))
