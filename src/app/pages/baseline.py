"""OpenCV Baseline: run the traditional pipeline and show every processing step and rule decision."""
import pandas as pd
import streamlit as st

from src.app.common import current_settings, image_input, load_result_json, run_detection, show_result
from src.inference.image_processor import BASELINE
from src.inference.visualization import BASELINE_STAGES, COLOR_GROUP_LEGEND, region_rows, to_rgb


def render() -> None:
    st.title("Traditional CV Baseline (OpenCV)")
    st.markdown(
        "A hand-designed pipeline with no machine learning: **resize + blur → HSV color masks → morphology → "
        "contours → shape/color features → rules → NMS → counts**. Every threshold is in `config/config.yaml` "
        "(section `baseline`). It is the fallback when the YOLO model is unavailable.")
    note = "This page always uses the baseline, whatever model is selected in the sidebar. "
    cmp = load_result_json("outputs/comparisons/test/comparison.json")
    if cmp:
        r = cmp["results"]
        note += (f"It works best on single fruits against plain backgrounds: test mAP@0.50 = "
                 f"{r['plain_background']['baseline']['mAP50']:.3f} on plain backgrounds vs "
                 f"{r['all']['baseline']['mAP50']:.3f} on all test images (see Model Evaluation).")
    st.caption(note)

    picked = image_input("baseline")
    if picked is None:
        st.info("Upload a photo or choose a demo image to see each processing step.", icon=":material/upload:")
        return
    name, data = picked
    settings = current_settings() | {"method": BASELINE}
    result = run_detection(data, name, BASELINE, settings, keep_stages=True)
    if result is None:
        return

    show_result(result, name, key="baseline")

    st.subheader("Processing steps")
    legend = " ".join(
        f"<span style='display:inline-flex;align-items:center;gap:5px;margin-right:14px'>"
        f"<span style='width:12px;height:12px;border-radius:3px;background:{hex_};display:inline-block'></span>"
        f"{group}</span>" for group, hex_ in COLOR_GROUP_LEGEND.items())
    tabs = st.tabs([BASELINE_STAGES[k][0] for k in result.stages])
    for tab, (key, image) in zip(tabs, result.stages.items()):
        title, explanation = BASELINE_STAGES[key]
        with tab:
            st.caption(explanation)
            if key in ("2_color_masks", "3_after_morphology", "4_contours"):
                st.markdown(f"Color groups: {legend}", unsafe_allow_html=True)
            st.image(to_rgb(image), width=640)

    st.subheader("Candidate regions and rule decisions")
    st.caption("Every region large enough to be a fruit, its measured features, and the rule that accepted or "
               "rejected it. Accepted regions may still be removed by NMS if they overlap a higher-scoring one.")
    rows = region_rows(result.regions)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=min(38 + 35 * len(rows), 400))
    else:
        st.info("No region passed the color and size filters.")
