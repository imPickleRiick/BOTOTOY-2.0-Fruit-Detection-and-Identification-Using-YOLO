"""BOTOTOY 2.0 - Fruit Detection and Identification Using YOLO (Streamlit user interface).

Run from the project root, with the virtual environment activated:
    streamlit run app.py
"""
import streamlit as st

from src.app.common import PAGES, get_engine, render_sidebar
from src.app.pages import about, baseline, batch, dashboard, detection, evaluation, yolo

st.set_page_config(page_title="BOTOTOY 2.0 | Fruit Detection", page_icon=":material/nutrition:", layout="wide")

engine = get_engine()

pages = {
    "Overview": [
        st.Page(dashboard.render, title="Dashboard", icon=":material/dashboard:", url_path="dashboard", default=True),
    ],
    "Detect fruits": [
        st.Page(detection.render, title="Fruit Detection", icon=":material/image_search:", url_path="detect"),
        st.Page(batch.render, title="Batch Detection", icon=":material/photo_library:", url_path="batch"),
    ],
    "Methods": [
        st.Page(yolo.render, title="YOLO11n Model", icon=":material/neurology:", url_path="yolo"),
        st.Page(baseline.render, title="OpenCV Baseline", icon=":material/tune:", url_path="baseline"),
        st.Page(evaluation.render, title="Model Evaluation", icon=":material/monitoring:", url_path="evaluation"),
    ],
    "Project": [
        st.Page(about.render, title="About", icon=":material/info:", url_path="about"),
    ],
}
PAGES.update({p.url_path: p for group in pages.values() for p in group})
page = st.navigation(pages)
render_sidebar(engine)
page.run()
