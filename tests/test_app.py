"""Headless tests of the Streamlit pages (streamlit.testing): every page renders without errors
and the detection pages show real results for a demonstration image."""
import pytest
from streamlit.testing.v1 import AppTest

from src.utils.helpers import PROJECT_ROOT

TRAINED = PROJECT_ROOT / "models" / "trained" / "bototoy_yolo11n.pt"
needs_model = pytest.mark.skipif(not TRAINED.is_file(), reason="trained YOLO weights not installed")
TIMEOUT = 300


def _page_script(page_name):
    import importlib
    importlib.import_module(f"src.app.pages.{page_name}").render()


def run_page(page: str, **state) -> AppTest:
    at = AppTest.from_function(_page_script, args=(page,), default_timeout=TIMEOUT)
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_full_app_starts_on_fruit_reader():
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert at.title[0].value == "Fruit Reader"
    assert not at.sidebar.radio  # minimal reader: the v1 detection settings stay off its sidebar


@pytest.mark.parametrize("page", ["reader", "dashboard", "evaluation", "about", "batch", "detection", "baseline"])
def test_pages_render_without_input(page):
    at = run_page(page)
    assert not at.exception, at.exception


@needs_model
def test_detection_page_yolo_on_demo_image():
    at = run_page("detection", method="yolo", detect_source="Demo image")
    assert not at.exception
    assert int(metric(at, "Total fruits")) >= 0
    assert "ms" in metric(at, "End-to-end time")
    assert len(at.get("download_button")) == 2


def test_detection_page_baseline_on_demo_image():
    at = run_page("detection", method="baseline", detect_source="Demo image")
    assert not at.exception
    assert int(metric(at, "Total fruits")) >= 0


def test_baseline_page_shows_processing_steps():
    at = run_page("baseline", baseline_source="Demo image")
    assert not at.exception
    assert len(at.tabs) == 5  # one tab per processing stage


@needs_model
def test_yolo_page_threshold_explorer():
    at = run_page("yolo", yolo_source="Demo image")
    assert not at.exception
    assert any(s.label == "Confidence threshold" for s in at.slider)


@needs_model
def test_batch_page_processes_demo_images():
    at = run_page("batch", method="yolo", batch_demo=True)
    at.button[0].click().run()
    assert not at.exception
    assert metric(at, "Images processed") == "8"
    assert metric(at, "Failed") == "0"
