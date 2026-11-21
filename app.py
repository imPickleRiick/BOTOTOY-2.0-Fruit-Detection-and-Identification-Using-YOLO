"""BOTOTOY 2.0 - Fruit Detection and Identification Using YOLO (Streamlit user interface).

Run from the project root, with the virtual environment activated:
    streamlit run app.py
or simply:
    python app.py          (also the VS Code "Run" button)
    python app.py --https  (lets a phone on the same Wi-Fi use its LIVE camera; see the Fruit Reader page)
"""
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

if __name__ == "__main__" and get_script_run_ctx(suppress_warning=True) is None:
    # Started with plain `python app.py`: relaunch through Streamlit, which serves the app in the browser.
    import sys
    from pathlib import Path

    from streamlit.web import cli

    import socket

    args = sys.argv[1:]
    port = int(args[args.index("--server.port") + 1]) if "--server.port" in args else 8501
    with socket.socket() as probe:  # another copy of the app already running? Then stop with a clear message
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            print(f"\n  Port {port} is already in use: another copy of the app is probably still running.\n"
                  f"  Close it first (Ctrl+C in its terminal), or start this one on another port:\n"
                  f"    python app.py --server.port {port + 1}\n")
            sys.exit(1)
    if "--server.port" not in args:
        args += ["--server.port", str(port)]  # never let Streamlit silently switch ports
    if "--https" in args:
        # The app stays on http; a small HTTPS proxy in front of it lets phones use the live camera.
        import os

        from src.assist.https_cert import ensure_certificate
        from src.assist.https_proxy import start_https_proxy
        args.remove("--https")
        https_port = int(os.environ.get("BOTOTOY_HTTPS_PORT", port + 942))  # 8501 -> 9443
        cert, key, ip = ensure_certificate()
        start_https_proxy(cert, key, https_port, port)
        os.environ["BOTOTOY_HTTPS_PORT"] = str(https_port)
        print(f"\n  HTTPS for phones: https://{ip or 'localhost'}:{https_port}  "
              "(accept the security warning once on the phone)\n")
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *args]
    sys.exit(cli.main())

from src.app.common import PAGES, get_engine, render_sidebar
from src.app.pages import about, baseline, batch, dashboard, detection, evaluation, reader, yolo

st.set_page_config(page_title="Fruit Reader", page_icon=":material/nutrition:", layout="wide",
                   initial_sidebar_state="collapsed")

engine = get_engine()

reader_page = st.Page(reader.render, title="Fruit Reader", icon=":material/record_voice_over:", url_path="reader",
                      default=True)
pages = {
    "Reader": [reader_page],
    "Project": [
        st.Page(dashboard.render, title="Dashboard", icon=":material/dashboard:", url_path="dashboard"),
        st.Page(detection.render, title="Fruit Detection", icon=":material/image_search:", url_path="detect"),
        st.Page(batch.render, title="Batch Detection", icon=":material/photo_library:", url_path="batch"),
        st.Page(yolo.render, title="YOLO11n Model", icon=":material/neurology:", url_path="yolo"),
        st.Page(baseline.render, title="OpenCV Baseline", icon=":material/tune:", url_path="baseline"),
        st.Page(evaluation.render, title="Model Evaluation", icon=":material/monitoring:", url_path="evaluation"),
        st.Page(about.render, title="About", icon=":material/info:", url_path="about"),
    ],
}
PAGES.update({p.url_path: p for group in pages.values() for p in group})
page = st.navigation(pages)
if page is not reader_page:  # the reader has its own settings; keep its sidebar empty
    render_sidebar(engine)
page.run()
