"""Fruit Reader: the assistive page. Point the camera -> the app says which fruits it sees and where.

Accessibility choices:
- "Live reader" (default): the camera streams, boxes appear live, and the reader speaks by itself when
  the same fruit stays in view for 2-3 seconds. Also "take or upload a photo" (on a phone this opens the
  phone camera) and a single-photo camera; a QR code opens the reader on a phone on the same Wi-Fi.
- The answer is spoken (browser speech synthesis) and also placed in an ARIA live region, so screen
  readers (TalkBack, VoiceOver, NVDA) read it even if the browser blocks automatic speech.
- Large, high-contrast text; photo problems (dark, blurry) are said first, with what to do.
"""
from __future__ import annotations

import base64
import html
import json
import os

import streamlit as st
import streamlit.components.v1 as components

from src.app.components import live_camera
from src.assist.describe import describe
from src.assist.fruit_names import display_name
from src.assist.live import LiveAnnouncer
from src.assist.phone import phone_url, qr_image
from src.assist.reader import FruitReader
from src.utils.detections import class_color
from src.inference.image_processor import SUPPORTED_FORMATS, InvalidImageError
from src.inference.visualization import fit_within, to_rgb

STYLE = """
<style>
.reader-msg {font-size: 1.9rem; font-weight: 650; line-height: 1.35; padding: 18px 22px; margin: 8px 0 4px;
             border: 3px solid currentColor; border-radius: 14px;}
.reader-ok {color: #0b5d2a; background: #eaf6ee;}
.reader-warn {color: #7a2e00; background: #fff1e0;}
@media (prefers-color-scheme: dark) {
  .reader-ok {color: #b9f0c9; background: #12301d;} .reader-warn {color: #ffd2a8; background: #3a2208;}
}
div[data-testid="stCameraInput"] button {font-size: 1.3rem; padding: 0.8rem 1.4rem;}
</style>
"""


LIVE_READER = "Live reader (automatic)"
TAKE_OR_UPLOAD = "Take or upload a photo"
LIVE_CAMERA = "Live camera (this device)"


def _hex(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def live_mode(reader: FruitReader, conf: float, filipino: bool, rate: float) -> None:
    """Continuous reading: the camera sends ~1 frame per second, the reader speaks when the scene is stable."""
    state = st.session_state.setdefault("live_state", {"seq": 0, "result": None, "said": "",
                                                       "announcer": LiveAnnouncer()})

    def process(value: dict | None) -> bool:
        """Read one new frame from the browser; True if it was new."""
        if not isinstance(value, dict) or value.get("seq", 0) <= state["seq"]:
            return False
        state["seq"] = value["seq"]
        try:
            data = base64.b64decode(value["frame"].split(",", 1)[1])
            r = reader.read_bytes(data, "live.jpg", conf=conf, filipino=filipino)
        except (InvalidImageError, ValueError, KeyError, IndexError):
            state["result"] = {"seq": value["seq"], "w": 1, "h": 1, "boxes": [], "say": "",
                               "status": "Bad frame, trying again."}
            return True
        dets = r.detection.detections
        h, w = r.detection.image.shape[:2]
        sentence = describe(dets, w, filipino)
        issue = r.quality.issues[0] if r.quality.issues else ""
        say = state["announcer"].update({d.class_name for d in dets}, sentence, issue, r.quality.message)
        if say:
            state["said"] = say
        status = sentence + " Hold still..." if dets else (r.quality.message or "Looking for fruit...")
        state["result"] = {
            "seq": value["seq"], "w": w, "h": h, "say": say, "status": status,
            "boxes": [[*map(int, d.box), f"{display_name(d.class_name, False)} {d.confidence:.0%}",
                       _hex(class_color(d.class_name))] for d in dets]}
        return True

    # The browser's newest frame is already in session_state when this run starts: answer it BEFORE drawing,
    # so each frame costs one redraw and the page does not flicker.
    process(st.session_state.get("reader_live"))
    value = live_camera(result=state["result"], key="reader_live", rate=rate)
    if value is None:  # not started, or stopped
        state.update(seq=0, result=None, announcer=LiveAnnouncer())
    elif process(value):  # fallback if the value was not in session_state yet
        st.rerun()

    if state["said"]:
        st.markdown(f'<div class="reader-msg reader-ok">{html.escape(state["said"])}</div>', unsafe_allow_html=True)
    st.caption("The reader speaks when the same fruit stays in view for about 2–3 seconds, and again after "
               "15 seconds if you keep pointing at it. On a phone, open the https address (see "
               "\"Use your phone camera\").")


def phone_panel() -> None:
    """QR code + steps to open the reader on a phone and use its camera."""
    with st.expander("Use your phone camera", icon=":material/smartphone:"):
        https_port = os.environ.get("BOTOTOY_HTTPS_PORT")  # set by `python app.py --https`
        https = bool(https_port)
        url = phone_url(int(https_port or st.get_option("server.port")), https)
        if url is None:
            st.warning("This computer is not connected to a network, so a phone cannot open the reader.")
            return
        left, right = st.columns([1, 2])
        left.image(qr_image(url), caption=url, width=220)
        steps = ("1. Connect the phone to the **same Wi-Fi** as this computer.\n"
                 f"2. Scan the QR code with the phone camera, or type **{url}** in the phone's browser.\n")
        if https:
            steps += ("3. The phone warns that the connection is not private (the certificate is made by this "
                      "laptop). Android Chrome: **Advanced → Proceed**. iPhone Safari: **Show Details → visit "
                      "this website**. This is needed only once.\n"
                      f"4. Choose **{LIVE_CAMERA}** and allow camera access, or use **{TAKE_OR_UPLOAD}**.\n")
        else:
            steps += (f"3. Tap **{TAKE_OR_UPLOAD}**, then choose **Camera** and take the photo.\n\n"
                      f"For the **{LIVE_CAMERA}** option on the phone, stop the app and start it with "
                      "`python app.py --https` (phone browsers allow the live camera only on https pages).\n")
        right.markdown(steps + "\nIf the page does not open: when Windows asks whether to allow Python on the "
                       "network, choose **Allow**, or connect both devices to the phone's hotspot.")


@st.cache_resource(show_spinner="Loading the fruit reader ...")
def get_reader() -> FruitReader:
    return FruitReader()


@st.cache_data(max_entries=8, show_spinner=False)
def read_cached(data: bytes, name: str, conf: float, filipino: bool) -> dict:
    r = get_reader().read_bytes(data, name, conf=conf, filipino=filipino)
    return {"message": r.message, "issues": r.quality.issues, "quality": r.quality.as_dict(),
            "annotated": to_rgb(fit_within(r.annotated, 1200)), "timing": r.timing_ms,
            "count": r.detection.total, "notice": r.detection.notice}


def speak(text: str, rate: float, autoplay: bool) -> None:
    """Browser speech synthesis with a large 'read again' button (autoplay may need a first click)."""
    components.html(f"""
<button id="say" style="font-size:22px;font-weight:600;padding:14px 26px;border-radius:12px;border:2px solid #0b5d2a;
        background:#0b5d2a;color:#fff;cursor:pointer" aria-label="Read the result aloud again">🔊 Read aloud again</button>
<script>
const text = {json.dumps(text)};
function say() {{
  if (!('speechSynthesis' in window)) return;
  const u = new SpeechSynthesisUtterance(text); u.lang = 'en-US'; u.rate = {rate};
  window.speechSynthesis.cancel(); window.speechSynthesis.speak(u);
}}
document.getElementById('say').addEventListener('click', say);
{'say();' if autoplay else ''}
</script>""", height=74)


def render() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)
    reader = get_reader()
    st.title("Fruit Reader")
    st.markdown("#### Point the camera at the fruit and hold still. The answer is read aloud.")

    with st.expander("Reader settings", icon=":material/settings:"):
        filipino = st.toggle("Say Filipino names too (e.g. mango, mangga)", value=True, key="reader_fil")
        rate = st.slider("Speech speed", 0.6, 1.4, 0.95, 0.05, key="reader_rate")
        autoplay = st.toggle("Read the result automatically", value=True, key="reader_auto")
        conf = st.slider("Minimum confidence", 0.10, 0.80, float(reader.engine.yolo_defaults["conf"]), 0.05,
                         key="reader_conf", help="Lower finds more fruits but makes more mistakes.")
        st.caption(f"Model: {reader.model_note}.")

    phone_panel()
    source = st.segmented_control("Photo source", [LIVE_READER, TAKE_OR_UPLOAD, LIVE_CAMERA], default=LIVE_READER,
                                  key="reader_source", label_visibility="collapsed")
    if source == LIVE_READER:
        live_mode(reader, conf, filipino, rate)
        return
    if source == LIVE_CAMERA:
        file = st.camera_input("Take a photo of the fruit", key="reader_camera")
    else:
        file = st.file_uploader("Take a photo (on a phone, choose Camera) or choose a photo file",
                                type=list(SUPPORTED_FORMATS), key="reader_upload")
    if file is None:
        st.info("Tip: hold the fruit about an arm's length away, in good light, against a plain background.",
                icon=":material/lightbulb:")
        return

    data = file.getvalue()
    try:
        with st.spinner("Looking at the photo ..."):
            out = read_cached(data, getattr(file, "name", "camera.jpg"), conf, filipino)
    except InvalidImageError as exc:
        st.error(str(exc))
        speak(str(exc), rate, autoplay)
        return

    kind = "reader-warn" if out["issues"] or out["count"] == 0 else "reader-ok"
    st.markdown(f'<div class="reader-msg {kind}" role="status" aria-live="assertive">{html.escape(out["message"])}</div>',
                unsafe_allow_html=True)
    speak(out["message"], rate, autoplay)

    st.image(out["annotated"], caption=f"What the reader found ({out['count']} fruit(s)), for a sighted helper",
             width="stretch")
    with st.expander("Details", icon=":material/info:"):
        q = out["quality"]
        st.markdown(f"- Photo check: sharpness {q['sharpness']}, direction ratio {q['direction_ratio']}, "
                    f"brightness {q['brightness']} → {', '.join(q['issues']) or 'OK'}\n"
                    f"- Time: quality check {out['timing']['quality_check']} ms, detection {out['timing']['detection']} ms, "
                    f"total {out['timing']['total']} ms")
        if out["notice"]:
            st.warning(out["notice"])
