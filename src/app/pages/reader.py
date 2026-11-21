"""Fruit Reader: the assistive page. Point the camera -> the app says which fruits it sees and where.

Minimal on purpose: a title, a Live / Photo switch, and one Settings button (speech, confidence and the
QR code for a phone). Everything else stays out of the way.

Accessibility choices:
- "Live" (default): the camera streams, boxes appear live, and the reader speaks by itself when the same
  fruit stays in view for 2-3 seconds. "Photo" takes or uploads one photo (on a phone this opens the camera).
- The answer is spoken (browser speech synthesis) and also placed in an ARIA live region, so screen
  readers (TalkBack, VoiceOver, NVDA) read it even if the browser blocks automatic speech.
- Large text and large touch targets; photo problems (dark, blurry) are said first, with what to do.
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
from src.inference.image_processor import SUPPORTED_FORMATS, InvalidImageError
from src.inference.visualization import fit_within, to_rgb

ACCENT = "#1D6B3E"
BOX_COLOR = "#34C759"   # one color for every live box: the label already says which fruit

STYLE = """
<style>
.block-container {max-width: 760px; padding-top: 2.5rem;}
h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0;}
.reader-sub {color: #5f6b64; font-size: 1.1rem; margin: 0.2rem 0 1.2rem;}
.reader-msg {font-size: 1.6rem; font-weight: 600; line-height: 1.35; padding: 18px 22px; margin: 12px 0 6px;
             border-radius: 16px; border-left: 6px solid currentColor;}
.reader-ok {color: #14532d; background: #ecf5ef;}
.reader-warn {color: #7a3a00; background: #fbf1e4;}
/* keep the Live/Photo switch and the Settings button on one row, also on phones */
div[data-testid="stHorizontalBlock"] {flex-wrap: nowrap !important;}
div[data-testid="stColumn"] {min-width: 0 !important;}
</style>
"""

LIVE = "Live"
PHOTO = "Photo"


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
        status = sentence if dets else (r.quality.message or "Looking for fruit…")
        state["result"] = {
            "seq": value["seq"], "w": w, "h": h, "say": say, "status": status,
            "boxes": [[*map(int, d.box), f"{display_name(d.class_name, False)} {d.confidence:.0%}", BOX_COLOR]
                      for d in dets]}
        return True

    # The browser's newest frame is already in session_state when this run starts: answer it BEFORE drawing,
    # so each frame costs one redraw and the page does not flicker.
    process(st.session_state.get("reader_live"))
    value = live_camera(result=state["result"], key="reader_live", rate=rate)
    if value is None:  # not started, or stopped
        state.update(seq=0, result=None, announcer=LiveAnnouncer())
    elif process(value):  # fallback if the value was not in session_state yet
        st.rerun()


def phone_panel() -> None:
    """QR code + short steps to open the reader on a phone."""
    https_port = os.environ.get("BOTOTOY_HTTPS_PORT")  # set by `python app.py --https`
    https = bool(https_port)
    url = phone_url(int(https_port or st.get_option("server.port")), https)
    if url is None:
        st.write("This computer is not connected to a network, so a phone cannot open the reader.")
        return
    st.image(qr_image(url), width=200)
    st.markdown(f"Scan with the phone, or open **{url}**")
    if https:
        st.caption("Phone and laptop on the same Wi-Fi or hotspot. On the security warning, tap "
                   "Advanced → Proceed (only once).")
    else:
        st.caption("Phone and laptop on the same Wi-Fi. For the live camera on a phone, start the app with "
                   "`python app.py --https`.")


@st.cache_resource(show_spinner="Loading the fruit reader …")
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
<button id="say" aria-label="Read the result aloud again"
  style="font:600 18px 'Source Sans Pro','Source Sans 3',system-ui,sans-serif;min-height:52px;padding:0 26px;
         border-radius:999px;border:1.5px solid {ACCENT};background:transparent;color:{ACCENT};cursor:pointer">
  Read again</button>
<script>
const text = {json.dumps(text)};
function say() {{
  if (!('speechSynthesis' in window)) return;
  const u = new SpeechSynthesisUtterance(text); u.lang = 'en-US'; u.rate = {rate};
  window.speechSynthesis.cancel(); window.speechSynthesis.speak(u);
}}
document.getElementById('say').addEventListener('click', say);
{'say();' if autoplay else ''}
</script>""", height=64)


def settings(reader: FruitReader) -> tuple[bool, float, bool, float]:
    filipino = st.toggle("Say Filipino names (mango, mangga)", value=True, key="reader_fil")
    autoplay = st.toggle("Read results automatically", value=True, key="reader_auto")
    rate = st.slider("Speech speed", 0.6, 1.4, 0.95, 0.05, key="reader_rate")
    conf = st.slider("Minimum confidence", 0.10, 0.80, float(reader.engine.yolo_defaults["conf"]), 0.05,
                     key="reader_conf", help="Lower finds more fruits but makes more mistakes.")
    st.caption(reader.model_note)
    return filipino, rate, autoplay, conf


def render() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)
    reader = get_reader()
    st.title("Fruit Reader")
    st.markdown('<p class="reader-sub">Point the camera at a fruit and hold still. The answer is read aloud.</p>',
                unsafe_allow_html=True)

    mode_col, settings_col = st.columns([3, 2], vertical_alignment="center")
    source = mode_col.segmented_control("Mode", [LIVE, PHOTO], default=LIVE, key="reader_source",
                                        label_visibility="collapsed")
    with settings_col.popover("Settings", icon=":material/tune:", use_container_width=True):
        filipino, rate, autoplay, conf = settings(reader)
        st.divider()
        st.markdown("**Open on a phone**")
        phone_panel()

    if source != PHOTO:  # Live is the default; a deselected switch also falls back to Live
        live_mode(reader, conf, filipino, rate)
        return

    file = st.file_uploader("Take or choose a photo", type=list(SUPPORTED_FORMATS), key="reader_upload")
    if file is None:
        return

    data = file.getvalue()
    try:
        with st.spinner("Looking at the photo …"):
            out = read_cached(data, getattr(file, "name", "camera.jpg"), conf, filipino)
    except InvalidImageError as exc:
        st.markdown(f'<div class="reader-msg reader-warn" role="status" aria-live="assertive">'
                    f'{html.escape(str(exc))}</div>', unsafe_allow_html=True)
        speak(str(exc), rate, autoplay)
        return

    kind = "reader-warn" if out["issues"] or out["count"] == 0 else "reader-ok"
    st.markdown(f'<div class="reader-msg {kind}" role="status" aria-live="assertive">{html.escape(out["message"])}</div>',
                unsafe_allow_html=True)
    speak(out["message"], rate, autoplay)
    st.image(out["annotated"], width="stretch")
    st.caption(f"{out['count']} fruit(s) found in {out['timing']['total']:.0f} ms. Boxes are for a sighted helper.")
    if out["notice"]:
        st.warning(out["notice"])
