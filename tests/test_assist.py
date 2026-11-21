"""Tests for the assistive reader: spoken descriptions, fruit names and the photo-quality check."""
import cv2
import numpy as np
import pytest

from src.assist.describe import describe, full_message
from src.assist.fruit_names import FRUIT_NAMES, display_name, names_for
from src.assist.quality import check_quality
from src.utils.detections import Detection
from src.utils.helpers import load_config
from src.utils.image_utils import read_image

W = 600


def det(cls, conf, x):
    return Detection(cls, conf, (x, 10, x + 40, 50), 0)


def test_single_fruit_with_filipino_name():
    assert describe([det("mango", 0.9, 10)], W) == "I see 1 mango (mangga), on the left."


def test_counts_positions_and_plural():
    text = describe([det("mango", 0.9, 10), det("mango", 0.8, 60), det("mango", 0.7, 400)], W)
    assert text == "I see 3 mangoes (mangga): 2 on the left and 1 on the right."


def test_most_confident_type_first_and_certainty_words():
    text = describe([det("banana", 0.5, 250), det("guava", 0.95, 10), det("lychee", 0.3, 500)], W)
    assert text.index("guava") < text.index("banana") < text.index("lychee")
    assert "I probably see 1 banana" in text
    assert "lychee, on the right, but I am not sure" in text


def test_large_groups_are_many():
    text = describe([det("orange", 0.9, x) for x in range(0, 560, 70)], W)
    assert text == "I see many oranges (dalandan), across the photo."


def test_no_filipino_option_and_nothing_found():
    assert describe([det("mango", 0.9, 10)], W, filipino=False) == "I see 1 mango, on the left."
    assert full_message([], W).startswith("I don't see any fruit")
    assert full_message([], W, "The photo is too dark.").startswith("The photo is too dark.")


@pytest.mark.parametrize("config", ["config/config.yaml", "config/config_fruit30.yaml"])
def test_every_class_has_a_spoken_name(config):
    for cls in load_config(config)["classes"]:
        assert cls in FRUIT_NAMES, cls
        assert names_for(cls)[0]


def test_display_name():
    assert display_name("sapodilla") == "Sapodilla (chico)"
    assert display_name("durian") == "Durian"


@pytest.fixture(scope="module")
def photo():
    return read_image("assets/unseen/mandarins_on_plate.jpg")


def test_quality_ok_on_a_normal_photo(photo):
    assert check_quality(photo).ok


def test_quality_flags_dark_photo(photo):
    q = check_quality(cv2.convertScaleAbs(photo, alpha=0.3))
    assert q.issues[0] == "dark" and "too dark" in q.message


def test_quality_flags_blur_and_motion_blur(photo):
    assert "blurry" in check_quality(cv2.GaussianBlur(photo, (31, 31), 0)).issues
    k = np.zeros((31, 31), np.float32)
    k[15, :] = 1 / 31
    assert "motion_blur" in check_quality(cv2.filter2D(photo, -1, k)).issues


def test_quality_flags_overexposed_photo(photo):
    assert "overexposed" in check_quality(cv2.convertScaleAbs(photo, alpha=2.5)).issues


def test_phone_qr_code_decodes_to_the_url():
    from src.assist.phone import phone_url, qr_image
    url = "http://192.168.1.23:8501"
    text, _, _ = cv2.QRCodeDetector().detectAndDecode(qr_image(url))
    assert text == url
    found = phone_url(8501)
    assert found is None or (found.startswith("http://") and found.endswith(":8501"))


def test_https_proxy_forwards_to_the_app(tmp_path):
    """A request over TLS to the proxy reaches a plain-http server unchanged."""
    import http.server
    import socket
    import ssl
    import subprocess
    import threading
    import urllib.request

    from src.assist.https_cert import find_openssl
    from src.assist.https_proxy import start_https_proxy

    openssl = find_openssl()
    if openssl is None:
        pytest.skip("OpenSSL not available")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-keyout", str(key),
                    "-out", str(cert), "-subj", "/CN=test", "-addext", "subjectAltName=IP:127.0.0.1"],
                   check=True, capture_output=True, timeout=60)

    class Hello(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"hello " + self.path.encode())

        def log_message(self, *args):
            pass

    app = http.server.HTTPServer(("127.0.0.1", 0), Hello)
    threading.Thread(target=app.serve_forever, daemon=True).start()
    with socket.socket() as s:  # a free port for the proxy
        s.bind(("127.0.0.1", 0))
        proxy_port = s.getsockname()[1]
    proxy = start_https_proxy(cert, key, proxy_port, app.server_address[1], listen_host="127.0.0.1")
    try:
        ctx = ssl.create_default_context(cafile=str(cert))
        body = urllib.request.urlopen(f"https://127.0.0.1:{proxy_port}/fruit", context=ctx, timeout=10).read()
        assert body == b"hello /fruit"
    finally:
        proxy.shutdown()
        app.shutdown()


def test_live_announcer_waits_for_a_stable_scene_and_does_not_repeat():
    from src.assist.live import LiveAnnouncer
    a = LiveAnnouncer()
    say = lambda types, t, **kw: a.update(set(types), "I see a mango.", now=t, **kw)
    assert say(["mango"], 0) == "" and say(["mango"], 1) == ""
    assert say(["mango"], 2) == "I see a mango."           # 3rd stable frame
    assert say(["mango"], 3) == "" and say(["mango"], 10) == ""   # no repeat
    assert say(["mango"], 18) == "I see a mango."          # still pointing after 15 s
    assert say(["mango", "banana"], 19) == ""              # scene changed: wait again
    assert say(["mango", "banana"], 20) == ""
    assert say(["mango", "banana"], 21) == "I see a mango."


def test_live_announcer_resets_after_empty_frames_and_reports_photo_problems():
    from src.assist.live import LiveAnnouncer
    a = LiveAnnouncer()
    for t in range(3):
        a.update({"guava"}, "guava", now=t)
    for t in range(3, 6):
        assert a.update(set(), "", "dark", "Too dark.", now=t) == ("Too dark." if t == 5 else "")
    for t in range(6, 8):
        assert a.update({"guava"}, "guava", now=t) == ""
    assert a.update({"guava"}, "guava", now=8) == "guava"  # announced again after the scene was cleared
