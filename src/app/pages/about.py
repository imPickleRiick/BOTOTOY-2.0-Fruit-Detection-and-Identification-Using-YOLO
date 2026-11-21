"""About the project: the Fruit Reader (version 2) and the 31-fruit model behind it."""
import streamlit as st

from src.app.common import load_result_json
from src.app.pages.reader import get_reader

TEAM = ["Kyle Angelo D. Mercado", "Ian Patrick Z. Mesias", "James Paul L. Panal"]
REPO = "https://github.com/imPickleRiick/BOTOTOY-2.0-Fruit-Detection-and-Identification-Using-YOLO/tree/v2-assistive"

STYLE = """
<style>
.block-container {max-width: 760px; padding-top: 2.5rem;}
h1 {font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0;}
.about-sub {color: #5f6b64; font-size: 1.1rem; margin: 0.2rem 0 1.4rem;}
</style>
"""


def render() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)
    reader = get_reader()
    split = load_result_json("data/splits_fruit31/summary.json")
    test = load_result_json("outputs/yolo_fruit31/eval_test/metrics.json")

    st.title("About")
    st.markdown('<p class="about-sub">BOTOTOY 2.0 Fruit Reader: an assistive fruit identifier for blind and '
                'low-vision Filipinos.</p>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Course:** CSST 106 – Perception and Computer Vision · BSCS 4A  \n"
                    "**Instructor:** Mr. Jeremy Reyes  \n"
                    "**Team:** " + " · ".join(TEAM))

    st.subheader("The problem")
    st.markdown(
        "The Department of Health estimated in 2017 that about 332,000 Filipinos are blind in both eyes and about "
        "2.18 million have low vision. At home, at a sari-sari store or at the palengke, fruit is often sold "
        "loose and without labels, and many fruits feel alike. The Fruit Reader lets a person point a phone at a "
        "fruit and **hear** what it is: the name in English and Filipino, how many, and where.")

    st.subheader("How it works")
    st.markdown(
        "1. **Photo check** (classic computer vision): brightness, glare, sharpness and motion blur. If the photo "
        "is bad, the reader says what to fix.\n"
        "2. **YOLO11n** (deep learning) finds every fruit: a box, the fruit's name and a confidence score.\n"
        "3. **One short sentence**: the name, the count (\"many\" above five), left / middle / right, and "
        "\"probably\" or \"not sure\" when the model is uncertain.\n"
        "4. **Speech** on the phone's browser. In live mode it speaks when the same fruit is seen in 3 frames "
        "in a row, so no button is needed.")

    st.subheader("The model")
    facts = [f"YOLO11n, {len(reader.classes)} fruits, transfer learning from COCO-pretrained weights",
             "Trained on a Google Colab T4 GPU; the free GPU quota ran out at epoch 45 of 50, so the best "
             "checkpoint (epoch 44) is used",
             "Runs on an ordinary laptop CPU; the phone is only the camera, screen and speaker"]
    if test:
        facts.append(f"Locked test set ({test['shared']['images']:,} images): mAP@0.50 "
                     f"{test['ultralytics']['mAP50']:.3f}, mAP@0.50:0.95 {test['ultralytics']['mAP50-95']:.3f}, "
                     f"{test['shared']['timing_ms']['end_to_end_mean']:.0f} ms per photo")
    st.markdown("\n".join(f"- {f}" for f in facts))

    st.subheader("Data")
    total = f"{sum(s['images'] for s in split['final'].values()):,} images" if split else "Two datasets"
    st.markdown(
        f"{total}, merged from two public datasets, with near-duplicate photos removed so no test photo was "
        "seen in training:\n"
        "- [\"Fruit\" by trucly](https://universe.roboflow.com/trucly/fruit-5ix6x/dataset/1), Roboflow Universe "
        "(CC BY 4.0): 30 tropical and common fruits\n"
        "- [Fruit Detection](https://github.com/lightly-ai/dataset_fruits_detection) (CC0-1.0): adds apple and "
        "grape\n\n"
        "Tangerines are read as oranges (dalandan).")

    st.subheader("Limits")
    st.markdown(
        "- Not yet tested with blind or low-vision users.\n"
        "- Knows 31 fruits only (no calamansi, atis, santol or langka yet).\n"
        "- Unusual colors can fool it (a green orange, a yellow watermelon).\n"
        "- The photo check sometimes gives advice that is not needed (white backgrounds, smooth close-ups).\n"
        "- Does not say whether a fruit is ripe, spoiled or safe to eat.")

    st.subheader("Versions")
    st.markdown("- **Version 2 (this one):** the Fruit Reader, 31 fruits, spoken results, live phone camera.\n"
                "- **Version 1 (tag `v1.0`):** a 6-fruit detector with boxes on screen; its pages are still "
                "under *Project* in the menu.")

    st.caption("Built with Python · Ultralytics YOLO11 · PyTorch · OpenCV · Streamlit. "
               f"Source code and reports: [GitHub, branch v2-assistive]({REPO})")
