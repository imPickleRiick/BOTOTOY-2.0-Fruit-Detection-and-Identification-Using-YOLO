"""About the project."""
import streamlit as st

TEAM = ["Kyle Angelo D. Mercado", "Ian Patrick Z. Mesias", "James Paul L. Panal"]


def render() -> None:
    st.title("About")
    with st.container(border=True):
        st.markdown("**Project:** BOTOTOY 2.0  \n"
                    "**Title:** Fruit Detection and Identification Using YOLO  \n"
                    "**Course:** CSST 106 - Perception and Computer Vision")
        st.markdown("**Team members**\n" + "\n".join(f"- {name}" for name in TEAM))

    st.subheader("What the system does")
    st.markdown(
        "BOTOTOY 2.0 detects and identifies six kinds of fruit (apple, banana, grape, orange, pineapple, watermelon) "
        "in photos. For every fruit it reports a bounding box, the fruit type and a confidence score, and it counts "
        "the fruits per type and in total. Its scope is detection, identification and counting only.")

    st.subheader("Methods")
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("**YOLO11n (main method)**\n"
                    "- One-stage deep-learning object detector (Ultralytics)\n"
                    "- Transfer learning from COCO-pretrained weights\n"
                    "- Trained on a Google Colab T4 GPU; runs on a laptop CPU")
    with b.container(border=True):
        st.markdown("**OpenCV baseline (fallback)**\n"
                    "- HSV color segmentation, morphology, contours\n"
                    "- Shape and color features (circularity, elongation, solidity, hue, ...)\n"
                    "- Hand-written classification rules; no training")

    st.subheader("Dataset")
    st.markdown(
        "Kaggle fruit detection dataset "
        "[lakshaytyagi01/fruit-detection](https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection), "
        "used through the GitHub mirror "
        "[lightly-ai/dataset_fruits_detection](https://github.com/lightly-ai/dataset_fruits_detection). "
        "License: CC0-1.0 (public domain). 8,479 images with 36,922 bounding boxes, split 70/15/15 by unique photo "
        "with no leakage between splits.")

    st.subheader("Built with")
    st.markdown("Python · Ultralytics YOLO11 · PyTorch · OpenCV · NumPy · pandas · Streamlit · Altair")
    st.caption("Source code, reports and setup instructions: "
               "[github.com/imPickleRiick/BOTOTOY-2.0-Fruit-Detection-and-Identification-Using-YOLO]"
               "(https://github.com/imPickleRiick/BOTOTOY-2.0-Fruit-Detection-and-Identification-Using-YOLO)")
