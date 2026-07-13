"""
app.py
-------
Streamlit web app for oil-spill detection & classification.

Upload an aerial/satellite image -> choose a model (SegFormer-B2 or
Mask2Former) -> see the predicted segmentation mask overlaid on the image,
plus a class-by-class pixel coverage breakdown.

Run locally:
    streamlit run app.py

Deploy:
    Push this repo to GitHub, then deploy on https://share.streamlit.io
"""

import streamlit as st
from PIL import Image

from model_utils import (
    load_segformer,
    load_mask2former,
    run_inference,
    mask_to_rgb,
    overlay_mask_on_image,
    class_pixel_breakdown,
    LADOS_ID2LABEL,
    LADOS_PALETTE,
)

st.set_page_config(
    page_title="Oil Spill Detection & Classification",
    page_icon="🛢️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_segformer():
    return load_segformer()


@st.cache_resource(show_spinner=False)
def get_mask2former():
    return load_mask2former()


def main():
    st.title("🛢️ Oil Spill Detection & Classification")
    st.markdown(
        "Upload an aerial or satellite image of a marine/coastal scene. "
        "The model will segment the image into **Background, Oil, Emulsion, "
        "Sheen, Ship,** and **Oil-platform** regions."
    )

    with st.sidebar:
        st.header("Settings")
        model_choice = st.radio(
            "Choose model",
            options=["SegFormer-B2", "Mask2Former"],
            help="Two transformer-based segmentation models trained on the LADOS dataset.",
        )
        alpha = st.slider("Mask overlay opacity", 0.0, 1.0, 0.5, 0.05)
        st.markdown("---")
        st.markdown("**Class colour legend**")
        for cls_id, name in LADOS_ID2LABEL.items():
            r, g, b = LADOS_PALETTE[cls_id]
            st.markdown(
                f"<span style='display:inline-block;width:14px;height:14px;"
                f"background-color:rgb({r},{g},{b});margin-right:6px;'></span>{name}",
                unsafe_allow_html=True,
            )

    uploaded_file = st.file_uploader(
        "Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        st.info("👆 Upload an image to get started.")
        return

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original image")
        st.image(image, use_container_width=True)

    with st.spinner(f"Loading {model_choice} and running inference..."):
        if model_choice == "SegFormer-B2":
            model, processor = get_segformer()
            model_type = "segformer"
        else:
            model, processor = get_mask2former()
            model_type = "mask2former"

        pred_mask = run_inference(model, processor, image, model_type)
        mask_rgb = mask_to_rgb(pred_mask, LADOS_PALETTE)
        overlay = overlay_mask_on_image(image, mask_rgb, alpha=alpha)

    with col2:
        st.subheader(f"Prediction ({model_choice})")
        st.image(overlay, use_container_width=True)

    st.subheader("Class coverage breakdown")
    breakdown = class_pixel_breakdown(pred_mask, LADOS_ID2LABEL)
    cols = st.columns(len(breakdown))
    for col, (name, pct) in zip(cols, breakdown.items()):
        col.metric(name, f"{pct}%")

    oil_related_pct = sum(
        pct for name, pct in breakdown.items() if name in ("Oil", "Emulsion", "Sheen")
    )
    if oil_related_pct > 0:
        st.warning(
            f"⚠️ Approximately **{round(oil_related_pct, 2)}%** of the image is "
            f"classified as oil-related (Oil, Emulsion, or Sheen)."
        )
    else:
        st.success("✅ No oil-related classes detected in this image.")


if __name__ == "__main__":
    main()
