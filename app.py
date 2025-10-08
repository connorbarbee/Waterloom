"""Streamlit entry point for Waterloom's watercolor stylization."""

from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image
import streamlit as st

from waterloom_core import (
    WatercolorSettings,
    stylize_image,
    to_pil_image,
)


def set_page_config() -> None:
    """Configure Streamlit page properties once at startup."""

    st.set_page_config(
        page_title="Waterloom — Watercolor Your Photos",
        page_icon="🎨",
        layout="centered",
    )


def download_button(image: Image.Image) -> None:
    """Render a download button for the stylized image."""

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    st.download_button(
        label="Download watercolor PNG",
        data=buffer,
        file_name="waterloom-watercolor.png",
        mime="image/png",
    )


def render_sidebar() -> WatercolorSettings:
    """Display controls and return configured settings."""

    st.sidebar.header("🎚️ Watercolor controls")
    smoothness = st.sidebar.slider(
        "Watercolor smoothness",
        min_value=10,
        max_value=200,
        value=65,
        help=(
            "Higher values increase the size of watercolor strokes."
            " Lower values keep more detail."
        ),
    )
    fidelity = st.sidebar.slider(
        "Color fidelity",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Smaller values create a looser, dreamier effect.",
    )
    edge_strength = st.sidebar.slider(
        "Edge ink intensity",
        min_value=0,
        max_value=255,
        value=80,
        help="Controls how prominent the outlining ink effect is.",
    )
    edge_blur = st.sidebar.slider(
        "Edge softness",
        min_value=1,
        max_value=9,
        value=3,
        step=2,
        help="Larger values soften outlines, smaller values sharpen them.",
    )
    texture_intensity = st.sidebar.slider(
        "Paper texture",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        help="Blend in handmade watercolor paper grain for added realism.",
    )
    vibrance = st.sidebar.slider(
        "Vibrance",
        min_value=0.0,
        max_value=0.5,
        value=0.15,
        step=0.05,
        help="Boost color intensity while protecting existing saturation.",
    )
    brightness = st.sidebar.slider(
        "Brightness",
        min_value=-0.3,
        max_value=0.3,
        value=0.05,
        step=0.05,
        help="Fine-tune overall luminance after stylization.",
    )
    max_edge = st.sidebar.slider(
        "Max edge size",
        min_value=720,
        max_value=4096,
        value=1920,
        step=240,
        help="Larger values keep more detail but need more processing time.",
    )
    return WatercolorSettings(
        smoothness=smoothness,
        fidelity=fidelity,
        edge_strength=edge_strength,
        edge_blur=edge_blur,
        texture_intensity=texture_intensity,
        vibrance=vibrance,
        brightness=brightness,
        max_edge=max_edge,
    )


def main() -> None:
    set_page_config()
    st.title("Waterloom")
    st.write(
        "Upload any photo and transform it into a hand-painted watercolor in seconds."
    )

    settings = render_sidebar()

    uploaded_file = st.file_uploader(
        "Drop an image file (PNG, JPG, or JPEG)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is None:
        st.info("Start by uploading a photo — portraits and landscapes work beautifully!")
        return

    original_image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(original_image)

    with st.spinner("Mixing pigments and brushing strokes..."):
        stylized_array = stylize_image(
            image_array,
            settings.smoothness,
            settings.fidelity,
            settings.edge_strength,
            settings.edge_blur,
            settings.texture_intensity,
            settings.vibrance,
            settings.brightness,
            settings.max_edge,
        )

    stylized_image = to_pil_image(stylized_array)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original", use_column_width=True)
    with col2:
        st.image(stylized_image, caption="Watercolor", use_column_width=True)

    st.success("All done! Save your watercolor masterpiece below.")
    download_button(stylized_image)


if __name__ == "__main__":
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    main()
