"""Watercolorizer Streamlit App.

This application allows users to upload images and transform them into watercolor-style
paintings completely offline using OpenCV's stylization filter combined with edge
blending tweaks for a painterly look.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
import streamlit as st


def set_page_config() -> None:
    """Configure Streamlit page properties once at startup."""
    st.set_page_config(
        page_title="Waterloom — Watercolor Your Photos",
        page_icon="🎨",
        layout="centered",
    )


@dataclass
class WatercolorSettings:
    """Tunable parameters for the watercolor effect."""

    smoothness: int = 65
    fidelity: float = 0.45
    edge_strength: int = 80
    edge_blur: int = 3


@st.cache_data(show_spinner=False)
def stylize_image(
    image: np.ndarray,
    smoothness: int,
    fidelity: float,
    edge_strength: int,
    edge_blur: int,
) -> np.ndarray:
    """Convert an RGB image array to a watercolor-styled array."""
    # OpenCV expects BGR, so convert before processing.
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Apply the watercolor stylization filter from OpenCV.
    stylized = cv2.stylization(
        bgr,
        sigma_s=smoothness,
        sigma_r=fidelity,
    )

    # Create gentle ink-like edges to enhance the watercolor effect.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, max(1, edge_blur))
    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2,
    )
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edges = 255 - edges  # invert to get darker outlines

    edge_alpha = np.clip(edge_strength / 255.0, 0.0, 1.0)
    stylized = cv2.addWeighted(stylized, 1.0, edges, edge_alpha, 0)

    # Convert back to RGB for display.
    return cv2.cvtColor(stylized, cv2.COLOR_BGR2RGB)


def to_pil_image(image_array: np.ndarray) -> Image.Image:
    """Convert a numpy RGB array to a PIL Image."""
    return Image.fromarray(image_array.astype(np.uint8), mode="RGB")


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
    return WatercolorSettings(
        smoothness=smoothness,
        fidelity=fidelity,
        edge_strength=edge_strength,
        edge_blur=edge_blur,
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
    # Disable Streamlit telemetry to avoid unnecessary outbound calls so the app can
    # run entirely offline.
    import os

    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    main()
