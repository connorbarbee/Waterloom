# Waterloom

Waterloom is a tiny offline-friendly app that turns everyday photos into dreamy watercolor paintings. The interface runs locally using [Streamlit](https://streamlit.io/) and relies on OpenCV's stylization filters plus a touch of edge inking for a hand-painted finish.

## Features

- 🌄 Upload any PNG or JPEG and instantly preview the watercolor result side-by-side.
- 🎚️ Adjust brush smoothness, color fidelity, ink outlines, paper texture, and more to match your style.
- 💾 Download the finished artwork as a high-quality PNG, all without leaving your machine.

## Getting started

1. **Create a virtual environment (optional but recommended).**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
2. **Install the dependencies.**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the app locally.**
   ```bash
   streamlit run app.py
   ```
4. Visit the local URL that Streamlit prints (usually <http://localhost:8501>) and start transforming your images.

The app disables Streamlit telemetry so it can be used completely offline once the dependencies are installed. If you are on Linux
you may need the system package `libgl1` (or equivalent) so OpenCV can render images. No internet connection is required after
installing the Python packages.

## How it works

- Uploaded images are processed with OpenCV's `cv2.stylization`, which produces watercolor-like diffusion of colors.
- Adaptive thresholding generates subtle ink-like edges that are blended back into the painting for a handcrafted feel.
- A procedurally generated watercolor paper texture and optional vibrance/brightness tweaks provide gallery-ready results.
- Streamlit caches results for repeated adjustments, keeping experimentation snappy.

Enjoy turning your photographs into luminous watercolor artworks! 🎨
