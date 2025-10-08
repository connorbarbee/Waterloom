# Waterloom

Waterloom is a tiny offline-friendly app that turns everyday photos into dreamy watercolor paintings. It now ships with both a
desktop studio (built with PySide6/Qt) and the original Streamlit control panel so you can work the way you prefer—completely
offline on any machine.

## Features

- 🌄 Upload any PNG or JPEG and instantly preview the watercolor result side-by-side.
- 📸 Camera rotation metadata is respected so portraits and landscapes always display correctly.
- 🖥️ Choose between a traditional desktop window or the Streamlit web panel.
- 🎚️ Adjust brush smoothness, color fidelity, ink outlines, paper texture, and more to match your style.
- 💾 Download/save the finished artwork as a high-quality PNG, all without leaving your machine.

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
3. **Pick the interface you want to use.**

   - **Desktop studio (recommended for double-click use):**
     ```bash
     python desktop_app.py
     ```
     The PySide6 window opens instantly with open/save buttons and live previews. This is the easiest route to bundling a
     standalone `.exe`/`.app` later.

   - **Streamlit control panel (runs in your browser):**
     ```bash
     streamlit run app.py
     ```
     Visit the local URL that Streamlit prints (usually <http://localhost:8501>) to tweak settings from your browser.

Both interfaces disable telemetry and work entirely offline once dependencies are installed. On Linux you may need the system
package `libgl1` (or equivalent) so OpenCV can render images. No internet connection is required after installing the Python
packages.

## Build a standalone executable

You can bundle the PySide6 desktop studio into a single-file executable so it can be launched directly from your desktop without
Python. [PyInstaller](https://pyinstaller.org/) works well across platforms.

1. Install PyInstaller (only needed when building the bundle):
   ```bash
   pip install pyinstaller
   ```
2. Run the build (Windows example shown—macOS/Linux use the same command):
   ```bash
   pyinstaller --onefile --windowed desktop_app.py --name Waterloom
   ```
3. Your packaged app will appear in `dist/Waterloom.exe` (or just `Waterloom` on macOS/Linux). Copy this file anywhere and double
   click to run. To include a custom icon, place a `waterloom.ico`/`waterloom.icns` next to `desktop_app.py` and add
   `--icon waterloom.ico` (or `.icns`) to the command.

The generated executable embeds Python and all dependencies, so it works completely offline on any PC that supports the bundled
platform.

## How it works

- The shared logic in `waterloom_core.py` processes images with OpenCV's `cv2.stylization`, adds adaptive ink edges, respects EXIF
  orientation metadata, and mixes in a procedurally generated paper texture with optional vibrance/brightness tweaks.
- `desktop_app.py` wraps the core pipeline in a PySide6 window with file dialogs, live previews, and save/export buttons.
- `app.py` exposes the same controls through Streamlit for those who prefer a browser-based panel.

Enjoy turning your photographs into luminous watercolor artworks! 🎨
