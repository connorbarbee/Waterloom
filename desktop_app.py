"""PySide6 desktop interface for the Waterloom watercolor stylizer."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from waterloom_core import WatercolorSettings, stylize_image, to_pil_image


def numpy_to_qimage(image: np.ndarray) -> QImage:
    """Convert an RGB numpy array to a QImage for display."""

    image = np.require(image, np.uint8, "C")
    height, width, channel = image.shape
    bytes_per_line = channel * width
    qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    qimage.ndarray = image  # type: ignore[attr-defined] - keep reference alive
    return qimage


class LabeledSlider(QWidget):
    """A convenience widget combining a label, slider, and numeric display."""

    def __init__(
        self,
        title: str,
        minimum: int,
        maximum: int,
        value: int,
        step: int = 1,
    ) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._title = title
        self.label = QLabel(f"{title}: {value}")
        layout.addWidget(self.label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setSingleStep(step)
        self.slider.setValue(value)
        layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self._update_label)

    def _update_label(self, value: int) -> None:
        self.label.setText(f"{self._title}: {value}")

    def value(self) -> int:
        return int(self.slider.value())

    def setValue(self, value: int) -> None:  # noqa: N802 (Qt API)
        self.slider.setValue(value)


class FloatSlider(QWidget):
    """Slider helper that exposes float values via a paired spin box."""

    def __init__(
        self,
        title: str,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int = 2,
    ) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._title = title
        self.label = QLabel(f"{title}: {value:.2f}")
        layout.addWidget(self.label)

        control_layout = QHBoxLayout()
        layout.addLayout(control_layout)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, int(round((maximum - minimum) / step)))
        self.slider.setValue(int(round((value - minimum) / step)))
        control_layout.addWidget(self.slider)

        self.spinbox = QDoubleSpinBox()
        self.spinbox.setDecimals(decimals)
        self.spinbox.setRange(minimum, maximum)
        self.spinbox.setSingleStep(step)
        self.spinbox.setValue(value)
        control_layout.addWidget(self.spinbox)

        self._minimum = minimum
        self._step = step
        self._decimals = decimals

        self.slider.valueChanged.connect(self._slider_changed)
        self.spinbox.valueChanged.connect(self._spin_changed)

    def _slider_changed(self, raw_value: int) -> None:
        value = self._minimum + raw_value * self._step
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(value)
        self.spinbox.blockSignals(False)
        self.label.setText(f"{self._title}: {value:.2f}")

    def _spin_changed(self, raw_value: float) -> None:
        position = int(round((raw_value - self._minimum) / self._step))
        self.slider.blockSignals(True)
        self.slider.setValue(position)
        self.slider.blockSignals(False)
        self.label.setText(f"{self._title}: {raw_value:.2f}")

    def value(self) -> float:
        return round(self.spinbox.value(), self._decimals)

    def setValue(self, value: float) -> None:  # noqa: N802 (Qt API)
        position = int(round((value - self._minimum) / self._step))
        self.slider.setValue(position)
        self.spinbox.setValue(value)
        self.label.setText(f"{self._title}: {value:.2f}")


class ImagePreview(QLabel):
    """A QLabel that scales pixmaps while keeping aspect ratio."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 1px solid #ccc; background-color: #fdfdfd;")

    def set_image(self, image: Optional[np.ndarray]) -> None:
        if image is None:
            self.setText("No preview available")
            self.setPixmap(QPixmap())
            return

        self.setText("")
        qimage = numpy_to_qimage(image)
        pixmap = QPixmap.fromImage(qimage)
        self.setPixmap(pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def resizeEvent(self, event) -> None:  # noqa: D401, N802 - Qt override
        super().resizeEvent(event)
        if not self.pixmap():
            return
        self.setPixmap(self.pixmap().scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))


class WaterloomWindow(QMainWindow):
    """Main desktop window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Waterloom — Watercolor Studio")
        self.resize(1200, 720)
        self._current_image: Optional[np.ndarray] = None
        self._stylized_image: Optional[np.ndarray] = None

        self._build_ui()
        self._create_menu()

    # UI creation ---------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        self.setCentralWidget(central)

        controls = self._create_controls_panel()
        layout.addWidget(controls, 0)

        previews = self._create_preview_panel()
        layout.addWidget(previews, 1)

    def _create_controls_panel(self) -> QWidget:
        container = QGroupBox("Watercolor controls")
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        self.open_button = QPushButton("Open photo…")
        self.open_button.clicked.connect(self.open_image)
        layout.addWidget(self.open_button)

        self.smoothness_slider = LabeledSlider("Stroke smoothness", 10, 200, 65, 5)
        layout.addWidget(self.smoothness_slider)

        self.fidelity_slider = FloatSlider("Color fidelity", 0.1, 1.0, 0.45, 0.05)
        layout.addWidget(self.fidelity_slider)

        self.edge_strength_slider = LabeledSlider("Edge ink", 0, 255, 80, 5)
        layout.addWidget(self.edge_strength_slider)

        self.edge_blur_slider = LabeledSlider("Edge softness", 1, 9, 3, 2)
        layout.addWidget(self.edge_blur_slider)

        self.texture_slider = FloatSlider("Paper texture", 0.0, 1.0, 0.35, 0.05)
        layout.addWidget(self.texture_slider)

        self.vibrance_slider = FloatSlider("Vibrance", 0.0, 0.5, 0.15, 0.05)
        layout.addWidget(self.vibrance_slider)

        self.brightness_slider = FloatSlider("Brightness", -0.3, 0.3, 0.05, 0.05)
        layout.addWidget(self.brightness_slider)

        self.max_edge_slider = LabeledSlider("Max edge (px)", 720, 4096, 1920, 120)
        layout.addWidget(self.max_edge_slider)

        self.run_button = QPushButton("Create watercolor")
        self.run_button.clicked.connect(self.process_watercolor)
        layout.addWidget(self.run_button)

        self.save_button = QPushButton("Save watercolor…")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_image)
        layout.addWidget(self.save_button)

        layout.addStretch(1)
        return container

    def _create_preview_panel(self) -> QWidget:
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)

        self.original_preview = ImagePreview()
        self.original_preview.setText("Open a photo to begin")
        grid.addWidget(QLabel("Original"), 0, 0)
        grid.addWidget(QLabel("Watercolor"), 0, 1)
        grid.addWidget(self.original_preview, 1, 0)

        self.stylized_preview = ImagePreview()
        self.stylized_preview.setText("Watercolor preview will appear here")
        grid.addWidget(self.stylized_preview, 1, 1)

        return container

    def _create_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        open_action = QAction("Open…", self)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        save_action = QAction("Save watercolor…", self)
        save_action.triggered.connect(self.save_image)
        save_action.setEnabled(False)
        file_menu.addAction(save_action)
        self._save_action = save_action

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    # Image handling ------------------------------------------------------
    def open_image(self) -> None:
        dialog = QFileDialog(self, "Select a photo")
        dialog.setNameFilters([
            "Image files (*.png *.jpg *.jpeg *.bmp)",
            "All files (*)",
        ])
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        if dialog.exec() == QFileDialog.DialogCode.Accepted:
            path = dialog.selectedFiles()[0]
            self._load_image(path)

    def _load_image(self, path: str) -> None:
        try:
            with Image.open(path) as img:
                pil_image = img.convert("RGB")
        except Exception:  # pragma: no cover - user error path
            QMessageBox.warning(self, "Waterloom", "Unable to open that file.")
            return

        rgb = np.array(pil_image)

        self._current_image = rgb
        self._stylized_image = None
        self.original_preview.set_image(rgb)
        self.stylized_preview.setText("Watercolor preview will appear here")
        self.save_button.setEnabled(False)
        self._save_action.setEnabled(False)

    def _gather_settings(self) -> WatercolorSettings:
        return WatercolorSettings(
            smoothness=self.smoothness_slider.value(),
            fidelity=self.fidelity_slider.value(),
            edge_strength=self.edge_strength_slider.value(),
            edge_blur=self.edge_blur_slider.value(),
            texture_intensity=self.texture_slider.value(),
            vibrance=self.vibrance_slider.value(),
            brightness=self.brightness_slider.value(),
            max_edge=self.max_edge_slider.value(),
        )

    def process_watercolor(self) -> None:
        if self._current_image is None:
            QMessageBox.information(self, "Waterloom", "Please open a photo first.")
            return

        settings = self._gather_settings()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = stylize_image(
                self._current_image,
                settings.smoothness,
                settings.fidelity,
                settings.edge_strength,
                settings.edge_blur,
                settings.texture_intensity,
                settings.vibrance,
                settings.brightness,
                settings.max_edge,
            )
        except Exception as exc:  # pragma: no cover - user error path
            QMessageBox.critical(self, "Waterloom", f"Failed to create watercolor: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._stylized_image = result
        self.stylized_preview.set_image(result)
        self.save_button.setEnabled(True)
        self._save_action.setEnabled(True)

    def save_image(self) -> None:
        if self._stylized_image is None:
            QMessageBox.information(self, "Waterloom", "Create a watercolor first.")
            return

        dialog = QFileDialog(self, "Save watercolor")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setNameFilters(["PNG image (*.png)"])
        dialog.setDefaultSuffix("png")
        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return

        path = dialog.selectedFiles()[0]
        try:
            to_pil_image(self._stylized_image).save(path, format="PNG")
        except Exception as exc:  # pragma: no cover - user error path
            QMessageBox.critical(self, "Waterloom", f"Unable to save image: {exc}")


def run() -> None:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)

    if icon_path := _app_icon_path():
        app.setWindowIcon(QIcon(icon_path))

    window = WaterloomWindow()
    window.show()
    app.exec()


def _app_icon_path() -> Optional[str]:
    candidate = Path(__file__).with_name("waterloom.png")
    if candidate.exists():
        return str(candidate)
    return None


if __name__ == "__main__":
    run()
