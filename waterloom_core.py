"""Core watercolor rendering utilities shared by the Waterloom apps."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

_RNG = np.random.default_rng()


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class WatercolorSettings:
    """Tunable parameters for the watercolor effect."""

    smoothness: int = 65
    fidelity: float = 0.45
    edge_strength: int = 80
    edge_blur: int = 3
    texture_intensity: float = 0.35
    vibrance: float = 0.15
    brightness: float = 0.05
    max_edge: int = 1920

    def normalized(self) -> "WatercolorSettings":
        """Return a copy with all parameters clamped to safe ranges."""

        edge_blur = max(1, int(round(self.edge_blur)))
        if edge_blur % 2 == 0:
            edge_blur += 1

        return WatercolorSettings(
            smoothness=int(round(_clamp(self.smoothness, 10, 200))),
            fidelity=float(_clamp(self.fidelity, 0.05, 1.0)),
            edge_strength=int(round(_clamp(self.edge_strength, 0, 255))),
            edge_blur=edge_blur,
            texture_intensity=float(_clamp(self.texture_intensity, 0.0, 1.0)),
            vibrance=float(_clamp(self.vibrance, 0.0, 1.0)),
            brightness=float(_clamp(self.brightness, -1.0, 1.0)),
            max_edge=int(round(_clamp(self.max_edge, 64, 8192))),
        )


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Return a contiguous RGB array from diverse numpy inputs."""

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        channels = image.shape[2]
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif channels != 3:
            raise ValueError("Unsupported channel configuration")
    else:
        raise ValueError("Expected a 2D or 3D image array")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image)


def _normalize_image(image: np.ndarray, max_dimension: int) -> np.ndarray:
    """Resize overly large images to keep processing quick offline."""

    height, width = image.shape[:2]
    largest_side = max(height, width)
    if largest_side <= max_dimension:
        return image

    scale = max_dimension / float(largest_side)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _apply_vibrance(image: np.ndarray, vibrance: float) -> np.ndarray:
    """Boost saturation with a bias toward muted colors for a watercolor glow."""

    if vibrance <= 0:
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    boost = (1.0 - saturation) * vibrance
    hsv[:, :, 1] = np.clip((saturation + boost) * 255.0, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _adjust_brightness(image: np.ndarray, brightness: float) -> np.ndarray:
    """Shift brightness in LAB space to preserve color balance."""

    if abs(brightness) < 1e-4:
        return image

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + brightness * 255.0, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def _paper_texture(width: int, height: int, intensity: float) -> np.ndarray:
    """Generate a watercolor paper texture as a grayscale overlay."""

    if intensity <= 0:
        return np.zeros((height, width, 1), dtype=np.uint8)

    base = _RNG.normal(0.5, 0.18, size=(height, width)).astype(np.float32)
    soft = cv2.GaussianBlur(base, (0, 0), sigmaX=6, sigmaY=6)
    fibers = cv2.GaussianBlur(base, (0, 0), sigmaX=1.2, sigmaY=1.2)
    combined = 0.6 * soft + 0.4 * fibers
    contrast = 0.6 + 0.8 * intensity
    texture = np.clip(0.5 + (combined - 0.5) * contrast, 0.0, 1.0)
    return (texture * 255).astype(np.uint8)[:, :, None]


def _apply_texture(rgb: np.ndarray, intensity: float) -> np.ndarray:
    """Blend a generated paper texture into the RGB image."""

    if intensity <= 0:
        return rgb

    texture = _paper_texture(rgb.shape[1], rgb.shape[0], intensity)
    texture_rgb = np.repeat(texture, 3, axis=2)

    rgb_f = rgb.astype(np.float32)
    texture_f = texture_rgb.astype(np.float32)

    texture_weight = 0.18 + 0.32 * intensity
    blended = rgb_f * (1.0 - texture_weight) + texture_f * texture_weight

    # Add subtle contrast variation to mimic watercolor pooling.
    grain = (texture_f / 255.0 - 0.5) * 60.0 * intensity
    blended = np.clip(blended + grain, 0.0, 255.0)

    return blended.astype(np.uint8)


def stylize_image(
    image: np.ndarray,
    smoothness: int,
    fidelity: float,
    edge_strength: int,
    edge_blur: int,
    texture_intensity: float,
    vibrance: float,
    brightness: float,
    max_dimension: int,
) -> np.ndarray:
    """Backward compatible wrapper for the stylizer."""

    settings = WatercolorSettings(
        smoothness=smoothness,
        fidelity=fidelity,
        edge_strength=edge_strength,
        edge_blur=edge_blur,
        texture_intensity=texture_intensity,
        vibrance=vibrance,
        brightness=brightness,
        max_edge=max_dimension,
    )
    return stylize(image, settings)


def stylize(image: np.ndarray, settings: WatercolorSettings) -> np.ndarray:
    """Convert an RGB image array to a watercolor-styled array."""

    normalized = settings.normalized()
    prepared = _normalize_image(_ensure_rgb(image), normalized.max_edge)
    bgr = cv2.cvtColor(prepared, cv2.COLOR_RGB2BGR)

    smoothing_sigma = int(round(_clamp(normalized.smoothness, 10, 200)))
    smoothing = cv2.edgePreservingFilter(
        bgr,
        flags=1,
        sigma_s=_clamp(smoothing_sigma * 0.9, 10, 200),
        sigma_r=_clamp(normalized.fidelity * 0.6 + 0.2, 0.1, 1.0),
    )
    stylized = cv2.stylization(
        smoothing,
        sigma_s=smoothing_sigma,
        sigma_r=_clamp(normalized.fidelity, 0.05, 1.0),
    )

    detail = cv2.detailEnhance(smoothing, sigma_s=12, sigma_r=0.18)
    stylized = cv2.addWeighted(stylized, 0.82, detail, 0.18, 0)

    gray = cv2.cvtColor(smoothing, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, normalized.edge_blur)
    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2,
    )
    edges = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2BGR)

    edge_alpha = np.clip(normalized.edge_strength / 255.0, 0.0, 1.0)
    stylized = cv2.addWeighted(stylized, 1.0, edges, edge_alpha, 0)

    rgb = cv2.cvtColor(stylized, cv2.COLOR_BGR2RGB)
    rgb = _apply_vibrance(rgb, normalized.vibrance)
    rgb = _adjust_brightness(rgb, normalized.brightness)
    rgb = _apply_texture(rgb, normalized.texture_intensity)

    return rgb.astype(np.uint8)


def to_pil_image(image_array: np.ndarray) -> Image.Image:
    """Convert a numpy RGB array to a PIL Image."""

    return Image.fromarray(image_array.astype(np.uint8), mode="RGB")


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a contiguous RGB numpy array."""

    corrected = ImageOps.exif_transpose(image).convert("RGB")
    array = np.array(corrected)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an RGB image")
    return np.ascontiguousarray(array)


def load_image(path: str) -> np.ndarray:
    """Load an image file path into an RGB numpy array."""

    with Image.open(path) as pil_image:
        return pil_to_rgb_array(pil_image)


def save_image(path: str, image_array: np.ndarray) -> None:
    """Persist an RGB numpy array to disk as a PNG image."""

    destination = Path(path)
    if destination.parent and not destination.parent.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)

    image = to_pil_image(image_array)
    image.save(destination, format="PNG")
