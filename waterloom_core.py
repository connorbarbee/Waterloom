"""Core watercolor rendering utilities shared by the Waterloom apps."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
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

    noise = np.random.normal(0.5, 0.18, size=(height, width)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=8, sigmaY=8)
    noise = np.clip(noise, 0.0, 1.0)
    texture = (noise * 255).astype(np.uint8)
    return texture[:, :, None]


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
    """Convert an RGB image array to a watercolor-styled array."""

    prepared = _normalize_image(image, max_dimension)
    bgr = cv2.cvtColor(prepared, cv2.COLOR_RGB2BGR)

    smoothed = cv2.edgePreservingFilter(bgr, flags=1, sigma_s=60, sigma_r=0.45)
    stylized = cv2.stylization(
        smoothed,
        sigma_s=smoothness,
        sigma_r=fidelity,
    )

    detail = cv2.detailEnhance(smoothed, sigma_s=10, sigma_r=0.15)
    stylized = cv2.addWeighted(stylized, 0.85, detail, 0.15, 0)

    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
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
    edges = 255 - edges

    edge_alpha = np.clip(edge_strength / 255.0, 0.0, 1.0)
    stylized = cv2.addWeighted(stylized, 1.0, edges, edge_alpha, 0)

    rgb = cv2.cvtColor(stylized, cv2.COLOR_BGR2RGB)
    rgb = _apply_vibrance(rgb, vibrance)
    rgb = _adjust_brightness(rgb, brightness)

    texture = _paper_texture(rgb.shape[1], rgb.shape[0], texture_intensity)
    if texture_intensity > 0:
        texture_f = texture.astype(np.float32) / 255.0
        rgb = np.clip(
            rgb.astype(np.float32) * (1.0 - 0.25 * texture_intensity) + texture_f * 255,
            0,
            255,
        )

    return rgb.astype(np.uint8)


def to_pil_image(image_array: np.ndarray) -> Image.Image:
    """Convert a numpy RGB array to a PIL Image."""

    return Image.fromarray(image_array.astype(np.uint8), mode="RGB")


def load_image(path: str) -> np.ndarray:
    """Load an image file path into an RGB numpy array."""

    pil_image = Image.open(path).convert("RGB")
    return np.array(pil_image)


def save_image(path: str, image_array: np.ndarray) -> None:
    """Persist an RGB numpy array to disk as a PNG image."""

    image = to_pil_image(image_array)
    image.save(path, format="PNG")
