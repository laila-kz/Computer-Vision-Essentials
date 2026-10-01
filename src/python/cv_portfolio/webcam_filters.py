"""Real-time webcam filter simulation for live video and interactive processing."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from .filtering.edges import canny_edges
from .filtering.spatial import bilateral_denoise, unsharp_mask
from .utils.conversions import ensure_uint8, to_gray


def grayscale_frame(frame: np.ndarray) -> np.ndarray:
    """Convert input frame to 3-channel grayscale for seamless video streaming display.

    Args:
        frame: Input BGR frame.

    Returns:
        np.ndarray: Grayscale frame encoded in BGR channels.
    """
    gray = to_gray(frame)
    return cv2.cvtColor(ensure_uint8(gray), cv2.COLOR_GRAY2BGR)


def pencil_sketch_frame(frame: np.ndarray, blur_ksize: int = 21) -> np.ndarray:
    """Simulate a hand-drawn artistic pencil sketch filter.

    Uses Dodge color blend on inverted Gaussian blurred grayscale:
    $$I_{sketch} = \\frac{I_{gray}}{255 - G_\\sigma(255 - I_{gray})} \\times 256$$

    Args:
        frame: Input BGR frame.
        blur_ksize: Gaussian blur window size (odd integer).

    Returns:
        np.ndarray: Pencil sketch effect in BGR format.
    """
    gray = to_gray(frame)
    inverted = cv2.bitwise_not(gray)
    k = max(3, blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1)
    blurred = cv2.GaussianBlur(inverted, (k, k), sigmaX=0, sigmaY=0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    return cv2.cvtColor(ensure_uint8(sketch), cv2.COLOR_GRAY2BGR)


def edge_frame(frame: np.ndarray, lower: int = 60, upper: int = 140) -> np.ndarray:
    """Extract Canny edges and return as 3-channel BGR visualization.

    Args:
        frame: Input BGR frame.
        lower: Hysteresis lower threshold.
        upper: Hysteresis upper threshold.

    Returns:
        np.ndarray: Binary edge frame in BGR format.
    """
    edges = canny_edges(frame, lower_threshold=lower, upper_threshold=upper)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def sharpen_frame(frame: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """Apply high-pass unsharp mask sharpening filter.

    Args:
        frame: Input BGR frame.
        strength: Sharpening gain multiplier.

    Returns:
        np.ndarray: Sharpened BGR frame.
    """
    return unsharp_mask(frame, kernel_size=5, sigma=1.0, strength=strength)


def cartoonize_frame(frame: np.ndarray) -> np.ndarray:
    """Simulate a cartoon / cel-shaded aesthetic.

    Combines bilateral edge-preserving smoothing with adaptive threshold edge outlines.

    Args:
        frame: Input BGR frame.

    Returns:
        np.ndarray: Cartoonized color frame.
    """
    u8 = ensure_uint8(frame)
    gray = to_gray(u8)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    color = cv2.bilateralFilter(u8, d=9, sigmaColor=250, sigmaSpace=250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


def build_webcam_filter_map(frame: np.ndarray) -> Dict[str, np.ndarray]:
    """Apply a suite of real-time filters to a single input frame.

    Args:
        frame: Input BGR frame.

    Returns:
        dict[str, np.ndarray]: Mapping of filter names to processed frames.
    """
    return {
        "original": ensure_uint8(frame),
        "grayscale": grayscale_frame(frame),
        "edge": edge_frame(frame),
        "sketch": pencil_sketch_frame(frame),
        "sharpen": sharpen_frame(frame),
        "cartoon": cartoonize_frame(frame),
    }
