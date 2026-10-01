"""Image type conversions, color space transformations, and numerical normalization utilities.

Provides safe array conversions, color model switching (BGR, RGB, Gray, HSV, LAB),
and dynamic range scaling for computer vision pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageStats:
    """Metadata and dimensional statistics for an image array.

    Attributes:
        height: Image height in pixels (rows).
        width: Image width in pixels (columns).
        channels: Number of color channels (1 for grayscale, 3 for BGR/RGB).
        dtype: Data type string representation (e.g., 'uint8', 'float32').
        min_val: Minimum pixel intensity present in the array.
        max_val: Maximum pixel intensity present in the array.
        mean_val: Mean pixel intensity.
        std_val: Standard deviation of pixel intensities.
    """

    height: int
    width: int
    channels: int
    dtype: str
    min_val: float
    max_val: float
    mean_val: float
    std_val: float


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an image to single-channel grayscale if not already grayscale.

    Computes luminance using standard ITU-R BT.601 weights:
    $Y = 0.299 R + 0.587 G + 0.114 B$ (OpenCV uses BGR order: $0.114 B + 0.587 G + 0.299 R$).

    Args:
        image: Input array of shape (H, W), (H, W, 1), or (H, W, 3).

    Returns:
        np.ndarray: Grayscale image of shape (H, W) with uint8 precision.
    """
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image.squeeze(axis=2)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image array shape for grayscale conversion: {image.shape}")


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR or single-channel image into RGB for Matplotlib visualization.

    Args:
        image: Input BGR image (H, W, 3) or Grayscale image (H, W).

    Returns:
        np.ndarray: RGB formatted image array (H, W, 3).
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    return image


def to_hsv(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to Hue-Saturation-Value (HSV) color space.

    HSV decouples chromaticity (Hue: [0, 179], Saturation: [0, 255])
    from luminance (Value: [0, 255]), making it robust for color segmentation
    under varying illumination.

    Args:
        image: Input image in BGR format (H, W, 3).

    Returns:
        np.ndarray: Converted image in HSV color space.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("HSV conversion requires a 3-channel BGR image.")
    return cv2.cvtColor(ensure_uint8(image), cv2.COLOR_BGR2HSV)


def to_lab(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to CIE L*a*b* color space.

    L* represents perceptual lightness, while a* and b* represent chromaticity
    coordinates. Useful for color-difference evaluations and illumination correction.

    Args:
        image: Input image in BGR format (H, W, 3).

    Returns:
        np.ndarray: Converted image in L*a*b* space.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("L*a*b* conversion requires a 3-channel BGR image.")
    return cv2.cvtColor(ensure_uint8(image), cv2.COLOR_BGR2LAB)


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    """Linearly map an arbitrary float or integer array into [0, 255] uint8.

    Applies Min-Max scaling:
    $I_{norm}(x,y) = \\frac{I(x,y) - \\min(I)}{\\max(I) - \\min(I)} \\times 255$

    Args:
        image: Input numpy array of any data type.

    Returns:
        np.ndarray: Scaled array with dtype np.uint8.
    """
    if image.size == 0:
        return image.astype(np.uint8)
    norm = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure the input image has dtype np.uint8, normalizing if necessary.

    Args:
        image: Input numpy array.

    Returns:
        np.ndarray: Array guaranteed to be uint8.
    """
    if image.dtype == np.uint8:
        return image
    return normalize_uint8(image)


def image_stats(image: np.ndarray) -> ImageStats:
    """Extract comprehensive dimensions and statistical distribution parameters.

    Args:
        image: Input image array.

    Returns:
        ImageStats: Dataclass containing dimension and intensity summary.
    """
    channels = 1 if image.ndim == 2 else image.shape[2]
    return ImageStats(
        height=int(image.shape[0]),
        width=int(image.shape[1]),
        channels=channels,
        dtype=str(image.dtype),
        min_val=float(np.min(image)),
        max_val=float(np.max(image)),
        mean_val=float(np.mean(image)),
        std_val=float(np.std(image)),
    )
