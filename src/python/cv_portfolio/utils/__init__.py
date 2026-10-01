"""Utility functions for I/O, format conversion, normalization, and synthetic test data."""

from __future__ import annotations

import cv2
import numpy as np

from .conversions import (
    ImageStats,
    ensure_uint8,
    image_stats,
    normalize_uint8,
    to_gray,
    to_hsv,
    to_lab,
    to_rgb,
)
from .io import generate_synthetic_image, read_image, save_image


def count_nonzero_ratio(mask: np.ndarray) -> float:
    """Calculate the fraction of non-zero (foreground) pixels in a binary mask.

    Args:
        mask: 2D or 3D binary/grayscale array.

    Returns:
        float: Foreground ratio between 0.0 and 1.0.
    """
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def label_components(binary_mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Perform 8-connectivity Connected Component Labeling (CCL).

    Wrapper around OpenCV's connectedComponentsWithStats returning 4 standard elements.

    Args:
        binary_mask: Binary image (0 or 255) of shape (H, W).

    Returns:
        tuple containing:
            - int: Total number of connected components including background (label 0).
            - np.ndarray: Labeled map where pixel values correspond to label IDs [0, num_labels-1].
            - np.ndarray: Statistics array of shape (num_labels, 5) [CC_STAT_LEFT, TOP, WIDTH, HEIGHT, AREA].
            - np.ndarray: Floating-point centroid coordinates of shape (num_labels, 2) [x, y].
    """
    return cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)


__all__ = [
    "ImageStats",
    "image_stats",
    "to_gray",
    "to_rgb",
    "to_hsv",
    "to_lab",
    "ensure_uint8",
    "normalize_uint8",
    "read_image",
    "save_image",
    "generate_synthetic_image",
    "count_nonzero_ratio",
    "label_components",
]
