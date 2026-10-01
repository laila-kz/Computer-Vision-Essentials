"""Object counting, blob analysis, and shape-filtered geometric localization."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from ..features.contours import contour_properties
from ..utils.conversions import ensure_uint8


def count_contours(
    binary_mask: np.ndarray,
    min_area: int = 100,
    max_area: int = 1_000_000,
) -> Tuple[int, List[np.ndarray]]:
    """Count and filter foreground object contours based on minimum area threshold.

    Args:
        binary_mask: Cleaned binary mask in uint8 (0 and 255).
        min_area: Minimum area cutoff in pixels.
        max_area: Maximum area cutoff in pixels.

    Returns:
        tuple containing:
            - int: Number of objects satisfying area criteria.
            - list[np.ndarray]: List of passing contour point arrays.
    """
    contours, _ = cv2.findContours(
        ensure_uint8(binary_mask),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    filtered = [
        c for c in contours
        if min_area <= cv2.contourArea(c) <= max_area
    ]
    return len(filtered), filtered


def count_circular_objects(
    binary_mask: np.ndarray,
    min_area: float = 100.0,
    min_circularity: float = 0.7,
) -> Tuple[int, List[np.ndarray]]:
    """Count circular/round objects (e.g., coins, pills, cells) by combining area and circularity filters.

    Circularity definition:
    $$C = \\frac{4\\pi \\cdot \\mathrm{Area}}{\\mathrm{Perimeter}^2} \\in [0, 1]$$
    A perfect circle yields $C = 1.0$.

    Args:
        binary_mask: Cleaned binary mask.
        min_area: Minimum area cutoff.
        min_circularity: Minimum shape circularity score ($[0, 1]$).

    Returns:
        tuple containing:
            - int: Number of circular objects detected.
            - list[np.ndarray]: Filtered contour arrays.
    """
    total_count, all_contours = count_contours(binary_mask, min_area=int(min_area))
    circular_contours: List[np.ndarray] = []

    for c in all_contours:
        props = contour_properties(c)
        if props.circularity >= min_circularity:
            circular_contours.append(c)

    return len(circular_contours), circular_contours
