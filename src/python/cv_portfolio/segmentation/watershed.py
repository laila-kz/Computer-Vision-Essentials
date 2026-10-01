"""Marker-controlled Watershed segmentation and Euclidean distance transforms.

Implements the classical topological watershed algorithm to separate touching or overlapping objects:
1. Binary binarization and morphological cleaning.
2. Exact Euclidean Distance Transform ($D(p) = \\min_{q \\in \\mathrm{Background}} \\|p - q\\|_2$).
3. Peak finding to extract unambiguous object centers as seeds/markers.
4. Morphological dilation to define unambiguous background markers.
5. Flooding the gradient / color landscape via `cv2.watershed` to form dividing ridge lines.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from ..utils.conversions import ensure_uint8, to_gray
from .morphology import morphology_open


def distance_transform_watershed(
    image: np.ndarray,
    distance_threshold_ratio: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Execute marker-controlled watershed segmentation on touching objects.

    Args:
        image: Color or grayscale image with foreground objects.
        distance_threshold_ratio: Multiplier $\\beta \\in (0, 1)$ applied to max distance
            transform value to establish sure foreground seeds.

    Returns:
        dict[str, np.ndarray]: Dictionary containing intermediate and final watershed results:
            - 'binary': Initial Otsu threshold mask.
            - 'opening': Mask after noise removal.
            - 'sure_bg': Dilated background boundary.
            - 'dist_transform': Normalized floating-point distance transform map.
            - 'sure_fg': Thresholded peaks acting as marker seeds.
            - 'unknown': Ambiguous boundary transition zone.
            - 'markers': Labeled integer marker map before and after watershed.
            - 'segmented': Color visualization with red boundaries highlighting object separations.
    """
    gray = to_gray(image)
    if image.ndim == 2:
        color_img = cv2.cvtColor(ensure_uint8(image), cv2.COLOR_GRAY2BGR)
    else:
        color_img = ensure_uint8(image).copy()

    # 1. Otsu thresholding with inversion if background is brighter
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Noise removal via opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Sure background area via dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 4. Finding sure foreground area using Euclidean distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    max_dist = dist_transform.max()
    _, sure_fg = cv2.threshold(dist_transform, distance_threshold_ratio * max_dist, 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 5. Finding unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    # Add 1 to all labels so that background is not 0, but 1
    markers = markers + 1
    # Mark the unknown region as 0
    markers[unknown == 255] = 0

    # 7. Apply watershed
    markers_applied = cv2.watershed(color_img, markers.copy())

    # 8. Boundary visualization
    segmented = color_img.copy()
    segmented[markers_applied == -1] = [0, 0, 255]  # Mark boundaries in red (BGR)

    # Normalize distance transform for visualization
    dist_norm = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return {
        "binary": thresh,
        "opening": opening,
        "sure_bg": sure_bg,
        "dist_transform": dist_norm,
        "sure_fg": sure_fg,
        "unknown": unknown,
        "markers": markers_applied,
        "segmented": segmented,
    }
