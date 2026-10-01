"""Template matching algorithms for object localization and pattern recognition."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from ..utils.conversions import ensure_uint8, to_gray


def template_match(
    image: np.ndarray,
    template: np.ndarray,
    method: int = cv2.TM_CCOEFF_NORMED,
) -> Tuple[Tuple[int, int], float, np.ndarray]:
    """Locate a template pattern within an image using correlation matching.

    Normalized Cross-Correlation (TM_CCOEFF_NORMED):
    $$R(x, y) = \\frac{\\sum_{x',y'} (T'(x',y') \\cdot I'(x+x', y+y'))}{\\sqrt{\\sum_{x',y'} T'^2(x',y') \\cdot \\sum_{x',y'} I'^2(x+x',y+y')}}$$

    Args:
        image: Source search image.
        template: Target exemplar template patch.
        method: OpenCV template matching mode (default: cv2.TM_CCOEFF_NORMED).

    Returns:
        tuple containing:
            - tuple[int, int]: (top_left_x, top_left_y) location of highest correlation match.
            - float: Peak correlation score $\\in [-1.0, 1.0]$ or $[0.0, 1.0]$.
            - np.ndarray: Full 2D correlation score response map.
    """
    gray_image = to_gray(image)
    gray_template = to_gray(template)

    result = cv2.matchTemplate(
        ensure_uint8(gray_image),
        ensure_uint8(gray_template),
        method,
    )
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        best_loc = min_loc
        best_val = float(min_val)
    else:
        best_loc = max_loc
        best_val = float(max_val)

    return (int(best_loc[0]), int(best_loc[1])), best_val, result


def multi_scale_template_match(
    image: np.ndarray,
    template: np.ndarray,
    scales: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5],
    threshold: float = 0.8,
) -> Tuple[Tuple[int, int, int, int], float, float]:
    """Multi-scale template matching across an image pyramid.

    Resizes the template across multiple scale ratios to achieve scale invariance.

    Args:
        image: Source search image.
        template: Target template patch.
        scales: List of scaling multipliers.
        threshold: Minimum correlation score.

    Returns:
        tuple containing:
            - tuple[int, int, int, int]: (x, y, w, h) bounding rectangle of optimal match.
            - float: Optimal correlation score.
            - float: Best matching scale factor.
    """
    gray_image = to_gray(image)
    gray_template = to_gray(template)

    best_score = -1.0
    best_bbox = (0, 0, 0, 0)
    best_scale = 1.0

    for s in scales:
        tw = int(gray_template.shape[1] * s)
        th = int(gray_template.shape[0] * s)
        if tw > gray_image.shape[1] or th > gray_image.shape[0] or tw < 10 or th < 10:
            continue

        resized_template = cv2.resize(gray_template, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(gray_image, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_score:
            best_score = float(max_val)
            best_bbox = (int(max_loc[0]), int(max_loc[1]), tw, th)
            best_scale = float(s)

    return best_bbox, best_score, best_scale
