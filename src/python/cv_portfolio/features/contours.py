"""Contour extraction, shape analysis, geometric moments, and boundary descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ..utils.conversions import ensure_uint8


@dataclass(frozen=True)
class ContourProperties:
    """Geometric and morphological properties of a closed contour.

    Attributes:
        area: Enclosed spatial area in pixels ($M_{00}$).
        perimeter: Arc length of closed curve.
        circularity: Shape compactness $4\\pi \\cdot \\mathrm{area} / \\mathrm{perimeter}^2 \\in [0, 1]$.
        bounding_box: (x, y, w, h) bounding rectangle.
        aspect_ratio: Ratio of width to height ($w / h$).
        extent: Ratio of contour area to bounding box area.
        solidity: Ratio of contour area to convex hull area.
        centroid: (cx, cy) center of gravity coordinates.
    """

    area: float
    perimeter: float
    circularity: float
    bounding_box: Tuple[int, int, int, int]
    aspect_ratio: float
    extent: float
    solidity: float
    centroid: Tuple[float, float]


def find_external_contours(binary_mask: np.ndarray, min_area: float = 20.0) -> List[np.ndarray]:
    """Extract external boundary contours from a binary image.

    Args:
        binary_mask: Binary mask array (0 or 255).
        min_area: Minimum contour area threshold to reject noise artifacts.

    Returns:
        list[np.ndarray]: List of filtered contour point arrays.
    """
    contours, _ = cv2.findContours(
        ensure_uint8(binary_mask),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def contour_properties(contour: np.ndarray) -> ContourProperties:
    """Compute comprehensive geometric shape descriptors and central moments.

    Computes spatial moments:
    $$m_{pq} = \\sum_x \\sum_y x^p y^q I(x, y)$$
    Centroid: $(\\bar{x}, \\bar{y}) = (m_{10}/m_{00}, m_{01}/m_{00})$.

    Args:
        contour: Single 2D contour point array of shape (N, 1, 2).

    Returns:
        ContourProperties: Structured dataclass of geometric metrics.
    """
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, closed=True))
    circularity = (4.0 * np.pi * area) / (perimeter * perimeter + 1e-6) if perimeter > 0 else 0.0

    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = float(w * h) + 1e-6
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0
    extent = area / bbox_area

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull)) + 1e-6
    solidity = area / hull_area

    moments = cv2.moments(contour)
    m00 = moments["m00"]
    if abs(m00) > 1e-5:
        cx = float(moments["m10"] / m00)
        cy = float(moments["m01"] / m00)
    else:
        cx, cy = float(x + w / 2), float(y + h / 2)

    return ContourProperties(
        area=area,
        perimeter=perimeter,
        circularity=circularity,
        bounding_box=(int(x), int(y), int(w), int(h)),
        aspect_ratio=aspect_ratio,
        extent=extent,
        solidity=solidity,
        centroid=(cx, cy),
    )


def draw_annotated_contours(
    image: np.ndarray,
    contours: List[np.ndarray],
    draw_boxes: bool = True,
    draw_centroids: bool = True,
) -> np.ndarray:
    """Draw contour outlines, bounding boxes, and centroids on an image canvas.

    Args:
        image: Background canvas image.
        contours: List of contour arrays.
        draw_boxes: If True, draws green bounding rectangles.
        draw_centroids: If True, draws red centroid dots.

    Returns:
        np.ndarray: Annotated BGR visualization image.
    """
    annotated = cv2.cvtColor(ensure_uint8(image), cv2.COLOR_GRAY2BGR) if image.ndim == 2 else ensure_uint8(image).copy()

    for idx, c in enumerate(contours):
        # Draw contour boundary in cyan
        cv2.drawContours(annotated, [c], -1, (255, 255, 0), 2)
        props = contour_properties(c)

        if draw_boxes:
            x, y, w, h = props.bounding_box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"#{idx+1}",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        if draw_centroids:
            cx, cy = int(props.centroid[0]), int(props.centroid[1])
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

    return annotated
