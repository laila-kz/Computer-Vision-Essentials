"""Connected Component Labeling (CCL) and regional morphological measurements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ..utils.conversions import ensure_uint8
from ..utils.io import count_nonzero_ratio


@dataclass(frozen=True)
class ComponentInfo:
    """Quantitative spatial metrics for a single connected component."""

    label_id: int
    area: int
    left: int
    top: int
    width: int
    height: int
    centroid_x: float
    centroid_y: float
    aspect_ratio: float
    extent: float


def connected_components_summary(binary_mask: np.ndarray) -> Dict[str, Any]:
    """Analyze all connected components within a binary mask.

    Computes bounding boxes, pixel areas, centroids, and foreground occupancy.

    Args:
        binary_mask: Binary mask in uint8 (0 or 255).

    Returns:
        dict[str, Any]: Detailed dictionary of component statistics and label matrices.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        ensure_uint8(binary_mask), connectivity=8
    )

    components: List[ComponentInfo] = []
    # Skip label 0 (background)
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        left = int(stats[i, cv2.CC_STAT_LEFT])
        top = int(stats[i, cv2.CC_STAT_TOP])
        width = int(stats[i, cv2.CC_STAT_WIDTH])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx = float(centroids[i, 0])
        cy = float(centroids[i, 1])
        bbox_area = float(width * height) + 1e-6
        aspect_ratio = float(width) / (float(height) + 1e-6)
        extent = float(area) / bbox_area

        components.append(
            ComponentInfo(
                label_id=i,
                area=area,
                left=left,
                top=top,
                width=width,
                height=height,
                centroid_x=cx,
                centroid_y=cy,
                aspect_ratio=aspect_ratio,
                extent=extent,
            )
        )

    areas = stats[:, cv2.CC_STAT_AREA].tolist()
    return {
        "num_labels": num_labels,
        "labels": labels,
        "stats": stats,
        "centroids": centroids,
        "areas": areas,
        "components": components,
        "foreground_ratio": count_nonzero_ratio(binary_mask),
    }


def filter_components_by_area(
    binary_mask: np.ndarray,
    min_area: int = 50,
    max_area: int = 1_000_000,
) -> Tuple[np.ndarray, int]:
    """Filter out connected components outside the [min_area, max_area] range.

    Args:
        binary_mask: Input binary mask.
        min_area: Minimum pixel area to retain.
        max_area: Maximum pixel area to retain.

    Returns:
        tuple containing:
            - np.ndarray: Cleaned binary mask containing only retained components.
            - int: Number of components that passed the area threshold.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ensure_uint8(binary_mask), connectivity=8
    )
    output_mask = np.zeros_like(binary_mask, dtype=np.uint8)
    kept_count = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            output_mask[labels == i] = 255
            kept_count += 1

    return output_mask, kept_count
