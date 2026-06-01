from __future__ import annotations

import cv2
import numpy as np

from .utils import count_nonzero_ratio, ensure_uint8, label_components, to_gray


def otsu_threshold(image: np.ndarray) -> tuple[float, np.ndarray]:
    gray = to_gray(image)
    threshold_value, binary = cv2.threshold(ensure_uint8(gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold_value, binary


def morphology_cleanup(binary_mask: np.ndarray, open_size: int = 3, close_size: int = 5) -> np.ndarray:
    open_kernel = np.ones((open_size, open_size), dtype=np.uint8)
    close_kernel = np.ones((close_size, close_size), dtype=np.uint8)
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)
    return closed


def connected_components_summary(binary_mask: np.ndarray) -> dict[str, object]:
    num_labels, labels, stats, centroids = label_components(binary_mask)
    areas = stats[:, cv2.CC_STAT_AREA].tolist()
    return {
        "num_labels": num_labels,
        "labels": labels,
        "stats": stats,
        "centroids": centroids,
        "areas": areas,
        "foreground_ratio": count_nonzero_ratio(binary_mask),
    }


def segmentation_pipeline(image: np.ndarray) -> dict[str, object]:
    threshold_value, binary = otsu_threshold(image)
    cleaned = morphology_cleanup(binary)
    summary = connected_components_summary(cleaned)
    summary.update({"threshold": threshold_value, "binary": binary, "cleaned": cleaned})
    return summary
