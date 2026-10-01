"""Integrated segmentation pipeline combining thresholding, morphology, and region labeling."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .connected_components import connected_components_summary
from .morphology import morphology_cleanup
from .thresholding import otsu_threshold


def segmentation_pipeline(image: np.ndarray) -> Dict[str, Any]:
    """Execute end-to-end Otsu thresholding, morphological refinement, and CCL analysis.

    Standard classical pipeline:
    1. Global Otsu thresholding to separate foreground from background.
    2. Morphological opening (noise reduction) and closing (hole filling).
    3. 8-connectivity connected components labeling with geometric statistics.

    Args:
        image: Input grayscale or BGR image.

    Returns:
        dict[str, Any]: Consolidated dictionary containing:
            - 'threshold': Otsu cut-off scalar value.
            - 'binary': Raw binarized mask.
            - 'cleaned': Morphologically filtered mask.
            - 'num_labels': Number of distinct components detected.
            - 'labels': 2D integer component map.
            - 'stats': Bounding box and area metrics matrix.
            - 'centroids': Floating-point coordinate array of centers.
            - 'areas': List of component pixel sizes.
            - 'foreground_ratio': Relative foreground area ratio.
    """
    threshold_value, binary = otsu_threshold(image)
    cleaned = morphology_cleanup(binary)
    summary = connected_components_summary(cleaned)
    summary.update({
        "threshold": threshold_value,
        "binary": binary,
        "cleaned": cleaned,
    })
    return summary
