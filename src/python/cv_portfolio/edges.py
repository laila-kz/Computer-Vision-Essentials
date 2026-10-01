"""Edge detection, differential operators, and gradient analysis."""

from __future__ import annotations

from .filtering.edges import (
    canny_edges,
    compare_edge_detectors,
    gradient_vector_field,
    laplacian_edges,
    log_edges,
    prewitt_edges,
    scharr_edges,
    sobel_edges,
)
from .filtering.spatial import (
    bilateral_denoise,
    box_blur,
    custom_convolution,
    gaussian_denoise,
    median_filter,
    unsharp_mask,
)

__all__ = [
    "gaussian_denoise",
    "box_blur",
    "median_filter",
    "bilateral_denoise",
    "unsharp_mask",
    "custom_convolution",
    "gradient_vector_field",
    "sobel_edges",
    "prewitt_edges",
    "scharr_edges",
    "laplacian_edges",
    "log_edges",
    "canny_edges",
    "compare_edge_detectors",
]
