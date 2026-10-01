"""Spatial filtering, linear convolution, non-linear denoising, and edge detection operators."""

from .edges import (
    canny_edges,
    compare_edge_detectors,
    gradient_vector_field,
    laplacian_edges,
    log_edges,
    prewitt_edges,
    scharr_edges,
    sobel_edges,
)
from .spatial import (
    bilateral_denoise,
    box_blur,
    custom_convolution,
    gaussian_denoise,
    median_filter,
    unsharp_mask,
)

__all__ = [
    "custom_convolution",
    "box_blur",
    "gaussian_denoise",
    "median_filter",
    "bilateral_denoise",
    "unsharp_mask",
    "gradient_vector_field",
    "sobel_edges",
    "prewitt_edges",
    "scharr_edges",
    "laplacian_edges",
    "log_edges",
    "canny_edges",
    "compare_edge_detectors",
]
