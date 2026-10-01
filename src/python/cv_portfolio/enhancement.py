"""Intensity transformations, histogram equalization, and dynamic range enhancement."""

from __future__ import annotations

from .preprocessing.color import color_histogram, gray_world_white_balance, split_channels
from .preprocessing.enhancement import (
    build_enhancement_summary,
    clahe_enhancement,
    compute_histogram_and_cdf,
    contrast_stretch,
    equalize_histogram,
    gamma_correction,
    log_transform,
)

__all__ = [
    "contrast_stretch",
    "gamma_correction",
    "log_transform",
    "equalize_histogram",
    "clahe_enhancement",
    "compute_histogram_and_cdf",
    "build_enhancement_summary",
    "split_channels",
    "gray_world_white_balance",
    "color_histogram",
]
