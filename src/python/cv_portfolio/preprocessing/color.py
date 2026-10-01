"""Color channel operations, color balancing, and chromaticity analysis."""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from ..utils.conversions import ensure_uint8


def split_channels(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Split a 3-channel BGR image into Blue, Green, and Red grayscale representations.

    Args:
        image: 3-channel BGR image array (H, W, 3).

    Returns:
        dict[str, np.ndarray]: Dictionary with keys 'blue', 'green', 'red'.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("split_channels requires a 3-channel color image.")
    b, g, r = cv2.split(image)
    return {"blue": b, "green": g, "red": r}


def gray_world_white_balance(image: np.ndarray) -> np.ndarray:
    """Perform Gray-World chromatic adaptation / white balancing.

    Assumes the average scene reflectance is achromatic (neutral gray).
    Scales each channel such that channel averages equal the global mean:
    $$I_c'(x,y) = I_c(x,y) \\cdot \\frac{\\mu_{gray}}{\\mu_c}$$

    Args:
        image: Input 3-channel BGR image.

    Returns:
        np.ndarray: Color-balanced BGR image in uint8.
    """
    img_float = image.astype(np.float32)
    b_avg = np.mean(img_float[:, :, 0]) + 1e-6
    g_avg = np.mean(img_float[:, :, 1]) + 1e-6
    r_avg = np.mean(img_float[:, :, 2]) + 1e-6
    gray_avg = (b_avg + g_avg + r_avg) / 3.0

    img_float[:, :, 0] = np.clip(img_float[:, :, 0] * (gray_avg / b_avg), 0, 255)
    img_float[:, :, 1] = np.clip(img_float[:, :, 1] * (gray_avg / g_avg), 0, 255)
    img_float[:, :, 2] = np.clip(img_float[:, :, 2] * (gray_avg / r_avg), 0, 255)

    return img_float.astype(np.uint8)


def color_histogram(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 256-bin histograms for Blue, Green, and Red channels independently.

    Args:
        image: 3-channel BGR image.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (hist_b, hist_g, hist_r) normalized histograms.
    """
    b_hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    g_hist = cv2.calcHist([image], [1], None, [256], [0, 256]).ravel()
    r_hist = cv2.calcHist([image], [2], None, [256], [0, 256]).ravel()
    total_pixels = float(image.shape[0] * image.shape[1])
    return b_hist / total_pixels, g_hist / total_pixels, r_hist / total_pixels
