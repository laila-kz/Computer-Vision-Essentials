"""Image enhancement, dynamic range compression, and histogram equalization techniques.

Provides foundational intensity transformations:
- Linear Contrast Stretching (Min-Max normalization)
- Non-linear Gamma Correction (Power-law transformation: $s = c \\cdot r^{\\gamma}$)
- Global Histogram Equalization (Uniform CDF transformation)
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Logarithmic Dynamic Range Compression ($s = c \\cdot \\ln(1 + r)$)
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from ..utils.conversions import ensure_uint8, to_gray


def contrast_stretch(image: np.ndarray, low_percentile: float = 0.0, high_percentile: float = 100.0) -> np.ndarray:
    """Linearly stretch intensity values to occupy the full dynamic range [0, 255].

    Optionally clips intensity percentiles to resist outliers and specular highlights.

    Mathematical Formulation:
    $$I_{out}(x, y) = \\mathrm{clip}\\left( \\frac{I(x, y) - q_{low}}{q_{high} - q_{low}} \\times 255, 0, 255 \\right)$$

    Args:
        image: Input grayscale or multi-channel image array.
        low_percentile: Lower percentile threshold for minimum intensity (0.0 to 10.0).
        high_percentile: Upper percentile threshold for maximum intensity (90.0 to 100.0).

    Returns:
        np.ndarray: Contrast-stretched image in uint8.
    """
    gray = to_gray(image)
    if low_percentile <= 0.0 and high_percentile >= 100.0:
        return cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    p_low = np.percentile(gray, low_percentile)
    p_high = np.percentile(gray, high_percentile)
    if p_high <= p_low:
        return gray

    stretched = (gray.astype(np.float32) - p_low) / (p_high - p_low) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def gamma_correction(image: np.ndarray, gamma: float = 1.0, c: float = 1.0) -> np.ndarray:
    """Apply power-law (Gamma) intensity transformation.

    Transforms image intensities non-linearly:
    $$s = c \\cdot r^\\gamma$$
    where $r \\in [0, 1]$ is the normalized input intensity.
    - $\\gamma < 1.0$: Expands dark regions (brightens underexposed images).
    - $\\gamma > 1.0$: Compresses highlights and expands darks (darkens overexposed images).

    Args:
        image: Input image array (Grayscale or Color).
        gamma: Exponent parameter $\\gamma > 0$.
        c: Scaling constant (default 1.0).

    Returns:
        np.ndarray: Gamma-corrected image array.

    Raises:
        ValueError: If gamma <= 0.
    """
    if gamma <= 0:
        raise ValueError(f"Gamma must be strictly positive (>0), got {gamma}")

    lookup_table = np.array([
        np.clip(c * ((i / 255.0) ** gamma) * 255.0, 0, 255)
        for i in range(256)
    ], dtype=np.uint8)

    return cv2.LUT(ensure_uint8(image), lookup_table)


def log_transform(image: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Apply logarithmic dynamic range compression.

    Formula:
    $$s = c \\cdot \\frac{255}{\\ln(1 + 255)} \\cdot \\ln(1 + r)$$

    Maps a narrow range of low-intensity input values into a wider output range,
    commonly used for Fourier spectrum visualization.

    Args:
        image: Input grayscale or color image.
        c: Scaling factor (default 1.0).

    Returns:
        np.ndarray: Log-transformed image in uint8.
    """
    gray = to_gray(image).astype(np.float32)
    # Scale such that log(1 + 255) maps to 255
    scale = (255.0 / np.log(1.0 + 255.0)) * c
    log_img = scale * np.log(1.0 + gray)
    return np.clip(log_img, 0, 255).astype(np.uint8)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """Perform global histogram equalization.

    Flattens the probability density function (PDF) of pixel intensities,
    producing an approximately uniform cumulative distribution function (CDF).
    Maximizes global image entropy and contrast.

    Args:
        image: Grayscale or 3-channel BGR image. If 3-channel, equalization is
            applied to the Luminance (Y or L) channel in YCrCb or LAB to preserve chromaticity.

    Returns:
        np.ndarray: Histogram equalized image.
    """
    if image.ndim == 2:
        return cv2.equalizeHist(ensure_uint8(image))

    # For color images: equalize Luminance channel in YCrCb to prevent chromatic distortion
    ycrcb = cv2.cvtColor(ensure_uint8(image), cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def clahe_enhancement(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Overcomes over-amplification of noise in homogeneous regions by:
    1. Dividing the image into contextual rectangular tiles (e.g., 8x8).
    2. Computing localized histogram equalization per tile.
    3. Clipping histograms above `clip_limit` and redistributing excess uniformly.
    4. Eliminating boundary artifacts via bilinear interpolation.

    Args:
        image: Input grayscale or 3-channel BGR image.
        clip_limit: Threshold for contrast limiting (typically 2.0 to 4.0).
        tile_grid_size: Number of tiles in (row, column) grid, e.g. (8, 8).

    Returns:
        np.ndarray: CLAHE enhanced image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    if image.ndim == 2:
        return clahe.apply(ensure_uint8(image))

    # Process in LAB space to enhance Lightness (L*) channel independently
    lab = cv2.cvtColor(ensure_uint8(image), cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def compute_histogram_and_cdf(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the 256-bin normalized histogram and cumulative distribution function (CDF).

    Args:
        image: Grayscale image array (H, W).

    Returns:
        tuple containing:
            - np.ndarray: Normalized histogram (PDF) summing to 1.0 of shape (256,).
            - np.ndarray: Cumulative distribution function (CDF) spanning [0, 1.0] of shape (256,).
    """
    gray = to_gray(image)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    pdf = hist / (gray.size + 1e-8)
    cdf = np.cumsum(pdf)
    return pdf, cdf


def build_enhancement_summary(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Execute all enhancement methods to generate a side-by-side comparison dictionary.

    Args:
        image: Input image array.

    Returns:
        dict[str, np.ndarray]: Mapping of transformation name to enhanced image.
    """
    return {
        "original": ensure_uint8(to_gray(image)),
        "contrast_stretch": contrast_stretch(image),
        "hist_equalized": equalize_histogram(to_gray(image)),
        "clahe": clahe_enhancement(to_gray(image), clip_limit=2.5),
        "gamma_0_6": gamma_correction(image, gamma=0.6),
        "gamma_1_8": gamma_correction(image, gamma=1.8),
        "log_transform": log_transform(image),
    }
