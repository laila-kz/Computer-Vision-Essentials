"""Gradient estimation, edge detection operators, and contour boundary extraction.

Comprehensive implementations of classical 1st and 2nd derivative edge operators:
- Sobel Gradient ($S_x, S_y$ with separable 3x3 derivative filters)
- Prewitt Operator (Uniform difference kernel)
- Scharr Operator (High rotational symmetry gradient kernel)
- Laplacian Operator (2nd derivative isotropic operator: $\\nabla^2 I$)
- Laplacian of Gaussian (LoG / Marr-Hildreth zero-crossing operator)
- Canny Edge Detector (Multi-stage optimal detector: Smoothing -> Gradient -> NMS -> Hysteresis)
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from ..utils.conversions import ensure_uint8, to_gray
from .spatial import gaussian_denoise


def gradient_vector_field(image: np.ndarray, ksize: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute directional spatial gradients, magnitude, and orientation field using Sobel.

    Computes:
    - Horizontal gradient: $G_x = \\frac{\\partial I}{\\partial x}$
    - Vertical gradient: $G_y = \\frac{\\partial I}{\\partial y}$
    - Gradient Magnitude: $M(x, y) = \\sqrt{G_x^2 + G_y^2}$
    - Gradient Orientation: $\\theta(x, y) = \\mathrm{arctan2}(G_y, G_x) \\in [-\\pi, \\pi]$

    Args:
        image: Input grayscale or color image.
        ksize: Aperture size for the Sobel operator (1, 3, 5, or 7).

    Returns:
        tuple containing:
            - np.ndarray: $G_x$ floating point gradient.
            - np.ndarray: $G_y$ floating point gradient.
            - np.ndarray: Magnitude field $M(x,y)$ as float32.
            - np.ndarray: Angle field $\\theta(x,y)$ in radians [-pi, pi] as float32.
    """
    gray = to_gray(image)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=False)
    return grad_x, grad_y, magnitude, angle


def sobel_edges(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Compute edge magnitude using 3x3 or 5x5 Sobel discrete differentiation.

    Kernel $S_x$:
    $$\\begin{bmatrix} -1 & 0 & +1 \\\\ -2 & 0 & +2 \\\\ -1 & 0 & +1 \\end{bmatrix}$$

    Args:
        image: Input image array.
        ksize: Kernel window size (default 3).

    Returns:
        np.ndarray: Normalized Sobel magnitude response in uint8.
    """
    gray = to_gray(image)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return ensure_uint8(magnitude)


def prewitt_edges(image: np.ndarray) -> np.ndarray:
    """Compute edge magnitude using Prewitt spatial differentiation kernels.

    Horizontal kernel $P_x$:
    $$\\begin{bmatrix} -1 & 0 & +1 \\\\ -1 & 0 & +1 \\\\ -1 & 0 & +1 \\end{bmatrix}$$

    Args:
        image: Input image array.

    Returns:
        np.ndarray: Normalized Prewitt edge response in uint8.
    """
    gray = to_gray(image).astype(np.float32)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    px = cv2.filter2D(gray, -1, kernel_x)
    py = cv2.filter2D(gray, -1, kernel_y)
    magnitude = np.hypot(px, py)
    return ensure_uint8(magnitude)


def scharr_edges(image: np.ndarray) -> np.ndarray:
    """Compute edge magnitude using Scharr gradient operator.

    Provides superior rotational invariance compared to standard 3x3 Sobel:
    $$\\begin{bmatrix} -3 & 0 & +3 \\\\ -10 & 0 & +10 \\\\ -3 & 0 & +3 \\end{bmatrix}$$

    Args:
        image: Input image array.

    Returns:
        np.ndarray: Scharr edge response in uint8.
    """
    gray = to_gray(image)
    grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return ensure_uint8(magnitude)


def laplacian_edges(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Compute second-order spatial derivative isotropic edge response.

    Laplace operator:
    $$\\nabla^2 I = \\frac{\\partial^2 I}{\\partial x^2} + \\frac{\\partial^2 I}{\\partial y^2}$$

    Args:
        image: Input image array.
        ksize: Aperture size (1, 3, 5).

    Returns:
        np.ndarray: Absolute Laplacian response in uint8.
    """
    gray = to_gray(image)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    return ensure_uint8(np.abs(lap))


def log_edges(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.4) -> np.ndarray:
    """Apply Laplacian of Gaussian (LoG / Marr-Hildreth filter).

    Pre-filters with a Gaussian kernel to suppress noise before applying
    the Laplacian differential operator:
    $$\\mathrm{LoG}(x, y) = -\\frac{1}{\\pi\\sigma^4} \\left[ 1 - \\frac{x^2+y^2}{2\\sigma^2} \\right] e^{-\\frac{x^2+y^2}{2\\sigma^2}}$$

    Args:
        image: Input image array.
        kernel_size: Gaussian kernel dimension.
        sigma: Standard deviation $\\sigma$.

    Returns:
        np.ndarray: Absolute LoG response in uint8.
    """
    blurred = gaussian_denoise(image, kernel_size=kernel_size, sigma=sigma)
    return laplacian_edges(blurred, ksize=3)


def canny_edges(
    image: np.ndarray,
    lower_threshold: int = 80,
    upper_threshold: int = 160,
    aperture_size: int = 3,
    l2_gradient: bool = True,
) -> np.ndarray:
    """Apply Canny edge detection pipeline.

    Multi-stage algorithm:
    1. Gaussian smoothing to suppress noise.
    2. Sobel gradient magnitude and orientation calculation.
    3. Non-Maximum Suppression (NMS) along gradient normal lines to thin edges to 1-pixel width.
    4. Hysteresis thresholding ($T_{low}, T_{high}$) and edge tracking by connectivity.

    Args:
        image: Input image array.
        lower_threshold: Hysteresis lower threshold (weak edge candidate cutoff).
        upper_threshold: Hysteresis upper threshold (strong edge cutoff).
        aperture_size: Sobel aperture size (default: 3).
        l2_gradient: If True, uses Euclidean $L_2$ norm $\\sqrt{G_x^2 + G_y^2}$ instead of $L_1$ norm $|G_x| + |G_y|$.

    Returns:
        np.ndarray: Binary edge map (0 and 255) in uint8.
    """
    gray = to_gray(image)
    return cv2.Canny(
        ensure_uint8(gray),
        threshold1=lower_threshold,
        threshold2=upper_threshold,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )


def compare_edge_detectors(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Run all classical edge detectors on an image for comprehensive comparison.

    Args:
        image: Input image array.

    Returns:
        dict[str, np.ndarray]: Mapping of detector names to their respective output maps.
    """
    denoised = gaussian_denoise(image, kernel_size=5, sigma=1.2)
    return {
        "sobel": sobel_edges(denoised),
        "prewitt": prewitt_edges(denoised),
        "scharr": scharr_edges(denoised),
        "laplacian": laplacian_edges(denoised),
        "log": log_edges(image, kernel_size=5, sigma=1.4),
        "canny": canny_edges(denoised, lower_threshold=50, upper_threshold=150),
    }
