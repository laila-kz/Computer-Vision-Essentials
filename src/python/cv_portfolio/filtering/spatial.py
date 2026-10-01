"""Spatial filtering, linear smoothing, non-linear denoising, and image sharpening.

Implements core continuous and discrete 2D spatial convolution operators:
- Box Blur (Uniform moving average filter)
- Gaussian Blur ($G(x, y) = \\frac{1}{2\\pi\\sigma^2} e^{-\\frac{x^2+y^2}{2\\sigma^2}}$)
- Median Filtering (Order-statistic non-linear filter for salt-and-pepper noise)
- Bilateral Filtering (Edge-preserving smoothing via spatial & radiometric Gaussian kernels)
- Unsharp Masking ($I_{sharp} = I + \\alpha \\cdot (I - G_\\sigma(I))$)
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..utils.conversions import ensure_uint8, to_gray


def custom_convolution(image: np.ndarray, kernel: np.ndarray, border_type: int = cv2.BORDER_REFLECT) -> np.ndarray:
    """Apply an arbitrary 2D spatial convolution kernel to an image.

    Computes the discrete 2D cross-correlation / convolution:
    $$g(x, y) = \\sum_{i=-k}^k \\sum_{j=-k}^k f(x+i, y+j) \\cdot h(i, j)$$

    Args:
        image: Input grayscale or color image.
        kernel: 2D floating-point kernel matrix.
        border_type: OpenCV border extrapolation mode (default: cv2.BORDER_REFLECT).

    Returns:
        np.ndarray: Filtered image with the same depth and channel count as input.
    """
    return cv2.filter2D(ensure_uint8(image), ddepth=-1, kernel=kernel.astype(np.float32), borderType=border_type)


def box_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply uniform box (mean) spatial smoothing.

    Args:
        image: Input image array.
        kernel_size: Odd integer filter size $k \\times k$.

    Returns:
        np.ndarray: Smoothed image.
    """
    k = max(1, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    return cv2.blur(ensure_uint8(image), (k, k))


def gaussian_denoise(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.2) -> np.ndarray:
    """Apply 2D isotropic Gaussian blur.

    Isotropic 2D Gaussian density function:
    $$G(x, y) = \\frac{1}{2\\pi\\sigma^2} \\exp\\left( -\\frac{x^2 + y^2}{2\\sigma^2} \\right)$$

    Acts as an ideal low-pass filter, attenuating high-frequency Gaussian noise while
    minimizing ringing artifacts in the spatial domain.

    Args:
        image: Input image array.
        kernel_size: Filter dimension (must be positive odd integer, or 0 for auto-computation from sigma).
        sigma: Standard deviation $\\sigma$ of the Gaussian kernel along both axes.

    Returns:
        np.ndarray: Gaussian-smoothed image.
    """
    k = kernel_size if (kernel_size % 2 == 1 or kernel_size == 0) else kernel_size + 1
    return cv2.GaussianBlur(ensure_uint8(image), (k, k), sigmaX=sigma, sigmaY=sigma)


def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply order-statistic Median filter for impulsive (salt-and-pepper) noise removal.

    Replaces each pixel with the statistical median of its neighborhood:
    $$I_{out}(x, y) = \\mathrm{median}\\{ I(x+i, y+j) \\mid (i, j) \\in W \\}$$
    Preserves sharp step edges far better than linear mean or Gaussian smoothing.

    Args:
        image: Input image array.
        kernel_size: Odd integer kernel diameter (e.g., 3, 5, 7).

    Returns:
        np.ndarray: Median-filtered image.
    """
    k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
    return cv2.medianBlur(ensure_uint8(image), k)


def bilateral_denoise(
    image: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0,
) -> np.ndarray:
    """Apply edge-preserving Bilateral filter.

    Combines spatial closeness (geometric distance) with radiometric similarity (photometric distance):
    $$I_{out}(p) = \\frac{1}{W_p} \\sum_{q \\in S} I(q) \\cdot G_{\\sigma_s}(\\|p - q\\|) \\cdot G_{\\sigma_r}(|I(p) - I(q)|)$$

    Smooths texture within flat regions while preventing blur across prominent intensity boundaries.

    Args:
        image: Input image array.
        diameter: Diameter of each pixel neighborhood (e.g., 5-9 for real-time, large for heavy filtering).
        sigma_color: Filter sigma in the color space (larger means wider colors mixed).
        sigma_space: Filter sigma in the coordinate space.

    Returns:
        np.ndarray: Edge-preserved filtered image.
    """
    return cv2.bilateralFilter(ensure_uint8(image), d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def unsharp_mask(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    """Sharpen an image by subtracting an unsharp (low-pass) mask.

    $$I_{sharp} = I + \\alpha \\cdot (I - G_\\sigma(I))$$
    Amplifies high-frequency spatial components and localized gradient transitions.

    Args:
        image: Input image array.
        kernel_size: Gaussian kernel window size.
        sigma: Standard deviation of Gaussian blur.
        strength: Sharpening weight $\\alpha > 0$.

    Returns:
        np.ndarray: Sharpened image clamped to [0, 255] in uint8.
    """
    img_u8 = ensure_uint8(image)
    blurred = gaussian_denoise(img_u8, kernel_size=kernel_size, sigma=sigma)
    mask = img_u8.astype(np.float32) - blurred.astype(np.float32)
    sharpened = img_u8.astype(np.float32) + strength * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)
