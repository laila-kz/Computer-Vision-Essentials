"""Mathematical morphology operations and structuring element utilities.

Implements set-theoretic mathematical morphology on binary and grayscale images:
- Erosion: $A \\ominus B = \\{ z \\mid (B)_z \\subseteq A \\}$ (Shrinks foreground objects)
- Dilation: $A \\oplus B = \\{ z \\mid (\\hat{B})_z \\cap A \\neq \\emptyset \\}$ (Expands foreground objects)
- Opening: $A \\circ B = (A \\ominus B) \\oplus B$ (Eliminates small noise, smooths boundaries)
- Closing: $A \\bullet B = (A \\oplus B) \\ominus B$ (Fills small holes and bridges narrow gaps)
- Morphological Gradient: $(A \\oplus B) - (A \\ominus B)$ (Extracts boundary perimeter)
- Top-Hat: $A - (A \\circ B)$ (Isolates elements brighter than surroundings)
- Black-Hat: $(A \\bullet B) - A$ (Isolates elements darker than surroundings)
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..utils.conversions import ensure_uint8


def get_structuring_element(shape: str = "rect", size: int = 3) -> np.ndarray:
    """Create a 2D morphological structuring element ($B$).

    Args:
        shape: Kernel geometry: 'rect' (box), 'ellipse' / 'disk', or 'cross'.
        size: Width and height of the kernel window.

    Returns:
        np.ndarray: Structuring element array with uint8 precision.
    """
    k_shape = cv2.MORPH_RECT
    if shape.lower() in ("ellipse", "disk", "circle"):
        k_shape = cv2.MORPH_ELLIPSE
    elif shape.lower() == "cross":
        k_shape = cv2.MORPH_CROSS
    return cv2.getStructuringElement(k_shape, (size, size))


def erode_image(image: np.ndarray, kernel_size: int = 3, iterations: int = 1, shape: str = "rect") -> np.ndarray:
    """Perform morphological erosion ($A \\ominus B$).

    Args:
        image: Binary or grayscale image.
        kernel_size: Structuring element diameter.
        iterations: Number of consecutive erosion passes.
        shape: Kernel geometry ('rect', 'ellipse', 'cross').

    Returns:
        np.ndarray: Eroded image array.
    """
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.erode(ensure_uint8(image), kernel, iterations=iterations)


def dilate_image(image: np.ndarray, kernel_size: int = 3, iterations: int = 1, shape: str = "rect") -> np.ndarray:
    """Perform morphological dilation ($A \\oplus B$).

    Args:
        image: Binary or grayscale image.
        kernel_size: Structuring element diameter.
        iterations: Number of consecutive dilation passes.
        shape: Kernel geometry ('rect', 'ellipse', 'cross').

    Returns:
        np.ndarray: Dilated image array.
    """
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.dilate(ensure_uint8(image), kernel, iterations=iterations)


def morphology_open(image: np.ndarray, kernel_size: int = 3, shape: str = "rect") -> np.ndarray:
    """Perform morphological opening ($A \\circ B = (A \\ominus B) \\oplus B$).

    Removes isolated foreground noise points and thins isthmuses without changing area significantly.

    Args:
        image: Input image array.
        kernel_size: Structuring element dimension.
        shape: Geometry of structuring element.

    Returns:
        np.ndarray: Opened image array.
    """
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.morphologyEx(ensure_uint8(image), cv2.MORPH_OPEN, kernel)


def morphology_close(image: np.ndarray, kernel_size: int = 5, shape: str = "rect") -> np.ndarray:
    """Perform morphological closing ($A \\bullet B = (A \\oplus B) \\ominus B$).

    Bridges internal fractures, fills pinholes, and fuses close contours.

    Args:
        image: Input image array.
        kernel_size: Structuring element dimension.
        shape: Geometry of structuring element.

    Returns:
        np.ndarray: Closed image array.
    """
    kernel = get_structuring_element(shape, kernel_size)
    return cv2.morphologyEx(ensure_uint8(image), cv2.MORPH_CLOSE, kernel)


def morphological_gradient(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Compute basic morphological gradient ($Dilation - Erosion$).

    Highlights localized perimeter contours of objects in binary or grayscale images.

    Args:
        image: Input image array.
        kernel_size: Structuring element dimension.

    Returns:
        np.ndarray: Gradient outline image.
    """
    kernel = get_structuring_element("rect", kernel_size)
    return cv2.morphologyEx(ensure_uint8(image), cv2.MORPH_GRADIENT, kernel)


def tophat_transform(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Perform White Top-Hat transform ($I - \\mathrm{Open}(I)$).

    Isolates bright foreground structures smaller than the structuring element on uneven backgrounds.

    Args:
        image: Input image array.
        kernel_size: Kernel dimension.

    Returns:
        np.ndarray: Top-hat filtered image.
    """
    kernel = get_structuring_element("rect", kernel_size)
    return cv2.morphologyEx(ensure_uint8(image), cv2.MORPH_TOPHAT, kernel)


def blackhat_transform(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Perform Black Top-Hat / Bottom-Hat transform ($\\mathrm{Close}(I) - I$).

    Isolates dark structural details smaller than the structuring element.

    Args:
        image: Input image array.
        kernel_size: Kernel dimension.

    Returns:
        np.ndarray: Black-hat filtered image.
    """
    kernel = get_structuring_element("rect", kernel_size)
    return cv2.morphologyEx(ensure_uint8(image), cv2.MORPH_BLACKHAT, kernel)


def morphology_cleanup(binary_mask: np.ndarray, open_size: int = 3, close_size: int = 5) -> np.ndarray:
    """Sequential morphological filtering pipeline (Opening followed by Closing).

    1. Removes isolated speckle noise (Opening).
    2. Fills internal voids and pinholes within segmented objects (Closing).

    Args:
        binary_mask: Binarized input mask.
        open_size: Kernel size for noise removal opening.
        close_size: Kernel size for void filling closing.

    Returns:
        np.ndarray: Cleaned binary mask.
    """
    opened = morphology_open(binary_mask, kernel_size=open_size, shape="ellipse")
    closed = morphology_close(opened, kernel_size=close_size, shape="ellipse")
    return closed
