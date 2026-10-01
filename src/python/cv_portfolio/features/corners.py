"""Corner detection and interest point localization algorithms.

Implements mathematical interest point detectors:
- Harris Corner Detector: Second-moment matrix ($M = \\sum w(x,y) \\begin{bmatrix} I_x^2 & I_x I_y \\\\ I_x I_y & I_y^2 \\end{bmatrix}$),
  Corner response measure $R = \\det(M) - k \\cdot (\\mathrm{trace}(M))^2 = \\lambda_1 \\lambda_2 - k(\\lambda_1 + \\lambda_2)^2$.
- Shi-Tomasi (Good Features to Track): $R = \\min(\\lambda_1, \\lambda_2)$.
- FAST (Features from Accelerated Segment Test): High-speed corner detection via circular 16-pixel Bresenham neighborhood comparison.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from ..utils.conversions import ensure_uint8, to_gray


def harris_corners(
    image: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold_ratio: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect corner keypoints using the Harris corner response function.

    Args:
        image: Input grayscale or color image.
        block_size: Neighborhood size for second-moment matrix calculation ($M$).
        ksize: Aperture parameter for Sobel derivative.
        k: Harris detector free parameter in response equation ($k \\in [0.04, 0.06]$).
        threshold_ratio: Fraction of max response used to threshold positive corner detections.

    Returns:
        tuple containing:
            - np.ndarray: Dilated floating-point response map $R(x, y)$.
            - np.ndarray: BGR annotated image with detected corners marked in red circles.
    """
    gray = to_gray(image).astype(np.float32)
    response = cv2.cornerHarris(gray, blockSize=block_size, ksize=ksize, k=k)
    response_dilated = cv2.dilate(response, None)

    annotated = cv2.cvtColor(ensure_uint8(to_gray(image)), cv2.COLOR_GRAY2BGR)
    threshold = threshold_ratio * response_dilated.max()
    annotated[response_dilated > threshold] = [0, 0, 255]  # Red in BGR

    return response_dilated, annotated


def shi_tomasi_corners(
    image: np.ndarray,
    max_corners: int = 100,
    quality_level: float = 0.01,
    min_distance: float = 10.0,
    block_size: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect prominent corners using the Shi-Tomasi minimum eigenvalue criterion.

    Computes $R = \\min(\\lambda_1, \\lambda_2)$ and enforces minimum Euclidean distance between keypoints.

    Args:
        image: Input image array.
        max_corners: Maximum number of corners to return.
        quality_level: Minimal accepted quality of image corners (relative to best corner).
        min_distance: Minimum Euclidean distance between returned corners.
        block_size: Window size for autocorrelation matrix.

    Returns:
        tuple containing:
            - np.ndarray: Array of corner coordinates (N, 2) in (x, y) order.
            - np.ndarray: BGR annotated image with corners circled in green.
    """
    gray = to_gray(image)
    corners = cv2.goodFeaturesToTrack(
        ensure_uint8(gray),
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size,
    )

    annotated = cv2.cvtColor(ensure_uint8(gray), cv2.COLOR_GRAY2BGR)
    if corners is not None:
        corners_int = np.int0(corners)
        for pt in corners_int:
            x, y = pt.ravel()
            cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 0), -1)
        return corners.reshape(-1, 2), annotated

    return np.empty((0, 2), dtype=np.float32), annotated


def fast_corners(
    image: np.ndarray,
    threshold: int = 25,
    nonmax_suppression: bool = True,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """Detect interest points using FAST (Features from Accelerated Segment Test).

    Tests a Bresenham circle of 16 pixels surrounding candidate pixel $p$. If $N=9$ (or $N=12$)
    contiguous pixels are all brighter than $I_p + t$ or darker than $I_p - t$, $p$ is a corner.

    Args:
        image: Input image array.
        threshold: Threshold intensity difference $t$.
        nonmax_suppression: Whether to apply non-maximum suppression on adjacent keypoints.

    Returns:
        tuple containing:
            - list[cv2.KeyPoint]: Detected keypoints.
            - np.ndarray: BGR image with keypoints drawn.
    """
    gray = to_gray(image)
    fast = cv2.FastFeatureDetector_create(
        threshold=threshold,
        nonmaxSuppression=nonmax_suppression,
    )
    keypoints = fast.detect(ensure_uint8(gray), None)
    annotated = cv2.drawKeypoints(
        ensure_uint8(image),
        keypoints,
        None,
        color=(0, 255, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT,
    )
    return keypoints, annotated
