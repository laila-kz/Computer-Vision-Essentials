"""Feature extraction, binary descriptors, and robust feature correspondence matching.

Implements ORB (Oriented FAST and Rotated BRIEF) keypoint extraction and matching pipelines:
- ORB keypoint detection (multi-scale image pyramid + oriented FAST)
- BRIEF binary descriptor calculation (steered by intensity centroid orientation)
- Brute-Force Matcher with Hamming Distance ($d(a, b) = \\sum a_i \\oplus b_i$)
- FLANN (Fast Library for Approximate Nearest Neighbors) matching with LSH index
- Lowe's Ratio Test ($d(best_1) < \\tau \\cdot d(best_2)$) to filter ambiguous false matches
- RANSAC Homography Estimation to compute geometric transformation and filter spatial outliers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..utils.conversions import ensure_uint8, to_gray


@dataclass(frozen=True)
class MatchResult:
    """Encapsulates feature detection and correspondence matching outputs.

    Attributes:
        keypoints_query: Detected keypoints in the first/query image.
        keypoints_train: Detected keypoints in the second/train image.
        matches: List of validated OpenCV DMatch instances.
        matched_image: Visual montage showing correspondence lines between images.
        homography: 3x3 projective transformation matrix (None if homography was not fit).
        inliers_mask: Boolean mask indicating which matches are geometric inliers.
    """

    keypoints_query: List[cv2.KeyPoint]
    keypoints_train: List[cv2.KeyPoint]
    matches: List[cv2.DMatch]
    matched_image: np.ndarray
    homography: Optional[np.ndarray] = None
    inliers_mask: Optional[np.ndarray] = None


def orb_keypoints_and_descriptors(
    image: np.ndarray,
    nfeatures: int = 1000,
    scale_factor: float = 1.2,
    nlevels: int = 8,
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """Detect ORB keypoints and compute 256-bit binary descriptors.

    Args:
        image: Input grayscale or color image.
        nfeatures: Maximum number of strongest features to retain.
        scale_factor: Pyramid decimation ratio (>1.0).
        nlevels: Number of pyramid levels.

    Returns:
        tuple containing:
            - list[cv2.KeyPoint]: Detected keypoints.
            - np.ndarray or None: (N, 32) uint8 descriptor matrix (32 bytes = 256 bits).
    """
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=scale_factor,
        nlevels=nlevels,
        edgeThreshold=15,
        patchSize=31,
    )
    gray = to_gray(image)
    keypoints, descriptors = orb.detectAndCompute(ensure_uint8(gray), None)
    return list(keypoints), descriptors


def match_orb_features(
    image_a: np.ndarray,
    image_b: np.ndarray,
    max_matches: int = 50,
    cross_check: bool = True,
) -> MatchResult:
    """Match ORB descriptors between two images using Brute-Force Hamming distance.

    Args:
        image_a: Query image array.
        image_b: Reference/Train image array.
        max_matches: Maximum number of top matches to draw.
        cross_check: If True, only pairs where $A \\to B$ and $B \\to A$ are mutual nearest neighbors are returned.

    Returns:
        MatchResult: Complete match metadata and annotated visualization image.

    Raises:
        ValueError: If descriptors could not be computed in either image.
    """
    keypoints_a, descriptors_a = orb_keypoints_and_descriptors(image_a)
    keypoints_b, descriptors_b = orb_keypoints_and_descriptors(image_b)

    if descriptors_a is None or descriptors_b is None or len(descriptors_a) == 0 or len(descriptors_b) == 0:
        raise ValueError("Could not compute ORB descriptors for one or both images")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
    matches = matcher.match(descriptors_a, descriptors_b)
    matches = sorted(matches, key=lambda m: m.distance)
    good_matches = matches[:max_matches]

    matched_image = cv2.drawMatches(
        ensure_uint8(image_a),
        keypoints_a,
        ensure_uint8(image_b),
        keypoints_b,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return MatchResult(
        keypoints_query=keypoints_a,
        keypoints_train=keypoints_b,
        matches=good_matches,
        matched_image=matched_image,
    )


def match_features_ratio_test(
    image_a: np.ndarray,
    image_b: np.ndarray,
    ratio_threshold: float = 0.75,
    ransac_reproj_threshold: float = 3.0,
) -> MatchResult:
    """Match features using k-Nearest Neighbors (k=2) with Lowe's ratio test and RANSAC homography.

    1. Finds top 2 nearest neighbors for each descriptor $d_i$: $(m, n)$.
    2. Enforces Lowe's ratio test: $m.\\mathrm{distance} < \\tau \\cdot n.\\mathrm{distance}$.
    3. Fits a 2D Homography matrix $H$ via RANSAC to eliminate geometric outliers.

    Args:
        image_a: Query image array.
        image_b: Train image array.
        ratio_threshold: Ratio test factor $\\tau$ (typically 0.7 to 0.8).
        ransac_reproj_threshold: Maximum allowable reprojection error in pixels for RANSAC.

    Returns:
        MatchResult: Filtered matches, annotated visualization, and estimated 3x3 homography matrix.
    """
    keypoints_a, descriptors_a = orb_keypoints_and_descriptors(image_a)
    keypoints_b, descriptors_b = orb_keypoints_and_descriptors(image_b)

    if descriptors_a is None or descriptors_b is None:
        raise ValueError("Could not compute ORB descriptors for feature matching.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(descriptors_a, descriptors_b, k=2)

    good_matches: List[cv2.DMatch] = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    homography = None
    inliers_mask = None

    if len(good_matches) >= 4:
        src_pts = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_reproj_threshold)
        if mask is not None:
            inliers_mask = mask.ravel()

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if inliers_mask is not None:
        draw_params["matchesMask"] = inliers_mask.tolist()

    matched_image = cv2.drawMatches(
        ensure_uint8(image_a),
        keypoints_a,
        ensure_uint8(image_b),
        keypoints_b,
        good_matches,
        None,
        **draw_params,
    )

    return MatchResult(
        keypoints_query=keypoints_a,
        keypoints_train=keypoints_b,
        matches=good_matches,
        matched_image=matched_image,
        homography=homography,
        inliers_mask=inliers_mask,
    )
