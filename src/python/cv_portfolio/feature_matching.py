from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .utils import ensure_uint8, to_gray


@dataclass(frozen=True)
class MatchResult:
    keypoints_query: list[cv2.KeyPoint]
    keypoints_train: list[cv2.KeyPoint]
    matches: list[cv2.DMatch]
    matched_image: np.ndarray


def orb_keypoints_and_descriptors(image: np.ndarray, nfeatures: int = 1000):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    gray = to_gray(image)
    keypoints, descriptors = orb.detectAndCompute(ensure_uint8(gray), None)
    return keypoints, descriptors


def match_orb_features(image_a: np.ndarray, image_b: np.ndarray, max_matches: int = 50) -> MatchResult:
    keypoints_a, descriptors_a = orb_keypoints_and_descriptors(image_a)
    keypoints_b, descriptors_b = orb_keypoints_and_descriptors(image_b)
    if descriptors_a is None or descriptors_b is None:
        raise ValueError("Could not compute ORB descriptors for one or both images")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(descriptors_a, descriptors_b), key=lambda match: match.distance)
    good_matches = matches[:max_matches]
    matched_image = cv2.drawMatches(
        ensure_uint8(image_a), keypoints_a,
        ensure_uint8(image_b), keypoints_b,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return MatchResult(keypoints_a, keypoints_b, good_matches, matched_image)


def template_match(image: np.ndarray, template: np.ndarray) -> tuple[tuple[int, int], float, np.ndarray]:
    gray_image = to_gray(image)
    gray_template = to_gray(template)
    result = cv2.matchTemplate(ensure_uint8(gray_image), ensure_uint8(gray_template), cv2.TM_CCOEFF_NORMED)
    _, max_value, _, max_location = cv2.minMaxLoc(result)
    return max_location, max_value, result
