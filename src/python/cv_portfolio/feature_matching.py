"""Feature detection, descriptor extraction, and correspondence matching."""

from __future__ import annotations

from .detection.template import template_match
from .features.corners import fast_corners, harris_corners, shi_tomasi_corners
from .features.matching import (
    MatchResult,
    match_features_ratio_test,
    match_orb_features,
    orb_keypoints_and_descriptors,
)

__all__ = [
    "MatchResult",
    "orb_keypoints_and_descriptors",
    "match_orb_features",
    "match_features_ratio_test",
    "template_match",
    "harris_corners",
    "shi_tomasi_corners",
    "fast_corners",
]
