"""Feature detection, corner extraction, binary descriptors, and contour geometry."""

from .contours import (
    ContourProperties,
    contour_properties,
    draw_annotated_contours,
    find_external_contours,
)
from .corners import fast_corners, harris_corners, shi_tomasi_corners
from .matching import (
    MatchResult,
    match_features_ratio_test,
    match_orb_features,
    orb_keypoints_and_descriptors,
)

__all__ = [
    "harris_corners",
    "shi_tomasi_corners",
    "fast_corners",
    "MatchResult",
    "orb_keypoints_and_descriptors",
    "match_orb_features",
    "match_features_ratio_test",
    "ContourProperties",
    "find_external_contours",
    "contour_properties",
    "draw_annotated_contours",
]
