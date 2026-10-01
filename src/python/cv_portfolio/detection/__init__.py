"""Object detection, Viola-Jones cascades, template matching, and object counting."""

from .cascade import DetectionSummary, detect_eyes, detect_faces
from .counting import count_circular_objects, count_contours
from .template import multi_scale_template_match, template_match

__all__ = [
    "DetectionSummary",
    "detect_faces",
    "detect_eyes",
    "template_match",
    "multi_scale_template_match",
    "count_contours",
    "count_circular_objects",
]
