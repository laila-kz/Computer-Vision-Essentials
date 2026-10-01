"""Object detection, Viola-Jones face cascades, template matching, and contour counting."""

from __future__ import annotations

from .detection.cascade import DetectionSummary, detect_eyes, detect_faces
from .detection.counting import count_circular_objects, count_contours
from .detection.template import multi_scale_template_match, template_match

__all__ = [
    "DetectionSummary",
    "detect_faces",
    "detect_eyes",
    "template_match",
    "multi_scale_template_match",
    "count_contours",
    "count_circular_objects",
]
