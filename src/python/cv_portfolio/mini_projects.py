from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from .edges import compare_edge_detectors
from .enhancement import build_enhancement_summary
from .feature_matching import match_orb_features
from .object_detection import detect_faces
from .segmentation import segmentation_pipeline
from .utils import read_image


@dataclass(frozen=True)
class DemoResult:
    name: str
    outputs: dict[str, object]


def enhancement_toolkit(image_path: str | Path) -> DemoResult:
    image = read_image(image_path)
    return DemoResult("Image Enhancement Toolkit", build_enhancement_summary(image))


def edge_detection_comparison(image_path: str | Path) -> DemoResult:
    image = read_image(image_path)
    return DemoResult("Edge Detection Comparison", compare_edge_detectors(image))


def histogram_analysis_tool(image_path: str | Path) -> DemoResult:
    image = read_image(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
    equalized = cv2.equalizeHist(gray)
    return DemoResult("Histogram Analysis Tool", {"gray": gray, "histogram": histogram, "equalized": equalized})


def image_segmentation_demo(image_path: str | Path) -> DemoResult:
    image = read_image(image_path)
    return DemoResult("Image Segmentation Demo", segmentation_pipeline(image))


def face_detection_project(image_path: str | Path) -> DemoResult:
    image = read_image(image_path)
    detection = detect_faces(image)
    return DemoResult("Face Detection Project", {"boxes": detection.boxes, "annotated_image": detection.annotated_image})


def feature_matching_demo(image_path_a: str | Path, image_path_b: str | Path) -> DemoResult:
    image_a = read_image(image_path_a)
    image_b = read_image(image_path_b)
    matches = match_orb_features(image_a, image_b)
    return DemoResult("Feature Matching Demo", {"matches": matches.matches, "matched_image": matches.matched_image})
