"""High-level mini-project pipelines demonstrating end-to-end Computer Vision workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import cv2
import numpy as np

from .detection.cascade import detect_faces
from .filtering.edges import compare_edge_detectors
from .features.matching import match_orb_features
from .preprocessing.enhancement import build_enhancement_summary
from .segmentation.pipeline import segmentation_pipeline
from .utils.conversions import ensure_uint8
from .utils.io import read_image


@dataclass(frozen=True)
class DemoResult:
    """Standardized output container for course mini-projects.

    Attributes:
        name: Human-readable title of the project.
        outputs: Key-value map of stage names to intermediate images, metrics, or objects.
    """

    name: str
    outputs: Dict[str, Any]


def enhancement_toolkit(image_path: Union[str, Path]) -> DemoResult:
    """Execute complete contrast enhancement and histogram equalization suite.

    Args:
        image_path: Path to input image file.

    Returns:
        DemoResult: Results containing original, equalized, CLAHE, and gamma corrections.
    """
    image = read_image(image_path)
    return DemoResult("Image Enhancement Toolkit", build_enhancement_summary(image))


def edge_detection_comparison(image_path: Union[str, Path]) -> DemoResult:
    """Run Sobel, Prewitt, Scharr, Laplacian, LoG, and Canny edge detectors.

    Args:
        image_path: Path to input image file.

    Returns:
        DemoResult: Results containing output edge maps for all operators.
    """
    image = read_image(image_path)
    return DemoResult("Edge Detection Comparison", compare_edge_detectors(image))


def histogram_analysis_tool(image_path: Union[str, Path]) -> DemoResult:
    """Analyze grayscale intensity distribution and equalization.

    Args:
        image_path: Path to input image file.

    Returns:
        DemoResult: Results containing original grayscale, histogram array, and equalized image.
    """
    image = read_image(image_path)
    gray = cv2.cvtColor(ensure_uint8(image), cv2.COLOR_BGR2GRAY) if image.ndim == 3 else ensure_uint8(image)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
    equalized = cv2.equalizeHist(gray)
    return DemoResult("Histogram Analysis Tool", {"gray": gray, "histogram": histogram, "equalized": equalized})


def image_segmentation_demo(image_path: Union[str, Path]) -> DemoResult:
    """Execute Otsu binarization, morphological opening/closing, and connected component labeling.

    Args:
        image_path: Path to input image file.

    Returns:
        DemoResult: Results containing threshold value, binary mask, cleaned mask, and CCL stats.
    """
    image = read_image(image_path)
    return DemoResult("Image Segmentation Demo", segmentation_pipeline(image))


def face_detection_project(image_path: Union[str, Path]) -> DemoResult:
    """Execute Haar-cascade face detection on an input image.

    Args:
        image_path: Path to input image file.

    Returns:
        DemoResult: Results containing bounding box coordinates and annotated image.
    """
    image = read_image(image_path)
    detection = detect_faces(image)
    return DemoResult(
        "Face Detection Project",
        {"boxes": detection.boxes, "annotated_image": detection.annotated_image},
    )


def feature_matching_demo(image_path_a: Union[str, Path], image_path_b: Union[str, Path]) -> DemoResult:
    """Execute ORB keypoint extraction and brute-force feature matching between two views.

    Args:
        image_path_a: Query image path.
        image_path_b: Train/reference image path.

    Returns:
        DemoResult: Results containing matched feature list and correspondence visualization.
    """
    image_a = read_image(image_path_a)
    image_b = read_image(image_path_b)
    matches = match_orb_features(image_a, image_b)
    return DemoResult(
        "Feature Matching Demo",
        {"matches": matches.matches, "matched_image": matches.matched_image},
    )
