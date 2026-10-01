"""Object and face detection using Viola-Jones Haar-like feature cascade classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..utils.conversions import ensure_uint8, to_gray


@dataclass(frozen=True)
class DetectionSummary:
    """Bounding box coordinates and annotated visualization of detection results.

    Attributes:
        boxes: List of bounding rectangles in (x, y, w, h) format.
        annotated_image: Image with color-coded bounding rectangles and labels.
    """

    boxes: List[Tuple[int, int, int, int]]
    annotated_image: np.ndarray


def detect_faces(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (30, 30),
) -> DetectionSummary:
    """Detect human frontal faces using Viola-Jones Haar Cascade Classifier.

    The Viola-Jones detector relies on:
    1. Integral Image representation for constant-time rectangular feature evaluations.
    2. AdaBoost classifier selection to isolate key discriminative Haar features.
    3. Attentional cascade hierarchy to rapidly reject background regions.

    Args:
        image: Input color or grayscale image.
        scale_factor: Parameter specifying how much the image size is reduced at each image scale.
        min_neighbors: Parameter specifying how many neighbors each candidate rectangle should retain.
        min_size: Minimum possible object size in pixels (width, height).

    Returns:
        DetectionSummary: Detected bounding boxes and annotated BGR image.
    """
    gray = to_gray(image)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    classifier = cv2.CascadeClassifier(cascade_path)

    if classifier.empty():
        raise RuntimeError(f"Unable to load Haar cascade classifier from {cascade_path}")

    faces = classifier.detectMultiScale(
        ensure_uint8(gray),
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )

    annotated = cv2.cvtColor(ensure_uint8(gray), cv2.COLOR_GRAY2BGR) if image.ndim == 2 else ensure_uint8(image).copy()
    boxes: List[Tuple[int, int, int, int]] = []

    for x, y, w, h in faces:
        bx, by, bw, bh = int(x), int(y), int(w), int(h)
        boxes.append((bx, by, bw, bh))
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            "Face",
            (bx, max(15, by - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return DetectionSummary(boxes=boxes, annotated_image=annotated)


def detect_eyes(
    image: np.ndarray,
    face_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
) -> DetectionSummary:
    """Detect eyes either globally or constrained within detected facial ROIs.

    Args:
        image: Input image array.
        face_boxes: Optional list of previously detected face bounding boxes to constrain search.
        scale_factor: Multiscale reduction factor.
        min_neighbors: Neighbor threshold.

    Returns:
        DetectionSummary: Detected eye bounding boxes and annotated image.
    """
    gray = to_gray(image)
    cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
    classifier = cv2.CascadeClassifier(cascade_path)

    if classifier.empty():
        raise RuntimeError(f"Unable to load Eye cascade classifier from {cascade_path}")

    annotated = cv2.cvtColor(ensure_uint8(gray), cv2.COLOR_GRAY2BGR) if image.ndim == 2 else ensure_uint8(image).copy()
    eye_boxes: List[Tuple[int, int, int, int]] = []

    if face_boxes:
        for fx, fy, fw, fh in face_boxes:
            roi_gray = gray[fy : fy + fh, fx : fx + fw]
            eyes = classifier.detectMultiScale(roi_gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
            for ex, ey, ew, eh in eyes:
                gx, gy = fx + int(ex), fy + int(ey)
                eye_boxes.append((gx, gy, int(ew), int(eh)))
                cv2.rectangle(annotated, (gx, gy), (gx + int(ew), gy + int(eh)), (255, 0, 0), 2)
    else:
        eyes = classifier.detectMultiScale(ensure_uint8(gray), scaleFactor=scale_factor, minNeighbors=min_neighbors)
        for ex, ey, ew, eh in eyes:
            eye_boxes.append((int(ex), int(ey), int(ew), int(eh)))
            cv2.rectangle(annotated, (int(ex), int(ey)), (int(ex + ew), int(ey + eh)), (255, 0, 0), 2)

    return DetectionSummary(boxes=eye_boxes, annotated_image=annotated)
