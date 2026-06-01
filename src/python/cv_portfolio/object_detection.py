from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .utils import ensure_uint8, to_gray


@dataclass(frozen=True)
class DetectionSummary:
    boxes: list[tuple[int, int, int, int]]
    annotated_image: np.ndarray


def detect_faces(image: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 5) -> DetectionSummary:
    gray = to_gray(image)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    classifier = cv2.CascadeClassifier(cascade_path)
    if classifier.empty():
        raise RuntimeError(f"Unable to load Haar cascade at {cascade_path}")
    faces = classifier.detectMultiScale(ensure_uint8(gray), scaleFactor=scale_factor, minNeighbors=min_neighbors)
    annotated = ensure_uint8(image).copy()
    boxes: list[tuple[int, int, int, int]] = []
    for x, y, w, h in faces:
        boxes.append((int(x), int(y), int(w), int(h)))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return DetectionSummary(boxes=boxes, annotated_image=annotated)


def count_contours(binary_mask: np.ndarray, min_area: int = 100) -> tuple[int, list[np.ndarray]]:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    return len(filtered), filtered
