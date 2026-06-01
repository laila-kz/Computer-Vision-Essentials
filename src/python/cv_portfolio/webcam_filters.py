from __future__ import annotations

import cv2
import numpy as np

from .edges import canny_edges
from .utils import ensure_uint8, to_gray


def grayscale_frame(frame: np.ndarray) -> np.ndarray:
    gray = to_gray(frame)
    return cv2.cvtColor(ensure_uint8(gray), cv2.COLOR_GRAY2BGR)


def pencil_sketch_frame(frame: np.ndarray) -> np.ndarray:
    gray = to_gray(frame)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    return cv2.cvtColor(ensure_uint8(sketch), cv2.COLOR_GRAY2BGR)


def edge_frame(frame: np.ndarray) -> np.ndarray:
    edges = canny_edges(frame)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def sharpen_frame(frame: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(ensure_uint8(frame), -1, kernel)


def build_webcam_filter_map(frame: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "original": ensure_uint8(frame),
        "grayscale": grayscale_frame(frame),
        "edge": edge_frame(frame),
        "sketch": pencil_sketch_frame(frame),
        "sharpen": sharpen_frame(frame),
    }
