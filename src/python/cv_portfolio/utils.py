from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageStats:
    height: int
    width: int
    channels: int
    dtype: str


def read_image(path: str | Path, color_flag: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), color_flag)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


def image_stats(image: np.ndarray) -> ImageStats:
    channels = 1 if image.ndim == 2 else image.shape[2]
    return ImageStats(height=image.shape[0], width=image.shape[1], channels=channels, dtype=str(image.dtype))


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return normalize_uint8(image)


def count_nonzero_ratio(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / float(mask.size)


def label_components(binary_mask: np.ndarray):
    return cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
