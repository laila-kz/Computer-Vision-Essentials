from __future__ import annotations

import cv2
import numpy as np

from .utils import ensure_uint8, to_gray


def contrast_stretch(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    lookup = np.array([(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(ensure_uint8(image), lookup)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    return cv2.equalizeHist(ensure_uint8(gray))


def clahe_enhancement(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    gray = to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(ensure_uint8(gray))


def build_enhancement_summary(image: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "original": ensure_uint8(to_gray(image)),
        "contrast_stretch": contrast_stretch(image),
        "hist_equalized": equalize_histogram(image),
        "clahe": clahe_enhancement(image),
        "gamma_0_8": gamma_correction(image, gamma=0.8),
        "gamma_1_5": gamma_correction(image, gamma=1.5),
    }
