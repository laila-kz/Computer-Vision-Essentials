from __future__ import annotations

import cv2
import numpy as np

from .utils import ensure_uint8, to_gray


def gaussian_denoise(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.2) -> np.ndarray:
    return cv2.GaussianBlur(ensure_uint8(image), (kernel_size, kernel_size), sigma)


def sobel_edges(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return ensure_uint8(magnitude)


def laplacian_edges(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return ensure_uint8(np.abs(laplacian))


def canny_edges(image: np.ndarray, lower_threshold: int = 80, upper_threshold: int = 160) -> np.ndarray:
    gray = to_gray(image)
    return cv2.Canny(gray, lower_threshold, upper_threshold)


def compare_edge_detectors(image: np.ndarray) -> dict[str, np.ndarray]:
    denoised = gaussian_denoise(image)
    return {
        "sobel": sobel_edges(denoised),
        "laplacian": laplacian_edges(denoised),
        "canny": canny_edges(denoised),
    }
