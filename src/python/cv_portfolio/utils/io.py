"""I/O handling and synthetic test pattern generation for computer vision experiments.

Includes safe image file reading and writing with descriptive error messages,
along with procedural generators for test patterns (checkerboard, shapes, noise)
enabling standalone execution of all demos without external asset dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def read_image(path: str | Path, color_flag: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image from disk safely, raising FileNotFoundError if unavailable.

    Args:
        path: Path to the image file.
        color_flag: OpenCV imread flag (default: cv2.IMREAD_COLOR for BGR 3-channel).

    Returns:
        np.ndarray: Loaded image array.

    Raises:
        FileNotFoundError: If the file does not exist or OpenCV fails to decode it.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {file_path.resolve()}")

    image = cv2.imread(str(file_path), color_flag)
    if image is None:
        raise ValueError(f"OpenCV could not decode image at {file_path.resolve()}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> Path:
    """Save an image array to disk, creating parent directories if necessary.

    Args:
        path: Target file path.
        image: Numpy image array (uint8 or valid OpenCV type).

    Returns:
        Path: Resolved path where the file was written.

    Raises:
        IOError: If OpenCV fails to write the image file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(target), image)
    if not success:
        raise OSError(f"Failed to write image to {target.resolve()}")
    return target


def generate_synthetic_image(
    pattern: str = "shapes",
    size: Tuple[int, int] = (400, 400),
    noise_level: float = 0.0,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Procedurally generate synthetic images for testing CV pipelines.

    Supported patterns:
        - "shapes": Overlapping geometric shapes (circles, rectangles, triangles) with varied intensity.
        - "checkerboard": Alternating black-and-white grid pattern.
        - "coins": Simulated circular foreground disks on dark background for thresholding & watershed.
        - "gradient": Smooth 2D intensity ramp with optional noise.

    Args:
        pattern: Name of the pattern to generate.
        size: Tuple of (height, width).
        noise_level: Standard deviation of additive Gaussian noise (0.0 to 50.0).
        seed: Random seed for deterministic reproducibility.

    Returns:
        np.ndarray: Generated 3-channel BGR image array with uint8 precision.
    """
    if seed is not None:
        np.random.seed(seed)

    h, w = size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if pattern == "checkerboard":
        block_size = max(10, min(h, w) // 8)
        for r in range(0, h, block_size):
            for c in range(0, w, block_size):
                if ((r // block_size) + (c // block_size)) % 2 == 0:
                    canvas[r : r + block_size, c : c + block_size] = (240, 240, 240)
                else:
                    canvas[r : r + block_size, c : c + block_size] = (30, 30, 30)

    elif pattern == "coins":
        # Dark background with circular bright disks
        canvas.fill(25)
        centers = [
            (int(w * 0.25), int(h * 0.3)),
            (int(w * 0.45), int(h * 0.35)),
            (int(w * 0.75), int(h * 0.28)),
            (int(w * 0.30), int(h * 0.70)),
            (int(w * 0.60), int(h * 0.65)),
            (int(w * 0.80), int(h * 0.75)),
            (int(w * 0.50), int(h * 0.45)),  # touching disk
        ]
        radii = [35, 42, 30, 38, 45, 32, 40]
        for (cx, cy), r in zip(centers, radii):
            cv2.circle(canvas, (cx, cy), r, (180, 200, 220), -1)
            cv2.circle(canvas, (cx, cy), r, (120, 140, 160), 2)
            # Add interior highlight to mimic metallic coin reflection
            cv2.circle(canvas, (cx - r // 3, cy - r // 3), r // 4, (230, 240, 250), -1)

    elif pattern == "gradient":
        x = np.linspace(0, 255, w, endpoint=True)
        y = np.linspace(0, 255, h, endpoint=True)
        xx, yy = np.meshgrid(x, y)
        ramp = (0.5 * xx + 0.5 * yy).astype(np.uint8)
        canvas = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)

    else:  # "shapes"
        canvas.fill(40)
        # Rectangle
        cv2.rectangle(canvas, (int(w * 0.1), int(h * 0.15)), (int(w * 0.4), int(h * 0.55)), (60, 180, 75), -1)
        # Circle
        cv2.circle(canvas, (int(w * 0.65), int(h * 0.35)), int(min(h, w) * 0.2), (230, 120, 50), -1)
        # Polygon / Triangle
        pts = np.array([[int(w * 0.45), int(h * 0.9)], [int(w * 0.2), int(h * 0.65)], [int(w * 0.7), int(h * 0.7)]], np.int32)
        cv2.fillPoly(canvas, [pts], (220, 60, 150))
        # Nested smaller square
        cv2.rectangle(canvas, (int(w * 0.72), int(h * 0.6)), (int(w * 0.9), int(h * 0.85)), (240, 220, 80), -1)

    if noise_level > 0.0:
        noise = np.random.normal(0, noise_level, canvas.shape).astype(np.float32)
        noisy = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    return canvas
