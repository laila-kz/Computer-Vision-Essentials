"""Viola-Jones Face & Eye Detection Demo Script.

Demonstrates real-time object detection using Haar-like Feature Cascade Classifiers:
- Integral image representations for rapid feature computation
- Attentional cascade filtering
- Multi-scale sliding window detection
- Eye detection constrained to facial ROIs

Usage:
    python face_detection_project.py [--image path/to/image.png] [--save-plot output.png] [--show]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

# Allow execution from root or subfolder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_portfolio.detection.cascade import detect_eyes, detect_faces
from cv_portfolio.utils.io import read_image
from cv_portfolio.visualization.plotting import plot_image_grid


def generate_synthetic_face() -> np.ndarray:
    """Generate a stylized synthetic face avatar for demonstration when no image is provided."""
    canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    canvas.fill(230)
    # Head outline
    cv2.ellipse(canvas, (200, 200), (90, 120), 0, 0, 360, (190, 210, 240), -1)
    cv2.ellipse(canvas, (200, 200), (90, 120), 0, 0, 360, (100, 120, 150), 2)
    # Eyes
    cv2.circle(canvas, (165, 170), 12, (255, 255, 255), -1)
    cv2.circle(canvas, (235, 170), 12, (255, 255, 255), -1)
    cv2.circle(canvas, (165, 170), 6, (60, 40, 30), -1)
    cv2.circle(canvas, (235, 170), 6, (60, 40, 30), -1)
    # Nose
    cv2.line(canvas, (200, 180), (195, 215), (100, 100, 100), 2)
    cv2.line(canvas, (195, 215), (205, 215), (100, 100, 100), 2)
    # Smile
    cv2.ellipse(canvas, (200, 240), (35, 20), 0, 10, 170, (40, 40, 180), 3)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Viola-Jones Face & Eye Detection using Haar Feature Cascades.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image containing human faces.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Image pyramid scaling factor per stage.",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Minimum neighbor bounding boxes required for detection candidate.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="outputs/face_detection_result.png",
        help="Path to save output visualization figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively.",
    )
    args = parser.parse_args()

    if args.image and Path(args.image).exists():
        print(f"[+] Loading input image: {args.image}")
        image = read_image(args.image)
    else:
        print("[!] No image supplied or file not found. Generating synthetic avatar canvas...")
        image = generate_synthetic_face()

    print("[*] Running Haar cascade frontal face and eye detectors...")
    face_result = detect_faces(
        image, scale_factor=args.scale_factor, min_neighbors=args.min_neighbors
    )
    print(f"[✓] Faces Detected: {len(face_result.boxes)}")

    eye_result = detect_eyes(
        face_result.annotated_image,
        face_boxes=face_result.boxes,
        scale_factor=args.scale_factor,
        min_neighbors=args.min_neighbors,
    )
    print(f"[✓] Eyes Detected within Face ROIs: {len(eye_result.boxes)}")

    grid_images = {
        "Input Image": image,
        f"Face & Eye Detections ({len(face_result.boxes)} faces, {len(eye_result.boxes)} eyes)": eye_result.annotated_image,
    }

    if args.save_plot or args.show:
        plot_image_grid(
            images=grid_images,
            title="Viola-Jones Haar Cascade Face & Facial Feature Localization",
            ncols=2,
            figsize=(12, 5),
            save_path=args.save_plot,
            show=args.show,
        )
        if args.save_plot:
            print(f"[+] Detection results saved to: {args.save_plot}")


if __name__ == "__main__":
    main()
