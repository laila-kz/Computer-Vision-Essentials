"""Image Segmentation & Morphological Processing Demo Script.

Demonstrates classical binarization and regional morphological analysis:
- Otsu Global Variance Optimization
- Morphological Opening & Closing Filtering
- 8-Connectivity Connected Components Labeling (CCL)
- Regional Bounding Box, Centroid, and Area Statistics

Usage:
    python image_segmentation_demo.py [--image path/to/image.png] [--save-plot output.png] [--show]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

# Allow execution from root or subfolder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_portfolio.segmentation.pipeline import segmentation_pipeline
from cv_portfolio.utils.conversions import ensure_uint8, to_gray
from cv_portfolio.utils.io import generate_synthetic_image, read_image
from cv_portfolio.visualization.plotting import plot_image_grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demonstrate Otsu thresholding, morphological noise cleanup, and connected component labeling.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image. If omitted, generates synthetic shapes test pattern.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="outputs/segmentation_pipeline_stages.png",
        help="Path to save multi-panel segmentation stages plot.",
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
        print("[!] No image supplied or file not found. Generating synthetic shapes pattern...")
        image = generate_synthetic_image(pattern="shapes", noise_level=8.0)

    print("[*] Running end-to-end segmentation pipeline (Otsu -> Morphology -> CCL)...")
    results = segmentation_pipeline(image)

    print(f"[✓] Optimal Otsu Threshold: {results['threshold']:.2f}")
    print(f"[✓] Connected Components Detected: {results['num_labels'] - 1} foreground regions")
    print(f"[✓] Foreground Occupancy: {results['foreground_ratio'] * 100:.2f}%")

    # Render colored label map for visualization
    labels: np.ndarray = results["labels"]
    num_labels: int = results["num_labels"]
    label_hue = np.uint8(179 * labels / max(1, np.max(labels)))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[labels == 0] = 0

    # Draw bounding boxes on color image
    annotated = cv2.cvtColor(ensure_uint8(to_gray(image)), cv2.COLOR_GRAY2BGR)
    for comp in results.get("components", []):
        cv2.rectangle(annotated, (comp.left, comp.top), (comp.left + comp.width, comp.top + comp.height), (0, 255, 0), 2)
        cv2.circle(annotated, (int(comp.centroid_x), int(comp.centroid_y)), 3, (0, 0, 255), -1)

    grid_images = {
        "1. Input Grayscale": to_gray(image),
        f"2. Otsu Binary Mask (T={results['threshold']:.1f})": results["binary"],
        "3. Morphologically Cleaned": results["cleaned"],
        f"4. Connected Components ({num_labels-1} Objects)": labeled_img,
        "5. Region Bounding Boxes & Centroids": annotated,
    }

    if args.save_plot or args.show:
        plot_image_grid(
            images=grid_images,
            title="Otsu Binarization & Connected Component Analysis Pipeline",
            ncols=3,
            save_path=args.save_plot,
            show=args.show,
        )
        if args.save_plot:
            print(f"[+] Multi-panel segmentation figure saved to: {args.save_plot}")


if __name__ == "__main__":
    main()
