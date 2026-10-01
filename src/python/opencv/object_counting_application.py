"""Automated Object Counting & Circular Shape Analysis Demo Script.

Demonstrates automated industrial inspection / counting:
- Preprocessing & Otsu Adaptive Binarization
- Morphological Noise Cleanup
- Contour Feature Analysis (Area, Perimeter, Compactness / Circularity)
- Bounding Box and Centroid Visualization

Usage:
    python object_counting_application.py [--image path/to/image.png] [--min-area 100] [--save-plot output.png] [--show]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

# Allow execution from root or subfolder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_portfolio.features.contours import draw_annotated_contours
from cv_portfolio.detection.counting import count_circular_objects, count_contours
from cv_portfolio.segmentation.morphology import morphology_cleanup
from cv_portfolio.segmentation.thresholding import otsu_threshold
from cv_portfolio.utils.conversions import ensure_uint8, to_gray
from cv_portfolio.utils.io import generate_synthetic_image, read_image
from cv_portfolio.visualization.plotting import plot_image_grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated object and circular particle counting application.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image. If omitted, generates synthetic coins/particles pattern.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=150,
        help="Minimum contour area in pixels to filter out noise.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="outputs/object_counting_summary.png",
        help="Path to save multi-panel object counting diagnostic plot.",
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
        print("[!] No image supplied or file not found. Generating synthetic coin particles pattern...")
        image = generate_synthetic_image(pattern="coins", noise_level=6.0)

    print("[*] Segmenting image and extracting external contours...")
    _, binary = otsu_threshold(image)
    cleaned = morphology_cleanup(binary, open_size=3, close_size=5)

    total_count, contours = count_contours(cleaned, min_area=args.min_area)
    circular_count, circular_contours = count_circular_objects(
        cleaned, min_area=args.min_area, min_circularity=0.65
    )

    print(f"[✓] Total Counted Objects (Area >= {args.min_area}px): {total_count}")
    print(f"[✓] Classified Circular Objects (Circularity >= 0.65): {circular_count}")

    annotated = draw_annotated_contours(image, contours, draw_boxes=True, draw_centroids=True)

    grid_images = {
        "1. Input Image": image,
        "2. Otsu Threshold": binary,
        "3. Morphological Cleanup": cleaned,
        f"4. Annotated Detections ({total_count} items)": annotated,
    }

    if args.save_plot or args.show:
        plot_image_grid(
            images=grid_images,
            title=f"Automated Object Counting Pipeline ({total_count} Objects Detected)",
            ncols=2,
            figsize=(12, 10),
            save_path=args.save_plot,
            show=args.show,
        )
        if args.save_plot:
            print(f"[+] Counting diagnostics saved to: {args.save_plot}")


if __name__ == "__main__":
    main()
