"""Marker-Controlled Watershed Segmentation Demo Script.

Demonstrates topological watershed separation of touching/overlapping objects:
- Euclidean Distance Transform ($D_{L2}$)
- Peak Local Maxima Seed Marker Generation
- Dilated Background Marker Extraction
- Watershed Topological Flooding to identify dividing boundary ridges

Usage:
    python watershed_segmentation_demo.py [--image path/to/image.png] [--save-plot output.png] [--show]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow execution from root or subfolder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_portfolio.segmentation.watershed import distance_transform_watershed
from cv_portfolio.utils.io import generate_synthetic_image, read_image
from cv_portfolio.visualization.plotting import plot_image_grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Separate touching objects using Euclidean Distance Transform and Marker-Controlled Watershed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image with touching objects. If omitted, generates synthetic touching coins.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.45,
        help="Fraction of peak distance transform value used for marker generation.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="outputs/watershed_segmentation_stages.png",
        help="Path to save intermediate stages plot.",
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
        print("[!] Generating synthetic touching coins pattern...")
        image = generate_synthetic_image(pattern="coins", noise_level=4.0)

    print("[*] Running Marker-Controlled Watershed pipeline...")
    results = distance_transform_watershed(
        image, distance_threshold_ratio=args.distance_threshold
    )

    grid_images = {
        "1. Input Image": image,
        "2. Binary Otsu Mask": results["binary"],
        "3. Morphological Opening": results["opening"],
        "4. Distance Transform ($D_{L2}$)": results["dist_transform"],
        "5. Sure Foreground Seeds": results["sure_fg"],
        "6. Final Watershed Boundaries (Red)": results["segmented"],
    }

    if args.save_plot or args.show:
        plot_image_grid(
            images=grid_images,
            title="Marker-Controlled Watershed Segmentation for Touching Objects",
            ncols=3,
            save_path=args.save_plot,
            show=args.show,
        )
        if args.save_plot:
            print(f"[+] Watershed pipeline visualization saved to: {args.save_plot}")


if __name__ == "__main__":
    main()
