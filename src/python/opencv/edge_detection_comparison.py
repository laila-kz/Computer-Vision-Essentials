"""Edge Detection Comparison Demo Script.

Compares classical 1st- and 2nd-order spatial gradient and edge operators:
- Sobel Magnitude
- Prewitt Magnitude
- Scharr Gradient
- Laplacian of Gaussian (LoG)
- Canny Edge Detector with Hysteresis

Usage:
    python edge_detection_comparison.py [--image path/to/image.png] [--save-plot output.png] [--show]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow execution from repository root or subfolder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_portfolio.filtering.edges import compare_edge_detectors
from cv_portfolio.utils.io import generate_synthetic_image, read_image
from cv_portfolio.visualization.plotting import plot_image_grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare classical 1st and 2nd derivative edge detectors with visual outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image. If omitted, a procedural test pattern is generated.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="outputs/edge_detection_comparison.png",
        help="Path to save the multi-panel Matplotlib comparison plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the Matplotlib comparison figure interactively.",
    )
    args = parser.parse_args()

    if args.image and Path(args.image).exists():
        print(f"[+] Loading input image: {args.image}")
        image = read_image(args.image)
    else:
        print("[!] No image supplied or file not found. Generating synthetic geometric test pattern...")
        image = generate_synthetic_image(pattern="shapes", noise_level=12.0)

    print("[*] Computing edge maps (Sobel, Prewitt, Scharr, Laplacian, LoG, Canny)...")
    edge_outputs = compare_edge_detectors(image)

    # Prepare visual dictionary
    grid_images = {"Original Image": image}
    grid_images.update({f"{name.upper()} Edges": edge_map for name, edge_map in edge_outputs.items()})

    for name, edge_map in edge_outputs.items():
        edge_density = (edge_map > 0).mean() * 100.0
        print(f"  -> {name.upper():<12} shape={edge_map.shape} | active pixel density={edge_density:.2f}%")

    if args.save_plot or args.show:
        plot_image_grid(
            images=grid_images,
            title="Classical Edge Detection & Gradient Operators Comparison",
            ncols=3,
            save_path=args.save_plot,
            show=args.show,
        )
        if args.save_plot:
            print(f"[+] Multi-panel edge comparison figure saved to: {args.save_plot}")


if __name__ == "__main__":
    main()
