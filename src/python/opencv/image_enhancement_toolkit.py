"""Image Enhancement & Histogram Processing Toolkit Demo Script.

Demonstrates global and adaptive contrast enhancement techniques:
- Linear Contrast Stretching (Min-Max Scaling)
- Global Histogram Equalization
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Power-Law (Gamma) Corrections ($\\gamma = 0.6$ brightening, $\\gamma = 1.8$ darkening)
- Logarithmic Dynamic Range Compression

Usage:
    python image_enhancement_toolkit.py [--image path/to/image.png] [--save-plot output.png] [--show]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow execution from root or subfolder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_portfolio.preprocessing.enhancement import (
    build_enhancement_summary,
    compute_histogram_and_cdf,
)
from cv_portfolio.utils.io import generate_synthetic_image, read_image
from cv_portfolio.visualization.plotting import plot_histogram_and_cdf, plot_image_grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demonstrate linear, non-linear, and adaptive image contrast enhancement methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image. If omitted, generates low-contrast synthetic image.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="outputs/image_enhancement_summary.png",
        help="Path to save multi-panel enhancement comparison plot.",
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
        print("[!] No image supplied or file not found. Generating low-contrast gradient test image...")
        image = generate_synthetic_image(pattern="gradient", noise_level=5.0)

    print("[*] Generating enhancement suite (Contrast stretch, Hist Eq, CLAHE, Gamma, Log)...")
    summary = build_enhancement_summary(image)

    for name, img in summary.items():
        mean_val = float(img.mean())
        std_val = float(img.std())
        print(f"  -> {name:<18} | shape={img.shape} | mean={mean_val:6.2f} | std={std_val:6.2f}")

    if args.save_plot or args.show:
        plot_image_grid(
            images=summary,
            title="Image Enhancement & Dynamic Range Processing Suite",
            ncols=3,
            save_path=args.save_plot,
            show=args.show,
        )
        if args.save_plot:
            print(f"[+] Multi-panel enhancement figure saved to: {args.save_plot}")


if __name__ == "__main__":
    main()
