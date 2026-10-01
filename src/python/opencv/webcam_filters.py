"""Webcam & Live Stream Filter Suite Demo Script.

Demonstrates real-time image filtering transforms suitable for video pipelines:
- High-contrast Grayscale
- Edge Boundary Isolation (Canny)
- Hand-drawn Artistic Pencil Sketch (Dodge color blend)
- Edge-preserving Unsharp Mask Sharpening
- Cel-shaded Cartoonization

Usage:
    python webcam_filters.py [--frame-image path/to/frame.png] [--save-plot output.png] [--show]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow execution from root or subfolder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_portfolio.utils.io import generate_synthetic_image, read_image
from cv_portfolio.visualization.plotting import plot_image_grid
from cv_portfolio.webcam_filters import build_webcam_filter_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate real-time video stream filter transformations on single frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--frame-image",
        type=str,
        default=None,
        help="Path to frame image. If omitted, generates synthetic test frame.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="outputs/webcam_filters_montage.png",
        help="Path to save multi-panel filter gallery plot.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively.",
    )
    args = parser.parse_args()

    if args.frame_image and Path(args.frame_image).exists():
        print(f"[+] Loading input frame: {args.frame_image}")
        frame = read_image(args.frame_image)
    else:
        print("[!] No frame supplied. Generating synthetic video test frame...")
        frame = generate_synthetic_image(pattern="shapes", noise_level=4.0)

    print("[*] Processing live filter transformations (Grayscale, Canny, Sketch, Sharpen, Cartoon)...")
    filter_outputs = build_webcam_filter_map(frame)

    for name, filtered_frame in filter_outputs.items():
        print(f"  -> Filter [{name.upper():<10}] rendered successfully (shape: {filtered_frame.shape})")

    if args.save_plot or args.show:
        plot_image_grid(
            images=filter_outputs,
            title="Real-Time Video Filter Transformation Gallery",
            ncols=3,
            save_path=args.save_plot,
            show=args.show,
        )
        if args.save_plot:
            print(f"[+] Filter gallery figure saved to: {args.save_plot}")


if __name__ == "__main__":
    main()
