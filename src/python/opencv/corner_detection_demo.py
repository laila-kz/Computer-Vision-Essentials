"""Corner & Interest Point Detection Demo Script.

Demonstrates classical interest point localization algorithms:
- Harris Corner Detector ($R = \\det(M) - k \\cdot \\mathrm{trace}(M)^2$)
- Shi-Tomasi Good Features to Track ($R = \\min(\\lambda_1, \\lambda_2)$)
- FAST (Features from Accelerated Segment Test)

Usage:
    python corner_detection_demo.py [--image path/to/image.png] [--save-plot output.png] [--show]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow execution from root or subfolder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_portfolio.features.corners import fast_corners, harris_corners, shi_tomasi_corners
from cv_portfolio.utils.io import generate_synthetic_image, read_image
from cv_portfolio.visualization.plotting import plot_image_grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Harris, Shi-Tomasi, and FAST corner interest point detectors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image. If omitted, generates synthetic checkerboard/shapes pattern.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="outputs/corner_detection_comparison.png",
        help="Path to save multi-panel corner comparison plot.",
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
        print("[!] Generating synthetic geometric test pattern...")
        image = generate_synthetic_image(pattern="shapes", noise_level=3.0)

    print("[*] Detecting corners using Harris, Shi-Tomasi, and FAST algorithms...")
    harris_resp, harris_annotated = harris_corners(image)
    shi_pts, shi_annotated = shi_tomasi_corners(image, max_corners=100)
    fast_kps, fast_annotated = fast_corners(image, threshold=25)

    print(f"  -> Shi-Tomasi corners detected: {len(shi_pts)}")
    print(f"  -> FAST keypoints detected:     {len(fast_kps)}")

    grid_images = {
        "Original Image": image,
        "Harris Corner Response (Red)": harris_annotated,
        f"Shi-Tomasi Good Features ({len(shi_pts)} pts)": shi_annotated,
        f"FAST Interest Points ({len(fast_kps)} kps)": fast_annotated,
    }

    if args.save_plot or args.show:
        plot_image_grid(
            images=grid_images,
            title="Corner & Interest Point Detection: Harris vs. Shi-Tomasi vs. FAST",
            ncols=2,
            figsize=(12, 9),
            save_path=args.save_plot,
            show=args.show,
        )
        if args.save_plot:
            print(f"[+] Corner comparison visualization saved to: {args.save_plot}")


if __name__ == "__main__":
    main()
