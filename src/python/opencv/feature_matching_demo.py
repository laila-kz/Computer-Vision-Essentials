"""ORB Feature Detection & Descriptor Matching Demo Script.

Demonstrates 2D feature extraction and correspondence matching:
- Multi-scale ORB (Oriented FAST and Rotated BRIEF) keypoint extraction
- 256-bit Binary Descriptor Computation
- Brute-Force Matcher with Hamming Distance
- Lowe's Ratio Test & RANSAC Homography inlier verification

Usage:
    python feature_matching_demo.py [--image-a path/to/a.png] [--image-b path/to/b.png] [--save-plot output.png] [--show]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

# Allow execution from root or subfolder
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_portfolio.features.matching import (
    match_features_ratio_test,
    match_orb_features,
    orb_keypoints_and_descriptors,
)
from cv_portfolio.utils.io import generate_synthetic_image, read_image
from cv_portfolio.visualization.plotting import plot_image_grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ORB keypoints and establish feature correspondences with ratio test & RANSAC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-a",
        type=str,
        default=None,
        help="Path to query image A. If omitted, generates synthetic reference pattern.",
    )
    parser.add_argument(
        "--image-b",
        type=str,
        default=None,
        help="Path to target image B. If omitted, applies geometric rotation/affine transform on pattern A.",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=50,
        help="Maximum top feature matches to display.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="outputs/feature_matching_demo.png",
        help="Path to save correspondence visualization montage.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively.",
    )
    args = parser.parse_args()

    if args.image_a and Path(args.image_a).exists():
        print(f"[+] Loading image A: {args.image_a}")
        img_a = read_image(args.image_a)
    else:
        print("[!] Generating synthetic pattern for Image A...")
        img_a = generate_synthetic_image(pattern="shapes", noise_level=5.0)

    if args.image_b and Path(args.image_b).exists():
        print(f"[+] Loading image B: {args.image_b}")
        img_b = read_image(args.image_b)
    else:
        print("[!] Synthesizing Image B via affine rotation (15 deg) + scaling...")
        h, w = img_a.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle=15, scale=0.9)
        img_b = cv2.warpAffine(img_a, rot_mat, (w, h), borderValue=(40, 40, 40))

    print("[*] Detecting ORB keypoints and extracting 256-bit binary descriptors...")
    kps_a, desc_a = orb_keypoints_and_descriptors(img_a, nfeatures=1000)
    kps_b, desc_b = orb_keypoints_and_descriptors(img_b, nfeatures=1000)
    print(f"  -> Image A: {len(kps_a)} keypoints detected")
    print(f"  -> Image B: {len(kps_b)} keypoints detected")

    print("[*] Performing Hamming Distance Brute-Force matching with Lowe's Ratio Test & RANSAC...")
    result = match_features_ratio_test(img_a, img_b, ratio_threshold=0.75)
    print(f"[✓] Validated Matches: {len(result.matches)} pairs")

    if result.homography is not None:
        print("[✓] RANSAC Homography Matrix successfully estimated:")
        for row in result.homography:
            print(f"     [ {row[0]:8.4f}  {row[1]:8.4f}  {row[2]:8.4f} ]")

    # Visual keypoint overlays
    kp_img_a = cv2.drawKeypoints(img_a, kps_a, None, color=(0, 255, 0), flags=0)
    kp_img_b = cv2.drawKeypoints(img_b, kps_b, None, color=(0, 255, 0), flags=0)

    grid_images = {
        f"Image A: Keypoints (N={len(kps_a)})": kp_img_a,
        f"Image B: Keypoints (N={len(kps_b)})": kp_img_b,
        f"Matched Feature Correspondences (Matches={len(result.matches)})": result.matched_image,
    }

    if args.save_plot or args.show:
        plot_image_grid(
            images=grid_images,
            title="ORB Feature Detection, Descriptor Extraction & Robust Matching",
            ncols=2,
            figsize=(14, 10),
            save_path=args.save_plot,
            show=args.show,
        )
        if args.save_plot:
            print(f"[+] Feature matching visualization saved to: {args.save_plot}")


if __name__ == "__main__":
    main()
