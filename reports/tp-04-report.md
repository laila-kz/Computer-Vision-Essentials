# TP 4: Feature Extraction and Object Detection

## Objective

Extract meaningful descriptors from images and use them for matching, localization, or detection.

## Theoretical Background

Features summarize images with compact and discriminative measurements. Keypoints and descriptors support matching, while detection frameworks localize specific targets.

## Methodology

1. Detect keypoints or contours.
2. Compute descriptors or region statistics.
3. Match corresponding features or identify objects.
4. Visualize the detections and matches.

## Implementation

- MATLAB solution: [src/matlab/tp4_feature_matching_detection.m](../src/matlab/tp4_feature_matching_detection.m)
- OpenCV solution: [src/python/cv_portfolio/feature_matching.py](../src/python/cv_portfolio/feature_matching.py)

## Results

The expected output includes matched points, detected regions, or localized objects depending on the selected source image pair.

## Analysis

Feature extraction is the bridge from preprocessing to recognition. Good feature quality directly improves detection reliability.

## Conclusion

This TP closes the classical vision loop by connecting preprocessing, descriptors, and practical object-level interpretation.
