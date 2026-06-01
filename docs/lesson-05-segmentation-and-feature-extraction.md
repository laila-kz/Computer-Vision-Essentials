# Lesson 5: Segmentation and Feature Extraction

## Objective

Introduce image segmentation as a region-building task and feature extraction as the transition from pixels to measurable descriptors.

## Theory

Segmentation divides an image into meaningful regions. Feature extraction then summarizes those regions with contours, areas, moments, keypoints, or local descriptors that can be used for matching or classification.

## Mathematical Background

- Region properties: area, perimeter, centroid, solidity
- Moments: $m_{pq} = \sum_x \sum_y x^p y^q I(x,y)$
- Centroid: $(\bar{x},\bar{y}) = (m_{10}/m_{00}, m_{01}/m_{00})$

## Algorithms

- Connected components
- Contour extraction
- Region measurements
- Watershed-style separation
- ORB keypoint detection and descriptor extraction

## Applications

- Object counting
- Shape analysis
- Quality inspection
- Matching and recognition

## Advantages

- Produces compact and meaningful summaries.
- Bridges low-level vision and recognition.

## Limitations

- Segmentation errors propagate quickly.
- Feature quality depends on preprocessing.

## MATLAB Implementation

- Use `bwconncomp`, `regionprops`, `bwboundaries`, `detectORBFeatures`, and `extractFeatures`.

## OpenCV Implementation

- Use `connectedComponentsWithStats`, `findContours`, `cv2.ORB_create`, and descriptor matching.

## Key Takeaways

- Segmentation and features are core to interpretation.
- Good descriptors are compact but discriminative.
- Measurements are essential for engineering use cases.
