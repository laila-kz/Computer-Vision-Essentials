# TP 1: Image Fundamentals and Basic Operations

## Objective

Introduce image import, visualization, grayscale conversion, and point-wise transformations as the entry point to the Computer Vision pipeline.

## Theoretical Background

This TP focuses on the representation of images as numerical arrays. The key idea is that an image can be manipulated directly through pixel operations before moving to more advanced processing.

## Methodology

1. Load the source image.
2. Visualize the original image and its grayscale equivalent.
3. Normalize the intensity range.
4. Apply simple contrast and inversion transforms.
5. Compare the outputs visually.

## Implementation

- MATLAB solution: [src/matlab/tp1_image_basics.m](../src/matlab/tp1_image_basics.m)
- OpenCV solution: [src/python/cv_portfolio/enhancement.py](../src/python/cv_portfolio/enhancement.py)

## Results

The expected result is a clearer understanding of how pixel values behave under intensity transforms and why preprocessing matters before histogram and edge analysis.

## Analysis

Point-wise operations are simple but foundational. They are often the quickest way to improve readability or prepare an image for the next stage.

## Conclusion

This TP establishes the numerical view of images and prepares the learner for histogram processing, filtering, and segmentation.
