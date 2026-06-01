# TP 3: Thresholding, Morphology, and Segmentation

## Objective

Segment foreground objects from the background and refine the resulting binary mask using morphological tools.

## Theoretical Background

Thresholding creates a binary decision map. Morphological operators then remove noise, fill holes, and improve object connectivity.

## Methodology

1. Convert the image to grayscale.
2. Estimate a threshold or use Otsu's method.
3. Build a binary mask.
4. Clean the mask using erosion, dilation, opening, and closing.
5. Extract connected components and region statistics.

## Implementation

- MATLAB solution: [src/matlab/tp3_segmentation_morphology.m](../src/matlab/tp3_segmentation_morphology.m)
- OpenCV solution: [src/python/cv_portfolio/segmentation.py](../src/python/cv_portfolio/segmentation.py)

## Results

The output should contain a cleaner mask, separated objects, and measurable regions suitable for counting or inspection.

## Analysis

This TP shows why binary segmentation is useful but also fragile. Illumination and object overlap are the main practical challenges.

## Conclusion

Thresholding and morphology convert pixel-level intensity data into object-level information.
