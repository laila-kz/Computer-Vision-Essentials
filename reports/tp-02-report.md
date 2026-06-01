# TP 2: Histogram Processing and Edge Detection

## Objective

Analyze contrast, histogram distribution, and edge structure using classical enhancement and gradient operators.

## Theoretical Background

Histogram equalization redistributes intensities to improve contrast. Edge detection highlights strong local changes in intensity, which often correspond to object boundaries.

## Methodology

1. Compute the histogram of the input image.
2. Apply contrast enhancement and histogram equalization.
3. Smooth the image if necessary.
4. Extract edges using Sobel, Laplacian, and Canny.
5. Compare the outputs and discuss the differences.

## Implementation

- MATLAB solution: [src/matlab/tp2_histogram_edges.m](../src/matlab/tp2_histogram_edges.m)
- OpenCV solution: [src/python/cv_portfolio/edges.py](../src/python/cv_portfolio/edges.py)

## Results

The enhanced image should reveal more detail, and the edge maps should show the impact of derivative-based operators and thresholds.

## Analysis

Global enhancement can improve visibility, but edges remain sensitive to noise and threshold selection. This makes smoothing and parameter tuning important.

## Conclusion

This TP connects intensity analysis with structural analysis and demonstrates the transition from enhancement to local feature extraction.
