# Lesson 2: Image Enhancement and Histogram Processing

## Objective

Study contrast improvement, histogram analysis, and intensity transformations used to reveal hidden structure in images.

## Theory

Enhancement methods improve human visibility or prepare images for analysis. Histogram processing reveals how intensity values are distributed and supports operations such as equalization, stretching, and gamma correction.

## Mathematical Background

- Histogram bin count: $n_i$
- Probability form: $p(i) = n_i / N$
- Histogram equalization uses the cumulative distribution function.
- Gamma correction: $I_{out} = c I_{in}^{\gamma}$

## Algorithms

- Contrast stretching
- Global histogram equalization
- CLAHE-style local enhancement
- Gamma correction

## Applications

- Low-light image correction
- Industrial inspection
- Medical image enhancement

## Advantages

- Simple, fast, and interpretable.
- Often improves subsequent detection and segmentation.

## Limitations

- Over-enhancement can amplify noise.
- Global methods may fail on non-uniform illumination.

## MATLAB Implementation

- Use `imhist`, `histeq`, `imadjust`, and custom intensity transforms.

## OpenCV Implementation

- Use `cv2.equalizeHist`, CLAHE, and manual gamma/contrast transforms.

## Key Takeaways

- Histograms describe image tone.
- Contrast and illumination are often the first bottleneck.
- Enhancement is usually a preprocessing step, not the final goal.
