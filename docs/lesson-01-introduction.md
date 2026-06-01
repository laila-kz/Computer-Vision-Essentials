# Lesson 1: Introduction to Computer Vision and Image Representation

## Objective

Introduce the role of Computer Vision, distinguish it from Image Processing, and establish the way images are represented numerically for later algorithms.

## Theory

Computer Vision seeks to infer meaning from visual data. Image Processing is the set of operations that prepare, transform, or analyze images before higher-level understanding is attempted. This lesson covers the image as a 2D signal, pixel coordinates, intensity values, and the difference between grayscale and color representations.

## Mathematical Background

- A grayscale image can be treated as a function $I(x,y)$.
- A digital image is a sampled and quantized version of a continuous scene.
- RGB images can be represented as three channels $R$, $G$, and $B$.

## Algorithms

- Image loading and display
- Channel splitting and merging
- Intensity normalization
- Basic pixel-wise arithmetic

## Applications

- Inspection systems
- Medical image viewing
- Preprocessing for detection and segmentation

## Advantages

- Provides the vocabulary used by all later CV methods.
- Makes the rest of the pipeline easier to reason about.

## Limitations

- Does not solve interpretation on its own.
- Raw pixels are often noisy and poorly scaled for analysis.

## MATLAB Implementation

- Use `imread`, `imshow`, `rgb2gray`, and direct matrix arithmetic.
- Show channel visualization and normalization with `mat2gray`.

## OpenCV Implementation

- Use `cv2.imread`, `cv2.cvtColor`, `cv2.split`, `cv2.merge`, and `cv2.normalize`.

## Key Takeaways

- Images are signals, not just pictures.
- Good preprocessing begins with correct representation.
- Every later algorithm depends on this layer.
