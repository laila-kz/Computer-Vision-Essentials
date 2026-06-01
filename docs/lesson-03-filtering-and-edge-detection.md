# Lesson 3: Filtering and Edge Detection

## Objective

Understand spatial filtering, denoising, sharpening, and edge detection as the main tools for extracting local structure.

## Theory

Filtering combines each pixel neighborhood with a kernel to suppress noise or emphasize local changes. Edge detection uses derivatives to find rapid intensity transitions that often correspond to object boundaries.

## Mathematical Background

- Convolution: $(f * g)(x,y) = \sum_i \sum_j f(i,j) g(x-i,y-j)$
- Gradient magnitude: $\sqrt{G_x^2 + G_y^2}$
- Laplacian: $\nabla^2 I = I_{xx} + I_{yy}$

## Algorithms

- Mean and Gaussian filtering
- Median filtering
- Sobel and Scharr gradients
- Laplacian edge response
- Canny edge detection

## Applications

- Boundary extraction
- Noise suppression
- Defect detection
- Pre-segmentation cleanup

## Advantages

- Classic operators are easy to interpret.
- They work well as building blocks in larger pipelines.

## Limitations

- Sensitive to parameter choice.
- Edges can be broken or noisy if thresholds are poor.

## MATLAB Implementation

- Use `imfilter`, `fspecial`, `medfilt2`, `edge`, and `imgaussfilt`.

## OpenCV Implementation

- Use `cv2.GaussianBlur`, `cv2.medianBlur`, `cv2.Sobel`, `cv2.Laplacian`, and `cv2.Canny`.

## Key Takeaways

- Filtering changes the signal before interpretation.
- Gradients reveal structure.
- Edge detection is a bridge between enhancement and segmentation.
