# Artificial Vision / Computer Vision 



## Project Overview

Computer Vision is the field of building systems that can interpret images and video. It sits at the intersection of image processing, pattern recognition, geometry, and machine learning. In practice, a Computer Vision pipeline takes pixels as input and produces meaning: boundaries, objects, measurements, labels, or decisions.

Image Processing is the lower-level foundation of that pipeline. It focuses on improving, analyzing, and transforming images through operations such as filtering, histogram processing, edge detection, morphology, and segmentation.

The objectives of this course-based portfolio are to:

- consolidate the theoretical concepts covered in the lectures,
- reconstruct each TP as a professional solution and report,
- demonstrate MATLAB and OpenCV implementation skills,
- provide reusable code and mini-projects for a public GitHub profile,
- show progression from fundamentals to practical vision applications.

## Technologies Used

- MATLAB for classical image processing workflows, visualization, and algorithm prototyping.
- Python for reusable Computer Vision code organization.
- OpenCV for image processing, feature extraction, segmentation, and detection.
- NumPy for numerical operations.
- Matplotlib for analysis and visual reporting.

## Skills Acquired

### MATLAB

- Image import, display, and format conversion.
- Intensity normalization and contrast enhancement.
- Histogram analysis and equalization.
- Spatial filtering and convolution-based processing.
- Edge detection with classical operators.
- Binary morphology and connected components.
- Segmentation workflows and region analysis.
- Feature extraction and matching.
- Writing reproducible scripts and local helper functions.

### OpenCV

- Reading, preprocessing, and visualizing images.
- Color space conversion and channel analysis.
- Histogram processing and contrast enhancement.
- Gaussian, median, Sobel, Laplacian, and Canny filtering.
- Thresholding, morphology, and connected component analysis.
- Contour analysis and shape-based reasoning.
- Feature detection and descriptor matching with ORB.
- Face detection using Haar cascades.
- Modular Python code design for reusable vision workflows.

### Image Processing

- Sampling, quantization, and image representation.
- Intensity transformations and dynamic range handling.
- Histogram computation and equalization.
- Smoothing, sharpening, and denoising.
- Edge detection and gradient estimation.
- Morphological opening, closing, erosion, and dilation.
- Segmentation by thresholding and region analysis.
- Feature extraction and geometric descriptors.

### Computer Vision

- From pixel-level operations to semantic interpretation.
- Object localization and detection concepts.
- Feature-based matching and correspondence.
- Shape reasoning and region statistics.
- Pipeline design for robust visual analysis.
- Practical trade-offs between accuracy, speed, and interpretability.

### Mathematical Foundations

- Convolution: $(f * g)(x,y) = \sum_i \sum_j f(i,j)g(x-i,y-j)$
- Gradient magnitude: $\|\nabla I\| = \sqrt{G_x^2 + G_y^2}$
- Histogram normalization: $p(i) = \frac{n_i}{N}$
- Thresholding: $g(x,y) = 1$ if $f(x,y) \ge T$, else $0$
- Otsu criterion: maximize between-class variance
- Morphological dilation and erosion using a structuring element
- Distance transform and watershed segmentation concepts
- Feature matching by descriptor distance and ratio tests

## Repository Structure

```text
docs/                      Theory notes, dependency map, and portfolio notes
reports/                   Professional TP reports
src/python/cv_portfolio/   Reusable OpenCV implementations
src/matlab/                MATLAB scripts aligned with the same topics
```

## Learning Showcase

1. Problem: image data is noisy, low contrast, or poorly structured.
2. Theory: use histograms, filtering, gradients, and morphology to analyze the signal.
3. Implementation: reproduce the methods in MATLAB and OpenCV.
4. Results: compare outputs using consistent metrics and visual evidence.
5. Real-world applications: document how the same ideas support inspection, detection, and recognition.


