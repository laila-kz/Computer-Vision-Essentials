# Portfolio Mini-Projects

These mini-projects are selected to showcase the strongest classical Computer Vision topics from the course.

## Image Enhancement Toolkit

- Problem: images are dark, low-contrast, or unevenly lit.
- Theory: histogram equalization, CLAHE, gamma correction, and contrast stretching.
- Implementation: compare multiple enhancement strategies on the same input.
- Results: display before/after outputs and histogram changes.
- Real-world applications: inspection images, scanned documents, and low-light scenes.

## Edge Detection Comparison

- Problem: boundaries are difficult to isolate directly from noisy images.
- Theory: gradient operators, smoothing, and thresholding.
- Implementation: compare Sobel, Laplacian, and Canny outputs.
- Results: highlight robustness versus detail sensitivity.
- Real-world applications: defect detection and shape analysis.

## Histogram Analysis Tool

- Problem: intensity distribution is hard to inspect visually.
- Theory: histograms, cumulative distributions, and normalization.
- Implementation: generate image histograms and equalized versions.
- Results: explain how contrast changes the signal distribution.
- Real-world applications: quality control and preprocessing audits.

## Image Segmentation Demo

- Problem: objects must be separated from the background.
- Theory: thresholding, morphology, and connected components.
- Implementation: build a pipeline that cleans masks and labels regions.
- Results: count objects and measure region statistics.
- Real-world applications: sorting, counting, and inspection.

## Object Counting Application

- Problem: the number of items in a scene must be estimated reliably.
- Theory: segmentation, connected components, and contour filtering.
- Implementation: count objects after cleanup and reject small noise.
- Results: report object count and region measurements.
- Real-world applications: manufacturing and inventory analysis.

## Real-Time Webcam Filters

- Problem: interactive processing is useful for demos and interviews.
- Theory: frame-level transforms and performance trade-offs.
- Implementation: apply grayscale, blur, edge, and sketch filters to live frames.
- Results: show fast, visible transformations.
- Real-world applications: live demos and prototyping.

## Face Detection Project

- Problem: locate human faces in images or video frames.
- Theory: detection cascades and region scanning.
- Implementation: use OpenCV Haar cascades for face rectangles.
- Results: draw bounding boxes and count detections.
- Real-world applications: access control and media analysis.

## Feature Matching Demo

- Problem: identify correspondence between two images.
- Theory: keypoints, descriptors, and descriptor distances.
- Implementation: detect ORB features and match them with a ratio or distance filter.
- Results: visualize correspondences and match quality.
- Real-world applications: stitching, localization, and registration.
