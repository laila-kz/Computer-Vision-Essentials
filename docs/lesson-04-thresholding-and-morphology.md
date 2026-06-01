# Lesson 4: Thresholding and Morphological Operations

## Objective

Learn how to convert images into binary masks and refine them with morphology for object cleanup and structural analysis.

## Theory

Thresholding separates foreground from background using intensity criteria. Morphological operations then improve the mask using a structuring element to expand, shrink, clean, or connect regions.

## Mathematical Background

- Binary threshold: $g(x,y)=1$ if $f(x,y)\ge T$, otherwise $0$
- Dilation: $A \oplus B$
- Erosion: $A \ominus B$
- Opening: $(A \ominus B) \oplus B$
- Closing: $(A \oplus B) \ominus B$

## Algorithms

- Global thresholding
- Otsu thresholding
- Erosion and dilation
- Opening and closing
- Hole filling and small-object removal

## Applications

- Document analysis
- Industrial counting
- Medical mask cleanup
- Shape extraction

## Advantages

- Fast and practical.
- Strong when objects differ clearly from the background.

## Limitations

- Sensitive to illumination variation.
- Binary decisions may lose fine detail.

## MATLAB Implementation

- Use `imbinarize`, `graythresh`, `imerode`, `imdilate`, `imopen`, and `imclose`.

## OpenCV Implementation

- Use `cv2.threshold`, `cv2.morphologyEx`, `cv2.erode`, and `cv2.dilate`.

## Key Takeaways

- Masks are the gateway to object-level analysis.
- Morphology makes thresholded results usable.
- Structuring-element design matters.
