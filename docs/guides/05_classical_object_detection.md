# Study Guide 05: Classical Object Detection & Template Matching

## 1. Viola-Jones Object Detection Framework

Published by Paul Viola and Michael Jones (2001), this algorithm pioneered real-time face detection through three core algorithmic innovations:

```mermaid
flowchart LR
    A["Input Image"] --> B["Integral Image Transformation (O(1) lookups)"]
    B --> C["Haar-like Feature Computation"]
    C --> D["AdaBoost Classifier Selection"]
    D --> E["Attentional Cascade Filtering"]
    E --> F["Face Bounding Box Detections"]
```

### A. The Integral Image (Summed-Area Table)
Enables constant-time $\mathcal{O}(1)$ evaluation of rectangular pixel intensity sums regardless of bounding box scale:

$$II(x, y) = \sum_{x' \le x, y' \le y} I(x', y')$$

Given any rectangle $D = [x_1, y_1, x_2, y_2]$:

$$\sum_{(x, y) \in D} I(x, y) = II(x_2, y_2) + II(x_1 - 1, y_1 - 1) - II(x_1 - 1, y_2) - II(x_2, y_1 - 1)$$

### B. Haar-Like Features
Calculates intensity differentials between adjacent white and black rectangular sub-regions:

$$f_i = \sum_{\mathrm{White\ Pixels}} I(x, y) - \sum_{\mathrm{Black\ Pixels}} I(x, y)$$

Captures structural properties such as eyes being darker than the forehead/cheeks, or the nasal bridge being brighter than lateral valleys.

### C. Attentional Cascade Classifier
Arranges classifiers in a degenerated decision tree:
- **Stage 1-3:** Extremely lightweight classifiers with 1-10 Haar features. Reject $>50\%$ of non-face background windows in sub-microsecond time.
- **Deeper Stages (20+):** Increasingly complex AdaBoost ensembles evaluate only the minority of windows surviving preliminary stages.

---

## 2. Template Matching

Finds occurrences of an exemplar patch $T(x', y')$ of size $w \times h$ within search image $I(x, y)$ of size $W \times H$.

### Correlation Metrics

| Method | Metric Name | Mathematical Formula | Optimal Match |
| :--- | :--- | :--- | :--- |
| `TM_SQDIFF_NORMED` | Normalized Sum of Squared Differences | $R(x, y) = \frac{\sum (T - I)^2}{\sqrt{\sum T^2 \sum I^2}}$ | Minimum ($R = 0$) |
| `TM_CCORR_NORMED` | Normalized Cross-Correlation | $R(x, y) = \frac{\sum (T \cdot I)}{\sqrt{\sum T^2 \sum I^2}}$ | Maximum ($R = 1$) |
| `TM_CCOEFF_NORMED` | Normalized Zero-Mean Cross-Correlation | $R(x, y) = \frac{\sum (T' \cdot I')}{\sqrt{\sum T'^2 \sum I'^2}}$ | Maximum ($R = 1$) |

> **Note:** `TM_CCOEFF_NORMED` subtracts mean values ($T' = T - \bar{T}, I' = I - \bar{I}$), making it invariant to linear brightness and contrast shifts.

---

## 3. Classical vs. Deep Learning Object Detection

| Property | Classical Vision (Haar, HOG + SVM) | Deep Learning (YOLO, Faster R-CNN) |
| :--- | :--- | :--- |
| **Feature Extraction** | Handcrafted geometric / statistical features | End-to-end learned convolutional representations |
| **Compute Requirements** | Runs on CPU / embedded microcontrollers | Requires GPU / NPU hardware acceleration |
| **Data Requirements** | Few samples / lightweight cascades | Thousands of annotated training bounding boxes |
| **Occlusion / Pose Robustness** | Sensitive to out-of-plane rotation | High invariance to viewpoint, lighting, and occlusion |
| **Inference Latency** | Ultra-low memory footprint ($< 10$ MB) | High memory footprint ($50 - 500$ MB) |
