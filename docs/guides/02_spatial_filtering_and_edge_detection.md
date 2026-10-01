# Study Guide 02: Spatial Filtering, Convolution & Edge Detection

## 1. 2D Discrete Spatial Convolution

Spatial filtering modifies an image by sliding a discrete $k \times k$ matrix (kernel / mask) $h(i, j)$ across the image grid $f(x, y)$:

$$g(x, y) = (f * h)(x, y) = \sum_{i=-a}^a \sum_{j=-b}^b f(x - i, y - j) \cdot h(i, j)$$

where $a = \lfloor k_h / 2 \rfloor$ and $b = \lfloor k_w / 2 \rfloor$.

### Linear Smoothing Filters

#### A. Box / Mean Filter
Uniform average over a neighborhood window:

$$h_{\mathrm{box}} = \frac{1}{k^2} \begin{bmatrix} 1 & \dots & 1 \\ \vdots & \ddots & \vdots \\ 1 & \dots & 1 \end{bmatrix}$$

#### B. 2D Isotropic Gaussian Filter
Continuous density function with standard deviation $\sigma$:

$$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left( -\frac{x^2 + y^2}{2\sigma^2} \right)$$

- **Separability property:** $G(x, y) = G(x) \cdot G(y)$. Reduces 2D convolution complexity from $\mathcal{O}(k^2 \cdot N)$ to two 1D passes with complexity $\mathcal{O}(2k \cdot N)$.
- Minimizes spatial ringing and preserves rotational symmetry.

#### C. Non-Linear Bilateral Filter (Edge-Preserving)
Combines geometric domain kernel $g_s$ with radiometric range kernel $g_r$:

$$I_{\mathrm{bilateral}}(p) = \frac{1}{W_p} \sum_{q \in S} I(q) \cdot \exp\left( -\frac{\|p - q\|^2}{2\sigma_s^2} \right) \cdot \exp\left( -\frac{|I(p) - I(q)|^2}{2\sigma_r^2} \right)$$

---

## 2. Gradient Operators & Differential Edge Detection

Edges correspond to localized extrema of first-order derivatives or zero-crossings of second-order derivatives.

```
Intensity Profile f(x):        ____/''''''''
1st Derivative f'(x):          ___/^\_______  (Peak marks edge boundary)
2nd Derivative f''(x):         __/^\_v/_____  (Zero-crossing marks exact edge)
```

### Spatial Gradients Vector
$$\nabla f = \begin{bmatrix} G_x \\ G_y \end{bmatrix} = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix}$$

- **Gradient Magnitude:**
  $$M(x, y) = \|\nabla f\| = \sqrt{G_x^2 + G_y^2} \approx |G_x| + |G_y|$$

- **Gradient Direction (Normal to edge orientation):**
  $$\theta(x, y) = \mathrm{arctan2}(G_y, G_x) \in [-\pi, \pi]$$

### Standard Differentiation Kernels

#### Sobel Operator (Derivative + Smoothing)
$$S_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \quad S_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}$$

#### Scharr Operator (Enhanced Rotational Accuracy)
$$K_x = \begin{bmatrix} -3 & 0 & +3 \\ -10 & 0 & +10 \\ -3 & 0 & +3 \end{bmatrix}, \quad K_y = \begin{bmatrix} -3 & -10 & -3 \\ 0 & 0 & 0 \\ +3 & +10 & +3 \end{bmatrix}$$

#### Laplacian (Isotropic 2nd Derivative)
$$\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}, \quad L = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix} \quad \text{or} \quad \begin{bmatrix} 1 & 1 & 1 \\ 1 & -8 & 1 \\ 1 & 1 & 1 \end{bmatrix}$$

---

## 3. Canny Edge Detection Pipeline

John Canny (1986) formulated three optimal criteria:
1. **Low error rate:** Zero spurious edges and no missed actual edges.
2. **Localization:** Minimal distance between detected and real edge centers.
3. **Single response:** Exactly one response per true edge (no thick boundaries).

```mermaid
flowchart LR
    A["Raw Image"] --> B["Gaussian Blur (Noise Filtering)"]
    B --> C["Sobel Gradients (Magnitude & Angle)"]
    C --> D["Non-Maximum Suppression (1-px Thinning)"]
    D --> E["Double Thresholding (T_low, T_high)"]
    E --> F["Hysteresis Edge Tracking"]
    F --> G["Final Binary Edge Map"]
```

### Stage Details:
1. **Gaussian Smoothing:** Attenuates sensor noise ($I_\sigma = I * G_\sigma$).
2. **Gradient Estimation:** Computes $M(x, y)$ and $\theta(x, y)$. Quantizes $\theta$ into 4 primary directions: $0^\circ, 45^\circ, 90^\circ, 135^\circ$.
3. **Non-Maximum Suppression (NMS):** Compares $M(x, y)$ with its two neighbors along gradient normal $\theta$. If $M(x, y)$ is not strictly greater than both neighbors, it is suppressed to zero ($M \leftarrow 0$).
4. **Hysteresis Thresholding:**
   - $M(x, y) \ge T_{\mathrm{high}}$: Classified as **Strong Edge**.
   - $T_{\mathrm{low}} \le M(x, y) < T_{\mathrm{high}}$: Classified as **Weak Edge Candidate**.
   - $M(x, y) < T_{\mathrm{low}}$: Suppressed (**Non-edge**).
5. **Connectivity Tracking:** A weak edge is preserved if and only if it is connected to a strong edge via 8-connectivity.

---

## 4. Edge Operator Benchmark & Trade-Offs

| Operator | Derivative Order | Noise Robustness | Localization Precision | Output Format |
| :--- | :--- | :--- | :--- | :--- |
| **Roberts** | 1st | Low | Moderate | Continuous gradient |
| **Prewitt** | 1st | Moderate | Moderate | Continuous gradient |
| **Sobel** | 1st | Good | Good | Continuous gradient |
| **Scharr** | 1st | Good | High (Rotational) | Continuous gradient |
| **LoG** | 2nd | High (Gaussian pre-pass) | High (Zero-crossings) | Continuous / Signed |
| **Canny** | Multi-stage | Superior | Optimal (1-pixel thinned) | Clean Binary Edge Map |
