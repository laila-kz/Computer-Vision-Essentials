# Study Guide 01: Image Fundamentals & Intensity Enhancement

## 1. Mathematical Representation of Digital Images

A continuous 2D scene irradiance $f_c(x, y)$ is converted into a discrete digital image matrix $I[r, c]$ through **spatial sampling** and **radiometric quantization**:

$$I \in \mathbb{R}^{H \times W \times C}, \quad I(r, c) \in \{0, 1, \dots, 2^b - 1\}$$

where:
- $H$ is the image height (rows, $r \in [0, H-1]$).
- $W$ is the image width (columns, $c \in [0, W-1]$).
- $C$ is the channel depth ($C=1$ for grayscale, $C=3$ for RGB/BGR).
- $b$ is the bit depth ($b=8$ gives $L = 256$ intensity levels $[0, 255]$).

### Grayscale Conversion (ITU-R BT.601)
Human visual perception is unequally sensitive to spectral wavelengths (cones peak at green). Grayscale luminance $Y$ is weighted:

$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

---

## 2. Intensity Transformation Functions

Point processing operations map an input intensity $r \in [0, L-1]$ to an output intensity $s = T(r)$ without spatial neighborhood context.

```
                      Transformation Curves
       255 +----------------------------------------+
           |                                  ..../ |
           |                           ....'''      |  Gamma < 1 (Brighten)
           |                     ...'''             |
           |                 ..''                   |
   s   128 |              ./                        |  Linear (Identity)
           |            .'  ''..                    |
           |         ./         '''...              |
           |       ./                 '''...        |  Gamma > 1 (Darken)
           |    ./                          ''''... |
         0 +----------------------------------------+
           0                  128                 255
                                r
```

### A. Linear Dynamic Range Stretching (Min-Max Normalization)
Maps an arbitrary compressed dynamic range $[r_{min}, r_{max}]$ to full 8-bit range $[0, 255]$:

$$s = \mathrm{clip}\left( \frac{r - r_{min}}{r_{max} - r_{min}} \times 255, 0, 255 \right)$$

### B. Power-Law (Gamma) Transformation
Addresses non-linear capture/display characteristics (Cathode Ray Tube / monitor gamma):

$$s = c \cdot r^\gamma, \quad r \in [0, 1]$$

- **$\gamma < 1$ (e.g., 0.5 - 0.7):** Expands dark regions, lifting shadows and underexposed details.
- **$\gamma > 1$ (e.g., 1.5 - 2.2):** Compresses dark tones, suppressing background haze and high-exposure washouts.

### C. Logarithmic Compression
Compresses vast dynamic ranges (e.g., Fourier transform power spectra where DC components dominate by orders of magnitude):

$$s = c \cdot \ln(1 + |r|)$$

---

## 3. Histogram Analysis & Equalization

Let $n_k$ be the number of pixels with intensity $r_k$. The discrete **Probability Density Function (PDF)** is:

$$p(r_k) = \frac{n_k}{N}, \quad \sum_{k=0}^{L-1} p(r_k) = 1$$

where $N = H \times W$ is the total pixel count.

### Cumulative Distribution Function (CDF)
$$S(r_k) = \sum_{j=0}^k p(r_j)$$

### Global Histogram Equalization Formulation
Histogram Equalization derives a monotonic transformation $T(r)$ that transforms an arbitrary input distribution into a uniform probability density:

$$s_k = T(r_k) = \mathrm{round}\left( (L - 1) \cdot \sum_{j=0}^k p(r_j) \right) = \mathrm{round}\left( (L - 1) \cdot S(r_k) \right)$$

### Contrast-Limited Adaptive Histogram Equalization (CLAHE)
Global equalization often over-amplifies background sensor noise in homogeneous regions. CLAHE resolves this:
1. Partitions the image into $M \times N$ contextual grid tiles (typically $8 \times 8$).
2. Computes the histogram for each tile.
3. Clips the histogram at a predetermined threshold $\beta_{clip}$ and redistributes excess uniformly across all bins.
4. Performs histogram equalization locally.
5. Employs bilinear interpolation across tile boundaries to eliminate blocking artifacts.

---

## 4. Method Comparison & Trade-Off Matrix

| Enhancement Algorithm | Computational Complexity | Contrast Improvement | Noise Sensitivity | Best Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Stretching** | $\mathcal{O}(N)$ | Moderate | Low | Images utilizing only a subset of $[0, 255]$ |
| **Gamma Correction** | $\mathcal{O}(N)$ (via LUT) | Smooth / Non-linear | Very Low | Global underexposure or overexposure |
| **Global Hist Eq** | $\mathcal{O}(N + L)$ | Maximal (Global) | High (amplifies noise) | Scientific imaging with multimodal histograms |
| **CLAHE** | $\mathcal{O}(N \cdot K^2)$ | High (Localized) | Controlled ($\beta_{clip}$) | Medical X-rays, underwater, variable illumination |
