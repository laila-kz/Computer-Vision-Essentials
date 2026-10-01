"""Publication-quality visualization and diagnostic plotting using Matplotlib and OpenCV."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from ..utils.conversions import to_rgb


def plot_image_grid(
    images: Dict[str, np.ndarray],
    title: str = "Computer Vision Pipeline Visualizer",
    ncols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Render a dictionary of labeled images in a multi-panel visual grid.

    Args:
        images: Mapping of subplot titles to image numpy arrays.
        title: Global super-title for the figure canvas.
        ncols: Number of grid columns.
        figsize: Optional (width, height) figure size in inches.
        save_path: Optional file path to save the generated figure.
        show: Whether to invoke plt.show() interactively.

    Returns:
        plt.Figure: Matplotlib figure instance.
    """
    n_images = len(images)
    nrows = (n_images + ncols - 1) // ncols
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).reshape(-1)

    for idx, (label, img) in enumerate(images.items()):
        ax = axes_flat[idx]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(to_rgb(img))
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axis("off")

    # Turn off unused subplot axes
    for idx in range(n_images, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_histogram_and_cdf(
    image: np.ndarray,
    title: str = "Histogram and Cumulative Distribution Function (CDF)",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot an image alongside its 256-bin intensity histogram and cumulative distribution function.

    Args:
        image: Grayscale or single-channel image.
        title: Plot title.
        save_path: Optional output file path.
        show: Whether to display interactively.

    Returns:
        plt.Figure: Generated figure.
    """
    fig, (ax_img, ax_hist) = plt.subplots(1, 2, figsize=(12, 4.5))

    if image.ndim == 2:
        ax_img.imshow(image, cmap="gray")
    else:
        ax_img.imshow(to_rgb(image))
    ax_img.set_title("Input Image", fontsize=11, fontweight="bold")
    ax_img.axis("off")

    flat = image.ravel()
    counts, bins, _ = ax_hist.hist(
        flat, bins=256, range=[0, 256], color="#2b5c8f", alpha=0.75, density=True, label="PDF (Histogram)"
    )
    ax_hist.set_xlabel("Pixel Intensity [0 - 255]", fontsize=10)
    ax_hist.set_ylabel("Probability Density", color="#2b5c8f", fontsize=10)
    ax_hist.set_xlim([0, 256])

    # Secondary twin axis for CDF
    ax_cdf = ax_hist.twinx()
    cdf = np.cumsum(counts) * (bins[1] - bins[0])
    ax_cdf.plot(bins[:-1], cdf, color="#d9534f", linewidth=2.0, label="CDF")
    ax_cdf.set_ylabel("Cumulative Probability", color="#d9534f", fontsize=10)
    ax_cdf.set_ylim([0.0, 1.05])

    ax_hist.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(p), dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig
