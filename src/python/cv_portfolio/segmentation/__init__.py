"""Image segmentation, thresholding, morphological processing, and connected components."""

from .connected_components import (
    ComponentInfo,
    connected_components_summary,
    filter_components_by_area,
)
from .morphology import (
    blackhat_transform,
    dilate_image,
    erode_image,
    get_structuring_element,
    morphological_gradient,
    morphology_cleanup,
    morphology_close,
    morphology_open,
    tophat_transform,
)
from .pipeline import segmentation_pipeline
from .thresholding import (
    adaptive_threshold_gaussian,
    adaptive_threshold_mean,
    manual_threshold,
    otsu_threshold,
)
from .watershed import distance_transform_watershed

__all__ = [
    "manual_threshold",
    "otsu_threshold",
    "adaptive_threshold_mean",
    "adaptive_threshold_gaussian",
    "get_structuring_element",
    "erode_image",
    "dilate_image",
    "morphology_open",
    "morphology_close",
    "morphological_gradient",
    "tophat_transform",
    "blackhat_transform",
    "morphology_cleanup",
    "ComponentInfo",
    "connected_components_summary",
    "filter_components_by_area",
    "distance_transform_watershed",
    "segmentation_pipeline",
]
