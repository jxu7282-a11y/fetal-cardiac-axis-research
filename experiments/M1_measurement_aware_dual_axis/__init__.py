"""M1 measurement-aware dual-axis components (stage 1: dual-angle encoding + aux loss)."""

from .dual_axis_angle_loss import (
    AxisLossResult,
    DualAxisLossResult,
    angle_to_vector,
    axis_cosine_loss,
    cosine_similarity,
    dual_axis_cosine_loss,
    normalize_vector,
    vector_to_angle_deg,
)

__all__ = [
    "AxisLossResult",
    "DualAxisLossResult",
    "angle_to_vector",
    "axis_cosine_loss",
    "cosine_similarity",
    "dual_axis_cosine_loss",
    "normalize_vector",
    "vector_to_angle_deg",
]
