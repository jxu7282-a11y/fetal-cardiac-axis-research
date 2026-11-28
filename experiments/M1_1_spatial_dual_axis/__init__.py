"""M1.1: spatial-preserving dual-axis auxiliary supervision.

Data reading spec: CARDIAC_AXIS_DATA_READING_SPEC_v1.0

Only difference from the archived M1 experiment: the auxiliary heads pool the
final decoder feature with ``AdaptiveAvgPool2d((4, 4))`` instead of a global
average pool, so the 4x4 spatial layout survives into the axis prediction layer.
The backbone, segmentation head, loss, labels, weights, schedule, folds and seed
are inherited unchanged from M1.
"""
from __future__ import annotations

from .dual_axis_aux_heads import SpatialAxisHead, SpatialDualAxisAuxHeads
from .m1_1_trainer import M1_1SpatialDualAxisUNet, nnUNetTrainer_M1_1SpatialDualAxis

__all__ = [
    "SpatialAxisHead",
    "SpatialDualAxisAuxHeads",
    "M1_1SpatialDualAxisUNet",
    "nnUNetTrainer_M1_1SpatialDualAxis",
]
