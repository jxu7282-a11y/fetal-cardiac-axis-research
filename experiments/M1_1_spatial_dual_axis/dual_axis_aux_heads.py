"""Spatial-preserving dual-axis auxiliary heads (M1.1).

Sole change relative to the archived M1 heads
(`experiments/M1_measurement_aware_dual_axis/dual_axis_aux_heads.py`):

    M1   : AdaptiveAvgPool2d(1)        -> Flatten -> Linear(C, 2)
    M1.1 : AdaptiveAvgPool2d((4, 4))   -> Flatten -> Linear(16*C, 2)

Everything else is identical: two independent Card/Reference heads, the same
input feature map (final decoder stage, the tensor that feeds `seg_layers[-1]`),
the same un-normalised double-angle output `[cos 2theta, sin 2theta]` of shape
[B, 2], and the same random initialisation scheme. Only the FC input width grows
from C to 16*C because 16 spatial cells are now kept.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["SpatialAxisHead", "SpatialDualAxisAuxHeads", "POOL_SIZE"]

# Spatial grid kept by the auxiliary heads. Fixed by the M1.1 spec.
POOL_SIZE = 4


class SpatialAxisHead(nn.Module):
    """AdaptiveAvgPool2d((4,4)) -> Flatten -> Linear(16*C, 2).

    Returns an UNNORMALISED double-angle vector [cos 2theta, sin 2theta].
    """

    def __init__(self, in_channels: int, pool_size: int = POOL_SIZE):
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if int(pool_size) <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")
        self.in_channels = int(in_channels)
        self.pool_size = int(pool_size)
        self.pool = nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size))
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(self.in_channels * self.pool_size * self.pool_size, 2)

    @property
    def fc_in_features(self) -> int:
        return self.fc.in_features

    def pooled(self, feature: torch.Tensor) -> torch.Tensor:
        return self.pool(feature)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(f"expected [B, C, H, W], got {tuple(feature.shape)}")
        if feature.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, got {feature.shape[1]}")
        return self.fc(self.flatten(self.pool(feature)))


class SpatialDualAxisAuxHeads(nn.Module):
    """Two independent spatial axis heads reading the same decoder feature map."""

    def __init__(self, in_channels: int, pool_size: int = POOL_SIZE):
        super().__init__()
        self.card = SpatialAxisHead(in_channels, pool_size)
        self.ref = SpatialAxisHead(in_channels, pool_size)

    def forward(self, feature: torch.Tensor):
        return self.card(feature), self.ref(feature)
