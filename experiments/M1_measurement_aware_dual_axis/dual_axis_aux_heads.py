"""Lightweight dual-axis auxiliary heads for M1 (CPU-only, no nnU-Net wiring).

Each head takes the final decoder feature map [B, C, H, W], pools it globally,
and emits one double-angle vector [cos 2t, sin 2t] of shape [B, 2].
The two heads are independent: separate Linear layers and parameters.
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["AxisHead", "DualAxisAuxHeads"]


class AxisHead(nn.Module):
    """GAP -> Flatten -> Linear(C, 2); returns an UNNORMALISED double-angle vector."""

    def __init__(self, in_channels: int):
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        self.in_channels = int(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(self.in_channels, 2)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(f"expected [B, C, H, W], got {tuple(feature.shape)}")
        if feature.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, got {feature.shape[1]}")
        return self.fc(self.flatten(self.pool(feature)))


class DualAxisAuxHeads(nn.Module):
    """Two independent axis heads reading the same final decoder feature map."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.card = AxisHead(in_channels)
        self.ref = AxisHead(in_channels)

    def forward(self, feature: torch.Tensor):
        return self.card(feature), self.ref(feature)
