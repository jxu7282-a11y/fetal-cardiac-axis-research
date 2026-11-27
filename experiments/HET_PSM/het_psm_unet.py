"""HET-PSM network: stock PlainConvUNet + 1x1 left/right side membership head.

Subclassing (not wrapping) is deliberate: the module tree and therefore every
encoder ``state_dict`` key keeps the exact ``encoder.*`` / ``decoder.*`` names
that the E3 encoder-only transfer expects.  ``forward`` returns the untouched
nnU-Net segmentation output (deep supervision list or tensor); the side logits
are only available through ``forward_with_side``.
"""
from __future__ import annotations

import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch import nn

from .side_membership import SideMembershipHead, final_feature_channels

__all__ = ["HETPSMPlainConvUNet"]


class HETPSMPlainConvUNet(PlainConvUNet):
    """PlainConvUNet with an auxiliary 2-channel side membership head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        channels = final_feature_channels(self.decoder)
        self.side_head = SideMembershipHead(channels)
        self._side_feature = None
        self.decoder.stages[-1].register_forward_hook(self._capture_side_feature)

    def _capture_side_feature(self, module, inputs, output):
        self._side_feature = output

    @property
    def side_feature_channels(self) -> int:
        return self.side_head.in_channels

    def forward(self, x):
        return super().forward(x)

    def forward_with_side(self, x):
        """Return ``(segmentation_output, side_logits)``; seg output is unchanged."""
        self._side_feature = None
        seg = super().forward(x)
        feature = self._side_feature
        if feature is None:
            raise RuntimeError("failed to capture the final decoder feature map")
        self._side_feature = None
        return seg, self.side_head(feature)
