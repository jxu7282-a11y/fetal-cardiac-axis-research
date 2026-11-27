"""HET-PSM side membership head, target construction and masked BCE loss.

The side branch is a pure auxiliary prediction of left/right cardiac
membership on the *final* decoder feature map.  It never touches the
segmentation heads, so ``network(x)`` still returns exactly the nnU-Net
output (including deep supervision).

Targets (from the augmented segmentation GT, same patch as the image):

    ID 1,2 -> Left      (channel 0)
    ID 3,4 -> Right     (channel 1)
    everything else (incl. 5,6,7,8,9) -> ignore

No anatomical claim is made about IDs 5-9; they are simply outside the
membership definition, so they are masked out.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "LEFT_LABEL_IDS",
    "RIGHT_LABEL_IDS",
    "SideMembershipHead",
    "final_feature_channels",
    "side_membership_targets",
    "masked_side_bce",
]

LEFT_LABEL_IDS = (1, 2)
RIGHT_LABEL_IDS = (3, 4)


def final_feature_channels(decoder: nn.Module) -> int:
    """Read the channel count of the last decoder conv from the module itself."""
    stages = getattr(decoder, "stages", None)
    if not stages:
        raise ValueError("decoder exposes no stages; cannot infer final channels")
    last = stages[-1]
    convs = getattr(last, "convs", None)
    if not convs:
        raise ValueError("last decoder stage has no convs; cannot infer channels")
    conv = getattr(convs[-1], "conv", None)
    if not isinstance(conv, nn.Conv2d):
        raise ValueError(f"unexpected last conv type: {type(conv)!r}")
    return int(conv.out_channels)


class SideMembershipHead(nn.Module):
    """1x1 conv on the final decoder feature -> 2 channels (left / right logits)."""

    def __init__(self, in_channels: int):
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        self.in_channels = int(in_channels)
        self.conv = nn.Conv2d(self.in_channels, 2, kernel_size=1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError(f"expected [B, C, H, W], got {tuple(feature.shape)}")
        if feature.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} channels, got {feature.shape[1]}")
        return self.conv(feature)


def _as_seg_map(seg: torch.Tensor) -> torch.Tensor:
    """Accept [B, 1, H, W] or [B, H, W]; return int64 [B, H, W]."""
    if seg.ndim == 4:
        if seg.shape[1] != 1:
            raise ValueError(f"expected a single-channel GT, got {tuple(seg.shape)}")
        seg = seg[:, 0]
    if seg.ndim != 3:
        raise ValueError(f"expected [B, H, W] GT, got {tuple(seg.shape)}")
    return seg.long()


def side_membership_targets(seg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build left/right targets and the valid mask from the segmentation GT.

    Returns ``(target, valid)`` with shapes ``[B, 2, H, W]`` and ``[B, 1, H, W]``.
    Pixels not labelled 1-4 are ignored; they are zeroed in ``target`` so that
    they contribute nothing even if the mask is misused.
    """
    label = _as_seg_map(seg)
    left = (label == LEFT_LABEL_IDS[0]) | (label == LEFT_LABEL_IDS[1])
    right = (label == RIGHT_LABEL_IDS[0]) | (label == RIGHT_LABEL_IDS[1])
    valid = (left | right).unsqueeze(1)
    target = torch.cat([left.unsqueeze(1), right.unsqueeze(1)], dim=1)
    target = target.to(torch.float32) * valid.to(torch.float32)
    return target, valid.to(torch.float32)


def masked_side_bce(logits: torch.Tensor, seg: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """Masked BCEWithLogitsLoss, averaged per case then over valid cases.

    Cases without any left/right pixel contribute no gradient; if the whole
    batch is empty a differentiable zero is returned (never NaN).
    """
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError(f"expected side logits [B, 2, H, W], got {tuple(logits.shape)}")
    target, valid = side_membership_targets(seg)
    if target.shape != logits.shape:
        raise ValueError(
            f"target {tuple(target.shape)} does not match logits {tuple(logits.shape)}")

    target = target.to(device=logits.device, dtype=logits.dtype)
    valid = valid.to(device=logits.device, dtype=logits.dtype)

    per_pixel = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    per_pixel = per_pixel * valid
    per_case_count = valid.sum(dim=(1, 2, 3))
    per_case_sum = per_pixel.sum(dim=(1, 2, 3))
    has_pixels = per_case_count > 0
    per_case = per_case_sum / per_case_count.clamp_min(1.0)

    denom = has_pixels.to(logits.dtype).sum()
    if float(denom) == 0.0:
        loss = logits.sum() * 0.0
    else:
        loss = (per_case * has_pixels.to(logits.dtype)).sum() / denom

    stats = {
        "valid_cases": int(has_pixels.sum().item()),
        "valid_pixels": int(per_case_count.sum().item()),
    }
    return loss, stats
