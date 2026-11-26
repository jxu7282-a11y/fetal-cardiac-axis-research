"""Dual-angle (double-angle) axis encoding and auxiliary cosine loss.

Stage 1 of M1. Self-contained: no nnU-Net imports, no trainer wiring, CPU only.

Angle convention (matches src/cardiac_axis/angles.py)
-----------------------------------------------------
Axes are UNDIRECTED lines: theta and theta + 180 deg are the same axis, and the
useful error range is [0, 90] deg. Encoding an axis as

    v(theta) = [cos(2*theta), sin(2*theta)]

makes the representation 180-deg periodic, so 0 deg and 180 deg map to the same
vector, and 1 - cos(v_pred, v_gt) is maximal when the two axes are 90 deg apart.

Units caveat (documented, not worked around)
-------------------------------------------
Angles are given in DEGREES, so dL/d(theta_deg) = dL/d(theta_rad) * pi/180.
When a head predicts degrees, every gradient therefore carries a 1/57.3 factor
relative to a radian-parametrised equivalent. That is the correct derivative for
a degree-valued input, but it does change the effective gradient scale seen by
the optimiser; it is recorded in training_summary rather than silently rescaled.

Loss definition (fixed by the M1 spec, not tunable here)
--------------------------------------------------------
    L_axis = mean_{i in valid_axis} (1 - cos(v_pred_i, v_gt_i))
    L_total = (L_card + L_ref) / 2

Each axis is averaged over its own valid samples. The two axis weights are NOT
re-normalised when one axis has no valid samples; that case yields exactly 0.0
for the empty axis and the reported valid counts make the consequence explicit.
An axis with zero valid samples returns a differentiable zero (never NaN), so
backward() still works and no spurious gradient is produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

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

# Small constant so that normalising an (almost) zero vector cannot divide by 0.
_EPS = 1e-12
# Vectors shorter than this are treated as degenerate and replaced by zeros.
_MIN_NORM = 1e-8


def _as_float_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if x.is_complex():
            raise TypeError("complex tensors are not supported")
        return x if x.is_floating_point() else x.to(torch.get_default_dtype())
    return torch.as_tensor(x, dtype=torch.get_default_dtype())


def angle_to_vector(angle_deg):
    """Map angle(s) in DEGREES to the double-angle unit vector [cos 2t, sin 2t].

    Returns a tensor of shape (..., 2), where the last dimension holds
    [cos(2t), sin(2t)]. Input must be finite (NaN/Inf raise ValueError); invalid
    *samples* are handled by the validity masks of the loss functions instead.
    """
    t = _as_float_tensor(angle_deg)
    if not torch.isfinite(t).all():
        raise ValueError("angle_to_vector received non-finite angles")
    rad = torch.deg2rad(2.0 * t)
    return torch.stack((torch.cos(rad), torch.sin(rad)), dim=-1)


def vector_to_angle_deg(vec):
    """Inverse of `angle_to_vector`; returns the axis angle in [0, 180) degrees.

    Only meant for inspection/tests: the returned angle is the branch
    atan2(sin, cos)/2 mapped into [0, 180).
    """
    v = _as_float_tensor(vec)
    if v.shape[-1] != 2:
        raise ValueError(f"expected last dimension 2, got {tuple(v.shape)}")
    ang = torch.rad2deg(torch.atan2(v[..., 1], v[..., 0])) / 2.0
    return ang % 180.0


def normalize_vector(vec):
    """L2-normalise the last dimension, without ever producing NaN.

    Degenerate (near-zero or non-finite) rows are replaced by the zero vector and
    reported through `degenerate_mask` so the caller can drop them.

    Returns
    -------
    (unit, degenerate_mask)
        `unit` has the same shape as `vec`; `degenerate_mask` has shape `vec.shape[:-1]`.
    """
    v = _as_float_tensor(vec)
    if v.shape[-1] != 2:
        raise ValueError(f"expected last dimension 2, got {tuple(v.shape)}")
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    finite = torch.isfinite(v).all(dim=-1, keepdim=True) & torch.isfinite(norm)
    safe_norm = torch.where(finite & (norm > _MIN_NORM), norm, torch.ones_like(norm))
    unit = torch.where(finite & (norm > _MIN_NORM), v / (safe_norm + _EPS),
                       torch.zeros_like(v))
    degenerate = ~(finite.squeeze(-1) & (norm.squeeze(-1) > _MIN_NORM))
    return unit, degenerate


def cosine_similarity(pred_vec, target_vec):
    """Cosine similarity per sample, clamped to [-1, 1]; degenerate rows give 0."""
    p, p_bad = normalize_vector(pred_vec)
    t, t_bad = normalize_vector(target_vec)
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: {tuple(p.shape)} vs {tuple(t.shape)}")
    cos = (p * t).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.where(p_bad | t_bad, torch.zeros_like(cos), cos)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Mean of `values` over `mask`; differentiable zero when nothing is valid."""
    values = values.reshape(-1)
    mask = mask.reshape(-1).to(torch.bool)
    if values.shape != mask.shape:
        raise ValueError(f"value/mask shape mismatch: {tuple(values.shape)} vs {tuple(mask.shape)}")
    n_valid = int(mask.sum().item())
    total = (values * mask.to(values.dtype)).sum()
    if n_valid == 0:
        # `values.sum() * 0` stays attached to the autograd graph, so backward()
        # works on a fully masked axis and contributes exactly zero gradient.
        return values.sum() * 0.0, 0
    return total / n_valid, n_valid


@dataclass
class AxisLossResult:
    """Per-axis auxiliary loss."""
    loss: torch.Tensor
    valid_count: int
    total_count: int
    dropped_nonfinite: int = 0
    dropped_degenerate: int = 0

    @property
    def all_masked(self) -> bool:
        return self.valid_count == 0


@dataclass
class DualAxisLossResult:
    """Card + Reference auxiliary loss."""
    loss: torch.Tensor
    card: AxisLossResult
    ref: AxisLossResult
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "loss_total": float(self.loss.detach().cpu()),
            "loss_card": float(self.card.loss.detach().cpu()),
            "loss_ref": float(self.ref.loss.detach().cpu()),
            "card_valid": self.card.valid_count,
            "card_total": self.card.total_count,
            "ref_valid": self.ref.valid_count,
            "ref_total": self.ref.total_count,
            "card_dropped_nonfinite": self.card.dropped_nonfinite,
            "ref_dropped_nonfinite": self.ref.dropped_nonfinite,
            "card_dropped_degenerate": self.card.dropped_degenerate,
            "ref_dropped_degenerate": self.ref.dropped_degenerate,
        }


def axis_cosine_loss(pred_vec, target, valid: torch.Tensor | None = None,
                     *, target_is_angle: bool | None = None) -> AxisLossResult:
    """Cosine loss `1 - cos` for ONE axis, averaged over its valid samples.

    Parameters
    ----------
    pred_vec : tensor (..., 2)
        Predicted double-angle vector(s). Normalised internally.
    target : tensor (..., 2) or tensor (...,)
        Target vectors, or target angles in DEGREES (then set
        `target_is_angle=True`, or rely on the shape-based default).
    valid : bool tensor of shape (...), optional
        Per-sample validity mask. Defaults to all-True.
    target_is_angle : bool, optional
        Defaults to True when `target` has no trailing dimension of size 2.

    Non-finite predictions/targets are excluded from the mean instead of
    producing NaN; the counts are reported in the result.
    """
    p = _as_float_tensor(pred_vec)
    if p.shape[-1] != 2:
        raise ValueError(f"pred_vec must have last dimension 2, got {tuple(p.shape)}")
    sample_shape = p.shape[:-1]

    t = _as_float_tensor(target)
    # A single value broadcast against a single sample is unambiguous.
    if t.numel() == 1 and math.prod(sample_shape) == 1 and t.shape != tuple(sample_shape):
        t = t.reshape(sample_shape if sample_shape else ())
    if target_is_angle is None:
        # Disambiguate by SHAPE, not by "does the last dim equal 2": with two
        # samples, an angle target of shape (2,) would otherwise be mistaken for
        # a vector target.
        if tuple(t.shape) == tuple(sample_shape):
            target_is_angle = True
        elif tuple(t.shape) == tuple(p.shape):
            target_is_angle = False
        else:
            raise ValueError(
                f"target shape {tuple(t.shape)} matches neither the angle shape "
                f"{tuple(sample_shape)} nor the vector shape {tuple(p.shape)}; "
                "pass target_is_angle explicitly")
    if target_is_angle:
        if t.shape != sample_shape:
            raise ValueError(f"target angle shape {tuple(t.shape)} != {tuple(sample_shape)}")
        finite_t = torch.isfinite(t)
        target_vec = angle_to_vector(torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0))
    else:
        if t.shape != p.shape:
            raise ValueError(f"target vector shape {tuple(t.shape)} != {tuple(p.shape)}")
        finite_t = torch.isfinite(t).all(dim=-1)
        target_vec = t

    if valid is None:
        mask = torch.ones(sample_shape, dtype=torch.bool, device=p.device)
    else:
        mask = _as_float_tensor(valid).to(torch.bool)
        if tuple(mask.shape) != tuple(sample_shape):
            # A scalar mask (or one scalar per axis) broadcasts over all samples;
            # anything else must match the sample shape exactly.
            try:
                mask = mask.expand(sample_shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"valid mask shape {tuple(mask.shape)} is not broadcastable to "
                    f"{tuple(sample_shape)}") from exc

    finite_p = torch.isfinite(p).all(dim=-1)
    finite = finite_p & finite_t
    cos = cosine_similarity(p, target_vec)
    _, p_degenerate = normalize_vector(p)

    effective = mask & finite & ~p_degenerate
    loss, n_valid = _masked_mean(1.0 - cos, effective)

    return AxisLossResult(
        loss=loss,
        valid_count=n_valid,
        total_count=int(mask.numel()),
        dropped_nonfinite=int((mask & ~finite).sum().item()),
        dropped_degenerate=int((mask & finite & p_degenerate).sum().item()),
    )


def dual_axis_cosine_loss(pred_card_vec, pred_ref_vec,
                          target_card, target_ref,
                          card_valid: torch.Tensor | None = None,
                          ref_valid: torch.Tensor | None = None,
                          *, target_is_angle: bool | None = None) -> DualAxisLossResult:
    """Card + Reference auxiliary loss: `(L_card + L_ref) / 2`.

    `card_valid` and `ref_valid` are independent per-sample masks. Either or both
    may be all-False (or `None` with zero-length inputs); the corresponding axis
    then contributes exactly 0.0 without producing NaN, and the total remains
    `(L_card + L_ref) / 2` exactly as specified.
    """
    card = axis_cosine_loss(pred_card_vec, target_card, card_valid,
                            target_is_angle=target_is_angle)
    ref = axis_cosine_loss(pred_ref_vec, target_ref, ref_valid,
                           target_is_angle=target_is_angle)
    total = (card.loss + ref.loss) / 2.0
    return DualAxisLossResult(loss=total, card=card, ref=ref)
