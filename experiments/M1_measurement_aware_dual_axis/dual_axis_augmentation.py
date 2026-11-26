"""Spatial-transform synchronisation for M1 double-angle axis labels (CPU only).

Matrix convention (verified against nnU-Net 2.8.1 / batchgeneratorsv2)
----------------------------------------------------------------------
``batchgeneratorsv2.transforms.spatial.spatial.SpatialTransform.get_parameters``
returns ``affine = R @ diag(scales)`` from ``create_affine_matrix_2d``. That
matrix is applied to the *output* grid before ``grid_sample``
(``grid = base_grid @ affine.T``), and the grid holds, per output pixel, the
input coordinate to sample. It is therefore the **backward/sampling** matrix
``M_out_to_in``. The physical forward matrix is its inverse
``M_in_to_out = inv(affine)``.

The base grid is built over ``(axis0, axis1) = (row, col)`` and only reordered
(not re-derived) for ``grid_sample``. Project axis angles are defined in
``(x, y) = (col, row)``, so matrices are conjugated with the swap
``J = [[0, 1], [1, 0]]`` before touching direction vectors.

Only direction is transformed: translation and patch cropping cannot change an
axis orientation, so ``center_location_in_pixels`` is irrelevant here.
"""
from __future__ import annotations

import torch

from .dual_axis_angle_loss import normalize_vector, vector_to_angle_deg

__all__ = [
    "SPATIAL_AXIS_ORDER",
    "apply_forward_matrix_to_double_angle",
    "apply_mirror_to_double_angle",
    "angle_deg_to_double_angle",
    "double_angle_to_angle_deg",
    "forward_matrix_from_affine_row_col",
    "forward_matrix_xy",
    "sampling_matrix_from_affine_row_col",
    "transform_double_angle_label",
]

SPATIAL_AXIS_ORDER = "row_col"

_SWAP = torch.tensor([[0.0, 1.0], [1.0, 0.0]])


def _as_2x2_matrix(matrix) -> torch.Tensor:
    m = torch.as_tensor(matrix, dtype=torch.get_default_dtype())
    if tuple(m.shape) != (2, 2):
        raise ValueError(f"expected a 2x2 matrix, got {tuple(m.shape)}")
    if not torch.isfinite(m).all():
        raise ValueError("matrix contains non-finite values")
    return m


def sampling_matrix_from_affine_row_col(affine) -> torch.Tensor:
    """Return the output->input matrix exactly as SpatialTransform uses it.

    This is the ``affine`` entry of ``SpatialTransform.get_parameters`` in
    ``(row, col)`` axis order. It is NOT the matrix that moves a point of the
    image forward; it maps each output pixel to the input pixel it samples.
    """
    return _as_2x2_matrix(affine).clone()


def forward_matrix_from_affine_row_col(affine) -> torch.Tensor:
    """Return the input->output (physical) matrix in ``(row, col)`` order."""
    sampling = _as_2x2_matrix(affine)
    try:
        forward = torch.linalg.inv(sampling)
    except RuntimeError as exc:
        raise ValueError(f"sampling matrix is singular and cannot be inverted: {sampling}") from exc
    if not torch.isfinite(forward).all():
        raise ValueError("forward matrix is not finite; refusing to transform labels")
    return forward


def forward_matrix_xy(affine) -> torch.Tensor:
    """Return the input->output matrix in ``(x, y) = (col, row)`` order.

    Use this matrix directly on ``[cos(theta), sin(theta)]`` direction vectors
    whose angle is ``atan2(dy, dx)`` as defined by the project GT protocol.
    """
    forward_row_col = forward_matrix_from_affine_row_col(affine)
    swap = _SWAP.to(dtype=forward_row_col.dtype)
    return swap @ forward_row_col @ swap


def angle_deg_to_double_angle(angle_deg) -> torch.Tensor:
    """[theta_deg] -> [cos 2theta, sin 2theta]; non-finite input raises."""
    t = torch.as_tensor(angle_deg, dtype=torch.get_default_dtype())
    if not torch.isfinite(t).all():
        raise ValueError("angle contains non-finite values; mask the sample instead")
    rad = torch.deg2rad(2.0 * t)
    return torch.stack((torch.cos(rad), torch.sin(rad)), dim=-1)


def double_angle_to_angle_deg(vec) -> torch.Tensor:
    """Inverse of :func:`angle_deg_to_double_angle`, returned in [0, 180)."""
    return vector_to_angle_deg(vec)


def _unit_rows(vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    unit, degenerate = normalize_vector(vec)
    return unit, ~degenerate


def apply_forward_matrix_to_double_angle(vec, forward_xy) -> torch.Tensor:
    """Transform double-angle vectors by the physical forward matrix in xy order.

    The vector is decoded to a unit direction, multiplied by ``forward_xy``,
    renormalised and re-encoded with the double-angle map. An anisotropic
    scaling changes the encoded angle exactly as it changes the drawn line.
    Degenerate (zero/non-finite) rows stay zero instead of becoming NaN.
    """
    unit, finite = _unit_rows(torch.as_tensor(vec, dtype=torch.get_default_dtype()))
    matrix = _as_2x2_matrix(forward_xy)
    # decode cos(2t), sin(2t) -> the axial direction (cos t, sin t)
    theta = torch.atan2(unit[..., 1], unit[..., 0]) / 2.0
    direction = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
    moved = direction @ matrix.T
    result, moved_finite = _unit_rows(moved)
    # re-encode the moved direction as a double angle
    out = torch.stack((
        result[..., 0] ** 2 - result[..., 1] ** 2,
        2.0 * result[..., 0] * result[..., 1],
    ), dim=-1)
    valid = finite & moved_finite
    return torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))


def apply_mirror_to_double_angle(vec, mirror_axes=()) -> torch.Tensor:
    """Apply horizontal/vertical mirroring to double-angle labels.

    ``mirror_axes`` uses the tensor's spatial axes: 0 = rows (vertical flip),
    1 = columns (horizontal flip). For an UNDIRECTED line either single-axis
    flip gives theta -> -theta, i.e. only the sine component of the double
    angle changes sign. Two flips cancel.
    """
    axes = tuple(int(a) for a in mirror_axes)
    invalid = [a for a in axes if a not in (0, 1)]
    if invalid:
        raise ValueError(f"unsupported mirror axes {invalid}; expected subset of (0, 1)")
    v = torch.as_tensor(vec, dtype=torch.get_default_dtype())
    if len(set(axes)) % 2 == 0:
        return v.clone()
    sign = torch.tensor([1.0, -1.0], dtype=v.dtype, device=v.device)
    return v * sign


def transform_double_angle_label(vec, affine=None, mirror_axes=()) -> torch.Tensor:
    """Full label transform for one axis.

    ``affine`` is the raw 2x2 matrix from ``SpatialTransform.get_parameters``
    (``None`` when that transform was not applied). ``mirror_axes`` is the list
    of axes actually flipped by ``MirrorTransform``. Order matters only in that
    both act on the same image coordinate frame; rotation and mirroring do not
    commute in general, so this function applies the spatial affine first and
    the mirror second, matching the nnU-Net pipeline order.
    """
    out = torch.as_tensor(vec, dtype=torch.get_default_dtype()).clone()
    if affine is not None:
        out = apply_forward_matrix_to_double_angle(out, forward_matrix_xy(affine))
    if mirror_axes:
        out = apply_mirror_to_double_angle(out, mirror_axes)
    return out
