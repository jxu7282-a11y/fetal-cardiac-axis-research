"""CPU tests for M1 double-angle spatial-transform synchronisation.

Run with:
  python -m unittest experiments.M1_measurement_aware_dual_axis.tests.test_dual_axis_augmentation -v

The decisive tests do not trust the algebra alone: they draw a synthetic line
into an image, push it through the real batchgeneratorsv2 SpatialTransform /
MirrorTransform, and compare the measured line orientation with the label
produced by dual_axis_augmentation.py.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from experiments.M1_measurement_aware_dual_axis.dual_axis_augmentation import (  # noqa: E402
    angle_deg_to_double_angle,
    apply_forward_matrix_to_double_angle,
    apply_mirror_to_double_angle,
    double_angle_to_angle_deg,
    forward_matrix_from_affine_row_col,
    forward_matrix_xy,
    sampling_matrix_from_affine_row_col,
    transform_double_angle_label,
)
from batchgeneratorsv2.transforms.spatial.spatial import (  # noqa: E402
    SpatialTransform,
    create_affine_matrix_2d,
)

_PATCH = 129


def _axial(got: float, expected: float, tol: float = 1.0) -> bool:
    return abs((got - expected + 90.0) % 180.0 - 90.0) <= tol


class _FixedAngleSpatial(SpatialTransform):
    """SpatialTransform with a deterministic angle/scale (test-only subclass)."""

    def __init__(self, angle_rad: float, scales=(1.0, 1.0)):
        super().__init__((_PATCH, _PATCH), patch_center_dist_from_border=0,
                         random_crop=False, p_rotation=1.0,
                         rotation=(angle_rad, angle_rad), p_scaling=1.0,
                         scaling=tuple(scales), p_synchronize_scaling_across_axes=1)


def _line_image(angle_deg: float, size: int = _PATCH) -> torch.Tensor:
    """Draw a thin bright line through the centre at `angle_deg` in (x, y)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    c = (size - 1) / 2.0
    t = math.radians(angle_deg)
    dx, dy = math.cos(t), math.sin(t)
    # perpendicular distance from the line (x,y) = c + s*(dx,dy)
    dist = np.abs(-(dy) * (xx - c) + (dx) * (yy - c))
    return torch.from_numpy(np.exp(-(dist ** 2) / (2 * 1.0 ** 2))).float()[None]


def _measure_line_angle(img: torch.Tensor, bg_thresh: float = 0.25) -> float:
    w = img[0].detach().cpu().numpy().astype(np.float64)
    w = np.clip(w - bg_thresh, 0, None)
    yy, xx = np.mgrid[0:w.shape[0], 0:w.shape[1]].astype(np.float64)
    total = w.sum()
    if total <= 0:
        return float("nan")
    mx, my = (w * xx).sum() / total, (w * yy).sum() / total
    cxx = (w * (xx - mx) ** 2).sum() / total
    cyy = (w * (yy - my) ** 2).sum() / total
    cxy = (w * (xx - mx) * (yy - my)).sum() / total
    theta = 0.5 * math.atan2(2 * cxy, cxx - cyy)
    return math.degrees(theta) % 180.0


class TestMatrixDirection(unittest.TestCase):
    def test_affine_is_the_sampling_matrix_not_the_forward_matrix(self):
        angle = math.radians(37.0)
        affine = create_affine_matrix_2d(angle, [1.0, 1.0])
        sampling = sampling_matrix_from_affine_row_col(affine)
        forward = forward_matrix_from_affine_row_col(affine)
        self.assertTrue(torch.allclose(sampling, torch.as_tensor(affine).float(), atol=1e-6))
        self.assertTrue(torch.allclose(sampling @ forward, torch.eye(2), atol=1e-6))
        # a forward rotation by +37 deg in row/col order must not equal the affine
        self.assertFalse(torch.allclose(forward, sampling, atol=1e-3))

    def test_forward_matrix_is_correct_rotation_direction_row_col(self):
        angle = math.radians(30.0)
        forward = forward_matrix_from_affine_row_col(create_affine_matrix_2d(angle, [1.0, 1.0]))
        # row/col forward matrix = R(-angle) for the create_affine_matrix_2d sign convention
        expected = torch.tensor([[math.cos(angle), math.sin(angle)],
                                 [-math.sin(angle), math.cos(angle)]])
        self.assertTrue(torch.allclose(forward, expected, atol=1e-6))

    def test_xy_matrix_swaps_axes(self):
        # isotropic scaling: the angle must come through unchanged
        angle = math.radians(25.0)
        affine = create_affine_matrix_2d(angle, [1.4, 1.4])
        fwd_xy = forward_matrix_xy(affine)
        v = torch.tensor([1.0, 0.0])
        moved = v @ fwd_xy.T
        theta = math.degrees(math.atan2(float(moved[1]), float(moved[0]))) % 180.0
        self.assertTrue(_axial(theta, 25.0, tol=1e-4), f"got {theta}")

    def test_singular_matrix_is_rejected(self):
        with self.assertRaises(ValueError):
            forward_matrix_from_affine_row_col(np.array([[0.0, 0.0], [0.0, 1.0]]))

    def test_non_finite_matrix_is_rejected(self):
        with self.assertRaises(ValueError):
            forward_matrix_from_affine_row_col(np.array([[np.nan, 0.0], [0.0, 1.0]]))


class TestScaling(unittest.TestCase):
    def test_isotropic_scaling_preserves_angle(self):
        for s in (0.5, 0.7, 1.0, 1.4, 2.0):
            affine = create_affine_matrix_2d(0.0, [s, s])
            out = transform_double_angle_label(angle_deg_to_double_angle(32.0), affine)
            self.assertAlmostEqual(float(double_angle_to_angle_deg(out)), 32.0, places=3)

    def test_anisotropic_scaling_changes_angle_as_expected(self):
        # affine scales (row, col) by (1, 2); the image is therefore sampled
        # with row'=row, col'=0.5*col, so 45 deg -> atan(1 / 0.5) = 63.435 deg.
        affine = create_affine_matrix_2d(0.0, [1.0, 2.0])
        out = transform_double_angle_label(angle_deg_to_double_angle(45.0), affine)
        got = float(double_angle_to_angle_deg(out))
        self.assertTrue(_axial(got, math.degrees(math.atan2(1, 0.5)), tol=0.01), f"got {got}")

    def test_anisotropic_scaling_matches_measured_line(self):
        angle, scales = 40.0, (1.0, 1.9)
        tr = _FixedAngleSpatial(0.0, scales)
        img = _line_image(angle)
        params = tr.get_parameters(image=img)
        out = tr.apply({'image': img}, **params)['image']
        measured = _measure_line_angle(out)
        label = float(double_angle_to_angle_deg(transform_double_angle_label(
            angle_deg_to_double_angle(angle), params['affine'])))
        self.assertTrue(_axial(measured, label, tol=2.0),
                        f"measured {measured:.3f} vs label {label:.3f}")


class TestRealSpatialTransform(unittest.TestCase):
    def _check(self, angle_deg: float, delta_deg: float, scales=(1.0, 1.0)):
        tr = _FixedAngleSpatial(math.radians(delta_deg), scales)
        img = _line_image(angle_deg)
        params = tr.get_parameters(image=img)
        out = tr.apply({'image': img}, **params)['image']
        measured = _measure_line_angle(out)
        label = float(double_angle_to_angle_deg(transform_double_angle_label(
            angle_deg_to_double_angle(angle_deg), params['affine'])))
        self.assertTrue(_axial(measured, label, tol=2.5),
                        f"angle={angle_deg} delta={delta_deg} scales={scales}: "
                        f"measured {measured:.3f} vs label {label:.3f}")
        return measured, label

    def test_rotation_positive(self):
        self._check(20.0, 30.0)

    def test_rotation_negative(self):
        self._check(20.0, -30.0)

    def test_rotation_near_right_angle(self):
        self._check(10.0, 75.0)

    def test_rotation_and_anisotropic_scaling(self):
        self._check(25.0, 33.0, scales=(0.85, 1.3))

    def test_no_spatial_transform(self):
        tr = _FixedAngleSpatial(0.0)
        tr.p_rotation = 0.0
        tr.p_scaling = 0.0
        img = _line_image(52.0)
        params = tr.get_parameters(image=img)
        self.assertIsNone(params['affine'])
        out = tr.apply({'image': img}, **params)['image']
        self.assertTrue(_axial(_measure_line_angle(out), 52.0, tol=1.0))


class TestMirroring(unittest.TestCase):
    def test_single_axis_flip_negates_angle(self):
        v = angle_deg_to_double_angle(35.0)
        for axis in (0, 1):
            got = float(double_angle_to_angle_deg(apply_mirror_to_double_angle(v, (axis,))))
            self.assertTrue(_axial(got, 145.0, tol=1e-4), f"axis {axis}: got {got}")

    def test_double_flip_cancels(self):
        v = angle_deg_to_double_angle(35.0)
        got = float(double_angle_to_angle_deg(apply_mirror_to_double_angle(v, (0, 1))))
        self.assertTrue(_axial(got, 35.0, tol=1e-4))

    def test_vertical_flip_matches_real_mirror_transform(self):
        from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform

        angle = 28.0
        img = _line_image(angle)
        flipped = torch.flip(img, (1,))  # row axis flip == MirrorTransform axis 0
        measured = _measure_line_angle(flipped)
        label = float(double_angle_to_angle_deg(apply_mirror_to_double_angle(
            angle_deg_to_double_angle(angle), (0,))))
        self.assertTrue(_axial(measured, label, tol=1.0),
                        f"measured {measured:.3f} vs label {label:.3f}")
        # and the transform really is the one under test
        mirror = MirrorTransform((0,))
        out = mirror.apply({'image': img}, axes=[0])['image']
        self.assertTrue(torch.allclose(out, flipped))

    def test_horizontal_flip_matches_real_mirror_transform(self):
        from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform

        angle = 28.0
        img = _line_image(angle)
        flipped = torch.flip(img, (2,))
        measured = _measure_line_angle(flipped)
        label = float(double_angle_to_angle_deg(apply_mirror_to_double_angle(
            angle_deg_to_double_angle(angle), (1,))))
        self.assertTrue(_axial(measured, label, tol=1.0))
        mirror = MirrorTransform((1,))
        out = mirror.apply({'image': img}, axes=[1])['image']
        self.assertTrue(torch.allclose(out, flipped))

    def test_mirror_composes_after_rotation(self):
        angle, delta = 33.0, 21.0
        tr = _FixedAngleSpatial(math.radians(delta))
        img = _line_image(angle)
        params = tr.get_parameters(image=img)
        rotated = tr.apply({'image': img}, **params)['image']
        flipped = torch.flip(rotated, (1,))
        label = float(double_angle_to_angle_deg(transform_double_angle_label(
            angle_deg_to_double_angle(angle), params['affine'], mirror_axes=(0,))))
        self.assertTrue(_axial(_measure_line_angle(flipped), label, tol=2.5))

    def test_invalid_mirror_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_mirror_to_double_angle(angle_deg_to_double_angle(0.0), (2,))


class TestEncodingInvariance(unittest.TestCase):
    def test_identity_matrix_is_a_no_op_everywhere(self):
        angles = torch.linspace(0.0, 179.0, 180)
        out = apply_forward_matrix_to_double_angle(
            angle_deg_to_double_angle(angles), torch.eye(2))
        err = torch.abs((double_angle_to_angle_deg(out) - angles + 90.0) % 180.0 - 90.0)
        self.assertLess(float(err.max()), 1e-3)

    def test_quarter_turn_identity(self):
        # theta and theta+90 are NOT the same axis; a 90 deg rotation must move it
        out = apply_forward_matrix_to_double_angle(
            angle_deg_to_double_angle(20.0), torch.tensor([[0.0, -1.0], [1.0, 0.0]]))
        self.assertTrue(_axial(float(double_angle_to_angle_deg(out)), 110.0, tol=1e-4))

    def test_double_angle_is_180_periodic(self):
        for a in (0.0, 5.5, 44.0, 90.0, 137.0):
            v1 = angle_deg_to_double_angle(a)
            v2 = angle_deg_to_double_angle(a + 180.0)
            self.assertTrue(torch.allclose(v1, v2, atol=1e-5))

    def test_degenerate_input_stays_finite(self):
        zeros = torch.zeros(3, 2)
        out = apply_forward_matrix_to_double_angle(zeros, torch.eye(2))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))

    def test_nan_angle_is_rejected(self):
        with self.assertRaises(ValueError):
            angle_deg_to_double_angle(float("nan"))

    def test_batch_shape_is_preserved(self):
        v = angle_deg_to_double_angle(torch.rand(4, 5) * 179.0)
        out = transform_double_angle_label(v, create_affine_matrix_2d(0.2, [1.1, 0.9]), (0,))
        self.assertEqual(tuple(out.shape), (4, 5, 2))


if __name__ == "__main__":
    unittest.main()
