"""CPU unit tests for the M1 stage-1 dual-angle encoding + auxiliary loss.

Run with:  python -m unittest discover -s experiments/M1_measurement_aware_dual_axis/tests -v
No GPU, no nnU-Net, no training.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from experiments.M1_measurement_aware_dual_axis.dual_axis_angle_loss import (  # noqa: E402
    angle_to_vector,
    axis_cosine_loss,
    cosine_similarity,
    dual_axis_cosine_loss,
    normalize_vector,
    vector_to_angle_deg,
)


def _v(angle_deg: float) -> torch.Tensor:
    return angle_to_vector(torch.tensor(float(angle_deg)))


def _assert_close(test: unittest.TestCase, a, b, tol=1e-6, msg=""):
    test.assertTrue(abs(float(a) - float(b)) <= tol,
                    f"{msg}: {float(a)} vs {float(b)}")


class TestAngleEncoding(unittest.TestCase):
    def test_unit_norm(self):
        for a in (0.0, 17.0, 45.0, 90.0, 135.0, 179.9, 270.0, 359.0, -45.0):
            v = _v(a)
            _assert_close(self, torch.linalg.vector_norm(v), 1.0, msg=f"norm at {a}")

    def test_double_angle_values(self):
        # 0 deg -> [1, 0]; 45 deg -> [0, 1]; 90 deg -> [-1, 0]; 135 deg -> [0, -1]
        _assert_close(self, _v(0)[0], 1.0)
        _assert_close(self, _v(0)[1], 0.0, tol=1e-7)
        _assert_close(self, _v(45)[0], 0.0, tol=1e-7)
        _assert_close(self, _v(45)[1], 1.0)
        _assert_close(self, _v(90)[0], -1.0)
        _assert_close(self, _v(90)[1], 0.0, tol=1e-7)
        _assert_close(self, _v(135)[0], 0.0, tol=1e-7)
        _assert_close(self, _v(135)[1], -1.0)

    def test_zero_and_180_are_identical(self):
        self.assertTrue(torch.allclose(_v(0.0), _v(180.0), atol=1e-6))
        self.assertTrue(torch.allclose(_v(37.0), _v(217.0), atol=1e-6))
        self.assertTrue(torch.allclose(_v(-45.0), _v(135.0), atol=1e-6))

    def test_180_periodicity(self):
        for a in (0.0, 12.5, 60.0, 123.4, 179.99):
            self.assertTrue(torch.allclose(_v(a), _v(a + 180.0), atol=1e-5),
                            msg=f"periodicity at {a}")
            self.assertTrue(torch.allclose(_v(a), _v(a + 360.0), atol=1e-5),
                            msg=f"360 periodicity at {a}")

    def test_boundaries_and_roundtrip(self):
        for a in (0.0, 45.0, 90.0, 135.0, 179.0, 180.0, 270.0, 360.0, -90.0):
            back = float(vector_to_angle_deg(_v(a)))
            # round-trip is only defined modulo 180 deg
            _assert_close(self, (back - a) % 180.0, 0.0, tol=1e-4,
                          msg=f"roundtrip at {a}")

    def test_batched_shapes(self):
        ang = torch.tensor([[0.0, 30.0], [90.0, 150.0]])
        v = angle_to_vector(ang)
        self.assertEqual(tuple(v.shape), (2, 2, 2))
        _assert_close(self, v[0, 1, 0], math.cos(math.radians(60.0)), tol=1e-6)
        _assert_close(self, v[1, 1, 1], math.sin(math.radians(300.0)), tol=1e-6)

    def test_non_finite_angle_rejected(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                angle_to_vector(torch.tensor([bad]))


class TestNormalization(unittest.TestCase):
    def test_unnormalized_input(self):
        unit, bad = normalize_vector(torch.tensor([[3.0, 4.0]]))
        self.assertFalse(bool(bad.any()))
        _assert_close(self, torch.linalg.vector_norm(unit), 1.0)
        _assert_close(self, unit[0, 0], 0.6)
        _assert_close(self, unit[0, 1], 0.8)

    def test_zero_vector_is_safe(self):
        unit, bad = normalize_vector(torch.zeros(2, 2))
        self.assertTrue(bool(bad.all()))
        self.assertFalse(bool(torch.isnan(unit).any()))
        self.assertTrue(bool((unit == 0).all()))

    def test_non_finite_vector_is_safe(self):
        unit, bad = normalize_vector(torch.tensor([[float("nan"), 1.0], [1.0, 2.0]]))
        self.assertTrue(bool(bad[0]))
        self.assertFalse(bool(bad[1]))
        self.assertFalse(bool(torch.isnan(unit).any()))

    def test_cosine_of_identical_vectors(self):
        v = _v(23.0)
        _assert_close(self, cosine_similarity(v, v), 1.0)


class TestCosineLoss(unittest.TestCase):
    def test_zero_loss_on_exact_match(self):
        pred = _v(0.0).requires_grad_(True)
        res = axis_cosine_loss(pred, torch.tensor(0.0))
        _assert_close(self, res.loss, 0.0)
        self.assertEqual(res.valid_count, 1)

    def test_zero_loss_on_180_equivalent_target(self):
        # 0 deg and 180 deg are the SAME axis -> loss must be exactly 0
        res = axis_cosine_loss(_v(0.0), torch.tensor(180.0))
        _assert_close(self, res.loss, 0.0)
        res = axis_cosine_loss(_v(179.0), torch.tensor(-1.0))
        _assert_close(self, res.loss, 0.0)

    def test_quarter_turn_is_max_loss(self):
        # 90 deg apart -> cos(2*90)=cos(180)=-1 -> loss = 2
        res = axis_cosine_loss(_v(0.0), torch.tensor(90.0))
        _assert_close(self, res.loss, 2.0)
        res = axis_cosine_loss(_v(30.0), torch.tensor(120.0))
        _assert_close(self, res.loss, 2.0)

    def test_loss_matches_closed_form(self):
        for delta in (0.0, 5.0, 17.0, 45.0, 60.0, 89.0, 90.0):
            res = axis_cosine_loss(_v(10.0), torch.tensor(10.0 + delta))
            expected = 1.0 - math.cos(math.radians(2.0 * delta))
            _assert_close(self, res.loss, expected, tol=1e-5, msg=f"delta={delta}")

    def test_loss_symmetric_and_monotone(self):
        base = 10.0
        losses = []
        for delta in (0.0, 10.0, 20.0, 40.0, 60.0, 80.0, 90.0):
            losses.append(float(axis_cosine_loss(_v(base), torch.tensor(base + delta)).loss))
        self.assertEqual(losses, sorted(losses), "loss must increase with angular gap")
        self.assertTrue(torch.allclose(
            axis_cosine_loss(_v(base), torch.tensor(base + 25.0)).loss,
            axis_cosine_loss(_v(base + 25.0), torch.tensor(base)).loss,
            atol=1e-6), "loss must be symmetric")

    def test_axis_range_boundaries(self):
        # predictions and targets at the extremes of the undirected axis range
        for pred, tgt in ((0.0, 90.0), (90.0, 0.0), (179.9, 90.0), (0.0, 179.9),
                          (359.9, 0.0), (-90.0, 90.0)):
            res = axis_cosine_loss(_v(pred), torch.tensor(tgt))
            self.assertTrue(math.isfinite(float(res.loss)), msg=f"{pred}->{tgt}")
            self.assertGreaterEqual(float(res.loss), 0.0)
            self.assertLessEqual(float(res.loss), 2.0 + 1e-6)

    def test_vector_target_matches_angle_target(self):
        a = axis_cosine_loss(_v(12.0), torch.tensor(48.0))
        b = axis_cosine_loss(_v(12.0), _v(48.0), target_is_angle=False)
        _assert_close(self, a.loss, b.loss)


class TestValidityMask(unittest.TestCase):
    def _setup(self):
        pred = torch.stack((_v(0.0), _v(10.0), _v(20.0), _v(30.0)))
        tgt = torch.tensor([0.0, 10.0, 20.0, 30.0])
        return pred, tgt

    def test_mask_selects_subset(self):
        pred, tgt = self._setup()
        # sample 0 and 2 match; samples 1 and 3 are 90 deg off but masked out
        tgt = torch.tensor([0.0, 100.0, 20.0, 120.0])
        res = axis_cosine_loss(pred, tgt, torch.tensor([True, False, True, False]))
        self.assertEqual(res.valid_count, 2)
        self.assertEqual(res.total_count, 4)
        _assert_close(self, res.loss, 0.0)

    def test_mean_is_over_valid_samples_only(self):
        pred = torch.stack((_v(0.0), _v(0.0), _v(0.0)))
        tgt = torch.tensor([0.0, 90.0, 0.0])
        # only the middle (loss=2) and the last (loss=0) are valid -> mean = 1.0
        res = axis_cosine_loss(pred, tgt, torch.tensor([False, True, True]))
        self.assertEqual(res.valid_count, 2)
        _assert_close(self, res.loss, 1.0)

    def test_all_masked_gives_zero_not_nan(self):
        pred, tgt = self._setup()
        pred = pred.clone().requires_grad_(True)
        res = axis_cosine_loss(pred, tgt, torch.zeros(4, dtype=torch.bool))
        self.assertEqual(res.valid_count, 0)
        self.assertTrue(res.all_masked)
        self.assertFalse(torch.isnan(res.loss))
        _assert_close(self, res.loss, 0.0)
        res.loss.backward()   # must not raise
        self.assertFalse(bool(torch.isnan(pred.grad).any()))
        self.assertTrue(bool((pred.grad == 0).all()))

    def test_empty_input(self):
        pred = torch.zeros(0, 2)
        tgt = torch.zeros(0)
        res = axis_cosine_loss(pred, tgt)
        self.assertEqual(res.valid_count, 0)
        _assert_close(self, res.loss, 0.0)

    def test_non_finite_target_dropped(self):
        pred = torch.stack((_v(0.0), _v(0.0)))
        tgt = torch.tensor([0.0, float("nan")])
        res = axis_cosine_loss(pred, tgt)
        self.assertEqual(res.valid_count, 1)
        self.assertEqual(res.dropped_nonfinite, 1)
        _assert_close(self, res.loss, 0.0)

    def test_non_finite_prediction_dropped(self):
        pred = torch.stack((_v(0.0), torch.tensor([float("nan"), float("nan")])))
        tgt = torch.tensor([0.0, 0.0])
        res = axis_cosine_loss(pred, tgt)
        self.assertEqual(res.valid_count, 1)
        self.assertEqual(res.dropped_nonfinite, 1)
        _assert_close(self, res.loss, 0.0)

    def test_degenerate_prediction_dropped(self):
        pred = torch.stack((_v(0.0), torch.zeros(2)))
        tgt = torch.tensor([0.0, 0.0])
        res = axis_cosine_loss(pred, tgt)
        self.assertEqual(res.valid_count, 1)
        self.assertEqual(res.dropped_degenerate, 1)

    def test_mask_shape_mismatch_rejected(self):
        pred, tgt = self._setup()
        with self.assertRaises(ValueError):
            axis_cosine_loss(pred, tgt, torch.tensor([True, False]))

    def test_scalar_target_and_mask_broadcast_to_single_sample(self):
        res = axis_cosine_loss(_v(0.0), torch.tensor(90.0),
                               valid=torch.tensor(False))
        self.assertEqual(res.valid_count, 0)
        _assert_close(self, res.loss, 0.0)
        res = axis_cosine_loss(_v(0.0), torch.tensor(90.0), valid=torch.tensor(True))
        self.assertEqual(res.valid_count, 1)
        _assert_close(self, res.loss, 2.0)

    def test_scalar_mask_broadcasts_over_batch(self):
        pred = torch.stack((_v(0.0), _v(0.0)))
        res = axis_cosine_loss(pred, torch.tensor([0.0, 90.0]), valid=torch.tensor(True))
        self.assertEqual(res.valid_count, 2)
        _assert_close(self, res.loss, 1.0)
        res = axis_cosine_loss(pred, torch.tensor([0.0, 90.0]), valid=torch.tensor(False))
        self.assertEqual(res.valid_count, 0)
        _assert_close(self, res.loss, 0.0)


class TestDualAxisLoss(unittest.TestCase):
    def test_total_is_mean_of_two_axes(self):
        pred_c, pred_r = _v(0.0), _v(0.0)
        # card is exact (0), reference is a quarter turn off (2) -> total = 1
        res = dual_axis_cosine_loss(pred_c, pred_r,
                                    torch.tensor(0.0), torch.tensor(90.0))
        _assert_close(self, res.card.loss, 0.0)
        _assert_close(self, res.ref.loss, 2.0)
        _assert_close(self, res.loss, 1.0)
        _assert_close(self, res.loss, (res.card.loss + res.ref.loss) / 2.0)

    def test_independent_masks(self):
        pred = torch.stack((_v(0.0), _v(0.0)))
        card_t = torch.tensor([0.0, 90.0])     # only sample 0 valid -> 0
        ref_t = torch.tensor([90.0, 0.0])      # only sample 1 valid -> 0
        res = dual_axis_cosine_loss(
            pred, pred, card_t, ref_t,
            card_valid=torch.tensor([True, False]),
            ref_valid=torch.tensor([False, True]))
        self.assertEqual(res.card.valid_count, 1)
        self.assertEqual(res.ref.valid_count, 1)
        _assert_close(self, res.card.loss, 0.0)
        _assert_close(self, res.ref.loss, 0.0)
        _assert_close(self, res.loss, 0.0)

    def test_one_axis_fully_masked(self):
        pred = torch.stack((_v(0.0), _v(0.0)))
        res = dual_axis_cosine_loss(
            pred, pred, torch.tensor([0.0, 0.0]), torch.tensor([90.0, 90.0]),
            card_valid=torch.zeros(2, dtype=torch.bool),
            ref_valid=torch.ones(2, dtype=torch.bool))
        self.assertTrue(res.card.all_masked)
        _assert_close(self, res.card.loss, 0.0)
        _assert_close(self, res.ref.loss, 2.0)
        # spec formula is fixed: no renormalisation -> (0 + 2) / 2 = 1
        _assert_close(self, res.loss, 1.0)
        self.assertFalse(torch.isnan(res.loss))

    def test_both_axes_fully_masked(self):
        pred = torch.stack((_v(0.0), _v(0.0))).requires_grad_(True)
        res = dual_axis_cosine_loss(
            pred, pred, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]),
            card_valid=torch.zeros(2, dtype=torch.bool),
            ref_valid=torch.zeros(2, dtype=torch.bool))
        _assert_close(self, res.loss, 0.0)
        self.assertFalse(torch.isnan(res.loss))
        res.loss.backward()
        self.assertFalse(bool(torch.isnan(pred.grad).any()))

    def test_masks_none_defaults_to_all_valid(self):
        pred = torch.stack((_v(5.0), _v(5.0)))
        res = dual_axis_cosine_loss(pred, pred,
                                    torch.tensor([5.0, 5.0]), torch.tensor([5.0, 5.0]))
        self.assertEqual(res.card.valid_count, 2)
        self.assertEqual(res.ref.valid_count, 2)
        _assert_close(self, res.loss, 0.0)

    def test_summary_has_no_nan(self):
        pred = torch.stack((_v(0.0), torch.zeros(2)))
        res = dual_axis_cosine_loss(pred, pred,
                                    torch.tensor([0.0, float("nan")]),
                                    torch.tensor([float("nan"), 30.0]))
        summary = res.summary()
        for k, val in summary.items():
            self.assertTrue(math.isfinite(float(val)), msg=k)


class TestGradient(unittest.TestCase):
    def _loss_through_angle(self, pred_deg: float, tgt_deg: float):
        theta = torch.tensor(float(pred_deg), requires_grad=True)
        loss = axis_cosine_loss(angle_to_vector(theta), torch.tensor(float(tgt_deg))).loss
        loss.backward()
        return float(loss), theta.grad

    def test_gradient_matches_analytic(self):
        # loss = 1 - cos(2*(tau - target)), tau in DEGREES:
        #   dL/dtau_deg = 2*sin(2*(tau-target)) * pi/180
        deg2rad = math.pi / 180.0
        for tau, target in ((10.0, 0.0), (0.0, 20.0), (55.0, 30.0), (80.0, 0.0)):
            _, grad = self._loss_through_angle(tau, target)
            expected = 2.0 * math.sin(math.radians(2.0 * (tau - target))) * deg2rad
            _assert_close(self, grad, expected, tol=1e-6,
                          msg=f"grad at tau={tau}, target={target}")

    def test_gradient_pushes_toward_target(self):
        # gradient descent: tau <- tau - lr*grad, so the sign must point back to the target
        _, grad = self._loss_through_angle(15.0, 5.0)     # target below tau -> grad > 0
        self.assertGreater(grad, 0.0)
        _, grad = self._loss_through_angle(5.0, 15.0)     # target above tau -> grad < 0
        self.assertLess(grad, 0.0)
        _, grad = self._loss_through_angle(-15.0, -5.0)   # target above tau -> grad < 0
        self.assertLess(grad, 0.0)

    def test_zero_gradient_at_exact_match(self):
        _, grad = self._loss_through_angle(37.0, 37.0)
        _assert_close(self, grad, 0.0, tol=1e-6)

    def test_gradient_reaches_both_axes(self):
        c = torch.tensor(0.0, requires_grad=True)
        r = torch.tensor(0.0, requires_grad=True)
        res = dual_axis_cosine_loss(angle_to_vector(c), angle_to_vector(r),
                                    torch.tensor(20.0), torch.tensor(60.0))
        res.loss.backward()
        self.assertIsNotNone(c.grad)
        self.assertIsNotNone(r.grad)
        # both angles sit at 0 and must move UP toward their targets -> negative grad
        self.assertLess(float(c.grad), 0.0)
        self.assertLess(float(r.grad), 0.0)
        # each axis contributes with weight 1/2
        deg2rad = math.pi / 180.0
        expected = 2.0 * math.sin(math.radians(-40.0)) * deg2rad / 2.0
        _assert_close(self, c.grad, expected, tol=1e-6)
        expected_r = 2.0 * math.sin(math.radians(-120.0)) * deg2rad / 2.0
        _assert_close(self, r.grad, expected_r, tol=1e-6)

    def test_gradient_respects_mask(self):
        # misaligned predictions so that every valid sample has a non-zero gradient
        pred = torch.stack((_v(0.0), _v(50.0)))
        card = pred.clone().requires_grad_(True)
        ref = pred.clone().requires_grad_(True)
        res = dual_axis_cosine_loss(
            card, ref, torch.tensor([20.0, 30.0]), torch.tensor([20.0, 80.0]),
            card_valid=torch.tensor([False, True]),
            ref_valid=torch.tensor([True, False]))
        res.loss.backward()
        for g in (card.grad, ref.grad):
            self.assertFalse(bool(torch.isnan(g).any()))
        # sample 0 is valid only for reference, sample 1 only for card
        self.assertEqual(float(card.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(ref.grad[1].abs().sum()), 0.0)
        self.assertGreater(float(card.grad[1].abs().sum()), 0.0)
        self.assertGreater(float(ref.grad[0].abs().sum()), 0.0)

    def test_gradient_finite_on_empty_axis(self):
        c = torch.tensor(0.0, requires_grad=True)
        r = torch.tensor(0.0, requires_grad=True)
        res = dual_axis_cosine_loss(angle_to_vector(c), angle_to_vector(r),
                                    torch.tensor(10.0), torch.tensor(10.0),
                                    card_valid=torch.tensor(True),
                                    ref_valid=torch.tensor(False))
        res.loss.backward()
        self.assertTrue(math.isfinite(float(r.grad)))
        self.assertTrue(math.isfinite(float(c.grad)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
