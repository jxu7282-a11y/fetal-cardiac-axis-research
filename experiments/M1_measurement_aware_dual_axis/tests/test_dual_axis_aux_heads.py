"""CPU unit tests for the M1 dual-axis auxiliary heads.

Run with:
  python -m unittest experiments.M1_measurement_aware_dual_axis.tests.test_dual_axis_aux_heads -v
No GPU, no nnU-Net, no training.
"""
from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from experiments.M1_measurement_aware_dual_axis.dual_axis_aux_heads import (  # noqa: E402
    AxisHead,
    DualAxisAuxHeads,
)


class TestAxisHeadShape(unittest.TestCase):
    def test_output_shape_various_batch_sizes(self):
        for channels in (8, 512):
            for batch in (1, 3, 7):
                head = AxisHead(channels)
                feat = torch.randn(batch, channels, 5, 7)
                out = head(feat)
                self.assertEqual(tuple(out.shape), (batch, 2))

    def test_dual_heads_shape(self):
        heads = DualAxisAuxHeads(16)
        out = heads(torch.randn(4, 16, 9, 9))
        self.assertIsInstance(out, tuple)
        card, ref = out
        self.assertEqual(tuple(card.shape), (4, 2))
        self.assertEqual(tuple(ref.shape), (4, 2))

    def test_rejects_non_4d_input(self):
        head = AxisHead(4)
        for bad in (torch.randn(4), torch.randn(2, 4), torch.randn(2, 4, 3, 3, 3)):
            with self.assertRaises(ValueError):
                head(bad)

    def test_rejects_channel_mismatch(self):
        head = AxisHead(4)
        with self.assertRaises(ValueError):
            head(torch.randn(2, 5, 3, 3))

    def test_rejects_invalid_channels(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                AxisHead(bad)


class TestAxisHeadIsolation(unittest.TestCase):
    def test_heads_have_independent_parameters(self):
        heads = DualAxisAuxHeads(8)
        self.assertIsNot(heads.card.fc, heads.ref.fc)
        card_ids = {id(p) for p in heads.card.parameters()}
        ref_ids = {id(p) for p in heads.ref.parameters()}
        self.assertEqual(card_ids & ref_ids, set())

    def test_head_parameters_are_only_pool_and_linear(self):
        head = AxisHead(8)
        names = sorted(n for n, _ in head.named_parameters())
        self.assertEqual(names, ["fc.bias", "fc.weight"])

    def test_aux_heads_do_not_touch_other_modules(self):
        heads = DualAxisAuxHeads(8)
        self.assertEqual(len(list(heads.parameters())), 4)


class TestAxisHeadNumerics(unittest.TestCase):
    def test_output_is_finite(self):
        torch.manual_seed(0)
        head = AxisHead(32)
        for scale in (1e-3, 1.0, 1e3):
            out = head(torch.randn(5, 32, 4, 4) * scale)
            self.assertTrue(torch.isfinite(out).all())

    def test_pooling_is_global_average(self):
        head = AxisHead(2)
        with torch.no_grad():
            head.fc.weight.zero_()
            head.fc.bias.zero_()
            head.fc.weight[0, 0] = 1.0
        feat = torch.zeros(1, 2, 2, 3)
        feat[0, 0] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = head(feat)
        expected = feat[0, 0].mean()
        self.assertTrue(torch.allclose(out[0, 0], expected, atol=1e-6))
        self.assertTrue(torch.allclose(out[0, 1], torch.tensor(0.0), atol=1e-6))

    def test_output_is_not_normalized(self):
        head = AxisHead(4)
        with torch.no_grad():
            head.fc.weight.fill_(3.0)
            head.fc.bias.fill_(2.0)
        feat = torch.ones(6, 4, 3, 3)
        out = head(feat)
        expected = torch.full((6, 2), 4 * 3.0 + 2.0)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5))
        norms = torch.linalg.vector_norm(out, dim=-1)
        self.assertTrue(torch.all(norms > 1.5))


class TestGradientPropagation(unittest.TestCase):
    def test_both_heads_receive_gradients(self):
        torch.manual_seed(1)
        heads = DualAxisAuxHeads(16)
        feat = torch.randn(4, 16, 6, 6, requires_grad=True)
        card, ref = heads(feat)
        loss = card.sum() + ref.sum()
        loss.backward()

        for name, param in heads.named_parameters():
            self.assertIsNotNone(param.grad, f"no gradient for {name}")
            self.assertTrue(torch.isfinite(param.grad).all(), f"non-finite grad for {name}")
        self.assertIsNotNone(feat.grad)
        self.assertTrue(torch.isfinite(feat.grad).all())

    def test_each_head_gets_its_own_gradient(self):
        torch.manual_seed(2)
        heads = DualAxisAuxHeads(8)
        feat = torch.randn(3, 8, 4, 4)
        card, _ = heads(feat)
        card.sum().backward()
        self.assertIsNotNone(heads.card.fc.weight.grad)
        self.assertIsNone(heads.ref.fc.weight.grad)

    def test_gradient_reaches_every_channel(self):
        torch.manual_seed(3)
        head = AxisHead(5)
        out = head(torch.randn(2, 5, 3, 3))
        out.sum().backward()
        grad = head.fc.weight.grad
        self.assertEqual(tuple(grad.shape), (2, 5))
        self.assertTrue(torch.all(grad.abs().sum(dim=0) > 0))


class TestIntegrationWithAngleLoss(unittest.TestCase):
    def test_heads_feed_the_existing_loss(self):
        from experiments.M1_measurement_aware_dual_axis.dual_axis_angle_loss import (
            dual_axis_cosine_loss,
        )

        heads = DualAxisAuxHeads(8)
        feat = torch.randn(4, 8, 5, 5, requires_grad=True)
        card, ref = heads(feat)
        result = dual_axis_cosine_loss(
            card, ref,
            torch.tensor([10.0, 20.0, 30.0, 40.0]),
            torch.tensor([50.0, 60.0, 70.0, 80.0]),
        )
        self.assertTrue(torch.isfinite(result.loss))
        result.loss.backward()
        self.assertIsNotNone(feat.grad)
        self.assertTrue(torch.isfinite(feat.grad).all())


if __name__ == "__main__":
    unittest.main()
