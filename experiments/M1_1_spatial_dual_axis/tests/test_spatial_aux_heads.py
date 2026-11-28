"""CPU unit tests for the M1.1 spatial dual-axis auxiliary heads.

Run with:
  python -m unittest experiments.M1_1_spatial_dual_axis.tests.test_spatial_aux_heads -v
No GPU, no nnU-Net, no training.
"""
from __future__ import annotations

import os
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from experiments.M1_1_spatial_dual_axis.dual_axis_aux_heads import (  # noqa: E402
    POOL_SIZE,
    SpatialAxisHead,
    SpatialDualAxisAuxHeads,
)
from experiments.M1_measurement_aware_dual_axis.dual_axis_angle_loss import (  # noqa: E402
    angle_to_vector,
)


def _quadrant_feature(quadrant: int, channels: int = 2, size: int = 16) -> torch.Tensor:
    x = torch.zeros(1, channels, size, size)
    h = size // 2
    rows = (0, 0, h, h)[quadrant]
    cols = (0, h, 0, h)[quadrant]
    x[:, :, rows:rows + h, cols:cols + h] = 1.0
    return x


class TestPoolingContract(unittest.TestCase):
    def test_pool_size_is_4(self):
        self.assertEqual(POOL_SIZE, 4)

    def test_pool_keeps_4x4_grid(self):
        head = SpatialAxisHead(32)
        feat = torch.randn(3, 32, 96, 96)
        pooled = head.pooled(feat)
        self.assertEqual(tuple(pooled.shape), (3, 32, 4, 4))

    def test_fc_input_width_is_16x_channels(self):
        for channels in (8, 32, 512):
            head = SpatialAxisHead(channels)
            self.assertEqual(head.fc.in_features, channels * 16)

    def test_output_shape_and_finiteness(self):
        torch.manual_seed(0)
        for channels in (8, 32, 512):
            for batch in (1, 3, 7):
                head = SpatialAxisHead(channels)
                out = head(torch.randn(batch, channels, 48, 48))
                self.assertEqual(tuple(out.shape), (batch, 2))
                self.assertTrue(torch.isfinite(out).all())

    def test_dual_heads_are_independent(self):
        heads = SpatialDualAxisAuxHeads(16)
        card, ref = heads(torch.randn(4, 16, 9, 9))
        self.assertEqual(tuple(card.shape), (4, 2))
        self.assertEqual(tuple(ref.shape), (4, 2))
        cid = {id(p) for p in heads.card.parameters()}
        rid = {id(p) for p in heads.ref.parameters()}
        self.assertEqual(cid & rid, set())
        self.assertEqual(len(list(heads.parameters())), 4)

    def test_rejects_bad_input(self):
        head = SpatialAxisHead(4)
        with self.assertRaises(ValueError):
            head(torch.randn(4))
        with self.assertRaises(ValueError):
            head(torch.randn(2, 5, 3, 3))
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                SpatialAxisHead(bad)


class TestSpatialInformationIsPreserved(unittest.TestCase):
    """The whole point of M1.1: different spatial layouts must be distinguishable."""

    def test_pooled_vectors_differ_by_layout(self):
        head = SpatialAxisHead(2)
        pooled = [head.flatten(head.pooled(_quadrant_feature(q))) for q in range(4)]
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertFalse(torch.allclose(pooled[i], pooled[j]),
                                 f"layouts {i} and {j} collapse to the same pooled vector")

    def test_gap_would_collapse_the_same_layouts(self):
        # Documents the M1 failure mode this experiment removes.
        gap = nn.AdaptiveAvgPool2d(1)
        pooled = [gap(_quadrant_feature(q)).flatten(1) for q in range(4)]
        for p in pooled[1:]:
            self.assertTrue(torch.allclose(pooled[0], p))

    def test_head_can_learn_layout_dependent_axes(self):
        """With 4x4 pooling the head CAN fit four layout-dependent axis targets."""
        torch.manual_seed(0)
        head = SpatialAxisHead(1)
        feats = torch.cat([_quadrant_feature(q, channels=1) for q in range(4)], dim=0)
        targets = angle_to_vector(torch.tensor([0.0, 45.0, 90.0, 135.0]))

        opt = torch.optim.Adam(head.parameters(), lr=0.05)
        for _ in range(400):
            opt.zero_grad()
            out = head(feats)
            unit = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            loss = (1.0 - (unit * targets).sum(dim=-1)).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            out = head(feats)
            unit = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cos = (unit * targets).sum(dim=-1)
            axis_err_deg = torch.rad2deg(torch.arccos(cos.clamp(-1, 1))) / 2.0
        self.assertLess(float(axis_err_deg.max()), 5.0)

    def test_output_is_not_constant_across_inputs(self):
        torch.manual_seed(1)
        head = SpatialAxisHead(4)
        out = head(torch.randn(8, 4, 32, 32))
        self.assertGreater(float(out.std(dim=0).max()), 0.0)


class TestGradients(unittest.TestCase):
    def test_gradients_reach_heads_and_feature(self):
        torch.manual_seed(2)
        heads = SpatialDualAxisAuxHeads(16)
        feat = torch.randn(4, 16, 24, 24, requires_grad=True)
        card, ref = heads(feat)
        (card.sum() + ref.sum()).backward()
        for name, p in heads.named_parameters():
            self.assertIsNotNone(p.grad, f"no gradient for {name}")
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertIsNotNone(feat.grad)
        self.assertTrue(torch.isfinite(feat.grad).all())

    def test_each_head_gets_only_its_own_gradient(self):
        heads = SpatialDualAxisAuxHeads(8)
        card, _ = heads(torch.randn(3, 8, 16, 16))
        card.sum().backward()
        self.assertIsNotNone(heads.card.fc.weight.grad)
        self.assertIsNone(heads.ref.fc.weight.grad)

    def test_every_input_cell_influences_the_output(self):
        """No spatial cell may be dead: each pooled cell carries gradient."""
        torch.manual_seed(3)
        head = SpatialAxisHead(1)
        # A dense feature map: every 4x4 cell must influence the output.
        out = head(torch.randn(1, 1, 16, 16, requires_grad=True))
        out.sum().backward()
        grad_cells = head.fc.weight.grad.abs().sum(dim=0).reshape(4, 4)
        self.assertTrue(torch.all(grad_cells > 0))


if __name__ == "__main__":
    unittest.main()
