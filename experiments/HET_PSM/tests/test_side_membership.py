"""CPU tests: side membership targets, masked BCE, head shapes, gradients."""
from __future__ import annotations

import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from experiments.HET_PSM.side_membership import (  # noqa: E402
    SideMembershipHead,
    masked_side_bce,
    side_membership_targets,
)


def _seg(batch_maps):
    return torch.tensor(batch_maps, dtype=torch.long).unsqueeze(1)


class TestTargets(unittest.TestCase):
    def test_ids_1_to_4_map_to_left_right(self):
        seg = _seg([[
            [1, 2, 3, 4],
            [0, 0, 0, 0],
            [9, 7, 8, 5],
        ]])
        target, valid = side_membership_targets(seg)
        self.assertEqual(tuple(target.shape), (1, 2, 3, 4))
        self.assertEqual(tuple(valid.shape), (1, 1, 3, 4))
        left = target[0, 0]
        right = target[0, 1]
        self.assertEqual(left[0, 0].item(), 1.0)
        self.assertEqual(left[0, 1].item(), 1.0)
        self.assertEqual(right[0, 2].item(), 1.0)
        self.assertEqual(right[0, 3].item(), 1.0)
        self.assertEqual(valid[0, 0].sum().item(), 4.0)

    def test_ids_5_to_9_are_ignored(self):
        for ignored in (5, 6, 7, 8, 9):
            seg = _seg([[[ignored, ignored], [ignored, ignored]]])
            target, valid = side_membership_targets(seg)
            self.assertEqual(valid.sum().item(), 0.0, msg=f"ID {ignored} leaked")
            self.assertEqual(target.sum().item(), 0.0)

    def test_background_ignored(self):
        seg = _seg([[[0, 0], [0, 0]]])
        _, valid = side_membership_targets(seg)
        self.assertEqual(valid.sum().item(), 0.0)

    def test_mixed_patch_keeps_only_1_to_4(self):
        seg = _seg([[[1, 7], [8, 4]]])
        target, valid = side_membership_targets(seg)
        self.assertEqual(valid.sum().item(), 2.0)
        self.assertEqual(target[0, 0].sum().item(), 1.0)
        self.assertEqual(target[0, 1].sum().item(), 1.0)

    def test_batch_independence(self):
        seg = torch.zeros(3, 1, 4, 4, dtype=torch.long)
        seg[0, 0, 0, 0] = 1
        seg[2, 0, 1, 1] = 4
        _, valid = side_membership_targets(seg)
        self.assertEqual([valid[i].sum().item() for i in range(3)], [1.0, 0.0, 1.0])


class TestHead(unittest.TestCase):
    def test_shape_and_channels(self):
        for channels in (32, 512):
            head = SideMembershipHead(channels)
            self.assertEqual(head.conv.weight.shape, (2, channels, 1, 1))
            for batch in (1, 4):
                out = head(torch.randn(batch, channels, 13, 11))
                self.assertEqual(tuple(out.shape), (batch, 2, 13, 11))

    def test_channel_mismatch_raises(self):
        with self.assertRaises(ValueError):
            SideMembershipHead(8)(torch.randn(1, 7, 4, 4))


class TestMaskedBCE(unittest.TestCase):
    def test_empty_mask_is_differentiable_zero(self):
        logits = torch.randn(2, 2, 5, 5, requires_grad=True)
        seg = torch.zeros(2, 1, 5, 5, dtype=torch.long)
        loss, stats = masked_side_bce(logits, seg)
        self.assertEqual(stats["valid_cases"], 0)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)

    def test_single_axis_valid(self):
        logits = torch.zeros(1, 2, 4, 4, requires_grad=True)
        seg = torch.zeros(1, 1, 4, 4, dtype=torch.long)
        seg[0, 0, 0, 0] = 1
        loss, stats = masked_side_bce(logits, seg)
        self.assertEqual(stats["valid_cases"], 1)
        self.assertEqual(stats["valid_pixels"], 1)
        # one valid pixel supervises BOTH channels (left=1, right=0)
        self.assertAlmostEqual(float(loss), 2 * 0.693147, places=4)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertAlmostEqual(float(logits.grad[0, 0, 0, 0]), -0.5, places=5)
        # right channel is supervised as a hard 0 on the same valid pixel
        self.assertAlmostEqual(float(logits.grad[0, 1, 0, 0]), 0.5, places=5)

    def test_per_case_then_mean_of_cases(self):
        logits = torch.zeros(3, 2, 2, 2, requires_grad=True)
        seg = torch.zeros(3, 1, 2, 2, dtype=torch.long)
        seg[0, 0, 0, 0] = 1
        seg[2, 0, 0, 0] = 1
        seg[2, 0, 1, 1] = 1
        loss, stats = masked_side_bce(logits, seg)
        self.assertEqual(stats["valid_cases"], 2)
        self.assertEqual(stats["valid_pixels"], 3)
        self.assertAlmostEqual(float(loss), 2 * 0.693147, places=4)

    def test_no_nan_and_grad_flows(self):
        torch.manual_seed(0)
        seg = torch.randint(0, 10, (4, 1, 16, 16))
        head = SideMembershipHead(8)
        logits = head(torch.randn(4, 8, 16, 16, requires_grad=False))
        loss, stats = masked_side_bce(logits, seg)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in head.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertGreater(float(head.conv.weight.grad.abs().sum()), 0.0)

    def test_loss_prefers_correct_side(self):
        seg = torch.zeros(1, 1, 2, 2, dtype=torch.long)
        seg[0, 0, 0, :] = 1
        good = torch.full((1, 2, 2, 2), 6.0)
        good[:, 0, 0, :] = 6.0
        good[:, 1, 0, :] = -6.0
        bad = -good
        loss_good, _ = masked_side_bce(good, seg)
        loss_bad, _ = masked_side_bce(bad, seg)
        self.assertLess(float(loss_good), 0.01)
        self.assertGreater(float(loss_bad), 10.0)


if __name__ == "__main__":
    unittest.main()
