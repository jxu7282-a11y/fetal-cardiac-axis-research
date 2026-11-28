"""CPU tests: M1.1 wrapper only changes the auxiliary heads.

Run with:
  python -m unittest experiments.M1_1_spatial_dual_axis.tests.test_m1_1_trainer -v
"""
from __future__ import annotations

import os
import sys
import unittest

import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from experiments.M1_1_spatial_dual_axis.m1_1_trainer import (  # noqa: E402
    M1_1SpatialDualAxisUNet,
    nnUNetTrainer_M1_1SpatialDualAxis,
)
from experiments.M1_measurement_aware_dual_axis.m1_trainer import (  # noqa: E402
    M1DualAxisUNet,
    nnUNetTrainer_M1DualAxis,
)
from experiments.M1_measurement_aware_dual_axis.dual_axis_angle_loss import (  # noqa: E402
    dual_axis_cosine_loss,
)


def _tiny_backbone(num_classes=7, ds=True, in_ch=3):
    return PlainConvUNet(
        input_channels=in_ch, n_stages=3, features_per_stage=[4, 8, 16],
        conv_op=nn.Conv2d, kernel_sizes=3, strides=[1, 2, 2],
        n_conv_per_stage=1, num_classes=num_classes,
        n_conv_per_stage_decoder=1, conv_bias=True,
        norm_op=nn.InstanceNorm2d, norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None, nonlin=nn.LeakyReLU, nonlin_kwargs={"inplace": True},
        deep_supervision=ds)


class TestBackboneIsUnchanged(unittest.TestCase):
    def test_same_seed_gives_identical_backbone(self):
        torch.manual_seed(1234)
        m1 = M1DualAxisUNet(_tiny_backbone())
        torch.manual_seed(1234)
        m11 = M1_1SpatialDualAxisUNet(_tiny_backbone())

        a = m1.backbone.state_dict()
        b = m11.backbone.state_dict()
        self.assertEqual(sorted(a), sorted(b))
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), f"backbone tensor differs: {k}")

    def test_segmentation_forward_is_identical(self):
        torch.manual_seed(7)
        m1 = M1DualAxisUNet(_tiny_backbone())
        m11 = M1_1SpatialDualAxisUNet(_tiny_backbone())
        m11.backbone.load_state_dict(m1.backbone.state_dict())
        x = torch.randn(2, 3, 64, 64)
        m1.eval(); m11.eval()
        with torch.no_grad():
            out_m1 = m1(x)
            out_m11 = m11(x)
        for a, b in zip(out_m1, out_m11):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_forward_with_aux_does_not_change_segmentation(self):
        torch.manual_seed(8)
        net = M1_1SpatialDualAxisUNet(_tiny_backbone()).eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            plain = net(x)
            seg_aux, card, ref = net.forward_with_aux(x)
        for a, b in zip(plain, seg_aux):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertEqual(tuple(card.shape), (2, 2))
        self.assertEqual(tuple(ref.shape), (2, 2))


class TestAuxHeadDiff(unittest.TestCase):
    def test_only_aux_head_weights_change(self):
        torch.manual_seed(9)
        m1 = M1DualAxisUNet(_tiny_backbone())
        torch.manual_seed(9)
        m11 = M1_1SpatialDualAxisUNet(_tiny_backbone())

        sd_m1 = m1.state_dict()
        sd_m11 = m11.state_dict()
        self.assertEqual(sorted(sd_m1), sorted(sd_m11))

        # Everything outside the auxiliary heads must be bit-identical: the
        # backbone, decoder and segmentation heads are untouched.
        for k in sd_m1:
            if k.startswith("aux_heads."):
                continue
            self.assertTrue(torch.equal(sd_m1[k], sd_m11[k]),
                            f"unexpected change outside aux heads: {k}")

        # Inside the aux heads, only the FC layer exists; its weight widens and
        # its bias value can shift because a wider Linear consumes a different
        # RNG stream. No other aux tensor may appear or disappear.
        self.assertEqual(
            sorted(k for k in sd_m1 if k.startswith("aux_heads.")),
            ["aux_heads.card.fc.bias", "aux_heads.card.fc.weight",
             "aux_heads.ref.fc.bias", "aux_heads.ref.fc.weight"])

        c = m11._final_channels()
        for head in ("card", "ref"):
            key = f"aux_heads.{head}.fc.weight"
            self.assertEqual(tuple(sd_m1[key].shape), (2, c))
            self.assertEqual(tuple(sd_m11[key].shape), (2, 16 * c))
            # bias shape is unchanged; its value may differ because a wider
            # Linear consumes a different RNG stream for its weight draw.
            self.assertEqual(tuple(sd_m11[f"aux_heads.{head}.fc.bias"].shape), (2,))

    def test_parameter_count_delta(self):
        torch.manual_seed(10)
        m1 = M1DualAxisUNet(_tiny_backbone())
        m11 = M1_1SpatialDualAxisUNet(_tiny_backbone())
        c = m1._final_channels()
        delta = sum(p.numel() for p in m11.parameters()) - sum(p.numel() for p in m1.parameters())
        # 2 heads * 2 output rows * (16C - C) extra weights
        self.assertEqual(delta, 2 * 2 * 15 * c)

    def test_final_feature_and_pooling(self):
        net = M1_1SpatialDualAxisUNet(_tiny_backbone()).eval()
        with torch.no_grad():
            net.forward_with_aux(torch.randn(1, 3, 64, 64))
        feat = net._captured
        self.assertEqual(feat.shape[1], net._final_channels())
        self.assertEqual(tuple(net.aux_heads.card.pooled(feat).shape[-2:]), (4, 4))


class TestLossAndGradients(unittest.TestCase):
    def test_aux_loss_and_gradients_are_finite(self):
        torch.manual_seed(11)
        net = M1_1SpatialDualAxisUNet(_tiny_backbone())
        x = torch.randn(2, 3, 64, 64)
        seg, card, ref = net.forward_with_aux(x)
        target = torch.zeros(2, 2, 2)
        target[:, 0, 0] = 1.0
        target[:, 1, 1] = 1.0
        valid = torch.ones(2, 2, dtype=torch.bool)
        res = dual_axis_cosine_loss(card, ref, target[:, 0], target[:, 1],
                                    card_valid=valid[:, 0], ref_valid=valid[:, 1],
                                    target_is_angle=False)
        self.assertTrue(torch.isfinite(res.loss))
        (seg[0].sum() * 1e-3 + 0.1 * res.loss).backward()
        for name, p in net.aux_heads.named_parameters():
            self.assertIsNotNone(p.grad, f"no grad for {name}")
            self.assertTrue(torch.isfinite(p.grad).all())
        enc_w = net.encoder.stages[0][0].convs[0].all_modules[0].weight
        self.assertIsNotNone(enc_w.grad)
        self.assertTrue(torch.isfinite(enc_w.grad).all())

    def test_loss_helpers_are_inherited_from_m1(self):
        self.assertTrue(issubclass(nnUNetTrainer_M1_1SpatialDualAxis, nnUNetTrainer_M1DualAxis))
        self.assertIs(nnUNetTrainer_M1_1SpatialDualAxis._aux_loss,
                      nnUNetTrainer_M1DualAxis._aux_loss)
        self.assertIs(nnUNetTrainer_M1_1SpatialDualAxis.train_step,
                      nnUNetTrainer_M1DualAxis.train_step)


if __name__ == "__main__":
    unittest.main()
