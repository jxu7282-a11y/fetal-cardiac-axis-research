"""CPU tests: HET-PSM network wiring, DS invariance, encoder-key compatibility."""
from __future__ import annotations

import os
import sys
import unittest

import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from experiments.HET_PSM.het_psm_unet import HETPSMPlainConvUNet  # noqa: E402
from experiments.HET_PSM.encoder_transfer import (  # noqa: E402
    _is_encoder_key,
    transfer_encoder_weights,
)

ARCH = dict(
    n_stages=8,
    features_per_stage=(32, 64, 128, 256, 512, 512, 512, 512),
    conv_op=torch.nn.Conv2d,
    kernel_sizes=[[3, 3]] * 8,
    strides=[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
    n_conv_per_stage=[2] * 8,
    n_conv_per_stage_decoder=[2] * 7,
    conv_bias=True,
    norm_op=torch.nn.InstanceNorm2d,
    norm_op_kwargs={"eps": 1e-5, "affine": True},
    nonlin=torch.nn.LeakyReLU,
    nonlin_kwargs={"inplace": True},
)


def _build(cls, num_classes=10, ds=True):
    return cls(input_channels=3, num_classes=num_classes,
               deep_supervision=ds, **ARCH)


class TestArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.base = _build(PlainConvUNet).eval()
        torch.manual_seed(0)
        cls.het = _build(HETPSMPlainConvUNet).eval()

    def test_final_channels_read_from_module(self):
        self.assertEqual(self.het.side_feature_channels,
                         self.base.decoder.stages[-1].convs[-1].conv.out_channels)
        self.assertEqual(self.het.side_head.conv.weight.shape[1],
                         self.het.side_feature_channels)

    def test_state_dict_keys_identical_to_plain_conv_unet(self):
        base_keys = set(self.base.state_dict())
        het_keys = set(self.het.state_dict())
        self.assertEqual(base_keys - het_keys, set())
        extra = het_keys - base_keys
        self.assertTrue(extra)
        self.assertTrue(all(k.startswith("side_head.") for k in extra), extra)

    def test_encoder_keys_compatible(self):
        base_enc = {k for k in self.base.state_dict() if _is_encoder_key(k)}
        het_enc = {k for k in self.het.state_dict() if _is_encoder_key(k)}
        self.assertEqual(base_enc, het_enc)
        self.assertFalse(any("side_head" in k for k in het_enc))

    def test_segmentation_output_and_deep_supervision_unchanged(self):
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            torch.manual_seed(1)
            base_out = self.base(x)
            torch.manual_seed(1)
            het_out = self.het(x)
        self.assertIsInstance(het_out, list)
        self.assertEqual(len(het_out), len(base_out))
        for a, b in zip(base_out, het_out):
            self.assertEqual(tuple(a.shape), tuple(b.shape))
            self.assertTrue(torch.equal(a, b))

    def test_no_deep_supervision_returns_tensor(self):
        net = _build(HETPSMPlainConvUNet, ds=False).eval()
        with torch.no_grad():
            seg, side = net.forward_with_side(torch.randn(1, 3, 256, 256))
        self.assertIsInstance(seg, torch.Tensor)
        self.assertEqual(seg.shape[1], 10)
        self.assertEqual(side.shape[1], 2)
        self.assertEqual(side.shape[-2:], seg.shape[-2:])

    def test_side_head_does_not_change_seg_path(self):
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            before = self.het(x)
            with torch.no_grad():
                for p in self.het.side_head.parameters():
                    p.add_(3.0)
            after = self.het(x)
        for a, b in zip(before, after):
            self.assertTrue(torch.equal(a, b))

    def test_side_head_trains_with_grad(self):
        net = _build(HETPSMPlainConvUNet).train()
        seg, side = net.forward_with_side(torch.randn(1, 3, 256, 256))
        (seg[0].mean() + side.mean()).backward()
        w = net.side_head.conv.weight.grad
        self.assertIsNotNone(w)
        self.assertTrue(torch.isfinite(w).all())
        self.assertGreater(float(w.abs().sum()), 0.0)


class TestTransfer(unittest.TestCase):
    def test_encoder_transfer_ignores_side_head(self):
        torch.manual_seed(0)
        mid = _build(HETPSMPlainConvUNet).eval()
        mid_net = mid
        mid_sd = {k: v.clone() for k, v in mid_net.state_dict().items()}
        torch.manual_seed(7)
        early = _build(PlainConvUNet, num_classes=7).eval()

        before_decoder = {k: v.clone() for k, v in early.state_dict().items()
                          if not _is_encoder_key(k)}
        audit = transfer_encoder_weights(early, mid_sd)

        sd = early.state_dict()
        self.assertEqual(audit["encoder_entries_not_equal_to_mid"], [])
        for k, v in mid_sd.items():
            if _is_encoder_key(k):
                self.assertTrue(torch.equal(sd[k], v), k)
        self.assertFalse(any("side_head" in k for k in sd))
        for k, v in before_decoder.items():
            self.assertTrue(torch.equal(sd[k], v), f"decoder changed: {k}")
        self.assertEqual(audit["encoder_state_dict_entries"],
                         len([k for k in mid_sd if _is_encoder_key(k)]))

    def test_mid_with_side_head_extra_keys_do_not_break_transfer(self):
        torch.manual_seed(0)
        mid_sd = {k: v.clone() for k, v in _build(HETPSMPlainConvUNet).state_dict().items()}
        self.assertTrue(any(k.startswith("side_head.") for k in mid_sd))
        early = _build(PlainConvUNet, num_classes=7).eval()
        audit = transfer_encoder_weights(early, mid_sd)
        self.assertEqual(audit["encoder_entries_in_mid_but_not_in_net"], [])


if __name__ == "__main__":
    unittest.main()
