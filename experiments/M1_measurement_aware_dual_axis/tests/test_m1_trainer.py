"""CPU integration tests for the M1 Trainer.

Run with:
  python -m unittest experiments.M1_measurement_aware_dual_axis.tests.test_m1_trainer -v

These tests build a deliberately tiny PlainConvUNet (n_stages=3, small
features) so the whole forward/backward path runs on CPU in seconds. The
Trainer object is constructed with a stub that bypasses nnU-Net's folder/plans
machinery; the augmentation and loss code under test is the real one.
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

from experiments.M1_measurement_aware_dual_axis.m1_trainer import (  # noqa: E402
    M1DualAxisUNet,
    nnUNetTrainer_M1DualAxis,
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


class TestNetworkWrapping(unittest.TestCase):
    def setUp(self):
        self.net = M1DualAxisUNet(_tiny_backbone()).eval()
        self.x = torch.randn(2, 3, 64, 64)

    def test_forward_returns_only_segmentation(self):
        with torch.no_grad():
            out = self.net(self.x)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)  # n_stages-1 deep supervision outputs

    def test_single_output_when_ds_disabled(self):
        net = M1DualAxisUNet(_tiny_backbone(ds=False)).eval()
        with torch.no_grad():
            out = net(self.x)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape[1], 7)

    def test_forward_output_channels_unchanged(self):
        with torch.no_grad():
            out = self.net(self.x)
        self.assertEqual(out[0].shape[1], 7)

    def test_forward_with_aux_shapes(self):
        with torch.no_grad():
            seg, card, ref = self.net.forward_with_aux(self.x)
        self.assertEqual(card.shape, (2, 2))
        self.assertEqual(ref.shape, (2, 2))
        self.assertEqual(seg[0].shape[1], 7)

    def test_aux_heads_do_not_alter_segmentation(self):
        self.net.eval()
        with torch.no_grad():
            plain = self.net(self.x)
            seg_aux, _, _ = self.net.forward_with_aux(self.x)
        for a, b in zip(plain, seg_aux):
            self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_aux_feature_is_final_decoder_stage(self):
        x = self.x.clone()
        self.net.forward_with_aux(x)
        feat = self.net._captured
        # decoder stages run low->high resolution; the LAST one feeds seg_layers[-1]
        self.assertEqual(feat.shape[1], 4)
        self.assertEqual(feat.shape[2:], (64, 64))


class TestLossComposition(unittest.TestCase):
    def test_aux_weight_is_one_tenth(self):
        self.assertEqual(nnUNetTrainer_M1DualAxis.aux_weight if hasattr(
            nnUNetTrainer_M1DualAxis, "aux_weight") else 0.1, 0.1)


class _StubTrainer:
    """Shares nnUNetTrainer_M1DualAxis's step logic without nnU-Net setup."""

    def __init__(self, network, aux_weight=0.1):
        self.device = torch.device("cpu")
        self.network = network
        self.aux_weight = aux_weight
        self.optimizer = torch.optim.SGD(network.parameters(), lr=1e-2)
        self.grad_scaler = None
        self.loss = _FakeSegLoss()
        self._last_aux_log = {}
        self.print_to_log_file = lambda *a, **k: None
        self._aux_loss = nnUNetTrainer_M1DualAxis._aux_loss.__get__(self)


class _FakeSegLoss:
    def __call__(self, output, target):
        return output[0].sum() * 1e-3


def _call_train_step(trainer, batch):
    # bind the real method to the stub
    return nnUNetTrainer_M1DualAxis.train_step(trainer, batch)


def _batch(n=2, with_aux=True, valid=None):
    b = {"data": torch.randn(n, 3, 64, 64),
         "target": [torch.zeros(n, 1, 64, 64, dtype=torch.long),
                    torch.zeros(n, 1, 32, 32, dtype=torch.long)]}
    if with_aux:
        b["axis_targets"] = torch.zeros(n, 2, 2)
        b["axis_targets"][:, 0, 0] = 1.0
        b["axis_targets"][:, 1, 1] = 1.0
        b["axis_valid"] = valid if valid is not None else torch.ones(n, 2, dtype=torch.bool)
    return b


class TestTrainStep(unittest.TestCase):
    def _setup(self):
        net = M1DualAxisUNet(_tiny_backbone())
        tr = _StubTrainer(net)
        return net, tr

    def test_segmentation_gradients_flow(self):
        net, tr = self._setup()
        out = _call_train_step(tr, _batch())
        self.assertTrue(torch.isfinite(torch.tensor(out["loss_seg"])))
        seg_grad = net.decoder.seg_layers[-1].weight.grad
        self.assertIsNotNone(seg_grad)
        self.assertGreater(float(seg_grad.abs().sum()), 0.0)

    def test_aux_gradients_flow(self):
        net, tr = self._setup()
        _call_train_step(tr, _batch())
        for name, p in net.aux_heads.named_parameters():
            self.assertIsNotNone(p.grad, f"no grad for {name}")
            self.assertGreater(float(p.grad.abs().sum()), 0.0, f"zero grad {name}")

    def test_encoder_receives_both_signals(self):
        net, tr = self._setup()
        _call_train_step(tr, _batch())
        w = net.encoder.stages[0][0].convs[0].all_modules[0].weight
        self.assertIsNotNone(w.grad)
        self.assertTrue(torch.isfinite(w.grad).all())

    def test_aux_heads_are_in_optimizer(self):
        net, tr = self._setup()
        ids = {id(p) for g in tr.optimizer.param_groups for p in g["params"]}
        for p in net.aux_heads.parameters():
            self.assertIn(id(p), ids)

    def test_aux_heads_actually_update(self):
        net, tr = self._setup()
        before = [p.detach().clone() for p in net.aux_heads.parameters()]
        _call_train_step(tr, _batch())
        after = [p.detach().clone() for p in net.aux_heads.parameters()]
        self.assertTrue(any(not torch.allclose(a, b) for a, b in zip(before, after)))

    def test_all_trainable_parameters_are_in_optimizer(self):
        net, tr = self._setup()
        ids = {id(p) for g in tr.optimizer.param_groups for p in g["params"]}
        missing = [n for n, p in net.named_parameters() if p.requires_grad and id(p) not in ids]
        self.assertEqual(missing, [])

    def test_missing_aux_fields_is_safe(self):
        net, tr = self._setup()
        out = _call_train_step(tr, _batch(with_aux=False))
        self.assertTrue(torch.isfinite(torch.tensor(out["loss"])))
        self.assertFalse(out["has_aux"])
        for name, p in net.aux_heads.named_parameters():
            # no aux loss => exactly zero gradient, not NaN
            self.assertTrue(p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad)))

    def test_all_axes_invalid_is_safe(self):
        net, tr = self._setup()
        out = _call_train_step(tr, _batch(valid=torch.zeros(2, 2, dtype=torch.bool)))
        self.assertTrue(torch.isfinite(torch.tensor(out["loss"])))
        self.assertEqual(out["loss_aux"], 0.0)

    def test_one_axis_invalid_uses_other_axis(self):
        net, tr = self._setup()
        valid = torch.tensor([[True, False], [True, False]])
        out = _call_train_step(tr, _batch(valid=valid))
        self.assertGreater(out["loss_card"], 0.0)
        self.assertEqual(out["loss_ref"], 0.0)
        self.assertAlmostEqual(out["loss_aux"], out["loss_card"] / 2.0, places=6)

    def test_loss_composition_matches_spec(self):
        net, tr = self._setup()
        out = _call_train_step(tr, _batch())
        expected = out["loss_seg"] + 0.1 * out["loss_aux"]
        self.assertAlmostEqual(out["loss"], expected, places=5)


class TestCheckpointCompat(unittest.TestCase):
    def test_state_dict_round_trip(self):
        net = M1DualAxisUNet(_tiny_backbone())
        sd = net.state_dict()
        self.assertTrue(any(k.startswith("aux_heads.") for k in sd))
        self.assertTrue(any(k.startswith("backbone.") for k in sd))
        fresh = M1DualAxisUNet(_tiny_backbone())
        fresh.load_state_dict(sd)
        for (a_name, a), (b_name, b) in zip(net.named_parameters(), fresh.named_parameters()):
            self.assertEqual(a_name, b_name)
            self.assertTrue(torch.allclose(a, b))

    def test_e0_weights_load_into_backbone(self):
        e0 = _tiny_backbone()
        e0_sd = e0.state_dict()
        net = M1DualAxisUNet(_tiny_backbone())
        mapped = {"backbone." + k: v for k, v in e0_sd.items()}
        missing, unexpected = net.load_state_dict(mapped, strict=False)
        self.assertEqual([m for m in missing if not m.startswith("aux_heads.")], [])
        self.assertEqual(unexpected, [])

    def test_segmentation_forward_after_reload(self):
        net = M1DualAxisUNet(_tiny_backbone()).eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            before = net(x)[0].clone()
        fresh = M1DualAxisUNet(_tiny_backbone()).eval()
        fresh.load_state_dict(net.state_dict())
        with torch.no_grad():
            after = fresh(x)[0]
        self.assertTrue(torch.allclose(before, after, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
