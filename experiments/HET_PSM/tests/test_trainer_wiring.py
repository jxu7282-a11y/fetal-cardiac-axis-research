"""CPU tests: trainer construction, one training step, loss weighting."""
from __future__ import annotations

import os
import sys
import unittest

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss  # noqa: E402
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper  # noqa: E402
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: E402

from experiments.HET_PSM.early_trainer import nnUNetTrainer_HETPSMEarly  # noqa: E402
from experiments.HET_PSM.mid_trainer import nnUNetTrainer_HETPSMMid  # noqa: E402

ARCH = dict(
    n_stages=3,
    features_per_stage=(8, 16, 32),
    conv_op=torch.nn.Conv2d,
    kernel_sizes=[[3, 3]] * 3,
    strides=[[1, 1], [2, 2], [2, 2]],
    n_conv_per_stage=[2] * 3,
    n_conv_per_stage_decoder=[2] * 2,
    conv_bias=True,
    norm_op=torch.nn.InstanceNorm2d,
    norm_op_kwargs={"eps": 1e-5, "affine": True},
    nonlin=torch.nn.LeakyReLU,
    nonlin_kwargs={"inplace": True},
)


class _Cfg:
    network_arch_class_name = (
        "dynamic_network_architectures.architectures.unet.PlainConvUNet")
    network_arch_init_kwargs = dict(ARCH)
    network_arch_init_kwargs_req_import = []


def _trainer(num_classes=10):
    trainer = object.__new__(nnUNetTrainer_HETPSMMid)
    trainer.network = nnUNetTrainer_HETPSMMid.build_network_architecture(
        None, _Cfg(), 3, num_classes, True)
    trainer.device = torch.device("cpu")
    trainer.side_weight = 0.1
    trainer.optimizer = torch.optim.SGD(trainer.network.parameters(), lr=0.01)
    trainer.grad_scaler = None
    trainer._last_side_stats = {}
    base = DC_and_CE_loss(
        {"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": False},
        {"weight": None, "ignore_index": -100, "reduction": "mean"})
    trainer.loss = DeepSupervisionWrapper(base, weight_factors=[1.0, 0.5, 0.25])
    return trainer


def _batch():
    x = torch.randn(2, 3, 64, 64)
    g = torch.zeros(2, 1, 64, 64, dtype=torch.long)
    g[0, 0, 5:20, 5:20] = 1
    g[0, 0, 25:40, 25:40] = 3
    g[0, 0, 45:50, 45:50] = 8
    g[1, 0, 10:30, 10:30] = 4
    g[1, 0, 35:50, 35:50] = 2
    return {"data": x, "target": [g, g[:, :, ::2, ::2], g[:, :, ::4, ::4]]}


class TestTrainers(unittest.TestCase):
    def test_subclass_and_epochs(self):
        self.assertTrue(issubclass(nnUNetTrainer_HETPSMMid, nnUNetTrainer))
        self.assertTrue(issubclass(nnUNetTrainer_HETPSMEarly, nnUNetTrainer))
        self.assertEqual(nnUNetTrainer_HETPSMMid.side_weight, 0.1)

    def test_early_trainer_has_no_side_head(self):
        net = nnUNetTrainer_HETPSMEarly.build_network_architecture(
            None, _Cfg(), 3, 7, True)
        self.assertFalse(hasattr(net, "side_head"))
        self.assertEqual(len(net.decoder.seg_layers), 2)

    def test_early_trainer_rejects_side_network(self):
        class Cfg(_Cfg):
            network_arch_class_name = (
                "experiments.HET_PSM.het_psm_unet.HETPSMPlainConvUNet")
        with self.assertRaises(RuntimeError):
            nnUNetTrainer_HETPSMEarly.build_network_architecture(
                None, Cfg(), 3, 7, True)

    def test_one_training_step_is_finite_and_weighted(self):
        trainer = _trainer()
        out = trainer._run_train_step(_batch())
        self.assertTrue(torch.isfinite(torch.tensor(out["loss"])))
        self.assertEqual(out["side_valid_cases"], 2)
        self.assertGreater(out["loss_side"], 0.0)
        expected = out["loss_seg"] + 0.1 * out["loss_side"]
        self.assertAlmostEqual(out["loss"], expected, places=4)

        side_grad = trainer.network.side_head.conv.weight.grad
        seg_grad = trainer.network.decoder.seg_layers[-1].weight.grad
        self.assertIsNotNone(side_grad)
        self.assertGreater(float(side_grad.abs().sum()), 0.0)
        self.assertGreater(float(seg_grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(side_grad).all())

    def test_all_empty_side_mask_keeps_training_alive(self):
        trainer = _trainer()
        batch = _batch()
        batch["target"][0][:] = 8
        batch["target"][1][:] = 8
        batch["target"][2][:] = 8
        out = trainer._run_train_step(batch)
        self.assertEqual(out["side_valid_cases"], 0)
        self.assertEqual(out["loss_side"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(out["loss"])))
        self.assertGreater(float(
            trainer.network.decoder.seg_layers[-1].weight.grad.abs().sum()), 0.0)

    def test_side_head_does_not_receive_seg_only_output(self):
        trainer = _trainer()
        data = torch.randn(1, 3, 64, 64)
        seg, side = trainer.network.forward_with_side(data)
        self.assertEqual(side.shape[1], 2)
        self.assertEqual(side.shape[-2:], seg[0].shape[-2:])


if __name__ == "__main__":
    unittest.main()
