"""HET-PSM mid-trimester pretraining: HET-PSM U-Net, 10 labels, L_seg + 0.1*L_side."""
from __future__ import annotations

import numpy as np
import torch
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

from .het_psm_unet import HETPSMPlainConvUNet
from .side_membership import masked_side_bce

__all__ = ["nnUNetTrainer_HETPSMMid"]

SIDE_WEIGHT = 0.1
EXPECTED_LABEL_IDS = tuple(range(10))


def unwrap(net):
    from torch._dynamo import OptimizedModule
    from torch.nn.parallel import DistributedDataParallel as DDP
    if isinstance(net, DDP):
        net = net.module
    if isinstance(net, OptimizedModule):
        net = net._orig_mod
    return net


def full_resolution_target(target):
    if isinstance(target, list):
        return target[0]
    return target


class HETPSMBase(nnUNetTrainer):
    """Shared network construction, loss and train step for HET-PSM."""

    side_weight = SIDE_WEIGHT

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 300
        self._last_side_stats = {}

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        """Same architecture as the plans describe, but with the side head added."""
        import pydoc
        kwargs = dict(configuration_manager.network_arch_init_kwargs)
        for key in configuration_manager.network_arch_init_kwargs_req_import:
            value = kwargs.get(key)
            if isinstance(value, str):
                kwargs[key] = pydoc.locate(value)
        if not configuration_manager.network_arch_class_name.endswith("PlainConvUNet"):
            raise RuntimeError(
                "HET-PSM requires PlainConvUNet, plans say "
                f"{configuration_manager.network_arch_class_name}")
        net = HETPSMPlainConvUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            **kwargs)
        net.apply(net.initialize)
        return net

    def _side_loss(self, side_logits, target):
        loss, stats = masked_side_bce(side_logits, full_resolution_target(target))
        self._last_side_stats = stats
        return loss

    def _run_train_step(self, batch):
        data = batch["data"]
        target = batch["target"]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        ctx = (autocast(self.device.type, enabled=True)
               if self.device.type == "cuda" else dummy_context())
        with ctx:
            net = unwrap(self.network)
            seg_out, side_logits = net.forward_with_side(data)
            l_seg = self.loss(seg_out, target)
            l_side = self._side_loss(side_logits, target)
            loss = l_seg + self.side_weight * l_side

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            "loss": loss.detach().cpu().numpy(),
            "loss_seg": float(l_seg.detach().cpu()),
            "loss_side": float(l_side.detach().cpu()),
            "side_valid_cases": self._last_side_stats.get("valid_cases", 0),
        }

    def train_step(self, batch):
        return self._run_train_step(batch)

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        side = [o["loss_side"] for o in train_outputs if "loss_side" in o]
        if side:
            self.print_to_log_file(
                f"side membership loss (mean over batches): {float(np.mean(side)):.4f} "
                f"(weight {self.side_weight})")


class nnUNetTrainer_HETPSMMid(HETPSMBase):
    """Mid-trimester HET-PSM pretraining; identical schedule to E1/E3 (300 epochs)."""

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def initialize(self):
        labels = self.dataset_json.get("labels", {})
        ids = sorted(int(v) for v in labels.values())
        if tuple(ids) != EXPECTED_LABEL_IDS:
            raise RuntimeError(
                f"HET-PSM mid pretraining expects the original 10 class IDs 0-9 "
                f"with ID8 kept as its own class, got {ids}")
        super().initialize()
        net = unwrap(self.network)
        self.print_to_log_file(
            f"HET-PSM network: final decoder channels={net.side_feature_channels}, "
            f"side head params={sum(p.numel() for p in net.side_head.parameters())}")
