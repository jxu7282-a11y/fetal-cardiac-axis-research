"""M1 Trainer: E0 segmentation pipeline + dual-axis auxiliary supervision.

Inherits the E0 ``nnUNetTrainer_300Epochs`` unchanged. Only these additions:

* ``build_network_architecture`` wraps the E0 PlainConvUNet in ``M1DualAxisUNet``,
  which keeps the E0 forward path (and therefore ``network(x)`` returns exactly
  the original segmentation output) and adds a ``forward_with_aux(x)`` that also
  returns the two axis vectors.
* ``get_dataloaders`` swaps the training loader for ``M1DataLoader``; the
  validation loader is the stock ``nnUNetDataLoader``.
* ``train_step`` adds ``0.1 * (L_card + L_ref) / 2``.

Everything else (loss, deep supervision, optimizer, LR schedule, batch/patch
size, augmentation, epochs, checkpoint selection) is the E0 implementation.
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch import autocast, nn

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context

from .dual_axis_angle_loss import dual_axis_cosine_loss
from .dual_axis_aux_heads import DualAxisAuxHeads
from .m1_dataloader import M1DataLoader, load_axis_angles

__all__ = ["M1DualAxisUNet", "nnUNetTrainer_M1DualAxis"]

AUX_WEIGHT = 0.1


def _unwrap(net):
    from torch._dynamo import OptimizedModule
    from torch.nn.parallel import DistributedDataParallel as DDP
    if isinstance(net, DDP):
        net = net.module
    if isinstance(net, OptimizedModule):
        net = net._orig_mod
    return net


class M1DualAxisUNet(nn.Module):
    """E0 PlainConvUNet with two GAP+Linear axis heads on the final decoder feature.

    ``forward`` is inherited from PlainConvUNet and returns ONLY the original
    segmentation output, so inference and post-processing are untouched.
    """

    def __init__(self, backbone: PlainConvUNet):
        super().__init__()
        self.backbone = backbone
        self.aux_heads = DualAxisAuxHeads(self._final_channels())
        self._captured = None
        # Decoder stages run from low to high resolution, so stages[-1] is the
        # highest-resolution feature and is exactly what feeds seg_layers[-1].
        self.backbone.decoder.stages[-1].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._captured = output

    def _last_conv(self):
        stage = self.backbone.decoder.stages[-1]
        return stage.convs[-1]

    def _final_channels(self) -> int:
        return int(self._last_conv().conv.out_channels)

    @property
    def encoder(self):
        return self.backbone.encoder

    @property
    def decoder(self):
        return self.backbone.decoder

    def forward(self, x):
        return self.backbone(x)

    def forward_with_aux(self, x):
        self._captured = None
        seg = self.backbone(x)
        if self._captured is None:
            raise RuntimeError("failed to capture the final decoder feature")
        card, ref = self.aux_heads(self._captured)
        return seg, card, ref


class nnUNetTrainer_M1DualAxis(nnUNetTrainer):
    """E0 300-epoch trainer + M1 dual-axis auxiliary supervision."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cpu")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 300
        self.aux_weight = AUX_WEIGHT
        self._axis_angles = None
        self._last_aux_log = {}

    # ------------------------------------------------------------- network
    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        backbone = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)
        if not isinstance(backbone, PlainConvUNet):
            raise RuntimeError(
                f"M1 expects the E0 PlainConvUNet, got {type(backbone).__name__}")
        return M1DualAxisUNet(backbone)

    # ------------------------------------------------------------ data path
    def _axis_map(self):
        if self._axis_angles is None:
            base = self.preprocessed_dataset_folder_base
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            gt = os.path.join(root, "local", "phase0", "fixed_gt_v1.csv")
            cmap = os.path.join(root, "local", "phase0", "nnunet_case_map.csv")
            self._axis_angles = load_axis_angles(gt, cmap)
        return self._axis_angles

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        ds_scales = self._get_deep_supervision_scales()
        (rotation_for_DA, do_dummy_2d, initial_patch_size, mirror_axes
         ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, ds_scales, mirror_axes, do_dummy_2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions
            if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        val_transforms = self.get_validation_transforms(
            ds_scales, is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions
            if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = M1DataLoader(
            dataset_tr, self.batch_size, initial_patch_size,
            self.configuration_manager.patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            axis_angle_by_key=self._axis_map())
        dl_val = nnUNetDataLoader(
            dataset_val, self.batch_size, self.configuration_manager.patch_size,
            self.configuration_manager.patch_size, self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling)

        n = int(get_allowed_n_proc_DA())
        if n == 0:
            tr = SingleThreadedAugmenter(dl_tr, None)
            val = SingleThreadedAugmenter(dl_val, None)
        else:
            tr = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr, transform=None, num_processes=n,
                num_cached=max(6, n // 2), seeds=None,
                pin_memory=self.device.type == "cuda", wait_time=0.002)
            val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val, transform=None,
                num_processes=max(1, n // 2), num_cached=max(3, n // 4),
                seeds=None, pin_memory=self.device.type == "cuda",
                wait_time=0.002)
        _ = next(tr)
        _ = next(val)
        return tr, val

    # --------------------------------------------------------------- losses
    def _aux_loss(self, card_pred, ref_pred, batch):
        if "axis_targets" not in batch or "axis_valid" not in batch:
            zero = card_pred.sum() * 0.0
            self._last_aux_log = {"aux_present": False}
            return zero, zero, zero

        targets = batch["axis_targets"].to(self.device, non_blocking=True)
        valid = batch["axis_valid"].to(self.device, non_blocking=True)
        if targets.ndim != 3 or targets.shape[-1] != 2:
            raise ValueError(f"axis_targets must be [B,2,2], got {tuple(targets.shape)}")
        if valid.shape != targets.shape[:2]:
            raise ValueError(
                f"axis_valid {tuple(valid.shape)} != {tuple(targets.shape[:2])}")

        res = dual_axis_cosine_loss(
            card_pred, ref_pred,
            targets[:, 0, :], targets[:, 1, :],
            card_valid=valid[:, 0], ref_valid=valid[:, 1],
            target_is_angle=False)
        self._last_aux_log = {
            "aux_present": True,
            "aux_total": float(res.loss.detach().cpu()),
            "card_loss": float(res.card.loss.detach().cpu()),
            "ref_loss": float(res.ref.loss.detach().cpu()),
            "card_valid": res.card.valid_count,
            "ref_valid": res.ref.valid_count,
        }
        return res.loss, res.card.loss, res.ref.loss

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        target = batch["target"]
        has_aux = "axis_targets" in batch and "axis_valid" in batch

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with ctx:
            net = _unwrap(self.network)
            seg_out, card_pred, ref_pred = net.forward_with_aux(data)
            l_seg = self.loss(seg_out, target)
            l_aux, l_card, l_ref = self._aux_loss(card_pred, ref_pred, batch)
            loss = l_seg + self.aux_weight * l_aux

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

        out = {
            "loss": loss.detach().cpu().numpy(),
            "loss_seg": float(l_seg.detach().cpu()),
            "loss_aux": float(l_aux.detach().cpu()),
            "has_aux": has_aux,
        }
        if has_aux:
            out["loss_card"] = float(l_card.detach().cpu())
            out["loss_ref"] = float(l_ref.detach().cpu())
        return out

    def on_train_epoch_end(self, train_outputs: Sequence[dict]):
        super().on_train_epoch_end(train_outputs)
        aux = [o.get("loss_aux", 0.0) for o in train_outputs if o.get("has_aux")]
        if aux:
            self.print_to_log_file(f"aux loss (mean over batches): {float(np.mean(aux)):.4f}")

    # ---------------------------------------------------------- checkpoints
    def save_checkpoint(self, filename: str) -> None:
        super().save_checkpoint(filename)

    def load_checkpoint(self, filename_or_checkpoint) -> None:
        super().load_checkpoint(filename_or_checkpoint)
