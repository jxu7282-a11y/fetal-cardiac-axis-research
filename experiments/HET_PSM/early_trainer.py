"""HET-PSM early-trimester fine-tuning: E3 encoder-only transfer from HET-PSM mid.

Only the encoder is copied out of the HET-PSM mid-trimester checkpoint.  The
mid-trimester decoder (10 classes) and the side head are discarded; the early
network is a plain 7-class PlainConvUNet with freshly initialised decoder and
segmentation heads, exactly as in E3.  No side loss is used here.
"""
from __future__ import annotations

import json
import os

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from .encoder_transfer import load_network_weights, transfer_encoder_weights, unwrap_network

__all__ = ["nnUNetTrainer_HETPSMEarly"]


class nnUNetTrainer_HETPSMEarly(nnUNetTrainer):
    """E3-equivalent early fine-tuning, encoder initialised from HET-PSM mid."""

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        self.hetpsm_continue_training = bool(plans.get("continue_training", False))
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 300

    def initialize(self):
        super().initialize()

        if self.hetpsm_continue_training:
            self.print_to_log_file(
                "HET-PSM encoder transfer skipped: resuming (continue_training=True); "
                "weights come from the checkpoint")
            return

        ckpt = os.environ.get("HETPSM_MID_CHECKPOINT")
        if not ckpt:
            raise RuntimeError("HETPSM_MID_CHECKPOINT is not set")
        if not os.path.isfile(ckpt):
            raise RuntimeError(f"HETPSM_MID_CHECKPOINT does not exist: {ckpt}")

        net = unwrap_network(self.network)
        mid = load_network_weights(ckpt)
        audit = transfer_encoder_weights(net, mid)
        audit.update({
            "early_net": net.__class__.__name__,
            "early_has_side_head": hasattr(net, "side_head"),
            "mid_checkpoint": os.path.abspath(ckpt),
            "mid_decoder_and_side_head_used": False,
            "first_encoder_keys": sorted(audit["encoder_loaded_names"])[:3],
        })
        self.print_to_log_file(
            f"HET-PSM encoder transfer OK: {audit['encoder_state_dict_entries']} entries "
            f"({audit['encoder_unique_parameter_objects']} unique tensors); "
            f"decoder/seg untouched ({audit['decoder_and_seg_state_dict_entries']})")
        self._write_audit(audit)

    def _audit_path(self):
        folder = os.environ.get("HETPSM_AUDIT_DIR", "local/het_psm/encoder_audit")
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"encoder_transfer_fold_{self.fold}.json")

    def _write_audit(self, audit):
        path = self._audit_path()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(audit, fh, indent=2, ensure_ascii=False)
        self.print_to_log_file(f"encoder audit -> {path}")

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        net = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision)
        if hasattr(net, "side_head"):
            raise RuntimeError(
                "early network must not carry the mid-trimester side head")
        return net
