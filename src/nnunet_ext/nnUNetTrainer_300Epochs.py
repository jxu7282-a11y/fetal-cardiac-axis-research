"""E0 baseline trainer: stock nnUNetTrainer with num_epochs=300.

SOLE DIFFERENCE from nnunetv2.training.nnUNetTrainer.nnUNetTrainer:
    self.num_epochs = 300   (stock default: 1000)

Everything else -- network architecture, loss, optimizer, initial_lr,
batch size, patch size, augmentation, plans, preprocessing, validation
scheme, save_every -- is inherited unchanged.

Note: PolyLRScheduler is constructed with self.num_epochs, so the learning
rate schedule is automatically rescaled to the 300-epoch horizon. That is a
consequence of changing num_epochs, not an additional modification.

Used for formal experiment E0 (early-trimester, 2D, scratch, 5-fold).
"""
from __future__ import annotations

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_300Epochs(nnUNetTrainer):
    """nnUNetTrainer with num_epochs=300 (E0 baseline)."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # ---- the ONLY deviation from stock nnUNetTrainer ----
        self.num_epochs = 300
