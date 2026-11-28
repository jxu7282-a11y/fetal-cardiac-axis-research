"""M1.1 Trainer: M1 dual-axis supervision with SPATIAL auxiliary pooling.

Data reading spec: CARDIAC_AXIS_DATA_READING_SPEC_v1.0

The single model change relative to M1 is the auxiliary head pooling:

    M1   : AdaptiveAvgPool2d(1)        -> Flatten -> Linear(C, 2)
    M1.1 : AdaptiveAvgPool2d((4, 4))   -> Flatten -> Linear(16*C, 2)

To guarantee that nothing else drifts, this trainer subclasses the archived M1
trainer and inherits, unchanged:

  * data pipeline (`M1DataLoader`), so image/mask/augmentation and the
    double-angle axis targets are bit-identical to M1;
  * the auxiliary loss (`dual_axis_cosine_loss`), its validity masking and the
    `0.1 * (L_card + L_ref) / 2` weighting;
  * `train_step`, epoch logging, checkpointing;
  * `num_epochs = 300`, and every hyper-parameter from the E0 parent.

Only `build_network_architecture` is overridden, and only the auxiliary head
construction inside the wrapper differs. The E0 `PlainConvUNet` backbone, its
initialisation order, and the segmentation output are untouched: the aux heads
are created *after* the backbone and are never used at inference.
"""
from __future__ import annotations

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch import nn

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

try:  # loaded as top-level package by nnUNet's nnUNet_extTrainer lookup
    from M1_measurement_aware_dual_axis.m1_trainer import (
        nnUNetTrainer_M1DualAxis,
    )
except ImportError:  # loaded as experiments.<pkg> (unit tests, direct import)
    from experiments.M1_measurement_aware_dual_axis.m1_trainer import (
        nnUNetTrainer_M1DualAxis,
    )

from .dual_axis_aux_heads import POOL_SIZE, SpatialDualAxisAuxHeads

__all__ = ["M1_1SpatialDualAxisUNet", "nnUNetTrainer_M1_1SpatialDualAxis"]


class M1_1SpatialDualAxisUNet(nn.Module):
    """E0 PlainConvUNet + two 4x4-spatial-pooled axis heads.

    ``forward`` returns ONLY the original segmentation output (inherited from the
    wrapped backbone), so inference and post-processing are untouched.
    ``forward_with_aux`` additionally returns the two axis vectors for training.
    """

    def __init__(self, backbone: PlainConvUNet, pool_size: int = POOL_SIZE):
        super().__init__()
        if not isinstance(backbone, PlainConvUNet):
            raise RuntimeError(
                f"M1.1 expects the E0 PlainConvUNet, got {type(backbone).__name__}")
        self.backbone = backbone
        self.pool_size = int(pool_size)
        self.aux_heads = SpatialDualAxisAuxHeads(self._final_channels(), self.pool_size)
        self._captured = None
        # Decoder stages run low -> high resolution; stages[-1] feeds seg_layers[-1].
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


class nnUNetTrainer_M1_1SpatialDualAxis(nnUNetTrainer_M1DualAxis):
    """M1 trainer with `AdaptiveAvgPool2d((4,4))` auxiliary heads."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device=None):
        if device is None:
            import torch
            device = torch.device("cpu")
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.aux_pool_size = POOL_SIZE

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        # Identical backbone construction to M1/E0; only the wrapper differs.
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
                f"M1.1 expects the E0 PlainConvUNet, got {type(backbone).__name__}")
        return M1_1SpatialDualAxisUNet(backbone, POOL_SIZE)
