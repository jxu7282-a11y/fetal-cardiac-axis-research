"""M1 DataLoader: nnUNetDataLoader that also returns double-angle axis targets.

Why a subclass is required
--------------------------
``nnUNetDataLoader.generate_train_batch`` calls ``ComposeTransforms`` once per
sample, which internally calls ``t(**data_dict)`` (get_parameters + apply) and
throws the parameters away. The M1 axis labels need the *actual* affine and
mirror axes used for that sample, so the transform loop is reproduced here and
each transform's ``get_parameters`` is called exactly once, with ``apply``
reusing that same parameter dict. Random draws therefore stay in the original
order: with the same seed the produced image/mask are bit-identical to E0.

Cropping is not a problem for an axis label
-------------------------------------------
``get_bbox`` only translates the sampling window (plus zero padding), and a
translation cannot change the orientation of a line. Rotation, scaling and
mirroring are the only operations that change it, and all three are captured
from the transform parameters themselves.
"""
from __future__ import annotations

import csv
import math
from typing import Mapping, Sequence

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader

from .dual_axis_augmentation import (
    angle_deg_to_double_angle,
    apply_forward_matrix_to_double_angle,
    apply_mirror_to_double_angle,
    forward_matrix_xy,
)

__all__ = ["M1DataLoader", "load_axis_angles"]

NUM_AXES = 2  # 0 = card, 1 = reference


def load_axis_angles(gt_csv: str, case_map_csv: str) -> dict[str, tuple[float, float]]:
    """Map nnU-Net case key -> (manual_card_axis, manual_ref_axis) in degrees.

    Missing or non-finite angles are stored as NaN so the loader can mark the
    sample invalid without dropping it from segmentation training.
    """
    uid_to_key: dict[str, str] = {}
    with open(case_map_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            uid_to_key[row["uid"]] = row["nnunet_key"]

    out: dict[str, tuple[float, float]] = {}
    with open(gt_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = uid_to_key.get(row.get("uid", ""))
            if key is None:
                continue
            try:
                card = float(row.get("manual_card_axis", "nan"))
                ref = float(row.get("manual_ref_axis", "nan"))
            except (TypeError, ValueError):
                card = ref = float("nan")
            out[key] = (card, ref)
    return out


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


class M1DataLoader(nnUNetDataLoader):
    """nnUNetDataLoader + per-sample transformed double-angle axis targets.

    Extra batch fields
    ------------------
    ``axis_targets`` : float32 [B, 2, 2]  (axis 0 = card, axis 1 = reference;
                        last dim = [cos 2theta, sin 2theta])
    ``axis_valid``   : bool   [B, 2]
    ``keys``         : unchanged list of nnU-Net case identifiers
    """

    def __init__(self, *args, axis_angle_by_key: Mapping[str, Sequence[float]] | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.axis_angle_by_key = {str(k): v for k, v in (axis_angle_by_key or {}).items()}

    # ------------------------------------------------------------------ helpers
    def _transform_list(self):
        t = self.transforms
        if isinstance(t, ComposeTransforms):
            return list(t.transforms)
        if t is None:
            return []
        return [t]

    def _raw_axis_target(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (double-angle vectors [2,2], validity [2]) for one case."""
        angles = self.axis_angle_by_key.get(str(key))
        if angles is None:
            return np.zeros((NUM_AXES, 2), dtype=np.float32), np.zeros(NUM_AXES, dtype=bool)
        values = []
        valid = []
        for a in angles:
            if _finite(a):
                values.append(angle_deg_to_double_angle(float(a)))
                valid.append(True)
            else:
                values.append(torch.zeros(2))
                valid.append(False)
        vec = torch.stack(values).numpy().astype(np.float32)
        return vec, np.asarray(valid, dtype=bool)

    @staticmethod
    def _apply_ops(vec: torch.Tensor, ops) -> tuple[torch.Tensor, bool]:
        out = vec
        ok = True
        for kind, payload in ops:
            if kind == "affine":
                out = apply_forward_matrix_to_double_angle(out, forward_matrix_xy(payload))
            else:
                out = apply_mirror_to_double_angle(out, payload)
            if not bool(torch.isfinite(out).all()):
                ok = False
        return out, ok

    # -------------------------------------------------------------- batch build
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        transforms = self._transform_list()
        data_all = None
        seg_all = None
        axis_all = None
        valid_all = None

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, key in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)

                    data, seg, seg_prev, properties = self._data.load_case(key)
                    shape = data.shape[1:]
                    bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties["class_locations"])
                    bbox = [[i, k] for i, k in zip(bbox_lbs, bbox_ubs)]

                    data_cropped = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                    seg_cropped = torch.from_numpy(
                        crop_and_pad_nd(seg, bbox, -1, cast_cropped_to=np.int16)).to(torch.int16)
                    if seg_prev is not None:
                        seg_prev_cropped = torch.from_numpy(
                            crop_and_pad_nd(seg_prev, bbox, -1, cast_cropped_to=np.int16)).to(torch.int16)
                        seg_cropped = torch.cat((seg_cropped, seg_prev_cropped[None]), dim=0)

                    if self.patch_size_was_2d:
                        data_cropped = data_cropped[:, 0]
                        seg_cropped = seg_cropped[:, 0]

                    sample = {"image": data_cropped, "segmentation": seg_cropped}
                    ops = []
                    for t in transforms:
                        params = t.get_parameters(**sample) if hasattr(t, "get_parameters") else {}
                        if isinstance(t, SpatialTransform) and params.get("affine") is not None:
                            ops.append(("affine", np.asarray(params["affine"], dtype=np.float64)))
                        elif isinstance(t, MirrorTransform):
                            axes = list(params.get("axes", []))
                            if axes:
                                ops.append(("mirror", axes))
                        sample = t.apply(sample, **params)

                    axis_vec, axis_ok = self._raw_axis_target(key)
                    moved, move_ok = self._apply_ops(
                        torch.from_numpy(axis_vec), ops)

                    if data_all is None:
                        data_all = torch.empty((self.batch_size, *sample["image"].shape), dtype=torch.float32)
                        axis_all = torch.zeros((self.batch_size, NUM_AXES, 2), dtype=torch.float32)
                        valid_all = torch.zeros((self.batch_size, NUM_AXES), dtype=torch.bool)
                    data_all[j] = sample["image"]
                    axis_all[j] = moved.to(torch.float32)
                    valid_all[j] = torch.from_numpy(axis_ok) & move_ok

                    seg_sample = sample["segmentation"]
                    if isinstance(seg_sample, list):
                        if seg_all is None:
                            seg_all = [torch.empty((self.batch_size, *s.shape), dtype=s.dtype) for s in seg_sample]
                        for s_idx, s in enumerate(seg_sample):
                            seg_all[s_idx][j] = s
                    else:
                        if seg_all is None:
                            seg_all = torch.empty((self.batch_size, *seg_sample.shape), dtype=seg_sample.dtype)
                        seg_all[j] = seg_sample

        axis_all = torch.where(valid_all.unsqueeze(-1), axis_all, torch.zeros_like(axis_all))
        return {
            "data": data_all,
            "target": seg_all,
            "keys": selected_keys,
            "axis_targets": axis_all,
            "axis_valid": valid_all,
        }
