"""CPU tests for the M1 DataLoader (axis targets + E0 augmentation fidelity).

Run with:
  python -m unittest experiments.M1_measurement_aware_dual_axis.tests.test_m1_dataloader -v
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from experiments.M1_measurement_aware_dual_axis.dual_axis_augmentation import (  # noqa: E402
    angle_deg_to_double_angle,
    double_angle_to_angle_deg,
)
from experiments.M1_measurement_aware_dual_axis.m1_dataloader import (  # noqa: E402
    M1DataLoader,
    load_axis_angles,
)

_PATCH = 96


class _FakeDataset:
    """Minimal stand-in for nnUNetDataset: constant image + one label blob."""

    def __init__(self, keys, shape=(2, 1, 48, 64), fg_label=4):
        self.identifiers = list(keys)
        self.shape = shape
        self.fg_label = fg_label

    def load_case(self, key):
        data = np.random.RandomState(0).rand(*self.shape).astype(np.float32) + 0.5
        seg = np.zeros(self.shape, dtype=np.int16)
        seg[0, 0, 0:40, 0:50] = self.fg_label
        props = {"class_locations": {self.fg_label: np.array([[0, 0, 10, 10]])}}
        return data, seg, None, props


class _LabelManager:
    all_labels = [1, 2, 3, 4, 5, 6]
    has_ignore_label = False
    ignore_label = None


def _make_loader(cls, keys, transforms, axis_map=None, batch=2):
    kwargs = dict(data=_FakeDataset(keys), batch_size=batch,
                  patch_size=(_PATCH, _PATCH), final_patch_size=(_PATCH, _PATCH),
                  label_manager=_LabelManager(), transforms=transforms)
    if cls is M1DataLoader:
        return cls(axis_angle_by_key=axis_map, **kwargs)
    return cls(**kwargs)


def _fixed_affine(angle_rad, scales=(1.0, 1.0)):
    """A real SpatialTransform whose sampled angle/scale are deterministic.

    ``sample_scalar`` returns the tuple endpoint itself when both ends are
    equal, so ``get_parameters`` still builds the genuine grid + affine.
    """
    return SpatialTransform((_PATCH, _PATCH), patch_center_dist_from_border=0,
                            random_crop=False, p_rotation=1.0,
                            rotation=(angle_rad, angle_rad), p_scaling=1.0,
                            scaling=(scales[0], scales[0]) if scales[0] == scales[1]
                            else scales,
                            p_synchronize_scaling_across_axes=1)


def _no_affine():
    return SpatialTransform((_PATCH, _PATCH), patch_center_dist_from_border=0,
                            random_crop=False, p_rotation=0.0, p_scaling=0.0)


class _FixedMirror(MirrorTransform):
    def __init__(self, axes):
        super().__init__((0, 1))
        self._axes = list(axes)

    def get_parameters(self, **data_dict):
        return {"axes": list(self._axes)}


class TestAxisTargets(unittest.TestCase):
    def test_shapes_and_dtypes(self):
        keys = ["a", "b"]
        axis = {"a": (10.0, 50.0), "b": (30.0, 80.0)}
        dl = _make_loader(M1DataLoader, keys, ComposeTransforms([]), axis, batch=2)
        batch = dl.generate_train_batch()
        self.assertEqual(tuple(batch["axis_targets"].shape), (2, 2, 2))
        self.assertEqual(tuple(batch["axis_valid"].shape), (2, 2))
        self.assertEqual(batch["axis_targets"].dtype, torch.float32)
        self.assertEqual(batch["axis_valid"].dtype, torch.bool)
        # nnUNetDataLoader samples with replacement, so keys may repeat and are
        # not required to cover every input key in a single batch.
        self.assertEqual(len(batch["keys"]), 2)
        self.assertTrue(set(batch["keys"]).issubset(set(keys)))

    def test_no_grid_in_output(self):
        dl = _make_loader(M1DataLoader, ["a"], ComposeTransforms([]),
                          {"a": (10.0, 20.0)}, batch=1)
        batch = dl.generate_train_batch()
        self.assertNotIn("grid", batch)
        self.assertNotIn("aux_params", batch)

    def test_identity_augmentation_preserves_angles(self):
        dl = _make_loader(M1DataLoader, ["a"], ComposeTransforms([]),
                          {"a": (17.0, 63.0)}, batch=1)
        batch = dl.generate_train_batch()
        got = double_angle_to_angle_deg(batch["axis_targets"][0])
        self.assertAlmostEqual(float(got[0]), 17.0, places=3)
        self.assertAlmostEqual(float(got[1]), 63.0, places=3)
        self.assertTrue(bool(batch["axis_valid"].all()))

    def test_affine_from_transform_is_applied(self):
        tr = ComposeTransforms([_fixed_affine(math.radians(30.0))])
        dl = _make_loader(M1DataLoader, ["a"], tr, {"a": (20.0, 70.0)}, batch=1)
        batch = dl.generate_train_batch()
        got = double_angle_to_angle_deg(batch["axis_targets"][0])
        self.assertAlmostEqual(float(got[0]), 50.0, places=2)
        self.assertAlmostEqual(float(got[1]), 100.0, places=2)

    def test_mirror_from_transform_is_applied(self):
        tr = ComposeTransforms([_FixedMirror([0])])
        dl = _make_loader(M1DataLoader, ["a"], tr, {"a": (20.0, 70.0)}, batch=1)
        batch = dl.generate_train_batch()
        got = double_angle_to_angle_deg(batch["axis_targets"][0])
        self.assertAlmostEqual(float(got[0]), 160.0, places=2)
        self.assertAlmostEqual(float(got[1]), 110.0, places=2)

    def test_missing_gt_masks_only_that_axis(self):
        axis = {"a": (25.0, float("nan"))}
        dl = _make_loader(M1DataLoader, ["a"], ComposeTransforms([]), axis, batch=1)
        batch = dl.generate_train_batch()
        self.assertTrue(bool(batch["axis_valid"][0, 0]))
        self.assertFalse(bool(batch["axis_valid"][0, 1]))
        self.assertTrue(torch.allclose(batch["axis_targets"][0, 1], torch.zeros(2)))

    def test_unknown_case_is_fully_masked(self):
        dl = _make_loader(M1DataLoader, ["zzz"], ComposeTransforms([]), {}, batch=1)
        batch = dl.generate_train_batch()
        self.assertFalse(bool(batch["axis_valid"].any()))
        self.assertTrue(torch.allclose(batch["axis_targets"], torch.zeros_like(batch["axis_targets"])))

    def test_targets_are_unit_vectors_where_valid(self):
        dl = _make_loader(M1DataLoader, ["a"], ComposeTransforms([]),
                          {"a": (11.0, 155.0)}, batch=1)
        batch = dl.generate_train_batch()
        norm = torch.linalg.vector_norm(batch["axis_targets"][0], dim=-1)
        self.assertTrue(torch.allclose(norm, torch.ones_like(norm), atol=1e-5))


class TestE0Fidelity(unittest.TestCase):
    """The M1 loader must not perturb the E0 image/mask augmentation stream."""

    def test_image_and_mask_match_stock_loader_with_same_seed(self):
        keys = ["a", "b", "c"]
        axis = {k: (12.0, 44.0) for k in keys}
        for angles in (0.0, 0.4, -0.7):
            tr = ComposeTransforms([_fixed_affine(angles, (0.9, 1.2)), _FixedMirror([1])])
            np.random.seed(1234)
            torch.manual_seed(1234)
            base = _make_loader(nnUNetDataLoader, keys, tr, None, batch=3).generate_train_batch()
            np.random.seed(1234)
            torch.manual_seed(1234)
            m1 = _make_loader(M1DataLoader, keys, tr, axis, batch=3).generate_train_batch()
            self.assertTrue(torch.allclose(base["data"], m1["data"], atol=1e-6),
                            f"image differs at angle {angles}")
            self.assertEqual(list(base["keys"]), list(m1["keys"]))
            if isinstance(base["target"], list):
                for a, b in zip(base["target"], m1["target"]):
                    self.assertTrue(torch.equal(a, b))
            else:
                self.assertTrue(torch.equal(base["target"], m1["target"]))

    def test_duck_types_as_stock_loader(self):
        dl = _make_loader(M1DataLoader, ["a"], ComposeTransforms([]), {"a": (1.0, 2.0)}, batch=1)
        self.assertIsInstance(dl, nnUNetDataLoader)
        batch = dl.generate_train_batch()
        self.assertEqual(set(batch) - {"axis_targets", "axis_valid"},
                         {"data", "target", "keys"})


class TestPrepareAxisMap(unittest.TestCase):
    def test_load_axis_angles_from_csv_pair(self):
        with tempfile.TemporaryDirectory() as d:
            gt = os.path.join(d, "gt.csv")
            cm = os.path.join(d, "map.csv")
            with open(gt, "w", encoding="utf-8") as fh:
                fh.write("uid,manual_card_axis,manual_ref_axis\n")
                fh.write("u1|Ne|retrospective,12.5,48.0\n")
                fh.write("u2|Ne|retrospective,,\n")
            with open(cm, "w", encoding="utf-8") as fh:
                fh.write("uid,nnunet_key\n")
                fh.write("u1|Ne|retrospective,card_0000\n")
                fh.write("u2|Ne|retrospective,card_0001\n")
            out = load_axis_angles(gt, cm)
        self.assertEqual(sorted(out), ["card_0000", "card_0001"])
        self.assertAlmostEqual(out["card_0000"][0], 12.5)
        self.assertTrue(math.isnan(out["card_0001"][0]))
        self.assertTrue(math.isnan(out["card_0001"][1]))


if __name__ == "__main__":
    unittest.main()
