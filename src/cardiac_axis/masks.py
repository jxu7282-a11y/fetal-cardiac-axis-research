"""Mask loading and fixed-geometry axis computation.

Data reading spec: CARDIAC_AXIS_DATA_READING_SPEC_v1.0
Method stage: fixed_geometry (id4/id5 OBB weighted card axis; id6->id1 centroid ref axis)
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .angles import axial_weighted_mean, line_angle_from_points

EXPECTED_LABEL_IDS = (1, 4, 5, 6)
_EIGHT_CONN = np.ones((3, 3), np.uint8)


class MaskError(ValueError):
    pass


def load_label_mask(path: str) -> np.ndarray:
    """Read a 2D integer label mask; hard-fail on ambiguous 3D input."""
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise MaskError(f"unreadable mask: {path}")
    if mask.ndim == 2:
        return mask.astype(np.int32, copy=False)
    if mask.ndim == 3:
        # Only accept a single-channel-repeat encoding, never an RGB visualization.
        if mask.shape[2] < 1:
            raise MaskError(f"empty mask channels: {path}")
        best = None
        for c in range(mask.shape[2]):
            chan = mask[..., c]
            hits = int(np.isin(chan, EXPECTED_LABEL_IDS).sum())
            score = (hits, -int(np.unique(chan).size))
            if best is None or score > best[0]:
                best = (score, c, chan)
        (hits, _), channel, chan = best
        if hits <= 0:
            raise MaskError(f"no expected label ID in any channel: {path}")
        if not all(np.array_equal(chan, mask[..., c]) for c in range(mask.shape[2])):
            raise MaskError(f"3-channel mask is not a repeated label image: {path}")
        return chan.astype(np.int32, copy=False)
    raise MaskError(f"unsupported mask ndim={mask.ndim}: {path}")


def largest_component(binary: np.ndarray) -> np.ndarray | None:
    """Largest 8-connected foreground component, or None if empty."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8, copy=False), connectivity=8
    )
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8)


def _obb_long_axis(component: np.ndarray):
    """Return (angle_deg, area, aspect) of the OBB long edge, or None."""
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    if not (math.isfinite(w) and math.isfinite(h)) or min(w, h) <= 1e-6:
        return None
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    best = None
    for i in range(4):
        p1, p2 = box[i], box[(i + 1) % 4]
        length = float(np.hypot(*(p2 - p1)))
        if best is None or length > best[0]:
            best = (length, p1, p2)
    length, p1, p2 = best
    angle = line_angle_from_points(p1, p2)
    if not math.isfinite(angle):
        return None
    area = float(cv2.contourArea(contour))
    aspect = float(max(w, h) / min(w, h))
    return angle, area, aspect


def card_axis_from_mask(mask: np.ndarray) -> dict:
    """Fixed-geometry card axis from id4/id5 OBB long axes (spec 11)."""
    out = {
        "card_axis": float("nan"),
        "id4_angle": float("nan"), "id5_angle": float("nan"),
        "id4_area": 0.0, "id5_area": 0.0,
        "id4_aspect": float("nan"), "id5_aspect": float("nan"),
        "id4_weight": 0.0, "id5_weight": 0.0,
        "id4_present": False, "id5_present": False,
        "card_reason": "",
    }
    angles, weights = [], []
    for lid in (4, 5):
        comp = largest_component(mask == lid)
        if comp is None:
            continue
        out[f"id{lid}_present"] = True
        obb = _obb_long_axis(comp)
        if obb is None:
            continue
        angle, area, aspect = obb
        weight = area * max(aspect - 1.0, 0.05)
        out[f"id{lid}_angle"] = angle
        out[f"id{lid}_area"] = area
        out[f"id{lid}_aspect"] = aspect
        out[f"id{lid}_weight"] = weight
        angles.append(angle)
        weights.append(weight)
    if not angles:
        out["card_reason"] = "no_valid_id4_or_id5"
        return out
    out["card_axis"] = axial_weighted_mean(angles, weights)
    out["card_reason"] = "id4_id5_weighted" if len(angles) == 2 else "single_region"
    return out


def _centroid(component: np.ndarray):
    m = cv2.moments(component, binaryImage=True)
    if m["m00"] <= 0:
        return None
    return (m["m10"] / m["m00"], m["m01"] / m["m00"])


def reference_axis_from_mask(mask: np.ndarray) -> dict:
    """Fixed-geometry ref axis from id6 centroid -> id1 centroid (spec 12)."""
    out = {"ref_axis": float("nan"), "id1_present": False, "id6_present": False,
           "id1_centroid": (float("nan"), float("nan")),
           "id6_centroid": (float("nan"), float("nan")), "ref_reason": ""}
    comp6 = largest_component(mask == 6)
    comp1 = largest_component(mask == 1)
    out["id6_present"] = comp6 is not None
    out["id1_present"] = comp1 is not None
    if comp6 is None or comp1 is None:
        out["ref_reason"] = "missing_id1_or_id6"
        return out
    c6, c1 = _centroid(comp6), _centroid(comp1)
    if c6 is None or c1 is None:
        out["ref_reason"] = "degenerate_centroid"
        return out
    out["id6_centroid"], out["id1_centroid"] = c6, c1
    angle = line_angle_from_points(c6, c1)
    if not math.isfinite(angle):
        out["ref_reason"] = "degenerate_centroid_line"
        return out
    out["ref_axis"] = angle
    out["ref_reason"] = "id6_to_id1"
    return out


def axes_from_mask(mask: np.ndarray) -> dict:
    res = {}
    res.update(card_axis_from_mask(mask))
    res.update(reference_axis_from_mask(mask))
    return res
