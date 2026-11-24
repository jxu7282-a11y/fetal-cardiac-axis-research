"""Undirected line angle utilities (spec sections 8 and 15)."""
from __future__ import annotations

import math

import numpy as np


def normalize_angle180(angle: float) -> float:
    """Normalize to [0, 180)."""
    if angle is None:
        return float("nan")
    angle = float(angle)
    if not math.isfinite(angle):
        return float("nan")
    return angle % 180.0


def line_angle_from_points(p1, p2) -> float:
    """Undirected line orientation in [0, 180); NaN if degenerate."""
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return float("nan")
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return float("nan")
    return normalize_angle180(math.degrees(math.atan2(dy, dx)))


def axial_abs_error(pred: float, gt: float) -> float:
    """Absolute undirected axis error in [0, 90]."""
    if pred is None or gt is None:
        return float("nan")
    pred, gt = float(pred), float(gt)
    if not (math.isfinite(pred) and math.isfinite(gt)):
        return float("nan")
    return abs((pred - gt + 90.0) % 180.0 - 90.0)


def signed_axial_diff(target: float, source: float) -> float:
    """Signed undirected difference in [-90, 90)."""
    if target is None or source is None:
        return float("nan")
    return (float(target) - float(source) + 90.0) % 180.0 - 90.0


def acute_angle_between_lines(a: float, b: float) -> float:
    """Smallest angle between two undirected axes, in [0, 90]."""
    if a is None or b is None:
        return float("nan")
    a, b = float(a), float(b)
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    return axial_abs_error(a, b)


def axial_weighted_mean(angles, weights) -> float:
    """Undirected weighted mean via double-angle representation."""
    angles = np.asarray(angles, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(angles) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    rad = np.deg2rad(2.0 * angles[valid])
    w = weights[valid]
    x = float(np.sum(w * np.cos(rad)))
    y = float(np.sum(w * np.sin(rad)))
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return float("nan")
    return normalize_angle180(math.degrees(0.5 * math.atan2(y, x)))
