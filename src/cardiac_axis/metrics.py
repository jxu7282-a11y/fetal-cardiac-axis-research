"""Unified segmentation + geometric + clinical evaluation (spec 15, 23)."""
from __future__ import annotations

import numpy as np

from .angles import axial_abs_error, acute_angle_between_lines

LABEL_IDS = (1, 2, 3, 4, 5, 6)


def dice_iou(pred: np.ndarray, gold: np.ndarray, label_id: int):
    p = pred == label_id
    g = gold == label_id
    inter = int(np.logical_and(p, g).sum())
    ps, gs = int(p.sum()), int(g.sum())
    if ps + gs == 0:
        return float("nan"), float("nan")  # undefined for absent-in-both
    dice = 2.0 * inter / (ps + gs)
    union = ps + gs - inter
    iou = inter / union if union > 0 else float("nan")
    return dice, iou


def segmentation_metrics(pred: np.ndarray, gold: np.ndarray, label_ids=LABEL_IDS) -> dict:
    out = {}
    dices, ious = [], []
    for lid in label_ids:
        d, i = dice_iou(pred, gold, lid)
        out[f"dice_id{lid}"] = d
        out[f"iou_id{lid}"] = i
        if np.isfinite(d):
            dices.append(d)
            ious.append(i)
    out["dice_mean_present"] = float(np.mean(dices)) if dices else float("nan")
    out["iou_mean_present"] = float(np.mean(ious)) if ious else float("nan")
    return out


def distribution(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = int(v.size)
    if n == 0:
        return {"N_valid": 0, "mean": float("nan"), "sd": float("nan"),
                "median": float("nan"), "p90": float("nan"), "p95": float("nan")}
    return {
        "N_valid": n,
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if n > 1 else float("nan"),
        "median": float(np.median(v)),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
    }
