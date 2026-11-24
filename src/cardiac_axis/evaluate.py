"""Unified evaluation: same GT, same fixed geometry, same metrics for every model.

Data reading spec: CARDIAC_AXIS_DATA_READING_SPEC_v1.0
Method stage: fixed_geometry (evaluation only; never reselects manual GT)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .angles import axial_abs_error, acute_angle_between_lines
from .masks import axes_from_mask, load_label_mask
from .metrics import distribution, segmentation_metrics


def evaluate_mask_dir(gt_df: pd.DataFrame, pred_dir: str, uid_to_filename: dict,
                      label_ids=(1, 2, 3, 4, 5, 6)) -> pd.DataFrame:
    """Evaluate a directory of predicted masks against the frozen fixed GT.

    gt_df must contain: uid, cohort, class, manual_card_axis, manual_ref_axis,
    manual_full_angle, label_gold_path.
    """
    rows = []
    for _, g in gt_df.iterrows():
        uid = g.uid
        rec = {"uid": uid, "cohort": g.cohort, "class": g["class"],
               "split_role": g.get("split_role", "")}
        target = uid_to_filename.get(uid)
        if target is None:
            path = None
        elif os.path.isabs(target):
            path = target
        else:
            path = os.path.join(pred_dir, target)
        if not path or not os.path.isfile(path):
            rec["valid_pred"] = False
            rec["invalid_reason"] = "prediction_missing"
            rows.append(rec)
            continue
        try:
            pred = load_label_mask(path)
            gold = load_label_mask(g.label_gold_path)
        except Exception as exc:
            rec["valid_pred"] = False
            rec["invalid_reason"] = f"read_error:{exc}"
            rows.append(rec)
            continue
        rec.update(segmentation_metrics(pred, gold, label_ids))
        axes = axes_from_mask(pred)
        pred_card, pred_ref = axes["card_axis"], axes["ref_axis"]
        pred_full = acute_angle_between_lines(pred_card, pred_ref)
        rec.update({
            "valid_pred": True,
            "pred_card_axis": pred_card, "pred_ref_axis": pred_ref,
            "pred_full_angle": pred_full,
            "pred_card_reason": axes["card_reason"], "pred_ref_reason": axes["ref_reason"],
            "manual_card_axis": g.manual_card_axis, "manual_ref_axis": g.manual_ref_axis,
            "manual_full_angle": g.manual_full_angle,
            "card_error": axial_abs_error(pred_card, g.manual_card_axis),
            "ref_error": axial_abs_error(pred_ref, g.manual_ref_axis),
            "full_error": abs(pred_full - g.manual_full_angle)
            if np.isfinite(pred_full) and np.isfinite(g.manual_full_angle) else float("nan"),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(eval_df: pd.DataFrame, cohort: str | None = None) -> dict:
    d = eval_df if cohort is None else eval_df[eval_df.cohort == cohort]
    out = {"cohort": cohort or "ALL", "N_cases": int(len(d)),
           "N_valid_pred": int(d.get("valid_pred", pd.Series(dtype=bool)).sum())}
    for label in ("card_error", "ref_error", "full_error"):
        dist = distribution(d[label] if label in d else [])
        out[f"{label}_N"] = dist["N_valid"]
        out[f"{label}_mean"] = dist["mean"]
        out[f"{label}_median"] = dist["median"]
        out[f"{label}_p90"] = dist["p90"]
        out[f"{label}_p95"] = dist["p95"]
    for key in ("dice_mean_present", "iou_mean_present"):
        v = pd.to_numeric(d.get(key, pd.Series(dtype=float)), errors="coerce")
        out[f"{key}_N"] = int(v.notna().sum())
        out[f"{key}_mean"] = float(v.mean()) if v.notna().any() else float("nan")
    return out
