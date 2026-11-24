"""Fixed manual GT: gold-assisted Card/Reference semantic assignment (spec 13-14)."""
from __future__ import annotations

from .angles import acute_angle_between_lines, axial_abs_error


def assign_manual_semantics(manual_line_angles, gold_card_axis, gold_ref_axis) -> dict:
    a0, a1 = manual_line_angles
    out = {
        "manual_card_axis": float("nan"),
        "manual_ref_axis": float("nan"),
        "manual_assignment": "unknown_gold_axis_failed",
        "manual_assignment_cost01": float("nan"),
        "manual_assignment_cost10": float("nan"),
        "manual_assignment_margin": float("nan"),
    }
    if not _finite(gold_card_axis) or not _finite(gold_ref_axis):
        return out
    if not (_finite(a0) and _finite(a1)):
        return out
    cost01 = axial_abs_error(a0, gold_card_axis) + axial_abs_error(a1, gold_ref_axis)
    cost10 = axial_abs_error(a1, gold_card_axis) + axial_abs_error(a0, gold_ref_axis)
    out["manual_assignment_cost01"] = cost01
    out["manual_assignment_cost10"] = cost10
    out["manual_assignment_margin"] = abs(cost10 - cost01)
    if cost01 <= cost10:
        out["manual_card_axis"], out["manual_ref_axis"] = a0, a1
        out["manual_assignment"] = "line0_card_line1_ref"
    else:
        out["manual_card_axis"], out["manual_ref_axis"] = a1, a0
        out["manual_assignment"] = "line1_card_line0_ref"
    return out


def _finite(x) -> bool:
    try:
        x = float(x)
    except Exception:
        return False
    return x == x and abs(x) != float("inf")


def predicted_full_angle(pred_card_axis, pred_ref_axis) -> float:
    return acute_angle_between_lines(pred_card_axis, pred_ref_axis)
