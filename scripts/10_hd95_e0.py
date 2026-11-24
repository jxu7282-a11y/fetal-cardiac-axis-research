#!/usr/bin/env python
"""Step 10: HD95 per label for E0 5-fold validation predictions (CPU only).

Surface-distance HD95: for each label, compare prediction vs label_gold in the
same image grid (same shape, no resampling). Cases where either side is empty
are marked invalid for that label and counted, never silently dropped.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

OUT = os.path.join(os.path.dirname(__file__), "..", "local", "phase0")
RES = os.path.join(os.path.dirname(__file__), "..", "local", "nnunet_results",
                   "Dataset501_EarlyRetroCardiac",
                   "nnUNetTrainer_300Epochs__nnUNetPlans__2d")
IDS = (1, 2, 3, 4, 5, 6)
STRUCT = np.ones((3, 3), bool)


def surface(binary: np.ndarray) -> np.ndarray:
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    return binary & ~ndimage.binary_erosion(binary, structure=STRUCT, border_value=0)


def hd95(pred_b: np.ndarray, gold_b: np.ndarray) -> float:
    sp, sg = surface(pred_b), surface(gold_b)
    if not sp.any() or not sg.any():
        return float("nan")
    dt_g = ndimage.distance_transform_edt(~sg)
    dt_p = ndimage.distance_transform_edt(~sp)
    d = np.concatenate([dt_g[sp], dt_p[sg]])
    return float(np.percentile(d, 95))


def main() -> int:
    man = pd.read_csv(os.path.join(OUT, "manifest_local_v1.csv"), dtype={"uid": str})
    gold = dict(zip(man.uid, man.label_gold_path))
    cmap = pd.read_csv(os.path.join(OUT, "nnunet_case_map.csv"), dtype={"uid": str})
    key2uid = dict(zip(cmap.nnunet_key, cmap.uid))

    rows = []
    for k in range(5):
        vdir = os.path.join(RES, f"fold_{k}", "validation")
        for fn in sorted(os.listdir(vdir)):
            if not fn.endswith(".png"):
                continue
            key = fn[:-4]
            uid = key2uid.get(key)
            if uid is None:
                continue
            p = cv2.imread(os.path.join(vdir, fn), cv2.IMREAD_UNCHANGED)
            g = cv2.imread(gold[uid], cv2.IMREAD_UNCHANGED)
            if p is None or g is None:
                rows.append({"fold": k, "uid": uid, "valid_shape": False})
                continue
            if p.shape != g.shape:
                rows.append({"fold": k, "uid": uid, "valid_shape": False,
                             "pred_shape": str(p.shape), "gold_shape": str(g.shape)})
                continue
            rec = {"fold": k, "uid": uid, "valid_shape": True}
            for lid in IDS:
                rec[f"hd95_{lid}"] = hd95(p == lid, g == lid)
            vals = [rec[f"hd95_{lid}"] for lid in IDS]
            rec["hd95_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
            rows.append(rec)
        print(f"fold{k} done ({len(rows)} cases so far)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "eval_E0_5fold_hd95_cases.csv"), index=False)
    print("\ncases:", len(df), "bad shape/read:", int((~df.valid_shape).sum()))
    summ = []
    for lid in IDS:
        v = df[f"hd95_{lid}"].dropna()
        summ.append({"label": lid, "N_valid": len(v), "N_invalid_empty": int(len(df) - len(v)),
                     "HD95_mean": v.mean(), "HD95_median": v.median(),
                     "HD95_p90": v.quantile(0.9), "HD95_p95": v.quantile(0.95)})
    sm = pd.DataFrame(summ)
    print(sm.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    sm.to_csv(os.path.join(OUT, "eval_E0_5fold_hd95_summary.csv"), index=False)
    v = df["hd95_mean"].dropna()
    print(f"\noverall mean-of-labels HD95: N={len(v)} mean={v.mean():.3f} median={v.median():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
