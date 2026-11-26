#!/usr/bin/env python
"""E3 internal five-fold OOF evaluation.

Byte-identical evaluation protocol to E0/E2:
  * frozen GT   : local/phase0/fixed_gt_v1.csv
  * frozen folds: local/phase0/fold_v2.csv
  * geometry    : src/cardiac_axis/evaluate.py (unchanged)
  * uid mapping : local/phase0/nnunet_case_map.csv
  * HD95        : scipy surface-distance 95th percentile, label loop 1..6
Only the prediction root differs (E3 folds).

Writes only under local/e3_encoder_only_transfer/evaluation/.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PHASE0 = os.path.join(ROOT, "local/phase0")
E3RES = os.path.join(ROOT, "local/e3_encoder_only_transfer/results_nnunet",
                     "Dataset501_EarlyRetroCardiac",
                     "nnUNetTrainer_E3EncoderOnly__nnUNetPlans__2d")
OUT = os.path.join(ROOT, "local/e3_encoder_only_transfer/evaluation")
sys.path.insert(0, os.path.join(ROOT, "src"))
from cardiac_axis.evaluate import evaluate_mask_dir, summarize  # noqa: E402

LABEL_IDS = (1, 2, 3, 4, 5, 6)


def build_uid_map():
    cmap = pd.read_csv(os.path.join(PHASE0, "nnunet_case_map.csv"), dtype={"uid": str})
    key2uid = dict(zip(cmap.nnunet_key, cmap.uid))
    fold = pd.read_csv(os.path.join(PHASE0, "fold_v2.csv"), dtype={"uid": str})
    uid_map, per_fold = {}, {}
    for k in range(5):
        vdir = os.path.join(E3RES, f"fold_{k}", "validation")
        files = sorted(glob.glob(os.path.join(vdir, "*.png")))
        m = {}
        for f in files:
            key = os.path.splitext(os.path.basename(f))[0]
            uid = key2uid.get(key)
            if uid is None:
                raise RuntimeError(f"unmapped key {key} fold {k}")
            m[uid] = os.path.abspath(f)
        expect = set(fold.loc[(fold.fold == k) & (fold.cohort == "retrospective"), "uid"])
        got = set(m)
        if expect != got:
            raise RuntimeError(f"fold {k} mismatch missing={len(expect-got)} extra={len(got-expect)}")
        uid_map.update(m)
        per_fold[k] = len(m)
    if len(uid_map) != 621:
        raise RuntimeError(f"OOF union {len(uid_map)} != 621")
    return uid_map, per_fold


def hd95(pred_b, gold_b):
    from scipy import ndimage
    S = np.ones((3, 3), bool)
    def surf(b):
        return b & ~ndimage.binary_erosion(b, structure=S, border_value=0) if b.any() else np.zeros_like(b, bool)
    sp, sg = surf(pred_b), surf(gold_b)
    if not sp.any() or not sg.any():
        return float("nan")
    return float(np.percentile(np.concatenate([ndimage.distance_transform_edt(~sg)[sp],
                                               ndimage.distance_transform_edt(~sp)[sg]]), 95))


def main():
    os.makedirs(OUT, exist_ok=True)
    import cv2
    gt = pd.read_csv(os.path.join(PHASE0, "fixed_gt_v1.csv"), dtype={"uid": str, "case_id": str})
    fold = pd.read_csv(os.path.join(PHASE0, "fold_v2.csv"), dtype={"uid": str})
    gt = gt.merge(fold[["uid", "fold", "split_role"]], on="uid", how="left", validate="one_to_one")
    man = pd.read_csv(os.path.join(PHASE0, "manifest_local_v1.csv"), dtype={"uid": str})
    gt = gt.merge(man[["uid", "label_gold_path"]], on="uid", how="left", validate="one_to_one")
    gt = gt[gt.cohort == "retrospective"].copy()
    uid_map, per_fold = build_uid_map()

    ev = evaluate_mask_dir(gt, "", uid_map, LABEL_IDS)
    ev.to_csv(os.path.join(OUT, "internal_case_level_metrics.csv"), index=False)

    summ = summarize(ev)

    man_df = pd.read_csv(os.path.join(PHASE0, "manifest_local_v1.csv"), dtype={"uid": str})
    man_gold = dict(zip(man_df.uid, man_df.label_gold_path))
    hd_rows = []
    for _, g in gt.iterrows():
        uid = g.uid
        pf = uid_map.get(uid)
        rec = {"uid": uid}
        if not pf or not os.path.isfile(pf):
            rec["valid_shape"] = False; hd_rows.append(rec); continue
        p = cv2.imread(pf, cv2.IMREAD_UNCHANGED)
        gg = cv2.imread(man_gold[uid], cv2.IMREAD_UNCHANGED)
        if p is None or gg is None or p.shape != gg.shape:
            rec["valid_shape"] = False; hd_rows.append(rec); continue
        rec["valid_shape"] = True
        for lid in LABEL_IDS:
            rec[f"hd95_{lid}"] = hd95(p == lid, gg == lid)
        vals = [rec[f"hd95_{lid}"] for lid in LABEL_IDS]
        rec["hd95_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        hd_rows.append(rec)
    hd = pd.DataFrame(hd_rows)
    hd.to_csv(os.path.join(OUT, "internal_hd95_case_level.csv"), index=False)

    out = {"per_fold_pred_count": per_fold, "overall": summ,
           "hd95_mean": float(hd.hd95_mean.mean()) if hd.hd95_mean.notna().any() else float("nan"),
           "hd95_by_label": {f"id{l}": float(hd[f'hd95_{l}'].mean()) for l in LABEL_IDS}}
    with open(os.path.join(OUT, "internal_summary.json"), "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
