#!/usr/bin/env python
"""E3 external evaluation (multicenter 442 + prospective 411 = 853).

Reuses E0 GT (fixed_gt_v1.csv), E0 folds (fold_v2.csv), E0 GT/geometry pipeline
(src/cardiac_axis/evaluate.py), and E0 external uid->file convention
(external basenames identical to local/phase0/uid_to_file_E0_external.json).
Only the prediction root differs (E3 external ensemble).

Writes only under local/e3_encoder_only_transfer/evaluation/.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PHASE0 = os.path.join(ROOT, "local/phase0")
OUT = os.path.join(ROOT, "local/e3_encoder_only_transfer/evaluation")
E3EXT = os.path.join(OUT, "external_pred_ensemble")
sys.path.insert(0, os.path.join(ROOT, "src"))
from cardiac_axis.evaluate import evaluate_mask_dir, summarize  # noqa: E402
LABEL_IDS = (1, 2, 3, 4, 5, 6)

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
    import cv2
    gt = pd.read_csv(os.path.join(PHASE0, "fixed_gt_v1.csv"), dtype={"uid": str, "case_id": str})
    fold = pd.read_csv(os.path.join(PHASE0, "fold_v2.csv"), dtype={"uid": str})
    gt = gt.merge(fold[["uid", "fold", "split_role"]], on="uid", how="left", validate="one_to_one")
    man = pd.read_csv(os.path.join(PHASE0, "manifest_local_v1.csv"), dtype={"uid": str})
    gt = gt.merge(man[["uid", "label_gold_path"]], on="uid", how="left", validate="one_to_one")
    gt = gt[gt.cohort.isin(["multicenter", "prospective"])].copy()
    e0map = json.load(open(os.path.join(PHASE0, "uid_to_file_E0_external.json")))
    uid_map = {uid: os.path.join(E3EXT, os.path.basename(p)) for uid, p in e0map.items()}
    assert set(gt.uid) == set(uid_map), (len(gt), len(uid_map))

    ev = evaluate_mask_dir(gt, "", uid_map, LABEL_IDS)
    ev.to_csv(os.path.join(OUT, "external_case_level_metrics.csv"), index=False)

    gold = dict(zip(gt.uid, gt.label_gold_path))
    hd_rows = []
    for _, g in gt.iterrows():
        uid = g.uid; pf = uid_map[uid]; rec = {"uid": uid, "cohort": g.cohort, "class": g["class"]}
        p = cv2.imread(pf, cv2.IMREAD_UNCHANGED); gg = cv2.imread(gold[uid], cv2.IMREAD_UNCHANGED)
        if p is None or gg is None or p.shape != gg.shape:
            rec["valid_shape"] = False; hd_rows.append(rec); continue
        rec["valid_shape"] = True
        for lid in LABEL_IDS:
            rec[f"hd95_{lid}"] = hd95(p == lid, gg == lid)
        vals = [rec[f"hd95_{lid}"] for lid in LABEL_IDS]
        rec["hd95_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        hd_rows.append(rec)
    hd = pd.DataFrame(hd_rows)
    hd.to_csv(os.path.join(OUT, "external_hd95_case_level.csv"), index=False)

    hdmap = dict(zip(hd.uid, hd.hd95_mean))
    ev["hd95_mean"] = ev.uid.map(hdmap)
    out = {"overall": summarize(ev),
           "by_cohort": {c: summarize(ev, c) for c in sorted(ev.cohort.unique())},
           "hd95_mean": float(hd.hd95_mean.mean()),
           "hd95_by_cohort": {c: float(ev[ev.cohort == c].hd95_mean.mean()) for c in sorted(ev.cohort.unique())},
           "hd95_by_label": {f"id{l}": float(hd[f'hd95_{l}'].mean()) for l in LABEL_IDS}}
    with open(os.path.join(OUT, "external_summary.json"), "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    for c in ("multicenter", "prospective"):
        s = out["by_cohort"][c]
        print(c, "N", s["N_cases"], "Dice %.4f"%s["dice_mean_present_mean"],
              "FullMAE %.3f MedAE %.3f P95 %.3f"%(s["full_error_mean"], s["full_error_median"], s["full_error_p95"]))
    s = out["overall"]
    print("external ALL N", s["N_cases"], "Dice %.4f"%s["dice_mean_present_mean"],
          "FullMAE %.3f P90 %.3f P95 %.3f"%(s["full_error_mean"], s["full_error_p90"], s["full_error_p95"]),
          "HD95 %.3f"%out["hd95_mean"])

if __name__ == "__main__":
    raise SystemExit(main())
