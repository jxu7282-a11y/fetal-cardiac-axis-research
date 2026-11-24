#!/usr/bin/env python
"""Step 12: HD95 per label for E0 external predictions (853 cases, CPU only)."""
from __future__ import annotations
import os, sys
import cv2, numpy as np, pandas as pd
from scipy import ndimage

OUT = os.path.join(os.path.dirname(__file__), "..", "local", "phase0")
IDS = (1, 2, 3, 4, 5, 6)
STRUCT = np.ones((3, 3), bool)


def surface(b):
    return b & ~ndimage.binary_erosion(b, structure=STRUCT, border_value=0) if b.any() else np.zeros_like(b, bool)


def hd95(p, g):
    sp, sg = surface(p), surface(g)
    if not sp.any() or not sg.any():
        return float("nan")
    return float(np.percentile(np.concatenate([ndimage.distance_transform_edt(~sg)[sp],
                                               ndimage.distance_transform_edt(~sp)[sg]]), 95))


man = pd.read_csv(os.path.join(OUT, "manifest_local_v1.csv"), dtype={"uid": str})
gold = dict(zip(man.uid, man.label_gold_path))
pmap = __import__("json").load(open(os.path.join(OUT, "uid_to_file_E0_external.json")))
fold = pd.read_csv(os.path.join(OUT, "fold_v2.csv"), dtype={"uid": str})
meta = fold.set_index("uid")[["cohort", "class"]]

rows = []
for uid, pf in pmap.items():
    p = cv2.imread(pf, cv2.IMREAD_UNCHANGED)
    g = cv2.imread(gold[uid], cv2.IMREAD_UNCHANGED)
    rec = {"uid": uid, "cohort": meta.loc[uid, "cohort"], "class": meta.loc[uid, "class"]}
    if p is None or g is None or p.shape != g.shape:
        rec["valid_shape"] = False
        rows.append(rec); continue
    rec["valid_shape"] = True
    for lid in IDS:
        rec[f"hd95_{lid}"] = hd95(p == lid, g == lid)
    vals = [rec[f"hd95_{lid}"] for lid in IDS]
    rec["hd95_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "eval_E0_external_hd95_cases.csv"), index=False)
print("cases:", len(df), "bad:", int((~df.valid_shape).sum()))
for lid in IDS:
    v = df[f"hd95_{lid}"].dropna()
    print(f"id{lid}: N={len(v)} HD95 mean={v.mean():.2f} median={v.median():.2f}")
v = df.hd95_mean.dropna()
print(f"mean-of-labels: N={len(v)} mean={v.mean():.2f}")
for coh in ("multicenter", "prospective"):
    d = df[df.cohort == coh].hd95_mean.dropna()
    print(f"  {coh}: N={len(d)} mean={d.mean():.2f}")
