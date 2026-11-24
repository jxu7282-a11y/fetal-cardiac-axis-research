#!/usr/bin/env python
"""Step 9: build uid -> nnUNet validation prediction file maps for E0 folds.

For each fold, reads <results>/fold_k/validation/*.png (nnUNet named them by
nnunet_key) and maps them back to the formal uid via nnunet_case_map.csv.
Only validation cases of that fold are included; nothing is invented.

Output: local/phase0/uid_to_file_E0_fold{k}.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "local", "phase0")
RES = os.path.join(os.path.dirname(__file__), "..", "local", "nnunet_results",
                   "Dataset501_EarlyRetroCardiac",
                   "nnUNetTrainer_300Epochs__nnUNetPlans__2d")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    cmap = pd.read_csv(os.path.join(OUT, "nnunet_case_map.csv"), dtype={"uid": str})
    key2uid = dict(zip(cmap.nnunet_key, cmap.uid))
    fold = pd.read_csv(os.path.join(OUT, "fold_v2.csv"), dtype={"uid": str})
    report = {}
    for k in args.folds:
        vdir = os.path.join(RES, f"fold_{k}", "validation")
        files = sorted(glob.glob(os.path.join(vdir, "*.png")))
        uid_map = {}
        for f in files:
            key = os.path.splitext(os.path.basename(f))[0]
            uid = key2uid.get(key)
            if uid is None:
                raise RuntimeError(f"unmapped nnunet key {key} in fold {k}")
            uid_map[uid] = os.path.abspath(f)
        expect = set(fold.loc[(fold.fold == k) & (fold.cohort == "retrospective"), "uid"])
        got = set(uid_map)
        out = os.path.join(OUT, f"uid_to_file_E0_fold{k}.json")
        with open(out, "w") as fh:
            json.dump(uid_map, fh, indent=1, sort_keys=True)
        report[k] = {"n_files": len(files), "n_mapped": len(uid_map),
                     "n_expected_val": len(expect),
                     "missing": sorted(expect - got)[:10], "extra": sorted(got - expect)[:10],
                     "map": out}
        print(f"fold{k}: files={len(files)} mapped={len(uid_map)} expected={len(expect)} "
              f"missing={len(expect-got)} extra={len(got-expect)} -> {out}")
    bad = [k for k, v in report.items() if v["missing"] or v["extra"]]
    if bad:
        print("WARNING folds with mismatch:", bad)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
