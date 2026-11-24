#!/usr/bin/env python
"""Step 4: unified evaluation CLI.

Usage:
  python scripts/04_evaluate.py --pred-dir DIR --uid-map local/phase0/uid_to_file_infer.json --tag infer_historical
  python scripts/04_evaluate.py --pred-dir DIR --uid-map ... --tag E0_fold0 --fold 0

Outputs local/phase0/eval_<tag>_cases.csv and eval_<tag>_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cardiac_axis.evaluate import evaluate_mask_dir, summarize  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "local", "phase0")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--uid-map", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fold", type=int, default=None, help="restrict to one validation fold")
    ap.add_argument("--cohort", default=None, help="restrict to a cohort")
    args = ap.parse_args()

    gt = pd.read_csv(os.path.join(OUT, "fixed_gt_v1.csv"), dtype={"uid": str, "case_id": str})
    fold = pd.read_csv(os.path.join(OUT, "fold_v2.csv"), dtype={"uid": str})
    gt = gt.merge(fold[["uid", "fold", "split_role"]], on="uid", how="left", validate="one_to_one")
    man = pd.read_csv(os.path.join(OUT, "manifest_local_v1.csv"), dtype={"uid": str})
    gt = gt.merge(man[["uid", "img_path", "label_gold_path", "label_infer_path"]],
                  on="uid", how="left", validate="one_to_one")
    if gt.label_gold_path.isna().any():
        raise RuntimeError("missing label_gold_path after merge")
    if args.fold is not None:
        gt = gt[gt.fold == args.fold]
    if args.cohort is not None:
        gt = gt[gt.cohort == args.cohort]
    with open(args.uid_map) as fh:
        uid_map = json.load(fh)

    eval_df = evaluate_mask_dir(gt, args.pred_dir, uid_map)
    eval_df.to_csv(os.path.join(OUT, f"eval_{args.tag}_cases.csv"), index=False)

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tag": args.tag, "pred_dir": args.pred_dir, "fold": args.fold,
        "data_reading_spec_version": "CARDIAC_AXIS_DATA_READING_SPEC_v1.0",
        "gt_source": "local/phase0/fixed_gt_v1.csv",
        "overall": summarize(eval_df),
        "by_cohort": {c: summarize(eval_df, c) for c in sorted(eval_df.cohort.unique())},
    }
    with open(os.path.join(OUT, f"eval_{args.tag}_summary.json"), "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
