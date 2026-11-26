#!/usr/bin/env python
"""E3 statistical analysis vs E0 and E2.

Inputs are the frozen case-level evaluation CSVs only (no retraining, no
re-prediction, no post-processing change). Bootstrap is paired by case uid.

Outputs (all under local/e3_encoder_only_transfer/analysis/):
  cohort_match_audit.json
  formal_comparison_table.csv
  paired_bootstrap.csv
  stratified_metrics.csv
  error_threshold_counts.csv
  failure_cases.csv
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
OUT = os.path.join(ROOT, "local/e3_encoder_only_transfer/analysis")
os.makedirs(OUT, exist_ok=True)

SRC = {
    "E0_int": "local/phase0/eval_E0_5fold_pooled_cases_withfold.csv",
    "E2_int": "local/e2_mid_to_early_finetune/evaluation/internal_case_level_metrics.csv",
    "E3_int": "local/e3_encoder_only_transfer/evaluation/internal_case_level_metrics.csv",
    "E0_ext": "local/phase0/eval_E0_external_cases_extonly.csv",
    "E2_ext": "local/e2_mid_to_early_finetune/evaluation/external_case_level_metrics.csv",
    "E3_ext": "local/e3_encoder_only_transfer/evaluation/external_case_level_metrics.csv",
}
METRICS = ["full_error", "card_error", "ref_error"]
N_BOOT = 10000
SEED = 42


def load_all():
    return {k: pd.read_csv(os.path.join(ROOT, v), dtype={"uid": str}) for k, v in SRC.items()}


def inner_and_ext(scope):
    if scope == "internal":
        return {"E0": D["E0_int"], "E2": D["E2_int"], "E3": D["E3_int"]}
    return {"E0": D["E0_ext"], "E2": D["E2_ext"], "E3": D["E3_ext"]}


def audit(dfs_by_scope):
    out = {}
    for scope, sets in dfs_by_scope.items():
        e0, e2, e3 = sets["E0"], sets["E2"], sets["E3"]
        rec = {
            "n_E0": len(e0), "n_E2": len(e2), "n_E3": len(e3),
            "uid_sets_identical": set(e0.uid) == set(e2.uid) == set(e3.uid),
            "duplicate_uid": {k: int(v.uid.duplicated().sum()) for k, v in sets.items()},
            "gt_identical": {},
            "invalid_pred": {k: int((~v.valid_pred).sum()) if "valid_pred" in v else None
                             for k, v in sets.items()},
            "nan_error_counts": {k: {m: int(v[m].isna().sum()) for m in METRICS} for k, v in sets.items()},
            "cohort_counts": {k: {c: int(n) for c, n in v.cohort.value_counts().items()} for k, v in sets.items()},
            "class_counts": {k: {c: int(n) for c, n in v["class"].value_counts().items()} for k, v in sets.items()},
        }
        for col in ("manual_full_angle", "manual_card_axis", "manual_ref_axis"):
            m = e0[["uid", col]].merge(e2[["uid", col]], on="uid", suffixes=("_0", "_2")) \
                               .merge(e3[["uid", col]], on="uid")
            same_02 = np.allclose(m[col + "_0"], m[col + "_2"], equal_nan=True)
            same_03 = np.allclose(m[col + "_0"], m[col], equal_nan=True)
            rec["gt_identical"][col] = {"E0==E2": bool(same_02), "E0==E3": bool(same_03)}
        out[scope] = rec
    return out


def paired_values(scope, target, other, mask):
    """Aligned metric vectors for the subset selected by `mask`, plus uid order."""
    a = inner_and_ext(scope)[other]
    b = inner_and_ext(scope)[target]
    a = a[mask(a)].copy()
    b = b[mask(b)].copy()
    m = a[["uid", "cohort", "class"] + METRICS].merge(
        b[["uid"] + METRICS], on="uid", how="inner", suffixes=("_o", "_t"))
    m = m.dropna(subset=["full_error_o", "full_error_t"])
    return m


def boot_ci(x, n=N_BOOT, seed=SEED):
    """Percentile bootstrap CI of the mean of x (x is already case-level)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n, x.size))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def p90(x):
    return float(np.nanpercentile(x, 90)) if np.isfinite(x).any() else np.nan


def p95(x):
    return float(np.nanpercentile(x, 95)) if np.isfinite(x).any() else np.nan


def main():
    global D
    D = load_all()
    scopes = {"internal": None, "external": None}
    sets = {s: inner_and_ext(s) for s in ("internal", "external")}

    audit_out = audit(sets)
    with open(os.path.join(OUT, "cohort_match_audit.json"), "w") as fh:
        json.dump(audit_out, fh, indent=2, ensure_ascii=False)

    # ---------------- formal comparison table ----------------
    rows = []
    for scope in ("internal", "external"):
        subsets = {"ALL": None}
        if scope == "internal":
            subsets.update({"Ne": "Ne", "Po": "Po"})
        else:
            for c in ("multicenter", "prospective"):
                subsets[c] = c
        for sub_name, sub in subsets.items():
            for model, df in sets[scope].items():
                if sub_name in ("Ne", "Po"):
                    d = df[df["class"] == sub_name]
                elif sub_name == "ALL":
                    d = df
                else:
                    d = df[df.cohort == sub_name]
                d = d[d.valid_pred]
                fe = d.full_error.dropna()
                rows.append({
                    "scope": scope, "subset": sub_name, "model": model,
                    "n_cases": int(len(d)), "n_full_error": int(len(fe)),
                    "full_MAE": float(fe.mean()), "full_MedAE": float(fe.median()),
                    "full_P90": p90(fe), "full_P95": p95(fe),
                    "card_MAE": float(d.card_error.dropna().mean()),
                    "ref_MAE": float(d.ref_error.dropna().mean()),
                    "dice_mean": float(d.dice_mean_present.mean()),
                    "iou_mean": float(d.iou_mean_present.mean()),
                })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "formal_comparison_table.csv"), index=False)

    # ---------------- paired bootstrap ----------------
    boot_rows = []
    for scope in ("internal", "external"):
        pairs = [("E3", "E0"), ("E3", "E2")]
        subsets = {"ALL": lambda d: pd.Series(True, index=d.index)}
        if scope == "internal":
            subsets["Ne"] = lambda d: d["class"] == "Ne"
            subsets["Po"] = lambda d: d["class"] == "Po"
        else:
            subsets["multicenter"] = lambda d: d.cohort == "multicenter"
            subsets["prospective"] = lambda d: d.cohort == "prospective"
        for target, other in pairs:
            for sub_name, mask in subsets.items():
                m = paired_values(scope, target, other, mask)
                for metric in METRICS:
                    o = m[f"{metric}_o"].to_numpy(float)
                    t = m[f"{metric}_t"].to_numpy(float)
                    diff = t - o
                    lo, hi = boot_ci(diff)
                    boot_rows.append({
                        "scope": scope, "subset": sub_name,
                        "comparison": f"{target}_vs_{other}", "metric": metric,
                        "n_paired": int(len(m)),
                        "mean_other": float(o.mean()), "mean_target": float(t.mean()),
                        "delta_mean": float(diff.mean()),
                        "delta_ci_lo": lo, "delta_ci_hi": hi,
                        "ci_excludes_zero": bool(lo > 0 or hi < 0),
                        "n_improved": int((diff < 0).sum()), "n_worsened": int((diff > 0).sum()),
                        "n_unchanged": int((diff == 0).sum()),
                    })
    pd.DataFrame(boot_rows).to_csv(os.path.join(OUT, "paired_bootstrap.csv"), index=False)

    # ---------------- stratified metrics (mean/median/P90/P95 + CI) ----------------
    strat_rows = []
    for scope in ("internal", "external"):
        strata = (["Ne", "Po"] if scope == "internal"
                  else ["multicenter", "prospective"])
        for model, df in sets[scope].items():
            for s in strata:
                d = df[(df["class"] == s) if scope == "internal" else (df.cohort == s)]
                d = d[d.valid_pred]
                for metric in ("full_error", "card_error", "ref_error"):
                    x = d[metric].dropna().to_numpy(float)
                    lo, hi = boot_ci(x)
                    strat_rows.append({
                        "scope": scope, "stratum": s, "model": model, "metric": metric,
                        "n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
                        "p90": p90(x), "p95": p95(x),
                        "mean_ci_lo": lo, "mean_ci_hi": hi,
                    })
    pd.DataFrame(strat_rows).to_csv(os.path.join(OUT, "stratified_metrics.csv"), index=False)

    # ---------------- threshold counts ----------------
    thr_rows = []
    for scope in ("internal", "external"):
        strata = (["ALL", "Ne", "Po"] if scope == "internal"
                  else ["ALL", "multicenter", "prospective"])
        for model, df in sets[scope].items():
            for s in strata:
                if s == "ALL":
                    d = df
                elif s in ("Ne", "Po"):
                    d = df[df["class"] == s]
                else:
                    d = df[df.cohort == s]
                d = d[d.valid_pred]
                fe = d.full_error
                n_total = int(len(d))
                thr_rows.append({
                    "scope": scope, "stratum": s, "model": model, "n_cases": n_total,
                    "n_full_error_available": int(fe.notna().sum()),
                    "n_error_ge_10deg": int((fe >= 10).sum()),
                    "pct_error_ge_10deg": float((fe >= 10).mean() * 100),
                    "n_error_ge_15deg": int((fe >= 15).sum()),
                    "n_measurement_failure": int(fe.isna().sum()),
                })
    pd.DataFrame(thr_rows).to_csv(os.path.join(OUT, "error_threshold_counts.csv"), index=False)

    # ---------------- per-case improvement/worsening (E3 vs E0/E2) ----------------
    imp_rows = []
    for scope in ("internal", "external"):
        for other in ("E0", "E2"):
            m = paired_values(scope, "E3", other, lambda d: pd.Series(True, index=d.index))
            m["delta_full"] = m["full_error_t"] - m["full_error_o"]
            for s in (["ALL", "Ne", "Po"] if scope == "internal"
                      else ["ALL", "multicenter", "prospective"]):
                if s == "ALL":
                    mm = m
                elif s in ("Ne", "Po"):
                    mm = m[m["class"] == s]
                else:
                    mm = m[m.cohort == s]
                d = mm.delta_full
                imp_rows.append({
                    "scope": scope, "stratum": s, "comparison": f"E3_vs_{other}",
                    "n_paired": int(len(mm)),
                    "n_improved_ge_1deg": int((d <= -1).sum()),
                    "n_worsened_ge_1deg": int((d >= 1).sum()),
                    "n_within_1deg": int((d.abs() < 1).sum()),
                    "mean_delta": float(d.mean()), "median_delta": float(d.median()),
                })
    pd.DataFrame(imp_rows).to_csv(os.path.join(OUT, "improvement_worsening_counts.csv"), index=False)

    # ---------------- failure cases ----------------
    fail_rows = []
    for scope in ("internal", "external"):
        for model, df in sets[scope].items():
            d = df.copy()
            bad = d[(~d.valid_pred) | d.full_error.isna() | d.card_error.isna() | d.ref_error.isna()]
            for _, r in bad.iterrows():
                reasons = []
                if not bool(r.valid_pred):
                    reasons.append("invalid_prediction")
                if pd.isna(r.card_error):
                    reasons.append("card_axis_unavailable")
                if pd.isna(r.ref_error):
                    reasons.append("ref_axis_unavailable")
                if pd.isna(r.full_error) and pd.notna(r.card_error):
                    reasons.append("full_angle_unavailable")
                fail_rows.append({
                    "scope": scope, "model": model, "uid": r.uid, "cohort": r.cohort,
                    "class": r["class"], "valid_pred": bool(r.valid_pred),
                    "pred_card_reason": r.get("pred_card_reason", ""),
                    "pred_ref_reason": r.get("pred_ref_reason", ""),
                    "card_error": r.card_error, "ref_error": r.ref_error,
                    "full_error": r.full_error, "failure_reasons": ";".join(reasons),
                })
    pd.DataFrame(fail_rows).to_csv(os.path.join(OUT, "failure_cases.csv"), index=False)

    # ---------------- console summary ----------------
    print("AUDIT")
    for s, rec in audit_out.items():
        print(f"  {s}: ids_identical={rec['uid_sets_identical']} "
              f"nan_full_error={ {k: v['full_error'] for k, v in rec['nan_error_counts'].items()} }")
    print("\nFORMAL (ALL)")
    print(table[table.subset == "ALL"].round(4).to_string(index=False))
    print("\nBOOTSTRAP full_error deltas")
    bt = pd.DataFrame(boot_rows)
    print(bt[(bt.metric == "full_error")].round(4).to_string(index=False))
    print(f"\nwrote 7 files to {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
