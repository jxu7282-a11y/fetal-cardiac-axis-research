# M1 — Measurement-aware Segmentation with Dual-axis Auxiliary Supervision

Version: `m1-dual-axis-v1.0`
Status: complete (5/5 folds × 300 epochs; formal evaluation and statistics done)

## What this is

M1 is an **independent method experiment**, run in parallel to E3. It is **not** a transfer
experiment: it trains from **scratch** on the early-trimester `Dataset501` (no mid-trimester
data, no pretrained weights). It asks a different question:

> *If we attach two lightweight axes-regression heads to the segmentation decoder and train
> them jointly with the segmentation loss, does that **measurement-aware supervision** improve
> the downstream cardiac-axis angle (Full Angle) — even though inference uses only the
> segmentation output?*

The proposed mechanism: the network is nudged during training toward representations that
encode **Card** and **Reference** axis direction, with the hope that better internal geometry
propagates into a lower Full Angle error after the frozen post-processing/geometry pipeline.

**Answer recorded here: no demonstrated clinical gain, and the internal cohort is
significantly *worse* than E0.** The auxiliary heads collapse to a near-constant predictor
because the chosen pooling (global average) discards spatial direction information. Full Angle
MAE/P95, not Dice, is the primary criterion.

## Contents

| File | Content |
|---|---|
| `README.md` | this file |
| `report.md` | method, network/loss, training config, results, statistics, GAP-collapse diagnosis, conclusion, limitations |
| `config.json` | dataset, network architecture, loss, training, evaluation protocol, headline results, diagnosis |
| `metrics/formal_comparison_table.csv` | M1/E0/E2/E3 × scope × subgroup × metric (MAE/MedAE/P90/P95/≥10%/Card/Ref/Dice/IoU) |
| `metrics/paired_bootstrap.csv` | paired bootstrap Δ vs E0/E2/E3, 10000 resamples, 95% CI |
| `metrics/stratified_metrics.csv` | Ne/Po and multicenter/prospective strata (mean/median/P90/P95/CI) |
| `metrics/improvement_worsening_counts.csv` | per-case improved/worsened ≥1° bands |
| `metrics/error_threshold_counts.csv` | ≥10°, ≥15° and measurement-failure counts |
| `metrics/internal_summary.json` | internal OOF aggregate (621; 620 valid Full) with Ne/Po |
| `metrics/external_summary.json` | external aggregate (853) plus per-cohort |
| `metrics/failure_cases_anonymized.csv` | the single measurement failure, `case_hash` only |
| `metrics/cohort_match_audit.json` | M1/E0/E2/E3 uid-set, duplicate, GT and NaN audit (aggregate only) |
| `*.py`, `tests/` | trainer, aux heads, loss, augmentation, dataloader and their CPU unit tests |

## Headline numbers

| | Full MAE | Full P95 | Card MAE | Ref MAE | Dice |
|---|---:|---:|---:|---:|---:|
| Internal M1 (n=620) | **3.815** | **10.084** | 3.135 | 2.530 | 0.933 |
| Internal E0 | 3.639 | 9.923 | 3.088 | 2.575 | 0.935 |
| Internal E3 | 3.647 | 9.173 | 3.188 | 2.404 | 0.934 |
| External M1 (n=853) | 3.692 | 9.369 | 2.830 | 2.747 | 0.929 |
| External E0 | 3.739 | 9.351 | 2.906 | 2.701 | 0.929 |
| External E3 | 3.693 | 8.848 | 2.768 | 2.922 | 0.928 |

Paired bootstrap (Δ = M1 − baseline, negative = M1 better):

- Internal vs E0: **+0.181° [0.023, 0.359] — significantly WORSE.**
- Internal vs E3: **+0.168° [0.009, 0.337] — significantly WORSE.**
- External vs E0: −0.048° [−0.151, 0.057] — null. External vs E3: −0.001° — null.

**Bottom line:** internal harm, external null. The method does not improve clinical
measurement and should not be read as a success on any cohort.
