# E3 — Mid-Trimester **Encoder-Only** Transfer → Early-Trimester Fine-Tuning

Version: `e3-encoder-only-v1.0`
Status: complete (5/5 folds, 300 epochs each; formal evaluation and statistics done)

## What this is

E2 transferred the *entire* mid-trimester network (all 382 non-segmentation keys) and
recorded a null result. E3 isolates one variable: **only the encoder** is initialised from
mid-trimester pretraining; the decoder (including its `ConvTranspose2d` layers) and all
segmentation heads are freshly randomly initialised, then the whole network is fine-tuned on
early-trimester `Dataset501`.

The question: *is it the encoder — and only the encoder — that carries transferable
information across trimesters?*

Answer recorded here: **no overall clinical advantage over the E0 scratch baseline.** E3 does
beat E2 on the external set, but the gain is concentrated in subgroups, and the Reference axis
moves in opposite directions inside versus outside. Full Angle MAE/P95, not Dice, is the
primary criterion.

## Contents

| File | Content |
|---|---|
| `README.md` | this file |
| `report.md` | method, training config, evaluation protocol, results, statistics, conclusion, limitations |
| `config.json` | dataset, encoder-only transfer contract, training, evaluation protocol, headline results |
| `checkpoints_sha256/fold_{0..4}.sha256` | SHA256 of each fold's best/final checkpoints (weights stay on the server) |
| `metrics/internal_summary.json` | internal OOF aggregate (621) |
| `metrics/external_summary.json` | external aggregate (853) plus per-cohort |
| `metrics/internal_case_level_anonymized.csv` | per-case internal metrics, `case_hash` only |
| `metrics/external_case_level_anonymized.csv` | per-case external metrics, `case_hash` only |
| `analysis/cohort_match_audit.json` | E0/E2/E3 uid-set, GT and NaN audit |
| `analysis/formal_comparison_table.csv` | E0/E2/E3 × scope × subgroup × metric |
| `analysis/paired_bootstrap.csv` | paired bootstrap deltas, 10000 resamples, 95% CI |
| `analysis/stratified_metrics.csv` | Ne/Po and multicenter/prospective strata |
| `analysis/error_threshold_counts.csv` | ≥10°, ≥15° and failure counts |
| `analysis/improvement_worsening_counts.csv` | per-case improved/worsened ≥1° |
| `analysis/failure_cases_anonymized.csv` | the single measurement failure, anonymised |
| `scripts/` | the three evaluation scripts plus the analysis driver actually used |

## Headline numbers (Full Angle MAE is the clinical endpoint)

| Set | N | Dice E3 / E0 | Full MAE E3 / E0 | ΔMAE [95% CI] |
|---|---:|---:|---:|---:|
| Internal 5-fold | 621 (620 measureable) | 0.9339 / 0.9354 | 3.647 / 3.639 | +0.013 [−0.156, +0.195] |
| Multicenter | 442 | 0.9262 / 0.9267 | 4.030 / 4.088 | −0.058 [−0.389, +0.323] |
| Prospective | 411 | 0.9303 / 0.9325 | 3.330 / 3.365 | −0.034 [−0.156, +0.081] |
| External total | 853 | 0.9282 / 0.9295 | 3.693 / 3.740 | −0.047 [−0.226, +0.161] |

Segmentation is unchanged; every E3-vs-E0 Full Angle delta has a CI spanning zero.
The only significant advantages are internal **Po** vs E0 and external **E3 vs E2**.
