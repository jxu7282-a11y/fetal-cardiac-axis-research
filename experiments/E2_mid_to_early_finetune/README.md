# E2 — Mid-Trimester Pretraining → Early-Trimester Fine-Tuning

Version: `e2-transfer-v1.0`
Status: complete (5/5 folds, 300 epochs each; formal evaluation done)

## What this is

The flagship transfer experiment: early-trimester nnUNetv2 2D models are **initialized from
the mid-trimester pretrained checkpoint** (`Mid_pretraining`) via nnU-Net's official
`-pretrained_weights` loader, then fine-tuned on the early-trimester `Dataset501` five-fold
split. It answers: *does cross-trimester pretraining improve early-trimester cardiac-axis
measurement versus the E0 from-scratch baseline?*

The answer recorded here is a **neutral / null result**: no clinically meaningful Full Angle
improvement at the population level. Full Angle MAE and P95 — not Dice — are the primary
criteria.

## Contents

| File | Content |
|---|---|
| `README.md` | this file |
| `report.md` | setup, 382/382 transfer, full internal/external metrics, E0 vs E2 paired comparison with CIs, Card/Reference/Full geometry decomposition, Gold-vs-Infer, 64 hard cases, 5 anonymized severe-failure findings, limitations, neutral conclusion |
| `config.json` | configuration: dataset, transfer contract, training, evaluation protocol, headline + geometry metrics |
| `metrics/segmentation_metrics.csv` | Dice/IoU/HD95 per label, per set, E2 and E0 |
| `metrics/angle_metrics.csv` | Full Angle MAE/MedAE/P90/P95, Card/Ref error, failure rate, per set & Ne/Po, E2 and E0 |
| `metrics/E2_vs_E0_comparison.csv` | E2−E0 deltas with paired-bootstrap CIs (MAE, P95) |
| `metrics/axis_error_comparison.csv` | Card/Ref/Full mean error decomposition by population |
| `metrics/gold_infer_comparison.csv` | Gold Mask vs E0-Infer vs E2-Infer per axis |

## Headline numbers (Full Angle MAE is the clinical endpoint)

| Set | N | Dice E2 / E0 | Full MAE E2 / E0 | ΔMAE [95% CI] |
|---|---:|---:|---:|---:|
| Internal 5-fold | 621 | 0.9337 / 0.9354 | 3.607 / 3.639 | −0.031 [−0.170, 0.111] |
| Multicenter | 442 | 0.9247 / 0.9267 | 4.317 / 4.088 | +0.229 [−0.152, 0.676] |
| Prospective | 411 | 0.9304 / 0.9325 | 3.368 / 3.365 | +0.004 [−0.123, 0.126] |
| External total | 853 | 0.9275 / 0.9295 | 3.860 / 3.739 | +0.120 [−0.082, 0.356] |

All Full Angle deltas have 95% CIs spanning zero; segmentation is essentially unchanged.
Cross-trimester pretraining did **not** beat the E0 scratch baseline on the clinical endpoint.

## Geometry insight

E2 has **opposite-signed effects on the two axes**: it improves the Cardiac axis
(ΔCard MAE −0.11° overall, −0.20° external) but degrades the Reference axis
(ΔRef MAE +0.17° overall, +0.38° external). The rare severe negative-transfer cases
(5 Multicenter normal cases) are a **Reference-axis** failure (ID6 largest-component /
centroid flip), while the Cardiac axis is fine or better. See `report.md` for detail.

## Not in this repository

Deliberately excluded: original images, masks, TAR archives, model weights and
`checkpoint_*.pth`, all prediction PNGs, all case-level CSVs (identifiers), manifests /
file-path maps, training logs, and predictions. Only anonymized aggregate metrics are
archived here. Weights and case-level tables remain server-side under
`local/e2_mid_to_early_finetune/`.
