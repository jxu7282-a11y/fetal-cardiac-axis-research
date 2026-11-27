# HET-PSM — 中孕期层次预训练（Side Membership）→ 早孕期 Encoder-only 微调

Version: `het-psm-v1.0`
Status: complete — B1 300/300 epoch, B2 5/5 folds × 300 epochs; formal evaluation and statistics done.

中孕期预训练在 decoder 末端加一个左右侧成员（side membership, `1×1 Conv(32→2)`）辅助头，
`L_total = L_seg + 0.1·L_side`；随后只把 **encoder** 迁移到早孕期 `Dataset501`（decoder 与 7 个
分割头全新随机），与无 side 监督的 E3 构成对照。数据/Fold/Plans/GT/几何/评价协议与 E0/E3 完全一致。

结论：Full Angle MAE 数值略优（Internal 3.636、External 3.648），但所有配对 Bootstrap 95% CI 均跨 0，
**无统计学显著优势**，不能宣称临床测量改善。

## 内容
| 文件 | 内容 |
|---|---|
| `report.md` | 方法、训练配置、迁移规则、评价协议、结果、统计、阴性结论、限制、复现 |
| `config.json` | 数据、网络、side 映射/Loss、B1/B2 完成状态、评价协议、headline 结果 |
| `results/metrics/internal_summary.json` / `external_summary.json` | 汇总（621 / 853） |
| `results/metrics/*_case_level_anonymized.csv` | 逐病例指标（仅 `case_hash`） |
| `results/analysis/formal_comparison_table.csv` | E0/E3/HET-PSM × scope × subgroup |
| `results/analysis/paired_bootstrap.csv` | 配对 bootstrap（10000，seed 42，95% CI） |
| `results/analysis/stratified_metrics.csv` | Ne/Po 与 multicenter/prospective |
| `results/analysis/error_threshold_counts.csv` | ≥10°/≥15°/失败计数 |
| `results/analysis/improvement_worsening_counts.csv` | 逐例改善/恶化 ≥1° |
| `results/analysis/failure_cases_anonymized.csv` | 测量失败病例（匿名） |
| `results/analysis/cohort_match_audit.json` | uid 集/GT/NaN 审计 |
| `results/checkpoints_sha256/` | B1+B2 权重 SHA256；plans/splits/GT/fold 哈希 |
| `*.py`, `tests/` | HET-PSM 网络/side/Loss/迁移代码与 27 项 CPU 单测 |

权重、影像、Mask、预测与含 uid 的逐病例 CSV 均保留在 `local/het_psm/`，不入 Git。
