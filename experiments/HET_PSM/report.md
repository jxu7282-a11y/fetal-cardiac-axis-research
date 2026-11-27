# HET-PSM — 中孕期层次预训练（Side Membership）→ 早孕期 Encoder-only 微调

**Version:** `het-psm-v1.0`
**Archived:** 2026-11-28
**Status:** training complete (B1 300/300; B2 5/5 folds × 300 epochs), evaluation and statistics complete
**Primary endpoint:** Full Angle MAE（临床测量主指标），非 Dice。

---

## 1. 假设与定位

中孕期分类分割包含左右心结构。HET-PSM 在中孕期预训练中，于 decoder 末端特征上加一个**左右侧成员（side membership）辅助头**，用对抗式（成员）监督注入左右先验；再把得到的 **encoder 单独**迁移到早孕期微调（Early Decoder 与分割头全新初始化），与无 side 监督的 E3 形成对照。

- 对照：E0（早孕期 scratch）、E3（中孕期 encoder-only 迁移、无 side 监督）。
- 唯一方法差异：中孕期多一个 `L_side` 辅助监督。
- 所有数据、Fold、Plans、GT、几何、评价协议与 E0/E3 完全一致。

## 2. 数据

| 队列 | N | Ne / Po | 角色 |
|---|---:|---|---|
| Retrospective | 621 | 571 / 50 | 五折开发，OOF 评价 |
| Multicenter | 442 | 434 / 8 | 外部测试 |
| Prospective | 411 | 400 / 11 | 前瞻测试 |
| **合计** | **1474** | 1405 / 69 | |

- Fold：E0 冻结 `fold_v2.csv`（未重新生成）；`splits_final.json` 为 E0 版本。
- GT：仅 `label_gold`；`label_infer` 不参与。
- 外部/前瞻队列从未用于训练、选 epoch、调参或后处理。

## 3. 方法

### 3.1 网络结构（B1，中孕期）
- 基座 `PlainConvUNet`（10 类，2D，patch 768, batch 5，Plans `nnUNetPlansFrom501`）。
- `HETPSMPlainConvUNet` 在**最终 decoder 特征**（最后一 decoder stage 输出，运行时读取 `out_channels=32`）上接 `1×1 Conv2d(32→2)` side 头，输出 left/right logits（全分辨率），仅 66 参数。
- `network(x)` 仍返回原 nnUNet 输出（含 deep supervision），side 头不进入分割头。

### 3.2 Side Membership 映射与 Loss
- 目标（来自增强后的分割 GT，同 patch）： `ID 1,2 → Left(ch0)`；`ID 3,4 → Right(ch1)`；其余（含 5–9）**忽略/掩码**，不宣称 5–9 解剖含义。
- Loss：`L_total = L_seg + 0.1 · L_side`。
  - `L_seg`：nnUNet 默认 DC+CE + deep supervision（不变）。
  - `L_side`：`MaskedBCEWithLogits`，逐例 mean，再对有效例 mean；权重 0.1。

### 3.3 B1 训练
- `nnUNetTrainer_HETPSMMid`，Dataset503，fold=all，300 epoch，GPU 0，单进程。
- 状态：DONE，300/300，结束于 2026-11-27 06:50:53，`Mean Validation Dice 0.9728`。
- `checkpoint_final.pth` SHA256 `5900ae4166379861b36fef8eec351e5f0aacd07a472fee08bd71292fbebd5b30`。

### 3.4 B2 五折微调与权重迁移
- `nnUNetTrainer_HETPSMEarly`，Dataset501，2D，300 epoch，五折串行，GPU 0（GPU 3 的 M1.1 未触碰）。
- **仅迁移 encoder**：从 B1 `checkpoint_final.pth` 拷贝 encoder（256 state_dict 项 / 64 唯一张量）；Early Decoder（含 `transpconvs`）与 7 个分割头**全新随机初始化**；严格沿用 E3 `encoder_transfer` 规则与结构断言（decoder/seg 140 项 transfer 前后快照比对未改动，无 side head 写入），不使用 mid decoder 与 side 头。
- 每折独立从 B1 初始化的网络起训；无跨折继承。
- 完成记录：

| Fold | 状态 | 开始 | 结束 | exit |
|---|---|---|---|---|
| 0 | DONE | 2026-11-27 13:15:45 | 16:14:53 | 0 |
| 1 | DONE | 16:14:56 | 19:28:46 | 0 |
| 2 | DONE | 19:28:49 | 21:37:28 | 0 |
| 3 | DONE | 21:37:30 | 23:26:31 | 0 |
| 4 | DONE | 23:26:33 | 2026-11-28 01:16:17 | 0 |

- 全部 checkpoint_best / checkpoint_final 可读，SHA256 见 `results/checkpoints_sha256/`。

## 4. 评价协议（严格复用 E3）
- GT：`local/phase0/fixed_gt_v1.csv`（SHA256 `ca5edd88…`）。
- Fold：`local/phase0/fold_v2.csv`（SHA256 `88b29820…`）。
- 几何：`src/cardiac_axis/evaluate.py`（冻结 E0 管线；ID4/ID5 OBB 加权 card 轴，ID6→ID1 质心 reference 轴）。
- 推理：`checkpoint_best.pth`，`--disable_tta`。
- Internal：B2 fold_{0..4}/validation 的 OOF，覆盖 621。
- External：B2 五折 `checkpoint_best` 集成预测 imagesTs（853）。
- 分层：Internal Ne/Po；External multicenter/prospective。
- 统计：Paired Bootstrap 10000 次，seed=42，95% percentile CI；按 case uid 配对（HET-PSM vs E3、HET-PSM vs E0）。
- NaN / 分割失败保留并单独报告，不静默删除。

## 5. 结果（Full Angle，度）

### 5.1 汇总
| 集合 | N | MAE | MedAE | P90 | P95 | ≥10° | Card MAE | Ref MAE | Dice | IoU | HD95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Internal OOF | 621 (620 可测) | 3.636 | 2.755 | 7.298 | 9.378 | 4.03% | 3.182 | 2.440 | 0.9338 | 0.8883 | 10.79 |
| External ALL | 853 | 3.648 | 3.014 | 7.144 | 9.224 | 2.93% | 2.729 | 2.919 | 0.9281 | 0.8746 | 16.49 |
| – Multicenter | 442 | 3.908 | 3.111 | 7.760 | 9.607 | 4.07% | 2.938 | 3.281 | 0.9260 | 0.8731 | 23.73 |
| – Prospective | 411 | 3.369 | 2.872 | 6.900 | 8.214 | 1.70% | 2.504 | 2.530 | 0.9303 | 0.8763 | 8.71 |

### 5.2 E0 / E3 / HET-PSM 对照（Full Angle MAE）
| 集合 | E0 | E3 | HET-PSM |
|---|---:|---:|---:|
| Internal MAE | 3.639 | 3.647 | **3.636** |
| External MAE | 3.739 | 3.693 | **3.648** |

分层完整指标见 `results/analysis/formal_comparison_table.csv`、`stratified_metrics.csv`。

### 5.3 Paired Bootstrap（full_error Δ = HET − 对照；负=更优）
| scope | comparison | n_paired | Δmean | 95% CI | 显著 |
|---|---|---:|---:|---|---|
| Internal | HET vs E3 | 620 | −0.0108 | [−0.1957, 0.1702] | 否 |
| Internal | HET vs E0 | 620 | +0.0025 | [−0.1450, 0.1503] | 否 |
| External | HET vs E3 | 853 | −0.0448 | [−0.1778, 0.0732] | 否 |
| External | HET vs E0 | 853 | −0.0914 | [−0.2511, 0.0905] | 否 |

全部 comparison（含分层、card_error、ref_error）95% CI 均跨 0。完整表：`paired_bootstrap.csv`、`improvement_worsening_counts.csv`、`error_threshold_counts.csv`。

### 5.4 失败病例
- Internal：1 例 Reference Axis 失败（ID1 像素丢失 → reference 轴与 full 角不可用），E3 与 HET 均同时保留并单独报告；有效配对 620。
- External：无失败；有效配对 853。
- 详见 `results/analysis/failure_cases_anonymized.csv`（`case_hash` 匿名）。

## 6. 结论（阴性）
HET-PSM 的 Full Angle MAE 在 Internal（3.636 vs E0 3.639 / E3 3.647）与 External（3.648 vs E0 3.739 / E3 3.693）上**数值略优**，但**所有 Full Angle 配对 Bootstrap 95% CI 均跨 0**，**不能宣称具有统计学显著优势**，也不能据此宣称临床测量改善。加入 side membership 辅助监督相对 E3 未产生可测量、可显著区分的增益；Dice/IoU 亦基本持平。

## 7. 限制
- 外部 multicenter 方向略优但不显著；Internal Po 子集仅 50 例，波动大。
- Fold 分组基于 case_id，非独立核实的患者级。
- 单次训练、单入口，无重复种子方差估计。

## 8. 复现要点
- 环境变量见 `config.json.environment_variables_required`；`nnUNet_extTrainer=$PWD/experiments`（Early trainer 位于 `experiments/HET_PSM/early_trainer.py`）。
- B1：`bash local/het_psm/mid_pretraining/run_mid.sh`（B1 实际使用，mid_pretraining 目录）。
- B2：`bash scripts/19_train_het_psm_b2.sh`（本归档构建，串行五折，拒绝覆盖已有 final）。
- 评价：`local/het_psm/evaluation/scripts/eval_het_psm.py`（Internal/External 指标）与 `run_het_psm_analysis.py`（对照/bootstrap/分层/失败）；External 预测 `external_predict_het_psm.sh`。
- 原始产物（权重、预测、含 uid 逐病例 CSV）仅保留在 `local/het_psm/`，不入 Git；本目录只保留汇总与匿名化结果。

## 9. 归档清单
`results/metrics/`（匿名 case-level + summary）、`results/analysis/`（对照/ bootstrap/分层/阈值/改善/失败/审计）、`results/checkpoints_sha256/`（B1+B2 权重哈希、plans/splits/GT/fold 哈希）。代码：`experiments/HET_PSM/*.py`、`tests/`（27 CPU 单测）。
