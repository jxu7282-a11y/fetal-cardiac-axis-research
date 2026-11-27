# 03｜实验计划与进度日志（EXPERIMENT LOG）

**文档版本：** v1.0  
**整理日期：** 2026-11-23  
**状态：** E0 早孕期五折训练已完成；中孕期全量预训练已完成；早孕期微调尚未开始

## 1. 当前研究假设

中孕期数据包含相对丰富的解剖分割标注。尽管中孕期与早孕期来自不同胎儿、没有样本级配对，且标签并不完全一致，中孕期数据仍可能帮助模型学习一部分可迁移的解剖表示，从而改善早孕期分割，并最终降低早孕期心轴角测量误差。

该假设必须通过对照实验验证，不得预设成立。

## 2. 实验前提

- 早孕期正式数据：1474例。
- 中孕期数据：原始905例；按图像SHA256去除15组重复后，890份独立图像用于预训练。
- 两批数据来自完全独立的不同胎儿。
- 每个胎儿一张静态二维图像，无视频。
- 标签空间仅部分相关。
- 最终任务只在早孕期图像上进行心轴角测量。
- 中孕期图像不作为早孕期模型推理时的必要输入。
- 早孕期正式GT遵循`02_GT_PROTOCOL.md`。

## 3. 建议分阶段推进

### Phase 0：数据与评价流程冻结

目标：先保证所有实验使用相同的数据版本、划分、后处理和评价。

任务： - 核对正式Manifest和Fold； - 处理阳性病例207条路径不匹配； -
核实跨队列重复ID候选； - 明确早孕期/中孕期标签语义； -
固定TAR读取、Gold语义分配和Full Angle计算； - 复核图像与Mask读取； -
保存版本、哈希、脚本版本和有效N。

验收条件： - 正式病例集合固定； - 所有输入路径可复现； -
GT处理可重复； - 每个病例的分割指标与角度指标可追溯。

### Phase 1：早孕期单独训练基线

优先使用用户倾向的`nnUNetv2`建立早孕期分割基线。

目的： - 确认当前早孕期标签能否稳定训练； - 建立分割指标基准； -
建立从预测Mask到Full Angle的完整评价基准。

初期不增加复杂多任务分支或端到端角度回归。

### Phase 2：中孕期预训练 → 早孕期微调

核心对照：

| 实验 | 训练方式                     | 要回答的问题             |
|------|------------------------------|--------------------------|
| E0   | 早孕期数据从头训练           | 早孕期基线性能           |
| E1   | 中孕期预训练后，在早孕期微调 | 中孕期知识是否有迁移价值 |

标签空间不一致时，先依据真实Label字典确定可用监督。不要将无关类别硬对齐。共享Encoder/独立Decoder、部分标签监督等复杂方案先作为备选，不必在第一轮全部实现。

中孕期数据自身的训练/验证划分已由用户选定：方案A，890例去重后病例全部用于
fold=all 预训练，不设独立中孕验证/测试集；同数据监控指标不能作为泛化结论。

### Phase 3：联合评价分割、几何与临床测量

同一批早孕期病例、同一后处理、同一GT规则下比较：

**分割层** - Dice； - IoU； - 必要时HD95/ASD等边界指标。

**几何层** - Card Axis Error； - Reference Axis Error。

**临床层** - Full Angle MAE； - Median Absolute Error； - P90/P95； -
预先定义的误差阈值达标率； - 各指标N_valid。

Full Angle MAE为主评价目标。分割指标不能替代临床测量指标。

### Phase 4：分割误差与角度误差关系分析

如果出现Dice改善但Full Angle没有改善，进一步检查： -
哪些结构的分割误差影响Card方向； -
哪些结构的分割误差影响Reference方向； - 质心偏移与方向误差的关系； -
边界局部误差是否影响OBB长轴； - 不同结构对Full Angle的敏感性。

这一阶段用于决定是否需要Measurement-aware或Geometry-aware训练约束。

### Phase 5：方法改进与消融

只有在前面实验发现明确瓶颈后，才考虑： - 测量任务辅助损失； -
几何方向辅助头； - 共享Encoder与独立Decoder； - 部分标签监督； -
其他跨域迁移策略。

避免在第一轮同时引入多个新机制，导致结果难以归因。

## 4. 数据划分与评价边界

早孕期： - Retrospective 621例：五折开发，正式内部性能使用OOF结果； -
Multicenter 442例：外部测试； - Prospective 411例：前瞻测试。

外部队列只在方法和超参数冻结后进行评价，不能用于模型选择。

现有分组依据是`case_id`，患者身份尚未独立核实。实验报告应准确描述分组层级。

中孕期： - 原始905例；图像SHA256重复15组，去重后890例； -
本阶段按方案A使用全部890例做预训练，不划分独立验证/测试集； -
需先核实是否存在内部来源/设备/中心差异； -
不得假设与早孕期有病例配对。

## 5. 每次实验必须记录

建议每个实验以独立ID记录：

| 字段          | 内容                                         |
|---------------|----------------------------------------------|
| Experiment ID | 唯一编号                                     |
| 日期          | 开始/结束日期                                |
| 假设          | 本次验证的问题                               |
| 数据版本      | Manifest、Fold或中孕数据清单版本             |
| 数据哈希      | 关键输入SHA256                               |
| 模型          | nnUNetv2配置/模型结构                        |
| 初始化        | 随机初始化或中孕预训练权重                   |
| 标签映射      | 使用的类别映射版本                           |
| 训练配置      | Epoch、batch、patch、optimizer等             |
| 随机种子      | Seed                                         |
| 训练划分      | Fold与训练数据来源                           |
| 分割结果      | Dice/IoU等                                   |
| 几何结果      | Card/Reference误差                           |
| 临床结果      | Full Angle MAE及分布                         |
| N_valid       | 各指标有效样本数                             |
| 结论          | 支持/不支持/不确定                           |
| 产物路径      | 日志、权重、预测、CSV                        |
| 状态          | planned/running/completed/failed/invalidated |

## 6. 实验记录区

### E0｜早孕期nnUNetv2基线

- 状态：completed（五折训练已完成；正式结果汇总以现有 E0 产物为准）
- 研究问题：早孕期单独训练的分割和心轴角测量基线是多少？
- 数据版本：Dataset501_EarlyRetroCardiac，Retrospective 621例五折开发集
- 训练配置：nnUNetTrainer_300Epochs，2D，nnUNetPlans，300 epochs/fold
- 产物：`local/nnunet_results/Dataset501_EarlyRetroCardiac/nnUNetTrainer_300Epochs__nnUNetPlans__2d/fold_0..4`
- 说明：本任务未修改 E0 的任何数据、Plans、Fold 或权重。

### E1｜中孕期预训练→早孕期微调

- 状态：planned（中孕预训练阶段已完成，早孕微调尚未开始）
- 研究问题：中孕期预训练是否降低早孕期Full Angle误差？
- 中孕数据清单：905原始 → 890去重训练例；见 `local/mid_pretraining/data_audit/mid_case_manifest.csv`
- 标签映射：恒等保留 0–9；Label 1–9 解剖语义仍待确认；不得与早孕 1–6 强行对应
- 中孕预训练配置：Dataset503_MidPregnancyCardiac，nnUNetPlansFrom501，2D，fold=all，300 epochs，随机初始化
- 中孕最终权重：`local/mid_pretraining/results/checkpoint_final.pth`
- 结果：中孕预训练已完成；早孕微调尚未启动，暂无 E1 迁移效果结论
- 结论：待后续五折微调与 Full Angle 评价

#### E1-pre｜中孕期全量预训练记录

- Experiment ID：MID-PRETRAIN-503-2D-ALL-300EP
- 状态：completed
- 原始/去重/训练例数：905 / 890 / 890
- 重复组：15组×2文件；排除15例，均保留原始文件并记录于 `mid_dedup_excluded.csv`
- 标签：0–9，共10类；1–9语义待确认
- 网络：PlainConvUNet，3通道输入，8级，patch 768×768，batch 5
- Plans：`nnUNetPlansFrom501`（由E0 Dataset501 `nnUNetPlans` 经官方 `nnUNetv2_move_plans_between_datasets` 迁移）
- Trainer：`nnUNetTrainer_300Epochs`（相对 stock 仅 `num_epochs=300`）
- Fold：all（全部890例既训练又参与同数据监控）
- Epoch：300/300
- GPU：物理 GPU 3，NVIDIA H20-3e；进程内 cuda:0
- 训练时长：约2.81小时（300轮 epoch 计时合计）；平均33.71秒/轮
- 同数据监控：Mean foreground Dice 0.9784 / IoU 0.9581；**不得作为独立泛化或临床结论**
- 权重：`local/mid_pretraining/results/checkpoint_final.pth`，SHA256 `df94a1d9aac1557a50d63856d684ef236508b45731dbcfd3479c85493314cd9f`
- 日志与配置：`local/mid_pretraining/results/`、`local/mid_pretraining/experiment_config.json`
- E0兼容性：382个非分割键名称/形状全部一致；仅14个 `.seg_layers.` 因7类 vs 10类不同，官方加载器会跳过并重新初始化

### M1｜Measurement-aware 双轴辅助监督

- 状态：archived（五折 × 300 epoch 完成，评价与统计完成）
- 研究问题：在 E0 分割 decoder 末端加 Card/Reference 双角度辅助头，能否降低 Full Angle 误差？
- 结论：内部显著变差（vs E0 +0.181°，vs E3 +0.168°），外部不显著；辅助头因 GAP 退化为近常数预测器
- 归档：`experiments/M1_measurement_aware_dual_axis/`；原始产物保留在 `local/m1_measurement_aware/`，未修改

### M1.1｜Spatial-preserving 双轴辅助监督

- Experiment ID：M1.1
- 状态：prepared（smoke test 通过；**正式五折训练未启动，等待批准**）
- 研究问题：把辅助头 `AdaptiveAvgPool2d(1)` 换成 `AdaptiveAvgPool2d((4,4))`，其余完全不变，能否修复 M1 的辅助头常数化失效？
- 唯一改动：辅助头池化 `1×1 → 4×4`；Flatten 后 FC 输入 `32 → 512`；辅助参数每头 `66 → 1026`
- 保持不变：双轴任务与标签、loss 类型/权重/归一化、主分割网络与分割头、数据、Fold、Plans、300 epoch、随机种子与初始化方式
- 初始化：scratch，不加载 E2/E3/M1 权重
- 验证：21 项 CPU 单测通过；GPU 3 smoke（1 个缩短 epoch）通过；backbone 与同种子 M1 逐张量一致，分割输出一致；辅助输出不再为常数
- 代码：`experiments/M1_1_spatial_dual_axis/`
- 产物目录：`local/m1_1_spatial_dual_axis/`（`run_m1_1_5fold.sh`、`smoke_m1_1.py`、`smoke/smoke_report.json`）
- GPU：物理 GPU 3（smoke 时该卡有他人约 13 GB 进程，剩余显存充足，未抢占）
- 下一步：用户批准后启动五折正式训练

## 7. 结果解释规则

- Dice提升、Full Angle MAE不变：不能宣称临床测量改善。
- Dice下降、Full Angle
  MAE下降：需要分析是否关键几何结构更准确，并检查其他风险。
- Card改善、Reference不变：需要判断Full Angle是否随之改善。
- 只有某一外部队列改善：需分析队列差异，不得据此反向调参。
- N_valid变化：必须显式报告，不能用不同分母的MAE直接比较。

### HET-PSM｜中孕期层次预训练（Side Membership）→ 早孕期 Encoder-only 迁移

- Experiment ID：HET-PSM（B1 中孕期预训练 + B2 早孕期五折微调）
- 状态：archived（B1 300/300；B2 五折 × 300 epoch 完成；评价与统计完成）
- 研究问题：中孕期加左右侧成员辅助监督（`L_seg + 0.1·L_side`），其 encoder 迁移到早孕期后，能否优于 E0（scratch）与 E3（纯 encoder 迁移、无 side 监督）？
- 方法：B1 在 decoder 末端特征加 `1×1 Conv(32→2)` side 头（ID1,2→Left；ID3,4→Right；其余掩码），MaskedBCE；B2 仅迁移 encoder（256 项/64 张量），decoder 与 7 分割头全新，严格沿用 E3 transfer 规则。数据/Fold/Plans/GT/几何/评价协议与 E0/E3 一致。
- 结果（Full Angle MAE，度）：Internal 621 例 E0/E3/HET-PSM = 3.639/3.647/3.636；External 853 例 = 3.739/3.693/3.648。
- 统计：Paired Bootstrap（10000，seed42，95% CI）full_error Δ 全部跨 0（Internal −0.011/−0.003；External −0.045/−0.091，vs E3/vs E0），**无统计学显著优势**。
- 失败病例：Internal 1 例 Reference Axis 失败（保留未删），External 无；有效配对 620 / 853。
- 归档：`experiments/HET_PSM/`（report.md、config.json、results/）；原始产物保留在 `local/het_psm/`，未修改 E0/E1/E2/E3/M1.1。
