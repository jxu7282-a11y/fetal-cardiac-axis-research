# CARDIAC_AXIS_METHOD_HISTORY

> **文档状态：** Active  
> **文档版本：** 1.0  
> **适用项目：** 早孕期胎儿心轴角自动测量  
> **创建日期：** 2026-08-04  
> **建议保存位置：** `/data5/xjx/zaoyun/data/j/docs/CARDIAC_AXIS_METHOD_HISTORY.md`  
> **配套数据规范：** `CARDIAC_AXIS_DATA_READING_SPEC_v1.md`  
> **文档目标：** 记录项目已经尝试过的方法、每个阶段的输入输出、设计动机、已知优点、已知缺陷、历史结果状态、被暂停的结论，以及当前正式工作应从哪里继续。

---

## 0. 文档定位

本文件是项目的**方法历史与决策记录**，不是数据读取规范，也不是当前最终论文方法说明。

本文件用于回答：

1. 过去做过哪些方法；
2. 每种方法解决什么问题；
3. 方法之间如何串联；
4. 哪些结果只是历史探索；
5. 哪些结论仍然可信；
6. 哪些结论因版本、候选定义或评估链问题而暂停；
7. 为什么当前不继续增加V2D/V2E等补丁；
8. 下一步应回退检查哪一层。

基础数据读取、人工线读取、card/ref语义分配、mask读取、角度定义和队列划分，统一以：

```text
CARDIAC_AXIS_DATA_READING_SPEC_v1.md
```

为准。

任何方法脚本不得通过本文件重新解释数据读取逻辑。

---

## 1. 当前正式研究目标

项目的最终任务是：

```text
由早孕期胎儿超声图像及其区域分割结果，自动估计胎儿心轴角。
```

心轴角由两条无方向轴构成：

```text
card axis
reference axis
```

最终临床角度为两条轴之间的锐角：

```text
full cardiac-axis angle ∈ [0°,90°]
```

### 1.1 当前主要瓶颈

现阶段已明确：

- `reference axis`结果已经较好；
- 当前主要攻克对象是`card axis`；
- 后续不应无理由扩展为“双轴一起修正”；
- ref轴主要作为稳定参考，用于由card轴重新计算full角。

### 1.2 端点约束

人工测量线的端点本身没有稳定、明确的解剖学意义。

因此主方法不应包装成：

```text
关键点/端点定位
```

历史和未来方案均应优先围绕：

```text
区域分割
→ 几何方向
→ card轴估计
```

展开。

---

## 2. 正式数据基准

当前正式样本基准：

| 队列 | Ne | Po | 总数 |
|---|---:|---:|---:|
| retrospective | 571 | 50 | 621 |
| multicenter | 434 | 8 | 442 |
| prospective | 400 | 11 | 411 |
| **总计** | **1405** | **69** | **1474** |

数据角色：

- 回顾性621例：患者级五折OOF开发与方法选择；
- 多中心442例：外部测试；
- 前瞻411例：前瞻测试；
- 外部数据不得参与选参、候选删除或方法选择。

---

## 3. 方法历史总览

项目历史方法链可概括为：

```text
区域分割结果
    ↓
固定几何基线
    ↓
Tree/表格残差修正
    ↓
风险/置信度识别
    ↓
高风险病例候选生成
    ↓
V2B/V2候选视觉排序
    ↓
V2C效用与伤害控制
    ↓
只修改card轴并重新计算full角
```

该链路后期逐渐形成：

```text
主估计器
+
风险门
+
候选选择器
+
安全门
+
幅度裁剪
+
混合
```

当前已认识到：方法越来越像在原估计器后面连续打补丁，因此不应继续直接增加新的门控层，而应回退诊断card轴候选和前置融合规则。

---

# 第一部分：固定几何基线

## 4. Fixed Geometry Baseline v2

### 4.1 直接生成文件

历史脚本直接生成：

```text
fixed_geometry_baseline_v2/
├── fixed_geometry_baseline_v2_case_detail.csv
└── fixed_geometry_baseline_v2_results.xlsx
```

输入：

```text
clean_dataset_manifest_v1/clean_manifest.csv
clean_dataset_manifest_v1/clean_5fold_split.csv
```

### 4.2 人工GT读取

人工tar中恰好两条测量线。

脚本：

1. 从tar内JSON递归查找`MeasureModel`；
2. 读取`Pos.p1`和`Pos.p2`；
3. 在人工标注自身坐标系中计算两条无方向轴；
4. 从`label_gold`生成`gold_card_axis`和`gold_ref_axis`；
5. 比较两种一一排列总代价；
6. 确定固定的`manual_card_axis`和`manual_ref_axis`。

后续所有card方法统一与固定`manual_card_axis`比较。

### 4.3 Card固定几何

对`label_infer`中的id4和id5分别：

```text
最大连通域
→ OBB
→ OBB长轴方向
```

区域权重：

```text
area × max(aspect - 1.0, 0.05)
```

使用双角度无方向加权平均得到：

```text
infer_card_axis
```

### 4.4 Ref固定几何

```text
id6最大连通域质心
→
id1最大连通域质心
```

两质心连线方向作为：

```text
infer_ref_axis
```

### 4.5 Full角

```text
infer_full_angle
=
acute_angle(infer_card_axis, infer_ref_axis)
```

### 4.6 优点

1. 不依赖无解剖意义的端点；
2. 计算简单、可复现；
3. 每个输出方向可追溯到分割区域；
4. ref轴定义清楚；
5. 可作为所有后续方法的统一基线；
6. 不需要端到端大模型；
7. 适合先验证“区域是否足以推导心轴”。

### 4.7 缺陷

1. id4/id5方向只依赖OBB长轴；
2. aspect接近1时长轴可能不稳定；
3. id4与id5方向差异较大时，固定加权平均可能得到两者都不理想的中间方向；
4. 权重由面积和elongation决定，但“权重大”不一定等于“方向更可信”；
5. 分割缺损、尾部、小伪连通域或区域不完整会影响OBB；
6. 该规则没有学习“什么形态下应信id4，什么形态下应信id5”。

### 4.8 当前状态

固定几何应保留为：

```text
基础基线
+
候选来源诊断对象
```

但不能默认它就是最终card轴主方法。

---

# 第二部分：Tree/残差修正

## 5. Dual-axis Residual Correction v2

### 5.1 历史脚本目标

输入：

```text
fixed_geometry_baseline_v2_case_detail.csv
```

从`label_infer`重新提取：

- id1/id4/id5/id6区域特征；
- 面积；
- 最大连通域质心；
- OBB；
- PCA；
- 距离；
- 角度的sin(2θ)、cos(2θ)表示；
- 固定几何已有字段。

模型包括：

```text
ExtraTreesRegressor
RandomForestRegressor
```

目标是预测：

```text
delta_card
delta_ref
```

随后修正：

```text
corrected_card_axis
corrected_ref_axis
```

并重新计算：

```text
corrected_full_angle
```

### 5.2 “Tree”的含义

下游文件中的：

```text
B0_Current_Tree
line1_card_corrected_axis
line2_ref_corrected_axis
full_corrected_angle
```

通常不是最原始OBB规则，而是已经经过表格特征残差修正的当前Tree结果。

因此后续文件中看到“Tree”时，必须确认具体来源版本，不能自动等同于固定几何原始轴。

### 5.3 训练协议

- 回顾性五折OOF；
- 每折训练四折、预测一折；
- 外部队列使用五折模型集成；
- 外部队列不参与模型选择。

### 5.4 优点

1. 能利用分割形态和区域几何特征；
2. 不改变端点定义；
3. card/ref/full均可重新计算；
4. 比固定权重规则更灵活；
5. 保留明确的病例级delta；
6. 可作为稳定的表格模型基线。

### 5.5 缺陷

1. 同时修card和ref，但当前主要瓶颈其实是card；
2. Tree模型学习的是“对现有几何结果的修正”，仍受固定候选/特征表达限制；
3. 如果card轴来源本身没有提供正确方向，残差模型只能有限修补；
4. 表格特征不能充分利用超声纹理和局部区域关系；
5. 后续方法逐渐把Tree当成不可动的默认结果，导致所有改进都变成fallback。

### 5.6 当前状态

保留为历史基线和对照，不直接认定为最终主方法。

下一步应检查：

```text
Tree问题是否主要来自：
- id4/id5候选方向
- 固定融合
- 残差修正
- 分割误差
```

---

# 第三部分：风险与置信度

## 6. Image-mask Confidence / Selective Prediction

### 6.1 设计动机

发现Tree在大多数病例中表现尚可，但少数病例存在明显尾部误差，因此尝试预测：

```text
Tree是否可能失败
```

并生成：

```text
selected_risk_score
```

后续只让高风险病例进入候选修正。

### 6.2 典型输出

```text
tree_confidence_selective_prediction_v1/
└── tree_confidence_selective_prediction_v1_case_flags.csv
```

### 6.3 优点

1. 保护低风险病例；
2. 减少不必要修改；
3. 适合临床安全型策略；
4. 可输出风险分数和门控状态；
5. 便于分析尾部病例。

### 6.4 缺陷

1. “Tree误差大”不等于“存在可被候选救回的方向”；
2. “Tree误差中等”也可能存在明显更优候选；
3. 风险模型解决的是失败检测，不是候选选择；
4. 高风险门覆盖范围有限；
5. 后续候选改进空间被门控提前截断。

### 6.5 历史判断

历史分析认为：

- 风险门只覆盖部分可改善病例；
- 大量中等误差病例没有进入候选选择；
- 因此“只在高风险病例上修正”可能是架构层面的限制。

相关数值属于历史分析记录，需在新正式版本链上重新计算。

### 6.6 当前状态

风险分数可保留用于诊断，但暂不应继续作为所有card候选方法的强制入口。

---

# 第四部分：候选池与Tail Failure Analysis

## 7. Card Candidate Pool

### 7.1 历史候选名称

历史候选池曾包括：

```text
B0_Current_Tree
corrected_tree
fixed_weighted_obb
id4_id5_centroid
id4_obb_feat
id4_obb_mask
id4_pca_feat
id4_pca_mask
id5_obb_feat
id5_obb_mask
id5_pca_feat
id5_pca_mask
union45_obb_mask
union45_pca_mask
weighted_obb_feat
weighted_pca_feat
```

### 7.2 必须注意

上述名称不能直接视为完整定义。

必须进一步确认：

- 输入是`label_gold`还是`label_infer`；
- 使用最大连通域还是全部区域；
- OBB/PCA如何实现；
- `feat`是已有字段还是现场重算；
- `mask`版本是否使用不同连通域；
- 缺失时是否回退；
- 是否存在相同方向的重复实现。

在候选生成脚本完全确认前，不得仅凭名称得出结论。

### 7.3 候选池设计动机

固定Tree使用一种融合规则，但不同病例可能更适合：

```text
id4单独
id5单独
OBB
PCA
联合区域
其他几何方向
```

候选池希望提供多个可能的card方向，再由选择器选出最优方向。

### 7.4 历史优点

1. 很多病例中存在比Tree更好的方向；
2. 可以解释“选择了哪种几何来源”；
3. 不预测端点；
4. 可将Tree保留为安全候选；
5. 允许研究不同区域和几何方法的适用条件。

### 7.5 历史缺陷

#### 7.5.1 候选名称多，真实方向少

多个候选名在同一病例中可能产生近似相同方向。

因此：

```text
candidate count
```

不能等同于：

```text
distinct-axis count
```

#### 7.5.2 候选语义可能不适合作为card轴

例如：

```text
id4_id5_centroid
```

描述的是两个区域质心关系，不一定等价于心脏card方向，可能产生接近垂直的系统性偏差。

#### 7.5.3 候选总体危险但局部有用

某个候选可能平均误差很差，但能唯一救回少数Tree灾难病例。

因此不能只按平均MAE自动删除。

#### 7.5.4 候选池覆盖失败

部分病例中，所有候选都远离人工card轴。这属于候选生成问题，不是选择器问题。

#### 7.5.5 候选不可辨识

即使oracle能选出最小误差候选，最佳与第二候选真实误差可能非常接近，图像上也未必有稳定证据区分。

### 7.6 历史Oracle结果

历史项目分析曾记录：

```text
Tree card MAE      ≈ 2.705641°
V2 card MAE        ≈ 2.671752°
V2C card MAE       ≈ 2.663590°
Candidate oracle   ≈ 1.029197°
```

这些数值只用于记录当时为何继续研究候选选择。

它们当前状态为：

```text
Historical / pending version-chain revalidation
```

不得直接写入新正式论文结果，原因包括：

1. 旧目录与新正式clean manifest尚未完全重建；
2. 候选定义未全部完成源码追溯；
3. 后期Core7、删除regret、盲评等分析存在方法学问题；
4. 正式数据需要重新从clean manifest运行。

### 7.7 当前状态

候选池不是直接废弃，而是进入：

```text
card候选来源诊断
```

重点判断：

- 候选池是否覆盖正确方向；
- gold与infer候选差异；
- id4/id5融合是否有问题；
- 候选是否重复；
- 候选是否可辨识；
- 哪些失败属于候选生成；
- 哪些失败属于选择器或门控。

---

# 第五部分：V2B / V2候选视觉排序

## 8. High-risk Candidate Line CNN Scorer v2 Rank Strip

### 8.1 命名说明

当前源码确认的脚本名为：

```text
run_highrisk_candidate_line_cnn_scorer_v2_rank_strip.py
```

在项目口头讨论中可能被称为：

```text
V2
V2B
CandidateLineCNN v2
rank-strip
```

在新文档或新代码中必须固定一个正式名称，并注明历史别名，避免版本混淆。

### 8.2 输入

典型输入：

```text
tree_confidence_selective_prediction_v1_case_flags.csv
cardiac_tail_failure_analysis_v1_casewise.csv
cardiac_tail_failure_analysis_v1_candidate_errors.csv
```

### 8.3 核心思想

每条card候选不只表示成表格角度，而是画成视觉假设，与超声图、mask、Tree轴和ref轴共同输入CNN。

历史12通道包括：

1. 超声ROI；
2. id1 mask；
3. id4 mask；
4. id5 mask；
5. id6 mask；
6. Tree cardiac axis；
7. candidate cardiac axis；
8. reference axis；
9. union45 boundary；
10. candidate local strip；
11. strip内超声；
12. strip内boundary。

### 8.4 训练目标

模型输出候选代理误差，越小越好。

历史损失包括：

```text
soft same-case ranking
+
pairwise ranking
+
auxiliary SmoothL1 regression
```

### 8.5 应用规则

- 候选池中始终加入`B0_Current_Tree`；
- 低风险病例保持Tree；
- 高风险病例才允许使用候选；
- 需要预测收益margin；
- 需要top1-top2 margin；
- 修正delta需要clip；
- 最终使用blend；
- ref轴保持不变；
- full角由修正card轴与原ref轴重新计算。

### 8.6 优点

1. 将候选作为明确视觉假设；
2. 能利用超声纹理和局部边界；
3. 不预测无意义端点；
4. Tree始终可作为安全候选；
5. 同病例排序比全局角度回归更贴合候选任务；
6. 输出候选级分数，具有病例可解释性。

### 8.7 缺陷

#### 8.7.1 高风险入口限制

大量可改善病例可能未进入风险门。

#### 8.7.2 候选重复干扰margin

多个候选名可能是同一方向，导致top1-top2 margin并不代表真正不同方向之间的置信度。

#### 8.7.3 候选质量决定上限

候选池没有正确方向时，CNN无法修复。

#### 8.7.4 视觉可辨识性不确定

人工GT最优候选不一定能从现有图像、mask和局部条带中稳定判断。

#### 8.7.5 只修card，不解决候选来源

CNN只是选择器，不改变id4/id5几何定义和融合规则。

#### 8.7.6 修改被clip/blend压缩

即使选中正确候选，最终应用幅度可能只利用一部分理论收益。

### 8.8 当前状态

保留为历史方法和诊断对象。

不建议在其后直接继续增加更复杂的V2D选择层，除非候选来源诊断证明：

```text
候选池有清晰好候选
+
人能够稳定识别
+
主要瓶颈确实是V2B排序模型
```

---

# 第六部分：V2C效用与伤害控制

## 9. Candidate Utility / Harm Control v2C

### 9.1 设计动机

V2选择“预测误差最小”的候选，但临床决策更关心：

```text
相对Tree是否有真实收益
是否可能造成伤害
是否值得修改
```

因此V2C在候选预测基础上增加：

- gain预测；
- harm概率；
- better-than-Tree概率；
- utility；
- consensus；
- 修正幅度惩罚；
- 更严格门控。

### 9.2 概念性效用

历史思想可概括为：

```text
utility
=
expected gain
- harm penalty
- delta penalty
- isolation / inconsistency penalty
```

### 9.3 优点

1. 明确优化“是否值得改Tree”；
2. 更关注伤害控制；
3. 可保护Tree已好的病例；
4. 能输出拒绝原因；
5. 适合作为临床安全策略研究；
6. 比单纯候选排名更符合决策逻辑。

### 9.4 缺陷

#### 9.4.1 不能创造新候选

V2C只能在V2候选池和预测基础上做决策。

#### 9.4.2 不能修复风险门漏检

未进入前置选择链的病例仍无法改善。

#### 9.4.3 层层门控进一步压缩收益

V2C增加：

```text
utility threshold
harm threshold
better probability
consensus
delta restriction
```

在安全性提高的同时，也可能拒绝真实好候选。

#### 9.4.4 方法复杂度增加

最终链路难以解释：

```text
到底是Tree、风险门、候选池、V2、V2C、clip还是blend导致结果
```

#### 9.4.5 论文叙事像补丁

如果最终提升很小，而模块很多，审稿人可能认为：

```text
核心方法不清晰
经验门控较多
后处理堆叠
```

### 9.5 当前状态

V2C保留为：

```text
历史安全策略
+
候选来源诊断中的归因环节
```

当前不继续增加新安全层。

---

# 第七部分：历史方案的总体优点

## 10. Tree + V2B + V2C总体优点

### 10.1 安全性

Tree是默认结果，候选只有在满足多项条件时才修改。

### 10.2 不依赖端点

全链路围绕区域和方向，不预测人工端点。

### 10.3 可追溯

每例可记录：

- Tree轴；
- 风险分数；
- 候选集合；
- 候选预测分数；
- 选择结果；
- V2C接受或拒绝；
- clip/blend后的最终delta；
- card和full误差变化。

### 10.4 数据划分相对规范

历史设计强调：

- 回顾性五折OOF；
- 外部队列最终评估；
- 人工轴只作为监督和评估；
- ref轴保持固定。

### 10.5 适合失败机制研究

该链路暴露了多个可分析环节，有助于判断失败发生在：

```text
候选生成
风险检测
候选选择
安全拒绝
幅度稀释
```

---

# 第八部分：历史方案的总体缺陷

## 11. Tree + V2B + V2C总体缺陷

### 11.1 架构逐渐补丁化

整体变成：

```text
固定几何不够
→ Tree残差修正
→ 风险模型
→ 高风险候选选择
→ ranking
→ margin
→ harm control
→ consensus
→ clip
→ blend
```

继续增加模块会进一步削弱核心叙事。

### 11.2 主要瓶颈可能在前面

若id4/id5方向、OBB/PCA候选或加权融合本身有问题，后面的选择器和安全层只能有限补救。

### 11.3 风险门与候选收益不是同一任务

高Tree误差预测不能替代“是否存在可改善候选”的判断。

### 11.4 Oracle低不代表任务可学

低oracle可能来自：

- 大量近重复候选；
- 很小的真实误差差距；
- 每例事后选最优；
- 少数危险候选的偶然救援。

必须同时分析：

```text
distinct axes
true margin
visual distinguishability
source consistency
```

### 11.5 只利用少量理论空间

历史记录显示V2/V2C相对Tree改善很小，而oracle空间较大。

该差距可能来自：

- 风险门漏检；
- 选择错误；
- 候选模糊；
- V2C拒绝；
- clip/blend稀释；
- 候选定义混乱。

### 11.6 Ref不是主要瓶颈

继续修ref会增加复杂度，却不一定解决当前card问题。

---

# 第九部分：历史结论状态

## 12. 仍可保留的结论

以下结论目前仍可作为工作假设：

1. ref轴当前不是主要瓶颈；
2. card轴需要重点分析；
3. Tree应保留为基础基线；
4. 候选池中在部分病例存在比Tree更好的方向；
5. 高风险fallback可能漏掉中等误差但可改善病例；
6. V2C更像安全决策层，而不是新的主估计器；
7. 方法已经出现补丁化趋势；
8. 下一步应回退诊断候选来源和id4/id5融合。

## 13. 暂停使用的结论

以下历史结论不得直接用于新正式方案：

1. `Original16 oracle ≈ 1.03°`作为正式性能结论；
2. `Core7`是最佳候选集合；
3. 某候选应根据平均MAE自动删除；
4. feat版本与mask版本一定互补或一定冗余；
5. 候选之间接近90°必然来自主轴/短轴错误；
6. removal regret可直接决定候选去留；
7. 旧盲评结果可直接决定M1/M2；
8. 外部队列上的候选表现可用于选择候选；
9. 旧目录结果可直接写入新论文。

原因：

- 新正式clean manifest尚需重建；
- 候选生成源码未完全追溯；
- 部分自动删除分析存在累计删除或代表轴不一致问题；
- 历史文件可能混用不同版本。

## 14. 已纠正的错误理解

### 14.1 不需要跨图统一坐标

人工线与mask之间未知裁剪关系，不做端点坐标映射，只比较方向角。

### 14.2 Card候选使用固定manual GT

人工两条线先由gold方向完成card/ref一一语义分配。

随后所有card候选统一与固定：

```text
manual_card_axis
```

比较。

不是每个候选重新在两条人工线中选最近线。

### 14.3 Card误差不等于Full误差

```text
card MAE
```

是card方向误差。

```text
full MAE
```

才是最终心轴夹角误差。

### 14.4 “候选有问题”不是单一结论

候选问题至少包括：

- pool没有正确方向；
- 候选重复；
- 候选语义错误；
- 候选不可辨识；
- 融合规则错误；
- 分割导致候选偏移。

同时还可能存在：

- 风险门漏检；
- V2选择错误；
- V2C拒绝；
- clip/blend稀释。

---

# 第十部分：当前工作主线

## 15. 当前正式方向

暂不继续：

```text
V2D
V2E
新的harm gate
新的clip/blend组合
```

当前主线为：

```text
恢复并冻结数据读取
→ 追溯候选定义
→ 进行card候选来源诊断
→ 判断问题位于分割、几何候选、融合、选择还是门控
→ 再决定新主方法
```

## 16. Card候选来源诊断目标

需要把每例失败分成：

### A. 候选池缺失

Tree不好，所有非Tree候选也不好。

### B. 候选重复/模糊

有多个候选名，但真实方向重复，或最佳与第二方向真实误差差很小。

### C. 风险门漏检

存在清晰好候选，但病例未进入V2选择。

### D. V2B选择错误

进入风险门且有清晰好候选，但排序器没有选中。

### E. V2C拒绝正确候选

V2已选中真实更优候选，但V2C保留Tree。

### F. Clip/Blend稀释

选择和门控正确，但最终应用幅度过小。

### G. 成功救援

Tree错误被候选链真正修复。

---

# 第十一部分：下一步方法决策规则

## 17. 若Gold候选好、Infer候选差

说明主要问题可能在：

```text
分割质量
连通域
边界缺损
mask形态稳定性
```

应优先提升候选对分割误差的鲁棒性或分割本身。

## 18. 若Gold与Infer候选都差

说明OBB/PCA/联合区域等候选定义无法表达人工card方向。

应考虑：

```text
重新设计区域到方向映射
或
region-to-axis直接学习
```

## 19. 若id4或id5单区域经常优于加权Tree

说明问题可能位于：

```text
固定融合规则
```

应替换融合层，而不是继续在Tree后面加安全门。

## 20. 若候选池好但风险门漏检多

应取消：

```text
仅高风险病例进入候选选择
```

考虑：

```text
所有病例统一估计/排序
Tree作为候选之一
```

## 21. 若候选清晰但V2选错多

说明候选生成可以保留，需要重做：

```text
候选表示
训练目标
图像区域
排序模型
```

## 22. 若V2选对但V2C拒绝多

说明安全层过度保守，应简化决策，不应继续叠加安全模块。

## 23. 若候选人眼也难以区分

说明候选选择任务缺乏可学习证据，应停止复杂ranker路线。

---

# 第十二部分：与论文叙事的关系

## 24. 不建议的最终叙事

不建议将最终方法写成：

```text
固定几何
+
残差Tree
+
风险检测
+
候选CNN
+
utility
+
harm control
+
clip/blend
```

原因：

- 核心贡献不清晰；
- 模块过多；
- 容易被认为是经验补丁；
- 性能提升与复杂度不匹配；
- 消融难解释。

## 25. 更理想的未来叙事

最终方法应尽量收敛到一个核心机制，例如：

### 方向A：学习式区域融合

```text
id4/id5区域与形态
→ 学习每个区域可靠性
→ 统一card轴
```

### 方向B：区域到方向直接估计

```text
分割区域/超声ROI
→ 无方向card轴表示
```

输出可使用：

```text
sin(2θ), cos(2θ)
```

避免180°周期问题。

### 方向C：少量、语义清晰的候选统一选择

```text
Tree + id4 + id5 +少量明确候选
→ 全病例统一选择
```

前提是候选来源诊断证明任务可辨识。

---

# 第十三部分：文件与版本管理

## 26. 每个方法脚本顶部必须声明

```text
Data reading spec:
CARDIAC_AXIS_DATA_READING_SPEC_v1.0

Method history:
CARDIAC_AXIS_METHOD_HISTORY_v1.0

Method stage:
例如 fixed_geometry / tree / candidate_diagnosis / new_method

Historical inputs used:
明确列出

External cohorts used for selection:
False
```

## 27. 每个结果文件必须记录

```text
data_reading_spec_version
method_history_version
method_name
method_version
input_manifest
input_fold
input_baseline
generator_script
generator_script_hash
generated_at
```

## 28. 旧结果状态标签

建议所有历史结果增加：

```text
result_status
```

允许值：

```text
formal
historical
diagnostic_only
invalidated
pending_revalidation
```

当前Tree/V2/V2C旧目录结果建议标为：

```text
historical / pending_revalidation
```

---

# 第十四部分：当前方法状态表

| 方法阶段 | 作用 | 优点 | 主要缺陷 | 当前状态 |
|---|---|---|---|---|
| Fixed Geometry v2 | 从mask生成card/ref | 简单、可解释 | 固定OBB与融合可能失真 | 保留基线 |
| Dual-axis Tree | 表格残差修正 | 灵活、OOF规范 | 仍受前置候选限制 | 历史基线 |
| Risk/Confidence | 识别Tree失败 | 安全、可解释 | 漏掉可改善病例 | 诊断用途 |
| Candidate Pool | 提供多个card方向 | 有理论空间 | 重复、语义和覆盖问题 | 重点诊断 |
| V2B/V2 Rank-strip | 候选视觉排序 | 利用图像与mask | 风险门、可辨识性、clip | 历史方法 |
| V2C Utility/Harm | 控制修改风险 | 安全性强 | 过度保守、补丁化 | 暂停扩展 |
| Card Source Diagnosis | 定位候选来源问题 | 能回退到真正瓶颈 | 尚未正式执行 | 当前主线 |

---

# 第十五部分：变更记录

## v1.0 — 2026-08-04

首次建立方法历史文档，记录：

- 固定几何基线；
- Tree双轴残差修正；
- 风险/置信度；
- card候选池；
- V2B/V2 rank-strip；
- V2C utility/harm control；
- 各阶段输入输出、优点与缺陷；
- 历史结果状态；
- 已纠正的评估理解；
- 已暂停的候选删除与oracle结论；
- 当前card候选来源诊断主线；
- 未来方法决策规则。

后续新增方法必须：

1. 在本文件增加新章节或新版本；
2. 明确其相对前一阶段解决的问题；
3. 记录输入输出；
4. 记录是否使用外部数据；
5. 记录接受、暂停或废弃原因；
6. 不得删除旧历史，只能追加状态更新。
