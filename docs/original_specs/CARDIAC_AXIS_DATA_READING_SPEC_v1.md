# CARDIAC_AXIS_DATA_READING_SPEC_v1

> **文档状态：** Active  
> **规范版本：** 1.0  
> **适用项目：** 早孕期胎儿心轴角自动测量  
> **创建日期：** 2026-08-04  
> **建议保存位置：** `/data5/xjx/zaoyun/data/j/docs/CARDIAC_AXIS_DATA_READING_SPEC_v1.md`  
> **规范目标：** 固定病例读取、人工测量线读取、分割标签读取、card/ref语义分配、角度定义、数据划分与下游使用规则，防止不同实验脚本或不同对话窗口重新解释数据。

---

## 0. 文档定位

本文件是胎儿心轴角项目的**唯一权威数据读取规范**。它只定义：

1. 正式样本和队列；
2. 病例唯一键与manifest读取；
3. 人工心轴标注tar/JSON的读取；
4. `label_gold`与`label_infer`的读取；
5. 人工两条线的card/ref语义分配；
6. Tree固定几何基线中的card轴与ref轴定义；
7. 方向角、最终心轴角及误差定义；
8. 数据划分、外部测试边界；
9. 下游脚本必须执行的校验；
10. 禁止重新解释或静默修改的规则。

本文件**不定义**具体候选池、V2B、V2C、候选CNN、风险门、效用模型或未来方法。这些内容应记录在独立的方法文档中。

---

## 1. 规范优先级

当不同来源发生冲突时，优先级如下：

1. **本规范当前有效版本；**
2. 经人工确认且已写入本规范的事实；
3. 当前正式公共读取模块；
4. 直接生成正式基线文件的源码；
5. 历史实验脚本；
6. 历史结果CSV/Excel中的字段名称；
7. 根据文件名或候选名作出的推断。

任何下游脚本不得仅凭字段名、候选名或旧注释重新定义读取含义。

---

## 2. 当前已确认的逻辑来源

本规范的核心读取逻辑来自已确认的固定几何基线生成脚本。该脚本：

- 读取：
  - `clean_dataset_manifest_v1/clean_manifest.csv`
  - `clean_dataset_manifest_v1/clean_5fold_split.csv`
- 生成：
  - `fixed_geometry_baseline_v2/fixed_geometry_baseline_v2_case_detail.csv`
  - `fixed_geometry_baseline_v2/fixed_geometry_baseline_v2_results.xlsx`
- 从`axis_label_path`指向的tar包中读取JSON和`MeasureModel`；
- 从`Pos.p1`、`Pos.p2`读取两条人工测量线；
- 使用`label_gold`生成的card/ref方向，对人工两条线进行方向角域内的一一语义分配；
- 使用`label_infer`生成实际预测card/ref方向。

### 2.1 路径边界

历史逻辑曾位于：

```text
/data5/xjx/zaoyun/data/region_axis_process_A_infer
```

当前正式数据整理目标位于：

```text
/data5/xjx/zaoyun/data/j
```

历史目录只可作为逻辑追溯来源。新正式实验不得在未注明来源的情况下混用旧目录中的manifest、模型输出或中间CSV。

---

## 3. 正式样本基准

| 队列 | 类别 | 数量 |
|---|---|---:|
| retrospective | Ne | 571 |
| retrospective | Po | 50 |
| multicenter | Ne | 434 |
| multicenter | Po | 8 |
| prospective | Ne | 400 |
| prospective | Po | 11 |
| **总计** | **Ne + Po** | **1474** |

队列总数：

| 队列 | 总数 |
|---|---:|
| retrospective | 621 |
| multicenter | 442 |
| prospective | 411 |
| **总计** | **1474** |

### 3.1 数据角色

- `retrospective`：开发集，采用患者级五折交叉验证和OOF评估。
- `multicenter`：外部测试集。
- `prospective`：前瞻测试集。
- 外部队列不得用于：
  - 候选删除；
  - 阈值设定；
  - 风险门选择；
  - 模型选择；
  - 超参数选择；
  - 方法接受/拒绝判断。

### 3.2 有效样本数

每个指标必须单独报告`N_valid`。不得因为总队列为1474例，就默认card、ref、full三个指标均有1474个有效值。

---

## 4. 病例唯一键

正式病例唯一键定义为：

```text
uid = case_id + "|" + class + "|" + cohort
```

示例：

```text
241097500001|Ne|multicenter
```

### 4.1 唯一性要求

以下组合必须唯一：

```text
(case_id, class, cohort)
```

以及：

```text
uid
```

### 4.2 禁止静默去重

禁止直接使用：

```python
drop_duplicates(..., keep="first")
```

来掩盖重复病例或fold冲突。

若发现重复：

1. 若所有关键字段完全相同，记录为重复输入并停止，由人工确认是否去重；
2. 若路径、fold、类别、队列或角度字段不同，定义为版本冲突，必须停止；
3. 不得自动保留第一条或最后一条。

---

## 5. Manifest读取规范

### 5.1 权威输入

正式读取必须从明确指定的clean manifest开始，不得在实验脚本中重新遍历目录并按文件名猜测病例关系。

必需字段至少包括：

```text
case_id
class
cohort
ready_for_axis_measurement
axis_label_path
label_gold_path
label_infer_path
```

若需要视觉输入，还应包括：

```text
img_path
```

### 5.2 Ready筛选

仅保留：

```text
ready_for_axis_measurement == True
```

正式manifest中该字段必须是标准布尔值。若原文件中存在字符串形式：

```text
"True", "true", "1", "yes"
```

必须先统一转换，并输出转换统计；不得让字符串与布尔值混杂。

### 5.3 路径检查

每例必须检查：

- `axis_label_path`存在；
- `label_gold_path`存在；
- `label_infer_path`存在；
- 需要时`img_path`存在；
- 路径不是空字符串、`None`或`NaN`；
- 同一UID对应的路径唯一。

路径缺失不得静默回退到其他目录或模糊匹配同名文件。

---

## 6. Fold读取规范

### 6.1 Fold文件

回顾性五折由明确的fold文件提供。合并键为：

```text
case_id + class + cohort
```

### 6.2 Fold约束

- 回顾性病例必须具有合法fold；
- 合法fold集合应明确固定，例如`0,1,2,3,4`；
- 多中心与前瞻病例不得参与fold训练；
- 同一病例不得出现在多个fold；
- fold表重复时必须先报错，不能静默保留第一条。

### 6.3 合并校验

manifest与fold合并后必须检查：

- 回顾性621例均成功匹配fold；
- 每个回顾性UID只匹配一行；
- 外部病例不会被错误赋予训练角色；
- `split_role`若存在，应与cohort和fold逻辑一致。

---

## 7. 人工心轴标注读取

### 7.1 数据来源

人工心轴标注由：

```text
axis_label_path
```

指向的tar包提供。

人工测量线不是通过彩色截图OCR或图像Hough检测获得，而是从tar内部JSON结构中直接读取。

### 7.2 tar成员读取

tar可使用：

```python
tarfile.open(path, "r:*")
```

打开。

只解析可判定为JSON内容的成员。每个成功解析的JSON对象必须记录其tar成员名，便于追溯。

建议输出：

```text
axis_tar_member_name
axis_json_encoding
axis_json_object_index
```

### 7.3 JSON编码

允许依次尝试：

```text
utf-8
gbk
cp936
```

若无任何JSON可解析，病例读取失败。

### 7.4 MeasureModel递归查找

在JSON对象中递归查找键：

```text
MeasureModel
```

且其值应为列表。

每个列表元素若含有：

```text
Pos.p1
Pos.p2
```

则视为一条候选人工测量线。

### 7.5 人工线数量

项目已确认：每个tar中应有且只有两条有效人工测量线。

正式规则：

```text
valid_manual_line_count == 2
```

- 少于2条：失败；
- 多于2条：失败；
- 不允许使用`lines[:2]`静默截断。

### 7.6 端点有效性

每条人工线的`p1`和`p2`必须满足：

- 均至少包含两个数值；
- 坐标为有限数；
- `p1 != p2`；
- 线段长度大于最小容差；
- 不允许`NaN`或无穷值。

建议保存：

```text
manual_line0_p1_x
manual_line0_p1_y
manual_line0_p2_x
manual_line0_p2_y
manual_line0_length

manual_line1_p1_x
manual_line1_p1_y
manual_line1_p2_x
manual_line1_p2_y
manual_line1_length
```

---

## 8. 人工线方向角

### 8.1 单条线方向

对端点：

```text
p1 = (x1, y1)
p2 = (x2, y2)
```

方向角定义为：

```text
theta = degrees(atan2(y2 - y1, x2 - x1))
```

归一化：

```text
theta_180 = theta mod 180
```

最终：

```text
theta_180 ∈ [0°, 180°)
```

### 8.2 无方向轴

人工测量线被视为无方向直线：

- 交换`p1`和`p2`不得改变结果；
- 相差180°表示同一条轴；
- 不使用0°—360°的有向角误差。

### 8.3 禁止跨图坐标匹配

人工标注图、原图和分割标签之间可能存在未知裁剪、显示视野变化或其他不可恢复处理。

因此禁止：

- 将人工端点缩放到mask坐标；
- 根据图像宽高强行恢复坐标；
- 使用端点距离匹配；
- 使用线中心距离匹配；
- 使用线段重叠匹配；
- 将人工线空间叠加结果当作定量证据。

不同来源之间只比较方向角。

---

## 9. Mask读取规范

### 9.1 输入

- `label_gold_path`：人工/金标准分割标签；
- `label_infer_path`：模型推理分割标签。

### 9.2 类别ID

| ID | 用途 |
|---:|---|
| 1 | reference-center相关区域 |
| 4 | cardiac区域之一 |
| 5 | cardiac区域之一 |
| 6 | body/thorax参考区域 |

类别ID不得在实验脚本中重新定义。

### 9.3 二维mask

若：

```text
mask.ndim == 2
```

直接作为标签图读取。

### 9.4 三通道mask

若：

```text
mask.ndim == 3
```

仅可在确认其本质仍是“单通道标签重复保存”时选择有效通道。

历史选择逻辑为：

1. 统计各通道中`id4 + id5`像素数；
2. 统计`id1 + id4 + id5 + id6`像素数；
3. 在得分相同情况下倾向唯一值较少的通道。

正式实现必须增加硬校验：

- 选中通道必须包含至少一个预期ID；
- 输出选中通道索引；
- 输出各预期ID像素数；
- 若所有通道均不含预期ID，停止；
- 不允许把RGB彩色可视化图误当整数标签图。

建议保存：

```text
mask_ndim
mask_selected_channel
mask_unique_values
id1_pixel_count
id4_pixel_count
id5_pixel_count
id6_pixel_count
```

### 9.5 Mask尺寸

card/ref方向均在各自mask坐标系内计算。

本规范不要求将`label_gold`、`label_infer`或人工线端点映射到同一空间坐标系。

### 9.6 插值限制

若仅为视觉显示而调整超声图到mask尺寸：

- 图像可使用线性插值；
- 标签图只能使用最近邻插值。

视觉resize不得改变方向角GT或用于人工线空间匹配。

---

## 10. 最大连通域规则

固定几何基线对以下类别均使用最大连通域：

```text
id1
id4
id5
id6
```

流程：

1. 对指定类别生成二值mask；
2. 使用8连通域；
3. 排除背景；
4. 选择面积最大的前景连通域；
5. 后续面积、质心、OBB均基于该最大连通域。

若无前景连通域：

- 对应区域记为缺失；
- 不允许自动使用其他类别替代；
- 不允许自动使用全部非零区域替代。

若未来候选使用“全部同类像素”而非最大连通域，必须在候选定义中显式声明，不能默认继承Tree逻辑。

---

## 11. Card轴固定几何定义

### 11.1 id4单区域方向

对id4最大连通域：

1. 提取外轮廓；
2. 使用`cv2.minAreaRect`计算OBB；
3. 使用`cv2.boxPoints`得到四个角点；
4. 计算四条边长度和方向；
5. 选择最长边对应方向；
6. 归一化到`[0°,180°)`。

输出至少包括：

```text
id4_angle
id4_area
id4_aspect
id4_weight
```

### 11.2 id5单区域方向

id5与id4采用相同流程。

### 11.3 OBB有效性

以下情况视为该区域方向无效：

- 无连通域；
- 无轮廓；
- 轮廓点少于3；
- OBB宽或高接近0；
- 计算结果为非有限数。

### 11.4 形状权重

每个区域权重定义为：

```text
weight = area × max(aspect - 1.0, 0.05)
```

其中：

```text
aspect = long_side / short_side
```

### 11.5 无方向加权平均

对有效区域方向`theta_i`和权重`w_i`，使用双角度表示：

```text
x = Σ w_i cos(2 theta_i)
y = Σ w_i sin(2 theta_i)
theta_card = 0.5 × atan2(y, x)
```

再归一化到：

```text
[0°,180°)
```

### 11.6 区域缺失

- id4、id5均有效：按上述权重融合；
- 仅一个有效：card轴使用该单一区域方向；
- 两者均无效：card轴失败。

---

## 12. Ref轴固定几何定义

### 12.1 区域

- id6：取最大连通域质心；
- id1：取最大连通域质心。

### 12.2 方向

ref轴定义为从id6质心到id1质心的无方向直线：

```text
theta_ref = angle(id6_centroid, id1_centroid) mod 180
```

由于最终按无方向轴处理，端点先后不影响误差。

### 12.3 缺失

若id1或id6任一缺失：

```text
ref_axis = invalid
```

不得用card中心、图像中心或其他类别自动代替。

---

## 13. 人工两条线的card/ref语义分配

人工tar中两条线的读取顺序不保证card/ref语义。

### 13.1 Gold参考方向

分别从`label_gold`生成：

```text
gold_card_axis
gold_ref_axis
```

### 13.2 两种一一排列

设人工线方向为：

```text
a0 = manual_line0_angle
a1 = manual_line1_angle
```

计算：

```text
cost01 =
    axial_abs_diff(a0, gold_card_axis)
  + axial_abs_diff(a1, gold_ref_axis)
```

以及：

```text
cost10 =
    axial_abs_diff(a1, gold_card_axis)
  + axial_abs_diff(a0, gold_ref_axis)
```

### 13.3 分配规则

若：

```text
cost01 <= cost10
```

则：

```text
manual_card_axis = a0
manual_ref_axis  = a1
manual_assignment = "line0_card_line1_ref"
```

否则：

```text
manual_card_axis = a1
manual_ref_axis  = a0
manual_assignment = "line1_card_line0_ref"
```

### 13.4 分配margin

定义：

```text
manual_assignment_margin = |cost10 - cost01|
```

margin越小，说明两种语义分配越接近，应进入审计清单。

### 13.5 固定GT原则

语义分配完成后：

```text
manual_card_axis
manual_ref_axis
```

成为该病例固定的单轴GT。

所有后续card候选统一与：

```text
manual_card_axis
```

比较。

所有后续ref结果统一与：

```text
manual_ref_axis
```

比较。

禁止为每个候选重新在两条人工线中选择最近方向。否则不同候选将使用不同GT，候选误差不再可比。

---

## 14. Gold语义分配失败

若`label_gold`无法同时生成有效的：

```text
gold_card_axis
gold_ref_axis
```

则：

```text
manual_card_axis = NaN
manual_ref_axis = NaN
manual_assignment = "unknown_gold_axis_failed"
```

此时：

- card单轴误差不可计算；
- ref单轴误差不可计算；
- 人工两条线之间的`manual_full_angle`仍可计算；
- infer full angle仍可与manual full angle比较。

必须分别报告card/ref/full的有效N。

---

## 15. 角度与误差定义

### 15.1 方向归一化

```text
normalize_angle180(a) = a mod 180
```

结果范围：

```text
[0°,180°)
```

### 15.2 有符号无方向差

```text
signed_axial_diff(target, source)
=
(target - source + 90) mod 180 - 90
```

范围：

```text
[-90°,90°)
```

### 15.3 无方向绝对误差

```text
axial_abs_error(pred, gt)
=
abs((pred - gt + 90) mod 180 - 90)
```

范围：

```text
[0°,90°]
```

### 15.4 两轴锐角

```text
acute_angle_between_lines(a, b)
```

定义为两条无方向轴之间的最小夹角，范围：

```text
[0°,90°]
```

### 15.5 Card方向角

```text
manual_card_axis
pred_card_axis
card_error = axial_abs_error(pred_card_axis, manual_card_axis)
```

这是card线自身方向误差，不是最终临床心轴角误差。

### 15.6 Ref方向角

```text
manual_ref_axis
pred_ref_axis
ref_error = axial_abs_error(pred_ref_axis, manual_ref_axis)
```

### 15.7 最终Full心轴角

人工：

```text
manual_full_angle
=
acute_angle_between_lines(manual_card_axis, manual_ref_axis)
```

等价地，由人工原始两条线直接计算，交换语义不影响结果。

预测：

```text
pred_full_angle
=
acute_angle_between_lines(pred_card_axis, pred_ref_axis)
```

误差：

```text
full_error
=
abs(pred_full_angle - manual_full_angle)
```

### 15.8 指标命名约束

禁止把：

```text
card_MAE
```

写成“最终心轴角MAE”。

正式论文主结果应使用：

```text
full_MAE
```

card/ref用于误差归因。

---

## 16. 单病例标准处理流程

每例必须按以下顺序处理：

```text
1. 从manifest读取病例和路径
2. 校验uid唯一
3. 读取axis_label tar
4. 解析JSON
5. 找到恰好两条有效MeasureModel线
6. 读取两条线端点并计算manual_line0/1_angle
7. 读取label_gold
8. 读取label_infer
9. 从label_gold生成gold_card_axis和gold_ref_axis
10. 用gold方向为人工两条线分配card/ref语义
11. 得到固定manual_card_axis和manual_ref_axis
12. 计算manual_full_angle
13. 从label_infer生成infer_card_axis和infer_ref_axis
14. 计算infer_full_angle
15. 分别计算card/ref/full误差
16. 保存所有来源、状态和审计字段
```

---

## 17. 固定几何基线输出字段

每个病例至少保存以下字段。

### 17.1 身份与划分

```text
uid
case_id
class
cohort
fold
split_role
```

### 17.2 路径

```text
img_path
axis_label_path
label_gold_path
label_infer_path
```

### 17.3 人工原始线

```text
manual_line_count
manual_line0_angle
manual_line1_angle
manual_line0_length
manual_line1_length
```

建议同时保存端点和tar成员来源。

### 17.4 人工语义

```text
manual_card_axis
manual_ref_axis
manual_full_angle
manual_assignment
manual_assignment_margin
gold_available_for_semantic_assignment
```

### 17.5 Gold几何

```text
gold_card_axis
gold_ref_axis
gold_full_angle
gold_card_error
gold_ref_error
gold_full_error
```

### 17.6 Infer几何

```text
infer_card_axis
infer_ref_axis
infer_full_angle
infer_card_error
infer_ref_error
infer_full_error
```

### 17.7 区域信息

```text
gold_id1_area
gold_id6_area
infer_id1_area
infer_id6_area

gold_card_id4_angle
gold_card_id5_angle
infer_card_id4_angle
infer_card_id5_angle

gold_card_id4_area
gold_card_id5_area
infer_card_id4_area
infer_card_id5_area

gold_card_id4_aspect
gold_card_id5_aspect
infer_card_id4_aspect
infer_card_id5_aspect
```

### 17.8 状态

```text
ok
fail_reason
traceback
```

---

## 18. 下游脚本数据合同

任何下游脚本读取固定几何基线时，必须：

1. 明确指定输入文件；
2. 检查本规范版本；
3. 检查UID唯一；
4. 检查人数；
5. 检查manual字段完整性；
6. 检查角度范围；
7. 检查Tree字段与基线一致；
8. 不重新定义manual card/ref；
9. 不覆盖已有正确误差而不留痕；
10. 输出来源文件和版本。

### 18.1 Card候选误差

任何card候选：

```text
candidate_card_error
=
axial_abs_error(candidate_axis, manual_card_axis)
```

前提是`manual_card_axis`有效。

### 18.2 Ref固定原则

当前研究主攻card轴。若方法声明ref轴不修改，则：

```text
refined_ref_axis = baseline_ref_axis
```

并必须验证ref字段没有被意外覆盖。

### 18.3 Full重算

任何card轴发生变化后，必须使用同一病例固定ref轴重新计算：

```text
refined_full_angle
=
acute_angle_between_lines(refined_card_axis, fixed_ref_axis)
```

不得直接对旧full angle加减card delta。

---

## 19. 强制自动校验

每次正式运行前，至少执行以下断言。

### 19.1 样本校验

- retrospective Ne = 571
- retrospective Po = 50
- multicenter Ne = 434
- multicenter Po = 8
- prospective Ne = 400
- prospective Po = 11
- total = 1474

### 19.2 唯一性校验

- `uid`唯一；
- `(case_id,class,cohort)`唯一；
- fold表无冲突；
- 下游候选表中`(uid,candidate_name)`按设计唯一。

### 19.3 人工标注校验

- tar存在；
- JSON可解析；
- 有效人工线数恰好为2；
- 端点有限；
- 线长非零；
- 两条角度均在`[0,180)`。

### 19.4 Mask校验

- mask可读取；
- 维度为2或3；
- 选中通道有效；
- 唯一值可解释；
- 预期ID统计已输出；
- 不能全部关键ID均为0却继续。

### 19.5 角度校验

- 所有轴角在`[0,180)`；
- 所有full角在`[0,90]`；
- 交换人工线端点不改变方向；
- 交换card/ref顺序不改变full角；
- 保存的full角与card/ref重算一致；
- 保存的误差与公式重算一致。

### 19.6 队列边界校验

- 回顾性五折完整；
- 外部队列未进入训练；
- 外部结果未用于选参；
- OOF结果来自对应fold未见过的模型。

---

## 20. 禁止事项

以下行为明确禁止：

1. 不知道裁剪关系时强行统一人工线与mask坐标；
2. 使用端点或中心距离跨图匹配；
3. 固定按人工线读取顺序定义card/ref；
4. 对每个候选重新选择不同人工GT；
5. 将RGB可视化mask当整数标签；
6. 三通道mask全部无效时仍选择一个通道；
7. 用`drop_duplicates(keep="first")`掩盖版本冲突；
8. 缺id4/id5时静默使用其他类别；
9. 直接把card delta加到full angle；
10. 将card MAE表述为最终full心轴角MAE；
11. 使用外部队列选择方案；
12. 混用旧目录和新正式目录而不记录；
13. 仅凭候选名推断其生成逻辑；
14. 下游脚本自行复制并修改读取规则；
15. 发现读取异常后继续训练并只在日志中警告。

---

## 21. 已知限制

### 21.1 Gold辅助语义分配

人工card/ref语义由`label_gold`产生的card/ref轴辅助分配。因此：

- `manual_full_angle`不依赖card/ref交换；
- 单轴card/ref GT依赖gold几何能够正确区分两条人工线；
- `manual_assignment_margin`较小的病例需要专项复核。

### 21.2 Mask通道启发式

历史脚本使用预期ID像素数选择三通道mask中的有效通道。正式版本必须在此基础上增加硬校验，避免错误图像被接受。

### 21.3 固定几何不是最终方法

本规范固定的是读取和基线定义，不代表id4/id5 OBB加权一定是最终最优card轴方法。未来可更换card方法，但不能更换人工GT、mask含义、角度定义和数据划分口径。

---

## 22. 版本追踪

每次输出应记录：

```text
data_reading_spec_version
manifest_path
manifest_hash
fold_path
fold_hash
baseline_input_path
baseline_input_hash
generator_script_name
generator_script_hash
generated_at
git_commit_or_code_version
```

推荐：

```text
data_reading_spec_version = "CARDIAC_AXIS_DATA_READING_SPEC_v1.0"
```

### 22.1 版本指纹字段

验证某个旧CSV是否由当前固定几何逻辑生成时，重点比较：

```text
manual_line0_angle
manual_line1_angle
manual_assignment
manual_assignment_margin
manual_card_axis
manual_ref_axis
manual_full_angle
gold_card_id4_angle
gold_card_id5_angle
infer_card_id4_angle
infer_card_id5_angle
```

只看文件名相同不能证明版本相同。

---

## 23. 公共读取模块要求

正式项目建议建立统一公共模块：

```text
scripts/common/
├── data_contract.py
├── manifest_loader.py
├── manual_axis_loader.py
├── mask_loader.py
├── geometry_utils.py
├── angle_utils.py
└── validation.py
```

所有实验脚本调用同一公共模块，不得复制后自行修改。

每个实验脚本顶部只声明：

```text
Data reading spec:
CARDIAC_AXIS_DATA_READING_SPEC_v1.0
```

并说明本脚本新增的方法逻辑，不重复定义基础读取规则。

---

## 24. 建议的读取审计输出

在任何正式训练之前，先生成只读审计结果，至少包括：

### Sheet / CSV 1：Count

- 各队列Ne/Po数量；
- 总数；
- 与预期差值。

### Sheet / CSV 2：UID Audit

- 重复UID；
- 路径冲突；
- fold冲突。

### Sheet / CSV 3：Manual Line Audit

- tar成员；
- JSON数量；
- MeasureModel数量；
- 两条端点；
- 两条线长；
- 两条方向。

### Sheet / CSV 4：Mask Audit

- mask维度；
- 选中通道；
- 唯一值；
- 各类别像素数；
- 各类别最大连通域面积。

### Sheet / CSV 5：Semantic Assignment

- gold card/ref；
- 两种assignment cost；
- 最终assignment；
- assignment margin。

### Sheet / CSV 6：Geometry Audit

- gold/infer id4、id5方向；
- card轴；
- ref轴；
- full角；
- 三类误差。

### Sheet / CSV 7：Failure Cases

- 所有失败病例；
- 明确失败原因；
- 不允许只保存traceback而无分类。

---

## 25. 正式运行验收清单

在任何Tree、候选诊断或模型训练开始前，必须逐项确认：

- [ ] 使用的规范版本为`CARDIAC_AXIS_DATA_READING_SPEC_v1.0`
- [ ] 正式样本总数为1474
- [ ] 六个cohort/class数量完全匹配
- [ ] UID无重复
- [ ] fold无重复或冲突
- [ ] 所有必需路径存在
- [ ] 每个tar恰好两条有效人工线
- [ ] 人工线端点和长度有效
- [ ] mask编码合法
- [ ] id1/id4/id5/id6统计已输出
- [ ] manual card/ref由gold方向一一分配
- [ ] manual assignment margin已保存
- [ ] card/ref/full角度范围合法
- [ ] 下游脚本未重新定义manual GT
- [ ] 外部数据未参与选参
- [ ] 输出中保存了规范版本和文件hash

任何一项未通过，正式实验不得继续。

---

## 26. 核心原则摘要

1. **人工tar内恰好两条测量线。**
2. **人工线只在自身坐标系中计算无方向角。**
3. **不做人工线与mask之间的坐标映射。**
4. **使用label_gold的card/ref方向，为人工两条线进行一一语义分配。**
5. **分配后得到固定的manual_card_axis与manual_ref_axis。**
6. **所有card候选统一与固定manual_card_axis比较。**
7. **Tree card轴由id4/id5最大连通域OBB长轴加权得到。**
8. **Tree ref轴由id6与id1最大连通域质心连线得到。**
9. **card/ref是单轴方向误差，full才是最终临床心轴角误差。**
10. **回顾性用于开发与OOF，外部队列只用于最终测试。**
11. **禁止静默去重、静默回退和跨版本混用。**
12. **所有新方法必须引用本规范，而不是重新解释数据。**

---

## 27. 变更记录

### v1.0 — 2026-08-04

首次建立权威读取规范，固定：

- 正式样本基准；
- manifest与fold读取；
- tar/JSON/MeasureModel人工线读取；
- 两条人工线方向角；
- label_gold辅助card/ref语义分配；
- label_gold/label_infer mask读取；
- id4/id5 OBB加权card轴；
- id6-id1质心连线ref轴；
- card/ref/full角度和误差；
- 下游数据合同；
- 禁止跨图坐标映射；
- 强制校验和版本追踪规则。

后续任何修改必须：

1. 新增版本号；
2. 在CHANGELOG中记录；
3. 明确是否需要重算历史结果；
4. 不得直接覆盖本版本后不留记录。
