# 02｜人工GT与心轴角计算规范（GT PROTOCOL）

**文档版本：** v1.0摘要  
**整理日期：** 2026-11-23  
**权威来源：** `CARDIAC_AXIS_DATA_READING_SPEC_v1.md` v1.0  
**状态：** 按现有权威规范整理；不重新定义GT

## 1. 最重要的原则

早孕期人工测量图像与分割图像属于同一病例，但经过不同裁剪、缩放或其他预处理，当前无法可靠恢复像素级变换。

因此，本项目采用：

**病例级关联 + 方向角域比较，不做跨图像像素坐标映射。**

明确禁止： - 把TAR人工线端点缩放到分割Mask坐标； -
根据两张图像宽高猜测变换； -
用端点距离、线中心距离或线段重叠进行跨图匹配； -
将空间叠加可视化作为跨图配准证据。

## 2. GT数据的三个概念

### 2.1 人工测量线（来自TAR）

TAR内JSON提供两条临床人工测量线。它们提供人工方向角和人工最终Full
Angle。

### 2.2 Gold Mask

`label_gold`是分割图像坐标系下的人工分割标签。它通过固定几何规则生成Gold
Card Axis和Gold Reference
Axis，用于给TAR中的两条人工线分配Card/Reference语义。

Gold Mask不负责提供TAR端点的像素级位置映射。

### 2.3 Infer Mask

`label_infer`是模型预测分割标签。模型预测的Card/Reference方向从Infer
Mask自身坐标系中计算，再形成预测Full Angle。

## 3. TAR读取

### 3.1 文件结构

使用Python `tarfile`读取TAR：

``` python
with tarfile.open(tar_path, "r:*") as tar:
    ...
```

从TAR成员中读取JSON。原始样本确认的结构为：

``` text
TAR
└── JSON
    └── Models
        └── MeasureModel
            ├── item 0
            │   └── Pos
            │       ├── p1
            │       └── p2
            └── item 1
                └── Pos
                    ├── p1
                    └── p2
```

权威规范要求递归查找`MeasureModel`列表。实际样本中路径为`Models.MeasureModel`。

### 3.2 有效测量线数量

每个正式TAR必须恰好包含两条有效测量线：

``` text
valid_manual_line_count == 2
```

少于或多于两条都应报错，不能静默截取前两条。

### 3.3 端点

每条线读取： - `Pos.p1` - `Pos.p2`

实际样本中坐标是三元素数组`[x, y, z]`，当前方向角计算使用x、y坐标。端点必须是有限数值，且两点不能重合。

## 4. 人工线方向角

对端点：

``` text
p1 = (x1, y1)
p2 = (x2, y2)
```

计算：

``` python
theta = degrees(atan2(y2 - y1, x2 - x1)) % 180
```

方向角范围：

``` text
[0°, 180°)
```

人工线是无方向直线： - 交换p1、p2不改变轴方向； -
相差180°表示同一条轴； - 不使用有方向的0°—360°误差。

两条人工线的锐角：

``` text
manual_full_angle = acute_angle_between_lines(line0, line1)
```

范围为`[0°,90°]`。

## 5. Mask类别与读取

权威规范中与固定几何方向相关的类别：

| Label ID | 规范中的用途             |
|---------:|--------------------------|
|        1 | reference-center相关区域 |
|        4 | cardiac区域之一          |
|        5 | cardiac区域之一          |
|        6 | body/thorax参考区域      |

不得仅凭ID数字自行重新解释类别语义。

### 5.1 最大连通域

固定几何基线对ID1、ID4、ID5、ID6分别： 1. 生成该ID的二值Mask； 2.
使用8连通域； 3. 选择最大前景连通域； 4.
后续面积、质心、轮廓和OBB均基于该连通域。

没有前景连通域时标记该区域无效，不得自动用其他类别替代。

## 6. Gold Card Axis

对Gold Mask中的ID4和ID5分别： 1. 取最大连通域； 2. 提取外轮廓； 3.
使用`cv2.minAreaRect`计算OBB； 4. 使用OBB最长边方向作为该区域方向； 5.
计算区域面积和OBB aspect ratio。

区域权重：

``` text
weight = area * max(aspect - 1.0, 0.05)
```

对有效方向采用双角度表示进行无方向加权平均：

``` text
x = Σ weight_i * cos(2 * theta_i)
y = Σ weight_i * sin(2 * theta_i)
theta_card = 0.5 * atan2(y, x)
```

归一化到`[0°,180°)`。

规则： - ID4、ID5都有效：按权重融合； - 仅一个有效：使用该区域方向； -
两者均无效：Gold Card Axis无效。

## 7. Gold Reference Axis

对Gold Mask： - ID6：取最大连通域质心； - ID1：取最大连通域质心。

以ID6质心到ID1质心的连线方向作为Gold Reference
Axis，按无方向轴处理并归一化到`[0°,180°)`。

若ID1或ID6任一缺失，Gold Reference Axis无效；不得自动使用其他结构代替。

## 8. 人工两条线的Card/Reference语义分配

TAR中两条线的顺序不保证语义。

设人工线方向为`a0`、`a1`，Gold几何方向为`gold_card_axis`、`gold_ref_axis`。

计算两种排列代价：

``` text
cost01 =
    axial_abs_diff(a0, gold_card_axis)
  + axial_abs_diff(a1, gold_ref_axis)

cost10 =
    axial_abs_diff(a1, gold_card_axis)
  + axial_abs_diff(a0, gold_ref_axis)
```

若`cost01 <= cost10`：

``` text
manual_card_axis = a0
manual_ref_axis  = a1
```

否则：

``` text
manual_card_axis = a1
manual_ref_axis  = a0
```

并记录： - 采用的分配方式； - `cost01`； - `cost10`； -
`manual_assignment_margin = abs(cost10 - cost01)`。

## 9. 固定人工GT

每个病例只进行一次Gold辅助语义分配。

分配后固定：

``` text
manual_card_axis
manual_ref_axis
manual_full_angle
```

后续所有Card候选或模型预测必须使用同一套固定人工GT。

禁止为每个候选重新从两条人工线中选择最近的一条，否则不同候选的误差不再可比。

如果Gold几何无法生成有效Card/Reference： - manual
Card/Reference单轴GT不可用； - 但TAR两条线之间的manual Full
Angle仍可计算； - 必须分别报告各指标的有效样本数。

## 10. 预测轴与最终Full Angle

预测方向从`label_infer`自身坐标系中的Mask计算：

- Predicted Card Axis：固定几何基线由ID4/ID5区域OBB方向加权得到；
- Predicted Reference Axis：ID6与ID1最大连通域质心连线。

预测Full Angle：

``` text
pred_full_angle =
    acute_angle_between_lines(pred_card_axis, pred_ref_axis)
```

最终误差：

``` text
full_error = abs(pred_full_angle - manual_full_angle)
```

Card/Reference单轴误差用于误差归因；**Full
Angle误差才是最终临床测量主指标。**

## 11. 角度误差定义

无方向轴绝对误差：

``` text
axial_abs_error(pred, gt)
= abs((pred - gt + 90) % 180 - 90)
```

范围`[0°,90°]`。

最终Full Angle均位于`[0°,90°]`，因此Full
Angle误差是两个锐角数值的绝对差。

## 12. 跨图像处理的边界

| 数据         | 所在坐标系         | 用途                        |
|--------------|--------------------|-----------------------------|
| 已标心轴图像 | 人工测量图像坐标系 | 对应TAR人工线               |
| TAR JSON     | 人工测量图像坐标系 | 提供两条人工线端点          |
| 分割PNG      | 分割图像坐标系     | 分割模型输入                |
| Gold Mask    | 分割图像坐标系     | Gold Card/Reference语义方向 |
| Infer Mask   | 分割图像坐标系     | 预测Card/Reference方向      |

不需要把上述坐标系统一。人工线和Mask之间通过方向角进行语义匹配。

## 13. 指标报告要求

至少分别报告： - manual Card Axis有效N； - manual Reference
Axis有效N； - manual Full Angle有效N； - predicted Card Axis有效N； -
predicted Reference Axis有效N； - predicted Full Angle有效N； - Full
Angle MAE及误差分布。

不能因为总队列为1474例就默认所有指标有效N都是1474。

## 14. 代码实施纪律

1.  以`CARDIAC_AXIS_DATA_READING_SPEC_v1.md`为唯一权威读取规范。
2.  不从历史代码中猜测或重新定义GT。
3.  用户明确说明，先前上传代码中的测量方法只是历史尝试，不应直接沿用；应沿用已确认的GT处理规则。
4.  任何GT规则修改必须先由用户确认，并更新规范版本。
5.  所有读取失败、缺失结构和无效角度必须显式记录，不允许静默回退。
