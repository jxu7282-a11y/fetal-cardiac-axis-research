# 论文 Project 公共文档包

**版本：** v1.0  
**整理日期：** 2026-11-23  
**更新：** 2026-11-28（新增 `06_LABEL_SEMANTICS.md`）
**适用项目：** 中孕期胎儿心脏超声数据辅助早孕期心轴角自动测量

## 使用目的

这组文档用于让同一 Project
下不同职责的对话共享稳定、可追溯的课题背景，减少重复解释。它们是项目级摘要和工作记录，不替代原始数据规范、机器可读数据合同或服务器上的正式数据文件。

## 七份核心文档

| 文件                   | 用途                                                   | 主要使用者         |
|------------------------|--------------------------------------------------------|--------------------|
| `00_PROJECT_MASTER.md` | 课题目标、科学问题、当前主线和职责划分                 | 所有对话           |
| `01_DATA_SPEC.md`      | 数据来源、目录逻辑、数量、划分、审计结果               | 数据工程、实验设计 |
| `02_GT_PROTOCOL.md`    | TAR读取、人工线角度、Gold Mask语义分配、Full Angle评价 | 数据工程、实验工程 |
| `03_EXPERIMENT_LOG.md` | 实验阶段、方案、结果记录模板                           | 实验设计、Codex    |
| `04_DECISION_LOG.md`   | 已确认决策及其边界                                     | 所有对话           |
| `05_OPEN_QUESTIONS.md` | 未解决问题、待确认事项和下一步                         | 主线对话           |
| `06_LABEL_SEMANTICS.md` | 早孕（7类）/中孕（10类）标签语义已确认的对照表 | 所有对话           |

## 建议放入 Project Sources 的顺序

优先上传： 1. `00_PROJECT_MASTER.md` 2. `01_DATA_SPEC.md` 3.
`02_GT_PROTOCOL.md` 4. `04_DECISION_LOG.md` 5. `05_OPEN_QUESTIONS.md` 6.
`03_EXPERIMENT_LOG.md` 7. `06_LABEL_SEMANTICS.md`

## 重要：文档权威性与冲突处理

本项目现有原始规范仍然有效，不能被本摘要包静默替代：

- `CARDIAC_AXIS_DATA_READING_SPEC_v1.md`：早孕期人工测量线、Mask、Card/Reference语义分配和角度定义的权威读取规范。
- `cardiac_axis_data_contract_v1.yaml`：正式Manifest、Fold、数据约束、版本链和硬停止条件的机器可读合同。
- `CARDIAC_AXIS_METHOD_HISTORY.md`：此前固定几何、Tree、候选池、V2B/V2C等方法的历史记录；按原合同属于历史参考，不代表当前新的跨孕周研究主线。
- `CARDIAC_AXIS_CANDIDATE_DICTIONARY_v1.csv`：此前15个候选方向定义，属于旧方法分支资料。

若摘要与上述原始规范冲突，数据读取与GT细节以原始权威规范为准；当前研究方向以用户最近确认的项目决策为准。发现冲突时先标记并询问，不得自行覆盖。

## 本版内容来源

- 用户在当前 Project 中明确确认的信息。
- 已上传 `j.zip` 中的原始规范、数据合同、方法历史、候选字典和Manifest。
- 已上传 `audit_recheck_v2.zip` 中的只读审计结果。
- 本文档不把未核实的推测写成事实。

## 更新规则

每次有重要变化时： 1. 修改对应文档，不要在多个文档重复维护同一细节； 2.
在 `04_DECISION_LOG.md` 或 `03_EXPERIMENT_LOG.md` 记录变化原因； 3.
更新版本号和日期； 4. 保留旧版本，不静默覆盖； 5.
专题对话结束时，输出可合并进公共文档的结论。
