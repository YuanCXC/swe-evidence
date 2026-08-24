# Evaluation 评价模块

本目录负责评价 Evidence Agent 及各类 Baseline 的检索结果、策略决策和最终证据包。评价对象统一为已有 Evidence Unit，不在本目录中重新抽取、切分或改写仓库代码。

模块同时支持两条充分性评价轨道：

- **确定性评价：** 使用七类 Evidence Obligation 与 OR-of-AND Witness 计算覆盖率和充分性。
- **语义评价：** 使用大模型结合 Issue、Evidence Package、Gold Patch 与 Test Patch 判断证据是否足以支持参考修复。

Gold Patch、Test Patch、Obligation 和 Witness 仅供离线评价使用，不能进入在线检索、候选生成、Policy 打分或 Agent 路由。

## 数据流

```text
不同方法的原始输出
文件 / 符号 / 行号区间 / evidence_id
                ↓
          output_adapter.py
                ↓
       统一 Evidence Package
                ↓
      方法原生输出或排名截断点
                ↓
    ┌───────────┴───────────┐
    ↓                       ↓
确定性指标              大模型语义指标
retrieval /             semantic_judge /
localization /          semantic_metrics /
sufficiency /           judge_calibration
trajectory / interaction /
policy / cost
    └───────────┬───────────┘
                ↓
          aggregation.py
                ↓
          实验汇总结果
```

## 文件说明

| 文件 | 作用 | 主要接口 |
|---|---|---|
| `__init__.py` | 定义评价包的公共入口，导出最常用的适配、预算、充分性和语义 Judge 接口。 | `adapt_outputs`、`apply_budget`、`evaluate_sufficiency`、`build_semantic_judge_prompt`、`judge_evidence_package` |
| `output_adapter.py` | 将文件、符号、代码区间或 `evidence_id` 等异构输出映射到冻结数据集中的现有 Evidence Unit。映射过程保持原始排名并去除重复单元。 | `adapt_outputs` |
| `budget.py` | 为 Ours、消融及需要预算控制的内部方法施加 Evidence Unit 与 Token 安全上限。外部方法保留原生输出规模。 | `evidence_token_count`、`apply_budget` |
| `retrieval_metrics.py` | 评价 Retriever 是否将正确 Evidence 放入候选池，以及正确 Evidence 的排序位置。 | `recall_at_k`、`reciprocal_rank`、`ndcg_at_k`、`retrieval_metrics`、`structure_increment` |
| `localization_metrics.py` | 评价方法对正确文件、Evidence Unit、符号和代码行区间的定位能力。 | `localization_metrics`、`span_recall` |
| `policy_metrics.py` | 评价统一 Single、Pair、STOP 动作空间中的状态级 Policy 排序结果。该模块只消费动作分数，不负责加载或运行训练模型。 | `evaluate_policy_states` |
| `sufficiency_metrics.py` | 根据七类 Evidence Obligation 和 OR-of-AND Witness 计算确定性证据充分性。一个 Witness Group 内的 Evidence 为 AND，不同 Group 之间为 OR。 | `group_is_covered`、`obligation_is_covered`、`covered_obligation_ids`、`evaluate_sufficiency` |
| `trajectory_metrics.py` | 评价从空 Evidence State 到 STOP 的完整获取轨迹，包括成功率、过早停止、停止过晚、轮数、Tool Call 和预算终止。 | `evaluate_trajectories` |
| `interaction_metrics.py` | 评价 Evidence 之间的互补、替代和冗余关系，重点支持 Pair 与 Interaction 消融。 | `evaluate_interactions` |
| `cost_metrics.py` | 统计 Evidence Unit、Token、步骤和 Tool Call 成本，并计算充分性—成本曲线下面积。 | `evaluate_costs`、`auc_sufficiency_cost` |
| `semantic_judge.py` | 构造参考修复约束的大模型评审提示，并将 Evidence Package 交给调用方提供的大模型函数。模块不绑定具体 API。 | `render_evidence_package`、`build_semantic_judge_prompt`、`judge_evidence_package` |
| `semantic_metrics.py` | 聚合大模型 Judge 的结构化 JSON 输出，计算语义充分率、关键需求覆盖、因果正确性、执行相关性、语义精度和冗余率。 | `aggregate_semantic_judgments`、`repeated_judge_agreement` |
| `judge_calibration.py` | 构造关键证据删除、无关证据注入和跨任务证据注入等反事实样例，评价语义 Judge 对证据变化的敏感性。 | `delete_critical_evidence`、`inject_evidence`、`calibration_sensitivity` |
| `aggregation.py` | 按方法、预算或实验变体聚合逐任务数值结果，生成后续表格和绘图所需的汇总记录。 | `aggregate_rows` |
| `README.md` | 说明本目录的职责、文件分工、指标体系和使用边界。 | — |

## 指标分层

### Retriever 指标

用于回答「正确 Evidence 是否进入候选池」：

- Recall@1/5/10/20/64
- MRR
- NDCG@K
- Online Positive Coverage
- File Recall
- Evidence Unit Recall
- Structure Increment

### Localization 指标

用于回答「问题相关的文件、函数或代码区间是否定位正确」：

- Gold Evidence Recall
- Gold Evidence MRR
- File Recall
- File MRR
- Span Recall

### Policy 指标

用于回答「候选可达时，Policy 是否选对 Single、Pair 或 STOP」：

- Action Hit@1
- MRR
- NDCG
- Single Accuracy
- Pair Accuracy
- STOP Accuracy

### 确定性充分性指标

用于回答「最终 Evidence Package 是否满足结构化修复信息需求」：

- Evidence Sufficiency Rate
- Critical Requirement Coverage
- Obligation Coverage
- Witness Group Coverage
- Complementary Group Coverage
- 7 个 Evidence Dimension 的逐维覆盖率

### Agent 轨迹指标

用于回答「Agent 是否能以合理成本逐步获取充分证据并正确停止」：

- Trajectory Success Rate
- Premature STOP Rate
- Never STOP Rate
- Late STOP Overhead
- Average / Median Acquisition Steps
- Hard-budget Termination Rate
- Mean Evidence Count
- Mean Evidence Tokens
- Mean Tool Calls

### Evidence Interaction 指标

用于评价证据之间的互补、替代和冗余关系：

- Redundant Evidence Rate
- Substitute Duplication Rate
- Complementary Evidence Completion Rate

### 大模型语义指标

用于回答「Evidence 的实际语义是否足以理解并支持参考修复」：

- Reference-Grounded Semantic Sufficiency Rate
- Partial-credit Sufficiency Score
- Semantic Critical Requirement Coverage
- Causal Correctness Score
- Execution Relevance Score
- Semantic Precision
- Semantic Redundancy Rate
- Misleading Evidence Task Rate
- Judge Agreement

### 成本指标

用于比较各方法在原生输出规模下的证据获取效率，并评价排名方法的效果—成本曲线：

- Evidence Units
- Evidence Tokens
- Acquisition Steps
- Tool Calls
- AUC-Sufficiency-Cost
- 语义充分率—Token AUC
- 外部输出映射率
- 共同任务交集数量
- API Prompt / Completion / Total Tokens（本地 API 方法按任务记录；外部冻结结果仅在官方产物提供 usage 时统计）
- API Calls、外部 Agent 轮数、外部 Tool Calls 和运行耗时（仅在官方轨迹可观测时统计）
- Execution Cost Observation Rate，用于说明成本均值实际覆盖了多少共同任务

## 七类 Evidence Obligation

确定性和语义评价统一使用以下七类修复信息需求：

| 类型 | 含义 |
|---|---|
| `fault_location` | 故障所在文件、符号或代码位置。 |
| `fault_logic` | 导致异常行为的实现逻辑和根本原因。 |
| `dependency_context` | 修复所依赖的接口、配置、导入和外部约束。 |
| `state_flow` | 参数、状态、调用链和返回值的传播过程。 |
| `behavior_constraint` | Issue 描述的实际行为、预期行为和兼容性要求。 |
| `repair_scope` | 修复涉及的实现范围、调用方和潜在影响面。 |
| `validation_constraint` | 验证修复所需的测试、断言和边界条件。 |

## OR-of-AND Witness 语义

每个 Obligation 可以包含多个 `witness_groups`：

```text
[[A, B], [C]]
```

含义为：

```text
(A AND B) OR C
```

一个 Witness Group 内的全部 Evidence Unit 都被选中后，该 Group 才算覆盖。任意一个 Group 被完整覆盖后，对应 Obligation 即视为满足。

## 大模型 Judge 边界

`semantic_judge.py` 接受调用方传入的 `call_model(prompt) -> str` 函数，因此可接入任意返回文本的模型 API。Judge 必须在方法输出冻结后运行，并遵守以下边界：

- 不向 Judge 暴露方法名称，避免方法身份偏见。
- Gold Patch 与 Test Patch 只用于离线参考修复核对。
- Judge 结果不能反向影响 Retriever、Policy、Agent routing 或候选预算。
- Judge 输出使用固定 JSON 字段，理由使用简体中文。
- 正式实验需要报告重复判断一致率和反事实校准结果。

## 使用约定

- 本目录只负责评价，不负责读取 66 GB runtime 数据库或执行在线检索。
- 实验入口脚本放在 `exp/`，由入口脚本组织数据读取、方法运行和结果写出。
- 所有 Baseline 必须先通过 `output_adapter.py` 映射为统一 Evidence Package。
- Ours 及其消融使用相同的安全上限，但由 Policy 自主 STOP，不要求选满预算。
- Dense 与 Rerank 使用冻结的 `BAAI/bge-m3` tokenizer 对文件正文分块：每块最多 1024 Token、相邻块重叠 80 Token、每个文件最多保留前 8 块，并以最高块得分作为文件得分。分词器仅用于本地切分，Embedding 与 Rerank 推理仍通过 API 执行。
- Agentless、LocAgent 等具有原生终止行为的方法保留其最终输出，不强制对齐证据数量。
- Agentless 必须显式区分 `file`、`related`、`edit` 阶段；LocAgent 必须显式区分 `file`、`module`、`function` 层级，不跨阶段或层级拼接结果。
- SweRank 等只输出排名的方法使用论文报告的标准 `k` 截断点，分别统计效果、Evidence Token 和效果—成本曲线。
- 冻结外部结果没有 usage 或轨迹时，对应 API 成本、Agent 轮数和 Tool Call 记为缺失，不能记为 0；Evidence Unit 数与 Evidence Token 始终由映射后的证据包统一计算。
- 跨方法汇总只使用各运行均成功覆盖的共同任务交集，同时保留每个运行的原始任务数量。
- 实验结果按 `exp/results/{split}/{method}/` 存放，固定使用 `results.jsonl` 和 `manifest.json`；只有 SweRank 的 `k` 或显式 `--run-variant` 才增加变体目录。
- 离线 Judge 默认写入对应方法目录的 `judgments.jsonl` 和 `judgments.manifest.json`。manifest 冻结代码、数据、checkpoint、模型、提示版本与实验参数；配置不一致时禁止续写已有 JSONL。
- 论文正式汇总使用 `aggregate --inputs ... --expected-task-count N` 显式选择运行，避免调试 JSONL 意外缩小共同任务交集。
- 外部方法进入主表时可同时启用 `--require-external-snapshot-verified` 与 `--min-external-mapping-rate`。
- 机器读取的指标字段使用稳定英文名称；README、注释和大模型可读提示使用中文。
