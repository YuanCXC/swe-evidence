# Unified SWE Dataset 发布结构与任务 Schema 设计

## 1. 目标

本设计定义 Unified SWE Dataset 的最终发布产物，以及训练、验证、评测三个任务文件的统一字段结构。数据集服务于预算约束下的仓库证据获取：模型根据问题、当前证据状态和候选动作，判断下一步应获取单个证据、证据组合，还是停止检索。

数据处理阶段使用 JSONL 作为可审计真值；正式发布阶段转换为 Parquet。最终发布只保留 5 个文件：

```text
unified_swe_dataset_v1/
├── train.parquet
├── validation.parquet
├── benchmark.parquet
├── repository_corpus.parquet
└── manifest.json
```

设计遵循以下原则：

- 一个任务只占一行。
- 同一任务的多数据集来源合并为 provenance，不重复计数。
- 模型输入与离线监督严格分区。
- train、validation、benchmark 使用完全相同的物理 Schema。
- 三个任务文件在发布前完成物理切分，不依赖运行时随机划分。
- 完整代码只保存在 `repository_corpus.parquet`，每个唯一文件版本的正文只保存一次。
- 任务文件通过稳定的 `evidence_id` 引用代码。
- SWE-bench 是唯一任务基准；外部来源必须可靠对齐到 SWE-bench，否则不下载、不合并、不发布。
- 监督区分确定性标签、跨来源标签和教师伪标签，不把三者混成同等可信的 Gold。
- Cross-Encoder 只读取有界 Evidence Unit，不读取整个仓库或不受控的完整文件正文。
- 最终只训练并部署一个 Cross-Encoder Evidence Policy Ranker；候选召回、候选配对和动作选择规则不引入其他可训练模型。

### 1.1 研究目标

模型学习的不是补丁生成，而是状态相关的证据动作价值：

```text
(q, K, A) -> utility(A | q, K)
```

- `q`：SWE-bench 问题描述。
- `K`：当前已获取的证据集合。
- `A`：单证据动作 `[u]`、双证据动作 `[u, v]` 或 `STOP`。
- `utility`：动作对证据义务完成度和进度的增益，训练时可扣除正文 Token 成本。

研究重点是证据之间的互补、替代、冗余、独立和冲突关系，以及严格的停止条件。补丁和测试只用于离线监督与最终系统评测，不进入 Cross-Encoder 在线输入。

### 1.2 非目标

- 不从零训练大语言模型，只微调预训练 Cross-Encoder Ranker。
- 不把百万级样本数作为目标；必须同时报告独立任务数、状态数、动作数和候选数。
- 不把完整仓库拼接到模型输入。
- 不把成功轨迹中的所有读取自动视为必要证据。
- 不把教师模型输出直接当作 benchmark 真值。
- 不引入无法与 SWE-bench 可靠对齐的新任务数据集。

## 2. 文件职责

| 文件 | 一行表示 | 用途 |
|------|----------|------|
| `train.parquet` | 一个可训练的统一任务 | 参数训练、行为模仿、弱监督预训练 |
| `validation.parquet` | 一个冻结的验证任务 | 调参、早停、阈值选择、回归验证 |
| `benchmark.parquet` | 一个正式评测任务 | 证据检索、充分性判断、代码修复评测 |
| `repository_corpus.parquet` | 一个唯一文件版本 | 保存正文、快照成员关系及嵌套 Evidence Unit |
| `manifest.json` | 整个发布版本的元数据 | Schema、来源、哈希、统计和审计 |

## 3. 公共任务 Schema

三个任务文件共享以下顶层结构：

```json
{
  "schema_version": "1.0",
  "task_id": "task_django_12345",
  "task_group_id": "group_django_issue_12345",
  "snapshot_id": "snapshot_django_abc123",
  "input": {},
  "provenance": [],
  "supervision": {},
  "trajectories": [],
  "evaluation": null,
  "split_info": {},
  "quality": {}
}
```

### 3.1 身份字段

| 字段 | Parquet 类型 | 必填 | 含义 |
|------|--------------|------|------|
| `schema_version` | string | 是 | 当前记录使用的 Schema 版本 |
| `task_id` | string | 是 | 去重后的唯一任务 ID |
| `task_group_id` | string | 是 | 同一 Issue、PR 或任务谱系的泄漏隔离 ID |
| `snapshot_id` | string | 是 | 修复前仓库快照 ID |

约束：

- `task_id` 在三个任务文件的并集中唯一。
- 同一 `task_group_id` 只能出现在一个任务文件中。
- `snapshot_id` 必须在 `repository_corpus.parquet` 中至少出现一次。
- 不得仅凭 `repo + base_commit` 判断两个任务相同。

### 3.2 `input`

`input` 是模型和 Retriever 可以访问的唯一任务输入区。

| 子字段 | 类型 | 必填 | 含义 |
|--------|------|------|------|
| `repo` | string | 是 | 规范化仓库名 |
| `base_commit` | string | 是 | 修复发生前的 Git commit |
| `language` | string | 是 | 主要编程语言 |
| `issue_id` | string | 否 | Issue、PR 或上游任务编号 |
| `problem_statement` | string | 是 | 待解决的问题描述 |
| `hints` | list\<string> | 是 | 上游公开提示；没有时为空数组 |
| `created_at` | timestamp | 否 | Issue 或任务创建时间 |
| `environment` | struct | 否 | 安装、运行和测试环境 |
| `retrieval_scope` | struct | 是 | 允许检索的快照和 Evidence Unit 类型 |

`environment`：

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `runtime` | string | 运行时及版本，例如 `python:3.11` |
| `install_command` | string | 安装命令 |
| `test_command` | string | 测试命令 |
| `container_image` | string | 可复现执行环境的镜像标识 |

`retrieval_scope`：

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `snapshot_id` | string | 允许检索的唯一代码快照 |
| `allowed_unit_types` | list\<string> | 可检索的证据粒度 |

禁止写入 `input`：

- Gold patch；
- test patch；
- Gold 文件、函数、span；
- obligation 和 witness；
- 成功轨迹；
- 由 Gold 派生的搜索范围。

### 3.3 `provenance`

`provenance` 类型为 `list<struct>`。同一任务可以关联多个来源。

| 子字段 | 类型 | 必填 | 含义 |
|--------|------|------|------|
| `dataset` | string | 是 | 数据集名称 |
| `subset` | string | 否 | train、test、verified 等上游子集 |
| `source_id` | string | 是 | 上游记录 ID |
| `version` | string | 是 | 数据集版本 |
| `revision` | string | 是 | commit、tag 或不可变 revision |
| `license` | string | 是 | 许可证标识 |
| `trust_tier` | string | 是 | `strong`、`support`、`weak` 或 `observed` |
| `raw_record_sha256` | string | 是 | 原始记录的稳定哈希 |

同一来源的 overlay 只增加 provenance 或监督标签，不创建新的任务行。

### 3.4 `supervision`

`supervision` 保存离线训练标签和评分答案，不属于模型输入。

| 子字段 | 类型 | 必填 | 含义 |
|--------|------|------|------|
| `level` | string | 是 | `strong`、`support`、`weak` 或 `none` |
| `training_targets` | list\<string> | 是 | 可用于训练的能力 |
| `recommended_weight` | float32 | 否 | 建议基础权重，不是固定采样概率 |
| `evidence_labels` | list\<struct> | 是 | 证据相关性标签 |
| `modified_files` | list\<string> | 是 | Gold patch 修改的文件 |
| `gold_patch` | string | 否 | 正确修复补丁 |
| `test_patch` | string | 否 | 复现或验证问题的测试补丁 |
| `hard_negative_evidence_ids` | list\<string> | 是 | 高相关但不支持结论的证据 |
| `obligations` | list\<struct> | 是 | 证据义务及 witness |
| `policy_states` | list\<struct> | 是 | 状态、候选动作、动作增益和 STOP 标签 |
| `label_provenance` | list\<struct> | 是 | 确定性、跨来源、教师和人工标签的逐项来源记录 |

`label_provenance` 中每项包含：

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `annotation_id` | string | 任务内唯一标注 ID |
| `source` | string | `deterministic`、`cross_source`、`teacher_consensus` 或 `human` |
| `source_record_ids` | list\<string> | 参与该判断的原始记录 ID |
| `teacher_model` | string | 教师模型及 revision；非教师标签为 null |
| `prompt_version` | string | Prompt 版本；非教师标签为 null |
| `vote_count` | int32 | 一致票数；非投票标签为 null |
| `agreement` | float32 | 一致率；非投票标签为 null |
| `rule_verified` | bool | 是否通过确定性约束校验 |
| `input_sha256` | string | 标注输入包的稳定哈希 |

`evidence_labels` 中每项包含：

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `evidence_id` | string | `repository_corpus.parquet` 中的证据 ID |
| `relevance` | string | `positive`、`hard_negative`、`unknown`；未被 Gold 选中不能直接标为负例 |
| `granularity` | string | file、class、function、span 或 code_block |
| `source` | string | 标签来源 |
| `confidence` | float32 | 映射或人工标注置信度，范围为 `[0, 1]` |
| `annotation_ids` | list\<string> | 对应的 `label_provenance.annotation_id` |

`obligations` 中每项包含：

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `obligation_id` | string | 任务内唯一义务 ID |
| `type` | string | 固定的义务类型 |
| `description` | string | 必须得到证据支持的判断 |
| `applicable` | bool | 该义务是否适用于当前任务 |
| `mandatory` | bool | 是否为严格 STOP 的必要条件 |
| `confidence` | float32 | 义务定义置信度 |
| `construction_method` | string | 规则、跨来源一致、教师共识或人工审核 |
| `witness_groups` | list\<struct> | 可以满足该义务的一个或多个证据组 |
| `annotation_ids` | list\<string> | 对应的标注来源 ID |

`witness_groups` 中每项包含：

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `group_id` | string | 任务内唯一 witness group ID |
| `evidence_ids` | list\<string> | 该组引用的 Evidence Unit |
| `logic` | string | 固定为 `AND`；同一义务下不同 group 之间按 `OR` 解释 |
| `source` | string | ContextBench、SWE-Explore、patch、结构规则、教师或人工 |
| `confidence` | float32 | 证据组置信度 |
| `annotation_ids` | list\<string> | 对应的标注来源 ID |

同一 group 内的证据必须共同获得；不同 group 是可替代路径。例如，`caller + callee` 可以组成一个 AND group，而集成测试可以组成另一个单证据 group。前者内部是互补关系，两个 group 之间是替代关系。

`policy_states` 中每项包含：

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `state_id` | string | 任务内唯一状态 ID |
| `step` | int32 | 状态序号；初始状态为 0 |
| `evidence_ids` | list\<string> | 当前状态 `K` 已获得的 Evidence Unit |
| `completed_obligation_ids` | list\<string> | 已完整满足的义务 |
| `completion_score` | float32 | 当前强制义务完成度 `C(K)` |
| `progress_score` | float32 | 当前 witness 进度 `P(K)` |
| `candidate_actions` | list\<struct> | 单证据、双证据和 STOP 动作 |
| `label_source` | string | 状态及动作标签的构造来源 |
| `confidence` | float32 | 状态级标签置信度 |

`candidate_actions` 中每项包含：

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `action_id` | string | 规范化动作 ID |
| `action_type` | string | `single`、`pair` 或 `stop` |
| `evidence_ids` | list\<string> | 单证据长度为 1，双证据长度为 2，STOP 为空 |
| `candidate_scope` | string | `online`、`offline_injected` 或 `stop`；只用于数据审计，不进入模型输入 |
| `candidate_sources` | list\<string> | BM25、路径、符号、结构关系、witness、教师或困难负例等候选来源 |
| `online_retrieval_rank` | int32 | 在线召回时的原始名次；离线注入动作和 STOP 为 null |
| `completion_gain` | float32 | 动作带来的 `C` 增量 |
| `progress_gain` | float32 | 动作带来的 `P` 增量 |
| `completion_interaction` | float32 | 双证据的完成度交互增益 `I_C`；单证据和 STOP 为 null |
| `progress_interaction` | float32 | 双证据的 witness 进度交互增益 `I_P`；单证据和 STOP 为 null |
| `token_cost` | int32 | 加入状态的正文 Token 数 |
| `relations` | list\<struct> | 当前状态下按 obligation 记录的双证据关系；单证据和 STOP 为空数组 |
| `relation_targets` | struct | 多标签关系头的 complement、substitute、redundant、independent、conflict 目标值 |
| `covered_obligation_ids` | list\<string> | 动作完成或推进的义务 |
| `semantic_useful` | bool | 动作是否对 `C` 或 `P` 产生正语义增益；标签未知时为 null |
| `policy_acceptable` | bool | 动作是否属于当前状态下值得执行的非支配动作集合；标签未知时为 null |
| `pareto_dominated` | bool | 证据动作是否被同状态中的其他证据动作 Pareto 支配；STOP 为 null |
| `dominated_by_action_ids` | list\<string> | 严格支配当前动作的同状态动作 ID；不可判定时为空列表 |
| `label_source` | string | 规则、跨来源、轨迹、教师共识或人工 |
| `confidence` | float32 | 动作标签置信度 |
| `relation_loss_mask` | bool | 是否至少存在一个可参与多标签关系损失的已知目标 |
| `annotation_ids` | list\<string> | 对应的标注来源 ID |

`relations` 中每项包含 `obligation_id`、`relation`、`confidence`、`label_source` 和 `annotation_ids`。单证据和 STOP 动作的 `relations=[]`、`relation_targets=null` 且 `relation_loss_mask=false`。双证据动作的 `evidence_ids` 必须排序，保证 `[u, v]` 与 `[v, u]` 只有一个物理动作。训练时可以随机交换两段正文的输入顺序，避免模型学习位置偏差。

允许的 `training_targets`：

- `evidence_action_ranking`
- `interaction_classification`

`evidence_action_ranking` 同时覆盖单证据、双证据和 STOP 的统一排序。`interaction_classification` 只表示同一模型关系辅助头的标签可用性。候选召回不是训练目标，STOP 也不作为独立模型或独立训练目标。

补丁生成、测试生成和失败轨迹建模不属于本数据集第一版的训练目标。

### 3.5 `trajectories`

`trajectories` 类型为 `list<struct>`。只有 `train.parquet` 可以包含非空轨迹。

轨迹字段：

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `trajectory_id` | string | 全局唯一轨迹 ID |
| `source` | string | 轨迹来源 |
| `model` | string | 生成轨迹的模型 |
| `resolved` | bool | 是否成功解决任务 |
| `reward` | float32 | 上游奖励；缺失时为 null |
| `steps` | list\<struct> | 有序行为步骤 |

步骤字段：

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `step` | int32 | 从 0 开始的步骤序号 |
| `action_type` | string | search、open、edit、test、stop 等 |
| `action` | string | 规范化后的动作内容 |
| `observation` | string | 工具返回或环境观察 |
| `evidence_ids` | list\<string> | 此步骤访问的 Evidence Unit |

失败轨迹只能用于 `failure_modeling`，不得自动生成 Gold evidence 或 Gold patch。

### 3.6 `evaluation`

`evaluation` 是可空 struct。`validation.parquet` 和 `benchmark.parquet` 必填；`train.parquet` 默认为 null。

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `benchmark_memberships` | list\<struct> | 上游评测集身份 |
| `targets` | list\<string> | 可评测能力 |
| `primary_metric` | string | 主要选择或报告指标 |
| `metrics` | list\<string> | 可计算指标 |
| `gold_visibility` | string | `evaluator_only` 或 `private` |
| `timeout_seconds` | int32 | 单任务最大执行时间 |
| `execution_required` | bool | 是否必须运行测试环境 |

`benchmark_memberships` 中每项包含：

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `suite` | string | 评测集名称 |
| `subset` | string | 上游子集 |
| `version` | string | 上游版本 |
| `original_source_id` | string | 上游实例 ID |

同一任务属于多个评测集时，只保留一行，并记录多个 membership。

### 3.7 `split_info`

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `split` | string | `train`、`validation` 或 `benchmark` |
| `trainable` | bool | 是否允许参数训练 |
| `split_reason` | string | 进入当前 split 的原因 |
| `split_policy_version` | string | 切分规则版本 |
| `leakage_group` | string | 等于 `task_group_id` |
| `frozen` | bool | 发布后是否禁止自动迁移 |

三个任务文件都要求 `frozen=true`。

### 3.8 `quality`

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `status` | string | `passed` 或 `passed_with_warnings` |
| `identity_confidence` | float32 | 身份合并置信度 |
| `label_confidence` | float32 | 标签综合置信度 |
| `executable` | bool | 测试环境是否可执行 |
| `snapshot_available` | bool | 是否具备完整仓库快照 |
| `evidence_mapping_rate` | float32 | Gold 到 Evidence Unit 的映射率 |
| `warnings` | list\<string> | 非阻断质量提示 |

高严重度身份冲突、悬空证据引用和快照缺失不得以 warning 形式进入发布。

## 4. `train.parquet`

### 4.1 用途

`train.parquet` 用于唯一 Cross-Encoder Evidence Policy Ranker 的参数训练。关系标签可以作为同一模型的辅助监督；单证据、双证据和 STOP 必须由同一个动作排序头统一评分。Retriever 只使用不可训练的词法、路径、符号和仓库结构规则，不使用本文件训练第二个模型。

### 4.2 字段约束

| 字段 | 约束 |
|------|------|
| `split_info.split` | 固定为 `train` |
| `split_info.trainable` | 固定为 `true` |
| `evaluation` | null |
| `supervision.level` | `strong`、`support` 或 `weak` |
| `supervision.training_targets` | 至少包含 1 项 |
| `trajectories` | 可以为空或非空 |

监督等级与默认建议权重：

| 等级 | 典型来源 | 建议权重 |
|------|----------|---------:|
| `strong` | ContextBench、跨来源一致、人工确认 | 1.0 |
| `support` | SWE-bench patch、SWE-Explore core、规则验证教师标签 | 0.7 |
| `weak` | SWE-Explore optional、单教师或行为轨迹 | 0.4 |

实际采样策略由训练配置控制，数据集只记录建议值。

### 4.3 来源合并

- SWE-bench 提供任务、仓库、commit 和 patch。
- ContextBench 仅为能够严格对齐的 SWE-bench 任务补充 Gold evidence，不新增任务。
- SWE-Explore 仅为能够严格对齐的 SWE-bench 任务补充共识区域、可选区域和读取顺序，不新增任务。
- 无法与 SWE-bench 对齐的 ContextBench、SWE-Explore 或其他来源记录不下载、不合并、不发布。
- 第一版不接入 SWE-rebench、SWE-Gym、SWE-smith、Nebius 或其他扩展任务集。
- SWE-bench train 中缺少强语义标签的任务，可以通过受约束教师标注补足训练监督。

## 5. `validation.parquet`

### 5.1 用途

`validation.parquet` 用于调参、早停、阈值选择和回归验证，不参与参数训练。

### 5.2 字段约束

| 字段 | 约束 |
|------|------|
| `split_info.split` | 固定为 `validation` |
| `split_info.trainable` | 固定为 `false` |
| `supervision.level` | 只能是 `strong` 或 `support` |
| `trajectories` | 固定为空数组 |
| `evaluation` | 必填 |
| `evaluation.gold_visibility` | 固定为 `evaluator_only` |

模型只能读取 `input` 和 `snapshot_id`。Evaluator 可以读取 `supervision` 和 `evaluation`。

### 5.3 切分要求

- `train.parquet` 继承 SWE-bench train，共 19,008 个任务。
- `validation.parquet` 继承 SWE-bench dev，共 225 个任务。
- `benchmark.parquet` 继承 SWE-bench test，共 2,294 个任务。
- ContextBench、SWE-Explore 和教师派生标签必须继承对应 SWE-bench 任务的 split，不得重新随机切分。
- 同一 `task_group_id`、Issue、PR 或派生变体不得跨 split。
- train、validation 和 benchmark 在任何标签生成、教师调用和候选挖掘之前冻结。
- benchmark 任务及其标签不得参与模型参数更新、阈值拟合或困难负例挖掘。

当前冻结来源的强监督分布为：train 中 38 个 ContextBench Poly 对齐任务；benchmark 中 313 个 ContextBench Verified 对齐任务和 451 个 SWE-Explore Verified 对齐任务。由于大部分跨来源强标签位于 benchmark，教师标注的主要用途是补足 train 和 validation，而不是把 benchmark 数据迁移到 train。

## 6. `benchmark.parquet`

### 6.1 用途

`benchmark.parquet` 是正式、冻结、不可训练的评测集。

### 6.2 字段约束

| 字段 | 约束 |
|------|------|
| `split_info.split` | 固定为 `benchmark` |
| `split_info.trainable` | 固定为 `false` |
| `trajectories` | 固定为空数组 |
| `evaluation` | 必填 |
| `evaluation.benchmark_memberships` | 至少包含 1 项 |
| `split_info.split_reason` | 必须说明上游评测身份或冻结策略 |

评测轨道：

| 轨道 | 主要来源 | 主要指标 |
|------|----------|----------|
| 代码修复 | SWE-bench test | `patch_resolved`、测试通过率 |
| 证据定位 | ContextBench、SWE-Explore、patch 派生位置 | Recall@K、MRR、nDCG |
| 证据充分性 | ContextBench、SWE-Explore、人工审核样本 | obligation coverage、STOP accuracy |
| 证据交互 | witness group 与人工审核样本 | multi-label pair relation F1、pair utility ranking |

所有证据类评测必须分别报告两种候选模式：

- `oracle_candidate_ranking`：向候选池注入缺失 Gold，只评估唯一 Cross-Encoder 对已给定候选的排序、组合与 STOP 能力；
- `end_to_end_online`：候选只能由问题、当前证据和 pre-fix 仓库生成，不允许 Gold 注入，用于评估规则召回与 Cross-Encoder 组成的完整系统。

两个模式必须分别报告样本数和指标，不得用 Oracle 候选结果代替端到端结果。

### 6.3 Gold 发布策略

内部完整版：

- `supervision` 保存完整 Gold；
- `evaluation.gold_visibility=evaluator_only`；
- 模型加载器必须丢弃 `supervision`。

对外发布版：

- `supervision` 删除 Gold 内容或置空；
- `evaluation.gold_visibility=private`；
- 使用私有 Evaluator 评分。

同一发布目录只能选择一种 flavor，并在 `manifest.json` 中记录 `release_flavor`。内部完整版的逻辑隔离不等于保密，不能直接作为公开 benchmark 发布。

## 7. 三个任务文件的字段矩阵

| 字段 | train | validation | benchmark |
|------|-------|------------|-----------|
| `input` | 必填 | 必填 | 必填 |
| `provenance` | 必填 | 必填 | 必填 |
| `supervision` | 必填 | 必填 | 内部版必填，公开版脱敏 |
| `trajectories` | 可非空 | 空数组 | 空数组 |
| `evaluation` | null | 必填 | 必填 |
| `split_info.trainable` | true | false | false |
| `split_info.frozen` | true | true | true |
| Gold 对模型可见 | 否 | 否 | 否 |

## 8. `repository_corpus.parquet`

### 8.1 行粒度

每行表示一个唯一文件版本。唯一键为 `repo + path + blob_oid`：同一仓库、同一路径、同一内容只保存一行，使用该版本的全部快照记录在 `snapshot_ids` 中。

```json
{
  "file_version_id": "fv_django_query_7a80d963",
  "repo": "django/django",
  "path": "django/db/models/query.py",
  "blob_oid": "4d5c6e7f8a9b",
  "snapshot_ids": [
    "snapshot_django_abc123",
    "snapshot_django_def456"
  ],
  "language": "python",
  "content": "class QuerySet:",
  "content_sha256": "7a80d963",
  "line_count": 850,
  "attributes": {
    "is_test": false,
    "is_generated": false,
    "is_vendor": false,
    "is_binary": false,
    "searchable": true
  },
  "evidence_units": [
    {
      "evidence_id": "ev_django_query_7a80d963_file",
      "unit_type": "file",
      "symbol": null,
      "qualified_name": null,
      "start_line": 1,
      "end_line": 850,
      "parent_evidence_id": null,
      "content_sha256": "7a80d963"
    }
  ],
  "imports": [],
  "extraction": {
    "parser": "tree-sitter-python",
    "parser_version": "0.23.0",
    "status": "success"
  }
}
```

### 8.2 文件字段

| 字段 | 类型 | 必填 | 含义 |
|------|------|------|------|
| `file_version_id` | string | 是 | 唯一文件版本 ID |
| `repo` | string | 是 | 规范化仓库名 |
| `path` | string | 是 | 仓库内相对路径 |
| `blob_oid` | string | 是 | Git tree 中的文件内容对象 ID |
| `snapshot_ids` | list\<string> | 是 | 使用此文件版本的全部快照 ID |
| `language` | string | 是 | 编程语言或文档类型 |
| `content` | string | 否 | 完整文件正文 |
| `content_sha256` | string | 是 | 文件内容哈希 |
| `line_count` | int32 | 是 | 文件总行数 |
| `attributes` | struct | 是 | 文件分类和检索控制 |
| `evidence_units` | list\<struct> | 是 | 文件内的证据单元 |
| `imports` | list\<struct> | 是 | 文件的静态依赖 |
| `extraction` | struct | 是 | 结构提取过程和状态 |

唯一性和成员关系：

- `file_version_id` 在整个 corpus 中唯一。
- `repo + path + blob_oid` 在整个 corpus 中唯一。
- `file_version_id` 由 `repo + path + blob_oid` 的稳定哈希生成。
- `snapshot_ids` 必须经过排序和去重。
- 展开所有 `snapshot_ids` 后，`snapshot_id + path` 必须唯一命中一个 `file_version_id`。
- 同一路径内容变化时创建新的文件版本；内容恢复时重新关联已有版本。
- 同一内容出现在不同路径或不同仓库时，不跨路径或仓库合并。
- `content_sha256` 用于正文完整性校验，不单独承担文件版本身份。

二进制文件可以保留路径和哈希，但 `content=null`、`attributes.searchable=false` 且 `evidence_units=[]`。

### 8.3 `attributes`

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `is_test` | bool | 是否为测试文件 |
| `is_generated` | bool | 是否为自动生成文件 |
| `is_vendor` | bool | 是否为第三方依赖 |
| `is_binary` | bool | 是否为二进制文件 |
| `searchable` | bool | 是否允许 Retriever 检索 |

默认情况下，二进制、vendor 和 generated 文件不可检索。测试文件是否可检索由任务的 `input.retrieval_scope` 决定。

### 8.4 `evidence_units`

`evidence_units` 保存文件内的可检索结构，不重复保存正文：

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `evidence_id` | string | 全局唯一 Evidence Unit ID |
| `unit_type` | string | file、class、function、method、code_block 或 doc_section |
| `symbol` | string | 简短符号名；不适用时为 null |
| `qualified_name` | string | 完整限定名；不适用时为 null |
| `start_line` | int32 | 起始行，包含该行 |
| `end_line` | int32 | 结束行，包含该行 |
| `parent_evidence_id` | string | 父级 Evidence Unit；没有时为 null |
| `content_sha256` | string | 对行号范围对应正文切片计算的哈希 |
| `token_count` | int32 | 使用冻结 Tokenizer 计算的正文 Token 数 |
| `scoreable` | bool | 是否允许作为 Cross-Encoder 候选动作 |

行号采用 1-based、闭区间语义。Evidence Unit 的正文通过以下规则恢复：

```python
unit_content = file_content_lines[start_line - 1:end_line]
```

每个可搜索文本文件必须生成且只生成一个 `unit_type=file` 的 Evidence Unit，范围为 `1..line_count`。解析成功时继续生成 class、function、method 和 doc_section。无法解析的可搜索文本文件按固定窗口降级生成 `code_block`，不得因为解析失败丢弃完整文件。

文件级 Evidence Unit 用于成员关系、文件级召回和审计。如果正文超过 Cross-Encoder 上限，则必须设置 `scoreable=false`，不能直接成为动作。过大的类、函数或 ContextBench/SWE-Explore 区域继续按语法边界或固定窗口切分，直到形成有界单元。双证据动作的两段正文加上问题和状态摘要后仍必须满足模型最大长度；无法满足时拆分动作或放弃该 pair，不允许静默截断关键证据。

所有监督标签统一引用 `evidence_id`：文件级标签指向 `unit_type=file`，细粒度标签指向对应的类、函数、方法、代码块或文档章节。`file_version_id` 只用于标识 corpus 的物理文件版本行，不作为监督标签 ID。

训练加载器根据 `file_version_id + start_line + end_line` 恢复 Evidence Unit 正文。任务文件不重复保存代码正文，只保存 `evidence_id`、状态和标签。

### 8.5 `imports`

`imports` 保存第一版检索扩展所需的静态依赖：

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `module` | string | 源文件声明的模块或包 |
| `declared_at_line` | int32 | import 声明所在行 |

`imports` 只保存由文件内容决定的声明，不保存 `resolved_path`。同一文件版本在不同 snapshot 中可能解析到不同目标，具体路径必须在加载任务快照时，根据该 snapshot 的文件成员关系确定。第一版不发布完整调用图、继承图或模型生成的关系边。

### 8.6 `extraction`

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `parser` | string | 解析器名称；未使用解析器时为 `line-window` |
| `parser_version` | string | 解析器版本 |
| `status` | string | `success`、`fallback` 或 `unsupported` |

任何任务只能访问 `snapshot_ids` 包含自身 `snapshot_id`、且 `attributes.searchable=true` 的文件版本和 Evidence Unit。训练加载器可以在内存中建立 `snapshot_id → file_version_id` 索引，并展开 `evidence_units` 形成函数级候选；索引和展开结果都不是新的发布文件。

## 9. 单命令构建与状态管理

### 9.1 用户接口

完整数据集通过一条命令生成：

```powershell
python scripts/build_unified_dataset.py --format jsonl
```

用户可见输入：

```text
data/raw/
data/cache/repos/
```

第一版只允许在 `scripts/` 下新增或保留一个最终构建入口 `build_unified_dataset.py`。不得为 adapter、标签阶段或发布阶段新增其他 Python 文件，也不使用外部配置文件。来源 revision、字段映射、质量门槛和 Prompt 版本全部作为版本化常量集中在该脚本内。教师 API 地址、密钥和模型名通过环境变量传入，密钥不得写入脚本、SQLite、manifest 或发布文件。

缺少已启用的 SWE-bench、ContextBench 或 SWE-Explore 原始文件时，脚本从固定官方地址下载并校验 revision 与哈希。无法与 SWE-bench 对齐的来源记录不进入后续下载和仓库快照准备。

唯一构建状态文件：

```text
data/.build/unified_swe_v1.sqlite3
```

最终发布目录：

```text
data/unified_swe_dataset_v1/
├── train.jsonl
├── validation.jsonl
├── benchmark.jsonl
├── repository_corpus.jsonl
└── manifest.json
```

数据处理和实验阶段默认使用 JSONL，方便逐行检查。正式发布时运行：

```powershell
python scripts/build_unified_dataset.py --format parquet --release
```

正式发布保持第 1 节定义的 4 个 Parquet 文件和 1 个 `manifest.json`。JSONL 与 Parquet 共用同一逻辑 Schema 和稳定 ID，格式转换不得重新生成标签或改变 split。

构建过程不得再生成 `normalized_instances.jsonl`、`master_instances.jsonl`、`split_assignments.jsonl`、`candidate_files.jsonl`、`evidence_units.jsonl` 或 `certification_labels.jsonl` 等独立中间产物。

### 9.2 SQLite 状态表

单个 SQLite 状态库至少包含：

```text
source_records
canonical_tasks
task_aliases
supervision
trajectories
split_assignments
snapshots
file_versions
snapshot_file_memberships
evidence_units
obligations
witness_groups
policy_states
candidate_actions
teacher_cache
conflicts
build_phases
```

SQLite 只承担规范化、关联、断点和审计状态，不属于最终发布数据。文件正文可以存储在 `file_versions` 中，也可以在本地 Git blob 可用时仅保存 `blob_oid` 和提取状态；无论采用哪种方式，最终内容必须从已校验的 Git blob 生成。

### 9.3 内部阶段

单次命令按以下阶段执行：

```text
校验原始来源和哈希
→ 来源专用规范化
→ 任务身份合并
→ train / validation / benchmark 切分
→ Git snapshot 校验
→ 唯一文件版本和成员关系枚举
→ Evidence Unit 提取
→ 监督与轨迹映射
→ 义务与 witness group 构建
→ 必要时执行受约束教师标注
→ 状态与单/双证据动作标签构建
→ 流式写入临时 JSONL 或 Parquet
→ 完整性审计
→ 原子发布
```

每个阶段在 `build_phases` 中记录：

- 阶段名称和版本；
- 输入指纹；
- 开始、完成和失败时间；
- 已处理数量；
- 输出表行数；
- 错误摘要；
- 是否可以断点续跑。

输入指纹或阶段版本变化时，只使受影响阶段及其下游失效，不得静默复用不兼容状态。

### 9.4 临时文件和原子发布

JSONL 逐行写入；Parquet 使用分批 row group 流式写入，不要求将完整数据载入内存。构建器先写入与目标格式一致的临时目录。正式 Parquet 发布示例如下：

```text
data/unified_swe_dataset_v1.tmp/
├── train.parquet
├── validation.parquet
├── benchmark.parquet
├── repository_corpus.parquet
└── manifest.json
```

只有 5 个文件全部通过发布硬门槛后，才将临时目录原子替换为正式目录。失败时：

- 已发布的旧版本保持不变；
- 临时目录不得被标记为 release；
- 下次从 SQLite 中最近的兼容阶段继续；
- 已完成的 Git inventory 和 Evidence Unit 提取不得无条件重跑。

### 9.5 状态清理

构建成功后默认保留 SQLite，便于复现和增量更新。用户可以显式删除：

```powershell
python scripts/build_unified_dataset.py `
  --clean-state
```

`--clean-state` 只能在正式目录完成哈希复核后执行。删除状态库不影响 5 个发布文件，但下一次构建必须从头开始。

## 10. `manifest.json`

`manifest.json` 至少记录：

- 数据集名称和版本；
- `schema_version`；
- `release_flavor`；
- 每个来源的版本、revision、许可证和用途；
- 4 个数据文件的行数、大小和 SHA-256；`manifest.json` 不记录自身哈希；
- train、validation、benchmark 的任务数量；
- strong、support、weak 的数量；
- 独立任务数、状态数、单证据动作数、双证据动作数和候选数；
- 义务类型、witness group 大小、关系类别和 STOP 标签分布；
- 教师模型、Prompt 版本、调用数量、一致率、规则拒绝率和人工抽检结果；
- 唯一文件版本数、snapshot-file 成员关系数和正文去重率；
- 去重和身份冲突统计；
- split 防泄漏审计；
- 文件和 Evidence Unit 引用完整率；
- 构建命令、随机种子和工具版本。

## 11. 发布硬门槛

以下任一条件不满足时禁止发布：

- 三个任务文件存在重复 `task_id`；
- `task_group_id` 跨 split；
- 模型可见输入中包含 Gold；
- validation 或 benchmark 存在轨迹；
- benchmark 任务或派生任务进入 train；
- 任一 mandatory obligation 没有可解析的 witness group；
- 任一正 STOP 状态仍存在未完成的 mandatory obligation；
- 任一负 STOP 状态已经完成全部 mandatory obligations；
- 重复、包含或相同内容证据被标成强 `complement`；
- 教师输出引用不存在、跨 snapshot 或未提供给教师的 `evidence_id`；
- teacher-only 标签进入 benchmark 主要指标；
- ContextBench overlay 被重复计为新任务；
- SWE-Explore 的 `meta.num_read_core` 被用作真实列表长度；
- SWE-Explore 的整文件读取被无界展开成 Cross-Encoder 正文；
- corpus 中存在重复 `file_version_id` 或重复 `repo + path + blob_oid`；
- 展开成员关系后，同一 `snapshot_id + path` 命中多个文件版本；
- Git tree 中的 `path + blob_oid` 与 corpus 成员关系不一致；
- `blob_oid`、文件正文和 `content_sha256` 校验不一致；
- `evidence_id` 引用完整率低于 100%；
- manifest 记录的任一数据文件哈希与实际文件不一致；
- 高严重度身份冲突进入任一任务文件；
- benchmark 的 split 或 membership 未冻结；
- 任何临时文件或未完成阶段被标记为正式 release。

## 12. 已确认决策

- 原始来源保留自身格式；内部派生状态统一存入单个 SQLite，正式发布使用 Parquet。
- 最终发布包含 5 个文件。
- 完整数据集由一条命令生成。
- 最终构建只使用 `scripts/build_unified_dataset.py`，不新增配置文件或分阶段脚本。
- 数据处理和实验阶段使用 JSONL，正式发布转换为 Parquet。
- 构建过程只保留一个可断点续跑的 SQLite 状态库，不发布独立中间 JSONL。
- 5 个最终文件通过临时目录流式生成，并在完整性审计后原子发布。
- train、validation、benchmark 提前物理分开。
- 三个任务文件共用同一个 Schema。
- 使用 `input` 与 `supervision` 隔离模型输入和答案。
- 一行代表一个去重任务。
- 多来源通过 provenance、监督和轨迹合并到任务。
- validation 和 benchmark 不包含轨迹。
- benchmark 支持内部完整版和对外脱敏版。
- `repository_corpus.parquet` 每行代表一个唯一文件版本，唯一键为 `repo + path + blob_oid`。
- 多个代码快照通过 `snapshot_ids` 共享相同文件版本，文件正文只保存一次。
- Evidence Unit 作为嵌套行号范围保存，训练时按行号展开。
- import 声明随文件版本保存，snapshot 相关的目标路径在加载时解析。
- SWE-bench 是唯一任务基准，ContextBench 和 SWE-Explore 只补充能够严格对齐的监督。
- 采用语义义务、AND witness group 和 group 间 OR 的方案 C。
- 评分采用完成度 `C(K)` 和进度 `P(K)` 两个原始分量，STOP 使用严格 mandatory 覆盖规则。
- 动作空间包含单证据、双证据和 STOP；双证据关系支持互补、替代、冗余、独立、冲突和未知。
- 标签不足时使用受约束教师模型，但完成度、进度、动作增益和 STOP 始终由程序计算。

## 13. 冻结数据源与实测规模

### 13.1 数据源范围

第一版只使用以下来源：

| 来源 | 角色 | 是否创建新任务 | 对齐要求 |
|------|------|----------------|----------|
| SWE-bench | 唯一任务基准、问题、commit、patch 和测试 | 是 | 不适用 |
| ContextBench | Gold context 和细粒度强证据 | 否 | 必须可靠映射到 SWE-bench `instance_id` |
| SWE-Explore | 多成功轨迹共识区域、可选区域和读取顺序 | 否 | 必须精确命中 SWE-bench `instance_id` |
| repository snapshot | 修复前完整仓库语料 | 否 | 必须匹配 SWE-bench `repo + base_commit` |

来源缺失时，单脚本从官方固定 revision 下载。下载顺序先取任务元数据和 ID，完成对齐后才准备仓库快照，避免下载最终会被拒绝的非对齐任务。

### 13.2 当前冻结数据的实测规模

| 项目 | 数量 |
|------|-----:|
| SWE-bench train | 19,008 |
| SWE-bench dev | 225 |
| SWE-bench test | 2,294 |
| SWE-bench 总任务 | 21,527 |
| ContextBench 原始主任务 | 1,136 |
| ContextBench 严格对齐任务 | 351 |
| SWE-Explore 原始任务 | 848 |
| SWE-Explore 严格对齐任务 | 451 |
| ContextBench 与 SWE-Explore 交集 | 283 |
| 两类强来源并集 | 519 |

519 个强来源任务中，38 个属于 SWE-bench train，481 个属于 SWE-bench test。不得为了增加训练标签而把 481 个 test 任务迁移到 train。训练标签不足时使用受约束教师标注 SWE-bench train，而不是破坏 benchmark 隔离。

### 13.3 ContextBench 文件角色

`data/raw/contextbench/full.parquet` 是唯一主任务表。其他 Parquet 和 CSV 只承担成员关系、选择状态、split 或 Gold context variant，不新增任务：

| 文件 | 角色 |
|------|------|
| `full.parquet` | 唯一主任务表 |
| `contextbench_verified.parquet` | 精选成员关系和监督 variant |
| `contextbench_verified_train.parquet` | 精选训练 overlay |
| `contextbench_verified_test.parquet` | 精选评测 overlay |
| `train.parquet` | 普通训练 overlay |
| `test.parquet` | 普通评测 overlay |
| `selected_500_instances.csv` | 选择状态和统计元数据 |

overlay 找不到 `full.parquet` 主记录时进入 SQLite 冲突表，不创建残缺任务。相同 variant 通过稳定 JSON 哈希去重；同键不同内容必须记录冲突。

## 14. 义务、Witness Group 与评分

### 14.1 固定义务类型

每个任务只生成确实适用且能够获得证据支持的义务。允许类型如下：

| 类型 | 含义 |
|------|------|
| `fault_location` | 故障发生的位置 |
| `fault_logic` | 错误机制或错误逻辑 |
| `dependency_context` | 调用、继承、导入等依赖约束 |
| `state_flow` | 状态、参数或数据如何流动 |
| `behavior_constraint` | 问题要求的正确行为 |
| `repair_scope` | 修复可能影响的范围 |
| `validation_constraint` | 测试、边界条件和回归约束 |

不是每个任务都必须拥有全部 7 类义务。无法可靠定义的义务不创建；不能为了让模板完整而补造义务。

### 14.2 方案 C 的逻辑语义

义务是“必须弄清楚的问题”，不是某个固定文件或 span。一个义务可以有多个 witness group：

```text
obligation
├── witness group 1: evidence A AND evidence B
└── witness group 2: evidence C
```

- 同一 group 内按 AND 解释。
- 同一义务下不同 group 按 OR 解释。
- `A + B` 形成互补关系。
- `[A, B]` 与 `C` 形成可替代路径。
- 重叠窗口、包含窗口或相同内容不能组成互补 group。

只有同时满足以下条件的义务才能设为 `mandatory=true`：

1. 义务适用于当前问题。
2. 至少存在一个能够解析到 repository corpus 的 witness group。
3. 定义来自确定性规则、跨来源一致、教师共识且通过规则验证，或人工审核。
4. 义务对理解故障或验证预期行为不可省略。

### 14.3 完成度

对义务 `r` 和当前状态 `K`：

```text
covered(r, K) = 1，当且仅当 r 的至少一个 witness group 完全包含于 K
covered(r, K) = 0，其他情况
```

强制义务完成度为：

```text
C(K) = 已完成的 mandatory obligations 数量 / mandatory obligations 总数
```

如果任务没有可靠的 mandatory obligation，则 `C(K)` 为 null，不能生成强 STOP 标签，相关状态标记为 `unknown`。

### 14.4 Witness 进度

只用完成度会把互补证据组的第一个证据错误标成零价值。因此同时保存进度：

```text
progress(r, K) = max_g |K ∩ g| / |g|
P(K) = 所有 applicable obligations 的 progress 平均值
```

例如 witness group 为 `[caller, callee]`：

- 两者都没有：进度为 0；
- 只有其中一个：进度为 0.5；
- 两者都有：进度为 1。

数据集保存 `C` 和 `P` 两个原始分量，不把人为权重固化为唯一真值。

### 14.5 动作增益

对当前状态 `K` 和动作 `A`：

```text
completion_gain = C(K ∪ A) - C(K)
progress_gain   = P(K ∪ A) - P(K)
```

训练需要单一排序值时，可以在 validation 上选择参数：

```text
utility = completion_gain + beta * progress_gain - gamma * token_cost
```

`beta` 和 `gamma` 属于训练配置或实验参数，不属于不可变 Gold。主要实验必须报告参数敏感性，并同时报告不扣成本和扣成本结果。

### 14.6 关系标签

关系优先由 witness graph 和确定性结构得到：

| 关系 | 判定原则 |
|------|----------|
| `complement` | 位于同一 AND group，任一单证据无法完成该义务 |
| `substitute` | 位于同一义务的不同 group，任一 group 都能独立满足义务 |
| `redundant` | 内容相同、区间包含、高度重叠或相同语义的重复表示 |
| `independent` | 分别推进不同义务，且不存在明显交互 |
| `conflict` | 版本、行为或约束指向矛盾结论 |
| `unknown` | 现有证据不足以可靠判断 |

候选共现、共同被轨迹读取或同时属于 Gold context，只能用于产生 pair 候选，不能直接决定 `complement`。

### 14.7 严格 STOP

STOP 使用已确认的严格规则：

```text
全部 mandatory obligations 已完成 -> STOP 可接受
任一 mandatory obligation 未完成   -> STOP 不可接受
义务定义或覆盖关系不可靠             -> STOP unknown
```

可选义务和 `strong_support_untyped` 可以用于相关性或排序训练，但不允许单独阻止或允许 STOP。成功轨迹结束也不能覆盖这一规则。

## 15. ContextBench 利用规则

### 15.1 数据审计结果

ContextBench 的每个 `gold_context` 项只有：

```text
file
start_line
end_line
content
```

它没有义务类型、必要性、替代性或证据关系标签。严格对齐的 351 个任务实测如下：

| 指标 | 数值 |
|------|-----:|
| Gold span 总数 | 2,207 |
| 每任务平均 span | 6.29 |
| 多 span 任务 | 349 |
| 多文件任务 | 174 |
| 存在区间重叠的任务 | 301 |
| 存在包含关系的任务 | 244 |
| 被其他 span 完全包含的 span | 398 |
| 合并重叠后每任务平均区间 | 4.51 |
| 与 patch 旧侧修改区间重叠的 span | 59.8% |
| 同时包含非 patch 上下文的任务 | 261 |
| 最大单 span 行数 | 880 |
| 空内容 span | 5 |

因此不能把每个 ContextBench span 直接当作一个 mandatory obligation。

### 15.2 映射与归一化

处理顺序如下：

```text
解析 gold_context
-> 校验路径、行号和正文
-> 映射修复前 snapshot
-> 合并完全重复和包含区间
-> 将过大区间切成 bounded Evidence Unit
-> 映射义务或保留为 untyped strong support
```

映射规则：

- 与 patch 旧侧修改区间重叠，可以支持 `fault_location`。
- 与测试断言或问题预期行为一致，可以支持 `behavior_constraint` 或 `validation_constraint`。
- 调用方、被调用方、导入定义可以支持 `dependency_context`。
- 状态读写和参数传递可以支持 `state_flow`。
- 无法可靠确定角色时标记为 `strong_support_untyped`，不得强制创建义务。
- 区间重叠、包含或内容相同优先产生 `redundant`，不得产生强互补标签。

ContextBench 决定“哪些区域是高质量证据候选”，不单独决定“任务有哪些证据义务”。

## 16. SWE-Explore 利用规则

### 16.1 字段语义

SWE-Explore 官方定义中，`read_core` 是所有成功修复轨迹共同读取的区域，`optional` 是部分模型读取的诊断上下文。第一版使用方式如下：

| 字段 | 用途 | 禁止解释 |
|------|------|----------|
| `read_core_regions` | 高置信度 witness 候选、定位正例 | 不能直接全部设为 mandatory |
| `read_core_files` | 文件级候选召回 | 不能把整文件送入 Cross-Encoder |
| `read_optional_regions_map` | 替代 witness、困难正例 | 不能当负例 |
| `modified_core_files` | 离线构造 `fault_location` 和 `repair_scope` | 不能进入在线输入 |
| `main_files` | 强故障位置锚点 | 不能代表完整证据集 |
| `read_step_info` | 构造轨迹前缀和下一动作候选 | 不能证明语义必要性或 STOP |
| `meta` | 仅作来源参考 | core 数量必须重新计算 |

官方说明：https://github.com/Qiushao-E/SWE-Explore-Bench

### 16.2 本地数据审计

严格对齐的 451 个任务全部来自 SWE-Explore `verified`，每个任务有 3 条成功轨迹：

| 指标 | 数值 |
|------|-----:|
| 平均 core 文件 | 3.29 |
| 平均 core region | 3.33 |
| 多 core region 任务 | 430 |
| 多读取路径任务 | 432 |
| 每条轨迹平均读取事件 | 16.52 |
| 整文件读取事件比例 | 61.5% |
| 存在 optional context 的任务 | 410 |

`meta.num_read_core` 在 252/451 个任务中与实际数组长度不一致，`meta.num_read_core_regions` 在 253/451 个任务中不一致。构建器必须以真实数组为准，重新计算统计。

### 16.3 监督强度

```text
SWE-Explore core ∩ ContextBench Gold
    -> strong typed witness 候选

SWE-Explore core ∩ patch/main file
    -> strong fault_location witness 候选

SWE-Explore core only
    -> trajectory_consensus_support

SWE-Explore optional
    -> alternative_support

只在 read_step_info 中出现
    -> behavioral_support
```

283 个 ContextBench 与 SWE-Explore 交集任务提供最高质量的跨来源一致监督，但全部属于 benchmark，只能用于评测和标签校准，不能用于训练。

### 16.4 轨迹前缀

按 `traj_path` 分组并按 `step_idx` 排序读取事件：

```text
K0 = empty
K1 = 第一次有效读取后状态
K2 = 前两次有效读取后状态
...
```

处理约束：

- 合并对同一 Evidence Unit 的重复读取，保留 `first_step` 和 `visit_count`。
- 有界读取映射到一个或多个 bounded Evidence Unit。
- `end=-1` 的整文件读取只记录 `visited_file`，不展开为全部文件正文。
- 实际下一读取只提供 `semantic_useful` 的行为支持，不能自动视为 `policy_acceptable=true`。
- 其他能够推进同一未完成义务的 witness 同样进入有用动作集合，再通过 Pareto 判断形成策略可接受集合。
- 若实际读取被更低成本且增益不低的动作支配，保留轨迹 provenance，但不能把被支配动作作为主要排序正例。
- 轨迹末尾只有在全部 mandatory obligations 已完成时才能产生正 STOP。

## 17. 受约束教师标注

### 17.1 采用范围

教师标注只补足以下缺口：

- ContextBench 或 SWE-Explore 证据无法确定语义角色；
- SWE-bench train/dev 缺少完整义务图；
- pair 无法区分互补、替代和独立；
- 某状态没有可靠的 `policy_acceptable` 动作；
- 需要构造跨文件困难正例或困难负例。

不对 21,527 个任务和全仓库候选做无差别教师调用。候选必须先由 patch、ContextBench、SWE-Explore、Retriever 或结构关系缩小到有界集合。

### 17.2 参考 Nips 的原则

Nips 项目使用 LLM 对已有 Gold supporting evidence 做受限角色分类，再由程序根据覆盖增益和成本选择动作，而不是让 LLM 无约束地产生最终训练标签：

- https://github.com/lmy020520/Nips/blob/da41359/scripts/rebuild_hotpotqa_targets_v2.py
- https://github.com/lmy020520/Nips/blob/da41359/scripts/build_hotpotqa_teacher_select_v2.py

本项目沿用“语义判断交给教师、数值标签交给程序”的边界。

### 17.3 教师输入

教师每次只接收一个任务的受控包：

```text
problem_statement
候选 Evidence Unit 的路径、符号、行号和 bounded 正文
候选之间的确定性结构关系
ContextBench/SWE-Explore/patch/test 的离线来源信号
固定义务类型、关系枚举和 JSON Schema
```

patch、test patch 和 Gold 信号只能用于离线标注，不得复制到 `input`、状态正文或候选正文。

### 17.4 教师输出

教师只能：

- 从固定 7 种义务类型中选择；
- 引用输入中真实存在的 `evidence_id`；
- 使用 AND group 和 group 间 OR；
- 针对明确的 `obligation_id` 从固定关系枚举中选择；
- 为同一个 pair 输出多条不同 obligation 的关系记录；
- 输出 `unknown`；
- 给出结构化置信度和简短依据。

教师不得：

- 创建新的代码正文或不存在的 Evidence Unit；
- 直接给出最终 `C`、`P`、动作增益或 STOP；
- 直接聚合 `relation_targets`；该多标签目标必须由程序根据义务级记录和置信度生成；
- 把未提供的仓库文件加入 witness；
- 覆盖确定性冲突标签。

### 17.5 一致性与验证

每个难例进行 3 次独立判断：

| 一致性 | 标签来源 | 处理 |
|--------|----------|------|
| 3/3 | `teacher_consensus_strong` | 通过规则验证后可进入 train |
| 2/3 | `teacher_consensus` | 通过规则验证后降低权重进入 train |
| 低于 2/3 | `unknown` | 不参与对应监督损失 |
| 与确定性标签冲突 | `conflict` | 不覆盖规则，进入人工抽检 |

程序验证至少包括：

- 所有 Evidence Unit 存在且属于同一 snapshot；
- 教师只能引用 Prompt 中提供的 ID；
- AND group 不得由重复、包含或相同内容证据组成；
- mandatory obligation 至少拥有一个有效 witness group；
- 关系标签与 witness graph 一致；
- 输出满足严格 JSON Schema；
- Prompt、模型 revision、输入哈希和原始响应可从 SQLite 审计。

### 17.6 Split 使用边界

- train：允许规则验证后的教师共识标签。
- validation：教师标签必须与确定性信号一致，或经过固定种子人工校准；teacher-only 状态不用于选择主要阈值。
- benchmark：teacher-only 标签不进入主要指标，只能用于辅助分析；主要真值必须来自确定性规则、跨来源一致或人工确认。

## 18. 状态与候选动作构造

### 18.1 状态来源

第一版支持 3 类状态：

| 状态类型 | 构造方式 | 可信度 |
|----------|----------|--------|
| `initial` | `K = empty` | 确定性 |
| `gold_prefix` | 按 witness 或可靠轨迹前缀逐步加入证据 | strong/support |
| `controlled_corruption` | 从充分证据集中删除一个 witness 或加入已验证冗余项 | support |

不得通过随机加入任意文件制造“困难状态”。受控扰动必须保留状态来源和被删除、加入的 Evidence Unit。

### 18.2 在线候选与离线标签候选分离

候选分离发生在 repository corpus 和 Evidence Unit 构建完成之后、状态与动作标签生成之前。它是候选可见性边界，不增加发布文件，也不引入第二个训练模型。

在线候选只允许使用真实运行时可见的信息：

```text
online_candidates = retrieve(q, K, pre_fix_snapshot)
```

在线候选可以来自：

- BM25 等冻结词法检索；
- 文件路径、文件名和问题关键词匹配；
- 符号定义与引用匹配；
- 从已召回单元扩展出的调用、导入、继承和读写关系；
- 按固定规则从高优先级单证据构造的有限 pair；
- STOP。

在线候选生成禁止读取参考补丁、测试答案、ContextBench Gold、SWE-Explore core/optional、witness group、教师解释或人工标签。在线 pair 也不能依据 Gold witness 关系构造。

离线标签候选用于确定训练真值和补足正例，可以使用：

- SWE-bench patch 的 pre-fix 映射位置；
- 严格对齐的 ContextBench 和 SWE-Explore 证据；
- obligation witness group；
- 规则验证后的教师共识或人工标注；
- 已验证的替代、冗余、独立、冲突和困难负例。

训练状态的物理候选集合为：

```text
training_candidates =
    online_candidates
    union missing_positive_injections
    union controlled_hard_negatives
    union STOP
```

在线已召回动作设置 `candidate_scope=online`。只因离线 Gold 或对照构造而加入的动作设置 `candidate_scope=offline_injected`。STOP 设置为 `candidate_scope=stop`。`candidate_scope`、`candidate_sources` 和 `online_retrieval_rank` 只用于采样、审计与分轨评测，不能出现在 Cross-Encoder 的文本输入中。

离线注入保证召回器漏掉的高质量正例仍能参与 Ranker 训练，但不得伪装成在线召回成功。validation 和 benchmark 的 Oracle 候选轨道允许注入；端到端在线轨道必须删除所有 `offline_injected` 动作后重新执行候选构造与排序。

所有候选必须属于当前 `snapshot_id`。未被 Gold 选中只表示未知，只有满足明确反证规则的单元才能标为 hard negative。

### 18.3 单证据与双证据动作

每个状态先生成单证据动作，再生成有限的双证据动作。禁止对全部候选做 `O(n^2)` 枚举。在线 pair 和离线注入 pair 必须分别构造：

- 在线 pair：只从在线高优先级单证据以及调用、导入、继承、读写、同文件相邻等可见结构关系生成；
- 离线注入 pair：允许依据 witness group、跨来源证据和已验证关系补入训练所需的正例与对照。

离线双证据候选优先级：

1. 同一 AND witness group。
2. 同一义务的不同替代 group。
3. caller/callee、writer/reader、implementation/test 等结构对。
4. ContextBench/SWE-Explore 共现但关系尚未确认的 pair。
5. 重叠、包含和内容重复的冗余对照。

只有前 3 类在证据充分时可以产生强关系标签；第 4 类默认 `unknown`，除非教师共识或人工确认。离线 Gold 产生的 pair 必须设置 `candidate_scope=offline_injected`，不能因为其成员曾被在线召回就把 Gold 配对关系伪装为在线构造结果。

### 18.4 语义有用性与策略可接受性

动作标签必须区分“提供信息”和“当前值得执行”。`semantic_useful` 只描述动作相对于当前状态 `K` 是否产生语义增益：

```text
semantic_useful(A | K) =
    completion_gain > 0
    或 completion_gain = 0 且 progress_gain > 0
```

`semantic_useful=true` 不等于动作应被模型优先选择。例如，替代证据 `[u]` 已能完成某个义务时，`[u, v]` 仍具有正语义增益，但如果没有比 `[u]` 完成更多内容，就不应为重复正文支付额外 Token 成本。

对于同一状态中的两个非 STOP 证据动作 `A` 和 `B`，若满足：

```text
completion_gain(B) >= completion_gain(A)
progress_gain(B)   >= progress_gain(A)
token_cost(B)      <= token_cost(A)
```

且至少一项严格不等，则 `B` Pareto 支配 `A`。此时：

```text
pareto_dominated(A) = true
dominated_by_action_ids(A) 包含 B.action_id
```

证据尚不充分，即 `C(K) < 1` 时：

```text
policy_acceptable(A | K) =
    semantic_useful(A | K)
    且 pareto_dominated(A) = false
```

同一状态允许多个非支配正确动作，数据集不强迫只有一个正动作。Pareto 判断不预设 `beta` 和 `gamma`，因此不会在构造 Gold 时提前固定信息增益与 Token 成本之间的标量权衡；训练阶段可以使用 validation split 选择这些系数。

STOP 不参与证据动作之间的 Pareto 比较。严格 STOP 规则优先于上述公式：

- `C(K) < 1`：STOP 的 `policy_acceptable=false`；
- `C(K) = 1`：STOP 的 `policy_acceptable=true`，所有继续获取证据的动作均为 false；
- `C(K)` 无法可靠计算：相关策略标签为 null，后续通过损失掩码排除。

STOP 不增加语义证据，因此在可判定状态下固定为 `semantic_useful=false`，`pareto_dominated=null`。证据动作即使因成本被支配，也必须保留其真实 `semantic_useful`，不能被错误改写为语义负例。

### 18.5 双证据交互增益

双证据的总增益不能直接表示两条证据是否产生组合价值。每个 pair 动作必须显式记录相对于当前状态 `K` 的二阶交互量：

```text
I_C(u, v | K) =
    completion_gain([u, v] | K)
    - completion_gain([u] | K)
    - completion_gain([v] | K)

I_P(u, v | K) =
    progress_gain([u, v] | K)
    - progress_gain([u] | K)
    - progress_gain([v] | K)
```

等价的反事实形式为：

```text
I_C(u, v | K) =
    C(K union {u, v})
    - C(K union {u})
    - C(K union {v})
    + C(K)
```

`I_P` 使用 `P` 按相同方式计算。数据集只保存 `completion_interaction=I_C` 和 `progress_interaction=I_P`，不复制保存 4 份反事实状态。

交互值的主要解释为：

| 取值 | 数值含义 | 关系证据 |
|------|----------|----------|
| `I > 0` | 组合增益大于单项增益之和 | 支持互补 |
| `I` 约等于 0 | 贡献基本可加，或存在单侧无增益证据 | 可能独立或单侧冗余 |
| `I < 0` | 两条证据存在价值重叠或边际收益递减 | 支持替代或冗余 |

数值比较使用固定容差 `epsilon=1e-6`，但存储值不得提前离散化。同一个 `[u, v]` 在不同 `K` 下可以拥有不同交互值，因此交互量属于状态—动作标签，不能提升为全局 pair 属性。

交互值不能机械替代关系标签：负值无法独立区分 substitute 与 redundant，零值无法独立区分 independent 与单侧 redundant，conflict 通常需要语义判断。最终关系仍由 witness graph、确定性结构规则、教师共识或人工标注确定。

用于交互监督的 pair 必须在同一状态中同时保留 `[u]`、`[v]` 和 `[u, v]` 三个动作，并且 3 个动作的 `completion_gain` 与 `progress_gain` 均可计算。条件不满足时两个交互字段为 null，不能参与交互数值监督。

### 18.6 状态与义务条件化的多标签关系

双证据关系不是全局 pair 属性。同一个 `[u, v]` 可以针对不同 obligation 同时具有不同关系，也可以随当前状态 `K` 改变。关系监督必须保存在具体 `policy_state.candidate_actions` 内，并按 obligation 展开：

```json
{
  "relations": [
    {
      "obligation_id": "obl_state_flow",
      "relation": "complement",
      "confidence": 1.0,
      "label_source": "witness_graph",
      "annotation_ids": []
    },
    {
      "obligation_id": "obl_validation",
      "relation": "substitute",
      "confidence": 0.9,
      "label_source": "teacher_consensus",
      "annotation_ids": ["ann_123"]
    }
  ],
  "relation_targets": {
    "complement": 1.0,
    "substitute": 0.9,
    "redundant": null,
    "independent": null,
    "conflict": null
  }
}
```

`relation_targets` 是固定字段的多标签目标，不设置 `primary_relation`。每个已确认类别的目标值取该类别义务级记录中的最大置信度；被可靠排除的类别为 `0.0`；证据不足、未判断或仅有 `unknown` 记录的类别为 null。null 不等于负例，训练时必须屏蔽。

关系辅助头对 5 个类别分别使用 Sigmoid，而不是在互斥类别上使用 Softmax。同一个 pair 可以同时预测 complement 和 substitute，因为两者可能对应不同 obligation。`unknown` 不作为模型需要预测的第 6 个类别，只表示对应关系监督不可用。

义务级关系只描述当前状态下 pair 对该义务的作用：

- 同一 AND witness group 中缺一不可的证据支持 complement；
- 同一义务不同 OR witness group 中可独立满足义务的证据支持 substitute；
- 相同内容、包含关系或已被 `K` 覆盖的边际贡献支持 redundant；
- 分别推进不同义务且无显著交互的证据支持 independent；
- 对同一义务给出不可同时成立的事实或约束时支持 conflict。

结构关系和交互值只能提供候选标签或一致性检查，不能替代 obligation 语义。例如，`I<0` 可以支持 substitute 或 redundant，但不能单独区分二者。

## 19. 唯一训练模型：Cross-Encoder Evidence Policy Ranker

### 19.1 模型定义

最终只训练一个面向软件仓库问题定位的、状态条件化且支持证据组合的 Cross-Encoder Evidence Policy Ranker。它不是代码生成模型，也不是只计算“问题—文件相关性”的静态 Reranker。模型学习在当前证据状态下，下一步应获取哪条证据、哪组证据，或者是否停止。

模型的核心函数为：

```text
s_A = f_theta(q, K, A)
```

- `theta`：唯一一套可训练参数。
- `q`：SWE-bench 问题描述。
- `K`：当前已获取的有界证据集合。
- `A`：单证据 `[u]`、双证据 `[u, v]` 或 `STOP`。
- `s_A`：动作效用排序分数，分数越高表示当前越值得执行。

模型采用预训练 Transformer Encoder 作为 Backbone，将 `q`、`K` 和 `A` 拼接后联合编码。Cross-Encoder 允许问题、已有证据和候选证据在同一注意力空间内交互，因此能够判断候选的增量价值，而不只是静态相关性。

### 19.2 单模型约束

“只训练一个模型”具有以下明确含义：

- 只产生一套参数和一个最终检查点；
- 单证据、双证据和 STOP 共用同一个 Backbone 与动作排序头；
- 不训练独立 Retriever、pair 选择器、STOP 分类器或另一个 Evidence Policy；
- 教师模型只参与离线标注，不属于训练产物或部署系统；
- 候选召回、结构扩展和有限 pair 构造均为不可训练的确定性过程；
- pair relation 辅助头与排序头共享 Backbone，保存在同一个检查点中，不构成第二个模型。

`completion_gain`、`progress_gain`、Token 成本和严格 STOP 条件首先用于构造监督信号，不要求分别训练预测模型。最终决策始终以统一的动作效用分数为准。

### 19.3 输入渲染

Cross-Encoder 的逻辑输入为：

```text
[QUESTION]
problem_statement

[CURRENT EVIDENCE]
K 中按固定顺序渲染的有界 Evidence Unit

[CANDIDATE ACTION]
单个 Evidence Unit、两个 Evidence Unit，或 [STOP]
```

初始状态为 `K = empty`，因此第一次评分等价于：

```text
f_theta(q, empty, A)
```

后续步骤必须显式输入 `K`，使同一候选在不同证据状态下可以得到不同分数。候选正文来自 `repository_corpus` 中长度受控的 Evidence Unit；完整仓库、Gold witness、参考补丁、测试答案和教师解释均不得进入在线模型输入。

双证据动作 `[u, v]` 作为一个整体动作输入模型。训练时可以随机交换 `u` 与 `v` 的渲染顺序，但其规范化 `action_id` 不变，防止模型学习固定位置偏差。

### 19.4 统一输出与动作选择

动作排序头对每个候选动作输出一个标量：

```text
score(q, K, [u])
score(q, K, [u, v])
score(q, K, STOP)
```

同一状态下的所有动作必须由同一个模型分别评分。系统执行：

```text
A_star = argmax_A score(q, K, A)
```

若 `A_star` 为单证据或双证据，系统将对应 Evidence Unit 加入 `K`，然后再次调用同一个模型；若 `A_star = STOP`，证据获取结束。STOP 不是另一个二分类模型，而是与证据动作竞争的特殊动作。

对于双证据动作，关系辅助头通过 5 个独立 Sigmoid 输出 complement、substitute、redundant、independent 和 conflict 的多标签概率。多个类别可以同时为真；unknown 通过目标值 null 和损失掩码表达，不作为第 6 个输出类别。关系输出用于辅助损失和评测，不替代主排序分数，也不单独决定动作。单证据和 STOP 不计算关系损失。

### 19.5 训练方式

训练样本按状态组织，而不是把每个候选当成互不相关的全局二分类样本。一个状态包含相同的 `(q, K)` 和多个候选动作：

```text
(q, K, {A_1, A_2, ..., A_n})
```

主目标是在同一状态内提高有增量价值动作的分数，降低无关、冗余、冲突动作和过早 STOP 的分数；证据充分时则提高 STOP 的分数。一个状态允许存在多个正确动作，因此主损失采用支持多正例的 listwise ranking loss。关系分类作为辅助损失，与排序任务共享同一个 Backbone：

```text
L_total = L_rank + lambda_relation * L_relation
```

`L_relation` 使用带掩码的多标签 Binary Cross-Entropy，只在拥有可靠义务级关系标签的双证据动作及非 null 类别上计算；其余动作通过 `relation_loss_mask=false` 屏蔽。`lambda_relation` 由 validation split 选择，不能使用 benchmark 调参。

第一版训练按同一个模型检查点逐步加入更难样本，不在各阶段分别训练不同模型：

```text
单证据初始状态预热
-> 状态感知单证据排序
-> 高置信度双证据动作
-> STOP 与关系联合校准
```

### 19.6 模型与外围系统的边界

模型只输出候选动作分数和 pair relation 多标签概率。以下工作属于外围系统，不属于模型本身：

- 从仓库语料中生成初始候选；
- 根据路径、符号、导入和调用关系扩展候选；
- 从高优先级单证据中构造有限双证据动作；
- 根据模型分数执行动作并更新 `K`；
- 运行测试、生成补丁或判断补丁是否解决任务。

因此，最终部署对象是一个被迭代调用的 Cross-Encoder Ranker；完整系统可以包含不可训练的检索与控制逻辑，但不会增加第二个训练模型。

## 20. 质量评估与最小试验

### 20.1 标签统计

每次构建必须分别报告：

```text
independent_task_count
state_count
single_action_count
pair_action_count
candidate_count
unique_snapshot_count
unique_repo_count
```

禁止把状态数、pair 数或候选数表述为独立 SWE 任务数量。

### 20.2 固定审计样本

正式扩展教师标签前，先构造 320 个反事实证据包：

```text
20 个任务 × 4 个候选 pair × 4 个证据包
```

每个 pair 检查：

```text
K
K + u
K + v
K + u + v
```

试验至少覆盖互补、替代、冗余和独立关系，并检查：

- `completion_gain` 和 `progress_gain` 是否符合 witness graph；
- 互补 pair 是否存在单独不完成、组合后完成的案例；
- 替代证据在一个 group 已满足后，另一个 group 的边际价值是否下降；
- 重复证据是否被错误赋予增益；
- 教师一致率和规则拒绝率是否可接受；
- 人工复核是否能依据原始证据重现标签。

如果试验无法观察到稳定的正交互、替代衰减或严格 STOP，不能扩展 pair 数据，也不能把证据交互作为主要实验结论。

### 20.3 发布前人工抽检

固定随机种子抽检：

- 100 个确定性或跨来源强监督任务；
- 100 个教师共识训练任务；
- 100 个跨文件 pair；
- 100 个 `unknown`、冲突或规则拒绝任务。

人工抽检记录只进入 SQLite 和 manifest 统计，不新增最终发布文件。

## 21. 端到端数据流

```text
冻结 SWE-bench split
-> 严格对齐 ContextBench 与 SWE-Explore
-> 准备修复前 repository snapshot
-> 构建唯一文件版本与 bounded Evidence Unit
-> 映射 patch、ContextBench 和 SWE-Explore 信号
-> 构建语义义务与 witness group
-> 对缺口执行受约束教师标注
-> 程序计算 C、P、动作增益、关系和 STOP
-> 构造 train / validation / benchmark 状态与候选动作
-> 完整性、泄漏和人工抽检
-> 输出 JSONL 实验版或 Parquet 正式版
```

任何阶段失败时保留旧 release，从 SQLite 最近的兼容阶段继续。最终文件只在所有硬门槛通过后原子发布。
