# Unified SWE Dataset 发布结构与任务 Schema 设计

## 1. 目标

本设计定义 Unified SWE Dataset 的最终发布产物，以及训练、验证、评测三个任务文件的统一字段结构。

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
- 完整代码只保存在 `repository_corpus.parquet`，任务文件通过稳定 ID 引用。

## 2. 文件职责

| 文件 | 一行表示 | 用途 |
|------|----------|------|
| `train.parquet` | 一个可训练的统一任务 | 参数训练、行为模仿、弱监督预训练 |
| `validation.parquet` | 一个冻结的验证任务 | 调参、早停、阈值选择、回归验证 |
| `benchmark.parquet` | 一个正式评测任务 | 证据检索、充分性判断、代码修复评测 |
| `repository_corpus.parquet` | 一个 Evidence Unit | 为三个任务 split 提供共享代码语料 |
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

`evidence_labels` 中每项包含：

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `evidence_id` | string | `repository_corpus.parquet` 中的证据 ID |
| `relevance` | string | `positive`、`negative`、`unknown` |
| `granularity` | string | file、class、function、span 或 code_block |
| `source` | string | 标签来源 |
| `confidence` | float32 | 映射或人工标注置信度，范围为 `[0, 1]` |

`obligations` 中每项包含：

| 子字段 | 类型 | 含义 |
|--------|------|------|
| `obligation_id` | string | 任务内唯一义务 ID |
| `description` | string | 必须得到证据支持的判断 |
| `witness_evidence_ids` | list\<string> | 支持该义务的 Evidence Unit |

允许的 `training_targets`：

- `retrieval`
- `reranking`
- `evidence_policy`
- `repair`
- `trajectory_imitation`
- `failure_modeling`
- `test_generation`
- `pretraining`

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

`train.parquet` 用于模型参数训练。任务可以同时服务多个训练目标，无需为 Retriever、Reranker、Evidence Policy 和 Repair 分别复制行。

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
| `strong` | ContextBench、人工确认 Gold | 1.0 |
| `support` | SWE-bench patch、Oracle、真实可执行任务 | 0.7 |
| `weak` 且真实 | 自动映射、模型轨迹 | 0.4 |
| `weak` 且合成 | SWE-smith | 0.2 |

实际采样策略由训练配置控制，数据集只记录建议值。

### 4.3 来源合并

- SWE-bench 提供任务、仓库、commit 和 patch。
- ContextBench 为已有任务补充 Gold evidence，不新增任务。
- Oracle 提供离线文件级监督。
- BM25 提供 hard negatives 和 baseline，不写入 `input`。
- SWE-Explore 与 SWE-agent trajectories 挂到已有任务。
- SWE-rebench 和 SWE-Gym 在无对应任务时创建新任务。
- SWE-smith 创建 `weak` 合成任务，只能进入 train。

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

- validation 占可训练真实任务的 5%～10%。
- 优先使用时间切分和仓库隔离。
- 至少覆盖 20 个仓库。
- 不包含 SWE-smith。
- 不包含正式 benchmark 任务或其变体。
- 不包含与 benchmark 共享 `task_group_id` 的任务。

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
| 代码修复 | SWE-bench Verified、Lite、冻结 holdout | `patch_resolved`、测试通过率 |
| 证据定位 | ContextBench、Oracle、MULocBench | Recall@K、MRR、nDCG |
| 证据充分性 | ContextBench、人工审核样本 | obligation coverage、STOP accuracy |

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

每行表示一个 Evidence Unit：

| 字段 | 类型 | 含义 |
|------|------|------|
| `evidence_id` | string | 全局唯一证据 ID |
| `snapshot_id` | string | 所属修复前快照 |
| `repo` | string | 规范化仓库名 |
| `commit` | string | 已解析 commit |
| `path` | string | 仓库内相对路径 |
| `language` | string | 内容语言 |
| `unit_type` | string | file、class、function、span 或 code_block |
| `symbol` | string | 符号名；不适用时为 null |
| `start_line` | int32 | 起始行 |
| `end_line` | int32 | 结束行 |
| `content` | string | 完整文本 |
| `content_sha256` | string | 内容哈希 |

任何任务只能检索自身 `snapshot_id` 对应的 Evidence Unit。

## 9. `manifest.json`

`manifest.json` 至少记录：

- 数据集名称和版本；
- `schema_version`；
- `release_flavor`；
- 每个来源的版本、revision、许可证和用途；
- 5 个发布文件的行数、大小和 SHA-256；
- train、validation、benchmark 的任务数量；
- strong、support、weak 的数量；
- 去重和身份冲突统计；
- split 防泄漏审计；
- Evidence Unit 引用完整率；
- 构建命令、随机种子和工具版本。

## 10. 发布硬门槛

以下任一条件不满足时禁止发布：

- 三个任务文件存在重复 `task_id`；
- `task_group_id` 跨 split；
- 模型可见输入中包含 Gold；
- validation 或 benchmark 存在轨迹；
- benchmark 任务或派生任务进入 train；
- Evidence Unit 引用完整率低于 100%；
- manifest 文件哈希与实际文件不一致；
- 高严重度身份冲突进入任一任务文件；
- benchmark 的 split 或 membership 未冻结。

## 11. 已确认决策

- 中间数据使用 JSONL，正式发布使用 Parquet。
- 最终发布包含 5 个文件。
- train、validation、benchmark 提前物理分开。
- 三个任务文件共用同一个 Schema。
- 使用 `input` 与 `supervision` 隔离模型输入和答案。
- 一行代表一个去重任务。
- 多来源通过 provenance、监督和轨迹合并到任务。
- validation 和 benchmark 不包含轨迹。
- benchmark 支持内部完整版和对外脱敏版。
