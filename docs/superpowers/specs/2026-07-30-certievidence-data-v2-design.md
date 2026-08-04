# CertiEvidence 数据流水线 v2 设计

## 1. 目标

本方案将现有候选级 MVP 重构为多源、分层监督、严格防泄漏的数据流水线。最终正式发布仍保持 3 个核心文件：

```text
data/release_v2/
├── benchmark.jsonl
├── repository_corpus.parquet
└── manifest.json
```

第一版以 Python 软件仓库为主。统一 schema 保留 `language` 字段，SWE-rebench V2 的其他语言先进入注册表，只有通过对应语言抽取器和质量门槛后才进入正式 release。

## 2. 非目标

- 不在数据构建阶段训练 Retriever、Reranker 或 Evidence Policy。
- 不把 Gold patch、Gold context 或成功轨迹直接作为在线 Agent 输入。
- 不把候选 pair 数量表述为独立任务数量。
- 不在 v2 验收前删除或覆盖当前 `data/processed`、`data/release` 和 Git 缓存。
- 不把合成任务、模型轨迹和人工审核数据混成同一可信度的 Gold 标签。

## 3. 数据源分层

| 层级 | 数据源 | 角色 | 默认可信度 | 默认用途 |
|------|--------|------|------------|----------|
| 强监督 | ContextBench | 文件、span、symbol 级相关上下文 | strong | 定位训练、充分性评测 |
| 强监督 | SWE-bench Oracle | Gold 修改文件对应上下文 | strong，但有 Gold 派生属性 | 离线定位标签、Oracle 上界 |
| 中等监督 | SWE-bench patch/test patch | 修改位置、测试、修复范围 | support | Retriever/Reranker 训练 |
| 检索基线 | SWE-bench BM25 13K/27K/40K | 不同预算的检索结果 | observed | baseline、hard-negative mining |
| 轨迹监督 | SWE-Explore | 真实或整理后的探索轨迹 | support/weak | 读取顺序、状态转移 |
| 轨迹监督 | Nebius SWE-agent trajectories | 80K 成功/失败轨迹 | weak | 行为模仿、失败反例、STOP |
| 可执行监督 | SWE-Gym | 真实任务和测试环境 | support | 行为价值、修复器校准 |
| 规模扩展 | SWE-rebench V2 | 大规模真实可执行任务 | support | 扩充独立训练任务 |
| 弱监督 | SWE-smith | 合成故障及专家轨迹 | weak | 预训练、策略 warm-up |
| 外部评测 | MULocBench | 代码与非代码位置 | eval_only | 泛化评测 |
| 外部方法 | SweLoc 构建流程 | Issue 到函数和 hard negatives | method_reference | 借鉴负例挖掘，不直接混入 Gold |

每条来源记录必须带：

```text
source_name
source_version
source_revision
source_record_id
license
trust_tier
visibility
ingested_at_utc
raw_record_sha256
```

## 4. ContextBench 去重方案

ContextBench 只生成 1,136 个主任务，不再把目录中的 7 个文件全部作为独立任务扫描。

### 4.1 文件角色

| 文件 | 新角色 |
|------|--------|
| `full.parquet` | 唯一主任务表 |
| `contextbench_verified.parquet` | 精选 500 条成员关系，不新增任务 |
| `contextbench_verified_train.parquet` | 精选任务的高优先级 `gold_context` overlay |
| `contextbench_verified_test.parquet` | 精选任务的高优先级 `gold_context` overlay |
| `train.parquet` | 保留原始 `gold_context` variant，不新增任务 |
| `test.parquet` | 保留原始 `gold_context` variant，不新增任务 |
| `selected_500_instances.csv` | 选择状态、难度和统计元数据，不新增任务 |

### 4.2 合并规则

以 `instance_id` 为第一键，以 `repo + base_commit + original_inst_id` 为一致性校验键：

1. 从 `full.parquet` 创建任务。
2. 将 `contextbench_verified.parquet` 标记为 `selected_500=true`。
3. 将 verified train/test 的 `gold_context` 记录为首选 variant。
4. 将普通 train/test 的不同 `gold_context` 保留为替代 variant。
5. CSV 字段写入 `source_metadata`，不得补造缺失的问题描述或仓库。
6. overlay 找不到主任务时进入 quarantine，不创建残缺任务。
7. 同一 variant 重复时比较稳定 JSON 哈希；内容不同时产生冲突记录并停止该任务的发布。

## 5. 统一实体模型

### 5.1 四层身份

```text
source_record_id
    ↓ 多来源别名
canonical_instance_id
    ↓ 同一 Issue/PR 的不同表示
task_group_id
    ↓ 多任务可共享同一修复前版本
snapshot_id
```

- `source_record_id`：数据源内一条原始记录。
- `canonical_instance_id`：统一后的一个任务表示。
- `task_group_id`：同一 Issue、PR 或任务谱系的泄漏隔离单元。
- `snapshot_id`：唯一绑定规范化后的 `repo + resolved_commit`。

### 5.2 禁止的错误合并

`repo + base_commit` 只能说明任务共享代码快照，不能单独证明是同一任务。v2 不再仅凭这一条件合并实例。

### 5.3 合并优先级

从强到弱依次为：

1. 数据源显式 alias 或 `original_inst_id`。
2. 完全一致的 Issue URL。
3. 完全一致的 PR URL。
4. `repo + PR number`。
5. `repo + base_commit + patch_sha256`。
6. `repo + base_commit + normalized_problem_sha256`。

弱规则产生的匹配必须记录 `match_method` 和 `match_confidence`。存在冲突时不自动跨来源合并。

## 6. Evidence Unit 身份

### 6.1 内容单元

```text
content_unit_id = hash(
    blob_oid,
    extractor_name,
    extractor_version,
    unit_type,
    start_line,
    end_line,
    content_sha256
)
```

保存代码正文及内容属性，同一 Git blob 的相同抽取结果只保存一次。

### 6.2 逻辑单元

```text
logical_unit_id = hash(
    snapshot_id,
    file_path,
    content_unit_id
)
```

保存某个内容单元在特定修复前快照中的逻辑身份。所有监督标签最终只能引用 `logical_unit_id`。

### 6.3 结构边

第一版支持：

```text
contains
same_file
imports
calls
inherits
reads
writes
raises
handles
test_covers
```

所有边必须满足源和目标属于同一 `snapshot_id`。无法可靠分析的语言只生成 `contains`、`same_file` 和固定行块关系。

## 7. 在线与离线可见性

### 7.1 在线字段

```text
canonical_instance_id
snapshot_id
repo
resolved_commit
problem_statement
language
candidate_scope
```

### 7.2 离线字段

```text
patch
test_patch
gold_context
oracle_context
trajectory
generated_patch
eval_logs
support_annotations
obligations
witness_groups
certificate_families
```

发布脚本采用允许列表写入在线字段，而不是依赖禁止列表删除字段。

## 8. 目录设计

```text
configs/
└── data_sources_v2.json

data/
├── raw/
│   ├── swebench/
│   ├── swebench_retrieval/
│   ├── contextbench/
│   ├── swe_explore/
│   ├── nebius/
│   ├── swe_gym/
│   ├── swe_rebench_v2/
│   ├── swe_smith/
│   └── mulocbench/
├── cache/
│   └── repos/
├── v2/
│   ├── registry/
│   ├── splits/
│   ├── snapshots/
│   ├── corpus/
│   ├── labels/
│   ├── trajectories/
│   ├── training/
│   └── reports/
└── release_v2/
```

原始下载文件不可原地修改。每个源目录保存 `SOURCE.json`，记录远程 revision、下载时间、文件哈希、许可证和数据卡 URL。

## 9. 处理流水线

### 阶段 0：冻结 v1

输入：当前 Git 状态和 `data/`。

输出：

```text
data/v2/reports/v1_inventory.json
data/v2/reports/v1_file_hashes.jsonl
```

验收：

- 不修改任何 v1 文件。
- 记录当前未提交文件、大小、mtime 和 SHA-256。
- 后续 v2 输出全部写入新目录。

### 阶段 1：数据源注册与审计

输入：`configs/data_sources_v2.json` 和 `data/raw`。

输出：

```text
data/v2/registry/source_manifest.jsonl
data/v2/reports/source_audit.json
data/v2/reports/source_schema.json
```

验收：

- 每个启用的数据文件都有 SHA-256、revision 和 license。
- ContextBench 只有 `full.parquet` 被标记为 `task_table`。
- 缺失必需文件时停止，不静默跳过。

### 阶段 2：来源专用规范化

每个来源使用独立 adapter，不再用字段关键词猜测所有来源。

输出：

```text
data/v2/registry/source_records.jsonl
data/v2/registry/source_overlays.jsonl
data/v2/registry/rejected_source_records.jsonl
```

验收：

- 原始记录数 = 规范化记录数 + overlay 数 + rejected 数。
- 每条记录保留来源和原始哈希。
- `patch`、`gold_context` 和轨迹默认标记为 `label_only`。

### 阶段 3：身份解析和 Master Registry

输出：

```text
data/v2/registry/master_instances.jsonl
data/v2/registry/source_aliases.jsonl
data/v2/registry/identity_conflicts.jsonl
data/v2/registry/quarantined_instances.jsonl
```

验收：

- `canonical_instance_id` 唯一。
- 不存在仅由 `repo + base_commit` 合并的任务。
- 所有自动合并记录均有证据和置信度。
- 高严重度冲突任务不得进入 release。

### 阶段 4：冻结 split

上游官方评测 split 优先于本项目自定义 split：

- SWE-bench test、Verified、Lite：`eval_only`。
- MULocBench：`eval_only`。
- 与上述任务发生身份重叠的派生记录继承 `eval_only`。
- SWE-bench train：可进入训练。
- SWE-rebench V2、SWE-Gym：按 `task_group_id` 分组，并保留 repo-disjoint 内部测试。
- SWE-smith：只进入 `train_weak`。

输出：

```text
data/v2/splits/split_assignments.jsonl
data/v2/splits/split_lock.json
data/v2/reports/split_audit.json
```

验收：

- `task_group_id` 跨 split 泄漏数为 0。
- eval 仓库进入 repo-disjoint 训练集的数量为 0。
- 同一原始 Issue/PR 的派生数据全部继承同一 split。

### 阶段 5：Git 快照准备

沿用 partial bare clone 和批量 fetch，缓存键为规范化 repo 名。

输出：

```text
data/v2/snapshots/git_snapshots.jsonl
data/v2/snapshots/snapshot_failures.jsonl
data/v2/reports/snapshot_report.json
```

验收：

- 每个可发布任务唯一解析到一个 `snapshot_id`。
- `snapshot_id → repo + resolved_commit` 为一对一。
- 未解析任务进入 quarantine。
- 不因 inventory 操作触发全仓库 blob 下载。

### 阶段 6：完整仓库语料

执行 inventory、extract、export 三阶段：

```text
Git tree inventory
→ 唯一 blob 抽取
→ content_unit 去重
→ snapshot_unit 归属
→ repository_corpus.parquet
```

内部输出：

```text
data/v2/corpus/build_state.sqlite3
data/v2/corpus/repository_corpus.parquet
data/v2/reports/corpus_report.json
data/v2/reports/corpus_failures.jsonl
```

验收：

- 每个 `snapshot_unit` 引用存在的 `content_unit`。
- 所有结构边 snapshot-local。
- 相同 blob 的正文只物理保存一次。
- 语料选择不读取 patch、gold context 或 trajectory。
- 状态库为空时 extract 必须失败。

### 阶段 7：监督映射

按以下优先级将标签映射到 `logical_unit_id`：

1. 精确路径 + 精确行区间。
2. 精确路径 + symbol。
3. 精确路径 + 内容哈希。
4. 重命名检测后的路径 + 行区间。
5. 模糊内容匹配，标记低置信度。

分别处理：

```text
ContextBench gold_context
SWE-bench Oracle
patch/test patch
SWE-Explore
Nebius trajectories
SWE-Gym execution
SWE-smith synthetic labels
```

输出：

```text
data/v2/labels/evidence_annotations.jsonl
data/v2/trajectories/normalized_events.parquet
data/v2/reports/mapping_report.json
data/v2/reports/unmapped_annotations.jsonl
```

验收：

- Oracle 文件映射率不低于 99%。
- ContextBench 文件映射率不低于 98%，span 映射率不低于 90%。
- 所有映射都保留 provenance 和 mapping method。
- 模糊映射不得标记为 `strong_support`。

### 阶段 8：证据图与参考证据全集

合并多源标签，但保留各来源独立判断：

```text
strong_support
support
weak_support
unresolved
contradicted
```

输出：

```text
data/v2/labels/reference_evidence.jsonl
data/v2/labels/evidence_graph.parquet
```

规则：

- patch 修改位置可支持 fault location，但不能单独证明依赖或状态义务。
- 成功轨迹读取过某单元，不等于该单元必需。
- 失败轨迹读取单元可作为行为反例，不自动成为语义负例。
- 未被 Gold 选中的候选不能直接标记为负例。

### 阶段 9：义务、Witness Group 与证书

义务类型：

```text
fault_location
fault_logic
dependency_context
state_flow
behavior_constraint
repair_scope
validation_constraint
```

每个义务包含：

```text
applicable
mandatory
witness_groups
construction_method
confidence
uncertainty
```

只在所有 mandatory obligations 被覆盖时生成 certificate。保留多组等价证书，不只保存单一最小集合。

### 阶段 10：训练样本

内部训练资产不进入三文件正式 release：

```text
data/v2/training/retrieval_pairs.parquet
data/v2/training/reranker_lists.parquet
data/v2/training/policy_states.parquet
data/v2/training/behavior_values.parquet
```

- Retriever：query、positive set、snapshot-local hard negatives。
- Reranker：同一 snapshot 的候选列表和 set-valued labels。
- Policy：证据前缀、可接受的下一单元集合、STOP/EXPAND/ABSTAIN。
- Behavior value：固定修复器在受控证据包上的测试结果。

训练统计必须同时报告：

```text
independent_task_count
state_count
pair_count
candidate_count
unique_snapshot_count
unique_repo_count
```

### 阶段 11：质量审计

硬性失败条件：

- 任意 `task_group_id` 跨 split。
- 任意在线字段包含 patch、gold context、trajectory 或 labels。
- 任意标签引用不存在的 `logical_unit_id`。
- 任意候选跨 `snapshot_id`。
- 任意发布文件哈希与 manifest 不一致。
- 任意高严重度身份冲突进入 release。

警告条件：

- 自动认证率超过 95%。
- 单单元证书比例超过 50%。
- 任务正例比例中位数超过 30%。
- 缺少至少 10 个 snapshot-local 候选的任务超过 5%。
- 某个来源的标签覆盖率或映射率显著低于历史版本。

另外抽取固定种子的人工审计样本：

```text
100 个强监督任务
100 个轨迹任务
100 个跨文件任务
100 个失败或 unresolved 任务
```

### 阶段 12：发布

`benchmark.jsonl` 每行一个任务，明确分离 `input` 与 `labels`。

`repository_corpus.parquet` 包含：

```text
content_unit
snapshot_unit
structure_edge
```

`manifest.json` 包含：

```text
数据源与 revision
schema 和构建版本
文件 SHA-256
独立任务、snapshot、repo 和候选规模
split 统计
映射与失败统计
许可证摘要
可见性规则
泄漏审计
已知局限
人工审计结果
```

## 10. 版本与可复现性

- 数据版本使用 `major.minor.patch`。
- schema 不兼容变更提升 major。
- 数据源或标签新增提升 minor。
- 仅修复错误记录提升 patch。
- 所有 ID 由规范化字段和版本化算法确定性生成。
- 每次构建写入 Git commit、Python 版本、依赖版本和命令参数。
- SQLite 只作为内部断点状态，不作为正式数据发布。

## 11. 迁移顺序

1. 冻结并校验当前 v1。
2. 实现 source manifest 和 ContextBench 文件角色规则。
3. 接入 SWE-bench Oracle/BM25，验证 21,527 条对齐。
4. 重建 v2 Master Registry 和 split。
5. 复用现有 Git 缓存，重新生成 v2 snapshots。
6. 完成完整仓库 corpus。
7. 映射 ContextBench、Oracle、patch 和 SWE-Explore。
8. 接入 Nebius 轨迹并映射读取事件。
9. 接入 SWE-Gym 和 SWE-rebench V2 的 Python 子集。
10. 生成证据图、训练样本和质量报告。
11. 构建并验收 `release_v2`。
12. v2 稳定后再决定是否归档 v1 大文件。

## 12. 成功标准

- 21,527 条 SWE-bench 的 Oracle/BM25 监督完成对齐。
- ContextBench 只生成 1,136 个任务且所有 variant 有明确 provenance。
- 完整 corpus 可按 `snapshot_id` 独立检索。
- 正式评测任务不进入训练。
- 标签引用完整率为 100%。
- 在线字段泄漏数为 0。
- 自动证书不再出现当前候选集合近乎全部为 witness 的塌缩。
- 最终 3 个发布文件可由 manifest 中记录的输入和命令重复生成。
