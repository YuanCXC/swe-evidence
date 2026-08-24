# Data 数据访问层

本目录为 Evidence Agent 提供统一的数据读取接口。数据层只读取冻结数据，不重新构建数据集，不重新抽取 Evidence Units，也不修改 SQLite 或 Parquet 文件。

最重要的设计目标是隔离在线数据与离线 Gold：

```text
在线 Retriever / Policy / Agent
只能使用 TaskReader 与 RuntimeRepository

状态级 Policy 训练和评价
使用 TaskReader 与 PolicyEvidenceReader

离线充分性评价 / Semantic Judge
显式使用 SupervisionReader
```

`SupervisionReader` 不从 `src.data` 默认导出，避免在线代码无意读取 Gold Patch、Test Patch、Obligation 或 Witness。

## 文件说明

| 文件 | 作用 | 数据来源 |
|---|---|---|
| `__init__.py` | 导出在线安全的 `TaskReader`、`RuntimeRepository`、`PolicyEvidenceReader` 和统一在线问题构造函数。不导出离线监督接口。 | — |
| `task_reader.py` | 读取 Issue、repo、snapshot、split 和实验资格等任务输入，并主动移除监督、轨迹和 Gold 字段；`build_online_issue` 统一拼接 Problem Statement 与公开 hints。 | `tasks.parquet` |
| `policy_evidence_reader.py` | 读取冻结 Policy states/actions 引用的 Evidence Unit 文本子集，供 Cross-Encoder 输入构造、状态级评价和消融使用。 | `policy_evidence.parquet` |
| `runtime_repository.py` | 只读访问完整修复前仓库，查询 snapshot、文件版本、已有 Evidence Units、文件级 FTS 结果和结构上下文。 | `repository_runtime.sqlite3` |
| `supervision_reader.py` | 读取 Gold Patch、Test Patch、七类 Evidence Obligation、OR-of-AND Witness、Policy states 和 Strong-Teacher 审计字段。只能用于训练、审计与离线评价。 | `tasks.parquet` |
| `README.md` | 说明数据边界、文件职责和主要数据流。 | — |

## 正式数据文件

数据层面向以下冻结 bundle：

```text
data/evidence_agent_dataset_v1/
├── tasks.parquet
├── policy_evidence.parquet
├── repository_runtime.sqlite3
└── manifest.json
```

### `tasks.parquet`

同时包含在线任务输入和离线监督，因此必须通过不同 Reader 隔离：

```text
TaskReader
└── Issue、repo、base_commit、snapshot_id、retrieval_scope

SupervisionReader
└── Gold、obligations、witnesses、policy_states、evaluation
```

### `policy_evidence.parquet`

包含冻结 Policy states/actions 引用的约 99.9 万个 Evidence Units。每行直接保存 Evidence 正文、路径、符号、行号和 Token 数，适合批量训练和状态级评价。

它不是完整仓库，不能作为在线 Retriever 的搜索空间。

### `repository_runtime.sqlite3`

包含完整在线仓库数据：

```text
snapshots
snapshot_file_memberships
file_versions
evidence_units
policy_file_fts
policy_file_fts_map
```

Runtime 中约有 2,549.6 万个 Evidence Units。文件级 FTS5 索引用于候选文件召回，Evidence 正文按 `file_versions.payload_json.content` 和单元的 1-based 行区间实时切片。

## 在线数据边界

在线代码允许读取：

- `task_id`、`task_group_id` 和 `snapshot_id`
- Issue `problem_statement` 与公开 hints
- repo、base commit、language 和 retrieval scope
- 修复前文件正文和已有 Evidence Units
- 文件路径、符号、imports 和冻结结构关系
- FTS 排名与在线候选

在线代码禁止读取：

- Gold Patch 与 Test Patch
- modified files
- Evidence labels
- Evidence Obligations 与 Witness Groups
- Policy action labels
- Strong-Teacher 输出
- benchmark 参考答案

## RuntimeRepository 接口

| 接口 | 作用 |
|---|---|
| `get_snapshot` | 读取冻结仓库快照元数据。 |
| `list_snapshot_files` | 列出快照中的路径和文件版本，不加载正文。 |
| `get_file` | 按 snapshot 与 path 读取完整文件。 |
| `get_file_version` | 按 `file_version_id` 读取完整文件版本。 |
| `search_files` | 使用现有 FTS5 索引查询当前 snapshot 中的相关文件。 |
| `get_evidence` | 按 `evidence_id` 读取并补全一个 Evidence Unit。 |
| `get_evidence_many` | 批量读取并补全 Evidence Units。 |
| `get_file_evidence` | 读取一个文件版本下的已有 Evidence Units。 |
| `get_structure_context` | 读取 parent、children、siblings、同文件单元和 imports。 |

`search_files` 只接受已经生成的查询词。Agent 在 `src/agents/` 中制定检索计划；Path、Symbol、Content、Structure 通道、RRF 融合和候选排序由 `src/retrieval/` 执行，不属于数据访问层。

## 使用示例

### 读取在线任务

```python
from src.data import TaskReader

reader = TaskReader("data/evidence_agent_dataset_v1/tasks.parquet")
task = reader.get_task("task_0004c3905bec5145332e53ba")
print(task["input"]["problem_statement"])
```

### 查询 Runtime

```python
from src.data import RuntimeRepository

with RuntimeRepository(
    "data/evidence_agent_dataset_v1/repository_runtime.sqlite3"
) as repository:
    files = repository.search_files(
        snapshot_id=task["snapshot_id"],
        repo=task["input"]["repo"],
        terms=["golden", "section", "search"],
        limit=20,
    )
```

### 显式读取离线监督

```python
from src.data.supervision_reader import SupervisionReader

reader = SupervisionReader("data/evidence_agent_dataset_v1/tasks.parquet")
references = reader.get_references("task_0004c3905bec5145332e53ba")
```

只有离线评价代码可以使用最后一种接口。
