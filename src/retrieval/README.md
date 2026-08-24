# Retrieval：检索计划执行层

本目录只执行 Agent 给出的 `RetrievalPlan`，不负责制定计划，也不代替训练模型选择证据或判断 `STOP`。

每一轮接收当前任务、Agent 生成的检索计划、历史已检索证据 ID 和当前已选证据集合 K。随后按计划调用 FTS、路径、符号、内容和结构通道，通过 RRF 融合排名，并在返回候选前排除所有历史已检索证据。七类证据维度是计划的目标空间，不是检索层自动生成的七组固定查询。

## 文件说明

| 文件 | 作用 |
|---|---|
| `__init__.py` | 导出统一入口 `RepositoryRAG`。 |
| `rag.py` | 执行结构化检索计划，协调各检索通道并过滤历史证据。 |
| `fts_retriever.py` | 使用 runtime 中的 FTS5 索引召回候选文件。 |
| `path_symbol_retriever.py` | 根据计划中的显式路径、符号和词项对证据单元排序。 |
| `unit_retriever.py` | 从候选文件读取可评分证据单元，并执行内容 BM25 排序。 |
| `structure_expander.py` | 从当前 K 扩展父子、同级、同文件邻域和导入关系。 |
| `rank_fusion.py` | 使用 Reciprocal Rank Fusion 合并不同通道的排名。 |
| `README.md` | 说明检索边界、执行流程与文件职责。 |

## 执行流程

```text
RetrievalPlan + 历史 retrieved_ids + 当前 K
                    ↓
      FTS / Path / Symbol / Content / Structure
                    ↓
                  RRF 融合
                    ↓
             排除所有历史已检索证据
                    ↓
              本轮新候选 Evidence Units
```

检索层只读取修复前任务字段和 `repository_runtime.sqlite3`，不读取 Gold Patch、Test Patch、Teacher supervision 或 Policy 标签。

## 使用示例

```python
from src.agents import RetrievalPlan
from src.data import RuntimeRepository, TaskReader
from src.retrieval import RepositoryRAG

task = TaskReader("data/evidence_agent_dataset_v1/tasks.parquet").get_task("task_id")
plan = RetrievalPlan.from_mapping(
    {
        "target_dimensions": ["fault_location", "fault_logic"],
        "queries": ["parser parse error", "invalid token branch"],
        "paths": ["src/parser.py"],
        "symbols": ["parse"],
        "retrieval_channels": ["fts", "path", "symbol", "content"],
        "reason": "先定位报错位置及其触发逻辑",
    }
)

with RuntimeRepository(
    "data/evidence_agent_dataset_v1/repository_runtime.sqlite3"
) as repository:
    candidates = RepositoryRAG(repository).retrieve(
        task,
        plan,
        exclude_evidence_ids=set(),
        current_evidence=[],
    )
```

候选如何组成 `Single`、`Pair`、`STOP`，以及最终选择哪个动作，属于 `src/policy/`。
