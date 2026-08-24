# Agents：有状态检索规划 Agent

本目录实现单 Agent 的在线轨迹控制。Agent 的职责是决定“下一轮该检索什么”，不是直接挑选具体证据。

- 当 K 为空时，根据 Problem Statement 与公开 hints 生成首轮 `RetrievalPlan` 并调用 RAG。
- 当 K 非空时，根据统一在线问题与当前 K 规划新的检索内容。
- 每轮把全部返回候选写入 `retrieved_ids` 和候选账本，后续 RAG 不再返回这些证据。
- 未被选中的历史候选保留在账本中，Policy 后续仍可选择，但它们不会被重复检索。
- 训练好的 Evidence Policy 从本轮候选动作中选择 `Single`、`Pair` 或 `STOP`。

## 文件说明

| 文件 | 作用 |
|---|---|
| `__init__.py` | 导出 Agent、状态和检索计划接口。 |
| `retrieval_plan.py` | 定义七类目标维度、检索通道和结构化 `RetrievalPlan`。 |
| `planner.py` | 将 Issue 或 Issue + K 交给规划模型，解析本轮检索计划。 |
| `state.py` | 保存 K、历史检索 ID、候选账本、本轮候选、预算和终止状态。 |
| `rollout.py` | 执行 `Plan → RAG → Policy → Update` 完整循环。 |
| `README.md` | 说明 Agent 边界、状态和运行方式。 |

Planner 始终接收 K 中全部 Evidence 的 ID、路径、类型、符号和行号元数据。正文使用独立上下文预算，默认优先保留最近获取 Evidence 的 8192 Token；超出预算的正文显示为省略标记，避免长轨迹无限扩大提示。该预算由实验入口的 `--planner-evidence-body-token-budget` 配置。

## Agent 模式

```text
Issue + K + retrieved_ids
          ↓
   Retrieval Planner
          ↓
   RetrievalPlan → RAG
          ↓
       新候选动作
          ↓
 Evidence Policy 选择动作
          ↓
  更新 K 或输出 STOP
```

这不是多智能体协作，也不是让 Agent 用规则替代 Policy。Planner 只输出查询、路径、符号、目标维度和检索通道；它不能输出证据 ID，不能读取 Gold，也不能决定 `STOP`。

规划模型通过一个可调用对象注入。该对象接收中文提示词，返回 JSON 对象或 JSON 字符串：

```python
from src.agents import RetrievalPlanner

planner = RetrievalPlanner(call_model=my_json_model)
```

完整 rollout 还需要 `RepositoryRAG`、`ActionBuilder` 和已加载 checkpoint 的 `EvidencePolicy`。模型训练未完成时，可以完成接口与静态验证，但不能据此运行正式评测。
