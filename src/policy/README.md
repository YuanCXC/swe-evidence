# Policy：Evidence Policy 推理层

本目录负责把 RAG 返回的候选证据构造成与训练一致的动作，并由训练好的 Cross-Encoder checkpoint 选择具体证据或 `STOP`。

Policy 不生成检索计划，不再次检索，也不把检索分数混入模型分数。

## 文件说明

| 文件 | 作用 |
|---|---|
| `__init__.py` | 导出动作构造、输入渲染和模型推理接口。 |
| `actions.py` | 构造 `Single`、结构合法的 `Pair` 和 `STOP` 动作。 |
| `input_renderer.py` | 严格复现训练时的 `q / K / action` 输入格式与长度裁剪。 |
| `evidence_policy.py` | 加载 Cross-Encoder checkpoint，对动作输出统一标量分数并排序。 |
| `README.md` | 说明 Policy 的输入、动作空间和职责边界。 |

## 动作空间

- `Single(e)`：选择一个本轮新候选证据。
- `Pair(e1, e2)`：只构造 runtime 中可验证的父子关系或同文件相邻关系。
- `STOP`：由模型与证据动作一起评分，分数最高时结束轨迹。

动作构造会遵守剩余 Evidence Unit 和 token 预算。`Pair` 不是检索策略，也不会用手写语义规则猜测两条证据是否相关。

## 模型输入

`PolicyInputRenderer` 复现当前训练脚本的输入：

```text
Question:
{q}

Current Evidence State:
{K}

Candidate Action:
{Single / Pair / STOP}
```

默认最大长度为 4096 token，问题视图最多 2048 token；对每个候选动作分别计算剩余空间，保留全部 K 元信息，并按最近获得优先加入 K 的正文，直到达到模型长度上限。

## 加载 checkpoint

```python
from src.policy import EvidencePolicy

policy = EvidencePolicy(
    checkpoint_dir="models/evidence_policy/checkpoint-best",
)
```

正式 checkpoint 尚未训练完成时，不应启动模型评测。该目录也不存放 baseline；所有对比实验入口和实现应放在 `exp/`。
