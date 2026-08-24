"""根据 Issue 与当前 Evidence State 规划下一轮仓库检索。"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

from .retrieval_plan import EVIDENCE_DIMENSIONS, RETRIEVAL_CHANNELS, RetrievalPlan


def render_current_evidence(
    current_evidence: Sequence[Mapping[str, Any]],
) -> str:
    """把已经获取的 Evidence 渲染为规划模型可见状态。"""

    if not current_evidence:
        return "[EMPTY]"
    return "\n\n---\n\n".join(
        "\n".join(
            (
                f"Evidence ID：{unit['evidence_id']}",
                f"位置：{unit['path']}:{unit['start_line']}-{unit['end_line']}",
                f"类型：{unit['unit_type']}",
                f"符号：{unit.get('qualified_name') or unit.get('symbol') or ''}",
                str(unit["content"]),
            )
        )
        for unit in current_evidence
    )


def build_planner_prompt(
    *,
    issue: str,
    current_evidence: Sequence[Mapping[str, Any]],
    retrieval_channels: Sequence[str] = RETRIEVAL_CHANNELS,
) -> str:
    """构造只负责检索规划、不选择具体 Evidence 的提示。"""

    dimensions = ", ".join(EVIDENCE_DIMENSIONS)
    channels = ", ".join(retrieval_channels)
    state = render_current_evidence(current_evidence)
    return f"""你是软件仓库 Evidence Agent 的检索规划器。你的任务是根据 Issue 和当前已经获取的 Evidence，决定下一轮应该通过 RAG 检索什么内容。

职责边界：
1. 只输出检索计划，不选择具体 Evidence，不判断 STOP。
2. 不能读取或猜测 Gold Patch、Test Patch、Gold obligation、Witness 或 Teacher answer。
3. K 为空时仅根据 Issue 规划首次检索。
4. K 非空时根据 Issue 与当前 Evidence 判断仍缺少什么信息，并规划新的检索。
5. RAG 会自动排除所有历史上已经返回过的 Evidence，无需在查询中重复旧内容。
6. structure 通道依赖 K；K 为空时必须选择 content、path 或 symbol 中至少一个通道。

可选证据维度：{dimensions}
可选检索通道：{channels}

Issue：
{issue}

当前 Evidence State (K)：
{state}

只输出一个 JSON 对象，不要输出 Markdown：
{{
  "information_gap": "本轮需要补充的信息",
  "target_dimensions": ["从可选证据维度中选择"],
  "queries": ["用于内容检索的查询"],
  "paths": ["已知或推测需要检索的仓库路径"],
  "symbols": ["需要检索的代码符号"],
  "retrieval_channels": ["从可选检索通道中选择"]
}}"""


class RetrievalPlanner:
    """调用一个冻结规划模型生成结构化 RetrievalPlan。"""

    def __init__(
        self,
        call_model: Callable[[str], str | Mapping[str, Any]],
        *,
        retrieval_channels: Sequence[str] = RETRIEVAL_CHANNELS,
    ) -> None:
        self.call_model = call_model
        self.retrieval_channels = tuple(map(str, retrieval_channels))

    def plan(
        self,
        *,
        issue: str,
        current_evidence: Sequence[Mapping[str, Any]],
    ) -> RetrievalPlan:
        """根据当前状态生成下一轮检索计划。"""

        prompt = build_planner_prompt(
            issue=issue,
            current_evidence=current_evidence,
            retrieval_channels=self.retrieval_channels,
        )
        result = self.call_model(prompt)
        payload = json.loads(result) if isinstance(result, str) else result
        return RetrievalPlan.from_mapping(payload)
