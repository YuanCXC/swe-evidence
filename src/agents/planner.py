"""根据 Issue 与当前 Evidence State 规划下一轮仓库检索。"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

from .retrieval_plan import EVIDENCE_DIMENSIONS, RETRIEVAL_CHANNELS, RetrievalPlan


PLANNER_PROMPT_VERSION = "2.0"


def _state_body_ids(
    current_evidence: Sequence[Mapping[str, Any]],
    body_token_budget: int,
) -> set[str]:
    """在预算内优先保留最近获取 Evidence 的正文。"""

    selected: set[str] = set()
    remaining = body_token_budget
    for unit in reversed(current_evidence):
        token_count = int(unit.get("rendered_token_count") or 0)
        if token_count <= remaining:
            selected.add(str(unit["evidence_id"]))
            remaining -= token_count
    return selected


def render_current_evidence(
    current_evidence: Sequence[Mapping[str, Any]],
    *,
    body_token_budget: int,
) -> str:
    """渲染全量元数据，并仅为预算内 Evidence 附带正文。"""

    if not current_evidence:
        return "[EMPTY]"
    body_ids = _state_body_ids(current_evidence, body_token_budget)
    return "\n\n---\n\n".join(
        "\n".join(
            (
                f"Evidence ID：{unit['evidence_id']}",
                f"位置：{unit['path']}:{unit['start_line']}-{unit['end_line']}",
                f"类型：{unit['unit_type']}",
                f"符号：{unit.get('qualified_name') or unit.get('symbol') or ''}",
                (
                    str(unit["content"])
                    if str(unit["evidence_id"]) in body_ids
                    else "[正文因 Planner 上下文预算省略]"
                ),
            )
        )
        for unit in current_evidence
    )


def build_planner_prompt(
    *,
    issue: str,
    current_evidence: Sequence[Mapping[str, Any]],
    retrieval_channels: Sequence[str] = RETRIEVAL_CHANNELS,
    evidence_body_token_budget: int = 8192,
) -> str:
    """构造只负责检索规划、不选择具体 Evidence 的提示。"""

    dimensions = ", ".join(EVIDENCE_DIMENSIONS)
    channels = ", ".join(retrieval_channels)
    state = render_current_evidence(
        current_evidence,
        body_token_budget=evidence_body_token_budget,
    )
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
        evidence_body_token_budget: int = 8192,
    ) -> None:
        self.call_model = call_model
        self.retrieval_channels = tuple(map(str, retrieval_channels))
        self.evidence_body_token_budget = evidence_body_token_budget

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
            evidence_body_token_budget=self.evidence_body_token_budget,
        )
        result = self.call_model(prompt)
        if isinstance(result, str):
            text = result.strip()
            if text.startswith("```") and text.endswith("```"):
                text = text.split("\n", maxsplit=1)[1].rsplit("```", maxsplit=1)[0]
            payload = json.loads(text)
        else:
            payload = result
        if not isinstance(payload, Mapping):
            raise ValueError("Planner 必须输出一个 JSON 对象")
        return RetrievalPlan.from_mapping(payload)
