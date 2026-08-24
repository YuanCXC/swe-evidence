"""面向最终证据包的参考修复约束大模型语义评审器。"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence


SEMANTIC_DIMENSIONS = (
    "fault_location",
    "fault_logic",
    "dependency_context",
    "state_flow",
    "behavior_constraint",
    "repair_scope",
    "validation_constraint",
)
SEMANTIC_JUDGE_PROMPT_VERSION = "2.0"


def _parse_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", maxsplit=1)[1].rsplit("```", maxsplit=1)[0]
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("语义 Judge 必须输出一个 JSON 对象")
    return payload


def _validate_judgment(result: Mapping[str, Any]) -> None:
    verdicts = {"sufficient", "partial", "insufficient"}
    if result.get("sufficiency_verdict") not in verdicts:
        raise ValueError("语义 Judge 的 sufficiency_verdict 非法")
    categorical_fields = {
        "causal_correctness": {"correct", "partial", "incorrect", "uncertain"},
        "execution_relevance": {"relevant", "partial", "irrelevant", "uncertain"},
        "repair_support": verdicts,
    }
    for field, choices in categorical_fields.items():
        if result.get(field) not in choices:
            raise ValueError(f"语义 Judge 的 {field} 非法")
    dimensions = result.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise ValueError("语义 Judge 的 dimensions 必须是 JSON 对象")
    for name in SEMANTIC_DIMENSIONS:
        dimension = dimensions.get(name)
        if not isinstance(dimension, Mapping):
            raise ValueError(f"语义 Judge 缺少维度 {name}")
        if dimension.get("coverage") not in {"sufficient", "partial", "none"}:
            raise ValueError(f"语义 Judge 维度 {name} 的 coverage 非法")
        if not isinstance(dimension.get("applicable"), bool) or not isinstance(
            dimension.get("critical"), bool
        ):
            raise ValueError(f"语义 Judge 维度 {name} 的布尔字段非法")
    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        raise ValueError("语义 Judge 的 confidence 必须位于 [0, 1]")
    for field in (
        "useful_evidence_ids",
        "irrelevant_evidence_ids",
        "redundant_evidence_ids",
        "misleading_evidence_ids",
        "missing_requirements",
    ):
        if not isinstance(result.get(field), list):
            raise ValueError(f"语义 Judge 字段 {field} 必须是 JSON 数组")


def render_evidence_package(evidence_package: Sequence[Mapping[str, Any]]) -> str:
    blocks = []
    for unit in evidence_package:
        blocks.append(
            "\n".join(
                (
                    f"证据 ID：{unit['evidence_id']}",
                    f"位置：{unit['path']}:{unit['start_line']}-{unit['end_line']}",
                    f"类型：{unit['unit_type']}",
                    str(unit["content"]),
                )
            )
        )
    return "\n\n---\n\n".join(blocks)


def build_semantic_judge_prompt(
    *,
    issue: str,
    evidence_package: Sequence[Mapping[str, Any]],
    gold_patch: str,
    test_patch: str,
) -> str:
    dimensions = ", ".join(SEMANTIC_DIMENSIONS)
    evidence = render_evidence_package(evidence_package)
    return f"""你是软件修复证据充分性评审器。你只评价给定 Evidence Package 是否包含理解并支持参考修复所需的信息，不生成补丁，也不评价方法名称。

判定原则：
1. Gold Patch 与 Test Patch 仅用于离线核对真实修复语义。
2. sufficient 表示证据覆盖了理解参考修复所需的关键位置、原因、行为和约束。
3. partial 表示包含部分关键证据，但仍缺少会影响正确理解或修复规划的信息。
4. insufficient 表示核心位置、因果、执行路径或约束缺失，或证据具有误导性。
5. 相关但可删除的内容标为 redundant；与参考修复无关的内容标为 irrelevant；包含错误因果或错误执行语义的内容标为 misleading。

七个评价维度：{dimensions}

问题描述：
{issue}

证据包：
{evidence}

参考 Gold Patch：
{gold_patch}

参考 Test Patch：
{test_patch}

只输出一个 JSON 对象，不要 Markdown。输出结构：
{{
  "sufficiency_verdict": "sufficient | partial | insufficient",
  "dimensions": {{
    "fault_location": {{"applicable": true, "critical": true, "coverage": "sufficient | partial | none", "reason": "简体中文理由"}},
    "fault_logic": {{"applicable": true, "critical": true, "coverage": "sufficient | partial | none", "reason": "简体中文理由"}},
    "dependency_context": {{"applicable": true, "critical": false, "coverage": "sufficient | partial | none", "reason": "简体中文理由"}},
    "state_flow": {{"applicable": true, "critical": false, "coverage": "sufficient | partial | none", "reason": "简体中文理由"}},
    "behavior_constraint": {{"applicable": true, "critical": true, "coverage": "sufficient | partial | none", "reason": "简体中文理由"}},
    "repair_scope": {{"applicable": true, "critical": true, "coverage": "sufficient | partial | none", "reason": "简体中文理由"}},
    "validation_constraint": {{"applicable": true, "critical": false, "coverage": "sufficient | partial | none", "reason": "简体中文理由"}}
  }},
  "causal_correctness": "correct | partial | incorrect | uncertain",
  "execution_relevance": "relevant | partial | irrelevant | uncertain",
  "repair_support": "sufficient | partial | insufficient",
  "useful_evidence_ids": [],
  "irrelevant_evidence_ids": [],
  "redundant_evidence_ids": [],
  "misleading_evidence_ids": [],
  "missing_requirements": [],
  "reason": "简体中文总体理由",
  "confidence": 0.0
}}"""


def judge_evidence_package(
    call_model: Callable[[str], str],
    *,
    issue: str,
    evidence_package: Sequence[Mapping[str, Any]],
    gold_patch: str,
    test_patch: str,
) -> dict[str, Any]:
    prompt = build_semantic_judge_prompt(
        issue=issue,
        evidence_package=evidence_package,
        gold_patch=gold_patch,
        test_patch=test_patch,
    )
    result = _parse_json_object(call_model(prompt))
    _validate_judgment(result)
    result["evidence_count"] = len(evidence_package)
    return result
