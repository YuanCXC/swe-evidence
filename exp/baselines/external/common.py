"""外部方法冻结输出的公共映射逻辑。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data import RuntimeRepository
from src.evaluation import adapt_outputs


def task_keys(task: Mapping[str, Any]) -> list[str]:
    """返回内部任务 ID 及其外部来源 ID。"""

    keys = [str(task["task_id"])]
    keys.extend(
        str(item["source_id"])
        for item in task.get("provenance") or []
        if item.get("source_id")
    )
    return list(dict.fromkeys(keys))


def verify_snapshot(
    task: Mapping[str, Any],
    external_task_id: str,
    metadata: Mapping[str, Any],
) -> bool:
    """核对外部产物中可用的 base_commit。"""

    base_commit = metadata.get("base_commit")
    if base_commit and str(base_commit) != str(task["input"]["base_commit"]):
        raise ValueError(f"{external_task_id} 的 base_commit 与当前任务快照不一致")
    return bool(base_commit)


def flatten_strings(value: Any) -> list[str]:
    """按原顺序展开官方多样本输出，并去除重复字符串。"""

    flattened: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            flattened.append(item)
        elif isinstance(item, list):
            for member in item:
                visit(member)

    visit(value)
    return list(dict.fromkeys(flattened))


def adapt_external_outputs(
    repository: RuntimeRepository,
    task: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    """将外部路径、符号或行号输出映射为冻结 Evidence Units。"""

    snapshot_files = {
        str(item["path"]): item
        for item in repository.list_snapshot_files(str(task["snapshot_id"]))
    }
    evidence_units = []
    for path in dict.fromkeys(str(item["path"]) for item in outputs):
        file_record = snapshot_files.get(path)
        if file_record is not None:
            evidence_units.extend(
                repository.get_file_evidence(
                    str(file_record["file_version_id"]),
                    scoreable_only=True,
                )
            )
    adapted = adapt_outputs(outputs, evidence_units)
    return [
        {
            **unit,
            "retrieval_score": float(
                unit.get("source_score")
                if unit.get("source_score") is not None
                else 1.0 / int(unit["source_rank"])
            ),
            "retrieval_sources": [source],
            "retrieval_source_ranks": {source: int(unit["source_rank"])},
        }
        for unit in adapted
    ]


def external_result(
    task: Mapping[str, Any],
    external_task_id: str,
    outputs: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    snapshot_verified: bool,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """构造统一结果，同时保留外部方法可观测的原生成本。"""

    evidence_ids = [str(unit["evidence_id"]) for unit in candidates]
    evidence_tokens = sum(
        int(unit.get("rendered_token_count") or 0) for unit in candidates
    )
    mapped_ranks = {int(unit["source_rank"]) for unit in candidates}
    result = {
        "task_id": str(task["task_id"]),
        "snapshot_id": str(task["snapshot_id"]),
        "external_task_id": external_task_id,
        "evidence_package": list(candidates),
        "final_evidence_ids": evidence_ids,
        "final_evidence_tokens": evidence_tokens,
        "retrieved_evidence_ids": evidence_ids,
        "external_outputs": [dict(item) for item in outputs],
        "external_output_count": len(outputs),
        "mapped_external_output_count": len(mapped_ranks),
        "external_mapping_rate": len(mapped_ranks) / len(outputs) if outputs else 0.0,
        "snapshot_verified": snapshot_verified,
        "execution_cost_observed": bool(metadata.get("execution_cost_observed")),
        "trajectory_observed": False,
        "retrieval_rounds": [
            {"round": 0, "candidate_evidence_ids": evidence_ids}
        ],
        "steps": [
            {
                "step": 0,
                "action_type": "stop",
                "action_id": "external_output_stop",
                "added_evidence_ids": evidence_ids,
                "added_token_count": evidence_tokens,
                "tool_calls": 0,
                "forced": True,
                "termination_reason": "external_native_output",
            }
        ],
        "termination_reason": "external_native_output",
        "planner_calls": 0,
    }
    result.update(
        {
            key: value
            for key, value in metadata.items()
            if value is not None and key != "execution_cost_observed"
        }
    )
    return result


def read_jsonl(path: Path | str) -> Iterable[dict[str, Any]]:
    """读取官方 UTF-8 JSONL 产物。"""

    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)
