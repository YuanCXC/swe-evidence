"""将官方 LocAgent 定位输出适配为统一 Evidence Package。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from src.data import RuntimeRepository

from .common import (
    adapt_external_outputs,
    external_result,
    flatten_strings,
    read_jsonl,
    task_keys,
    verify_snapshot,
)


def _parse_entity(identifier: str) -> dict[str, Any]:
    normalized = identifier.replace("\\", "/").removeprefix("./")
    if ".py:" in normalized:
        file_stem, symbol = normalized.split(".py:", maxsplit=1)
        return {
            "path": file_stem + ".py",
            "symbol": symbol,
            "external_id": identifier,
        }
    return {"path": normalized, "external_id": identifier}


def _trajectory_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    usage = row.get("usage")
    trajectories = (row.get("loc_trajs") or {}).get("trajs") or []
    messages = [
        message
        for trajectory in trajectories
        for message in trajectory.get("messages") or []
    ]
    api_calls = sum(1 for message in messages if message.get("role") == "assistant")
    tool_calls = sum(1 for message in messages if message.get("role") == "tool")
    elapsed = sum(float(trajectory.get("time") or 0) for trajectory in trajectories)
    observed = isinstance(usage, dict)
    prompt_tokens = float(usage.get("prompt_tokens") or 0) if observed else None
    completion_tokens = (
        float(usage.get("completion_tokens") or 0) if observed else None
    )
    return {
        "execution_cost_observed": observed,
        "api_prompt_tokens": prompt_tokens,
        "api_completion_tokens": completion_tokens,
        "api_total_tokens": prompt_tokens + completion_tokens if observed else None,
        "api_calls": float(api_calls) if messages else None,
        "external_agent_iterations": float(api_calls) if messages else None,
        "external_tool_calls": float(tool_calls) if messages else None,
        "external_elapsed_seconds": elapsed if trajectories else None,
    }


class LocAgentOutputStore:
    """读取 LocAgent 官方 loc_outputs 或 loc_trajs JSONL。"""

    def __init__(self, output_path: Path | str, *, level: str) -> None:
        self.path = Path(output_path).resolve()
        self.level = level
        self.rows = {
            str(row["instance_id"]): row for row in read_jsonl(self.path)
        }

    def task_key(self, task: Mapping[str, Any]) -> str | None:
        return next((key for key in task_keys(task) if key in self.rows), None)

    def contains(self, task: Mapping[str, Any]) -> bool:
        return self.task_key(task) is not None

    def get_for_task(
        self, task: Mapping[str, Any]
    ) -> tuple[str, list[dict[str, Any]], bool, dict[str, Any]]:
        task_key = self.task_key(task)
        if task_key is None:
            raise KeyError(str(task["task_id"]))
        row = self.rows[task_key]
        metadata = row.get("meta_data") or {}
        snapshot_verified = verify_snapshot(task, task_key, metadata)
        field = {
            "file": "found_files",
            "module": "found_modules",
            "function": "found_entities",
        }[self.level]
        identifiers = flatten_strings(row.get(field) or [])
        outputs = (
            [
                {"path": value.replace("\\", "/"), "external_id": value}
                for value in identifiers
            ]
            if self.level == "file"
            else [_parse_entity(value) for value in identifiers]
        )
        costs = {
            "external_level": self.level,
            **_trajectory_metrics(row),
        }
        return task_key, outputs, snapshot_verified, costs


class LocAgentBaseline:
    """映射 LocAgent 指定层级的原生最终输出。"""

    def __init__(
        self,
        repository: RuntimeRepository,
        outputs: LocAgentOutputStore,
    ) -> None:
        self.repository = repository
        self.outputs = outputs

    def run(self, task: Mapping[str, Any]) -> dict[str, Any]:
        external_task_id, outputs, snapshot_verified, costs = (
            self.outputs.get_for_task(task)
        )
        candidates = adapt_external_outputs(
            self.repository,
            task,
            outputs,
            source="locagent",
        )
        return external_result(
            task,
            external_task_id,
            outputs,
            candidates,
            snapshot_verified=snapshot_verified,
            metadata=costs,
        )
