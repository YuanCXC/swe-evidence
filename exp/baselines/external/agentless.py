"""将官方 Agentless 定位输出适配为统一 Evidence Package。"""

from __future__ import annotations

import re
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


def _parse_locations(value: Any) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for member in item:
                visit(member)
            return
        if not isinstance(item, dict):
            return
        for path, locations in item.items():
            lines = []
            for text in flatten_strings(locations):
                lines.extend(text.splitlines())
            for location in lines:
                location = location.strip()
                line_match = re.fullmatch(r"line:\s*(\d+)", location)
                if line_match:
                    line = int(line_match.group(1))
                    outputs.append(
                        {
                            "path": str(path),
                            "start_line": line,
                            "end_line": line,
                            "external_id": f"{path}:{location}",
                        }
                    )
                    continue
                symbol_match = re.fullmatch(
                    r"(?:function|class|variable):\s*(.+)", location
                )
                if symbol_match:
                    outputs.append(
                        {
                            "path": str(path),
                            "symbol": symbol_match.group(1).strip(),
                            "external_id": f"{path}:{location}",
                        }
                    )

    visit(value)
    return outputs


def _usage(value: Any) -> dict[str, float]:
    totals = {"prompt": 0.0, "completion": 0.0, "calls": 0.0}
    if isinstance(value, list):
        for item in value:
            item_totals = _usage(item)
            for key in totals:
                totals[key] += item_totals[key]
    elif isinstance(value, dict):
        usage = value.get("usage")
        if isinstance(usage, dict):
            totals["prompt"] += float(usage.get("prompt_tokens") or 0)
            totals["completion"] += float(usage.get("completion_tokens") or 0)
            response = value.get("response")
            totals["calls"] += float(len(response) if isinstance(response, list) else 1)
        else:
            for item in value.values():
                item_totals = _usage(item)
                for key in totals:
                    totals[key] += item_totals[key]
    return totals


class AgentlessOutputStore:
    """读取 Agentless 官方 loc_outputs JSONL。"""

    def __init__(self, output_path: Path | str, *, stage: str) -> None:
        self.path = Path(output_path).resolve()
        self.stage = stage
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
        snapshot_verified = verify_snapshot(
            task,
            task_key,
            {
                "base_commit": row.get("base_commit")
                or metadata.get("base_commit"),
            },
        )
        if self.stage == "file":
            outputs = [
                {"path": path, "external_id": path}
                for path in flatten_strings(row.get("found_files") or [])
            ]
            trajectory_names = ("file_traj",)
        elif self.stage == "related":
            outputs = _parse_locations(row.get("found_related_locs") or {})
            trajectory_names = ("file_traj", "related_loc_traj")
        else:
            outputs = _parse_locations(row.get("found_edit_locs") or {})
            trajectory_names = ("file_traj", "related_loc_traj", "edit_loc_traj")
        usage = _usage([row.get(name) for name in trajectory_names])
        execution_observed = usage["calls"] > 0
        costs = {
            "external_stage": self.stage,
            "execution_cost_observed": execution_observed,
            "api_prompt_tokens": usage["prompt"] if execution_observed else None,
            "api_completion_tokens": usage["completion"]
            if execution_observed
            else None,
            "api_total_tokens": usage["prompt"] + usage["completion"]
            if execution_observed
            else None,
            "api_calls": usage["calls"] if execution_observed else None,
        }
        return task_key, outputs, snapshot_verified, costs


class AgentlessBaseline:
    """映射 Agentless 指定定位阶段的原生最终输出。"""

    def __init__(
        self,
        repository: RuntimeRepository,
        outputs: AgentlessOutputStore,
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
            source="agentless",
        )
        return external_result(
            task,
            external_task_id,
            outputs,
            candidates,
            snapshot_verified=snapshot_verified,
            metadata=costs,
        )
