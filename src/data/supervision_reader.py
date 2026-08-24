"""读取只能用于训练、审计和离线评价的监督信息。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping

import pyarrow.dataset as ds


OFFLINE_COLUMNS = (
    "task_id",
    "task_group_id",
    "snapshot_id",
    "supervision",
    "trajectories",
    "evaluation",
    "strong_teacher_status",
    "strong_teacher_audit_status",
    "strong_teacher_risk_score",
    "strong_teacher_risk_flags_json",
    "strong_teacher_exclusion_reason",
    "strong_teacher_overridden_slots_json",
    "strong_teacher_blockers_json",
    "strong_teacher_semantic_slots_json",
    "strong_teacher_semantic_review_complete",
    "experiment_eligible",
    "experiment_exclusion_reason",
    "experiment_exclusion_source",
    "experiment_exclusion_details_json",
    "split",
)


def supervision_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """保留监督、Gold 和离线审计字段。"""

    return {name: row.get(name) for name in OFFLINE_COLUMNS}


class SupervisionReader:
    """读取与在线 Agent 严格隔离的任务监督。"""

    def __init__(self, tasks_path: Path | str) -> None:
        self.path = Path(tasks_path)
        self.dataset = ds.dataset(self.path, format="parquet")

    def iter_supervision(
        self,
        *,
        split: str | None = None,
        experiment_only: bool = True,
        batch_size: int = 64,
    ) -> Iterator[dict[str, Any]]:
        """流式读取离线监督记录。"""

        expression = None
        if split is not None:
            expression = ds.field("split") == split
        if experiment_only:
            eligible = ds.field("experiment_eligible") == True  # noqa: E712
            expression = eligible if expression is None else expression & eligible

        for batch in self.dataset.to_batches(
            columns=list(OFFLINE_COLUMNS),
            filter=expression,
            batch_size=batch_size,
        ):
            for row in batch.to_pylist():
                yield supervision_view(row)

    def get_task_supervision(self, task_id: str) -> dict[str, Any]:
        """按 task_id 读取完整离线监督记录。"""

        table = self.dataset.to_table(
            columns=list(OFFLINE_COLUMNS),
            filter=ds.field("task_id") == task_id,
        )
        return supervision_view(table.to_pylist()[0])

    def get_references(self, task_id: str) -> dict[str, Any]:
        """读取确定性评价和语义 Judge 使用的参考信息。"""

        row = self.get_task_supervision(task_id)
        supervision = row["supervision"]
        return {
            "task_id": task_id,
            "obligations": supervision.get("obligations") or [],
            "evidence_labels": supervision.get("evidence_labels") or [],
            "modified_files": supervision.get("modified_files") or [],
            "gold_patch": supervision.get("gold_patch") or "",
            "test_patch": supervision.get("test_patch") or "",
            "evaluation": row["evaluation"],
        }
