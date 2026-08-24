"""读取任务输入，并隔离所有离线监督字段。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping

import pyarrow.dataset as ds


ONLINE_COLUMNS = (
    "schema_version",
    "task_id",
    "task_group_id",
    "snapshot_id",
    "input",
    "provenance",
    "split_info",
    "quality",
    "experiment_eligible",
    "split",
)


def online_task_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """只保留允许进入 Retriever、Policy 和 Agent 的任务字段。"""

    return {name: row.get(name) for name in ONLINE_COLUMNS}


class TaskReader:
    """按 split 或 task_id 读取在线安全任务视图。"""

    def __init__(self, tasks_path: Path | str) -> None:
        self.path = Path(tasks_path)
        self.dataset = ds.dataset(self.path, format="parquet")

    def iter_tasks(
        self,
        *,
        split: str | None = None,
        experiment_only: bool = True,
        batch_size: int = 128,
    ) -> Iterator[dict[str, Any]]:
        """流式读取任务，默认排除不满足实验资格的记录。"""

        expression = None
        if split is not None:
            expression = ds.field("split") == split
        if experiment_only:
            eligible = ds.field("experiment_eligible") == True  # noqa: E712
            expression = eligible if expression is None else expression & eligible

        for batch in self.dataset.to_batches(
            columns=list(ONLINE_COLUMNS),
            filter=expression,
            batch_size=batch_size,
        ):
            for row in batch.to_pylist():
                yield online_task_view(row)

    def get_task(self, task_id: str) -> dict[str, Any]:
        """按稳定 task_id 读取一个在线安全任务。"""

        table = self.dataset.to_table(
            columns=list(ONLINE_COLUMNS),
            filter=ds.field("task_id") == task_id,
        )
        return online_task_view(table.to_pylist()[0])
