"""将逐任务数值指标聚合为实验汇总表。"""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any, Mapping, Sequence


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_by: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_by)].append(row)

    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items()):
        aggregate = dict(zip(group_by, key))
        aggregate["task_count"] = len(members)
        metric_names = sorted(
            {
                name
                for row in members
                for name, value in row.items()
                if name not in group_by
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            }
        )
        for name in metric_names:
            values = [float(row[name]) for row in members if name in row]
            aggregate[name] = fmean(values)
        output.append(aggregate)
    return output
