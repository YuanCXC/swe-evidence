"""将不同方法的输出映射到冻结的证据单元全集。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


def _span_overlaps(output: Mapping[str, Any], unit: Mapping[str, Any]) -> bool:
    return int(unit["start_line"]) <= int(output["end_line"]) and int(
        output["start_line"]
    ) <= int(unit["end_line"])


def _unit_sort_key(unit: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(unit["start_line"]),
        int(unit["end_line"]),
        str(unit["evidence_id"]),
    )


def adapt_outputs(
    outputs: Sequence[Mapping[str, Any]],
    evidence_units: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """把已排序的文件、符号、区间或单元输出转换为已有证据单元。

    每项输出必须包含以下一种身份信息：`evidence_id`、`path + symbol`、
    `path + start_line + end_line` 或 `path`。本函数不会创建、切分或改写证据单元。
    """

    units = [dict(unit) for unit in evidence_units if bool(unit.get("scoreable", True))]
    by_id = {str(unit["evidence_id"]): unit for unit in units}
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_symbol: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for unit in units:
        path = str(unit["path"])
        by_path[path].append(unit)
        for symbol_field in ("symbol", "qualified_name"):
            symbol = unit.get(symbol_field)
            if symbol:
                by_symbol[(path, str(symbol))].append(unit)

    for path_units in by_path.values():
        path_units.sort(key=_unit_sort_key)

    adapted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, output in enumerate(outputs, start=1):
        matches: list[dict[str, Any]]
        if output.get("evidence_id"):
            unit = by_id.get(str(output["evidence_id"]))
            matches = [unit] if unit is not None else []
        elif output.get("symbol"):
            matches = by_symbol.get((str(output["path"]), str(output["symbol"])), [])
        elif (
            output.get("start_line") is not None and output.get("end_line") is not None
        ):
            matches = [
                unit
                for unit in by_path.get(str(output["path"]), [])
                if _span_overlaps(output, unit)
            ]
        else:
            matches = by_path.get(str(output["path"]), [])

        for unit in matches:
            evidence_id = str(unit["evidence_id"])
            if evidence_id in seen:
                continue
            item = dict(unit)
            item["source_rank"] = rank
            item["source_score"] = output.get("score")
            adapted.append(item)
            seen.add(evidence_id)

    return adapted
