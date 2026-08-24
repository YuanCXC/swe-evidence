"""对排序后的证据包施加等单元和等 Token 预算。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def evidence_token_count(unit: Mapping[str, Any]) -> int:
    return int(unit.get("rendered_token_count") or unit.get("token_count") or 0)


def apply_budget(
    evidence_package: Sequence[Mapping[str, Any]],
    *,
    max_units: int | None = None,
    max_tokens: int | None = None,
) -> list[dict[str, Any]]:
    """按排名选择同时满足单元数和 Token 预算的证据。"""

    selected: list[dict[str, Any]] = []
    tokens = 0
    for unit in evidence_package:
        if max_units is not None and len(selected) >= max_units:
            break
        unit_tokens = evidence_token_count(unit)
        if max_tokens is not None and tokens + unit_tokens > max_tokens:
            continue
        selected.append(dict(unit))
        tokens += unit_tokens
    return selected
