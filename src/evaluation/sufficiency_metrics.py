"""基于七类证据需求与 OR-of-AND Witness 的确定性充分性评价。"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence


def group_is_covered(group: Mapping[str, Any], selected_ids: set[str]) -> bool:
    return set(map(str, group.get("evidence_ids") or [])) <= selected_ids


def obligation_is_covered(
    obligation: Mapping[str, Any], selected_ids: set[str]
) -> bool:
    return any(
        group_is_covered(group, selected_ids)
        for group in obligation.get("witness_groups") or []
    )


def covered_obligation_ids(
    selected_ids: Iterable[str],
    obligations: Sequence[Mapping[str, Any]],
    *,
    mandatory_only: bool = False,
) -> set[str]:
    selected = set(map(str, selected_ids))
    return {
        str(obligation["obligation_id"])
        for obligation in obligations
        if bool(obligation.get("applicable"))
        and (not mandatory_only or bool(obligation.get("mandatory")))
        and obligation_is_covered(obligation, selected)
    }


def evaluate_sufficiency(
    selected_ids: Iterable[str],
    obligations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = set(map(str, selected_ids))
    applicable = [item for item in obligations if bool(item.get("applicable"))]
    critical = [item for item in applicable if bool(item.get("mandatory"))]
    covered = [item for item in applicable if obligation_is_covered(item, selected)]
    covered_critical = [
        item for item in critical if obligation_is_covered(item, selected)
    ]
    witness_groups = [
        group for item in applicable for group in item.get("witness_groups") or []
    ]
    covered_groups = [
        group for group in witness_groups if group_is_covered(group, selected)
    ]
    complementary = [
        group for group in witness_groups if len(group.get("evidence_ids") or []) > 1
    ]
    covered_complementary = [
        group for group in complementary if group_is_covered(group, selected)
    ]

    by_dimension = {
        str(item["type"]): float(obligation_is_covered(item, selected))
        for item in applicable
    }
    return {
        "sufficient": bool(critical) and len(covered_critical) == len(critical),
        "critical_requirement_coverage": len(covered_critical) / len(critical)
        if critical
        else 0.0,
        "obligation_coverage": len(covered) / len(applicable) if applicable else 0.0,
        "witness_group_coverage": len(covered_groups) / len(witness_groups)
        if witness_groups
        else 0.0,
        "complementary_group_coverage": (
            len(covered_complementary) / len(complementary) if complementary else 0.0
        ),
        "covered_obligation_ids": [str(item["obligation_id"]) for item in covered],
        "missing_critical_obligation_ids": [
            str(item["obligation_id"])
            for item in critical
            if item not in covered_critical
        ],
        "coverage_by_dimension": by_dimension,
    }
