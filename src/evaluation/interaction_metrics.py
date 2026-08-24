"""证据互补性、替代性与冗余性诊断指标。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _obligation_progress(obligation: Mapping[str, Any], selected: set[str]) -> float:
    groups = obligation.get("witness_groups") or []
    progress = [
        len(set(map(str, group.get("evidence_ids") or [])) & selected)
        / len(group.get("evidence_ids") or [])
        for group in groups
        if group.get("evidence_ids")
    ]
    return max(progress, default=0.0)


def evaluate_interactions(
    ordered_evidence_ids: Sequence[str],
    obligations: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    applicable = [item for item in obligations if bool(item.get("applicable"))]
    selected: set[str] = set()
    redundant = 0
    substitute_duplicates = 0
    substitute_opportunities = 0

    for evidence_id in map(str, ordered_evidence_ids):
        before = {
            str(item["obligation_id"]): _obligation_progress(item, selected)
            for item in applicable
        }

        for obligation in applicable:
            groups = obligation.get("witness_groups") or []
            if len(groups) < 2 or before[str(obligation["obligation_id"])] < 1.0:
                continue
            substitute_opportunities += 1
            if any(
                evidence_id in set(map(str, group.get("evidence_ids") or []))
                for group in groups
            ):
                substitute_duplicates += 1

        selected.add(evidence_id)
        after = {
            str(item["obligation_id"]): _obligation_progress(item, selected)
            for item in applicable
        }
        if all(after[key] <= before[key] for key in before):
            redundant += 1

    count = len(ordered_evidence_ids)
    complementary_obligations = [
        item
        for item in applicable
        if any(
            len(group.get("evidence_ids") or []) > 1
            for group in item.get("witness_groups") or []
        )
    ]
    completed_complementary = [
        item
        for item in complementary_obligations
        if any(
            len(group.get("evidence_ids") or []) > 1
            and set(map(str, group.get("evidence_ids") or [])) <= selected
            for group in item.get("witness_groups") or []
        )
    ]
    return {
        "redundant_evidence_rate": redundant / count if count else 0.0,
        "substitute_duplication_rate": (
            substitute_duplicates / substitute_opportunities
            if substitute_opportunities
            else 0.0
        ),
        "complementary_evidence_completion_rate": (
            len(completed_complementary) / len(complementary_obligations)
            if complementary_obligations
            else 0.0
        ),
    }
