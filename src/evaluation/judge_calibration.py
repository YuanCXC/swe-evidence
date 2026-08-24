"""用于校准语义评审器的反事实样例与指标。"""

from __future__ import annotations

from statistics import fmean
from typing import Any, Mapping, Sequence

from .semantic_metrics import VERDICT_SCORE


def delete_critical_evidence(
    evidence_package: Sequence[Mapping[str, Any]],
    obligations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected_ids = {str(unit["evidence_id"]) for unit in evidence_package}
    critical_group = next(
        group
        for obligation in obligations
        if bool(obligation.get("applicable")) and bool(obligation.get("mandatory"))
        for group in obligation.get("witness_groups") or []
        if set(map(str, group.get("evidence_ids") or [])) <= selected_ids
    )
    deleted_id = str(critical_group["evidence_ids"][0])
    return [
        dict(unit)
        for unit in evidence_package
        if str(unit["evidence_id"]) != deleted_id
    ]


def inject_evidence(
    evidence_package: Sequence[Mapping[str, Any]],
    injected_unit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [*(dict(unit) for unit in evidence_package), dict(injected_unit)]


def calibration_sensitivity(pairs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    deletion_drops = []
    irrelevant_non_improvements = []
    cross_task_non_improvements = []
    for pair in pairs:
        original = VERDICT_SCORE[str(pair["original"]["sufficiency_verdict"])]
        perturbed = VERDICT_SCORE[str(pair["perturbed"]["sufficiency_verdict"])]
        case_type = str(pair["case_type"])
        if case_type == "critical_deletion":
            deletion_drops.append(float(perturbed < original))
        elif case_type == "irrelevant_injection":
            irrelevant_non_improvements.append(float(perturbed <= original))
        elif case_type == "cross_task_injection":
            cross_task_non_improvements.append(float(perturbed <= original))
    return {
        "critical_deletion_sensitivity": fmean(deletion_drops)
        if deletion_drops
        else 0.0,
        "irrelevant_injection_non_improvement_rate": (
            fmean(irrelevant_non_improvements) if irrelevant_non_improvements else 0.0
        ),
        "cross_task_injection_non_improvement_rate": (
            fmean(cross_task_non_improvements) if cross_task_non_improvements else 0.0
        ),
    }
