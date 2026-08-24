"""端到端证据获取轨迹指标。"""

from __future__ import annotations

from statistics import fmean, median
from typing import Any, Mapping, Sequence


HARD_BUDGET_REASONS = {
    "hard_budget",
    "unit_budget_exhausted",
    "token_budget_exhausted",
}


def _trace_summary(trace: Mapping[str, Any]) -> dict[str, Any]:
    steps = list(trace["steps"])
    stop_index = next(
        (
            index
            for index, step in enumerate(steps)
            if step["action_type"] == "stop" and not bool(step.get("forced"))
        ),
        None,
    )
    first_sufficient_index = next(
        (
            index
            for index, step in enumerate(steps)
            if bool(step.get("sufficient_after"))
        ),
        None,
    )
    final_sufficient = bool(steps and steps[-1].get("sufficient_after"))
    premature_stop = stop_index is not None and not bool(
        steps[stop_index].get("sufficient_after")
    )
    late_steps = (
        steps[first_sufficient_index + 1 : stop_index]
        if first_sufficient_index is not None and stop_index is not None
        else []
    )
    return {
        "trajectory_success": float(final_sufficient),
        "premature_stop": float(premature_stop),
        "never_stop": float(stop_index is None),
        "hard_budget_termination": float(
            trace.get("termination_reason") in HARD_BUDGET_REASONS
        ),
        "steps": len(steps),
        "evidence_count": len(trace.get("final_evidence_ids") or []),
        "evidence_tokens": int(trace.get("final_evidence_tokens") or 0),
        "tool_calls": sum(int(step.get("tool_calls") or 0) for step in steps),
        "late_stop_steps": len(late_steps),
        "late_stop_evidence": sum(
            len(step.get("added_evidence_ids") or []) for step in late_steps
        ),
        "late_stop_tokens": sum(
            int(step.get("added_token_count") or 0) for step in late_steps
        ),
    }


def evaluate_trajectories(
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    rows = [_trace_summary(trace) for trace in trajectories]
    if not rows:
        return {"trajectory_count": 0.0}

    result = {
        "trajectory_count": float(len(rows)),
        "trajectory_success_rate": fmean(row["trajectory_success"] for row in rows),
        "premature_stop_rate": fmean(row["premature_stop"] for row in rows),
        "never_stop_rate": fmean(row["never_stop"] for row in rows),
        "hard_budget_termination_rate": fmean(
            row["hard_budget_termination"] for row in rows
        ),
        "average_acquisition_steps": fmean(row["steps"] for row in rows),
        "median_acquisition_steps": float(median(row["steps"] for row in rows)),
        "mean_evidence_count": fmean(row["evidence_count"] for row in rows),
        "mean_evidence_tokens": fmean(row["evidence_tokens"] for row in rows),
        "mean_tool_calls": fmean(row["tool_calls"] for row in rows),
        "late_stop_overhead_steps": fmean(row["late_stop_steps"] for row in rows),
        "late_stop_overhead_evidence": fmean(row["late_stop_evidence"] for row in rows),
        "late_stop_overhead_tokens": fmean(row["late_stop_tokens"] for row in rows),
    }
    return result
