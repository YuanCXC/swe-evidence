"""证据获取成本与充分性—成本效率指标。"""

from __future__ import annotations

from statistics import fmean
from typing import Mapping, Sequence


def auc_sufficiency_cost(points: Sequence[tuple[float, float]]) -> float:
    """计算 `(Token 预算, 证据充分率)` 曲线的归一化梯形面积。"""

    ordered = sorted((float(cost), float(score)) for cost, score in points)
    area = sum(
        (right_cost - left_cost) * (left_score + right_score) / 2.0
        for (left_cost, left_score), (right_cost, right_score) in zip(
            ordered, ordered[1:]
        )
    )
    width = ordered[-1][0] - ordered[0][0]
    return area / width if width else ordered[0][1]


def evaluate_costs(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    return {
        "mean_evidence_tokens": fmean(float(row["evidence_tokens"]) for row in rows),
        "mean_evidence_units": fmean(float(row["evidence_units"]) for row in rows),
        "mean_acquisition_steps": fmean(float(row.get("steps", 1.0)) for row in rows),
        "mean_tool_calls": fmean(float(row.get("tool_calls", 0.0)) for row in rows),
    }
