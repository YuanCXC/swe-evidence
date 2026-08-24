"""状态级 Single、Pair 与 STOP 策略指标。"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence


def _is_acceptable(action: Mapping[str, Any]) -> bool:
    return bool(action.get("acceptable")) or action.get("action_label") == "positive"


def _ranked_actions(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return sorted(
        state["actions"], key=lambda action: float(action["score"]), reverse=True
    )


def evaluate_policy_states(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hit_at_1 = 0
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    expected_by_type: Counter[str] = Counter()
    correct_by_type: Counter[str] = Counter()
    top_action_types: Counter[str] = Counter()

    for state in states:
        ranked = _ranked_actions(state)
        positive_types = {
            str(action["action_type"]) for action in ranked if _is_acceptable(action)
        }
        for action_type in positive_types:
            expected_by_type[action_type] += 1

        top = ranked[0]
        top_type = str(top["action_type"])
        top_action_types[top_type] += 1
        if _is_acceptable(top):
            hit_at_1 += 1
            correct_by_type[top_type] += 1

        first_positive_rank = next(
            (
                rank
                for rank, action in enumerate(ranked, start=1)
                if _is_acceptable(action)
            ),
            None,
        )
        reciprocal_ranks.append(
            1.0 / first_positive_rank if first_positive_rank else 0.0
        )

        gains = [1.0 if _is_acceptable(action) else 0.0 for action in ranked]
        dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
        ideal = sorted(gains, reverse=True)
        idcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg else 0.0)

    count = len(states)
    result: dict[str, Any] = {
        "state_count": count,
        "action_hit@1": hit_at_1 / count if count else 0.0,
        "mrr": sum(reciprocal_ranks) / count if count else 0.0,
        "ndcg": sum(ndcgs) / count if count else 0.0,
        "top_action_type_counts": dict(top_action_types),
    }
    for action_type in ("single", "pair", "stop"):
        denominator = expected_by_type[action_type]
        result[f"{action_type}_accuracy"] = (
            correct_by_type[action_type] / denominator if denominator else 0.0
        )
        result[f"{action_type}_state_count"] = denominator
    return result
