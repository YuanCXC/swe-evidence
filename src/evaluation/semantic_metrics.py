"""聚合参考修复约束语义评审结果。"""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean
from typing import Any, Mapping, Sequence

from .semantic_judge import SEMANTIC_DIMENSIONS


VERDICT_SCORE = {"sufficient": 1.0, "partial": 0.5, "insufficient": 0.0}
COVERAGE_SCORE = {"sufficient": 1.0, "partial": 0.5, "none": 0.0}
CAUSAL_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0, "uncertain": 0.0}
EXECUTION_SCORE = {"relevant": 1.0, "partial": 0.5, "irrelevant": 0.0, "uncertain": 0.0}


def aggregate_semantic_judgments(
    judgments: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    dimension_scores: dict[str, list[float]] = defaultdict(list)
    critical_scores: list[float] = []
    semantic_precision: list[float] = []
    redundancy_rates: list[float] = []

    for judgment in judgments:
        for dimension in SEMANTIC_DIMENSIONS:
            item = judgment["dimensions"][dimension]
            if not bool(item["applicable"]):
                continue
            score = COVERAGE_SCORE[str(item["coverage"])]
            dimension_scores[dimension].append(score)
            if bool(item["critical"]):
                critical_scores.append(score)

        count = int(judgment["evidence_count"])
        semantic_precision.append(
            len(judgment["useful_evidence_ids"]) / count if count else 0.0
        )
        redundancy_rates.append(
            len(judgment["redundant_evidence_ids"]) / count if count else 0.0
        )

    result = {
        "semantic_judgment_count": float(len(judgments)),
        "reference_grounded_semantic_sufficiency_rate": fmean(
            float(item["sufficiency_verdict"] == "sufficient") for item in judgments
        ),
        "partial_credit_sufficiency_score": fmean(
            VERDICT_SCORE[str(item["sufficiency_verdict"])] for item in judgments
        ),
        "semantic_critical_requirement_coverage": fmean(critical_scores)
        if critical_scores
        else 0.0,
        "causal_correctness_score": fmean(
            CAUSAL_SCORE[str(item["causal_correctness"])] for item in judgments
        ),
        "execution_relevance_score": fmean(
            EXECUTION_SCORE[str(item["execution_relevance"])] for item in judgments
        ),
        "semantic_precision": fmean(semantic_precision),
        "semantic_redundancy_rate": fmean(redundancy_rates),
        "misleading_evidence_task_rate": fmean(
            float(bool(item["misleading_evidence_ids"])) for item in judgments
        ),
        "mean_judge_confidence": fmean(float(item["confidence"]) for item in judgments),
    }
    for dimension in SEMANTIC_DIMENSIONS:
        scores = dimension_scores[dimension]
        result[f"semantic_{dimension}_coverage"] = fmean(scores) if scores else 0.0
    return result


def repeated_judge_agreement(judgments: Sequence[Mapping[str, Any]]) -> float:
    groups: dict[str, list[str]] = defaultdict(list)
    for item in judgments:
        groups[str(item["case_id"])].append(str(item["sufficiency_verdict"]))
    agreements = [float(len(set(verdicts)) == 1) for verdicts in groups.values()]
    return fmean(agreements) if agreements else 0.0
