"""文件、证据单元、符号与代码区间定位指标。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .retrieval_metrics import recall_at_k, reciprocal_rank


def ranked_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def localization_metrics(
    evidence_package: Sequence[Mapping[str, Any]],
    *,
    gold_evidence_ids: set[str],
    gold_files: set[str],
    k: int | None = None,
) -> dict[str, float]:
    package = evidence_package[:k] if k is not None else evidence_package
    evidence_ids = [str(unit["evidence_id"]) for unit in package]
    paths = ranked_unique([str(unit["path"]) for unit in package])
    return {
        "gold_evidence_recall": recall_at_k(
            evidence_ids, gold_evidence_ids, len(evidence_ids)
        ),
        "gold_evidence_mrr": reciprocal_rank(evidence_ids, gold_evidence_ids),
        "file_recall": recall_at_k(paths, gold_files, len(paths)),
        "file_mrr": reciprocal_rank(paths, gold_files),
    }


def span_recall(
    predicted: Sequence[Mapping[str, Any]],
    gold_spans: Sequence[Mapping[str, Any]],
) -> float:
    hits = 0
    for gold in gold_spans:
        if any(
            str(item["path"]) == str(gold["path"])
            and int(item["start_line"]) <= int(gold["end_line"])
            and int(gold["start_line"]) <= int(item["end_line"])
            for item in predicted
        ):
            hits += 1
    return hits / len(gold_spans) if gold_spans else 0.0
