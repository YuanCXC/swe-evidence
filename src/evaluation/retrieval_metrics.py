"""候选证据检索与排序指标。"""

from __future__ import annotations

import math
from typing import Hashable, Mapping, Sequence


def recall_at_k(
    ranked_ids: Sequence[Hashable], relevant_ids: set[Hashable], k: int
) -> float:
    return (
        len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)
        if relevant_ids
        else 0.0
    )


def reciprocal_rank(
    ranked_ids: Sequence[Hashable], relevant_ids: set[Hashable]
) -> float:
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[Hashable],
    relevance: Mapping[Hashable, float],
    k: int,
) -> float:
    gains = [float(relevance.get(item_id, 0.0)) for item_id in ranked_ids[:k]]
    dcg = sum(
        (2.0**gain - 1.0) / math.log2(rank + 2) for rank, gain in enumerate(gains)
    )
    ideal = sorted((float(value) for value in relevance.values()), reverse=True)[:k]
    idcg = sum(
        (2.0**gain - 1.0) / math.log2(rank + 2) for rank, gain in enumerate(ideal)
    )
    return dcg / idcg if idcg else 0.0


def retrieval_metrics(
    ranked_ids: Sequence[Hashable],
    relevance: Mapping[Hashable, float],
    *,
    cutoffs: Sequence[int] = (1, 5, 10, 20, 64),
) -> dict[str, float]:
    relevant_ids = {
        item_id for item_id, value in relevance.items() if float(value) > 0.0
    }
    result = {
        "mrr": reciprocal_rank(ranked_ids, relevant_ids),
        "online_positive_coverage": float(bool(set(ranked_ids) & relevant_ids)),
    }
    for k in cutoffs:
        result[f"recall@{k}"] = recall_at_k(ranked_ids, relevant_ids, k)
        result[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, relevance, k)
    return result


def structure_increment(
    base_ranked_ids: Sequence[Hashable],
    expanded_ranked_ids: Sequence[Hashable],
    relevant_ids: set[Hashable],
    *,
    k: int,
) -> float:
    base_hits = set(base_ranked_ids[:k]) & relevant_ids
    expanded_hits = set(expanded_ranked_ids[:k]) & relevant_ids
    return len(expanded_hits - base_hits) / len(relevant_ids) if relevant_ids else 0.0
