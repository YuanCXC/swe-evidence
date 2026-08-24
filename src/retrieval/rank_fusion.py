"""使用 Reciprocal Rank Fusion 合并不同检索通道。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    rank_constant: int = 60,
    limit: int = 64,
) -> list[dict[str, Any]]:
    """融合不同通道的 ID 排名，并保留来源和通道内名次。"""

    scores: dict[str, float] = defaultdict(float)
    source_ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for source, ranked_ids in rankings.items():
        for rank, item_id in enumerate(dict.fromkeys(map(str, ranked_ids)), start=1):
            scores[item_id] += 1.0 / (rank_constant + rank)
            source_ranks[item_id][source] = rank

    ordered = sorted(scores, key=lambda item_id: (-scores[item_id], item_id))[:limit]
    return [
        {
            "item_id": item_id,
            "rrf_score": scores[item_id],
            "sources": sorted(source_ranks[item_id]),
            "source_ranks": source_ranks[item_id],
        }
        for item_id in ordered
    ]
