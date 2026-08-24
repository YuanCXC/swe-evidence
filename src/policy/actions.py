"""构造在线可见且满足预算的 Single、Pair、STOP 动作。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from src.data import RuntimeRepository


class ActionBuilder:
    """按冻结候选规则构造动作，不判断动作的语义价值。"""

    def __init__(self, repository: RuntimeRepository, *, pair_limit: int = 8) -> None:
        self.repository = repository
        self.pair_limit = pair_limit

    def _structural_pairs(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> list[tuple[str, str]]:
        """依据 canonical parent/child 与真实文件相邻关系构造 Pair。"""

        records = {str(unit["evidence_id"]): unit for unit in candidates}
        candidate_ids = list(records)
        candidate_set = set(candidate_ids)
        edges: dict[str, set[str]] = defaultdict(set)
        file_version_ids = list(
            dict.fromkeys(str(unit["file_version_id"]) for unit in candidates)
        )

        for file_version_id in file_version_ids:
            file_units = self.repository.get_file_evidence(
                file_version_id,
                scoreable_only=True,
            )
            for unit in file_units:
                evidence_id = str(unit["evidence_id"])
                parent_id = str(unit.get("parent_evidence_id") or "")
                if evidence_id in candidate_set and parent_id in candidate_set:
                    edges[evidence_id].add(parent_id)
                    edges[parent_id].add(evidence_id)
            for left, right in zip(file_units, file_units[1:]):
                left_id = str(left["evidence_id"])
                right_id = str(right["evidence_id"])
                if left_id in candidate_set and right_id in candidate_set:
                    edges[left_id].add(right_id)
                    edges[right_id].add(left_id)

        pairs = []
        for source in candidate_ids:
            for target in sorted(edges[source]):
                pair = tuple(sorted((source, target)))
                if pair not in pairs:
                    pairs.append(pair)
                if len(pairs) >= self.pair_limit:
                    return pairs
        return pairs

    def build(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        current_evidence: Sequence[Mapping[str, Any]],
        remaining_tokens: int,
        remaining_units: int,
    ) -> list[dict[str, Any]]:
        """生成当前状态下可执行的统一动作集合。"""

        selected_ids = {str(unit["evidence_id"]) for unit in current_evidence}
        eligible = [
            dict(unit)
            for unit in candidates
            if str(unit["evidence_id"]) not in selected_ids
            and int(unit.get("rendered_token_count") or 0) <= remaining_tokens
            and remaining_units >= 1
        ]
        records = {str(unit["evidence_id"]): unit for unit in eligible}
        actions = [
            {
                "action_id": f"single:{unit['evidence_id']}",
                "action_type": "single",
                "evidence_ids": [str(unit["evidence_id"])],
                "evidence": [unit],
                "token_cost": int(unit.get("rendered_token_count") or 0),
                "candidate_scope": "online",
                "candidate_sources": list(unit.get("retrieval_sources") or []),
            }
            for unit in eligible
        ]

        if remaining_units >= 2:
            for left_id, right_id in self._structural_pairs(eligible):
                pair_units = [records[left_id], records[right_id]]
                token_cost = sum(
                    int(unit.get("rendered_token_count") or 0) for unit in pair_units
                )
                if token_cost > remaining_tokens:
                    continue
                actions.append(
                    {
                        "action_id": f"pair:{left_id}:{right_id}",
                        "action_type": "pair",
                        "evidence_ids": [left_id, right_id],
                        "evidence": pair_units,
                        "token_cost": token_cost,
                        "candidate_scope": "online",
                        "candidate_sources": ["structure_pair"],
                    }
                )

        actions.append(
            {
                "action_id": "stop",
                "action_type": "stop",
                "evidence_ids": [],
                "evidence": [],
                "token_cost": 0,
                "candidate_scope": "stop",
                "candidate_sources": ["stop"],
            }
        )
        return actions
