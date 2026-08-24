"""维护 Agent 获取状态、历史检索记录和完整轨迹。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .retrieval_plan import RetrievalPlan


@dataclass
class AgentState:
    """一次任务从 K 为空到终止的在线状态。"""

    task_id: str
    snapshot_id: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    retrieved_ids: set[str] = field(default_factory=set)
    current_candidates: list[dict[str, Any]] = field(default_factory=list)
    candidate_pool: dict[str, dict[str, Any]] = field(default_factory=dict)
    retrieval_rounds: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    evidence_tokens: int = 0
    termination_reason: str | None = None

    @property
    def evidence_ids(self) -> set[str]:
        return {str(unit["evidence_id"]) for unit in self.evidence}

    def record_retrieval(
        self,
        plan: RetrievalPlan,
        candidates: Sequence[Mapping[str, Any]],
    ) -> None:
        """记录本轮 RAG 返回，并将全部候选加入历史排除集合。"""

        self.current_candidates = [dict(candidate) for candidate in candidates]
        candidate_ids = [str(candidate["evidence_id"]) for candidate in candidates]
        for candidate in self.current_candidates:
            self.candidate_pool[str(candidate["evidence_id"])] = candidate
        self.retrieved_ids.update(candidate_ids)
        self.retrieval_rounds.append(
            {
                "round": len(self.retrieval_rounds),
                "plan": plan.to_dict(),
                "candidate_evidence_ids": candidate_ids,
            }
        )

    def apply_action(self, action: Mapping[str, Any]) -> list[str]:
        """执行 Single 或 Pair，将新增 Evidence 写入 K。"""

        added_ids = []
        for unit in action["evidence"]:
            evidence_id = str(unit["evidence_id"])
            if evidence_id in self.evidence_ids:
                continue
            record = dict(unit)
            self.evidence.append(record)
            self.evidence_tokens += int(record.get("rendered_token_count") or 0)
            del self.candidate_pool[evidence_id]
            added_ids.append(evidence_id)
        return added_ids

    def finish(self, reason: str) -> None:
        """记录轨迹终止原因。"""

        self.termination_reason = reason

    def result(self) -> dict[str, Any]:
        """生成 Evidence Package 与完整在线轨迹。"""

        return {
            "task_id": self.task_id,
            "snapshot_id": self.snapshot_id,
            "evidence_package": list(self.evidence),
            "final_evidence_ids": [str(unit["evidence_id"]) for unit in self.evidence],
            "final_evidence_tokens": self.evidence_tokens,
            "retrieved_evidence_ids": sorted(self.retrieved_ids),
            "pending_candidate_evidence_ids": list(self.candidate_pool),
            "retrieval_rounds": list(self.retrieval_rounds),
            "steps": list(self.steps),
            "termination_reason": self.termination_reason,
        }
