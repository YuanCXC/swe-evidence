"""只执行一轮 Planner 与 RAG 的 One-shot Baseline。"""

from __future__ import annotations

from typing import Any, Mapping

from src.agents import RetrievalPlanner
from src.data import build_online_issue
from src.evaluation import apply_budget
from src.retrieval import RepositoryRAG


class OneShotBaseline:
    """使用与 Ours 相同的 Planner 和 RAG，但不运行 Evidence Policy。"""

    def __init__(
        self,
        planner: RetrievalPlanner,
        rag: RepositoryRAG,
        *,
        evidence_token_budget: int,
        evidence_unit_budget: int,
        retrieval_limit: int,
    ) -> None:
        self.planner = planner
        self.rag = rag
        self.evidence_token_budget = evidence_token_budget
        self.evidence_unit_budget = evidence_unit_budget
        self.retrieval_limit = retrieval_limit

    def run(self, task: Mapping[str, Any]) -> dict[str, Any]:
        """从空 K 规划并检索一次，将 RAG 排名直接作为最终证据包。"""

        issue = build_online_issue(task["input"])
        plan = self.planner.plan(issue=issue, current_evidence=[])
        candidates = self.rag.retrieve(task, plan, limit=self.retrieval_limit)
        evidence = apply_budget(
            candidates,
            max_units=self.evidence_unit_budget,
            max_tokens=self.evidence_token_budget,
        )
        evidence_ids = [str(unit["evidence_id"]) for unit in evidence]
        evidence_tokens = sum(
            int(unit.get("rendered_token_count") or 0) for unit in evidence
        )
        return {
            "task_id": str(task["task_id"]),
            "snapshot_id": str(task["snapshot_id"]),
            "evidence_package": evidence,
            "final_evidence_ids": evidence_ids,
            "final_evidence_tokens": evidence_tokens,
            "retrieved_evidence_ids": [str(unit["evidence_id"]) for unit in candidates],
            "retrieval_rounds": [
                {
                    "round": 0,
                    "plan": plan.to_dict(),
                    "candidate_evidence_ids": [
                        str(unit["evidence_id"]) for unit in candidates
                    ],
                }
            ],
            "steps": [
                {
                    "step": 0,
                    "retrieval_plan": plan.to_dict(),
                    "retrieved_evidence_ids": [
                        str(unit["evidence_id"]) for unit in candidates
                    ],
                    "action_type": "stop",
                    "action_id": "one_shot_stop",
                    "added_evidence_ids": evidence_ids,
                    "added_token_count": evidence_tokens,
                    "tool_calls": 1,
                    "forced": True,
                    "termination_reason": "one_shot",
                }
            ],
            "termination_reason": "one_shot",
            "planner_calls": 1,
        }
