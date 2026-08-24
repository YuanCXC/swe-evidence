"""固定轮数、按检索分数选择 Single 的迭代 Agent Baseline。"""

from __future__ import annotations

from typing import Any, Mapping

from src.agents import AgentState, RetrievalPlanner
from src.data import build_online_issue
from src.retrieval import RepositoryRAG


class FixedIterativeBaseline:
    """保留多轮 Planner/RAG，但不使用 Pair、Evidence Policy 或学习型 STOP。"""

    def __init__(
        self,
        planner: RetrievalPlanner,
        rag: RepositoryRAG,
        *,
        max_steps: int,
        evidence_token_budget: int,
        evidence_unit_budget: int,
        retrieval_limit: int,
    ) -> None:
        self.planner = planner
        self.rag = rag
        self.max_steps = max_steps
        self.evidence_token_budget = evidence_token_budget
        self.evidence_unit_budget = evidence_unit_budget
        self.retrieval_limit = retrieval_limit

    def _finish(
        self,
        state: AgentState,
        reason: str,
        *,
        tool_calls: int = 0,
    ) -> None:
        state.steps.append(
            {
                "step": len(state.steps),
                "action_type": "stop",
                "action_id": "fixed_stop",
                "added_evidence_ids": [],
                "added_token_count": 0,
                "tool_calls": tool_calls,
                "forced": True,
                "termination_reason": reason,
            }
        )
        state.finish(reason)

    def run(self, task: Mapping[str, Any]) -> dict[str, Any]:
        """每轮选择候选账本中检索分数最高且满足预算的 Single。"""

        issue = build_online_issue(task["input"])
        state = AgentState(
            task_id=str(task["task_id"]),
            snapshot_id=str(task["snapshot_id"]),
        )

        for _ in range(self.max_steps):
            if len(state.evidence) >= self.evidence_unit_budget:
                self._finish(state, "unit_budget_exhausted")
                break
            if state.evidence_tokens >= self.evidence_token_budget:
                self._finish(state, "token_budget_exhausted")
                break

            plan = self.planner.plan(issue=issue, current_evidence=state.evidence)
            candidates = self.rag.retrieve(
                task,
                plan,
                state.evidence,
                exclude_evidence_ids=tuple(state.retrieved_ids),
                limit=self.retrieval_limit,
            )
            state.record_retrieval(plan, candidates)
            eligible = [
                unit
                for unit in state.candidate_pool.values()
                if int(unit.get("rendered_token_count") or 0)
                <= self.evidence_token_budget - state.evidence_tokens
            ]
            if not eligible:
                self._finish(
                    state,
                    "candidate_exhausted"
                    if not state.candidate_pool
                    else "token_budget_exhausted",
                    tool_calls=1,
                )
                break

            chosen = max(
                eligible,
                key=lambda unit: float(unit.get("retrieval_score") or 0.0),
            )
            action = {
                "action_type": "single",
                "evidence": [chosen],
            }
            added_ids = state.apply_action(action)
            state.steps.append(
                {
                    "step": len(state.steps),
                    "retrieval_plan": plan.to_dict(),
                    "retrieved_evidence_ids": [
                        str(unit["evidence_id"]) for unit in candidates
                    ],
                    "action_type": "single",
                    "action_id": f"single:{chosen['evidence_id']}",
                    "action_score": float(chosen.get("retrieval_score") or 0.0),
                    "added_evidence_ids": added_ids,
                    "added_token_count": int(chosen.get("rendered_token_count") or 0),
                    "tool_calls": 1,
                    "forced": False,
                    "termination_reason": None,
                }
            )
        else:
            self._finish(state, "fixed_steps")

        result = state.result()
        result["planner_calls"] = len(state.retrieval_rounds)
        return result
