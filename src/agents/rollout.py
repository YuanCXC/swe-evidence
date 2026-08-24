"""执行 Plan、RAG、Policy、Update 组成的完整在线轨迹。"""

from __future__ import annotations

from typing import Any, Mapping

from src.data import build_online_issue
from src.policy import ActionBuilder, EvidencePolicy
from src.retrieval import RepositoryRAG

from .planner import RetrievalPlanner
from .retrieval_plan import RetrievalPlan
from .state import AgentState


class EvidenceAgent:
    """单 Agent 的状态化检索规划与 Evidence 获取循环。"""

    def __init__(
        self,
        planner: RetrievalPlanner,
        rag: RepositoryRAG,
        action_builder: ActionBuilder,
        policy: EvidencePolicy,
        *,
        evidence_token_budget: int = 32_768,
        evidence_unit_budget: int = 64,
        retrieval_limit: int = 64,
    ) -> None:
        self.planner = planner
        self.rag = rag
        self.action_builder = action_builder
        self.policy = policy
        self.evidence_token_budget = evidence_token_budget
        self.evidence_unit_budget = evidence_unit_budget
        self.retrieval_limit = retrieval_limit

    def _record_forced_stop(
        self,
        state: AgentState,
        reason: str,
        *,
        tool_calls: int = 0,
        plan: RetrievalPlan | None = None,
        candidates: list[dict[str, Any]] | None = None,
    ) -> None:
        step = {
            "step": len(state.steps),
            "action_type": "stop",
            "action_id": "forced_stop",
            "action_score": None,
            "added_evidence_ids": [],
            "added_token_count": 0,
            "tool_calls": tool_calls,
            "forced": True,
            "termination_reason": reason,
        }
        if plan is not None:
            step["retrieval_plan"] = plan.to_dict()
            step["retrieved_evidence_ids"] = [
                str(candidate["evidence_id"]) for candidate in candidates or []
            ]
        state.steps.append(step)
        state.finish(reason)

    def run(self, task: Mapping[str, Any]) -> dict[str, Any]:
        """从空 K 开始运行，直到模型 STOP 或系统预算终止。"""

        issue = build_online_issue(task["input"])
        state = AgentState(
            task_id=str(task["task_id"]),
            snapshot_id=str(task["snapshot_id"]),
        )

        while state.termination_reason is None:
            if len(state.evidence) >= self.evidence_unit_budget:
                self._record_forced_stop(state, "unit_budget_exhausted")
                break
            if state.evidence_tokens >= self.evidence_token_budget:
                self._record_forced_stop(state, "token_budget_exhausted")
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
            if not state.candidate_pool:
                self._record_forced_stop(
                    state,
                    "candidate_exhausted",
                    tool_calls=1,
                    plan=plan,
                    candidates=candidates,
                )
                break

            actions = self.action_builder.build(
                list(state.candidate_pool.values()),
                current_evidence=state.evidence,
                remaining_tokens=self.evidence_token_budget - state.evidence_tokens,
                remaining_units=self.evidence_unit_budget - len(state.evidence),
            )
            evidence_actions = [
                action for action in actions if action["action_type"] != "stop"
            ]
            if not evidence_actions:
                self._record_forced_stop(
                    state,
                    "token_budget_exhausted",
                    tool_calls=1,
                    plan=plan,
                    candidates=candidates,
                )
                break

            ranked = self.policy.rank_actions(
                task_input=task["input"],
                current_evidence=state.evidence,
                actions=actions,
            )
            scoreable_evidence_actions = [
                action for action in ranked if action["action_type"] != "stop"
            ]
            if not scoreable_evidence_actions:
                self._record_forced_stop(
                    state,
                    "no_scoreable_action",
                    tool_calls=1,
                    plan=plan,
                    candidates=candidates,
                )
                break

            chosen = ranked[0]
            added_ids = []
            if chosen["action_type"] == "stop":
                state.finish("model_stop")
            else:
                added_ids = state.apply_action(chosen)

            state.steps.append(
                {
                    "step": len(state.steps),
                    "retrieval_plan": plan.to_dict(),
                    "retrieved_evidence_ids": [
                        str(candidate["evidence_id"]) for candidate in candidates
                    ],
                    "action_type": str(chosen["action_type"]),
                    "action_id": str(chosen["action_id"]),
                    "action_score": float(chosen["score"]),
                    "added_evidence_ids": added_ids,
                    "added_token_count": int(chosen["token_cost"]),
                    "tool_calls": 1,
                    "forced": False,
                    "termination_reason": state.termination_reason,
                    "ranked_actions": [
                        {
                            "action_id": str(action["action_id"]),
                            "action_type": str(action["action_type"]),
                            "evidence_ids": list(action["evidence_ids"]),
                            "score": float(action["score"]),
                        }
                        for action in ranked
                    ],
                }
            )

        return state.result()
