"""在不修改 src 核心实现的条件下切换实验组件。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from src.agents import EvidenceAgent, RetrievalPlan, RetrievalPlanner
from src.data import RuntimeRepository
from src.policy import ActionBuilder
from src.retrieval import RepositoryRAG


@dataclass(frozen=True)
class Variant:
    """一个 Ours 运行时变体的组件开关。"""

    use_pair: bool = True
    use_stop: bool = True
    use_structure: bool = True
    adaptive_planning: bool = True


ABLATION_VARIANTS = {
    "ours_no_pair": Variant(use_pair=False),
    "ours_no_stop": Variant(use_stop=False),
    "ours_no_structure": Variant(use_structure=False),
    "ours_fixed_plan": Variant(adaptive_planning=False),
}


class VariantPlanner:
    """限制 Structure 或复用首轮计划的 Planner 包装器。"""

    def __init__(self, planner: RetrievalPlanner, variant: Variant) -> None:
        self.planner = planner
        self.variant = variant
        self.fixed_plans: dict[str, RetrievalPlan] = {}

    def plan(
        self,
        *,
        issue: str,
        current_evidence: Sequence[Mapping[str, Any]],
    ) -> RetrievalPlan:
        if not self.variant.adaptive_planning and issue in self.fixed_plans:
            return self.fixed_plans[issue]

        visible_evidence = current_evidence if self.variant.adaptive_planning else []
        plan = self.planner.plan(
            issue=issue,
            current_evidence=visible_evidence,
        )
        if not self.variant.use_structure:
            plan = replace(
                plan,
                retrieval_channels=tuple(
                    channel
                    for channel in plan.retrieval_channels
                    if channel != "structure"
                ),
            )
        if not self.variant.adaptive_planning:
            self.fixed_plans[issue] = plan
        return plan


class VariantActionBuilder:
    """从统一动作空间中移除 Pair 或学习型 STOP。"""

    def __init__(self, action_builder: ActionBuilder, variant: Variant) -> None:
        self.action_builder = action_builder
        self.variant = variant

    def build(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        current_evidence: Sequence[Mapping[str, Any]],
        remaining_tokens: int,
        remaining_units: int,
    ) -> list[dict[str, Any]]:
        actions = self.action_builder.build(
            candidates,
            current_evidence=current_evidence,
            remaining_tokens=remaining_tokens,
            remaining_units=remaining_units,
        )
        return [
            action
            for action in actions
            if (self.variant.use_pair or action["action_type"] != "pair")
            and (self.variant.use_stop or action["action_type"] != "stop")
        ]


def build_ablation(
    repository: RuntimeRepository,
    planner: RetrievalPlanner,
    policy: Any,
    variant: Variant,
    *,
    evidence_token_budget: int,
    evidence_unit_budget: int,
    retrieval_limit: int,
    pair_limit: int,
) -> EvidenceAgent:
    """按组件开关组装一个消融方法。"""

    return EvidenceAgent(
        VariantPlanner(planner, variant),
        RepositoryRAG(repository),
        VariantActionBuilder(
            ActionBuilder(repository, pair_limit=pair_limit),
            variant,
        ),
        policy,
        evidence_token_budget=evidence_token_budget,
        evidence_unit_budget=evidence_unit_budget,
        retrieval_limit=retrieval_limit,
    )
