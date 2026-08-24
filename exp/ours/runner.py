"""组装完整 Evidence Agent 主实验。"""

from __future__ import annotations

from typing import Any

from src.agents import EvidenceAgent, RetrievalPlanner
from src.data import RuntimeRepository
from src.policy import ActionBuilder
from src.retrieval import RepositoryRAG


def build_ours(
    repository: RuntimeRepository,
    planner: RetrievalPlanner,
    policy: Any,
    *,
    evidence_token_budget: int,
    evidence_unit_budget: int,
    retrieval_limit: int,
    pair_limit: int,
) -> EvidenceAgent:
    """使用完整 Planner、RAG、动作空间和训练 Policy 构造主方法。"""

    return EvidenceAgent(
        planner,
        RepositoryRAG(repository),
        ActionBuilder(repository, pair_limit=pair_limit),
        policy,
        evidence_token_budget=evidence_token_budget,
        evidence_unit_budget=evidence_unit_budget,
        retrieval_limit=retrieval_limit,
    )
