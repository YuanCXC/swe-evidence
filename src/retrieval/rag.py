"""按照 Agent 给出的检索计划执行仓库 RAG。"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from src.data import RuntimeRepository

from .fts_retriever import FTSRetriever
from .path_symbol_retriever import rank_path_units, rank_symbol_units
from .rank_fusion import reciprocal_rank_fusion
from .structure_expander import StructureExpander
from .unit_retriever import (
    collect_candidate_units,
    rank_units_by_query,
    retrieval_terms,
)


class RetrievalPlanLike(Protocol):
    """RAG 执行所需的最小检索计划接口。"""

    target_dimensions: Sequence[str]
    queries: Sequence[str]
    paths: Sequence[str]
    symbols: Sequence[str]
    retrieval_channels: Sequence[str]


class RepositoryRAG:
    """执行检索计划，不生成计划，也不选择具体 Evidence。"""

    def __init__(self, repository: RuntimeRepository) -> None:
        self.repository = repository
        self.fts = FTSRetriever(repository)
        self.structure = StructureExpander(repository)

    def _query_groups(self, plan: RetrievalPlanLike) -> dict[str, list[str]]:
        terms = list(
            dict.fromkeys(
                retrieval_terms(" ".join([*plan.queries, *plan.paths, *plan.symbols]))
            )
        )
        dimensions = list(dict.fromkeys(map(str, plan.target_dimensions))) or [
            "planned_query"
        ]
        return {dimension: terms for dimension in dimensions}

    def _explicit_path_file_ids(
        self,
        task: Mapping[str, Any],
        paths: Sequence[str],
    ) -> list[str]:
        normalized = [path.replace("\\", "/").lower() for path in paths]
        return [
            str(item["file_version_id"])
            for item in self.repository.list_snapshot_files(str(task["snapshot_id"]))
            if any(
                str(item["path"]).lower() == path
                or str(item["path"]).lower().endswith(path)
                for path in normalized
            )
        ]

    def retrieve(
        self,
        task: Mapping[str, Any],
        plan: RetrievalPlanLike,
        current_evidence: Sequence[Mapping[str, Any]] = (),
        *,
        exclude_evidence_ids: Sequence[str] = (),
        limit: int = 64,
        file_limit: int = 32,
        per_query_file_limit: int = 16,
    ) -> list[dict[str, Any]]:
        """返回本轮首次出现的 Evidence Unit 候选。"""

        channels = set(map(str, plan.retrieval_channels))
        query_groups = self._query_groups(plan)
        file_rankings: dict[str, list[str]] = {}
        if channels & {"content", "path", "symbol"}:
            file_groups = self.fts.retrieve(
                task,
                query_groups,
                per_dimension_limit=per_query_file_limit,
            )
            file_rankings = {
                f"fts:{name}": [str(item["file_version_id"]) for item in rows]
                for name, rows in file_groups.items()
            }

        explicit_file_ids = self._explicit_path_file_ids(task, plan.paths)
        if explicit_file_ids:
            file_rankings["explicit_path"] = explicit_file_ids
        fused_files = reciprocal_rank_fusion(file_rankings, limit=file_limit)
        file_version_ids = [str(item["item_id"]) for item in fused_files]

        excluded_ids = set(map(str, exclude_evidence_ids))
        excluded_ids.update(str(unit["evidence_id"]) for unit in current_evidence)
        evidence_units = [
            unit
            for unit in collect_candidate_units(self.repository, file_version_ids)
            if str(unit["evidence_id"]) not in excluded_ids
        ]
        records = {str(unit["evidence_id"]): unit for unit in evidence_units}
        terms = list(
            dict.fromkeys(term for values in query_groups.values() for term in values)
        )
        evidence_rankings: dict[str, list[str]] = {}

        if "content" in channels:
            content_rankings = rank_units_by_query(
                evidence_units,
                query_groups,
                limit=limit,
            )
            fused_content = reciprocal_rank_fusion(content_rankings, limit=limit)
            evidence_rankings["content"] = [
                str(item["item_id"]) for item in fused_content
            ]
        if "path" in channels:
            evidence_rankings["path"] = rank_path_units(
                evidence_units,
                terms,
                plan.paths,
                limit=limit,
            )
        if "symbol" in channels:
            evidence_rankings["symbol"] = rank_symbol_units(
                evidence_units,
                terms,
                plan.symbols,
                limit=limit,
            )
        if "structure" in channels and current_evidence:
            structure_units = self.structure.expand(
                task,
                current_evidence,
                excluded_evidence_ids=exclude_evidence_ids,
                limit=limit,
            )
            for unit in structure_units:
                records[str(unit["evidence_id"])] = unit
            evidence_rankings["structure"] = [
                str(unit["evidence_id"]) for unit in structure_units
            ]

        fused = reciprocal_rank_fusion(evidence_rankings, limit=limit)
        return [
            {
                **records[str(item["item_id"])],
                "retrieval_score": float(item["rrf_score"]),
                "retrieval_sources": list(item["sources"]),
                "retrieval_source_ranks": dict(item["source_ranks"]),
            }
            for item in fused
            if str(item["item_id"]) in records
        ]
