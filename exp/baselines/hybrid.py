"""BM25 与 Dense 文件排名融合 Baseline。"""

from __future__ import annotations

from typing import Any, Mapping

from src.data import RuntimeRepository
from src.retrieval.rank_fusion import reciprocal_rank_fusion
from src.retrieval.unit_retriever import retrieval_terms

from .bm25 import task_query
from .dense import DenseEncoder, file_ranking_result


class HybridBaseline:
    """使用 RRF 融合文件级 BM25 与 Dense 排名。"""

    def __init__(
        self,
        repository: RuntimeRepository,
        encoder: DenseEncoder,
        *,
        evidence_token_budget: int,
        evidence_unit_budget: int,
        file_limit: int,
        candidate_file_limit: int,
        rank_constant: int,
    ) -> None:
        self.repository = repository
        self.encoder = encoder
        self.evidence_token_budget = evidence_token_budget
        self.evidence_unit_budget = evidence_unit_budget
        self.file_limit = file_limit
        self.candidate_file_limit = candidate_file_limit
        self.rank_constant = rank_constant

    def run(self, task: Mapping[str, Any]) -> dict[str, Any]:
        """分别执行两路文件检索，再将融合结果映射为证据包。"""

        query = task_query(task["input"])
        bm25_files = self.repository.search_files(
            snapshot_id=str(task["snapshot_id"]),
            repo=str(task["input"]["repo"]),
            terms=retrieval_terms(query)[:64],
            limit=self.candidate_file_limit,
        )
        dense_files = self.encoder.rank_files(
            self.repository,
            snapshot_id=str(task["snapshot_id"]),
            query=query,
            limit=self.candidate_file_limit,
        )
        file_records = {
            str(item["file_version_id"]): item for item in (*bm25_files, *dense_files)
        }
        fused = reciprocal_rank_fusion(
            {
                "bm25_file": [str(item["file_version_id"]) for item in bm25_files],
                "dense_file": [
                    str(item["file_version_id"]) for item in dense_files
                ],
            },
            rank_constant=self.rank_constant,
            limit=self.file_limit,
        )
        ranked_files = [
            {
                "file_version_id": str(item["item_id"]),
                "path": str(file_records[str(item["item_id"])]["path"]),
                "score": float(item["rrf_score"]),
                "sources": list(item["sources"]),
                "source_ranks": dict(item["source_ranks"]),
            }
            for item in fused
        ]
        return file_ranking_result(
            task,
            self.repository,
            ranked_files,
            evidence_token_budget=self.evidence_token_budget,
            evidence_unit_budget=self.evidence_unit_budget,
            source_name="hybrid_file",
        )
