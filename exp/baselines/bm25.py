"""使用 runtime 文件级 FTS5 BM25 的一次性检索 Baseline。"""

from __future__ import annotations

from typing import Any, Mapping

from src.data import RuntimeRepository
from src.evaluation import apply_budget
from src.retrieval.unit_retriever import retrieval_terms


def task_query(task_input: Mapping[str, Any]) -> str:
    pieces = [str(task_input.get("problem_statement") or "")]
    hints = task_input.get("hints")
    if isinstance(hints, str) and hints.strip():
        pieces.append(hints)
    elif isinstance(hints, list):
        pieces.extend(str(item) for item in hints if str(item).strip())
    return "\n".join(pieces)


class BM25Baseline:
    """按文件 BM25 排名映射 Evidence Units，并施加统一预算。"""

    def __init__(
        self,
        repository: RuntimeRepository,
        *,
        evidence_token_budget: int,
        evidence_unit_budget: int,
        file_limit: int,
    ) -> None:
        self.repository = repository
        self.evidence_token_budget = evidence_token_budget
        self.evidence_unit_budget = evidence_unit_budget
        self.file_limit = file_limit

    def run(self, task: Mapping[str, Any]) -> dict[str, Any]:
        """运行一次文件检索，并按文件排名和代码顺序生成证据包。"""

        terms = retrieval_terms(task_query(task["input"]))[:64]
        ranked_files = self.repository.search_files(
            snapshot_id=str(task["snapshot_id"]),
            repo=str(task["input"]["repo"]),
            terms=terms,
            limit=self.file_limit,
        )
        candidates = []
        for file_rank, file_record in enumerate(ranked_files, start=1):
            units = self.repository.get_file_evidence(
                str(file_record["file_version_id"]),
                scoreable_only=True,
            )
            candidates.extend(
                {
                    **unit,
                    "retrieval_score": float(file_record["fts_score"]),
                    "retrieval_sources": ["bm25_file"],
                    "retrieval_source_ranks": {"bm25_file": file_rank},
                }
                for unit in units
            )

        evidence = apply_budget(
            candidates,
            max_units=self.evidence_unit_budget,
            max_tokens=self.evidence_token_budget,
        )
        evidence_ids = [str(unit["evidence_id"]) for unit in evidence]
        evidence_tokens = sum(
            int(unit.get("rendered_token_count") or 0) for unit in evidence
        )
        candidate_ids = [str(unit["evidence_id"]) for unit in candidates]
        return {
            "task_id": str(task["task_id"]),
            "snapshot_id": str(task["snapshot_id"]),
            "evidence_package": evidence,
            "final_evidence_ids": evidence_ids,
            "final_evidence_tokens": evidence_tokens,
            "retrieved_evidence_ids": candidate_ids,
            "ranked_files": [
                {
                    "path": str(item["path"]),
                    "file_version_id": str(item["file_version_id"]),
                    "score": float(item["fts_score"]),
                }
                for item in ranked_files
            ],
            "retrieval_rounds": [
                {
                    "round": 0,
                    "candidate_evidence_ids": candidate_ids,
                }
            ],
            "steps": [
                {
                    "step": 0,
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
            "planner_calls": 0,
        }
