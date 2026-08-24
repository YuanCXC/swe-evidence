"""将官方 SweRank 输出适配为统一 Evidence Package。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data import RuntimeRepository
from src.evaluation import adapt_outputs


def _parse_document_id(document_id: str, score: float | None) -> dict[str, Any]:
    normalized = document_id.replace("\\", "/").removeprefix("./")
    if ".py/" in normalized:
        file_stem, symbol = normalized.split(".py/", maxsplit=1)
        return {
            "path": file_stem + ".py",
            "symbol": symbol.strip("/").replace("/", "."),
            "score": score,
            "external_id": document_id,
        }
    if ".py:" in normalized:
        file_stem, symbol = normalized.split(".py:", maxsplit=1)
        return {
            "path": file_stem + ".py",
            "symbol": symbol,
            "score": score,
            "external_id": document_id,
        }
    return {
        "path": normalized,
        "score": score,
        "external_id": document_id,
    }


class SweRankOutputStore:
    """读取 SweRank Retriever JSONL 或官方 Reranker 输出目录。"""

    def __init__(self, output_path: Path | str) -> None:
        self.path = Path(output_path).resolve()
        self.metadata: dict[str, dict[str, Any]] = {}
        self.outputs = (
            self._read_reranker_directory()
            if self.path.is_dir()
            else self._read_retriever_jsonl()
        )

    def _read_retriever_jsonl(self) -> dict[str, list[dict[str, Any]]]:
        outputs: dict[str, list[dict[str, Any]]] = {}
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                task_id = str(row.get("task_id") or row["instance_id"])
                self.metadata[task_id] = {
                    "repo": row.get("repo"),
                    "base_commit": row.get("base_commit"),
                }
                outputs[task_id] = [
                    _parse_document_id(str(document_id), None)
                    for document_id in row["docs"]
                ]
        return outputs

    def _read_reranker_directory(self) -> dict[str, list[dict[str, Any]]]:
        outputs: dict[str, list[dict[str, Any]]] = {}
        for result_path in self.path.rglob("rerank_100_llm_gen_num.json"):
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                payload = payload[0]
            for task_id, scores in payload.items():
                ranked = sorted(
                    scores.items(),
                    key=lambda item: (-float(item[1]), str(item[0])),
                )
                outputs[str(task_id)] = [
                    _parse_document_id(str(document_id), float(score))
                    for document_id, score in ranked
                ]
        return outputs

    def get(self, task_id: str) -> list[dict[str, Any]]:
        """读取一个任务的有序定位结果。"""

        return self.outputs[str(task_id)]

    def task_key(self, task: Mapping[str, Any]) -> str | None:
        """把本项目 task_id 映射到外部结果使用的 instance_id。"""

        candidates = [str(task["task_id"])]
        candidates.extend(
            str(item["source_id"])
            for item in task.get("provenance") or []
            if item.get("source_id")
        )
        return next((key for key in candidates if key in self.outputs), None)

    def contains(self, task: Mapping[str, Any]) -> bool:
        """判断冻结输出是否包含当前任务。"""

        return self.task_key(task) is not None

    def get_for_task(
        self,
        task: Mapping[str, Any],
    ) -> tuple[str, list[dict[str, Any]], bool]:
        """读取任务输出，并核对 Retriever JSONL 中可用的 commit 元数据。"""

        task_key = self.task_key(task)
        if task_key is None:
            raise KeyError(str(task["task_id"]))
        metadata = self.metadata.get(task_key) or {}
        expected_commit = metadata.get("base_commit")
        snapshot_verified = bool(expected_commit)
        if expected_commit and str(expected_commit) != str(task["input"]["base_commit"]):
            raise ValueError(f"{task_key} 的 base_commit 与当前任务快照不一致")
        return task_key, self.outputs[task_key], snapshot_verified


def _result(
    task: Mapping[str, Any],
    external_task_id: str,
    outputs: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    *,
    requested_top_k: int,
    snapshot_verified: bool,
) -> dict[str, Any]:
    evidence_ids = [str(unit["evidence_id"]) for unit in candidates]
    evidence_tokens = sum(
        int(unit.get("rendered_token_count") or 0) for unit in candidates
    )
    mapped_ranks = {int(unit["source_rank"]) for unit in candidates}
    return {
        "task_id": str(task["task_id"]),
        "snapshot_id": str(task["snapshot_id"]),
        "external_task_id": external_task_id,
        "evidence_package": list(candidates),
        "final_evidence_ids": evidence_ids,
        "final_evidence_tokens": evidence_tokens,
        "retrieved_evidence_ids": evidence_ids,
        "external_outputs": [dict(item) for item in outputs],
        "rank_cutoff": requested_top_k,
        "external_output_count": len(outputs),
        "mapped_external_output_count": len(mapped_ranks),
        "external_mapping_rate": len(mapped_ranks) / len(outputs) if outputs else 0.0,
        "snapshot_verified": snapshot_verified,
        "execution_cost_observed": False,
        "trajectory_observed": False,
        "retrieval_rounds": [
            {
                "round": 0,
                "candidate_evidence_ids": evidence_ids,
            }
        ],
        "steps": [
            {
                "step": 0,
                "action_type": "stop",
                "action_id": "external_output_stop",
                "added_evidence_ids": evidence_ids,
                "added_token_count": evidence_tokens,
                "tool_calls": 0,
                "forced": True,
                "termination_reason": "rank_cutoff",
            }
        ],
        "termination_reason": "rank_cutoff",
        "planner_calls": 0,
    }


class SweRankBaseline:
    """把 SweRank 函数排名转换为当前 snapshot 中的 Evidence Units。"""

    def __init__(
        self,
        repository: RuntimeRepository,
        outputs: SweRankOutputStore,
        *,
        top_k: int,
    ) -> None:
        self.repository = repository
        self.outputs = outputs
        self.top_k = top_k

    def run(self, task: Mapping[str, Any]) -> dict[str, Any]:
        """读取一个任务的 SweRank 排名，并按官方 k 截取函数结果。"""

        external_task_id, all_outputs, snapshot_verified = self.outputs.get_for_task(task)
        outputs = all_outputs[: self.top_k]
        snapshot_files = {
            str(item["path"]): item
            for item in self.repository.list_snapshot_files(str(task["snapshot_id"]))
        }
        evidence_units = []
        for path in dict.fromkeys(str(item["path"]) for item in outputs):
            file_record = snapshot_files.get(path)
            if file_record is None:
                continue
            evidence_units.extend(
                self.repository.get_file_evidence(
                    str(file_record["file_version_id"]),
                    scoreable_only=True,
                )
            )
        adapted = adapt_outputs(outputs, evidence_units)
        candidates = [
            {
                **unit,
                "retrieval_score": float(
                    unit.get("source_score")
                    if unit.get("source_score") is not None
                    else 1.0 / int(unit["source_rank"])
                ),
                "retrieval_sources": ["swerank"],
                "retrieval_source_ranks": {
                    "swerank": int(unit["source_rank"]),
                },
            }
            for unit in adapted
        ]
        return _result(
            task,
            external_task_id,
            outputs,
            candidates,
            requested_top_k=self.top_k,
            snapshot_verified=snapshot_verified,
        )
