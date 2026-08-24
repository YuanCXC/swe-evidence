"""文件级 Dense Retrieval Baseline。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

import numpy as np
from openai import OpenAI

from exp.api_usage import record_api_usage
from src.data import RuntimeRepository
from src.evaluation import apply_budget

from .bm25 import task_query
from .chunking import TokenChunker


class DenseEncoder:
    """通过 OpenAI-compatible Embeddings API 编码查询和仓库文件。"""

    def __init__(
        self,
        model_name: str,
        *,
        api_base: str | None,
        api_keys: Sequence[str],
        timeout: float,
        max_retries: int,
        batch_size: int,
        cache_path: Path | str,
        chunker: TokenChunker,
    ) -> None:
        self.model_name = model_name
        self.clients = [
            OpenAI(
                api_key=api_key,
                base_url=api_base,
                timeout=timeout,
                max_retries=max_retries,
            )
            for api_key in api_keys
        ]
        self.batch_size = batch_size
        self.chunker = chunker
        self.call_count = 0
        self.pool_lock = Lock()
        self.cache_lock = Lock()
        cache = Path(cache_path).resolve()
        cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache
        self.cache = sqlite3.connect(cache, check_same_thread=False)
        self.cache.execute(
            "CREATE TABLE IF NOT EXISTS file_chunk_embeddings ("
            "file_version_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, "
            "vector BLOB NOT NULL, PRIMARY KEY(file_version_id, chunk_index))"
        )

    def _read_cached_vectors(self, file_ids: Sequence[str]) -> dict[str, np.ndarray]:
        grouped: dict[str, list[np.ndarray]] = {}
        with self.cache_lock:
            for offset in range(0, len(file_ids), 500):
                chunk = list(file_ids[offset : offset + 500])
                placeholders = ",".join("?" for _ in chunk)
                rows = self.cache.execute(
                    "SELECT file_version_id, vector FROM file_chunk_embeddings "
                    f"WHERE file_version_id IN ({placeholders}) "
                    "ORDER BY file_version_id, chunk_index",
                    chunk,
                )
                for file_version_id, vector in rows:
                    grouped.setdefault(str(file_version_id), []).append(
                        np.frombuffer(vector, dtype=np.float32).copy()
                    )
        return {
            file_version_id: np.stack(vectors)
            for file_version_id, vectors in grouped.items()
        }

    def _embed(self, texts: Sequence[str], *, usage_scope: str = "api") -> np.ndarray:
        vectors = []
        for offset in range(0, len(texts), self.batch_size):
            with self.pool_lock:
                client = self.clients[self.call_count % len(self.clients)]
                self.call_count += 1
            response = client.embeddings.create(
                model=self.model_name,
                input=list(texts[offset : offset + self.batch_size]),
                encoding_format="float",
            )
            record_api_usage(response.usage, scope=usage_scope)
            vectors.extend(
                np.asarray(item.embedding, dtype=np.float32)
                for item in sorted(response.data, key=lambda item: item.index)
            )
        matrix = np.stack(vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / norms

    def _document_vectors(
        self,
        repository: RuntimeRepository,
        files: Sequence[Mapping[str, Any]],
    ) -> list[np.ndarray]:
        file_ids = [str(item["file_version_id"]) for item in files]
        cached = self._read_cached_vectors(file_ids)
        missing_files = [
            item for item in files if str(item["file_version_id"]) not in cached
        ]
        if missing_files:
            chunk_records = []
            for item in missing_files:
                file_record = repository.get_file_version(str(item["file_version_id"]))
                file_id = str(item["file_version_id"])
                chunks = self.chunker.split(
                    str(file_record.get("content") or ""),
                    path=str(item["path"]),
                )
                chunk_records.extend(
                    (file_id, chunk_index, text)
                    for chunk_index, text in enumerate(chunks)
                )
            encoded = self._embed(
                [record[2] for record in chunk_records],
                usage_scope="index_api",
            )
            additions = []
            grouped: dict[str, list[np.ndarray]] = {}
            for (file_id, chunk_index, _), vector in zip(chunk_records, encoded):
                grouped.setdefault(file_id, []).append(vector)
                additions.append((file_id, chunk_index, vector.tobytes()))
            cached.update(
                {
                    file_id: np.stack(vectors)
                    for file_id, vectors in grouped.items()
                }
            )
            with self.cache_lock:
                self.cache.executemany(
                    "INSERT OR REPLACE INTO file_chunk_embeddings"
                    "(file_version_id, chunk_index, vector) VALUES (?, ?, ?)",
                    additions,
                )
                self.cache.commit()
        return [cached[file_id] for file_id in file_ids]

    def rank_files(
        self,
        repository: RuntimeRepository,
        *,
        snapshot_id: str,
        query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """按查询与文件向量的余弦相似度返回文件排名。"""

        files = repository.list_snapshot_files(snapshot_id)

        query_vector = self._embed([query])[0]
        document_vectors = self._document_vectors(repository, files)
        scores = np.asarray(
            [float(np.max(vectors @ query_vector)) for vectors in document_vectors]
        )

        ranked_indices = sorted(
            range(len(files)),
            key=lambda index: (-float(scores[index]), str(files[index]["path"])),
        )[:limit]
        return [
            {
                "file_version_id": str(files[index]["file_version_id"]),
                "path": str(files[index]["path"]),
                "dense_score": float(scores[index]),
            }
            for index in ranked_indices
        ]


def file_ranking_result(
    task: Mapping[str, Any],
    repository: RuntimeRepository,
    ranked_files: Sequence[Mapping[str, Any]],
    *,
    evidence_token_budget: int,
    evidence_unit_budget: int,
    source_name: str,
) -> dict[str, Any]:
    """把文件排名映射为统一 Evidence Package 与轨迹结构。"""

    candidates = []
    for file_rank, file_record in enumerate(ranked_files, start=1):
        units = repository.get_file_evidence(
            str(file_record["file_version_id"]),
            scoreable_only=True,
        )
        candidates.extend(
            {
                **unit,
                "retrieval_score": float(file_record["score"]),
                "retrieval_sources": list(
                    file_record.get("sources") or [source_name]
                ),
                "retrieval_source_ranks": dict(
                    file_record.get("source_ranks") or {source_name: file_rank}
                ),
            }
            for unit in units
        )

    evidence = apply_budget(
        candidates,
        max_units=evidence_unit_budget,
        max_tokens=evidence_token_budget,
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
        "ranked_files": [dict(item) for item in ranked_files],
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


class DenseBaseline:
    """使用 Issue 与仓库文件的向量相似度执行一次检索。"""

    def __init__(
        self,
        repository: RuntimeRepository,
        encoder: DenseEncoder,
        *,
        evidence_token_budget: int,
        evidence_unit_budget: int,
        file_limit: int,
    ) -> None:
        self.repository = repository
        self.encoder = encoder
        self.evidence_token_budget = evidence_token_budget
        self.evidence_unit_budget = evidence_unit_budget
        self.file_limit = file_limit

    def run(self, task: Mapping[str, Any]) -> dict[str, Any]:
        """检索文件并转换成遵守统一预算的 Evidence Package。"""

        ranked_files = self.encoder.rank_files(
            self.repository,
            snapshot_id=str(task["snapshot_id"]),
            query=task_query(task["input"]),
            limit=self.file_limit,
        )
        return file_ranking_result(
            task,
            self.repository,
            [
                {
                    **item,
                    "score": float(item["dense_score"]),
                    "sources": ["dense_file"],
                }
                for item in ranked_files
            ],
            evidence_token_budget=self.evidence_token_budget,
            evidence_unit_budget=self.evidence_unit_budget,
            source_name="dense_file",
        )
