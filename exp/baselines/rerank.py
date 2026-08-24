"""BM25 候选与 API 重排序 Baseline。"""

from __future__ import annotations

import random
import time
from threading import Lock
from typing import Any, Mapping, Sequence

import httpx

from exp.api_usage import record_api_usage
from src.data import RuntimeRepository
from src.retrieval.unit_retriever import retrieval_terms

from .bm25 import task_query
from .dense import file_ranking_result


class RerankCaller:
    """调用 OpenAI-compatible Provider 提供的独立 Rerank 接口。"""

    def __init__(
        self,
        model_name: str,
        *,
        api_base: str | None,
        api_keys: Sequence[str],
        timeout: float,
        max_retries: int,
        max_chunks_per_doc: int,
        overlap_tokens: int,
    ) -> None:
        self.model_name = model_name
        self.clients = [
            httpx.Client(
                base_url=str(api_base).rstrip("/") + "/",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout,
            )
            for api_key in api_keys
        ]
        self.max_chunks_per_doc = max_chunks_per_doc
        self.overlap_tokens = overlap_tokens
        self.max_retries = max_retries
        self.call_count = 0
        self.pool_lock = Lock()

    def rank(
        self,
        *,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> list[dict[str, Any]]:
        """返回文档原始下标与相关性分数。"""

        with self.pool_lock:
            client = self.clients[self.call_count % len(self.clients)]
            self.call_count += 1
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": list(documents),
            "top_n": top_n,
            "return_documents": False,
            "max_chunks_per_doc": self.max_chunks_per_doc,
            "overlap_tokens": self.overlap_tokens,
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = client.post("rerank", json=payload)
                if response.status_code in {429, 500, 502, 503, 504}:
                    if attempt == self.max_retries:
                        response.raise_for_status()
                else:
                    break
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
            time.sleep(min(2**attempt + random.random(), 30.0))
        response.raise_for_status()
        result = response.json()
        record_api_usage(result.get("usage"))
        if not isinstance(result.get("results"), list):
            raise ValueError("Rerank API 响应缺少 results 数组")
        return sorted(
            result["results"],
            key=lambda item: (-float(item["relevance_score"]), int(item["index"])),
        )


class RerankBaseline:
    """先执行 BM25 文件召回，再由 API Cross-Encoder 重排序。"""

    def __init__(
        self,
        repository: RuntimeRepository,
        caller: RerankCaller,
        *,
        evidence_token_budget: int,
        evidence_unit_budget: int,
        file_limit: int,
        candidate_file_limit: int,
    ) -> None:
        self.repository = repository
        self.caller = caller
        self.evidence_token_budget = evidence_token_budget
        self.evidence_unit_budget = evidence_unit_budget
        self.file_limit = file_limit
        self.candidate_file_limit = candidate_file_limit

    def run(self, task: Mapping[str, Any]) -> dict[str, Any]:
        """对 BM25 候选文件进行 Issue–文件相关性重排序。"""

        query = task_query(task["input"])
        candidates = self.repository.search_files(
            snapshot_id=str(task["snapshot_id"]),
            repo=str(task["input"]["repo"]),
            terms=retrieval_terms(query)[:64],
            limit=self.candidate_file_limit,
        )
        if not candidates:
            return file_ranking_result(
                task,
                self.repository,
                [],
                evidence_token_budget=self.evidence_token_budget,
                evidence_unit_budget=self.evidence_unit_budget,
                source_name="rerank_file",
            )
        documents = []
        for item in candidates:
            file_record = self.repository.get_file_version(
                str(item["file_version_id"])
            )
            documents.append(
                f"文件路径：{item['path']}\n\n{file_record.get('content') or ''}"
            )
        reranked = self.caller.rank(
            query=query,
            documents=documents,
            top_n=self.file_limit,
        )
        ranked_files = [
            {
                "file_version_id": str(candidates[int(item["index"])]["file_version_id"]),
                "path": str(candidates[int(item["index"])]["path"]),
                "score": float(item["relevance_score"]),
                "sources": ["bm25_file", "rerank_file"],
                "source_ranks": {
                    "bm25_file": int(item["index"]) + 1,
                    "rerank_file": rank,
                },
            }
            for rank, item in enumerate(reranked, start=1)
        ]
        return file_ranking_result(
            task,
            self.repository,
            ranked_files,
            evidence_token_budget=self.evidence_token_budget,
            evidence_unit_budget=self.evidence_unit_budget,
            source_name="rerank_file",
        )
