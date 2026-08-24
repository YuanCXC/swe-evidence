"""基于正式 runtime FTS5 索引执行文件召回。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.data import RuntimeRepository


class FTSRetriever:
    """执行检索计划给出的文件级 FTS 查询。"""

    def __init__(self, repository: RuntimeRepository) -> None:
        self.repository = repository

    def retrieve(
        self,
        task: Mapping[str, Any],
        query_groups: Mapping[str, Sequence[str]],
        *,
        per_dimension_limit: int = 16,
        term_limit: int = 12,
    ) -> dict[str, list[dict[str, Any]]]:
        """返回每组检索词对应的文件排名。"""

        return {
            name: self.repository.search_files(
                snapshot_id=str(task["snapshot_id"]),
                repo=str(task["input"]["repo"]),
                terms=list(terms)[:term_limit],
                limit=per_dimension_limit,
            )
            for name, terms in query_groups.items()
        }
