"""读取冻结 Policy states 引用的 Evidence Unit 文本子集。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pyarrow.dataset as ds


POLICY_EVIDENCE_COLUMNS = (
    "evidence_id",
    "file_version_id",
    "repo",
    "path",
    "blob_oid",
    "unit_type",
    "symbol",
    "qualified_name",
    "start_line",
    "end_line",
    "parent_evidence_id",
    "content",
    "content_sha256",
    "token_count",
    "rendered_token_count",
    "scoreable",
)


class PolicyEvidenceReader:
    """为状态级 Policy 训练、评价和消融提供 Evidence 文本。"""

    def __init__(self, evidence_path: Path | str) -> None:
        self.path = Path(evidence_path)
        self.dataset = ds.dataset(self.path, format="parquet")

    def iter_evidence(
        self,
        *,
        columns: Sequence[str] = POLICY_EVIDENCE_COLUMNS,
        scoreable_only: bool = False,
        batch_size: int = 1024,
    ) -> Iterator[dict[str, Any]]:
        """流式读取冻结 Evidence 子集。"""

        expression = ds.field("scoreable") == True if scoreable_only else None  # noqa: E712
        for batch in self.dataset.to_batches(
            columns=list(columns),
            filter=expression,
            batch_size=batch_size,
        ):
            yield from batch.to_pylist()

    def get_many(self, evidence_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """按 evidence_id 批量读取 Evidence，并以稳定 ID 建立映射。"""

        requested = list(dict.fromkeys(map(str, evidence_ids)))
        if not requested:
            return {}
        table = self.dataset.to_table(
            columns=list(POLICY_EVIDENCE_COLUMNS),
            filter=ds.field("evidence_id").isin(requested),
        )
        return {str(row["evidence_id"]): row for row in table.to_pylist()}

    def get(self, evidence_id: str) -> dict[str, Any]:
        """读取一个 Evidence Unit。"""

        return self.get_many([evidence_id])[evidence_id]
