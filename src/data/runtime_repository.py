"""只读访问完整修复前仓库、Evidence Units 和文件级 FTS。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence


def repository_token(repo: str) -> str:
    """生成与冻结 V2.10 FTS 索引一致的仓库 Token。"""

    digest = hashlib.sha1(repo.lower().encode("utf-8")).hexdigest()[:20]
    return f"repo{digest}"


def quote_fts_term(term: str) -> str:
    """将一个查询词编码为 FTS5 phrase。"""

    return '"' + term.replace('"', '""') + '"'


def slice_evidence_content(file_content: str, start_line: int, end_line: int) -> str:
    """按冻结数据的 1-based 闭区间规则截取 Evidence 正文。"""

    lines = file_content.splitlines()
    return "\n".join(lines[start_line - 1 : end_line])


class RuntimeRepository:
    """访问 `repository_runtime.sqlite3` 中的在线安全仓库数据。"""

    def __init__(self, database_path: Path | str) -> None:
        self.path = Path(database_path).resolve()
        self.connection = sqlite3.connect(
            self.path.as_uri() + "?mode=ro",
            uri=True,
        )
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        """关闭只读数据库连接。"""

        self.connection.close()

    def __enter__(self) -> RuntimeRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        """读取一个冻结仓库快照。"""

        row = self.connection.execute(
            "SELECT payload_json FROM snapshots WHERE snapshot_id=?",
            (snapshot_id,),
        ).fetchone()
        return json.loads(row["payload_json"])

    def list_snapshot_files(self, snapshot_id: str) -> list[dict[str, Any]]:
        """列出快照中的路径与文件版本，不加载文件正文。"""

        rows = self.connection.execute(
            "SELECT m.path, m.file_version_id, fv.repo, fv.blob_oid, fv.status "
            "FROM snapshot_file_memberships AS m "
            "JOIN file_versions AS fv ON fv.file_version_id=m.file_version_id "
            "WHERE m.snapshot_id=? ORDER BY m.path",
            (snapshot_id,),
        )
        return [dict(row) for row in rows]

    def snapshot_file_ids(self, snapshot_id: str) -> set[str]:
        """返回快照包含的全部 file_version_id。"""

        rows = self.connection.execute(
            "SELECT file_version_id FROM snapshot_file_memberships WHERE snapshot_id=?",
            (snapshot_id,),
        )
        return {str(row["file_version_id"]) for row in rows}

    def get_file(self, snapshot_id: str, path: str) -> dict[str, Any]:
        """按快照和路径读取完整文件版本。"""

        row = self.connection.execute(
            "SELECT fv.payload_json FROM snapshot_file_memberships AS m "
            "JOIN file_versions AS fv ON fv.file_version_id=m.file_version_id "
            "WHERE m.snapshot_id=? AND m.path=?",
            (snapshot_id, path.replace("\\", "/")),
        ).fetchone()
        return json.loads(row["payload_json"])

    def get_file_version(self, file_version_id: str) -> dict[str, Any]:
        """按 file_version_id 读取完整文件版本。"""

        row = self.connection.execute(
            "SELECT payload_json FROM file_versions WHERE file_version_id=?",
            (file_version_id,),
        ).fetchone()
        return json.loads(row["payload_json"])

    def search_files(
        self,
        *,
        snapshot_id: str,
        repo: str,
        terms: Sequence[str],
        limit: int = 64,
    ) -> list[dict[str, Any]]:
        """按冻结 FTS5 排名查询当前快照中的文件。"""

        unique_terms = list(dict.fromkeys(str(term) for term in terms if str(term)))
        if not unique_terms:
            return []
        match_query = (
            quote_fts_term(repository_token(repo))
            + " AND ("
            + " OR ".join(quote_fts_term(term) for term in unique_terms)
            + ")"
        )
        membership = self.snapshot_file_ids(snapshot_id)
        rows = self.connection.execute(
            "SELECT map.file_version_id, map.path, policy_file_fts.rank AS fts_score "
            "FROM policy_file_fts "
            "JOIN policy_file_fts_map AS map ON map.rowid=policy_file_fts.rowid "
            "WHERE policy_file_fts MATCH ? "
            "AND rank MATCH 'bm25(0.0, 1.5, 1.0)' "
            "ORDER BY rank ASC, policy_file_fts.rowid ASC",
            (match_query,),
        )

        selected: list[dict[str, Any]] = []
        for row in rows:
            file_version_id = str(row["file_version_id"])
            if file_version_id not in membership:
                continue
            selected.append(
                {
                    "file_version_id": file_version_id,
                    "path": str(row["path"]),
                    "fts_score": float(row["fts_score"]),
                }
            )
            if len(selected) >= limit:
                break
        return selected

    def _hydrate_evidence_row(self, row: sqlite3.Row) -> dict[str, Any]:
        unit = json.loads(row["unit_payload"])
        file_record = json.loads(row["file_payload"])
        start_line = int(unit["start_line"])
        end_line = int(unit["end_line"])
        unit.update(
            {
                "file_version_id": str(row["file_version_id"]),
                "repo": str(file_record["repo"]),
                "path": str(file_record["path"]),
                "blob_oid": str(file_record["blob_oid"]),
                "content": slice_evidence_content(
                    str(file_record["content"]),
                    start_line,
                    end_line,
                ),
            }
        )
        return unit

    def get_evidence(self, evidence_id: str) -> dict[str, Any]:
        """按 evidence_id 读取并补全一个 Evidence Unit。"""

        row = self.connection.execute(
            "SELECT eu.file_version_id, eu.payload_json AS unit_payload, "
            "fv.payload_json AS file_payload "
            "FROM evidence_units AS eu "
            "JOIN file_versions AS fv ON fv.file_version_id=eu.file_version_id "
            "WHERE eu.evidence_id=?",
            (evidence_id,),
        ).fetchone()
        return self._hydrate_evidence_row(row)

    def get_evidence_many(
        self, evidence_ids: Iterable[str]
    ) -> dict[str, dict[str, Any]]:
        """批量读取并补全 Evidence Units。"""

        requested = list(dict.fromkeys(map(str, evidence_ids)))
        if not requested:
            return {}
        placeholders = ",".join("?" for _ in requested)
        rows = self.connection.execute(
            "SELECT eu.file_version_id, eu.payload_json AS unit_payload, "
            "fv.payload_json AS file_payload "
            "FROM evidence_units AS eu "
            "JOIN file_versions AS fv ON fv.file_version_id=eu.file_version_id "
            f"WHERE eu.evidence_id IN ({placeholders})",
            requested,
        )
        records = [self._hydrate_evidence_row(row) for row in rows]
        return {str(record["evidence_id"]): record for record in records}

    def get_file_evidence(
        self,
        file_version_id: str,
        *,
        scoreable_only: bool = True,
    ) -> list[dict[str, Any]]:
        """读取一个文件版本下的全部 Evidence Units。"""

        scoreable_clause = " AND eu.scoreable=1" if scoreable_only else ""
        rows = self.connection.execute(
            "SELECT eu.file_version_id, eu.payload_json AS unit_payload, "
            "fv.payload_json AS file_payload "
            "FROM evidence_units AS eu "
            "JOIN file_versions AS fv ON fv.file_version_id=eu.file_version_id "
            "WHERE eu.file_version_id=?" + scoreable_clause,
            (file_version_id,),
        )
        records = [self._hydrate_evidence_row(row) for row in rows]
        return sorted(
            records,
            key=lambda item: (
                int(item["start_line"]),
                int(item["end_line"]),
                str(item["evidence_id"]),
            ),
        )

    def get_structure_context(self, evidence_id: str) -> dict[str, Any]:
        """读取 parent、children、siblings、同文件单元和 imports。"""

        current = self.get_evidence(evidence_id)
        file_units = self.get_file_evidence(
            str(current["file_version_id"]),
            scoreable_only=True,
        )
        parent_id = current.get("parent_evidence_id")
        parent = self.get_evidence(str(parent_id)) if parent_id else None
        children = [
            unit for unit in file_units if unit.get("parent_evidence_id") == evidence_id
        ]
        siblings = [
            unit
            for unit in file_units
            if parent_id
            and unit.get("parent_evidence_id") == parent_id
            and unit["evidence_id"] != evidence_id
        ]
        file_record = self.get_file_version(str(current["file_version_id"]))
        return {
            "current": current,
            "parent": parent,
            "children": children,
            "siblings": siblings,
            "same_file": [
                unit for unit in file_units if unit["evidence_id"] != evidence_id
            ],
            "imports": list(file_record.get("imports") or []),
        }
