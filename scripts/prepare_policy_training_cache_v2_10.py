#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 V2.10 Evidence Policy 训练构建只读数据侧 Evidence Cache。

这个脚本不会修改冻结的 V2.10 release。它只做两次顺序扫描：

1. 扫描 train / validation / benchmark 的 policy states，收集真正被引用的 evidence_id；
2. 顺序扫描 repository_corpus_v2_10.parquet，只物化上述 Evidence 的代码片段。

最终得到一个训练侧 SQLite：
    data/.train_cache/policy_evidence_v2_10.sqlite3

训练时无需把 7.3GB repository corpus 全部加载到内存，也无需随机扫描 Parquet。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_DATASET_DIR = Path("data/upstream/unified_swe_dataset_v2_10")
DEFAULT_OUTPUT = Path("data/.train_cache/policy_evidence_v2_10.sqlite3")
DEFAULT_SPLITS = ("train", "validation", "benchmark")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def open_cache(path: Path, *, rebuild: bool) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rebuild and path.exists():
        path.unlink()

    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-262144")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS requested_evidence (
            evidence_id TEXT PRIMARY KEY
        ) WITHOUT ROWID;

        CREATE TABLE IF NOT EXISTS evidence (
            evidence_id TEXT PRIMARY KEY,
            file_version_id TEXT NOT NULL,
            path TEXT NOT NULL,
            unit_type TEXT NOT NULL,
            symbol TEXT,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            content TEXT NOT NULL,
            rendered_body TEXT NOT NULL,
            metadata TEXT NOT NULL,
            rendered_token_count INTEGER NOT NULL
        ) WITHOUT ROWID;
        """
    )
    connection.commit()
    return connection


def set_meta(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value, ensure_ascii=False, sort_keys=True)),
    )


def render_unit(
    path: str,
    unit_type: str,
    symbol: str | None,
    start: int,
    end: int,
    text: str,
) -> str:
    """严格复制 V2.10 Builder 的 _render_unit 模板。"""

    symbol_line = f"\n[SYMBOL] {symbol}" if symbol else ""
    return (
        f"[PATH] {path}\n[TYPE] {unit_type}\n[LINES] {start}-{end}"
        f"{symbol_line}\n[CONTENT]\n{text}"
    )


def render_metadata(record: dict[str, Any]) -> str:
    """严格复制 V2.10 Builder 的 _evidence_metadata 模板。"""

    return (
        f"[EVIDENCE_META] id={record['evidence_id']} path={record.get('path')} "
        f"type={record.get('unit_type')} symbol={record.get('symbol')} "
        f"lines={record.get('start_line')}-{record.get('end_line')}"
    )


def iter_task_rows(parquet_path: Path, batch_size: int) -> Iterable[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["supervision"],
        use_threads=True,
    ):
        yield from batch.to_pylist()


def collect_requested_evidence(
    connection: sqlite3.Connection,
    *,
    dataset_dir: Path,
    splits: Sequence[str],
    batch_size: int,
) -> dict[str, int]:
    started = time.perf_counter()
    before = int(
        connection.execute("SELECT COUNT(*) FROM requested_evidence").fetchone()[0]
    )
    task_count = 0
    state_count = 0
    action_count = 0
    insert_buffer: list[tuple[str]] = []

    def flush() -> None:
        nonlocal insert_buffer
        if not insert_buffer:
            return
        connection.executemany(
            "INSERT OR IGNORE INTO requested_evidence(evidence_id) VALUES(?)",
            insert_buffer,
        )
        insert_buffer = []

    for split in splits:
        path = dataset_dir / f"{split}_v2_10.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)

        split_tasks = 0
        for row in iter_task_rows(path, batch_size):
            split_tasks += 1
            task_count += 1
            supervision = row.get("supervision") or {}
            for state in supervision.get("policy_states") or []:
                state_count += 1
                for evidence_id in state.get("evidence_ids") or []:
                    insert_buffer.append((str(evidence_id),))
                for action in state.get("candidate_actions") or []:
                    action_count += 1
                    for evidence_id in action.get("evidence_ids") or []:
                        insert_buffer.append((str(evidence_id),))
                    for evidence_id in (
                        action.get("rendered_state_body_evidence_ids") or []
                    ):
                        insert_buffer.append((str(evidence_id),))
                if len(insert_buffer) >= 50_000:
                    flush()
                    connection.commit()

        flush()
        connection.commit()
        unique_now = int(
            connection.execute("SELECT COUNT(*) FROM requested_evidence").fetchone()[0]
        )
        print(
            f"collect: split={split}, tasks={split_tasks}, "
            f"requested_unique={unique_now:,}",
            file=sys.stderr,
            flush=True,
        )

    after = int(
        connection.execute("SELECT COUNT(*) FROM requested_evidence").fetchone()[0]
    )
    return {
        "task_count": task_count,
        "state_count": state_count,
        "action_count": action_count,
        "unique_requested_evidence": after,
        "new_requested_evidence": after - before,
        "seconds": round(time.perf_counter() - started, 3),
    }


def materialize_evidence(
    connection: sqlite3.Connection,
    *,
    corpus_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    requested = {
        str(row[0])
        for row in connection.execute(
            "SELECT evidence_id FROM requested_evidence"
        )
    }
    existing = {
        str(row[0])
        for row in connection.execute("SELECT evidence_id FROM evidence")
    }
    requested.difference_update(existing)

    if not requested:
        return {
            "requested_remaining_at_start": 0,
            "inserted": 0,
            "missing": 0,
            "seconds": 0.0,
        }

    print(
        f"materialize: need={len(requested):,} evidence records",
        file=sys.stderr,
        flush=True,
    )

    parquet = pq.ParquetFile(corpus_path)
    started = time.perf_counter()
    inserted = 0
    scanned_files = 0
    insert_buffer: list[tuple[Any, ...]] = []

    def flush() -> None:
        nonlocal insert_buffer
        if not insert_buffer:
            return
        connection.executemany(
            """
            INSERT OR REPLACE INTO evidence(
                evidence_id, file_version_id, path, unit_type, symbol,
                start_line, end_line, content, rendered_body, metadata,
                rendered_token_count
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            insert_buffer,
        )
        insert_buffer = []

    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=[
            "file_version_id",
            "path",
            "content",
            "evidence_units",
        ],
        use_threads=True,
    ):
        for file_record in batch.to_pylist():
            scanned_files += 1
            units = file_record.get("evidence_units") or []
            matches = [
                unit
                for unit in units
                if str(unit.get("evidence_id")) in requested
            ]
            if not matches:
                continue

            file_content = str(file_record.get("content") or "")
            lines = file_content.splitlines()
            path = str(file_record.get("path") or "")
            file_version_id = str(file_record.get("file_version_id") or "")

            for unit in matches:
                evidence_id = str(unit["evidence_id"])
                start_line = max(1, int(unit.get("start_line") or 1))
                end_line = max(
                    start_line,
                    int(unit.get("end_line") or start_line),
                )
                content = "\n".join(lines[start_line - 1 : end_line])
                symbol = unit.get("qualified_name") or unit.get("symbol")
                unit_type = str(unit.get("unit_type") or "code_block")

                record = {
                    "evidence_id": evidence_id,
                    "file_version_id": file_version_id,
                    "path": path,
                    "unit_type": unit_type,
                    "symbol": symbol,
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": content,
                }
                insert_buffer.append(
                    (
                        evidence_id,
                        file_version_id,
                        path,
                        unit_type,
                        None if symbol is None else str(symbol),
                        start_line,
                        end_line,
                        content,
                        render_unit(
                            path,
                            unit_type,
                            None if symbol is None else str(symbol),
                            start_line,
                            end_line,
                            content,
                        ),
                        render_metadata(record),
                        int(unit.get("rendered_token_count") or 0),
                    )
                )
                requested.remove(evidence_id)
                inserted += 1

            if len(insert_buffer) >= 10_000:
                flush()
                connection.commit()

        if scanned_files % 50_000 < batch_size:
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"materialize: files={scanned_files:,}, inserted={inserted:,}, "
                f"remaining={len(requested):,}, rate={scanned_files/elapsed:.1f} file/s",
                file=sys.stderr,
                flush=True,
            )

        if not requested:
            break

    flush()
    connection.commit()

    return {
        "requested_remaining_at_start": inserted + len(requested),
        "inserted": inserted,
        "missing": len(requested),
        "missing_sample": sorted(requested)[:20],
        "scanned_files": scanned_files,
        "seconds": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        choices=list(DEFAULT_SPLITS),
    )
    parser.add_argument("--task-batch-size", type=int, default=64)
    parser.add_argument("--corpus-batch-size", type=int, default=128)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="删除现有训练 cache 后重新构建。",
    )
    parser.add_argument(
        "--verify-manifest-hash",
        action="store_true",
        help="额外计算 manifest 中任务文件的 SHA-256；默认依赖 release audit。",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    manifest_path = dataset_dir / "manifest_v2_10.json"
    corpus_path = dataset_dir / "repository_corpus_v2_10.parquet"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not corpus_path.is_file():
        raise FileNotFoundError(corpus_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_version") != "2.10.0":
        raise ValueError(
            f"只接受 V2.10 release，实际={manifest.get('dataset_version')}"
        )
    if manifest.get("audit_status") != "passed":
        raise ValueError("manifest audit_status 不是 passed")

    if args.verify_manifest_hash:
        for split in args.splits:
            filename = f"{split}_v2_10.parquet"
            expected = str(manifest["files"][filename]["sha256"])
            actual = sha256_file(dataset_dir / filename)
            if actual != expected:
                raise ValueError(
                    f"{filename} SHA-256 不匹配：expected={expected}, actual={actual}"
                )

    connection = open_cache(args.output.resolve(), rebuild=args.rebuild)
    try:
        set_meta(connection, "dataset_name", manifest["dataset_name"])
        set_meta(connection, "dataset_version", manifest["dataset_version"])
        set_meta(connection, "manifest_sha256", sha256_file(manifest_path))
        set_meta(connection, "splits", list(args.splits))
        set_meta(
            connection,
            "retriever_version",
            (manifest.get("retrieval") or {}).get("version"),
        )
        set_meta(connection, "schema_version", manifest.get("schema_version"))
        connection.commit()

        collect_report = collect_requested_evidence(
            connection,
            dataset_dir=dataset_dir,
            splits=args.splits,
            batch_size=args.task_batch_size,
        )
        materialize_report = materialize_evidence(
            connection,
            corpus_path=corpus_path,
            batch_size=args.corpus_batch_size,
        )

        requested_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM requested_evidence"
            ).fetchone()[0]
        )
        evidence_count = int(
            connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
        )
        missing_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM requested_evidence r "
                "LEFT JOIN evidence e ON e.evidence_id=r.evidence_id "
                "WHERE e.evidence_id IS NULL"
            ).fetchone()[0]
        )

        report = {
            "dataset_version": "2.10.0",
            "cache": str(args.output.resolve()),
            "collect": collect_report,
            "materialize": materialize_report,
            "requested_evidence_count": requested_count,
            "materialized_evidence_count": evidence_count,
            "missing_evidence_count": missing_count,
        }
        set_meta(connection, "build_report", report)
        set_meta(connection, "completed", missing_count == 0)
        connection.commit()

        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        if missing_count:
            return 2

        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
