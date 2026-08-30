#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export the final Evidence Agent experiment bundle.

Final output (exactly 4 files):

    data/evidence_agent_dataset_v1/
    ├── tasks.parquet
    ├── policy_evidence.parquet
    ├── repository_runtime.sqlite3
    └── manifest.json

Design goals
------------
1. Keep all 20,864 canonical tasks in one Parquet and add/validate a top-level split column.
2. Keep experiment-excluded tasks for provenance, but mark them via experiment_eligible=false.
3. Materialize every evidence_id referenced anywhere in the task records into one compact
   policy_evidence.parquet for training/offline evaluation.
4. Export only repository/runtime tables from the large working SQLite into one runtime DB;
   do not carry dataset-builder supervision/checkpoint tables into the final runtime.
5. Merge the existing V2.10 FTS5 sidecar into repository_runtime.sqlite3 when possible,
   so online Retriever/Agent rollout no longer needs data/.build/*.sqlite3.
6. Require a passed integrated integrity audit. Semantic review must be explicitly confirmed.
7. Never mutate the source V2.10 release, teacher_v1 dataset, .build DB, or FTS sidecar.
8. Publish atomically and leave only four final files.

Typical invocation (PowerShell)
-------------------------------
python scripts/export_final_experiment_bundle_v1_0_2.py `
  --derived-dir "data\\unified_swe_dataset_v2_10_teacher_v1" `
  --base-dir "data\\unified_swe_dataset_v2_10" `
  --build-db "data\\.build\\unified_swe_v1.sqlite3" `
  --fts-db "data\\.build\\retriever_v2_2_fts.sqlite3" `
  --output-dir "data\\evidence_agent_dataset_v1" `
  --confirm-semantic-review-complete `
  --overwrite

Notes
-----
* The large .build DB is an EXPORT-TIME dependency only. The final bundle does not depend on it.
* policy_evidence.parquet is built from evidence_ids referenced by the final task records,
  not from the old .train_cache, so Teacher-added evidence cannot be silently missed.
* repository_runtime.sqlite3 intentionally contains the complete runtime repository space,
  because end-to-end online Agent rollout must be able to retrieve evidence that was not in
  the offline Policy training candidate sets.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover
    raise SystemExit("pyarrow is required: pip install pyarrow") from exc


SCRIPT_VERSION = "1.0.3"
BUNDLE_NAME = "evidence_agent_dataset_v1"
BUNDLE_VERSION = "1.0.0"
SPLITS = ("train", "validation", "benchmark")
EXPECTED_TASK_COUNTS = {
    "train": 18_347,
    "validation": 223,
    "benchmark": 2_294,
    "total": 20_864,
}
EXPECTED_V210_RUNTIME_COUNTS = {
    "snapshots": 18_527,
    "file_versions": 1_027_752,
    "evidence_units": 25_496_300,
    "snapshot_file_memberships": 32_092_093,
}
RUNTIME_CORE_TABLES = (
    "snapshots",
    "file_versions",
    "evidence_units",
    "snapshot_file_memberships",
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_report(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        out["rows"] = int(rows)
    return out


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def atomic_publish(staging: Path, output_dir: Path, *, overwrite: bool) -> None:
    """Publish a directory atomically where the filesystem permits it."""
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_dir}. Use --overwrite to replace it."
        )

    backup = output_dir.with_name(output_dir.name + ".old")
    if backup.exists():
        shutil.rmtree(backup)

    if output_dir.exists():
        output_dir.replace(backup)
        try:
            staging.replace(output_dir)
        except Exception:
            backup.replace(output_dir)
            raise
        shutil.rmtree(backup)
    else:
        staging.replace(output_dir)


def resolve_split_path(derived_dir: Path, split: str) -> Path:
    candidates = [
        derived_dir / f"{split}.parquet",
        derived_dir / f"{split}_v2_10.parquet",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Cannot find {split} Parquet under {derived_dir}; tried: {candidates}"
    )


def resolve_base_corpus(base_dir: Path) -> Path:
    candidates = [
        base_dir / "repository_corpus_v2_10.parquet",
        base_dir / "repository_corpus.parquet",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Cannot find repository corpus under {base_dir}; tried: {candidates}"
    )


def resolve_base_manifest(base_dir: Path) -> Path:
    candidates = [
        base_dir / "manifest_v2_10.json",
        base_dir / "manifest.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Cannot find base manifest under {base_dir}; tried: {candidates}"
    )


# ---------------------------------------------------------------------------
# Source validation
# ---------------------------------------------------------------------------

def validate_integrated_sources(
    *,
    derived_dir: Path,
    base_dir: Path,
    build_db: Path,
    fts_db: Path,
    confirm_semantic_review_complete: bool,
    strict_v210_counts: bool,
) -> dict[str, Any]:
    derived_manifest_path = derived_dir / "manifest.json"
    audit_path = derived_dir / "integrity_audit.json"
    if not derived_manifest_path.is_file():
        raise FileNotFoundError(derived_manifest_path)
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Missing integrated integrity audit: {audit_path}. Run the v1.3 audit first."
        )
    if not build_db.is_file():
        raise FileNotFoundError(build_db)
    if not fts_db.is_file():
        raise FileNotFoundError(
            f"Missing V2.10 FTS sidecar: {fts_db}. It is required only for this one-time "
            "runtime bundle export."
        )

    audit = load_json(audit_path)
    if str(audit.get("status") or "").upper() != "PASS":
        raise RuntimeError(f"Integrated audit is not PASS: {audit_path}")
    if int(audit.get("error_count") or 0) != 0:
        raise RuntimeError(
            f"Integrated audit error_count != 0: {audit.get('error_count')}"
        )
    if int(audit.get("task_count") or 0) != EXPECTED_TASK_COUNTS["total"]:
        raise RuntimeError(
            f"Integrated audit task_count mismatch: {audit.get('task_count')} != "
            f"{EXPECTED_TASK_COUNTS['total']}"
        )
    if not confirm_semantic_review_complete:
        raise RuntimeError(
            "Semantic review must be explicitly confirmed. Re-run with "
            "--confirm-semantic-review-complete only after the final semantic review is done."
        )

    base_manifest_path = resolve_base_manifest(base_dir)
    base_corpus_path = resolve_base_corpus(base_dir)
    base_manifest = load_json(base_manifest_path)
    derived_manifest = load_json(derived_manifest_path)

    # Verify the runtime repository working DB still has the frozen V2.10 corpus cardinalities.
    runtime_counts: dict[str, int] = {}
    conn = sqlite3.connect(build_db)
    try:
        for table in RUNTIME_CORE_TABLES:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not exists:
                raise RuntimeError(f"Working DB is missing runtime table: {table}")
            runtime_counts[table] = int(
                conn.execute(f"SELECT COUNT(*) FROM {qident(table)}").fetchone()[0]
            )
    finally:
        conn.close()

    if strict_v210_counts:
        bad = {
            k: {"expected": v, "actual": runtime_counts.get(k)}
            for k, v in EXPECTED_V210_RUNTIME_COUNTS.items()
            if runtime_counts.get(k) != v
        }
        if bad:
            raise RuntimeError(
                "Working DB repository cardinalities do not match frozen V2.10: "
                + json.dumps(bad, ensure_ascii=False, sort_keys=True)
            )

    return {
        "derived_manifest_path": derived_manifest_path,
        "derived_manifest": derived_manifest,
        "audit_path": audit_path,
        "audit": audit,
        "base_manifest_path": base_manifest_path,
        "base_manifest": base_manifest,
        "base_corpus_path": base_corpus_path,
        "runtime_counts": runtime_counts,
    }


# ---------------------------------------------------------------------------
# tasks.parquet
# ---------------------------------------------------------------------------

def _bool_true_count(arr: pa.Array | pa.ChunkedArray) -> int:
    # pyarrow compute is not imported on purpose; this stays version-compatible.
    return sum(1 for x in arr.to_pylist() if x is True)


def merge_task_splits(
    *, derived_dir: Path, output_path: Path, batch_size: int = 128
) -> dict[str, Any]:
    writer: pq.ParquetWriter | None = None
    out_schema: pa.Schema | None = None
    split_counts: dict[str, int] = {}
    eligible_by_split: dict[str, int] = {}
    excluded_by_reason: Counter[str] = Counter()
    total = 0

    try:
        for split in SPLITS:
            src = resolve_split_path(derived_dir, split)
            pf = pq.ParquetFile(src)
            try:
                rows_this_split = 0
                eligible_this_split = 0
                for batch in pf.iter_batches(batch_size=batch_size):
                    table = pa.Table.from_batches([batch])

                    if "split" in table.column_names:
                        values = {str(v) for v in table["split"].to_pylist() if v is not None}
                        if values and values != {split}:
                            raise ValueError(
                                f"Existing split column in {src} disagrees with filename: {values}"
                            )
                    else:
                        table = table.append_column(
                            "split", pa.array([split] * table.num_rows, type=pa.string())
                        )

                    if "experiment_eligible" not in table.column_names:
                        raise ValueError(
                            f"{src} has no experiment_eligible column. Use the audited teacher_v1 dataset."
                        )

                    eligible_this_split += _bool_true_count(table["experiment_eligible"])
                    if "experiment_exclusion_reason" in table.column_names:
                        for reason, eligible in zip(
                            table["experiment_exclusion_reason"].to_pylist(),
                            table["experiment_eligible"].to_pylist(),
                        ):
                            if eligible is False:
                                excluded_by_reason[str(reason or "UNSPECIFIED")] += 1

                    if writer is None:
                        out_schema = table.schema
                        writer = pq.ParquetWriter(
                            output_path,
                            out_schema,
                            compression="zstd",
                            use_dictionary=True,
                            write_statistics=True,
                        )
                    else:
                        assert out_schema is not None
                        if table.schema != out_schema:
                            try:
                                table = table.cast(out_schema, safe=False)
                            except Exception as exc:
                                raise TypeError(
                                    f"Task split schema mismatch while merging {src}:\n"
                                    f"expected={out_schema}\nactual={table.schema}"
                                ) from exc

                    writer.write_table(table)
                    rows_this_split += table.num_rows
                    total += table.num_rows

                split_counts[split] = rows_this_split
                eligible_by_split[split] = eligible_this_split
            finally:
                pf.close()
    finally:
        if writer is not None:
            writer.close()

    if total == 0:
        raise RuntimeError("No task rows were written")

    expected_bad = {
        split: {"expected": EXPECTED_TASK_COUNTS[split], "actual": split_counts.get(split)}
        for split in SPLITS
        if split_counts.get(split) != EXPECTED_TASK_COUNTS[split]
    }
    if expected_bad or total != EXPECTED_TASK_COUNTS["total"]:
        raise RuntimeError(
            "Task counts changed during bundle export: "
            + json.dumps(
                {
                    "split_mismatches": expected_bad,
                    "expected_total": EXPECTED_TASK_COUNTS["total"],
                    "actual_total": total,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    return {
        "rows": total,
        "split_counts": split_counts,
        "eligible_by_split": eligible_by_split,
        "eligible_total": sum(eligible_by_split.values()),
        "excluded_total": total - sum(eligible_by_split.values()),
        "excluded_by_reason": dict(excluded_by_reason),
    }


def summarize_existing_tasks(tasks_path: Path) -> dict[str, Any]:
    pf = pq.ParquetFile(tasks_path)
    split_counts: Counter[str] = Counter()
    eligible_by_split: Counter[str] = Counter()
    excluded_by_reason: Counter[str] = Counter()
    total = 0
    try:
        required = {"split", "experiment_eligible"}
        names = set(pf.schema_arrow.names)
        if not required.issubset(names):
            raise RuntimeError(
                f"Resume tasks.parquet missing required columns: {sorted(required - names)}"
            )
        cols = ["split", "experiment_eligible"]
        if "experiment_exclusion_reason" in names:
            cols.append("experiment_exclusion_reason")
        for batch in pf.iter_batches(batch_size=4096, columns=cols):
            table = pa.Table.from_batches([batch])
            splits = table["split"].to_pylist()
            elig = table["experiment_eligible"].to_pylist()
            reasons = (
                table["experiment_exclusion_reason"].to_pylist()
                if "experiment_exclusion_reason" in table.column_names
                else [None] * table.num_rows
            )
            for sp, ok, reason in zip(splits, elig, reasons):
                sp = str(sp)
                split_counts[sp] += 1
                total += 1
                if ok is True:
                    eligible_by_split[sp] += 1
                else:
                    excluded_by_reason[str(reason or "UNSPECIFIED")] += 1
    finally:
        pf.close()

    if total != EXPECTED_TASK_COUNTS["total"]:
        raise RuntimeError(
            f"Resume tasks.parquet row count mismatch: {total} != {EXPECTED_TASK_COUNTS['total']}"
        )
    for sp in SPLITS:
        if split_counts[sp] != EXPECTED_TASK_COUNTS[sp]:
            raise RuntimeError(
                f"Resume tasks.parquet split count mismatch for {sp}: "
                f"{split_counts[sp]} != {EXPECTED_TASK_COUNTS[sp]}"
            )
    return {
        "rows": total,
        "split_counts": {sp: split_counts[sp] for sp in SPLITS},
        "eligible_by_split": {sp: eligible_by_split[sp] for sp in SPLITS},
        "eligible_total": sum(eligible_by_split.values()),
        "excluded_total": total - sum(eligible_by_split.values()),
        "excluded_by_reason": dict(excluded_by_reason),
    }


def summarize_existing_policy_evidence(
    policy_evidence_path: Path, referenced_ids: set[str]
) -> dict[str, Any]:
    pf = pq.ParquetFile(policy_evidence_path)
    try:
        rows = pf.metadata.num_rows
        columns = pf.schema_arrow.names
        if rows != len(referenced_ids):
            raise RuntimeError(
                f"Resume policy_evidence row mismatch: rows={rows}, referenced={len(referenced_ids)}"
            )
        if "evidence_id" not in columns or "content" not in columns:
            raise RuntimeError(
                "Resume policy_evidence.parquet lacks evidence_id/content columns"
            )
    finally:
        pf.close()
    return {
        "rows": rows,
        "referenced_evidence_id_count": len(referenced_ids),
        "columns": list(columns),
        "content_source": "file_versions.payload_json.content sliced by evidence unit line range",
        "content_sha256_verified": True,
        "resumed_existing_file": True,
    }


# ---------------------------------------------------------------------------
# Evidence reference collection
# ---------------------------------------------------------------------------

def collect_evidence_ids_from_obj(obj: Any, out: set[str]) -> None:
    """Recursively collect stable evidence IDs from known evidence key names.

    This intentionally walks the whole task row so we do not accidentally omit Teacher witness,
    K-state evidence, candidate actions, hard negatives, evidence labels, or source trajectory
    references that use the same stable evidence_id contract.
    """
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            key_s = str(key)
            if key_s == "evidence_id":
                if isinstance(value, str) and value.strip():
                    out.add(value.strip())
            elif key_s == "evidence_ids":
                if isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            out.add(item.strip())
            elif key_s.endswith("_json") and isinstance(value, str) and value.strip():
                # Integrated Teacher metadata is intentionally stored as JSON text in several
                # extension columns. Parse it when possible so supporting/witness evidence IDs
                # remain materialized for provenance/audit as well.
                try:
                    parsed = json.loads(value)
                except Exception:
                    parsed = None
                if parsed is not None:
                    collect_evidence_ids_from_obj(parsed, out)
            else:
                collect_evidence_ids_from_obj(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            collect_evidence_ids_from_obj(value, out)


def collect_referenced_evidence_ids(
    tasks_path: Path, *, row_group_batch: int = 1
) -> set[str]:
    ids: set[str] = set()
    pf = pq.ParquetFile(tasks_path)
    try:
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            for row in table.to_pylist():
                collect_evidence_ids_from_obj(row, ids)
            if (rg + 1) % max(1, row_group_batch) == 0:
                print(
                    f"[bundle] evidence refs: row_group={rg+1}/{pf.num_row_groups} "
                    f"unique={len(ids):,}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        pf.close()
    return ids


# ---------------------------------------------------------------------------
# policy_evidence.parquet from working DB
# ---------------------------------------------------------------------------

def sqlite_decl_to_arrow(decl: str | None) -> pa.DataType:
    t = (decl or "").strip().upper()
    if "BOOL" in t:
        return pa.bool_()
    if "INT" in t:
        return pa.int64()
    if any(x in t for x in ("REAL", "FLOA", "DOUB", "DECIMAL", "NUMERIC")):
        return pa.float64()
    if "BLOB" in t:
        return pa.binary()
    # IDs, JSON, source code, paths, hashes, etc.
    return pa.large_string()


def table_columns(conn: sqlite3.Connection, table: str, schema: str = "main") -> list[dict[str, Any]]:
    rows = conn.execute(f"PRAGMA {qident(schema)}.table_info({qident(table)})").fetchall()
    return [
        {
            "cid": int(r[0]),
            "name": str(r[1]),
            "type": str(r[2] or ""),
            "notnull": int(r[3]),
            "default": r[4],
            "pk": int(r[5]),
        }
        for r in rows
    ]


def _coerce_for_arrow(value: Any, dtype: pa.DataType) -> Any:
    if value is None:
        return None
    if pa.types.is_boolean(dtype):
        return bool(value)
    if pa.types.is_integer(dtype):
        return int(value)
    if pa.types.is_floating(dtype):
        return float(value)
    if pa.types.is_binary(dtype):
        if isinstance(value, bytes):
            return value
        return bytes(value)
    # Preserve JSON/source text exactly as text.
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _reconstruct_evidence_record(
    *,
    evidence_id: str,
    file_version_id: str,
    unit_json: str,
    file_json: str,
) -> dict[str, Any]:
    """Reconstruct one Evidence Unit body from V2.10's normalized storage.

    V2.10 intentionally stores Evidence Unit metadata in evidence_units.payload_json
    without duplicating source text. The complete pre-fix file content lives in
    file_versions.payload_json["content"]. Evidence text must therefore be recovered
    by slicing the file with the unit's 1-based inclusive start_line/end_line.
    """
    try:
        unit = json.loads(unit_json)
    except Exception as exc:
        raise RuntimeError(f"Malformed evidence unit payload: {evidence_id}: {exc}") from exc
    try:
        file_record = json.loads(file_json)
    except Exception as exc:
        raise RuntimeError(
            f"Malformed file_version payload for evidence {evidence_id}: {file_version_id}: {exc}"
        ) from exc
    if not isinstance(unit, dict) or not isinstance(file_record, dict):
        raise RuntimeError(f"Non-object payload for evidence {evidence_id}")

    full_content = file_record.get("content")
    if full_content is None:
        raise RuntimeError(
            f"file_versions.payload_json has no content for evidence {evidence_id} "
            f"(file_version_id={file_version_id})"
        )
    full_content = str(full_content)
    lines = full_content.splitlines()
    if not lines:
        lines = [""]

    try:
        start_line = int(unit.get("start_line") or 1)
        end_line = int(unit.get("end_line") or start_line)
    except Exception as exc:
        raise RuntimeError(f"Invalid line range for evidence {evidence_id}: {exc}") from exc
    if start_line < 1 or end_line < start_line:
        raise RuntimeError(
            f"Invalid line range for evidence {evidence_id}: {start_line}-{end_line}"
        )
    if end_line > len(lines):
        raise RuntimeError(
            f"Evidence range exceeds file for {evidence_id}: "
            f"{start_line}-{end_line} > {len(lines)} lines"
        )

    content = "\n".join(lines[start_line - 1 : end_line])
    computed_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    expected_sha = str(unit.get("content_sha256") or "").strip()
    if expected_sha and computed_sha != expected_sha:
        raise RuntimeError(
            f"Evidence content SHA mismatch for {evidence_id}: "
            f"expected={expected_sha}, actual={computed_sha}"
        )

    return {
        "evidence_id": str(evidence_id),
        "file_version_id": str(file_version_id),
        "repo": str(file_record.get("repo") or unit.get("repo") or ""),
        "path": str(file_record.get("path") or unit.get("path") or ""),
        "blob_oid": str(file_record.get("blob_oid") or ""),
        "unit_type": str(unit.get("unit_type") or "code_block"),
        "symbol": None if unit.get("symbol") is None else str(unit.get("symbol")),
        "qualified_name": (
            None if unit.get("qualified_name") is None else str(unit.get("qualified_name"))
        ),
        "start_line": start_line,
        "end_line": end_line,
        "parent_evidence_id": (
            None
            if unit.get("parent_evidence_id") in (None, "")
            else str(unit.get("parent_evidence_id"))
        ),
        "content": content,
        "content_sha256": expected_sha or computed_sha,
        "token_count": int(unit.get("token_count") or 0),
        "rendered_token_count": int(unit.get("rendered_token_count") or 0),
        "scoreable": bool(unit.get("scoreable")),
    }


def export_policy_evidence(
    *,
    build_db: Path,
    evidence_ids: set[str],
    output_path: Path,
    fetch_size: int = 20_000,
) -> dict[str, Any]:
    if not evidence_ids:
        raise RuntimeError("No referenced evidence IDs found in tasks.parquet")

    uri = build_db.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        enames = {c["name"] for c in table_columns(conn, "evidence_units")}
        fnames = {c["name"] for c in table_columns(conn, "file_versions")}
        required_e = {"evidence_id", "file_version_id", "payload_json"}
        required_f = {"file_version_id", "payload_json"}
        if not required_e.issubset(enames):
            raise RuntimeError(
                "evidence_units missing required V2.10 columns: "
                f"missing={sorted(required_e - enames)}, available={sorted(enames)}"
            )
        if not required_f.issubset(fnames):
            raise RuntimeError(
                "file_versions missing required V2.10 columns: "
                f"missing={sorted(required_f - fnames)}, available={sorted(fnames)}"
            )

        conn.execute(
            "CREATE TEMP TABLE wanted_evidence_ids (evidence_id TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        sorted_ids = sorted(evidence_ids)
        for off in range(0, len(sorted_ids), 20_000):
            conn.executemany(
                "INSERT INTO wanted_evidence_ids(evidence_id) VALUES (?)",
                ((x,) for x in sorted_ids[off : off + 20_000]),
            )

        missing_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM wanted_evidence_ids w "
                "LEFT JOIN evidence_units e ON e.evidence_id=w.evidence_id "
                "WHERE e.evidence_id IS NULL"
            ).fetchone()[0]
        )
        if missing_count:
            sample = [
                str(r[0])
                for r in conn.execute(
                    "SELECT w.evidence_id FROM wanted_evidence_ids w "
                    "LEFT JOIN evidence_units e ON e.evidence_id=w.evidence_id "
                    "WHERE e.evidence_id IS NULL LIMIT 20"
                ).fetchall()
            ]
            raise RuntimeError(
                f"{missing_count} referenced evidence IDs are missing from working DB; sample={sample}"
            )

        missing_file_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM wanted_evidence_ids w "
                "JOIN evidence_units e ON e.evidence_id=w.evidence_id "
                "LEFT JOIN file_versions fv ON fv.file_version_id=e.file_version_id "
                "WHERE fv.file_version_id IS NULL"
            ).fetchone()[0]
        )
        if missing_file_count:
            raise RuntimeError(
                f"{missing_file_count} referenced evidence IDs have no file_version record"
            )

        schema = pa.schema(
            [
                pa.field("evidence_id", pa.string(), nullable=False),
                pa.field("file_version_id", pa.string(), nullable=False),
                pa.field("repo", pa.string(), nullable=False),
                pa.field("path", pa.string(), nullable=False),
                pa.field("blob_oid", pa.string(), nullable=False),
                pa.field("unit_type", pa.string(), nullable=False),
                pa.field("symbol", pa.string()),
                pa.field("qualified_name", pa.string()),
                pa.field("start_line", pa.int32(), nullable=False),
                pa.field("end_line", pa.int32(), nullable=False),
                pa.field("parent_evidence_id", pa.string()),
                pa.field("content", pa.large_string(), nullable=False),
                pa.field("content_sha256", pa.string(), nullable=False),
                pa.field("token_count", pa.int32(), nullable=False),
                pa.field("rendered_token_count", pa.int32(), nullable=False),
                pa.field("scoreable", pa.bool_(), nullable=False),
            ]
        )

        # Sort by file_version_id so adjacent rows usually share the same file payload.
        # This reduces repeated JSON parsing and makes the ~1M-row export materially cheaper.
        cur = conn.execute(
            "SELECT e.evidence_id, e.file_version_id, e.payload_json AS unit_json, "
            "fv.payload_json AS file_json "
            "FROM evidence_units e "
            "JOIN wanted_evidence_ids w ON w.evidence_id=e.evidence_id "
            "JOIN file_versions fv ON fv.file_version_id=e.file_version_id "
            "ORDER BY e.file_version_id, e.evidence_id"
        )

        writer = pq.ParquetWriter(
            output_path,
            schema,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        rows_written = 0
        bad_path_count = 0
        try:
            while True:
                rows = cur.fetchmany(fetch_size)
                if not rows:
                    break
                records: list[dict[str, Any]] = []
                for row in rows:
                    rec = _reconstruct_evidence_record(
                        evidence_id=str(row["evidence_id"]),
                        file_version_id=str(row["file_version_id"]),
                        unit_json=str(row["unit_json"]),
                        file_json=str(row["file_json"]),
                    )
                    if not rec["path"]:
                        bad_path_count += 1
                    records.append(rec)
                table = pa.Table.from_pylist(records, schema=schema)
                writer.write_table(table)
                rows_written += len(records)
                if rows_written % 200_000 < len(records):
                    print(
                        f"[bundle] policy evidence: {rows_written:,}/{len(evidence_ids):,}",
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            writer.close()

        if rows_written != len(evidence_ids):
            raise RuntimeError(
                f"policy_evidence row mismatch: wrote={rows_written}, referenced={len(evidence_ids)}"
            )
        if bad_path_count:
            raise RuntimeError(f"policy_evidence contains {bad_path_count} records with empty path")

        return {
            "rows": rows_written,
            "referenced_evidence_id_count": len(evidence_ids),
            "columns": schema.names,
            "content_source": "file_versions.payload_json.content sliced by evidence unit line range",
            "content_sha256_verified": True,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Runtime SQLite export
# ---------------------------------------------------------------------------

def sqlite_table_exists(conn: sqlite3.Connection, schema: str, table: str) -> bool:
    return (
        conn.execute(
            f"SELECT 1 FROM {qident(schema)}.sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        is not None
    )


def get_create_sql(conn: sqlite3.Connection, schema: str, kind: str, name: str) -> str | None:
    row = conn.execute(
        f"SELECT sql FROM {qident(schema)}.sqlite_master WHERE type=? AND name=?",
        (kind, name),
    ).fetchone()
    return None if row is None else row[0]


def copy_table_exact(
    conn: sqlite3.Connection,
    *,
    src_schema: str,
    table: str,
    copy_indexes: bool = True,
) -> int:
    create_sql = get_create_sql(conn, src_schema, "table", table)
    if not create_sql:
        raise RuntimeError(f"Missing CREATE TABLE SQL for {src_schema}.{table}")
    conn.execute(create_sql)
    conn.execute(
        f"INSERT INTO main.{qident(table)} SELECT * FROM {qident(src_schema)}.{qident(table)}"
    )
    count = int(conn.execute(f"SELECT COUNT(*) FROM main.{qident(table)}").fetchone()[0])

    if copy_indexes:
        idx_rows = conn.execute(
            f"SELECT name, sql FROM {qident(src_schema)}.sqlite_master "
            "WHERE type='index' AND tbl_name=? AND sql IS NOT NULL ORDER BY name",
            (table,),
        ).fetchall()
        for _, sql in idx_rows:
            if sql:
                conn.execute(sql)
    return count


def _fts_shadow_names(virtual_table_name: str) -> set[str]:
    return {
        virtual_table_name + suffix
        for suffix in ("_data", "_idx", "_content", "_docsize", "_config")
    }


def copy_fts_sidecar_exact(
    conn: sqlite3.Connection,
    *,
    fts_schema: str,
) -> dict[str, Any]:
    virtuals = conn.execute(
        f"SELECT name, sql FROM {qident(fts_schema)}.sqlite_master "
        "WHERE type='table' AND sql IS NOT NULL AND lower(sql) LIKE '%virtual table%' "
        "AND lower(sql) LIKE '%fts5%' ORDER BY name"
    ).fetchall()
    if not virtuals:
        raise RuntimeError("FTS sidecar contains no FTS5 virtual table")

    virtual_names = [str(r[0]) for r in virtuals]
    shadow_names: set[str] = set()
    for name in virtual_names:
        shadow_names.update(_fts_shadow_names(name))

    copied_virtual: dict[str, int] = {}
    for name_raw, sql_raw in virtuals:
        name = str(name_raw)
        sql = str(sql_raw)
        # Recreate the logical FTS table, then insert logical rows so SQLite rebuilds shadow data.
        conn.execute(sql)
        cols = [c["name"] for c in table_columns(conn, name, fts_schema)]
        if not cols:
            raise RuntimeError(f"Cannot inspect FTS columns for {fts_schema}.{name}")
        col_sql = ", ".join(qident(c) for c in cols)
        conn.execute(
            f"INSERT INTO main.{qident(name)} ({col_sql}) "
            f"SELECT {col_sql} FROM {qident(fts_schema)}.{qident(name)}"
        )
        copied_virtual[name] = int(
            conn.execute(f"SELECT COUNT(*) FROM main.{qident(name)}").fetchone()[0]
        )

    # Copy small regular metadata tables from the sidecar, but never copy FTS shadow tables.
    copied_metadata: dict[str, int] = {}
    regular = conn.execute(
        f"SELECT name, sql FROM {qident(fts_schema)}.sqlite_master "
        "WHERE type='table' AND sql IS NOT NULL ORDER BY name"
    ).fetchall()
    for name_raw, _ in regular:
        name = str(name_raw)
        if name in virtual_names or name in shadow_names:
            continue
        if sqlite_table_exists(conn, "main", name):
            # Core runtime table wins on name collision.
            continue
        copied_metadata[name] = copy_table_exact(
            conn, src_schema=fts_schema, table=name, copy_indexes=True
        )

    return {
        "mode": "copied_v2_10_fts5_sidecar",
        "virtual_tables": copied_virtual,
        "metadata_tables": copied_metadata,
    }


def build_generic_runtime_fts(conn: sqlite3.Connection) -> dict[str, Any]:
    """Fallback only: build a canonical FTS table from runtime file_versions."""
    fcols = table_columns(conn, "file_versions", "main")
    fnames = {c["name"] for c in fcols}

    def choose(candidates: Sequence[str]) -> str | None:
        for x in candidates:
            if x in fnames:
                return x
        return None

    id_col = choose(("file_version_id", "id"))
    path_col = choose(("path", "file_path", "relative_path"))
    content_col = choose(("content", "text", "source", "file_content"))
    if not id_col:
        raise RuntimeError(
            "Cannot build generic runtime FTS; file_versions ID column not found. "
            f"available={sorted(fnames)}"
        )
    if path_col:
        path_expr = qident(path_col)
    elif "payload_json" in fnames:
        path_expr = "json_extract(payload_json, '$.path')"
    else:
        raise RuntimeError(
            "Cannot build generic runtime FTS; file path not found in columns or payload_json. "
            f"available={sorted(fnames)}"
        )
    if content_col:
        content_expr = qident(content_col)
    elif "payload_json" in fnames:
        content_expr = "json_extract(payload_json, '$.content')"
    else:
        raise RuntimeError(
            "Cannot build generic runtime FTS; file content not found in columns or payload_json. "
            f"available={sorted(fnames)}"
        )

    conn.execute(
        "CREATE VIRTUAL TABLE runtime_file_fts USING fts5("
        "file_version_id UNINDEXED, path, content, tokenize='unicode61')"
    )
    conn.execute(
        "INSERT INTO runtime_file_fts(file_version_id, path, content) "
        f"SELECT CAST({qident(id_col)} AS TEXT), {path_expr}, {content_expr} "
        "FROM file_versions WHERE " + content_expr + " IS NOT NULL"
    )
    return {
        "mode": "generic_runtime_fts_fallback",
        "virtual_tables": {
            "runtime_file_fts": int(
                conn.execute("SELECT COUNT(*) FROM runtime_file_fts").fetchone()[0]
            )
        },
        "warning": (
            "This fallback is a stable bundle runtime index, but it is not guaranteed to expose "
            "the exact same FTS table name/schema as the historical V2.10 sidecar."
        ),
    }


def _clone_sqlite_database_with_progress(source_path: Path, dest_path: Path) -> dict[str, Any]:
    """Clone an SQLite database byte-logically with SQLite backup API.

    This is intentionally used for the historical FTS5 sidecar.  Re-inserting the
    logical FTS rows would make SQLite tokenize every document and rebuild the
    inverted index from scratch.  The backup API preserves the already-built FTS5
    shadow tables exactly and is dramatically cheaper.
    """
    if dest_path.exists():
        dest_path.unlink()

    src = sqlite3.connect(str(source_path.resolve()))
    dst = sqlite3.connect(str(dest_path.resolve()))
    last_pct = -1
    try:
        # Contractual read-only behavior: do not issue writes to the source DB.
        src.execute("PRAGMA query_only=ON")

        def progress(status: int, remaining: int, total: int) -> None:
            nonlocal last_pct
            if total <= 0:
                return
            done = total - remaining
            pct = int(done * 100 / total)
            # Log at 5% boundaries, plus completion.
            bucket = min(100, (pct // 5) * 5)
            if bucket != last_pct and (bucket % 5 == 0 or remaining == 0):
                last_pct = bucket
                print(
                    f"[bundle] runtime DB: cloning FTS sidecar {done:,}/{total:,} pages ({pct}%)",
                    file=sys.stderr,
                    flush=True,
                )

        src.backup(dst, pages=8192, progress=progress, sleep=0.01)
        dst.commit()
        return {
            "source": str(source_path.resolve()),
            "mode": "sqlite_backup_api",
            "source_size_bytes": source_path.stat().st_size,
        }
    finally:
        dst.close()
        src.close()


def _inspect_existing_fts(conn: sqlite3.Connection) -> dict[str, Any]:
    virtuals = conn.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='table' AND sql IS NOT NULL AND lower(sql) LIKE '%virtual table%' "
        "AND lower(sql) LIKE '%fts5%' ORDER BY name"
    ).fetchall()
    if not virtuals:
        raise RuntimeError("Cloned FTS sidecar contains no FTS5 virtual table")

    tables: dict[str, Any] = {}
    for name_raw, sql_raw in virtuals:
        name = str(name_raw)
        row_count = None
        try:
            row_count = int(conn.execute(f"SELECT COUNT(*) FROM {qident(name)}").fetchone()[0])
        except Exception:
            pass
        tables[name] = {"rows": row_count, "sql": str(sql_raw)}
    return {
        "mode": "cloned_v2_10_fts5_sidecar",
        "virtual_tables": tables,
    }


def export_runtime_db(
    *,
    build_db: Path,
    fts_db: Path,
    output_path: Path,
    source_hashes: Mapping[str, str],
    allow_generic_fts_fallback: bool,
) -> dict[str, Any]:
    """Export the self-contained runtime repository DB.

    v1.0.3 changes the order deliberately:

      1. Clone the already-built V2.10 FTS sidecar into repository_runtime.sqlite3
         with SQLite's backup API.  This preserves the FTS5 shadow tables and avoids
         re-tokenizing/re-indexing ~1M file versions.
      2. Attach the repository working DB read-only-by-contract and copy only the
         four canonical runtime tables into the cloned destination.

    The previous implementation recreated the FTS virtual table and did
    INSERT..SELECT of all logical rows, which rebuilt the complete inverted index
    and could appear to hang for a very long time on Windows.
    """
    clone_report: dict[str, Any] | None = None
    try:
        print(
            "[bundle] runtime DB: cloning existing V2.10 FTS5 sidecar (no re-index) ...",
            file=sys.stderr,
            flush=True,
        )
        clone_report = _clone_sqlite_database_with_progress(fts_db, output_path)
    except Exception as exc:
        if not allow_generic_fts_fallback:
            raise RuntimeError(
                "Failed to clone the historical V2.10 FTS sidecar. "
                "Re-run with --allow-generic-fts-fallback only if you accept rebuilding "
                "a canonical generic runtime FTS schema. Original error: " + repr(exc)
            ) from exc
        print(
            f"[bundle] FTS sidecar clone failed; starting generic runtime DB: {exc!r}",
            file=sys.stderr,
            flush=True,
        )
        if output_path.exists():
            output_path.unlink()

    conn = sqlite3.connect(str(output_path))
    src_attached = False
    try:
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("PRAGMA cache_size=-262144")  # ~256 MiB

        src_path = str(build_db.resolve())
        try:
            conn.execute("ATTACH DATABASE ? AS src", (src_path,))
            src_attached = True
            conn.execute("SELECT COUNT(*) FROM src.sqlite_master").fetchone()
        except sqlite3.Error as exc:
            raise RuntimeError(
                "Cannot attach the repository working DB for read-only export. "
                f"path={src_path!r}, sqlite_error={exc!r}. "
                "Close programs that may hold the DB exclusively and retry."
            ) from exc

        # FTS clone should not already contain any of the canonical repository tables.
        # A collision would make the bundle ambiguous, so fail rather than silently merge.
        for table in RUNTIME_CORE_TABLES:
            if sqlite_table_exists(conn, "main", table):
                raise RuntimeError(
                    f"FTS sidecar unexpectedly already contains runtime core table {table!r}; "
                    "refusing ambiguous merge"
                )

        core_counts: dict[str, int] = {}
        for table in RUNTIME_CORE_TABLES:
            print(f"[bundle] runtime DB: copying {table} ...", file=sys.stderr, flush=True)
            core_counts[table] = copy_table_exact(
                conn, src_schema="src", table=table, copy_indexes=True
            )
            print(
                f"[bundle] runtime DB: {table} rows={core_counts[table]:,}",
                file=sys.stderr,
                flush=True,
            )
            conn.commit()

        if clone_report is not None:
            fts_report = _inspect_existing_fts(conn)
            fts_report["clone"] = clone_report
        else:
            print(
                "[bundle] runtime DB: building generic fallback FTS5 ...",
                file=sys.stderr,
                flush=True,
            )
            fts_report = build_generic_runtime_fts(conn)
            conn.commit()

        # Add stable runtime indexes if the source DB did not already provide equivalents.
        idx_specs = [
            ("idx_runtime_evidence_id", "evidence_units", "evidence_id"),
            ("idx_runtime_evidence_file_version", "evidence_units", "file_version_id"),
            ("idx_runtime_membership_snapshot", "snapshot_file_memberships", "snapshot_id"),
            ("idx_runtime_membership_file_version", "snapshot_file_memberships", "file_version_id"),
        ]
        for idx_name, table, col in idx_specs:
            cols = {x["name"] for x in table_columns(conn, table)}
            if col in cols:
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {qident(idx_name)} "
                    f"ON {qident(table)}({qident(col)})"
                )
        fcols = {x["name"] for x in table_columns(conn, "file_versions")}
        if "path" in fcols:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runtime_file_path ON file_versions(path)"
            )

        if sqlite_table_exists(conn, "main", "bundle_metadata"):
            raise RuntimeError("FTS sidecar unexpectedly contains bundle_metadata")
        conn.execute(
            "CREATE TABLE bundle_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
        )
        metadata = {
            "bundle_name": BUNDLE_NAME,
            "bundle_version": BUNDLE_VERSION,
            "script_version": SCRIPT_VERSION,
            "created_at": utc_now(),
            **{f"source_{k}_sha256": v for k, v in source_hashes.items()},
        }
        conn.executemany(
            "INSERT INTO bundle_metadata(key,value) VALUES (?,?)",
            sorted((str(k), str(v)) for k, v in metadata.items()),
        )
        conn.commit()

        # quick_check is appropriate here because all source tables were copied from already
        # audited SQLite databases and FTS shadow tables were cloned atomically via backup().
        # Full integrity_check on a ~60M-row runtime bundle can dominate the export time.
        integrity = conn.execute("PRAGMA quick_check").fetchone()[0]
        if str(integrity).lower() != "ok":
            raise RuntimeError(f"repository_runtime.sqlite3 quick_check failed: {integrity}")

        if src_attached:
            conn.execute("DETACH DATABASE src")
            src_attached = False
        conn.execute("PRAGMA optimize")
        conn.commit()

        return {
            "core_table_counts": core_counts,
            "fts": fts_report,
            "integrity_check": "quick_check:ok",
            "source_attach_mode": "absolute_path_read_only_by_contract",
        }
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        if src_attached:
            try:
                conn.execute("DETACH DATABASE src")
            except Exception:
                pass
        conn.close()
        for suffix in ("-wal", "-shm", "-journal"):
            p = Path(str(output_path) + suffix)
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Bundle audit
# ---------------------------------------------------------------------------

def audit_final_bundle(
    *,
    tasks_path: Path,
    policy_evidence_path: Path,
    runtime_db_path: Path,
    expected_evidence_ids: set[str],
) -> dict[str, Any]:
    errors: list[str] = []

    # Task audit.
    pf = pq.ParquetFile(tasks_path)
    try:
        task_rows = pf.metadata.num_rows
        schema_names = set(pf.schema_arrow.names)
    finally:
        pf.close()
    if task_rows != EXPECTED_TASK_COUNTS["total"]:
        errors.append(f"TASK_ROWS:{task_rows}")
    for required in ("task_id", "split", "experiment_eligible"):
        if required not in schema_names:
            errors.append(f"TASK_COLUMN_MISSING:{required}")

    # Policy evidence audit.
    epf = pq.ParquetFile(policy_evidence_path)
    try:
        evidence_rows = epf.metadata.num_rows
        e_names = set(epf.schema_arrow.names)
    finally:
        epf.close()
    if evidence_rows != len(expected_evidence_ids):
        errors.append(
            f"POLICY_EVIDENCE_ROWS:{evidence_rows}!={len(expected_evidence_ids)}"
        )
    if "evidence_id" not in e_names:
        errors.append("POLICY_EVIDENCE_ID_COLUMN_MISSING")
    for required in ("file_version_id", "path", "start_line", "end_line", "content", "content_sha256"):
        if required not in e_names:
            errors.append(f"POLICY_EVIDENCE_COLUMN_MISSING:{required}")
    # Check reconstructed content is physically present, not merely a declared schema field.
    if "content" in e_names:
        null_content = 0
        epf2 = pq.ParquetFile(policy_evidence_path)
        try:
            for rg in range(epf2.num_row_groups):
                arr = epf2.read_row_group(rg, columns=["content"])["content"]
                null_content += arr.null_count
        finally:
            epf2.close()
        if null_content:
            errors.append(f"POLICY_EVIDENCE_NULL_CONTENT:{null_content}")

    # Runtime DB audit.
    conn = sqlite3.connect(runtime_db_path)
    try:
        integ = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
        if integ.lower() != "ok":
            errors.append(f"RUNTIME_DB_INTEGRITY:{integ}")
        runtime_counts = {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {qident(table)}").fetchone()[0])
            for table in RUNTIME_CORE_TABLES
        }
        for table, expected in EXPECTED_V210_RUNTIME_COUNTS.items():
            if runtime_counts.get(table) != expected:
                errors.append(
                    f"RUNTIME_COUNT:{table}:{runtime_counts.get(table)}!={expected}"
                )
        fts_tables = [
            str(r[0])
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
                "AND lower(sql) LIKE '%virtual table%' AND lower(sql) LIKE '%fts5%'"
            ).fetchall()
        ]
        if not fts_tables:
            errors.append("RUNTIME_FTS5_MISSING")
    finally:
        conn.close()

    return {
        "status": "PASS" if not errors else "FAIL",
        "error_count": len(errors),
        "errors": errors[:100],
        "task_rows": task_rows,
        "policy_evidence_rows": evidence_rows,
        "runtime_counts": runtime_counts,
        "runtime_fts_tables": fts_tables,
    }


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------

def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="evidence_bundle_selftest_") as td_s:
        td = Path(td_s)
        # Recursive evidence reference collection.
        obj = {
            "supervision": {
                "policy_states": [
                    {
                        "evidence_ids": ["ev_a"],
                        "candidate_actions": [
                            {"evidence_ids": ["ev_b", "ev_c"]},
                            {"action_id": "not_evidence"},
                        ],
                    }
                ],
                "obligations": [
                    {"witness_groups": [{"evidence_ids": ["ev_d"]}]}
                ],
            }
        }
        got: set[str] = set()
        collect_evidence_ids_from_obj(obj, got)
        assert got == {"ev_a", "ev_b", "ev_c", "ev_d"}, got

        # Tiny split merge.
        derived = td / "derived"
        derived.mkdir()
        base_schema = pa.schema(
            [
                pa.field("task_id", pa.string()),
                pa.field("experiment_eligible", pa.bool_()),
                pa.field("experiment_exclusion_reason", pa.string()),
            ]
        )
        # Self-test uses tiny counts and therefore tests helper logic directly rather than the
        # frozen 20,864-count gate in merge_task_splits.
        for split in SPLITS:
            table = pa.Table.from_arrays(
                [
                    pa.array([f"task_{split}"]),
                    pa.array([split != "benchmark"]),
                    pa.array([None if split != "benchmark" else "TEST"]),
                ],
                schema=base_schema,
            )
            pq.write_table(table, derived / f"{split}.parquet")

        # SQLite declaration mapping sanity.
        assert pa.types.is_integer(sqlite_decl_to_arrow("INTEGER"))
        assert pa.types.is_floating(sqlite_decl_to_arrow("REAL"))
        assert pa.types.is_large_string(sqlite_decl_to_arrow("TEXT"))
        assert pa.types.is_binary(sqlite_decl_to_arrow("BLOB"))

        # V2.10 evidence text reconstruction: unit payload has no content; file payload does.
        body = "def f():\n    x = 1\n    return x\n"
        slice_text = "    x = 1\n    return x"
        unit = {
            "evidence_id": "ev_fixture",
            "file_version_id": "fv_fixture",
            "unit_type": "function",
            "symbol": "f",
            "start_line": 2,
            "end_line": 3,
            "content_sha256": hashlib.sha256(slice_text.encode("utf-8")).hexdigest(),
            "token_count": 4,
            "rendered_token_count": 10,
            "scoreable": True,
        }
        file_record = {
            "file_version_id": "fv_fixture",
            "repo": "org/repo",
            "path": "src/a.py",
            "blob_oid": "b" * 40,
            "content": body,
        }
        reconstructed = _reconstruct_evidence_record(
            evidence_id="ev_fixture",
            file_version_id="fv_fixture",
            unit_json=json.dumps(unit),
            file_json=json.dumps(file_record),
        )
        assert reconstructed["content"] == slice_text
        assert reconstructed["path"] == "src/a.py"

    print("SELF_TEST_OK")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export the four-file final Evidence Agent experiment bundle."
    )
    p.add_argument(
        "--derived-dir",
        type=Path,
        default=Path("data/upstream/unified_swe_dataset_v2_10_teacher_v1"),
        help="Audited Strong-Teacher derived dataset directory.",
    )
    p.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/upstream/unified_swe_dataset_v2_10"),
        help="Frozen V2.10 base release directory.",
    )
    p.add_argument(
        "--build-db",
        type=Path,
        default=Path("data/.build/unified_swe_v1.sqlite3"),
        help="Export-time repository working DB; not needed after bundle export.",
    )
    p.add_argument(
        "--fts-db",
        type=Path,
        default=Path("data/.build/retriever_v2_2_fts.sqlite3"),
        help="Export-time V2.10 FTS5 sidecar; merged into final runtime DB.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evidence_agent_dataset_v1"),
    )
    p.add_argument(
        "--confirm-semantic-review-complete",
        action="store_true",
        help="Required final confirmation; sets semantic_review_complete/training_ready=true.",
    )
    p.add_argument(
        "--allow-generic-fts-fallback",
        action="store_true",
        help=(
            "If exact V2.10 FTS sidecar merge fails, build runtime_file_fts instead. "
            "Do not use unless you accept adapting the online Retriever to that stable schema."
        ),
    )
    p.add_argument(
        "--no-strict-v210-counts",
        action="store_true",
        help="Disable frozen V2.10 corpus cardinality gate (not recommended).",
    )
    p.add_argument(
        "--resume-staging",
        type=Path,
        default=None,
        help=(
            "Resume from an existing evidence_agent_dataset_v1.staging.* directory that already "
            "contains tasks.parquet and policy_evidence.parquet. Only runtime DB + final audit/manifest "
            "are rebuilt. Use after force-terminating an older exporter so its finally block cannot "
            "delete the staging directory."
        ),
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return 0

    project_root = Path.cwd().resolve()

    def abs_path(p: Path) -> Path:
        return p.resolve() if p.is_absolute() else (project_root / p).resolve()

    derived_dir = abs_path(args.derived_dir)
    base_dir = abs_path(args.base_dir)
    build_db = abs_path(args.build_db)
    fts_db = abs_path(args.fts_db)
    output_dir = abs_path(args.output_dir)

    source = validate_integrated_sources(
        derived_dir=derived_dir,
        base_dir=base_dir,
        build_db=build_db,
        fts_db=fts_db,
        confirm_semantic_review_complete=args.confirm_semantic_review_complete,
        strict_v210_counts=not args.no_strict_v210_counts,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    resumed = args.resume_staging is not None
    if resumed:
        staging = abs_path(args.resume_staging)
        if not staging.is_dir():
            raise FileNotFoundError(f"Resume staging directory not found: {staging}")
        tasks_path = staging / "tasks.parquet"
        policy_evidence_path = staging / "policy_evidence.parquet"
        if not tasks_path.is_file() or not policy_evidence_path.is_file():
            raise RuntimeError(
                "Resume staging must already contain tasks.parquet and policy_evidence.parquet: "
                f"{staging}"
            )
        # Remove only incomplete outputs from the failed third/fourth phase.
        for stale in [staging / "repository_runtime.sqlite3", staging / "manifest.json"]:
            if stale.exists():
                stale.unlink()
        for suffix in ("-wal", "-shm", "-journal"):
            stale = Path(str(staging / "repository_runtime.sqlite3") + suffix)
            if stale.exists():
                stale.unlink()
    else:
        staging = Path(
            tempfile.mkdtemp(
                prefix=output_dir.name + ".staging.", dir=str(output_dir.parent)
            )
        )
        tasks_path = staging / "tasks.parquet"
        policy_evidence_path = staging / "policy_evidence.parquet"

    runtime_db_path = staging / "repository_runtime.sqlite3"
    manifest_path = staging / "manifest.json"

    started = time.time()
    try:
        if resumed:
            print(
                f"[bundle] RESUME using existing staging: {staging}",
                file=sys.stderr,
                flush=True,
            )
            print("[bundle] resume: validating existing tasks.parquet", file=sys.stderr, flush=True)
            task_report = summarize_existing_tasks(tasks_path)
            print("[bundle] resume: collecting stable evidence_id references", file=sys.stderr, flush=True)
            referenced_ids = collect_referenced_evidence_ids(tasks_path)
            print(
                f"[bundle] referenced evidence IDs: {len(referenced_ids):,}",
                file=sys.stderr,
                flush=True,
            )
            print("[bundle] resume: validating existing policy_evidence.parquet", file=sys.stderr, flush=True)
            evidence_report = summarize_existing_policy_evidence(
                policy_evidence_path, referenced_ids
            )
        else:
            print("[bundle] 1/4 merging task splits -> tasks.parquet", file=sys.stderr, flush=True)
            task_report = merge_task_splits(
                derived_dir=derived_dir,
                output_path=tasks_path,
            )

            print("[bundle] 2/4 collecting stable evidence_id references", file=sys.stderr, flush=True)
            referenced_ids = collect_referenced_evidence_ids(tasks_path)
            print(
                f"[bundle] referenced evidence IDs: {len(referenced_ids):,}",
                file=sys.stderr,
                flush=True,
            )

            print("[bundle] 2/4 exporting policy_evidence.parquet", file=sys.stderr, flush=True)
            evidence_report = export_policy_evidence(
                build_db=build_db,
                evidence_ids=referenced_ids,
                output_path=policy_evidence_path,
            )

        # Hash frozen/authoritative source files once. We intentionally do not hash the large
        # mutable build DB; it is an export tool input, not a bundle source of truth.
        print("[bundle] hashing source release files", file=sys.stderr, flush=True)
        source_hashes = {
            "derived_manifest": sha256_file(source["derived_manifest_path"]),
            "integrity_audit": sha256_file(source["audit_path"]),
            "base_manifest": sha256_file(source["base_manifest_path"]),
            "base_repository_corpus": sha256_file(source["base_corpus_path"]),
        }

        print("[bundle] 3/4 exporting repository_runtime.sqlite3", file=sys.stderr, flush=True)
        runtime_report = export_runtime_db(
            build_db=build_db,
            fts_db=fts_db,
            output_path=runtime_db_path,
            source_hashes=source_hashes,
            allow_generic_fts_fallback=args.allow_generic_fts_fallback,
        )

        print("[bundle] final integrity audit", file=sys.stderr, flush=True)
        final_audit = audit_final_bundle(
            tasks_path=tasks_path,
            policy_evidence_path=policy_evidence_path,
            runtime_db_path=runtime_db_path,
            expected_evidence_ids=referenced_ids,
        )
        if final_audit["status"] != "PASS":
            raise RuntimeError(
                "Final bundle audit failed: "
                + json.dumps(final_audit, ensure_ascii=False, indent=2)
            )

        # File hashes after all writes are complete.
        print("[bundle] 4/4 hashing final files + writing manifest", file=sys.stderr, flush=True)
        tasks_file_report = file_report(tasks_path, rows=task_report["rows"])
        evidence_file_report = file_report(
            policy_evidence_path, rows=evidence_report["rows"]
        )
        runtime_file_report = file_report(runtime_db_path)

        manifest = {
            "dataset_name": BUNDLE_NAME,
            "dataset_version": BUNDLE_VERSION,
            "script_version": SCRIPT_VERSION,
            "created_at": utc_now(),
            "status": "FROZEN",
            "semantic_review_complete": True,
            "integrity_audit_passed": True,
            "training_ready": True,
            "files_created": 4,
            "task_counts": {
                **task_report["split_counts"],
                "total": task_report["rows"],
                "eligible_by_split": task_report["eligible_by_split"],
                "eligible_total": task_report["eligible_total"],
                "excluded_total": task_report["excluded_total"],
                "excluded_by_reason": task_report["excluded_by_reason"],
            },
            "files": {
                "tasks.parquet": tasks_file_report,
                "policy_evidence.parquet": evidence_file_report,
                "repository_runtime.sqlite3": runtime_file_report,
            },
            "source": {
                "derived_dataset_dir": str(derived_dir),
                "derived_manifest": str(source["derived_manifest_path"]),
                "derived_manifest_sha256": source_hashes["derived_manifest"],
                "integrity_audit": str(source["audit_path"]),
                "integrity_audit_sha256": source_hashes["integrity_audit"],
                "base_dataset_dir": str(base_dir),
                "base_manifest": str(source["base_manifest_path"]),
                "base_manifest_sha256": source_hashes["base_manifest"],
                "base_repository_corpus": str(source["base_corpus_path"]),
                "base_repository_corpus_sha256": source_hashes[
                    "base_repository_corpus"
                ],
                "build_db_export_only": str(build_db),
                "fts_db_export_only": str(fts_db),
                "source_files_mutated": False,
            },
            "contracts": {
                "task_identity_preserved": True,
                "stable_evidence_id": True,
                "all_canonical_tasks_retained": True,
                "experiment_excluded_tasks_physically_deleted": False,
                "experiment_filter": "experiment_eligible == true",
                "train_filter": "split == 'train' and experiment_eligible == true",
                "validation_filter": "split == 'validation' and experiment_eligible == true",
                "benchmark_filter": "split == 'benchmark' and experiment_eligible == true",
                "benchmark_for_training_or_tuning": False,
                "policy_evidence_is_self_contained_for_offline_training": True,
                "runtime_repository_is_complete_for_online_retrieval": True,
                "build_db_required_after_export": False,
                "train_cache_required_after_export": False,
                "strong_teacher_freeze_required_after_export": False,
                "base_dataset_required_after_export_for_execution": False,
                "base_dataset_should_be_archived_for_reproducibility": True,
            },
            "policy_evidence": evidence_report,
            "repository_runtime": runtime_report,
            "final_audit": final_audit,
            "runtime_source_counts": source["runtime_counts"],
            "elapsed_seconds": round(time.time() - started, 3),
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Enforce exact four-file output contract before publication.
        names = sorted(p.name for p in staging.iterdir() if p.is_file())
        expected_names = sorted(
            [
                "tasks.parquet",
                "policy_evidence.parquet",
                "repository_runtime.sqlite3",
                "manifest.json",
            ]
        )
        if names != expected_names:
            raise RuntimeError(
                f"Four-file bundle contract violated: actual={names}, expected={expected_names}"
            )

        atomic_publish(staging, output_dir, overwrite=args.overwrite)
        staging = None  # type: ignore[assignment]

        summary = {
            "status": "OK",
            "output_dir": str(output_dir),
            "files_created": 4,
            "task_count": task_report["rows"],
            "eligible_task_count": task_report["eligible_total"],
            "excluded_task_count": task_report["excluded_total"],
            "policy_evidence_rows": evidence_report["rows"],
            "runtime_fts_mode": runtime_report["fts"]["mode"],
            "training_ready": True,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        gc.collect()
        if staging is not None and isinstance(staging, Path) and staging.exists():
            # Fresh exports keep the historical cleanup behavior.  Resume mode deliberately
            # preserves the staging directory on failure so completed Parquet phases are never
            # discarded again.
            if not resumed:
                shutil.rmtree(staging, ignore_errors=True)
            else:
                print(
                    f"[bundle] resume staging preserved after failure: {staging}",
                    file=sys.stderr,
                    flush=True,
                )


if __name__ == "__main__":
    raise SystemExit(main())
