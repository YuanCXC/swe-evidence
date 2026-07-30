#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""审计 data/raw 下已下载的数据集；不联网、不修改原始数据。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_VERSION = "1.0.0"

KNOWN_SOURCES: dict[str, dict[str, Any]] = {
    "swebench": {
        "origin": "princeton-nlp/SWE-bench",
        "required": True,
        "expected_any_fields": [
            ["instance_id", "id"],
            ["repo", "repository"],
            ["base_commit"],
            ["problem_statement", "issue"],
        ],
    },
    "contextbench": {
        "origin": "EuniAI/ContextBench",
        "required": True,
        "expected_any_fields": [["instance_id", "id"]],
    },
    "swe_explore": {
        "origin": "Qiushao-E/SWE-Explore-Bench",
        "required": True,
        "expected_any_fields": [["instance_id", "id"]],
    },
}

SUPPORTED_SUFFIXES = {".parquet", ".jsonl", ".json", ".csv"}


@dataclass(frozen=True)
class FileManifestRecord:
    source_name: str
    relative_path: str
    suffix: str
    size_bytes: int
    modified_time_utc: str
    sha256: str


@dataclass
class FileSchemaRecord:
    source_name: str
    relative_path: str
    format: str
    parse_status: str
    row_count: int | None
    fields: list[str]
    sampled_records: int
    invalid_records: int
    error: str | None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def inspect_parquet(path: Path) -> tuple[int, list[str], int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "读取 Parquet 需要 pyarrow，请执行：python -m pip install pyarrow"
        ) from exc

    parquet_file = pq.ParquetFile(path)
    row_count = parquet_file.metadata.num_rows
    fields = parquet_file.schema_arrow.names
    sampled_records = 0
    if row_count > 0 and parquet_file.num_row_groups > 0:
        table = parquet_file.read_row_group(
            0,
            columns=fields[: min(5, len(fields))],
        )
        sampled_records = min(table.num_rows, 100)
    return row_count, sorted(fields), sampled_records


def inspect_jsonl(
    path: Path,
    sample_limit: int,
) -> tuple[int, list[str], int, int]:
    row_count = 0
    sampled_records = 0
    invalid_records = 0
    fields: set[str] = set()

    with path.open("r", encoding="utf-8-sig") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_records += 1
                if invalid_records <= 5:
                    print(
                        f"[警告] {path} 第 {line_number} 行 JSON 无效：{exc}",
                        file=sys.stderr,
                    )
                continue

            if sampled_records < sample_limit:
                if isinstance(record, dict):
                    fields.update(str(key) for key in record.keys())
                sampled_records += 1

    return row_count, sorted(fields), sampled_records, invalid_records


def inspect_json(
    path: Path,
    sample_limit: int,
) -> tuple[int | None, list[str], int]:
    with path.open("r", encoding="utf-8-sig") as file_obj:
        payload = json.load(file_obj)

    fields: set[str] = set()
    if isinstance(payload, list):
        row_count: int | None = len(payload)
        sampled_records = min(len(payload), sample_limit)
        for record in payload[:sample_limit]:
            if isinstance(record, dict):
                fields.update(str(key) for key in record.keys())
    elif isinstance(payload, dict):
        row_count = 1
        sampled_records = 1
        fields.update(str(key) for key in payload.keys())
    else:
        row_count = 1
        sampled_records = 1

    return row_count, sorted(fields), sampled_records


def inspect_csv(
    path: Path,
    sample_limit: int,
) -> tuple[int, list[str], int]:
    row_count = 0
    sampled_records = 0
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        fields = sorted(reader.fieldnames or [])
        for _ in reader:
            row_count += 1
            if sampled_records < sample_limit:
                sampled_records += 1
    if not fields:
        raise ValueError("CSV 没有可识别的表头")
    return row_count, fields, sampled_records


def inspect_data_file(
    source_name: str,
    path: Path,
    project_root: Path,
    sample_limit: int,
) -> FileSchemaRecord:
    relative_path = path.relative_to(project_root).as_posix()
    suffix = path.suffix.lower()

    try:
        if suffix == ".parquet":
            row_count, fields, sampled_records = inspect_parquet(path)
            invalid_records = 0
            format_name = "parquet"
        elif suffix == ".jsonl":
            row_count, fields, sampled_records, invalid_records = inspect_jsonl(
                path, sample_limit
            )
            format_name = "jsonl"
        elif suffix == ".json":
            row_count, fields, sampled_records = inspect_json(path, sample_limit)
            invalid_records = 0
            format_name = "json"
        elif suffix == ".csv":
            row_count, fields, sampled_records = inspect_csv(path, sample_limit)
            invalid_records = 0
            format_name = "csv"
        else:
            raise ValueError(f"不支持的格式：{suffix}")

        parse_status = "passed" if invalid_records == 0 else "failed"
        error = None if invalid_records == 0 else f"发现 {invalid_records} 条无效记录"
        return FileSchemaRecord(
            source_name=source_name,
            relative_path=relative_path,
            format=format_name,
            parse_status=parse_status,
            row_count=row_count,
            fields=fields,
            sampled_records=sampled_records,
            invalid_records=invalid_records,
            error=error,
        )
    except Exception as exc:  # noqa: BLE001
        return FileSchemaRecord(
            source_name=source_name,
            relative_path=relative_path,
            format=suffix.lstrip("."),
            parse_status="failed",
            row_count=None,
            fields=[],
            sampled_records=0,
            invalid_records=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def find_supported_files(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def check_expected_fields(
    source_name: str,
    observed_fields: set[str],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for alternatives in KNOWN_SOURCES.get(source_name, {}).get(
        "expected_any_fields", []
    ):
        matched = sorted(set(alternatives) & observed_fields)
        checks.append(
            {
                "accepted_alternatives": alternatives,
                "matched_fields": matched,
                "status": "passed" if matched else "warning",
            }
        )
    return checks


def build_source_fingerprint(
    manifest_records: list[FileManifestRecord],
) -> str:
    digest = hashlib.sha256()
    for record in sorted(manifest_records, key=lambda item: item.relative_path):
        digest.update(record.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def audit_source(
    source_name: str,
    source_dir: Path,
    project_root: Path,
    sample_limit: int,
) -> tuple[dict[str, Any], list[FileManifestRecord], list[FileSchemaRecord]]:
    manifest_records: list[FileManifestRecord] = []
    schema_records: list[FileSchemaRecord] = []
    source_spec = KNOWN_SOURCES.get(source_name, {})

    if not source_dir.exists():
        return (
            {
                "source_name": source_name,
                "status": "failed",
                "required": source_spec.get("required", False),
                "local_path": source_dir.relative_to(project_root).as_posix(),
                "error": "数据源目录不存在",
            },
            manifest_records,
            schema_records,
        )

    data_files = find_supported_files(source_dir)
    if not data_files:
        return (
            {
                "source_name": source_name,
                "status": "failed",
                "required": source_spec.get("required", False),
                "local_path": source_dir.relative_to(project_root).as_posix(),
                "error": "目录中没有支持的数据文件",
            },
            manifest_records,
            schema_records,
        )

    for path in data_files:
        manifest_records.append(
            FileManifestRecord(
                source_name=source_name,
                relative_path=path.relative_to(project_root).as_posix(),
                suffix=path.suffix.lower(),
                size_bytes=path.stat().st_size,
                modified_time_utc=file_mtime_iso(path),
                sha256=sha256_file(path),
            )
        )
        schema_records.append(
            inspect_data_file(
                source_name=source_name,
                path=path,
                project_root=project_root,
                sample_limit=sample_limit,
            )
        )

    observed_fields = {field for record in schema_records for field in record.fields}
    failed_files = [
        record.relative_path
        for record in schema_records
        if record.parse_status != "passed"
    ]
    fingerprint = build_source_fingerprint(manifest_records)

    audit = {
        "source_name": source_name,
        "origin": source_spec.get("origin"),
        "required": source_spec.get("required", False),
        "local_path": source_dir.relative_to(project_root).as_posix(),
        "status": "passed" if not failed_files else "failed",
        "file_count": len(manifest_records),
        "total_size_bytes": sum(record.size_bytes for record in manifest_records),
        "dataset_fingerprint": fingerprint,
        "observed_fields": sorted(observed_fields),
        "expected_field_checks": check_expected_fields(source_name, observed_fields),
        "failed_files": failed_files,
        "error": None if not failed_files else "部分数据文件无法解析",
    }
    return audit, manifest_records, schema_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="审计 data/raw 下已下载的数据集，不访问网络。"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="项目根目录，默认当前目录。",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="只审计指定来源，可重复使用。",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=100,
        help="每个文件用于字段探测的最大记录数，默认 100。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    raw_root = project_root / "data" / "raw"
    registry_root = project_root / "data" / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        print(f"[错误] 原始数据目录不存在：{raw_root}", file=sys.stderr)
        return 2

    selected_sources = args.sources or [
        path.name for path in sorted(raw_root.iterdir()) if path.is_dir()
    ]

    all_audits: list[dict[str, Any]] = []
    all_manifest_records: list[FileManifestRecord] = []
    all_schema_records: list[FileSchemaRecord] = []

    for source_name in selected_sources:
        source_dir = raw_root / source_name
        print(f"[审计] {source_name}: {source_dir}")
        try:
            source_audit, manifest_records, schema_records = audit_source(
                source_name=source_name,
                source_dir=source_dir,
                project_root=project_root,
                sample_limit=args.sample_limit,
            )
        except Exception as exc:  # noqa: BLE001
            source_audit = {
                "source_name": source_name,
                "status": "failed",
                "required": KNOWN_SOURCES.get(source_name, {}).get("required", False),
                "local_path": source_dir.relative_to(project_root).as_posix(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            manifest_records = []
            schema_records = []

        all_audits.append(source_audit)
        all_manifest_records.extend(manifest_records)
        all_schema_records.extend(schema_records)

    write_jsonl(
        registry_root / "source_manifest.jsonl",
        [
            {**asdict(record), "script_version": SCRIPT_VERSION}
            for record in all_manifest_records
        ],
    )
    write_json(
        registry_root / "source_schema_report.json",
        {
            "schema_version": "1.0",
            "script_version": SCRIPT_VERSION,
            "generated_at_utc": utc_now_iso(),
            "files": [asdict(record) for record in all_schema_records],
        },
    )

    required_failures = [
        audit["source_name"]
        for audit in all_audits
        if audit.get("required") and audit.get("status") != "passed"
    ]
    overall_status = "passed" if not required_failures else "failed"

    write_json(
        registry_root / "source_audit_report.json",
        {
            "schema_version": "1.0",
            "script_version": SCRIPT_VERSION,
            "generated_at_utc": utc_now_iso(),
            "project_root": str(project_root),
            "overall_status": overall_status,
            "required_failures": required_failures,
            "sources": all_audits,
        },
    )
    write_json(
        registry_root / "local_source_lock.json",
        {
            "schema_version": "1.0",
            "script_version": SCRIPT_VERSION,
            "generated_at_utc": utc_now_iso(),
            "revision_status": "unknown_existing_download",
            "note": (
                "无法仅从本地数据文件可靠恢复远程 revision；"
                "dataset_fingerprint 冻结的是本次实际使用的本地文件字节。"
            ),
            "sources": [
                {
                    "source_name": audit["source_name"],
                    "origin": audit.get("origin"),
                    "local_path": audit.get("local_path"),
                    "dataset_fingerprint": audit.get("dataset_fingerprint"),
                    "file_count": audit.get("file_count", 0),
                    "status": audit.get("status"),
                }
                for audit in all_audits
            ],
        },
    )

    print(
        json.dumps(
            {
                "status": overall_status,
                "audited_sources": selected_sources,
                "manifest_files": len(all_manifest_records),
                "required_failures": required_failures,
                "output_directory": str(registry_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if overall_status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
