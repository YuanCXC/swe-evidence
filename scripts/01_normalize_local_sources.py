#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 data/raw 下的 SWE-bench、ContextBench、SWE-Explore 数据统一为同一实例格式。

特点：
- 不读取任何配置文件；
- 不联网；
- 自动扫描 data/raw/{swebench,contextbench,swe_explore}；
- 自动识别 Parquet、CSV、JSONL、JSON；
- 使用字段别名和内容规则映射公共字段；
- 无法可靠映射的记录不会静默丢弃，而是写入 rejected_records.jsonl；
- 不生成补丁，不执行测试，不判断真实修复成功。

输出：
data/processed/
├── normalized_instances.jsonl
├── rejected_records.jsonl
├── normalization_report.json
└── observed_field_catalog.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

SCRIPT_VERSION = "1.1.0"

SOURCE_DIRS = ("swebench", "contextbench", "swe_explore")
SUPPORTED_SUFFIXES = {".parquet", ".csv", ".jsonl", ".json"}

# 公共字段别名。脚本按顺序寻找第一个非空值。
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "source_instance_id": (
        "instance_id",
        "id",
        "task_id",
        "problem_id",
        "sample_id",
        "qid",
        "bug_id",
    ),
    "repo": (
        "repo",
        "repository",
        "repo_name",
        "repository_name",
        "project",
        "project_name",
    ),
    "base_commit": (
        "base_commit",
        "commit",
        "commit_id",
        "commit_sha",
        "base_sha",
        "sha",
    ),
    "problem_statement": (
        "problem_statement",
        "issue",
        "issue_text",
        "description",
        "problem",
        "query",
        "prompt",
        "task",
    ),
    "patch": (
        "patch",
        "gold_patch",
        "solution_patch",
        "diff",
        "fix_patch",
    ),
    "test_patch": (
        "test_patch",
        "tests_patch",
        "gold_test_patch",
    ),
    "issue_url": (
        "issue_url",
        "github_issue_url",
        "problem_url",
    ),
    "pr_url": (
        "pr_url",
        "pull_request_url",
        "github_pr_url",
    ),
    "version": (
        "version",
        "dataset_version",
        "release",
    ),
}

# 这些名字通常代表上下文、span、定位结果或证据标注。
CONTEXT_KEYWORDS = (
    "context",
    "gold",
    "span",
    "region",
    "location",
    "localization",
    "relevant",
    "evidence",
    "core",
    "optional",
    "file",
    "line",
    "symbol",
)

# 这些名字通常代表代理探索轨迹或读取过程。
TRAJECTORY_KEYWORDS = (
    "trajectory",
    "trace",
    "steps",
    "actions",
    "history",
    "exploration",
    "tool_calls",
    "messages",
    "rollout",
)


@dataclass
class NormalizedRecord:
    canonical_record_id: str
    source_name: str
    source_split: str | None
    source_file: str
    source_row: int
    source_instance_id: str
    repo: str | None
    base_commit: str | None
    problem_statement: str | None
    patch: str | None
    test_patch: str | None
    issue_url: str | None
    pr_url: str | None
    external_context: Any
    trajectory: Any
    source_record_sha256: str
    mapped_fields: dict[str, str]
    source_metadata: dict[str, Any]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    """用稳定 JSON 表示记录，便于生成可复现哈希。"""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_key(key: str) -> str:
    """统一字段名格式，便于兼容大小写、空格和短横线。"""
    key = str(key).strip()
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    key = key.replace("-", "_").replace(" ", "_").replace(".", "_")
    key = re.sub(r"_+", "_", key)
    return key.lower().strip("_")


def normalize_record_keys(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """
    返回：
    1. 规范化字段字典；
    2. 规范化字段到原字段名的映射。
    """
    normalized: dict[str, Any] = {}
    original_names: dict[str, str] = {}

    for original_key, value in record.items():
        normalized_key = normalize_key(original_key)
        # 避免两个不同原字段规范化为同名后覆盖：
        # 优先保留第一个非空值。
        if normalized_key not in normalized or is_empty(normalized[normalized_key]):
            normalized[normalized_key] = value
            original_names[normalized_key] = str(original_key)

    return normalized, original_names


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def clean_text(value: Any) -> str | None:
    """将标量文本统一成字符串；复杂对象不强行转成正文。"""
    if is_empty(value):
        return None
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None


def clean_repo(value: Any) -> str | None:
    """
    尽量把仓库字段统一为 owner/repo。
    对无法判断的值只做轻量清洗，不猜测仓库名。
    """
    text = clean_text(value)
    if not text:
        return None

    text = text.strip().rstrip("/")
    prefixes = (
        "https://github.com/",
        "http://github.com/",
        "git@github.com:",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break

    if text.endswith(".git"):
        text = text[:-4]

    # 去掉 GitHub URL 后可能附带的 issue、pull、commit 路径。
    parts = [part for part in text.split("/") if part]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return text


def find_value(
    normalized: dict[str, Any],
    original_names: dict[str, str],
    canonical_name: str,
) -> tuple[Any, str | None]:
    for alias in FIELD_ALIASES[canonical_name]:
        normalized_alias = normalize_key(alias)
        if normalized_alias in normalized and not is_empty(normalized[normalized_alias]):
            return normalized[normalized_alias], original_names.get(normalized_alias)
    return None, None


def infer_split(path: Path) -> str | None:
    """
    从文件名推断 split。
    不根据目录外部配置推断，避免人为指定错误。
    """
    name = path.stem.lower()
    for split in ("train", "test", "dev", "validation", "valid"):
        if re.search(rf"(^|[_\-.]){split}([_\-.]|$)", name):
            return "validation" if split in {"valid", "validation"} else split
    if "verified" in name:
        return "verified"
    if "full" in name:
        return "full"
    if "public" in name:
        return "public"
    return None


def select_keyword_payload(
    normalized: dict[str, Any],
    excluded_keys: set[str],
    keywords: tuple[str, ...],
) -> dict[str, Any]:
    """
    收集字段名中包含指定关键词的内容。
    这里只保留原值，不进行语义猜测。
    """
    selected: dict[str, Any] = {}
    for key, value in normalized.items():
        if key in excluded_keys or is_empty(value):
            continue
        if any(keyword in key for keyword in keywords):
            selected[key] = value
    return selected


def compact_metadata(
    normalized: dict[str, Any],
    excluded_keys: set[str],
    max_text_length: int = 1000,
) -> dict[str, Any]:
    """
    保留未映射的轻量来源元数据。
    大段文本、巨大列表或复杂轨迹不重复写入，避免输出膨胀。
    """
    metadata: dict[str, Any] = {}

    for key, value in normalized.items():
        if key in excluded_keys or is_empty(value):
            continue

        if isinstance(value, (bool, int, float)):
            metadata[key] = value
        elif isinstance(value, str) and len(value) <= max_text_length:
            metadata[key] = value
        elif isinstance(value, list) and len(value) <= 20:
            # 仅保留较小且内容简单的列表。
            if all(
                item is None or isinstance(item, (bool, int, float, str))
                for item in value
            ):
                metadata[key] = value
        elif isinstance(value, dict) and len(value) <= 20:
            # 对小字典保留其值；后续可以据此补充数据源专用映射。
            metadata[key] = value

    return metadata


def derive_instance_id(
    source_name: str,
    source_file: str,
    source_row: int,
    explicit_id: Any,
    repo: str | None,
    base_commit: str | None,
    problem_statement: str | None,
) -> str:
    explicit_text = clean_text(explicit_id)
    if explicit_text:
        return explicit_text

    # 没有显式 ID 时，使用稳定组合生成来源内 ID。
    fingerprint_payload = {
        "source_name": source_name,
        "source_file": source_file,
        "source_row": source_row,
        "repo": repo,
        "base_commit": base_commit,
        "problem_statement": problem_statement,
    }
    return f"generated-{sha256_text(stable_json(fingerprint_payload))[:20]}"


def build_canonical_record_id(
    source_name: str,
    source_instance_id: str,
    source_file: str,
    source_row: int,
    source_record_sha256: str,
) -> str:
    """
    生成来源记录级唯一 ID。

    source_instance_id 只标识“任务”，不能保证同一文件内每一行唯一。
    因此必须同时加入 source_row 和原始记录哈希，避免重复实例 ID
    导致不同来源记录得到相同 canonical_record_id。
    """
    payload = {
        "source_name": source_name,
        "source_instance_id": source_instance_id,
        "source_file": source_file,
        "source_row": source_row,
        "source_record_sha256": source_record_sha256,
    }
    return sha256_text(stable_json(payload))


def read_parquet(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "读取 Parquet 需要 pyarrow，请执行：python -m pip install pyarrow"
        ) from exc

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=1024):
        table = batch.to_pydict()
        if not table:
            continue
        columns = list(table.keys())
        row_count = len(table[columns[0]])
        for row_index in range(row_count):
            yield {column: table[column][row_index] for column in columns}


def read_csv(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            yield dict(row)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{path} 第 {line_number} 行不是 JSON object"
                )
            yield payload


def read_json(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file_obj:
        payload = json.load(file_obj)

    if isinstance(payload, list):
        for index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"{path} 第 {index} 项不是 JSON object")
            yield item
    elif isinstance(payload, dict):
        # 常见格式：{"data": [...]}、{"instances": [...]}、{"records": [...]}
        for container_key in ("data", "instances", "records", "items"):
            items = payload.get(container_key)
            if isinstance(items, list):
                for index, item in enumerate(items, start=1):
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"{path} 的 {container_key}[{index}] 不是 JSON object"
                        )
                    yield item
                return
        yield payload
    else:
        raise ValueError(f"{path} 顶层不是 object 或 array")


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        yield from read_parquet(path)
    elif suffix == ".csv":
        yield from read_csv(path)
    elif suffix == ".jsonl":
        yield from read_jsonl(path)
    elif suffix == ".json":
        yield from read_json(path)
    else:
        raise ValueError(f"不支持的输入格式：{suffix}")


def find_input_files(raw_root: Path) -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []
    for source_name in SOURCE_DIRS:
        source_dir = raw_root / source_name
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append((source_name, path))
    return files


def normalize_one_record(
    source_name: str,
    source_file: str,
    source_split: str | None,
    source_row: int,
    raw_record: dict[str, Any],
) -> NormalizedRecord:
    normalized, original_names = normalize_record_keys(raw_record)

    mapped_values: dict[str, Any] = {}
    mapped_fields: dict[str, str] = {}
    consumed_keys: set[str] = set()

    for canonical_name in FIELD_ALIASES:
        value, original_field = find_value(
            normalized,
            original_names,
            canonical_name,
        )
        mapped_values[canonical_name] = value
        if original_field is not None:
            mapped_fields[canonical_name] = original_field
            consumed_keys.add(normalize_key(original_field))

    repo = clean_repo(mapped_values["repo"])
    base_commit = clean_text(mapped_values["base_commit"])
    problem_statement = clean_text(mapped_values["problem_statement"])
    patch = clean_text(mapped_values["patch"])
    test_patch = clean_text(mapped_values["test_patch"])
    issue_url = clean_text(mapped_values["issue_url"])
    pr_url = clean_text(mapped_values["pr_url"])

    source_instance_id = derive_instance_id(
        source_name=source_name,
        source_file=source_file,
        source_row=source_row,
        explicit_id=mapped_values["source_instance_id"],
        repo=repo,
        base_commit=base_commit,
        problem_statement=problem_statement,
    )

    context_payload = select_keyword_payload(
        normalized=normalized,
        excluded_keys=consumed_keys,
        keywords=CONTEXT_KEYWORDS,
    )
    trajectory_payload = select_keyword_payload(
        normalized=normalized,
        excluded_keys=consumed_keys | set(context_payload.keys()),
        keywords=TRAJECTORY_KEYWORDS,
    )

    metadata = compact_metadata(
        normalized=normalized,
        excluded_keys=(
            consumed_keys
            | set(context_payload.keys())
            | set(trajectory_payload.keys())
        ),
    )

    raw_hash = sha256_text(stable_json(raw_record))
    canonical_record_id = build_canonical_record_id(
        source_name=source_name,
        source_instance_id=source_instance_id,
        source_file=source_file,
        source_row=source_row,
        source_record_sha256=raw_hash,
    )

    return NormalizedRecord(
        canonical_record_id=canonical_record_id,
        source_name=source_name,
        source_split=source_split,
        source_file=source_file,
        source_row=source_row,
        source_instance_id=source_instance_id,
        repo=repo,
        base_commit=base_commit,
        problem_statement=problem_statement,
        patch=patch,
        test_patch=test_patch,
        issue_url=issue_url,
        pr_url=pr_url,
        external_context=context_payload,
        trajectory=trajectory_payload,
        source_record_sha256=raw_hash,
        mapped_fields=mapped_fields,
        source_metadata=metadata,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl_record(file_obj: Any, record: dict[str, Any]) -> None:
    file_obj.write(
        json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="自动扫描并统一本地 SWE-bench、ContextBench、SWE-Explore 数据。",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="项目根目录，默认当前目录。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的规范化输出。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    raw_root = project_root / "data" / "raw"
    output_root = project_root / "data" / "processed"

    normalized_path = output_root / "normalized_instances.jsonl"
    rejected_path = output_root / "rejected_records.jsonl"
    report_path = output_root / "normalization_report.json"
    field_catalog_path = output_root / "observed_field_catalog.json"

    if not raw_root.exists():
        print(f"[错误] 原始数据目录不存在：{raw_root}", file=sys.stderr)
        return 2

    existing_outputs = [
        path for path in (normalized_path, rejected_path, report_path, field_catalog_path)
        if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        print(
            "[错误] 输出文件已经存在。确认需要重建时添加 --overwrite：\n"
            + "\n".join(str(path) for path in existing_outputs),
            file=sys.stderr,
        )
        return 2

    input_files = find_input_files(raw_root)
    if not input_files:
        print("[错误] 没有找到可处理的数据文件。", file=sys.stderr)
        return 2

    output_root.mkdir(parents=True, exist_ok=True)

    source_counts: Counter[str] = Counter()
    file_counts: Counter[str] = Counter()
    mapped_field_counts: dict[str, Counter[str]] = defaultdict(Counter)
    observed_fields: dict[str, Counter[str]] = defaultdict(Counter)
    missing_core_fields: dict[str, Counter[str]] = defaultdict(Counter)
    rejection_counts: Counter[str] = Counter()

    total_read = 0
    total_normalized = 0
    total_rejected = 0

    with (
        normalized_path.open("w", encoding="utf-8", newline="\n") as normalized_file,
        rejected_path.open("w", encoding="utf-8", newline="\n") as rejected_file,
    ):
        for source_name, input_path in input_files:
            relative_file = input_path.relative_to(project_root).as_posix()
            source_split = infer_split(input_path)
            print(f"[处理] {source_name}: {relative_file}")

            try:
                for source_row, raw_record in enumerate(
                    read_records(input_path),
                    start=1,
                ):
                    total_read += 1

                    if not isinstance(raw_record, dict):
                        total_rejected += 1
                        rejection_counts["record_not_object"] += 1
                        write_jsonl_record(
                            rejected_file,
                            {
                                "source_name": source_name,
                                "source_file": relative_file,
                                "source_row": source_row,
                                "reason": "record_not_object",
                            },
                        )
                        continue

                    for raw_key in raw_record.keys():
                        observed_fields[source_name][str(raw_key)] += 1

                    try:
                        normalized_record = normalize_one_record(
                            source_name=source_name,
                            source_file=relative_file,
                            source_split=source_split,
                            source_row=source_row,
                            raw_record=raw_record,
                        )
                    except Exception as exc:  # noqa: BLE001
                        total_rejected += 1
                        rejection_counts["normalization_error"] += 1
                        write_jsonl_record(
                            rejected_file,
                            {
                                "source_name": source_name,
                                "source_file": relative_file,
                                "source_row": source_row,
                                "reason": "normalization_error",
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(),
                                "source_record_sha256": sha256_text(
                                    stable_json(raw_record)
                                ),
                            },
                        )
                        continue

                    # 规范化阶段只要求存在来源 ID。
                    # repo/base_commit/problem_statement 缺失时保留记录并计数，
                    # 由后续主实例注册表阶段判断是否可参与主流程。
                    for core_field in ("repo", "base_commit", "problem_statement"):
                        if is_empty(getattr(normalized_record, core_field)):
                            missing_core_fields[source_name][core_field] += 1

                    for canonical_name, original_name in (
                        normalized_record.mapped_fields.items()
                    ):
                        mapped_field_counts[source_name][
                            f"{canonical_name}<-{original_name}"
                        ] += 1

                    write_jsonl_record(
                        normalized_file,
                        asdict(normalized_record),
                    )

                    total_normalized += 1
                    source_counts[source_name] += 1
                    file_counts[relative_file] += 1

            except Exception as exc:  # noqa: BLE001
                total_rejected += 1
                rejection_counts["file_read_error"] += 1
                write_jsonl_record(
                    rejected_file,
                    {
                        "source_name": source_name,
                        "source_file": relative_file,
                        "source_row": 0,
                        "reason": "file_read_error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                )

    # canonical_record_id 必须做到“一条来源记录一个 ID”。
    # 这里再次扫描输出，防止未来字段映射修改后重新引入碰撞。
    canonical_id_counts: Counter[str] = Counter()
    with normalized_path.open("r", encoding="utf-8") as check_file:
        for line_number, raw_line in enumerate(check_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            canonical_id = str(record.get("canonical_record_id", "")).strip()
            if not canonical_id:
                raise RuntimeError(
                    f"规范化输出第 {line_number} 行缺少 canonical_record_id"
                )
            canonical_id_counts[canonical_id] += 1

    duplicate_canonical_record_ids = sorted(
        record_id
        for record_id, count in canonical_id_counts.items()
        if count > 1
    )
    if duplicate_canonical_record_ids:
        raise RuntimeError(
            "canonical_record_id 仍存在重复，示例："
            + ", ".join(duplicate_canonical_record_ids[:20])
        )

    field_catalog = {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now_iso(),
        "sources": {
            source_name: {
                "observed_fields": [
                    {
                        "field": field,
                        "record_occurrences": count,
                    }
                    for field, count in counts.most_common()
                ],
                "applied_mappings": [
                    {
                        "mapping": mapping,
                        "record_occurrences": count,
                    }
                    for mapping, count in mapped_field_counts[
                        source_name
                    ].most_common()
                ],
            }
            for source_name, counts in observed_fields.items()
        },
    }
    write_json(field_catalog_path, field_catalog)

    report = {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now_iso(),
        "status": "passed" if total_normalized > 0 else "failed",
        "input_file_count": len(input_files),
        "total_records_read": total_read,
        "total_records_normalized": total_normalized,
        "total_records_rejected": total_rejected,
        "source_counts": dict(source_counts),
        "file_counts": dict(file_counts),
        "missing_core_fields": {
            source_name: dict(counts)
            for source_name, counts in missing_core_fields.items()
        },
        "rejection_counts": dict(rejection_counts),
        "outputs": {
            "normalized_instances": normalized_path.relative_to(
                project_root
            ).as_posix(),
            "rejected_records": rejected_path.relative_to(
                project_root
            ).as_posix(),
            "observed_field_catalog": field_catalog_path.relative_to(
                project_root
            ).as_posix(),
        },
        "notes": [
            "该阶段只统一字段，不去重、不合并跨来源实例。",
            "patch 和 test_patch 被保留为离线标签资产，后续不得进入在线证据获取模型输入。",
            "repo、base_commit 或 problem_statement 缺失的记录不会在本阶段被删除。",
            "external_context 和 trajectory 是基于字段名关键词自动收集的候选数据，后续仍需来源专用解析。",
        ],
    }
    write_json(report_path, report)

    print(
        json.dumps(
            {
                "status": report["status"],
                "total_records_read": total_read,
                "total_records_normalized": total_normalized,
                "total_records_rejected": total_rejected,
                "source_counts": dict(source_counts),
                "output_directory": str(output_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0 if total_normalized > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
