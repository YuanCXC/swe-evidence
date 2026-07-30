#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
冻结主实例数据划分，不读取配置文件。

输入：
    data/registry/master_instances.jsonl

输出：
    data/splits/train.jsonl
    data/splits/dev.jsonl
    data/splits/test_retrieval.jsonl
    data/splits/test_sufficiency.jsonl
    data/splits/split_assignments.jsonl
    data/splits/quarantined_instances.jsonl
    data/splits/split_summary.json
    data/splits/split_lock.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_VERSION = "1.0.0"
DEFAULT_SEED = 20260729
SPLIT_RATIOS = {
    "train": 0.70,
    "dev": 0.10,
    "test_retrieval": 0.10,
    "test_sufficiency": 0.10,
}
SPLIT_ORDER = tuple(SPLIT_RATIOS)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return isinstance(value, (list, tuple, dict, set)) and not value


def normalize_repo(value: Any) -> str | None:
    if is_empty(value):
        return None

    text = str(value).strip().rstrip("/")
    for prefix in (
        "https://github.com/",
        "http://github.com/",
        "git@github.com:",
    ):
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break

    if text.endswith(".git"):
        text = text[:-4]

    parts = [part for part in text.split("/") if part]
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}".lower()
    return text.lower() or None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8-sig") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} 第 {line_number} 行 JSON 无效：{exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path} 第 {line_number} 行不是 JSON object"
                )
            records.append(record)
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for record in records:
            file_obj.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )


def quarantine_reasons(record: dict[str, Any]) -> list[str]:
    reasons = []

    for field in (
        "canonical_instance_id",
        "task_group_id",
        "repo",
        "base_commit",
        "problem_statement",
    ):
        if is_empty(record.get(field)):
            reasons.append(f"missing_{field}")

    if bool(record.get("has_high_severity_conflict")):
        reasons.append("high_severity_registry_conflict")

    return reasons


def group_key(record: dict[str, Any]) -> str:
    repo = normalize_repo(record.get("repo"))
    if repo:
        return f"repo:{repo}"
    return f"task:{record['task_group_id']}"


def integer_targets(total: int) -> dict[str, int]:
    exact = {
        name: total * ratio
        for name, ratio in SPLIT_RATIOS.items()
    }
    targets = {
        name: math.floor(value)
        for name, value in exact.items()
    }
    remaining = total - sum(targets.values())

    ranked = sorted(
        SPLIT_ORDER,
        key=lambda name: (
            exact[name] - targets[name],
            -SPLIT_ORDER.index(name),
        ),
        reverse=True,
    )

    for name in ranked[:remaining]:
        targets[name] += 1

    return targets


def choose_split(
    size: int,
    counts: dict[str, int],
    targets: dict[str, int],
) -> str:
    def score(name: str) -> tuple[float, float, int]:
        simulated = dict(counts)
        simulated[name] += size

        total_error = sum(
            abs(simulated[split] - targets[split])
            / max(targets[split], 1)
            for split in SPLIT_ORDER
        )
        overflow = max(
            simulated[name] - targets[name],
            0,
        ) / max(targets[name], 1)

        return total_error, overflow, SPLIT_ORDER.index(name)

    return min(SPLIT_ORDER, key=score)


def assign_by_repo(
    records: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, str], dict[str, int], int]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[group_key(record)].append(record)

    targets = integer_targets(len(records))
    counts = {name: 0 for name in SPLIT_ORDER}

    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            sha256_text(f"{seed}|{item[0]}"),
        ),
    )

    assignments: dict[str, str] = {}
    for key, group_records in ordered_groups:
        split = choose_split(
            len(group_records),
            counts,
            targets,
        )
        assignments[key] = split
        counts[split] += len(group_records)

    return assignments, targets, len(groups)


def validate(
    assignments: list[dict[str, Any]],
) -> dict[str, Any]:
    canonical_splits: dict[str, set[str]] = defaultdict(set)
    task_splits: dict[str, set[str]] = defaultdict(set)
    repo_splits: dict[str, set[str]] = defaultdict(set)

    for item in assignments:
        split = item["split"]
        canonical_splits[
            item["canonical_instance_id"]
        ].add(split)
        task_splits[item["task_group_id"]].add(split)

        repo = normalize_repo(item.get("repo"))
        if repo:
            repo_splits[repo].add(split)

    canonical_leaks = {
        key: sorted(value)
        for key, value in canonical_splits.items()
        if len(value) > 1
    }
    task_leaks = {
        key: sorted(value)
        for key, value in task_splits.items()
        if len(value) > 1
    }
    repo_leaks = {
        key: sorted(value)
        for key, value in repo_splits.items()
        if len(value) > 1
    }

    return {
        "canonical_instance_leakage_count": len(canonical_leaks),
        "task_group_leakage_count": len(task_leaks),
        "repo_leakage_count": len(repo_leaks),
        "canonical_instance_leakage_examples": dict(
            list(canonical_leaks.items())[:20]
        ),
        "task_group_leakage_examples": dict(
            list(task_leaks.items())[:20]
        ),
        "repo_leakage_examples": dict(
            list(repo_leaks.items())[:20]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="冻结主实例数据划分，不读取配置文件。"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()

    input_path = (
        project_root
        / "data"
        / "registry"
        / "master_instances.jsonl"
    )
    output_root = project_root / "data" / "splits"

    split_paths = {
        name: output_root / f"{name}.jsonl"
        for name in SPLIT_ORDER
    }
    assignment_path = output_root / "split_assignments.jsonl"
    quarantine_path = output_root / "quarantined_instances.jsonl"
    summary_path = output_root / "split_summary.json"
    lock_path = output_root / "split_lock.json"

    output_paths = [
        *split_paths.values(),
        assignment_path,
        quarantine_path,
        summary_path,
        lock_path,
    ]

    if not input_path.exists():
        print(
            f"[错误] 输入不存在：{input_path}",
            file=sys.stderr,
        )
        return 2

    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        print(
            "[错误] 输出已存在，重建时添加 --overwrite：\n"
            + "\n".join(str(path) for path in existing),
            file=sys.stderr,
        )
        return 2

    master_records = read_jsonl(input_path)
    if not master_records:
        print("[错误] 主实例注册表为空。", file=sys.stderr)
        return 2

    canonical_ids = [
        str(record.get("canonical_instance_id", "")).strip()
        for record in master_records
    ]
    duplicated_ids = sorted(
        key
        for key, count in Counter(canonical_ids).items()
        if key and count > 1
    )
    if duplicated_ids:
        print(
            "[错误] canonical_instance_id 重复：\n"
            + "\n".join(duplicated_ids[:20]),
            file=sys.stderr,
        )
        return 2

    eligible = []
    quarantined = []

    for record in master_records:
        reasons = quarantine_reasons(record)
        if reasons:
            quarantined.append(
                {
                    "canonical_instance_id": record.get(
                        "canonical_instance_id"
                    ),
                    "task_group_id": record.get("task_group_id"),
                    "repo": record.get("repo"),
                    "base_commit": record.get("base_commit"),
                    "source_names": record.get("source_names", []),
                    "reasons": reasons,
                    "master_record": record,
                }
            )
        else:
            eligible.append(record)

    if not eligible:
        print("[错误] 没有可划分实例。", file=sys.stderr)
        return 2

    repo_assignments, targets, repo_group_count = (
        assign_by_repo(eligible, args.seed)
    )

    split_records = {
        name: []
        for name in SPLIT_ORDER
    }
    assignment_records = []

    for record in eligible:
        key = group_key(record)
        split = repo_assignments[key]

        output_record = dict(record)
        output_record["split"] = split
        output_record["split_group_key"] = key
        split_records[split].append(output_record)

        assignment_records.append(
            {
                "canonical_instance_id": record[
                    "canonical_instance_id"
                ],
                "task_group_id": record["task_group_id"],
                "repo": normalize_repo(record.get("repo")),
                "base_commit": record.get("base_commit"),
                "split": split,
                "group_key": key,
                "seed": args.seed,
                "script_version": SCRIPT_VERSION,
            }
        )

    for name in SPLIT_ORDER:
        split_records[name].sort(
            key=lambda item: item["canonical_instance_id"]
        )
    assignment_records.sort(
        key=lambda item: (
            item["split"],
            item["canonical_instance_id"],
        )
    )
    quarantined.sort(
        key=lambda item: str(
            item.get("canonical_instance_id", "")
        )
    )

    validation = validate(assignment_records)
    actual_counts = {
        name: len(split_records[name])
        for name in SPLIT_ORDER
    }
    repo_counts = {
        name: len(
            {
                normalize_repo(record.get("repo"))
                for record in split_records[name]
            }
        )
        for name in SPLIT_ORDER
    }

    status = "passed"
    if any(
        validation[key] != 0
        for key in (
            "canonical_instance_leakage_count",
            "task_group_leakage_count",
            "repo_leakage_count",
        )
    ):
        status = "failed"

    if any(actual_counts[name] == 0 for name in SPLIT_ORDER):
        status = "failed"

    output_root.mkdir(parents=True, exist_ok=True)

    for name, path in split_paths.items():
        write_jsonl(path, split_records[name])

    write_jsonl(assignment_path, assignment_records)
    write_jsonl(quarantine_path, quarantined)

    source_coverage = {}
    for name in SPLIT_ORDER:
        counter: Counter[str] = Counter()
        for record in split_records[name]:
            for source_name in record.get("source_names", []):
                counter[str(source_name)] += 1
        source_coverage[name] = dict(sorted(counter.items()))

    summary = {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now(),
        "status": status,
        "seed": args.seed,
        "split_ratios": SPLIT_RATIOS,
        "total_master_instances": len(master_records),
        "eligible_instance_count": len(eligible),
        "quarantined_instance_count": len(quarantined),
        "repo_group_count": repo_group_count,
        "target_instance_counts": targets,
        "actual_instance_counts": actual_counts,
        "repo_counts": repo_counts,
        "source_coverage": source_coverage,
        "validation": validation,
        "quarantine_reason_counts": dict(
            Counter(
                reason
                for record in quarantined
                for reason in record["reasons"]
            )
        ),
    }
    write_json(summary_path, summary)

    lock = {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now(),
        "seed": args.seed,
        "split_ratios": SPLIT_RATIOS,
        "input_path": input_path.relative_to(
            project_root
        ).as_posix(),
        "input_sha256": sha256_file(input_path),
        "assignment_path": assignment_path.relative_to(
            project_root
        ).as_posix(),
        "assignment_sha256": sha256_file(assignment_path),
        "eligible_instance_count": len(eligible),
        "quarantined_instance_count": len(quarantined),
    }
    write_json(lock_path, lock)

    print(
        json.dumps(
            {
                "status": status,
                "eligible_instance_count": len(eligible),
                "quarantined_instance_count": len(quarantined),
                "target_instance_counts": targets,
                "actual_instance_counts": actual_counts,
                "repo_counts": repo_counts,
                "task_group_leakage_count": validation[
                    "task_group_leakage_count"
                ],
                "repo_leakage_count": validation[
                    "repo_leakage_count"
                ],
                "output_directory": str(output_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0 if status == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
