#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将三个来源的规范化记录合并为 Master Instance Registry。

输入：
    data/processed/normalized_instances.jsonl

输出：
    data/registry/master_instances.jsonl
    data/registry/instance_aliases.jsonl
    data/registry/overlap_report.json
    data/registry/registry_conflicts.jsonl
    data/registry/unmatched_records.jsonl

本脚本：
- 不读取配置文件；
- 不联网；
- 不执行测试；
- 不生成修复补丁；
- 只做实例身份对齐、去重与冲突检测。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

SCRIPT_VERSION = "1.1.0"
MIN_PROBLEM_LENGTH = 80

# 只用于选择主实例的展示字段，不影响匹配逻辑。
SOURCE_PRIORITY = {
    "swebench": 0,
    "contextbench": 1,
    "swe_explore": 2,
}


class UnionFind:
    """并查集：把多条等价来源记录聚合为一个实例组。"""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return

        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_repo(value: Any) -> str | None:
    """把 GitHub 仓库统一为小写 owner/repo。"""
    text = clean_text(value)
    if not text:
        return None

    text = text.rstrip("/")
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
    return text.lower()


def normalize_commit(value: Any) -> str | None:
    text = clean_text(value)
    return text.lower() if text else None


def normalize_url(value: Any) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    return text.split("#", 1)[0].split("?", 1)[0].rstrip("/").lower()


def normalize_problem(value: Any) -> str | None:
    text = clean_text(value)
    if not text:
        return None

    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < MIN_PROBLEM_LENGTH:
        return None
    return text


def normalize_patch(value: Any) -> str | None:
    """Patch 仅用于离线实例对齐，不进入在线证据获取模型。"""
    text = clean_text(value)
    if not text:
        return None
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE).strip()
    return text or None


def source_id_key(record: dict[str, Any]) -> str | None:
    value = clean_text(record.get("source_instance_id"))
    if not value:
        return None

    value = value.lower()
    if value.startswith("generated-") or value.isdigit() or len(value) < 6:
        return None
    return value


def source_id_conflicts(
    left: dict[str, Any], right: dict[str, Any]
) -> bool:
    """两条记录都有可靠 source_instance_id 且不同时视为冲突。

    用于阻止同一仓库下不同 bug 被 repo+commit 或 repo+patch 规则误合并。
    """
    left_id = source_id_key(left)
    right_id = source_id_key(right)
    if left_id and right_id and left_id != right_id:
        return True
    return False


def issue_key(record: dict[str, Any]) -> str | None:
    return normalize_url(record.get("issue_url"))


def pr_key(record: dict[str, Any]) -> str | None:
    return normalize_url(record.get("pr_url"))


def repo_commit_key(record: dict[str, Any]) -> str | None:
    repo = normalize_repo(record.get("repo"))
    commit = normalize_commit(record.get("base_commit"))
    if not repo or not commit:
        return None

    # 只把标准 Git SHA 当强键，避免版本号或占位符误合并。
    if not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        return None
    return f"{repo}@{commit}"


def repo_patch_key(record: dict[str, Any]) -> str | None:
    repo = normalize_repo(record.get("repo"))
    patch = normalize_patch(record.get("patch"))
    if not repo or not patch:
        return None
    return f"{repo}@patch:{sha256_text(patch)}"


def repo_problem_key(record: dict[str, Any]) -> str | None:
    repo = normalize_repo(record.get("repo"))
    problem = normalize_problem(record.get("problem_statement"))
    if not repo or not problem:
        return None
    return f"{repo}@problem:{sha256_text(problem)}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON object")
            records.append(record)
    return records


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
            file_obj.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
                + "\n"
            )


def build_index(
    records: list[dict[str, Any]],
    key_function: Callable[[dict[str, Any]], str | None],
) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for record_index, record in enumerate(records):
        key = key_function(record)
        if key:
            index[key].append(record_index)
    return dict(index)


def union_exact_groups(
    union_find: UnionFind,
    records: list[dict[str, Any]],
    index: dict[str, list[int]],
    method: str,
    confidence: float,
    match_log: list[dict[str, Any]],
    conflict_check: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
) -> None:
    """将同一强键下的所有记录直接合并。

    若提供 conflict_check，则对每对记录检查是否冲突，冲突时跳过合并。
    """
    for key, indices in index.items():
        if len(indices) < 2:
            continue
        anchor = indices[0]
        for other in indices[1:]:
            if conflict_check and conflict_check(records[anchor], records[other]):
                continue
            union_find.union(anchor, other)
            match_log.append(
                {
                    "left_index": anchor,
                    "right_index": other,
                    "method": method,
                    "key": key,
                    "confidence": confidence,
                }
            )


def union_problem_groups(
    union_find: UnionFind,
    records: list[dict[str, Any]],
    index: dict[str, list[int]],
    match_log: list[dict[str, Any]],
) -> None:
    """
    问题描述哈希属于较弱键。

    只有在 repo 一致且双方 base_commit 不冲突时才合并。
    """
    for key, indices in index.items():
        if len(indices) < 2:
            continue

        for left_position, left_index in enumerate(indices):
            for right_index in indices[left_position + 1:]:
                left_commit = normalize_commit(
                    records[left_index].get("base_commit")
                )
                right_commit = normalize_commit(
                    records[right_index].get("base_commit")
                )

                if left_commit and right_commit and left_commit != right_commit:
                    continue

                if source_id_conflicts(
                    records[left_index], records[right_index]
                ):
                    continue

                union_find.union(left_index, right_index)
                match_log.append(
                    {
                        "left_index": left_index,
                        "right_index": right_index,
                        "method": "repo_problem_hash",
                        "key": key,
                        "confidence": 0.88,
                    }
                )


def choose_value(group: list[dict[str, Any]], field: str) -> Any:
    ordered = sorted(
        group,
        key=lambda record: (
            SOURCE_PRIORITY.get(str(record.get("source_name")), 99),
            str(record.get("source_file", "")),
            int(record.get("source_row", 0)),
        ),
    )
    for record in ordered:
        value = record.get(field)
        if value is not None and (not isinstance(value, str) or value.strip()):
            return value
    return None


def stable_master_id(group: list[dict[str, Any]]) -> str:
    """根据最强可用身份生成稳定主实例 ID。"""
    issue_keys = sorted(
        {key for record in group if (key := issue_key(record))}
    )
    pr_keys = sorted({key for record in group if (key := pr_key(record))})
    source_keys = sorted(
        {key for record in group if (key := source_id_key(record))}
    )
    commit_keys = sorted(
        {key for record in group if (key := repo_commit_key(record))}
    )

    # 优先用 issue/pr/source_instance_id 生成 ID。
    # 只有当它们都不可靠时才退回 commit：同一 commit 上可能有多个不同 bug，
    # 若用 commit 生成 ID 会导致不同 group 撞 ID。
    if len(issue_keys) == 1:
        identity = f"issue:{issue_keys[0]}"
    elif len(pr_keys) == 1:
        identity = f"pr:{pr_keys[0]}"
    elif len(source_keys) == 1:
        identity = f"source:{source_keys[0]}"
    elif len(commit_keys) == 1 and len(source_keys) == 0:
        # 仅当组内没有任何可靠 source_instance_id 时，才用 commit 作为身份。
        identity = f"commit:{commit_keys[0]}"
    else:
        # canonical_record_id 理论上唯一，但主实例 ID 不能只依赖该字段。
        # 同时加入来源、文件、行号和原记录哈希，避免历史规范化数据中的
        # 重复 canonical_record_id 继续传播为重复 canonical_instance_id。
        record_identities = sorted(
            stable_json(
                {
                    "canonical_record_id": record.get("canonical_record_id"),
                    "source_name": record.get("source_name"),
                    "source_file": record.get("source_file"),
                    "source_row": record.get("source_row"),
                    "source_record_sha256": record.get("source_record_sha256"),
                }
            )
            for record in group
        )
        identity = "records:" + "|".join(record_identities)

    return "mi-" + sha256_text(identity)[:24]


def collect_conflicts(
    canonical_instance_id: str,
    group: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """检查同一实例组中是否存在关键字段冲突。"""
    conflicts: list[dict[str, Any]] = []

    # 若组内所有可靠 source_instance_id 一致，repo 字段差异属于
    # 数据源写法不一致（如 material-ui vs mui/material），非误合并，
    # 降级为 medium，不触发退出码 3。
    reliable_ids = {
        source_id_key(record) for record in group if source_id_key(record)
    }
    all_same_source_id = len(reliable_ids) == 1

    checks: tuple[
        tuple[str, Callable[[Any], str | None], str], ...
    ] = (
        ("repo", normalize_repo, "medium" if all_same_source_id else "high"),
        ("base_commit", normalize_commit, "high"),
        (
            "problem_statement",
            lambda value: (
                sha256_text(text) if (text := normalize_problem(value)) else None
            ),
            "medium",
        ),
        (
            "patch",
            lambda value: (
                sha256_text(text) if (text := normalize_patch(value)) else None
            ),
            "medium",
        ),
    )

    for field_name, normalizer, severity in checks:
        values = sorted(
            {
                normalized
                for record in group
                if (normalized := normalizer(record.get(field_name)))
            }
        )
        if len(values) <= 1:
            continue

        # 仅保留定位信息与值摘要，避免深拷贝完整大文本（patch/problem
        # 可能达数 MB），否则大 group 会触发 MemoryError。
        conflicts.append(
            {
                "canonical_instance_id": canonical_instance_id,
                "field": field_name,
                "severity": severity,
                "distinct_normalized_values": values,
                "source_records": [
                    {
                        "source_name": record.get("source_name"),
                        "source_instance_id": record.get("source_instance_id"),
                        "source_file": record.get("source_file"),
                        "source_row": record.get("source_row"),
                        "value_length": (
                            len(str(record.get(field_name)))
                            if record.get(field_name) is not None
                            else 0
                        ),
                        "value_sha256_prefix": (
                            sha256_text(
                                str(record.get(field_name))
                            )[:12]
                            if record.get(field_name) is not None
                            else None
                        ),
                    }
                    for record in group
                    if record.get(field_name) not in (None, "")
                ],
            }
        )

    return conflicts


def build_match_evidence_by_root(
    match_log: list[dict[str, Any]],
    union_find: "UnionFind",
) -> dict[int, list[dict[str, Any]]]:
    """
    单次遍历 match_log，按每条记录的 root 聚合匹配证据。

    旧实现 group_match_methods 对每个 group 都遍历整个 match_log，
    复杂度 O(G * M)。这里改为 O(M) 一次扫描，再按 root 查询。
    """
    evidence_by_root: dict[int, Counter[tuple[str, float]]] = defaultdict(
        Counter
    )

    for item in match_log:
        left_root = union_find.find(item["left_index"])
        right_root = union_find.find(item["right_index"])
        # 合并是传递的，正常情况下 left_root == right_root。
        # 若因并发修改出现不一致，取 left_root 作为归属。
        root = left_root if left_root == right_root else left_root
        evidence_by_root[root][
            (item["method"], item["confidence"])
        ] += 1

    return {
        root: [
            {
                "method": method,
                "confidence": confidence,
                "match_count": count,
            }
            for (method, confidence), count in sorted(
                counts.items(),
                key=lambda item: (-item[0][1], item[0][0]),
            )
        ]
        for root, counts in evidence_by_root.items()
    }


def group_match_methods(
    group_indices: list[int],
    match_log: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    group_set = set(group_indices)
    counts: Counter[tuple[str, float]] = Counter()

    for item in match_log:
        if item["left_index"] in group_set and item["right_index"] in group_set:
            counts[(item["method"], item["confidence"])] += 1

    return [
        {
            "method": method,
            "confidence": confidence,
            "match_count": count,
        }
        for (method, confidence), count in sorted(
            counts.items(),
            key=lambda item: (-item[0][1], item[0][0]),
        )
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="建立 Master Instance Registry，不读取配置文件。"
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
        help="允许覆盖已有输出。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()

    input_path = (
        project_root
        / "data"
        / "processed"
        / "normalized_instances.jsonl"
    )
    registry_root = project_root / "data" / "registry"

    master_path = registry_root / "master_instances.jsonl"
    aliases_path = registry_root / "instance_aliases.jsonl"
    overlap_path = registry_root / "overlap_report.json"
    conflicts_path = registry_root / "registry_conflicts.jsonl"
    unmatched_path = registry_root / "unmatched_records.jsonl"

    if not input_path.exists():
        print(f"[错误] 输入不存在：{input_path}", file=sys.stderr)
        return 2

    output_paths = (
        master_path,
        aliases_path,
        overlap_path,
        conflicts_path,
        unmatched_path,
    )
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        print(
            "[错误] 输出已存在，确认重建时添加 --overwrite：\n"
            + "\n".join(str(path) for path in existing),
            file=sys.stderr,
        )
        return 2

    records = read_jsonl(input_path)
    if not records:
        print("[错误] normalized_instances.jsonl 为空。", file=sys.stderr)
        return 2

    union_find = UnionFind(len(records))
    match_log: list[dict[str, Any]] = []
    index_report: dict[str, dict[str, int]] = {}

    # 按置信度从高到低建立强匹配。
    # 需要检查 source_instance_id 冲突的方法：同仓库同 commit/patch
    # 但 instance_id 不同时，说明是不同 bug，不应合并。
    strong_specs: tuple[
        tuple[str, Callable[[dict[str, Any]], str | None], float, bool], ...
    ] = (
        ("issue_url", issue_key, 1.00, False),
        ("pr_url", pr_key, 1.00, False),
        ("source_instance_id", source_id_key, 0.99, False),
        ("repo_base_commit", repo_commit_key, 0.98, True),
        ("repo_patch_hash", repo_patch_key, 0.95, True),
    )

    for method, key_function, confidence, check_id_conflict in strong_specs:
        index = build_index(records, key_function)
        union_exact_groups(
            union_find,
            records,
            index,
            method,
            confidence,
            match_log,
            conflict_check=source_id_conflicts if check_id_conflict else None,
        )
        index_report[method] = {
            "unique_keys": len(index),
            "duplicate_keys": sum(
                1 for indices in index.values() if len(indices) > 1
            ),
            "records_with_key": sum(len(indices) for indices in index.values()),
        }

    problem_index = build_index(records, repo_problem_key)
    union_problem_groups(union_find, records, problem_index, match_log)
    index_report["repo_problem_hash"] = {
        "unique_keys": len(problem_index),
        "duplicate_keys": sum(
            1 for indices in problem_index.values() if len(indices) > 1
        ),
        "records_with_key": sum(
            len(indices) for indices in problem_index.values()
        ),
    }

    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for record_index in range(len(records)):
        grouped_indices[union_find.find(record_index)].append(record_index)

    # 预先按 root 聚合匹配证据，避免每个 group 重复遍历整个 match_log。
    # 旧实现复杂度 O(G * M)，这里降到 O(M) 一次扫描。
    evidence_by_root = build_match_evidence_by_root(match_log, union_find)

    master_records: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    for root, indices in grouped_indices.items():
        group = [records[index] for index in indices]
        canonical_instance_id = stable_master_id(group)
        source_names = sorted(
            {str(record.get("source_name")) for record in group}
        )

        group_conflicts = collect_conflicts(canonical_instance_id, group)
        conflicts.extend(group_conflicts)

        master_record = {
            "canonical_instance_id": canonical_instance_id,
            # 后续只能按 task_group_id 划分数据，防止同任务跨集合泄漏。
            "task_group_id": canonical_instance_id,
            "repo": choose_value(group, "repo"),
            "base_commit": choose_value(group, "base_commit"),
            "problem_statement": choose_value(group, "problem_statement"),
            "patch": choose_value(group, "patch"),
            "test_patch": choose_value(group, "test_patch"),
            "issue_url": choose_value(group, "issue_url"),
            "pr_url": choose_value(group, "pr_url"),
            "source_names": source_names,
            "source_splits": sorted(
                {
                    str(record.get("source_split"))
                    for record in group
                    if record.get("source_split") not in (None, "")
                }
            ),
            "source_record_count": len(group),
            "has_external_context": any(
                bool(record.get("external_context")) for record in group
            ),
            "has_trajectory": any(
                bool(record.get("trajectory")) for record in group
            ),
            "match_evidence": evidence_by_root.get(root, []),
            "conflict_count": len(group_conflicts),
            "has_high_severity_conflict": any(
                item["severity"] == "high" for item in group_conflicts
            ),
            "source_records": [
                {
                    "canonical_record_id": record.get("canonical_record_id"),
                    "source_name": record.get("source_name"),
                    "source_split": record.get("source_split"),
                    "source_file": record.get("source_file"),
                    "source_row": record.get("source_row"),
                    "source_instance_id": record.get("source_instance_id"),
                    "source_record_sha256": record.get("source_record_sha256"),
                    "external_context": record.get("external_context", {}),
                    "trajectory": record.get("trajectory", {}),
                    "source_metadata": record.get("source_metadata", {}),
                }
                for record in sorted(
                    group,
                    key=lambda item: (
                        SOURCE_PRIORITY.get(str(item.get("source_name")), 99),
                        str(item.get("source_file", "")),
                        int(item.get("source_row", 0)),
                    ),
                )
            ],
        }
        master_records.append(master_record)

        for record in group:
            aliases.append(
                {
                    "canonical_instance_id": canonical_instance_id,
                    "task_group_id": canonical_instance_id,
                    "canonical_record_id": record.get("canonical_record_id"),
                    "source_name": record.get("source_name"),
                    "source_instance_id": record.get("source_instance_id"),
                    "source_file": record.get("source_file"),
                    "source_row": record.get("source_row"),
                }
            )

        if len(group) == 1:
            record = group[0]
            unmatched.append(
                {
                    "canonical_instance_id": canonical_instance_id,
                    "canonical_record_id": record.get("canonical_record_id"),
                    "source_name": record.get("source_name"),
                    "source_instance_id": record.get("source_instance_id"),
                    "repo": record.get("repo"),
                    "base_commit": record.get("base_commit"),
                    "reason": "未找到其他来源或重复文件中的匹配记录",
                }
            )

    # 主实例 ID 必须全局唯一。发现碰撞时立即失败，不能让划分阶段兜底。
    master_id_counts = Counter(
        str(record["canonical_instance_id"])
        for record in master_records
    )
    duplicate_master_ids = sorted(
        master_id
        for master_id, count in master_id_counts.items()
        if count > 1
    )
    if duplicate_master_ids:
        collision_details = []
        for master_id in duplicate_master_ids[:20]:
            collision_details.append(
                {
                    "canonical_instance_id": master_id,
                    "groups": [
                        {
                            "repo": record.get("repo"),
                            "base_commit": record.get("base_commit"),
                            "source_names": record.get("source_names"),
                            "source_records": [
                                {
                                    "source_name": item.get("source_name"),
                                    "source_file": item.get("source_file"),
                                    "source_row": item.get("source_row"),
                                    "source_instance_id": item.get("source_instance_id"),
                                    "canonical_record_id": item.get("canonical_record_id"),
                                }
                                for item in record.get("source_records", [])
                            ],
                        }
                        for record in master_records
                        if record["canonical_instance_id"] == master_id
                    ],
                }
            )

        collision_path = registry_root / "master_id_collisions.json"
        write_json(
            collision_path,
            {
                "script_version": SCRIPT_VERSION,
                "duplicate_count": len(duplicate_master_ids),
                "collisions": collision_details,
            },
        )
        raise RuntimeError(
            "canonical_instance_id 仍存在重复；诊断已写入 "
            f"{collision_path}"
        )

    master_records.sort(key=lambda item: item["canonical_instance_id"])
    aliases.sort(
        key=lambda item: (
            item["canonical_instance_id"],
            str(item.get("source_name", "")),
            str(item.get("source_instance_id", "")),
        )
    )
    conflicts.sort(
        key=lambda item: (item["canonical_instance_id"], item["field"])
    )
    unmatched.sort(key=lambda item: item["canonical_instance_id"])

    write_jsonl(master_path, master_records)
    write_jsonl(aliases_path, aliases)
    write_jsonl(conflicts_path, conflicts)
    write_jsonl(unmatched_path, unmatched)

    source_combinations: Counter[str] = Counter()
    source_pair_overlap: Counter[str] = Counter()
    group_sizes: Counter[int] = Counter()

    for record in master_records:
        source_names = record["source_names"]
        source_combinations["+".join(source_names)] += 1
        group_sizes[record["source_record_count"]] += 1
        for left_position, left_source in enumerate(source_names):
            for right_source in source_names[left_position + 1:]:
                source_pair_overlap[f"{left_source}<->{right_source}"] += 1

    high_conflict_ids = {
        item["canonical_instance_id"]
        for item in conflicts
        if item["severity"] == "high"
    }

    report = {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now(),
        "status": "warning" if high_conflict_ids else "passed",
        "input_record_count": len(records),
        "master_instance_count": len(master_records),
        "merged_group_count": sum(
            1 for record in master_records if record["source_record_count"] > 1
        ),
        "cross_source_group_count": sum(
            1 for record in master_records if len(record["source_names"]) > 1
        ),
        "single_record_group_count": len(unmatched),
        "conflict_count": len(conflicts),
        "high_conflict_group_count": len(high_conflict_ids),
        "source_record_counts": dict(
            Counter(str(record.get("source_name")) for record in records)
        ),
        "source_combination_counts": dict(sorted(source_combinations.items())),
        "source_pair_overlap": dict(sorted(source_pair_overlap.items())),
        "group_size_distribution": {
            str(size): count for size, count in sorted(group_sizes.items())
        },
        "matching_index_statistics": index_report,
        "matching_policy": [
            "issue_url 完全一致",
            "pr_url 完全一致",
            "可靠 source_instance_id 完全一致",
            "repo + base_commit 完全一致",
            "repo + patch 哈希完全一致",
            "repo + 问题描述哈希一致且 base_commit 不冲突",
        ],
        "outputs": {
            "master_instances": master_path.relative_to(project_root).as_posix(),
            "instance_aliases": aliases_path.relative_to(project_root).as_posix(),
            "registry_conflicts": conflicts_path.relative_to(project_root).as_posix(),
            "unmatched_records": unmatched_path.relative_to(project_root).as_posix(),
        },
    }
    write_json(overlap_path, report)

    print(
        json.dumps(
            {
                "status": report["status"],
                "input_record_count": len(records),
                "master_instance_count": len(master_records),
                "merged_group_count": report["merged_group_count"],
                "cross_source_group_count": report["cross_source_group_count"],
                "single_record_group_count": len(unmatched),
                "high_conflict_group_count": len(high_conflict_ids),
                "output_directory": str(registry_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    # 输出已经完整生成；退出码 3 表示需要人工检查高严重度冲突。
    return 3 if high_conflict_ids else 0


if __name__ == "__main__":
    raise SystemExit(main())
