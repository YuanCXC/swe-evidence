#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从已有标签资产中抽取轻量 Evidence Anchors。

本阶段不会：
- 全量扫描仓库；
- 全量下载 Git blob；
- 解析全仓库 AST；
- 生成补丁；
- 执行测试；
- 判断修复是否成功。

本阶段只做：
1. 从 patch/test_patch 中提取修复前文件与 old-side 行号；
2. 从 external_context/trajectory 中提取文件、行号和符号候选；
3. 使用 Git tree 验证候选文件确实存在于 base_commit；
4. 输出后续 Evidence Unit 解析所需的候选文件清单。

输入：
    data/registry/master_instances.jsonl
    data/registry/git_snapshots.jsonl

输出：
    data/processed/evidence_anchors/evidence_anchors.jsonl
    data/processed/evidence_anchors/candidate_files.jsonl
    data/processed/evidence_anchors/anchor_failures.jsonl
    data/processed/evidence_anchors/anchor_report.json

注意：
- patch、test_patch、gold context 和 trajectory 都属于 label_only；
- 后续在线证据获取模型不得直接读取这些标签；
- 本脚本输出的 anchors 用于离线数据构造和监督对齐。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator

SCRIPT_VERSION = "1.0.0"

VALID_SPLITS = {
    "train",
    "dev",
    "test_retrieval",
    "test_sufficiency",
}

PATH_KEYS = {
    "file",
    "file_path",
    "filepath",
    "filename",
    "path",
    "relative_path",
    "source_file",
    "target_file",
}

START_LINE_KEYS = {
    "start_line",
    "line_start",
    "start",
    "begin_line",
    "from_line",
}

END_LINE_KEYS = {
    "end_line",
    "line_end",
    "end",
    "stop_line",
    "to_line",
}

SINGLE_LINE_KEYS = {
    "line",
    "line_no",
    "line_number",
    "lineno",
}

SYMBOL_KEYS = {
    "symbol",
    "symbol_name",
    "function",
    "function_name",
    "method",
    "method_name",
    "class",
    "class_name",
    "name",
}

# 仅将这些后缀视为较可信的仓库文件路径。
KNOWN_FILE_SUFFIXES = {
    ".py", ".pyi",
    ".js", ".jsx", ".mjs", ".cjs",
    ".ts", ".tsx",
    ".java", ".kt", ".kts",
    ".go", ".rs",
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp",
    ".cs", ".rb", ".php", ".swift", ".scala",
    ".sh", ".bash", ".zsh", ".fish",
    ".sql", ".r", ".R",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".xml", ".json", ".jsonl", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".conf", ".properties",
    ".md", ".rst", ".txt",
    ".gradle", ".cmake",
}

SPECIAL_FILE_NAMES = {
    "Dockerfile",
    "Makefile",
    "CMakeLists.txt",
    "BUILD",
    "WORKSPACE",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "tox.ini",
    "requirements.txt",
    "package.json",
    "tsconfig.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
}


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


def normalize_key(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = text.replace("-", "_").replace(" ", "_").replace(".", "_")
    text = re.sub(r"_+", "_", text)
    return text.lower().strip("_")


def normalize_path(value: Any) -> str | None:
    """
    将外部路径统一为仓库相对 POSIX 路径。

    拒绝：
    - 绝对路径；
    - URL；
    - 含 .. 的路径；
    - /dev/null；
    - 看起来不像源码、测试、配置或文档文件的字符串。
    """
    if value is None:
        return None

    text = str(value).strip().strip("`'\"")
    if not text:
        return None

    # 去除常见行号后缀，例如 src/a.py:12、src/a.py#L12-L20。
    text = re.sub(r"#L\d+(?:-L?\d+)?$", "", text)
    text = re.sub(r":\d+(?:-\d+)?$", "", text)

    # 去掉 unified diff 前缀。
    if text.startswith("a/") or text.startswith("b/"):
        text = text[2:]

    text = text.replace("\\", "/")
    text = re.sub(r"/+", "/", text)
    text = text.strip("/")

    if not text or text == "dev/null" or text == "/dev/null":
        return None

    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", text):
        return None

    if re.match(r"^[A-Za-z]:/", text):
        return None

    parts = PurePosixPath(text).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None

    file_name = parts[-1]
    suffix = PurePosixPath(file_name).suffix

    if suffix not in KNOWN_FILE_SUFFIXES and file_name not in SPECIAL_FILE_NAMES:
        return None

    return PurePosixPath(*parts).as_posix()


def parse_positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


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


def run_git(
    project_root: Path,
    arguments: list[str],
    *,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.setdefault("GIT_TERMINAL_PROMPT", "0")

    return subprocess.run(
        ["git", *arguments],
        cwd=str(project_root),
        env=environment,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def make_anchor(
    *,
    canonical_instance_id: str,
    snapshot_id: str,
    repo: str,
    resolved_commit: str,
    file_path: str,
    start_line: int | None,
    end_line: int | None,
    symbol: str | None,
    anchor_type: str,
    evidence_role: str,
    provenance: str,
    source_detail: str,
    confidence: float,
) -> dict[str, Any]:
    normalized_start = start_line if start_line and start_line > 0 else None
    normalized_end = end_line if end_line and end_line > 0 else normalized_start

    if (
        normalized_start is not None
        and normalized_end is not None
        and normalized_end < normalized_start
    ):
        normalized_start, normalized_end = (
            normalized_end,
            normalized_start,
        )

    identity = {
        "canonical_instance_id": canonical_instance_id,
        "snapshot_id": snapshot_id,
        "file_path": file_path,
        "start_line": normalized_start,
        "end_line": normalized_end,
        "symbol": symbol,
        "anchor_type": anchor_type,
        "provenance": provenance,
        "source_detail": source_detail,
    }

    return {
        "anchor_id": "anchor-" + sha256_text(stable_json(identity))[:24],
        "canonical_instance_id": canonical_instance_id,
        "snapshot_id": snapshot_id,
        "repo": repo,
        "resolved_commit": resolved_commit,
        "file_path": file_path,
        "start_line": normalized_start,
        "end_line": normalized_end,
        "symbol": symbol,
        "anchor_type": anchor_type,
        "evidence_role": evidence_role,
        "provenance": provenance,
        "visibility": "label_only",
        "confidence": confidence,
        "source_detail": source_detail,
        "file_exists_at_base_commit": None,
    }


def parse_unified_diff(
    diff_text: Any,
    *,
    canonical_instance_id: str,
    snapshot: dict[str, Any],
    provenance: str,
    evidence_role: str,
) -> list[dict[str, Any]]:
    """
    解析 unified diff 的 old-side 行号。

    对新增文件：
        修复前不存在，无法生成 pre-fix 文件证据锚点。

    对纯插入 hunk：
        old_count=0，记录 insertion_boundary。
    """
    if not isinstance(diff_text, str) or not diff_text.strip():
        return []

    anchors: list[dict[str, Any]] = []
    current_old_path: str | None = None
    current_new_path: str | None = None
    hunk_index = 0

    diff_git_pattern = re.compile(
        r"^diff --git a/(.+?) b/(.+?)$"
    )
    hunk_pattern = re.compile(
        r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
        r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
    )

    for raw_line in diff_text.splitlines():
        diff_match = diff_git_pattern.match(raw_line)
        if diff_match:
            current_old_path = normalize_path(diff_match.group(1))
            current_new_path = normalize_path(diff_match.group(2))
            continue

        if raw_line.startswith("--- "):
            raw_old = raw_line[4:].strip()
            current_old_path = normalize_path(raw_old)
            continue

        if raw_line.startswith("+++ "):
            raw_new = raw_line[4:].strip()
            current_new_path = normalize_path(raw_new)
            continue

        hunk_match = hunk_pattern.match(raw_line)
        if not hunk_match:
            continue

        hunk_index += 1
        old_start = int(hunk_match.group("old_start"))
        old_count = int(hunk_match.group("old_count") or "1")

        # 新文件 old-side 是 /dev/null，不产生修复前代码锚点。
        target_path = current_old_path
        if target_path is None:
            continue

        if old_count == 0:
            start_line = max(1, old_start)
            end_line = start_line
            anchor_type = "patch_insertion_boundary"
        else:
            start_line = old_start
            end_line = old_start + old_count - 1
            anchor_type = "patch_old_hunk"

        anchors.append(
            make_anchor(
                canonical_instance_id=canonical_instance_id,
                snapshot_id=snapshot["snapshot_id"],
                repo=snapshot["repo"],
                resolved_commit=snapshot["resolved_commit"],
                file_path=target_path,
                start_line=start_line,
                end_line=end_line,
                symbol=None,
                anchor_type=anchor_type,
                evidence_role=evidence_role,
                provenance=provenance,
                source_detail=f"hunk:{hunk_index}",
                confidence=1.0,
            )
        )

    return anchors


def extract_path_from_string(text: str) -> tuple[str | None, int | None, int | None]:
    """
    从字符串中抽取：
        path
        path:12
        path:12-20
        path#L12-L20
    """
    stripped = text.strip().strip("`'\"")

    hash_match = re.search(
        r"(?P<path>[A-Za-z0-9_./\\-]+)"
        r"#L(?P<start>\d+)(?:-L?(?P<end>\d+))?",
        stripped,
    )
    if hash_match:
        path = normalize_path(hash_match.group("path"))
        if path:
            start = parse_positive_int(hash_match.group("start"))
            end = parse_positive_int(hash_match.group("end")) or start
            return path, start, end

    colon_match = re.search(
        r"(?P<path>[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+)"
        r":(?P<start>\d+)(?:-(?P<end>\d+))?",
        stripped,
    )
    if colon_match:
        path = normalize_path(colon_match.group("path"))
        if path:
            start = parse_positive_int(colon_match.group("start"))
            end = parse_positive_int(colon_match.group("end")) or start
            return path, start, end

    path = normalize_path(stripped)
    return path, None, None


def walk_payload(
    payload: Any,
    *,
    location: str = "$",
) -> Iterator[tuple[Any, str]]:
    """递归遍历外部标签 payload。

    遇到 JSON 字符串（strip 后以 { 或 [ 开头）时自动解析后递归遍历，
    以支持 ContextBench gold_context 这类被序列化为字符串的嵌套结构。
    """
    yield payload, location

    if isinstance(payload, dict):
        for key, value in payload.items():
            child_location = f"{location}.{key}"
            yield from walk_payload(
                value,
                location=child_location,
            )
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            child_location = f"{location}[{index}]"
            yield from walk_payload(
                value,
                location=child_location,
            )
    elif isinstance(payload, str):
        stripped = payload.strip()
        if stripped and stripped[0] in "[{":
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                parsed = None
            if parsed is not None and parsed != payload:
                yield from walk_payload(
                    parsed,
                    location=f"{location}#parsed",
                )


def extract_structured_anchor(
    value: dict[str, Any],
) -> tuple[str | None, int | None, int | None, str | None]:
    normalized = {
        normalize_key(key): item
        for key, item in value.items()
    }

    path_value = None
    for key in PATH_KEYS:
        if key in normalized:
            path_value = normalized[key]
            break

    path = normalize_path(path_value)
    if path is None:
        return None, None, None, None

    start_line = None
    end_line = None
    symbol = None

    for key in START_LINE_KEYS:
        if key in normalized:
            start_line = parse_positive_int(normalized[key])
            if start_line:
                break

    for key in END_LINE_KEYS:
        if key in normalized:
            end_line = parse_positive_int(normalized[key])
            if end_line:
                break

    if start_line is None:
        for key in SINGLE_LINE_KEYS:
            if key in normalized:
                start_line = parse_positive_int(normalized[key])
                if start_line:
                    end_line = start_line
                    break

    for key in SYMBOL_KEYS:
        if key in normalized and normalized[key] is not None:
            text = str(normalized[key]).strip()
            if text and len(text) <= 300:
                symbol = text
                break

    return path, start_line, end_line, symbol


def extract_payload_anchors(
    payload: Any,
    *,
    canonical_instance_id: str,
    snapshot: dict[str, Any],
    provenance: str,
    evidence_role: str,
    confidence: float,
) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    seen_candidates: set[tuple[Any, ...]] = set()

    for value, location in walk_payload(payload):
        path: str | None = None
        start_line: int | None = None
        end_line: int | None = None
        symbol: str | None = None
        anchor_type = "external_file"

        if isinstance(value, dict):
            path, start_line, end_line, symbol = (
                extract_structured_anchor(value)
            )
            if path:
                anchor_type = (
                    "external_span"
                    if start_line is not None
                    else "external_file"
                )

        elif isinstance(value, str) and len(value) <= 2000:
            path, start_line, end_line = extract_path_from_string(value)
            if path:
                anchor_type = (
                    "external_span"
                    if start_line is not None
                    else "external_file"
                )

        if path is None:
            continue

        dedup_key = (
            path,
            start_line,
            end_line,
            symbol,
            provenance,
        )
        if dedup_key in seen_candidates:
            continue
        seen_candidates.add(dedup_key)

        anchors.append(
            make_anchor(
                canonical_instance_id=canonical_instance_id,
                snapshot_id=snapshot["snapshot_id"],
                repo=snapshot["repo"],
                resolved_commit=snapshot["resolved_commit"],
                file_path=path,
                start_line=start_line,
                end_line=end_line,
                symbol=symbol,
                anchor_type=anchor_type,
                evidence_role=evidence_role,
                provenance=provenance,
                source_detail=location,
                confidence=confidence,
            )
        )

    return anchors


def load_snapshot_index(
    snapshot_path: Path,
    project_root: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    """
    返回：
    1. canonical_instance_id -> snapshot
    2. snapshot_id -> snapshot
    """
    by_instance: dict[str, dict[str, Any]] = {}
    by_snapshot: dict[str, dict[str, Any]] = {}

    for snapshot in read_jsonl(snapshot_path):
        if snapshot.get("status") != "passed":
            continue

        snapshot_id = str(snapshot.get("snapshot_id", "")).strip()
        git_dir_value = str(snapshot.get("git_dir", "")).strip()
        resolved_commit = str(
            snapshot.get("resolved_commit", "")
        ).strip()

        if not snapshot_id or not git_dir_value or not resolved_commit:
            continue

        git_dir = Path(git_dir_value)
        if not git_dir.is_absolute():
            git_dir = project_root / git_dir

        normalized_snapshot = {
            **snapshot,
            "snapshot_id": snapshot_id,
            "git_dir_absolute": str(git_dir.resolve()),
        }
        by_snapshot[snapshot_id] = normalized_snapshot

        for instance_id in snapshot.get(
            "canonical_instance_ids",
            [],
        ):
            instance_text = str(instance_id).strip()
            if instance_text:
                by_instance[instance_text] = normalized_snapshot

    return by_instance, by_snapshot


def validate_paths_for_snapshot(
    project_root: Path,
    snapshot: dict[str, Any],
    paths: list[str],
    *,
    batch_size: int,
) -> tuple[set[str], str | None]:
    """
    使用 git ls-tree 验证路径存在。

    ls-tree 只读取 commit/tree 对象，不读取文件 blob；
    因此不会因为验证路径而下载源码内容。
    """
    existing: set[str] = set()
    git_dir = snapshot["git_dir_absolute"]
    commit = snapshot["resolved_commit"]

    for start in range(0, len(paths), batch_size):
        batch = paths[start:start + batch_size]
        result = run_git(
            project_root,
            [
                "--git-dir",
                git_dir,
                "ls-tree",
                "-r",
                "--name-only",
                "-z",
                commit,
                "--",
                *batch,
            ],
            timeout=180,
        )

        if result.returncode != 0:
            return existing, (
                result.stderr.strip()
                or result.stdout.strip()
                or "git ls-tree 失败"
            )

        for path in result.stdout.split("\0"):
            normalized = normalize_path(path)
            if normalized:
                existing.add(normalized)

    return existing, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "从 patch、external_context 和 trajectory 中抽取轻量证据锚点。"
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="项目根目录，默认当前目录。",
    )

    parser.add_argument(
        "--split",
        action="append",
        choices=sorted(VALID_SPLITS),
        dest="splits",
        help="只处理指定 split，可重复使用；默认处理全部已有快照。",
    )

    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="最多处理多少个实例，用于小规模验证。",
    )

    parser.add_argument(
        "--path-batch-size",
        type=int,
        default=100,
        help="每次 git ls-tree 校验的路径数量，默认 100。",
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

    master_path = (
        project_root
        / "data"
        / "registry"
        / "master_instances.jsonl"
    )
    snapshot_path = (
        project_root
        / "data"
        / "registry"
        / "git_snapshots.jsonl"
    )
    output_root = (
        project_root
        / "data"
        / "processed"
        / "evidence_anchors"
    )

    anchors_path = output_root / "evidence_anchors.jsonl"
    candidates_path = output_root / "candidate_files.jsonl"
    failures_path = output_root / "anchor_failures.jsonl"
    report_path = output_root / "anchor_report.json"

    output_paths = (
        anchors_path,
        candidates_path,
        failures_path,
        report_path,
    )

    if not master_path.exists():
        print(
            f"[错误] 主实例文件不存在：{master_path}",
            file=sys.stderr,
        )
        return 2

    if not snapshot_path.exists():
        print(
            f"[错误] Git 快照索引不存在：{snapshot_path}",
            file=sys.stderr,
        )
        return 2

    existing_outputs = [
        path for path in output_paths if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        print(
            "[错误] 输出已存在，重建时添加 --overwrite：\n"
            + "\n".join(str(path) for path in existing_outputs),
            file=sys.stderr,
        )
        return 2

    if args.path_batch_size <= 0:
        print(
            "[错误] --path-batch-size 必须大于 0。",
            file=sys.stderr,
        )
        return 2

    masters = read_jsonl(master_path)
    snapshot_by_instance, snapshot_by_id = load_snapshot_index(
        snapshot_path,
        project_root,
    )

    selected_splits = set(args.splits or [])

    selected_masters: list[dict[str, Any]] = []
    for master in masters:
        instance_id = str(
            master.get("canonical_instance_id", "")
        ).strip()
        snapshot = snapshot_by_instance.get(instance_id)
        if snapshot is None:
            continue

        if selected_splits:
            snapshot_splits = set(
                str(value)
                for value in snapshot.get("split_names", [])
            )
            if not snapshot_splits.intersection(selected_splits):
                continue

        selected_masters.append(master)

    selected_masters.sort(
        key=lambda item: str(
            item.get("canonical_instance_id", "")
        )
    )

    if args.max_instances is not None:
        if args.max_instances <= 0:
            print(
                "[错误] --max-instances 必须大于 0。",
                file=sys.stderr,
            )
            return 2
        selected_masters = selected_masters[:args.max_instances]

    if not selected_masters:
        print("[错误] 没有可处理的实例。", file=sys.stderr)
        return 2

    raw_anchors: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, master in enumerate(selected_masters, start=1):
        instance_id = str(master["canonical_instance_id"])
        snapshot = snapshot_by_instance[instance_id]

        print(
            f"[{index}/{len(selected_masters)}] "
            f"{instance_id} @ {snapshot['repo']}"
        )

        # Gold patch：定位疑似修复位置，属于离线标签。
        raw_anchors.extend(
            parse_unified_diff(
                master.get("patch"),
                canonical_instance_id=instance_id,
                snapshot=snapshot,
                provenance="gold_patch",
                evidence_role="fault_location",
            )
        )

        # Test patch：提供验证约束位置，同样属于离线标签。
        raw_anchors.extend(
            parse_unified_diff(
                master.get("test_patch"),
                canonical_instance_id=instance_id,
                snapshot=snapshot,
                provenance="test_patch",
                evidence_role="validation_constraint",
            )
        )

        for source_record in master.get("source_records", []):
            source_name = str(
                source_record.get("source_name", "unknown")
            )

            external_context = source_record.get(
                "external_context",
                {},
            )
            raw_anchors.extend(
                extract_payload_anchors(
                    external_context,
                    canonical_instance_id=instance_id,
                    snapshot=snapshot,
                    provenance=f"{source_name}:external_context",
                    evidence_role="external_context",
                    confidence=0.95,
                )
            )

            trajectory = source_record.get("trajectory", {})
            raw_anchors.extend(
                extract_payload_anchors(
                    trajectory,
                    canonical_instance_id=instance_id,
                    snapshot=snapshot,
                    provenance=f"{source_name}:trajectory",
                    evidence_role="exploration_trace",
                    confidence=0.80,
                )
            )

    # anchor_id 去重。
    anchors_by_id: dict[str, dict[str, Any]] = {}
    for anchor in raw_anchors:
        anchors_by_id.setdefault(anchor["anchor_id"], anchor)

    anchors = list(anchors_by_id.values())

    # 按 snapshot 批量校验候选路径。
    paths_by_snapshot: dict[str, set[str]] = defaultdict(set)
    for anchor in anchors:
        paths_by_snapshot[anchor["snapshot_id"]].add(
            anchor["file_path"]
        )

    existing_paths_by_snapshot: dict[str, set[str]] = {}

    for snapshot_id, paths in sorted(paths_by_snapshot.items()):
        snapshot = snapshot_by_id.get(snapshot_id)
        if snapshot is None:
            failures.append(
                {
                    "snapshot_id": snapshot_id,
                    "reason": "snapshot_not_found",
                }
            )
            continue

        existing_paths, error = validate_paths_for_snapshot(
            project_root,
            snapshot,
            sorted(paths),
            batch_size=args.path_batch_size,
        )
        existing_paths_by_snapshot[snapshot_id] = existing_paths

        if error:
            failures.append(
                {
                    "snapshot_id": snapshot_id,
                    "repo": snapshot.get("repo"),
                    "resolved_commit": snapshot.get(
                        "resolved_commit"
                    ),
                    "reason": "git_tree_validation_failed",
                    "error": error,
                }
            )

    valid_anchors: list[dict[str, Any]] = []
    missing_path_count = 0

    for anchor in anchors:
        existing_paths = existing_paths_by_snapshot.get(
            anchor["snapshot_id"],
            set(),
        )
        exists = anchor["file_path"] in existing_paths
        anchor["file_exists_at_base_commit"] = exists

        if exists:
            valid_anchors.append(anchor)
        else:
            missing_path_count += 1
            failures.append(
                {
                    "canonical_instance_id": anchor[
                        "canonical_instance_id"
                    ],
                    "snapshot_id": anchor["snapshot_id"],
                    "repo": anchor["repo"],
                    "resolved_commit": anchor[
                        "resolved_commit"
                    ],
                    "file_path": anchor["file_path"],
                    "provenance": anchor["provenance"],
                    "reason": "file_not_found_at_base_commit",
                }
            )

    valid_anchors.sort(
        key=lambda item: (
            item["canonical_instance_id"],
            item["file_path"],
            item["start_line"] or 0,
            item["end_line"] or 0,
            item["provenance"],
        )
    )

    # 为下一阶段生成按实例、文件聚合的候选文件清单。
    grouped_candidates: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}

    for anchor in valid_anchors:
        key = (
            anchor["canonical_instance_id"],
            anchor["snapshot_id"],
            anchor["file_path"],
        )

        if key not in grouped_candidates:
            grouped_candidates[key] = {
                "candidate_file_id": (
                    "candidate-file-"
                    + sha256_text("|".join(key))[:24]
                ),
                "canonical_instance_id": anchor[
                    "canonical_instance_id"
                ],
                "snapshot_id": anchor["snapshot_id"],
                "repo": anchor["repo"],
                "resolved_commit": anchor[
                    "resolved_commit"
                ],
                "file_path": anchor["file_path"],
                "anchor_ids": [],
                "provenances": set(),
                "evidence_roles": set(),
                "line_ranges": set(),
                "symbols": set(),
                "priority_score": 0.0,
                "visibility": "label_only",
            }

        candidate = grouped_candidates[key]
        candidate["anchor_ids"].append(anchor["anchor_id"])
        candidate["provenances"].add(anchor["provenance"])
        candidate["evidence_roles"].add(
            anchor["evidence_role"]
        )

        if anchor["start_line"] is not None:
            candidate["line_ranges"].add(
                (
                    anchor["start_line"],
                    anchor["end_line"],
                )
            )

        if anchor["symbol"]:
            candidate["symbols"].add(anchor["symbol"])

        # 简单确定性优先级，只用于后续解析顺序。
        candidate["priority_score"] += {
            "gold_patch": 4.0,
            "test_patch": 3.0,
        }.get(
            anchor["provenance"],
            2.0
            if anchor["provenance"].endswith(
                ":external_context"
            )
            else 1.0,
        )

    candidate_files: list[dict[str, Any]] = []

    for candidate in grouped_candidates.values():
        candidate["anchor_ids"] = sorted(
            set(candidate["anchor_ids"])
        )
        candidate["provenances"] = sorted(
            candidate["provenances"]
        )
        candidate["evidence_roles"] = sorted(
            candidate["evidence_roles"]
        )
        candidate["line_ranges"] = [
            {
                "start_line": start_line,
                "end_line": end_line,
            }
            for start_line, end_line in sorted(
                candidate["line_ranges"]
            )
        ]
        candidate["symbols"] = sorted(
            candidate["symbols"]
        )
        candidate["priority_score"] = round(
            candidate["priority_score"],
            3,
        )
        candidate_files.append(candidate)

    candidate_files.sort(
        key=lambda item: (
            item["canonical_instance_id"],
            -item["priority_score"],
            item["file_path"],
        )
    )

    write_jsonl(anchors_path, valid_anchors)
    write_jsonl(candidates_path, candidate_files)
    write_jsonl(failures_path, failures)

    provenance_counts = Counter(
        anchor["provenance"]
        for anchor in valid_anchors
    )
    role_counts = Counter(
        anchor["evidence_role"]
        for anchor in valid_anchors
    )
    type_counts = Counter(
        anchor["anchor_type"]
        for anchor in valid_anchors
    )

    instances_with_candidates = {
        candidate["canonical_instance_id"]
        for candidate in candidate_files
    }

    # 路径验证失败（file_not_found_at_base_commit / git_tree_validation_failed）
    # 属于预期行为：文件重命名、新增文件、外部路径格式差异（如 .po 引用）
    # 都会触发。只要产出了有效锚点，整体状态即为 passed。
    status = (
        "passed"
        if valid_anchors
        else "failed"
    )

    report = {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now(),
        "status": status,
        "processed_instance_count": len(selected_masters),
        "instance_with_candidate_count": len(
            instances_with_candidates
        ),
        "raw_anchor_count": len(raw_anchors),
        "deduplicated_anchor_count": len(anchors),
        "valid_anchor_count": len(valid_anchors),
        "missing_path_anchor_count": missing_path_count,
        "candidate_file_count": len(candidate_files),
        "failure_count": len(failures),
        "provenance_counts": dict(
            sorted(provenance_counts.items())
        ),
        "evidence_role_counts": dict(
            sorted(role_counts.items())
        ),
        "anchor_type_counts": dict(
            sorted(type_counts.items())
        ),
        "network_behavior": (
            "本脚本只执行 git ls-tree；"
            "不会读取候选文件 blob，也不会主动下载源码内容。"
        ),
        "label_visibility": (
            "所有输出均为 label_only；"
            "不得直接进入在线 evidence agent 输入。"
        ),
        "outputs": {
            "evidence_anchors": anchors_path.relative_to(
                project_root
            ).as_posix(),
            "candidate_files": candidates_path.relative_to(
                project_root
            ).as_posix(),
            "anchor_failures": failures_path.relative_to(
                project_root
            ).as_posix(),
        },
    }

    write_json(report_path, report)

    print(
        json.dumps(
            {
                "status": status,
                "processed_instance_count": len(
                    selected_masters
                ),
                "instance_with_candidate_count": len(
                    instances_with_candidates
                ),
                "valid_anchor_count": len(valid_anchors),
                "candidate_file_count": len(candidate_files),
                "missing_path_anchor_count": missing_path_count,
                "failure_count": len(failures),
                "output_directory": str(output_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0 if status == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
