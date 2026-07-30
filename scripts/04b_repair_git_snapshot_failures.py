#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
只修复 Git 快照阶段的失败项。

适用场景：
- git_snapshots.jsonl 已经包含绝大多数成功快照；
- git_snapshot_failures.jsonl 中仍有少量仓库名错误或提交归属错误；
- 不希望重新处理全部 19311 个 repo + commit。

本脚本不读取配置文件，不修改 split，不创建 worktree。

仓库恢复依据：
1. source_records[*].source_instance_id
   例如 fasterxml__jackson-databind-4320
   可恢复为 fasterxml/jackson-databind；
2. issue_url / pr_url；
3. 来源元数据中的 repo / repository 字段；
4. 原始 repo 作为最后回退。

输入：
    data/registry/git_snapshots.jsonl
    data/registry/git_snapshot_failures.jsonl
    data/splits/*.jsonl

输出：
    更新 data/registry/git_snapshots.jsonl
    更新 data/registry/git_snapshot_failures.jsonl
    新增 data/registry/git_snapshot_repair_report.json

运行：
    python scripts/04b_repair_git_snapshot_failures.py --project-root .

只预览候选仓库，不执行 clone/fetch：
    python scripts/04b_repair_git_snapshot_failures.py --project-root . --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

SCRIPT_VERSION = "1.0.0"

VALID_SPLITS = (
    "train",
    "dev",
    "test_retrieval",
    "test_sufficiency",
)


def utc_now() -> str:
    """返回当前 UTC 时间。"""
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    """计算文本 SHA-256。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_repo(value: Any) -> str | None:
    """把 GitHub 仓库统一为 owner/repo。"""
    if value is None:
        return None

    text = str(value).strip().rstrip("/")
    if not text:
        return None

    github_match = re.search(
        r"github\.com[/:]([^/\s]+)/([^/\s?#]+)",
        text,
        flags=re.IGNORECASE,
    )
    if github_match:
        text = f"{github_match.group(1)}/{github_match.group(2)}"

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
    if len(parts) < 2:
        return None

    owner = parts[0].strip()
    repo = parts[1].strip()

    # 排除明显不是仓库名的 URL 后续段。
    if repo.lower() in {"issues", "pull", "pulls", "commit", "tree", "blob"}:
        return None

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
        return None

    return f"{owner}/{repo}"


def normalize_commit(value: Any) -> str | None:
    """校验 Git commit 基本格式。"""
    if value is None:
        return None

    text = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{7,40}", text):
        return None

    return text


def repo_slug(repo: str) -> str:
    """生成 Windows 兼容的稳定缓存目录名。"""
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "__", repo)
    digest = sha256_text(repo.lower())[:10]
    return f"{safe_name}__{digest}"


def run_git(
    arguments: list[str],
    *,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """执行 Git 命令，不使用 shell=True。"""
    environment = os.environ.copy()
    environment.setdefault("GIT_TERMINAL_PROMPT", "0")

    return subprocess.run(
        ["git", *arguments],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
        env=environment,
    )


def require_git() -> str:
    """确认 Git 已安装。"""
    result = run_git(["--version"], timeout=30)
    if result.returncode != 0:
        raise RuntimeError("未找到 git，请先安装 Git 并加入 PATH。")
    return result.stdout.strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件。"""
    records: list[dict[str, Any]] = []

    if not path.exists():
        return records

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


def atomic_write_jsonl(
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    """通过临时文件原子替换 JSONL，降低中断损坏风险。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file_obj:
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

    temporary_path.replace(path)


def write_json(path: Path, value: Any) -> None:
    """写 JSON 报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    temporary_path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def backup_once(path: Path) -> Path | None:
    """首次修复时保留原始清单备份。"""
    if not path.exists():
        return None

    backup_path = path.with_name(
        f"{path.stem}.before_repo_repair{path.suffix}"
    )
    if not backup_path.exists():
        shutil.copy2(path, backup_path)

    return backup_path


def repo_from_source_instance_id(value: Any) -> str | None:
    """
    从 SWE-bench 风格实例 ID 中恢复仓库。

    示例：
        fasterxml__jackson-databind-4320
        mui__material-ui-12345
        zeromicro__go-zero-1000
    """
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.lower().startswith("generated-"):
        return None

    match = re.match(
        r"^(.+?)__(.+?)-\d+(?:[-_].*)?$",
        text,
    )
    if not match:
        return None

    owner, repo = match.groups()
    return normalize_repo(f"{owner}/{repo}")


def repo_from_github_url(value: Any) -> str | None:
    """从 GitHub Issue、PR 或普通仓库 URL 恢复 owner/repo。"""
    if value is None:
        return None

    text = str(value).strip()
    match = re.search(
        r"github\.com/([^/\s]+)/([^/\s?#]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    return normalize_repo(
        f"{match.group(1)}/{match.group(2)}"
    )


def walk_repo_metadata(
    value: Any,
    *,
    parent_key: str = "",
    depth: int = 0,
) -> Iterator[str]:
    """
    从来源元数据中保守提取仓库候选。

    只读取：
    - 字段名含 repo/repository 的字符串；
    - 含 github.com 的 URL；
    - 最多递归三层。
    """
    if depth > 3:
        return

    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = str(key).lower()
            yield from walk_repo_metadata(
                child,
                parent_key=normalized_key,
                depth=depth + 1,
            )
        return

    if isinstance(value, list):
        for child in value[:100]:
            yield from walk_repo_metadata(
                child,
                parent_key=parent_key,
                depth=depth + 1,
            )
        return

    if not isinstance(value, str):
        return

    if "github.com" in value.lower():
        repo = repo_from_github_url(value)
        if repo:
            yield repo
        return

    if "repo" in parent_key or "repository" in parent_key:
        repo = normalize_repo(value)
        if repo:
            yield repo


def build_split_instance_index(
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    """按 canonical_instance_id 索引四个 split 中的主实例记录。"""
    index: dict[str, dict[str, Any]] = {}

    for split_name in VALID_SPLITS:
        split_path = (
            project_root
            / "data"
            / "splits"
            / f"{split_name}.jsonl"
        )

        if not split_path.exists():
            continue

        for record in read_jsonl(split_path):
            instance_id = str(
                record.get("canonical_instance_id", "")
            ).strip()
            if not instance_id:
                continue

            # 理论上同一实例只出现在一个 split。
            record_copy = dict(record)
            record_copy["_loaded_split"] = split_name
            index[instance_id] = record_copy

    return index


def collect_candidate_repos(
    failure: dict[str, Any],
    instance_index: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    """
    为一个失败快照收集候选仓库，并记录推断来源。

    返回顺序即尝试顺序：
    1. source_instance_id；
    2. Issue/PR URL；
    3. 来源元数据；
    4. 原始 repo。
    """
    candidates: dict[str, dict[str, str]] = {}

    def add(repo: str | None, method: str) -> None:
        if not repo:
            return

        normalized = normalize_repo(repo)
        if not normalized:
            return

        key = normalized.lower()
        if key not in candidates:
            candidates[key] = {
                "repo": normalized,
                "method": method,
            }

    instance_ids = failure.get("canonical_instance_ids")
    if not isinstance(instance_ids, list):
        single_id = failure.get("canonical_instance_id")
        instance_ids = [single_id] if single_id else []

    related_records = [
        instance_index[instance_id]
        for instance_id in instance_ids
        if instance_id in instance_index
    ]

    # 1. 来源实例 ID。
    for record in related_records:
        for source_record in record.get("source_records", []):
            if not isinstance(source_record, dict):
                continue

            add(
                repo_from_source_instance_id(
                    source_record.get("source_instance_id")
                ),
                "source_instance_id",
            )

    # 2. 主记录和来源元数据中的 URL。
    for record in related_records:
        add(
            repo_from_github_url(record.get("issue_url")),
            "issue_url",
        )
        add(
            repo_from_github_url(record.get("pr_url")),
            "pr_url",
        )

        for source_record in record.get("source_records", []):
            if not isinstance(source_record, dict):
                continue

            for key in (
                "issue_url",
                "pr_url",
                "github_issue_url",
                "github_pr_url",
            ):
                add(
                    repo_from_github_url(source_record.get(key)),
                    key,
                )

    # 3. 来源元数据。
    for record in related_records:
        for source_record in record.get("source_records", []):
            if not isinstance(source_record, dict):
                continue

            for repo in walk_repo_metadata(
                source_record.get("source_metadata", {})
            ):
                add(repo, "source_metadata")

    # 4. 主实例 repo 和失败记录 repo 放到最后。
    for record in related_records:
        add(record.get("repo"), "master_repo")

    add(
        failure.get("repo")
        or failure.get("requested_repo"),
        "failed_repo",
    )

    return list(candidates.values())


def is_valid_bare_repo(git_dir: Path) -> bool:
    """判断目录是否为可用 bare 仓库。"""
    return (
        git_dir.is_dir()
        and (git_dir / "HEAD").is_file()
        and (git_dir / "objects").is_dir()
    )


def ensure_repo_cache(
    repo: str,
    git_dir: Path,
    *,
    no_fetch: bool,
    timeout: int,
) -> str:
    """
    确保候选仓库缓存存在。

    每个候选仓库在本次修复中只调用一次。
    默认使用 blobless partial bare clone。
    """
    remote_url = f"https://github.com/{repo}.git"

    if not git_dir.exists():
        git_dir.parent.mkdir(parents=True, exist_ok=True)

        result = run_git(
            [
                "clone",
                "--bare",
                "--filter=blob:none",
                remote_url,
                str(git_dir),
            ],
            timeout=timeout,
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip()
                or result.stdout.strip()
                or "git clone --bare 失败"
            )

        return "cloned_partial"

    if not is_valid_bare_repo(git_dir):
        raise RuntimeError(
            f"缓存目录存在但不是有效 bare 仓库：{git_dir}"
        )

    if no_fetch:
        return "reused_without_fetch"

    set_url = run_git(
        [
            "--git-dir",
            str(git_dir),
            "remote",
            "set-url",
            "origin",
            remote_url,
        ],
        timeout=60,
    )
    if set_url.returncode != 0:
        raise RuntimeError(
            set_url.stderr.strip()
            or set_url.stdout.strip()
            or "更新 origin URL 失败"
        )

    fetch_result = run_git(
        [
            "--git-dir",
            str(git_dir),
            "fetch",
            "--prune",
            "--tags",
            "--filter=blob:none",
            "origin",
            "+refs/heads/*:refs/heads/*",
        ],
        timeout=timeout,
    )
    if fetch_result.returncode != 0:
        raise RuntimeError(
            fetch_result.stderr.strip()
            or fetch_result.stdout.strip()
            or "git fetch 失败"
        )

    return "fetched_once"


def resolve_commit(
    git_dir: Path,
    requested_commit: str,
) -> str | None:
    """在候选仓库中解析完整提交 SHA。"""
    result = run_git(
        [
            "--git-dir",
            str(git_dir),
            "rev-parse",
            "--verify",
            f"{requested_commit}^{{commit}}",
        ],
        timeout=60,
    )

    if result.returncode != 0:
        return None

    resolved = result.stdout.strip().splitlines()[-1].lower()
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        return None

    return resolved


def fetch_commit(
    git_dir: Path,
    requested_commit: str,
    *,
    timeout: int,
) -> None:
    """仅在本地无法解析时尝试显式获取一次目标提交。"""
    run_git(
        [
            "--git-dir",
            str(git_dir),
            "fetch",
            "--filter=blob:none",
            "origin",
            requested_commit,
        ],
        timeout=timeout,
    )


def failure_identity(
    record: dict[str, Any],
) -> tuple[str, str, tuple[str, ...]]:
    """生成失败记录稳定身份，用于删除已修复项。"""
    repo = normalize_repo(
        record.get("repo")
        or record.get("requested_repo")
    ) or ""
    commit = normalize_commit(
        record.get("requested_base_commit")
        or record.get("base_commit")
    ) or ""

    instance_ids = record.get("canonical_instance_ids")
    if not isinstance(instance_ids, list):
        single_id = record.get("canonical_instance_id")
        instance_ids = [single_id] if single_id else []

    return (
        repo.lower(),
        commit,
        tuple(sorted(str(value) for value in instance_ids if value)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "自动恢复失败快照的真实仓库，"
            "只重试失败项。"
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="项目根目录，默认当前目录。",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只输出候选仓库，不执行 clone/fetch，不修改清单。",
    )

    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="只使用现有 Git 缓存。",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="单个 clone/fetch 命令超时秒数，默认 3600。",
    )

    parser.add_argument(
        "--max-failures",
        type=int,
        default=None,
        help="最多处理多少条失败记录，用于小规模验证。",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    registry_root = project_root / "data" / "registry"

    manifest_path = registry_root / "git_snapshots.jsonl"
    failure_path = registry_root / "git_snapshot_failures.jsonl"
    report_path = registry_root / "git_snapshot_repair_report.json"

    if not manifest_path.exists():
        print(
            f"[错误] 成功快照清单不存在：{manifest_path}",
            file=sys.stderr,
        )
        return 2

    if not failure_path.exists():
        print(
            f"[错误] 失败清单不存在：{failure_path}",
            file=sys.stderr,
        )
        return 2

    try:
        git_version = require_git()
        success_records = read_jsonl(manifest_path)
        failure_records = read_jsonl(failure_path)
        instance_index = build_split_instance_index(project_root)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[错误] {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    retryable_failures = [
        record
        for record in failure_records
        if normalize_commit(
            record.get("requested_base_commit")
            or record.get("base_commit")
        )
    ]

    if args.max_failures is not None:
        if args.max_failures <= 0:
            print(
                "[错误] --max-failures 必须大于 0。",
                file=sys.stderr,
            )
            return 2
        retryable_failures = retryable_failures[
            : args.max_failures
        ]

    if not retryable_failures:
        print("[完成] 没有可重试的失败快照。")
        return 0

    repaired_records: list[dict[str, Any]] = []
    unresolved_records: list[dict[str, Any]] = []
    repaired_failure_ids: set[
        tuple[str, str, tuple[str, ...]]
    ] = set()

    repo_cache_state: dict[str, dict[str, Any]] = {}
    repo_action_counts: Counter[str] = Counter()
    resolution_method_counts: Counter[str] = Counter()

    for index, failure in enumerate(
        retryable_failures,
        start=1,
    ):
        original_repo = normalize_repo(
            failure.get("repo")
            or failure.get("requested_repo")
        )
        requested_commit = normalize_commit(
            failure.get("requested_base_commit")
            or failure.get("base_commit")
        )

        if not requested_commit:
            continue

        candidates = collect_candidate_repos(
            failure,
            instance_index,
        )

        print(
            f"[{index}/{len(retryable_failures)}] "
            f"{original_repo or '<unknown>'}@{requested_commit}"
        )
        print(
            "  候选："
            + (
                ", ".join(
                    f"{item['repo']}[{item['method']}]"
                    for item in candidates
                )
                if candidates
                else "<无>"
            )
        )

        if args.dry_run:
            continue

        resolved_result: dict[str, Any] | None = None
        candidate_errors: list[dict[str, str]] = []

        for candidate in candidates:
            candidate_repo = candidate["repo"]
            candidate_key = candidate_repo.lower()
            git_dir = (
                project_root
                / "data"
                / "cache"
                / "repos"
                / f"{repo_slug(candidate_repo)}.git"
            )

            cache_state = repo_cache_state.get(candidate_key)
            if cache_state is None:
                try:
                    cache_action = ensure_repo_cache(
                        candidate_repo,
                        git_dir,
                        no_fetch=args.no_fetch,
                        timeout=args.timeout,
                    )
                    cache_state = {
                        "status": "passed",
                        "git_dir": git_dir,
                        "cache_action": cache_action,
                    }
                    repo_action_counts[cache_action] += 1
                except Exception as exc:  # noqa: BLE001
                    cache_state = {
                        "status": "failed",
                        "git_dir": git_dir,
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                repo_cache_state[candidate_key] = cache_state

            if cache_state["status"] != "passed":
                candidate_errors.append(
                    {
                        "repo": candidate_repo,
                        "error": cache_state["error"],
                    }
                )
                continue

            resolved_commit = resolve_commit(
                cache_state["git_dir"],
                requested_commit,
            )

            if resolved_commit is None and not args.no_fetch:
                fetch_commit(
                    cache_state["git_dir"],
                    requested_commit,
                    timeout=args.timeout,
                )
                resolved_commit = resolve_commit(
                    cache_state["git_dir"],
                    requested_commit,
                )

            if resolved_commit is None:
                candidate_errors.append(
                    {
                        "repo": candidate_repo,
                        "error": "候选仓库中找不到目标 commit",
                    }
                )
                continue

            instance_ids = failure.get(
                "canonical_instance_ids"
            )
            if not isinstance(instance_ids, list):
                single_id = failure.get(
                    "canonical_instance_id"
                )
                instance_ids = (
                    [single_id]
                    if single_id
                    else []
                )

            split_names = failure.get("split_names", [])
            if not isinstance(split_names, list):
                split_names = []

            snapshot_id = (
                "snapshot-"
                + sha256_text(
                    f"{candidate_repo.lower()}@{resolved_commit}"
                )[:24]
            )

            resolved_result = {
                "snapshot_id": snapshot_id,
                "repo": candidate_repo,
                "requested_repo": original_repo,
                "requested_base_commit": requested_commit,
                "resolved_commit": resolved_commit,
                "git_dir": cache_state[
                    "git_dir"
                ].relative_to(
                    project_root
                ).as_posix(),
                "split_names": sorted(
                    str(value)
                    for value in split_names
                ),
                "canonical_instance_count": len(
                    set(
                        str(value)
                        for value in instance_ids
                        if value
                    )
                ),
                "canonical_instance_ids": sorted(
                    set(
                        str(value)
                        for value in instance_ids
                        if value
                    )
                ),
                "status": "passed",
                "storage_mode": "partial_bare_blobless",
                "cache_action": cache_state["cache_action"],
                "repo_resolution_method": candidate["method"],
                "repo_candidates": candidates,
                "script_version": SCRIPT_VERSION,
            }
            break

        if resolved_result is None:
            unresolved_copy = dict(failure)
            unresolved_copy[
                "repair_candidates"
            ] = candidates
            unresolved_copy[
                "repair_candidate_errors"
            ] = candidate_errors
            unresolved_copy[
                "repair_status"
            ] = "unresolved"
            unresolved_records.append(unresolved_copy)
            print("  [未修复] 所有候选仓库均未包含该提交")
            continue

        repaired_records.append(resolved_result)
        repaired_failure_ids.add(failure_identity(failure))
        resolution_method_counts[
            resolved_result["repo_resolution_method"]
        ] += 1

        print(
            f"  [修复] {original_repo} -> "
            f"{resolved_result['repo']}"
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "retryable_failure_count": len(
                        retryable_failures
                    ),
                    "note": "未修改任何文件。",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    # 保留未在本轮处理范围内的失败项，并删除本轮已修复项。
    remaining_failures: list[dict[str, Any]] = []
    selected_identities = {
        failure_identity(record)
        for record in retryable_failures
    }
    unresolved_by_identity = {
        failure_identity(record): record
        for record in unresolved_records
    }

    for record in failure_records:
        identity = failure_identity(record)

        if identity in repaired_failure_ids:
            continue

        if identity in selected_identities:
            remaining_failures.append(
                unresolved_by_identity.get(
                    identity,
                    record,
                )
            )
        else:
            remaining_failures.append(record)

    # 通过 canonical_instance_ids + resolved_commit 去重，防止重复追加。
    merged_successes: dict[
        tuple[str, str, tuple[str, ...]],
        dict[str, Any],
    ] = {}

    for record in [*success_records, *repaired_records]:
        resolved_commit = str(
            record.get("resolved_commit", "")
        ).lower()
        repo = normalize_repo(record.get("repo")) or ""

        instance_ids = record.get(
            "canonical_instance_ids",
            [],
        )
        if not isinstance(instance_ids, list):
            instance_ids = []

        key = (
            repo.lower(),
            resolved_commit,
            tuple(sorted(str(value) for value in instance_ids)),
        )
        merged_successes[key] = record

    final_success_records = sorted(
        merged_successes.values(),
        key=lambda record: (
            str(record.get("repo", "")).lower(),
            str(record.get("resolved_commit", "")),
            str(record.get("snapshot_id", "")),
        ),
    )

    remaining_failures.sort(
        key=lambda record: (
            str(record.get("repo", "")).lower(),
            str(
                record.get("requested_base_commit", "")
            ),
        )
    )

    backup_manifest = backup_once(manifest_path)
    backup_failures = backup_once(failure_path)

    atomic_write_jsonl(
        manifest_path,
        final_success_records,
    )
    atomic_write_jsonl(
        failure_path,
        remaining_failures,
    )

    report = {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now(),
        "status": (
            "passed"
            if not unresolved_records
            else "partial"
        ),
        "git_version": git_version,
        "input_failure_count": len(failure_records),
        "retryable_failure_count": len(
            retryable_failures
        ),
        "repaired_count": len(repaired_records),
        "unresolved_retry_count": len(
            unresolved_records
        ),
        "remaining_failure_count": len(
            remaining_failures
        ),
        "final_success_count": len(
            final_success_records
        ),
        "resolution_method_counts": dict(
            resolution_method_counts
        ),
        "repo_action_counts": dict(
            repo_action_counts
        ),
        "backup_manifest": (
            backup_manifest.relative_to(
                project_root
            ).as_posix()
            if backup_manifest
            else None
        ),
        "backup_failures": (
            backup_failures.relative_to(
                project_root
            ).as_posix()
            if backup_failures
            else None
        ),
    }
    write_json(report_path, report)

    print(
        json.dumps(
            {
                "status": report["status"],
                "repaired_count": len(repaired_records),
                "unresolved_retry_count": len(
                    unresolved_records
                ),
                "remaining_failure_count": len(
                    remaining_failures
                ),
                "final_success_count": len(
                    final_success_records
                ),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0 if not unresolved_records else 3


if __name__ == "__main__":
    raise SystemExit(main())
