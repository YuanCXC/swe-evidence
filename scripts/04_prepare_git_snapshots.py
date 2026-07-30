#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
轻量 Git 快照准备（按仓库批处理版）。

修复点：
1. 不再对每个 repo + commit 重复执行 git fetch；
2. 每个仓库只 clone/fetch 一次；
3. 默认使用 partial bare clone：--filter=blob:none；
4. clone/fetch 后，在本地批量验证该仓库下的全部 base_commit；
5. 不创建 worktree，不展开源码目录；
6. 支持断点续跑：已有成功快照默认跳过。

输入：
    data/splits/train.jsonl
    data/splits/dev.jsonl
    data/splits/test_retrieval.jsonl
    data/splits/test_sufficiency.jsonl

输出：
    data/cache/repos/<repo_slug>.git/
    data/registry/git_snapshots.jsonl
    data/registry/git_snapshot_failures.jsonl
    data/registry/git_snapshot_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_VERSION = "1.1.0"

VALID_SPLITS = (
    "train",
    "dev",
    "test_retrieval",
    "test_sufficiency",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_repo(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().rstrip("/")
    if not text:
        return None

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

    return f"{parts[0]}/{parts[1]}"


def normalize_commit(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{7,40}", text):
        return None

    return text


def repo_slug(repo: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "__", repo)
    digest = sha256_text(repo.lower())[:10]
    return f"{safe_name}__{digest}"


def run_git(
    arguments: list[str],
    *,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
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
    result = run_git(["--version"], timeout=30)
    if result.returncode != 0:
        raise RuntimeError("未找到 git，请先安装 Git 并加入 PATH。")
    return result.stdout.strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if not path.exists():
        return records

    with path.open("r", encoding="utf-8-sig") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} 第 {line_number} 行 JSON 无效：{exc}"
                ) from exc

            if not isinstance(payload, dict):
                raise ValueError(
                    f"{path} 第 {line_number} 行不是 JSON object"
                )

            records.append(payload)

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


def load_repo_tasks(
    project_root: Path,
    selected_splits: list[str],
) -> tuple[dict[str, dict[str, dict[str, set[str]]]], list[dict[str, Any]]]:
    """
    返回：
        repo_tasks[repo][commit] = {
            "split_names": set(...),
            "canonical_instance_ids": set(...)
        }
    """
    repo_tasks: dict[
        str,
        dict[str, dict[str, set[str]]],
    ] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "split_names": set(),
                "canonical_instance_ids": set(),
            }
        )
    )
    invalid_inputs: list[dict[str, Any]] = []

    for split_name in selected_splits:
        split_path = (
            project_root
            / "data"
            / "splits"
            / f"{split_name}.jsonl"
        )

        if not split_path.exists():
            raise FileNotFoundError(f"划分文件不存在：{split_path}")

        for record in read_jsonl(split_path):
            repo = normalize_repo(record.get("repo"))
            commit = normalize_commit(record.get("base_commit"))
            instance_id = str(
                record.get("canonical_instance_id", "")
            ).strip()

            reasons: list[str] = []
            if not repo:
                reasons.append("invalid_repo")
            if not commit:
                reasons.append("invalid_base_commit")
            if not instance_id:
                reasons.append("missing_canonical_instance_id")

            if reasons:
                invalid_inputs.append(
                    {
                        "split": split_name,
                        "canonical_instance_id": instance_id or None,
                        "repo": record.get("repo"),
                        "base_commit": record.get("base_commit"),
                        "status": "invalid_input",
                        "reasons": reasons,
                    }
                )
                continue

            repo_tasks[repo][commit]["split_names"].add(split_name)
            repo_tasks[repo][commit][
                "canonical_instance_ids"
            ].add(instance_id)

    return {
        repo: dict(commits)
        for repo, commits in repo_tasks.items()
    }, invalid_inputs


def is_valid_bare_repo(git_dir: Path) -> bool:
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
    full_clone: bool,
    timeout: int,
) -> str:
    """
    每个仓库只调用一次。

    默认 partial bare clone，仅下载提交和目录树；
    读取具体文件时 Git 会按需获取 blob。
    """
    remote_url = f"https://github.com/{repo}.git"

    if not git_dir.exists():
        git_dir.parent.mkdir(parents=True, exist_ok=True)

        clone_args = ["clone", "--bare"]
        if not full_clone:
            clone_args.extend(["--filter=blob:none"])
        clone_args.extend([remote_url, str(git_dir)])

        result = run_git(clone_args, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip()
                or result.stdout.strip()
                or "git clone --bare 失败"
            )

        return "cloned_full" if full_clone else "cloned_partial"

    if not is_valid_bare_repo(git_dir):
        raise RuntimeError(f"缓存目录不是有效 bare 仓库：{git_dir}")

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

    if no_fetch:
        return "reused_without_fetch"

    fetch_args = [
        "--git-dir",
        str(git_dir),
        "fetch",
        "--prune",
        "--tags",
    ]
    if not full_clone:
        fetch_args.append("--filter=blob:none")
    fetch_args.extend(
        [
            "origin",
            "+refs/heads/*:refs/heads/*",
        ]
    )

    result = run_git(fetch_args, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or result.stdout.strip()
            or "git fetch 失败"
        )

    return "fetched_once"


def resolve_commit(
    git_dir: Path,
    requested_commit: str,
) -> str | None:
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


def fetch_missing_commits_once(
    git_dir: Path,
    commits: list[str],
    *,
    full_clone: bool,
    timeout: int,
) -> subprocess.CompletedProcess[str] | None:
    """
    尝试一次性获取该仓库中缺失的提交。

    不再为每个 commit 单独 fetch。
    """
    if not commits:
        return None

    arguments = [
        "--git-dir",
        str(git_dir),
        "fetch",
    ]
    if not full_clone:
        arguments.append("--filter=blob:none")
    arguments.extend(["origin", *commits])

    return run_git(arguments, timeout=timeout)


def load_existing_successes(
    manifest_path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    existing: dict[tuple[str, str], dict[str, Any]] = {}

    for record in read_jsonl(manifest_path):
        repo = normalize_repo(record.get("repo"))
        requested_commit = normalize_commit(
            record.get("requested_base_commit")
        )
        if (
            repo
            and requested_commit
            and record.get("status") == "passed"
        ):
            existing[(repo, requested_commit)] = record

    return existing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按仓库批量准备 Git 快照；"
            "每个仓库只 clone/fetch 一次。"
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
    )

    parser.add_argument(
        "--split",
        action="append",
        choices=VALID_SPLITS,
        dest="splits",
        help="只处理指定 split，可重复使用；默认全部。",
    )

    parser.add_argument(
        "--max-repos",
        type=int,
        default=None,
        help="最多处理多少个仓库，用于小规模验证。",
    )

    parser.add_argument(
        "--max-commits-per-repo",
        type=int,
        default=None,
        help="每个仓库最多验证多少个 commit，用于小规模验证。",
    )

    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="只使用已有缓存。",
    )

    parser.add_argument(
        "--full-clone",
        action="store_true",
        help="禁用 blob:none，下载完整 Git 对象；默认使用 partial clone。",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help="忽略已有成功快照，从头生成清单。",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    selected_splits = args.splits or list(VALID_SPLITS)

    registry_root = project_root / "data" / "registry"
    manifest_path = registry_root / "git_snapshots.jsonl"
    failure_path = registry_root / "git_snapshot_failures.jsonl"
    report_path = registry_root / "git_snapshot_report.json"

    try:
        git_version = require_git()
        repo_tasks, invalid_inputs = load_repo_tasks(
            project_root,
            selected_splits,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[错误] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    existing_successes = (
        {}
        if args.restart
        else load_existing_successes(manifest_path)
    )

    ordered_repos = sorted(repo_tasks, key=str.lower)
    if args.max_repos is not None:
        if args.max_repos <= 0:
            print("[错误] --max-repos 必须大于 0。", file=sys.stderr)
            return 2
        ordered_repos = ordered_repos[: args.max_repos]

    registry_root.mkdir(parents=True, exist_ok=True)

    successful_records: dict[
        tuple[str, str],
        dict[str, Any],
    ] = dict(existing_successes)
    failure_records: list[dict[str, Any]] = list(invalid_inputs)

    repo_action_counts: Counter[str] = Counter()
    newly_verified = 0
    skipped_existing = 0
    unresolved_count = 0

    for repo_index, repo in enumerate(ordered_repos, start=1):
        commit_map = repo_tasks[repo]
        commits = sorted(commit_map)

        if args.max_commits_per_repo is not None:
            if args.max_commits_per_repo <= 0:
                print(
                    "[错误] --max-commits-per-repo 必须大于 0。",
                    file=sys.stderr,
                )
                return 2
            commits = commits[: args.max_commits_per_repo]

        pending_commits = [
            commit
            for commit in commits
            if (repo, commit) not in existing_successes
        ]
        skipped_existing += len(commits) - len(pending_commits)

        if not pending_commits:
            print(
                f"[{repo_index}/{len(ordered_repos)}] "
                f"{repo}: 已全部验证，跳过"
            )
            continue

        slug = repo_slug(repo)
        git_dir = (
            project_root
            / "data"
            / "cache"
            / "repos"
            / f"{slug}.git"
        )

        print(
            f"[{repo_index}/{len(ordered_repos)}] "
            f"{repo}: {len(pending_commits)} 个待验证 commit"
        )

        repo_started = time.monotonic()

        try:
            cache_action = ensure_repo_cache(
                repo,
                git_dir,
                no_fetch=args.no_fetch,
                full_clone=args.full_clone,
                timeout=args.timeout,
            )
            repo_action_counts[cache_action] += 1
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            print(f"  [仓库失败] {error}", file=sys.stderr)

            for commit in pending_commits:
                values = commit_map[commit]
                failure_records.append(
                    {
                        "repo": repo,
                        "requested_base_commit": commit,
                        "git_dir": git_dir.relative_to(
                            project_root
                        ).as_posix(),
                        "split_names": sorted(values["split_names"]),
                        "canonical_instance_ids": sorted(
                            values["canonical_instance_ids"]
                        ),
                        "status": "failed",
                        "error": error,
                    }
                )
            continue

        resolved_by_commit = {
            commit: resolve_commit(git_dir, commit)
            for commit in pending_commits
        }

        missing_commits = [
            commit
            for commit, resolved in resolved_by_commit.items()
            if resolved is None
        ]

        if missing_commits and not args.no_fetch:
            print(
                f"  [补取] 一次性尝试获取 "
                f"{len(missing_commits)} 个缺失 commit"
            )

            fetch_result = fetch_missing_commits_once(
                git_dir,
                missing_commits,
                full_clone=args.full_clone,
                timeout=args.timeout,
            )

            if fetch_result is not None and fetch_result.returncode != 0:
                print(
                    "  [警告] 批量补取未完全成功："
                    + (
                        fetch_result.stderr.strip()
                        or fetch_result.stdout.strip()
                    ),
                    file=sys.stderr,
                )

            for commit in missing_commits:
                resolved_by_commit[commit] = resolve_commit(
                    git_dir,
                    commit,
                )

        for commit in pending_commits:
            values = commit_map[commit]
            resolved_commit = resolved_by_commit[commit]

            if resolved_commit is None:
                unresolved_count += 1
                failure_records.append(
                    {
                        "repo": repo,
                        "requested_base_commit": commit,
                        "git_dir": git_dir.relative_to(
                            project_root
                        ).as_posix(),
                        "split_names": sorted(values["split_names"]),
                        "canonical_instance_ids": sorted(
                            values["canonical_instance_ids"]
                        ),
                        "status": "failed",
                        "error": "base_commit 无法在仓库缓存中解析",
                    }
                )
                continue

            snapshot_id = (
                "snapshot-"
                + sha256_text(
                    f"{repo.lower()}@{resolved_commit}"
                )[:24]
            )

            successful_records[(repo, commit)] = {
                "snapshot_id": snapshot_id,
                "repo": repo,
                "requested_base_commit": commit,
                "resolved_commit": resolved_commit,
                "git_dir": git_dir.relative_to(
                    project_root
                ).as_posix(),
                "split_names": sorted(values["split_names"]),
                "canonical_instance_count": len(
                    values["canonical_instance_ids"]
                ),
                "canonical_instance_ids": sorted(
                    values["canonical_instance_ids"]
                ),
                "status": "passed",
                "storage_mode": (
                    "full_bare"
                    if args.full_clone
                    else "partial_bare_blobless"
                ),
                "script_version": SCRIPT_VERSION,
            }
            newly_verified += 1

        print(
            f"  [完成] 仓库仅处理一次，耗时 "
            f"{time.monotonic() - repo_started:.1f}s"
        )

        # 每处理完一个仓库就落盘，支持中断后续跑。
        write_jsonl(
            manifest_path,
            [
                successful_records[key]
                for key in sorted(
                    successful_records,
                    key=lambda item: (
                        item[0].lower(),
                        item[1],
                    ),
                )
            ],
        )
        write_jsonl(failure_path, failure_records)

    passed_records = [
        successful_records[key]
        for key in sorted(
            successful_records,
            key=lambda item: (
                item[0].lower(),
                item[1],
            ),
        )
    ]
    write_jsonl(manifest_path, passed_records)
    write_jsonl(failure_path, failure_records)

    total_requested_commits = sum(
        min(
            len(repo_tasks[repo]),
            args.max_commits_per_repo
            if args.max_commits_per_repo is not None
            else len(repo_tasks[repo]),
        )
        for repo in ordered_repos
    )

    status = (
        "passed"
        if unresolved_count == 0 and not invalid_inputs
        else "partial"
        if passed_records
        else "failed"
    )

    report = {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now(),
        "status": status,
        "git_version": git_version,
        "selected_splits": selected_splits,
        "repository_count": len(ordered_repos),
        "requested_commit_count": total_requested_commits,
        "existing_success_count": len(existing_successes),
        "skipped_existing_count": skipped_existing,
        "newly_verified_count": newly_verified,
        "total_success_count": len(passed_records),
        "unresolved_commit_count": unresolved_count,
        "invalid_input_count": len(invalid_inputs),
        "repo_action_counts": dict(repo_action_counts),
        "storage_mode": (
            "full_bare"
            if args.full_clone
            else "partial_bare_blobless"
        ),
        "note": (
            "每个仓库仅 clone/fetch 一次；"
            "commit 验证均在本地批量执行。"
        ),
    }
    write_json(report_path, report)

    print(
        json.dumps(
            {
                "status": status,
                "repository_count": len(ordered_repos),
                "requested_commit_count": total_requested_commits,
                "skipped_existing_count": skipped_existing,
                "newly_verified_count": newly_verified,
                "total_success_count": len(passed_records),
                "unresolved_commit_count": unresolved_count,
                "storage_mode": report["storage_mode"],
                "output_directory": str(registry_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0 if status == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
