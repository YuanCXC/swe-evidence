#!/usr/bin/env python3
"""Unified SWE Dataset 单脚本构建器。"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from typing import Any, Sequence


DATASET_NAME = "unified_swe_dataset_v1"
DATASET_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"
SCRIPT_VERSION = "0.1.0"

RELEASE_FILES = (
    "train.parquet",
    "validation.parquet",
    "benchmark.parquet",
    "repository_corpus.parquet",
    "manifest.json",
)
EXPECTED_SPLIT_COUNTS = {"train": 18_336, "validation": 223, "benchmark": 2_294}
EXPECTED_RAW_SWEBENCH_COUNTS = {"train": 19_008, "dev": 225, "test": 2_294}
EXPECTED_TEACHER_PACKETS = {
    "train_main": 12_000,
    "validation": 1_784,
    "train_rare": 1_216,
}

RETRIEVAL_CHANNELS = ("bm25_content", "path_name", "symbol", "structure")
RRF_K = 64
CHANNEL_DEPTH = 64
FINAL_DEPTH = 64
REGULAR_PAIR_CAP = 8

TOKENIZER_NAME = "BAAI/bge-reranker-v2-m3"
TOKENIZER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MODEL_MAX_LENGTH = 4_096
QUESTION_MAX_TOKENS = 2_048
SCOREABLE_UNIT_MAX_TOKENS = 1_024
EVIDENCE_TOKEN_BUDGET = 32_768
SELECTED_EVIDENCE_UNIT_CAP = 64

BUILD_PHASES = (
    "sources",
    "normalize",
    "identity",
    "split",
    "snapshots",
    "corpus",
    "supervision",
    "teacher",
    "policy",
    "write",
    "audit",
    "publish",
)

SOURCE_DATASET_REVISIONS = {
    "swebench": "sha256:ca8ccb0733c6722b9023e85c46eb9a6eda1e694c956bfe2fa527e4a13b8a5eed",
    "contextbench": "sha256:4667f05863f22e88ea6a1e0627ad45f866421b7d293b408a3a737dd73b7c90ea",
    "swe_explore": "sha256:66eeaa78bd71ff3972a5c63a4a4d63e1c0c5ad7dadfeb6774c522f2c163873c5",
}
SOURCE_LICENSES = {
    "swebench": "MIT",
    "contextbench": "Apache-2.0",
    "swe_explore": "MIT",
}
FROZEN_SOURCE_FILES = {
    "swebench": {
        "dev-00000-of-00001.parquet": "4703031c73d7137217f5936adaf59301442d02a41f699b12e3b8637508e2b5a1",
        "test-00000-of-00001.parquet": "0996c4bd66e647ecef89d3ef57e527dcc0ea4e8369ac6f740cdd570734188df2",
        "train-00000-of-00001.parquet": "0ee19c80623ebc6eeef483b597dd38f27c1dda22054e00210976d315cea87a69",
    },
    "contextbench": {
        "contextbench_verified.parquet": "e9dcfd504cbfb849ac815a79040c793d0d92f94eecc9b5a4ee3e1445a2f8a791",
        "contextbench_verified_test.parquet": "4b560777bc8ba4061c7afa5f98ca2b5c793d0a32f085f25c5c817db0708b3629",
        "contextbench_verified_train.parquet": "cd1bc4ddefa69271d4f073f31d54bf33a1c5671800211d9db7e6a3c24435a75d",
        "full.parquet": "2f56535bdc73eb8a68bf4ebb49789d8e9cd4f219ea60df6290b85278aee61ca8",
        "selected_500_instances.csv": "03aace3b95956bb93decb2d8dd6a0c96f7b9d4f89febcdee97ff3e5b73da482f",
        "test.parquet": "42142b51e0fb5a25f227541e80ad659ed522b094f35969c03830b5d8f25af3dc",
        "train.parquet": "34482f20a65dbe1bd6cdc928aa5322bea2143a7ab26363145274581019f02e76",
    },
    "swe_explore": {
        "bench.final.public.jsonl": "dc4f114ececd0bfb987361c26ae5e2440456e2cccb36adfccb09ea5385aec202",
    },
}
SOURCE_FILE_URLS = {
    (
        "swebench",
        name,
    ): (
        "https://huggingface.co/datasets/princeton-nlp/SWE-bench/resolve/"
        f"e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6/data/{name}",
        None,
    )
    for name in FROZEN_SOURCE_FILES["swebench"]
}
SOURCE_FILE_URLS.update(
    {
        (
            "contextbench",
            name,
        ): (
            "https://huggingface.co/datasets/Contextbench/ContextBench/resolve/"
            f"c2855792b006af41c67202d33883fb9d46362853/data/{name}",
            None,
        )
        for name in FROZEN_SOURCE_FILES["contextbench"]
        if name != "selected_500_instances.csv"
    }
)
SOURCE_FILE_URLS[("contextbench", "selected_500_instances.csv")] = (
    "https://raw.githubusercontent.com/EuniAI/ContextBench/"
    "2b43909d120c915b403dd0e15d80450b101c9c4b/data/selected_500_instances.csv",
    "crlf",
)
SOURCE_FILE_URLS[("swe_explore", "bench.final.public.jsonl")] = (
    "https://huggingface.co/datasets/SWE-Explore-Bench/SWE-Explore-Bench/resolve/"
    "bdb0ae45d7c337d9e1dc3ebfe2a0af6bc7c1fbd9/bench.final.public.jsonl",
    None,
)

REQUIRED_STATE_TABLES = (
    "source_records",
    "canonical_tasks",
    "task_aliases",
    "supervision",
    "trajectories",
    "split_assignments",
    "snapshots",
    "file_versions",
    "snapshot_file_memberships",
    "evidence_units",
    "obligations",
    "witness_groups",
    "policy_states",
    "candidate_actions",
    "teacher_cache",
    "conflicts",
    "build_phases",
)


def stable_json_dumps(value: Any) -> str:
    """返回用于哈希和 JSONL 的稳定、无 ASCII 转义 JSON。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_id(prefix: str, *parts: object, length: int = 24) -> str:
    """由带类型的有序组成部分生成确定性 SHA-256 ID。"""

    if not prefix or not prefix.replace("-", "_").isalnum():
        raise ValueError("ID prefix 必须是非空字母数字标识，可包含 '-' 或 '_'。")
    if length < 8 or length > 64:
        raise ValueError("ID hash 长度必须位于 [8, 64]。")
    payload = stable_json_dumps([str(part) for part in parts]).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:length]
    return f"{prefix}_{digest}"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """流式计算文件 SHA-256，不把大文件整体载入内存。"""

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数。")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_sources(raw_root: Path) -> dict[str, str]:
    """验证全部冻结来源文件，并返回各来源的目录字节指纹。"""

    fingerprints: dict[str, str] = {}
    for dataset, files in FROZEN_SOURCE_FILES.items():
        digest = hashlib.sha256()
        for relative_name, expected_sha256 in sorted(files.items()):
            path = raw_root / dataset / relative_name
            if not path.is_file():
                raise FileNotFoundError(f"缺少冻结来源文件：{path}")
            actual_sha256 = sha256_file(path)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"来源文件哈希不匹配：{path}，actual={actual_sha256}，"
                    f"expected={expected_sha256}"
                )
            manifest_path = f"data/raw/{dataset}/{relative_name}"
            digest.update(manifest_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(actual_sha256.encode("ascii"))
            digest.update(b"\n")
        fingerprint = digest.hexdigest()
        expected_fingerprint = SOURCE_DATASET_REVISIONS[dataset].removeprefix("sha256:")
        if fingerprint != expected_fingerprint:
            raise ValueError(
                f"来源目录指纹不匹配：{dataset}，actual={fingerprint}，"
                f"expected={expected_fingerprint}"
            )
        fingerprints[dataset] = fingerprint
    return fingerprints


def _download_file(url: str, destination: Path, newline_mode: str | None) -> None:
    """从固定地址流式下载单个来源文件，并在同目录原子落盘。"""

    import httpx

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            with temporary.open("wb") as file:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    file.write(chunk)
        if newline_mode == "crlf":
            payload = temporary.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            temporary.write_bytes(payload)
        elif newline_mode is not None:
            raise ValueError(f"未知下载换行模式：{newline_mode!r}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_frozen_sources(raw_root: Path) -> dict[str, str]:
    """只下载缺失的固定来源文件，随后逐文件和逐来源校验哈希。"""

    for dataset, files in FROZEN_SOURCE_FILES.items():
        for relative_name in files:
            destination = raw_root / dataset / relative_name
            if destination.is_file():
                continue
            try:
                url, newline_mode = SOURCE_FILE_URLS[(dataset, relative_name)]
            except KeyError as error:
                raise ValueError(
                    f"缺失来源文件且没有固定下载地址：{dataset}/{relative_name}"
                ) from error
            _download_file(url, destination, newline_mode)
    return validate_frozen_sources(raw_root)


def build_parser() -> argparse.ArgumentParser:
    """构造唯一入口的命令行参数解析器。"""

    parser = argparse.ArgumentParser(
        description="从冻结原始来源和 pre-fix Git 快照构建 Unified SWE Dataset。"
    )
    parser.add_argument("--format", choices=("jsonl", "parquet"), default="jsonl")
    parser.add_argument("--release", action="store_true", help="通过硬门禁后原子发布正式版。")
    parser.add_argument("--clean-state", action="store_true", help="复核正式文件哈希后删除 SQLite 状态。")
    parser.add_argument("--audit-only", action="store_true", help="只运行已完成阶段的审计，不发布。")
    parser.add_argument("--through-phase", choices=BUILD_PHASES, help="执行到指定内部阶段后停止。")
    parser.add_argument("--self-test", action="store_true", help="运行脚本内置契约与单元测试。")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def normalize_repo(value: object) -> str:
    """将 GitHub 仓库标识规范化为 owner/repo。"""

    text = str(value or "").strip().replace("\\", "/")
    for prefix in ("https://github.com/", "http://github.com/", "git@github.com:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :]
            break
    text = text.removesuffix(".git").strip("/")
    if len(text.split("/")) != 2:
        raise ValueError(f"非法仓库标识：{value!r}")
    return text


def normalize_hints(value: object) -> list[str]:
    """把 SWE-bench 的 hints_text 稳定转换为字符串列表。"""

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(decoded, list):
        return [str(item).strip() for item in decoded if str(item).strip()]
    if decoded is None:
        return []
    return [str(decoded).strip()] if str(decoded).strip() else []


def map_upstream_split(upstream_split: str) -> str:
    """把 SWE-bench 原始 split 映射到冻结物理 split。"""

    mapping = {"train": "train", "dev": "validation", "test": "benchmark"}
    try:
        return mapping[upstream_split]
    except KeyError as error:
        raise ValueError(f"未知 SWE-bench split：{upstream_split!r}") from error


def normalize_swebench_task(record: dict[str, Any], upstream_split: str) -> dict[str, Any]:
    """把一个 SWE-bench 记录转换为公共任务 Schema 的未增强基线。"""

    final_split = map_upstream_split(upstream_split)
    source_id = str(record.get("instance_id") or "").strip()
    repo = normalize_repo(record.get("repo"))
    base_commit = str(record.get("base_commit") or "").strip().lower()
    problem = str(record.get("problem_statement") or "").strip()
    if not source_id or not base_commit or not problem:
        raise ValueError("SWE-bench 记录缺少 instance_id、base_commit 或 problem_statement。")

    task_id = stable_id("task", "swebench", source_id)
    task_group_id = stable_id("group", "swebench", source_id)
    snapshot_id = stable_id("snapshot", repo, base_commit)
    raw_sha256 = hashlib.sha256(stable_json_dumps(record).encode("utf-8")).hexdigest()
    evaluation = None
    if final_split == "benchmark":
        evaluation = {
            "benchmark_memberships": [
                {
                    "suite": "swe-bench",
                    "subset": upstream_split,
                    "version": str(record.get("version") or "unknown"),
                    "original_source_id": source_id,
                }
            ],
            "targets": ["code_repair", "evidence_localization", "evidence_sufficiency"],
            "gold_visibility": "evaluator_only",
            "timeout_seconds": 1_800,
            "execution_required": True,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "task_group_id": task_group_id,
        "snapshot_id": snapshot_id,
        "input": {
            "repo": repo,
            "base_commit": base_commit,
            "language": str(record.get("language") or "unknown").strip().lower(),
            "issue_id": source_id,
            "problem_statement": problem,
            "hints": normalize_hints(record.get("hints_text")),
            "created_at": record.get("created_at"),
            "environment": None,
            "retrieval_scope": {
                "snapshot_id": snapshot_id,
                "allowed_unit_types": [
                    "class",
                    "function",
                    "method",
                    "code_block",
                    "doc_section",
                ],
            },
        },
        "provenance": [
            {
                "dataset": "swebench",
                "subset": upstream_split,
                "source_id": source_id,
                "version": str(record.get("version") or "unknown"),
                "revision": SOURCE_DATASET_REVISIONS["swebench"],
                "license": SOURCE_LICENSES["swebench"],
                "trust_tier": "support",
                "raw_record_sha256": raw_sha256,
            }
        ],
        "supervision": {
            "level": "none",
            "training_targets": [],
            "recommended_weight": None,
            "evidence_labels": [],
            "modified_files": [],
            "gold_patch": record.get("patch") or None,
            "test_patch": record.get("test_patch") or None,
            "hard_negative_evidence_ids": [],
            "obligations": [],
            "policy_states": [],
            "label_provenance": [],
        },
        "trajectories": [],
        "evaluation": evaluation,
        "split_info": {
            "split": final_split,
            "trainable": final_split == "train",
            "split_reason": f"swebench_original_{upstream_split}",
            "split_policy_version": "swebench_frozen_v1",
            "leakage_group": task_group_id,
            "frozen": True,
        },
        "quality": {
            "status": "passed_with_warnings",
            "identity_confidence": 1.0,
            "label_confidence": 0.0,
            "executable": False,
            "snapshot_available": False,
            "evidence_mapping_rate": 0.0,
            "problem_token_count": 0,
            "model_question_token_count": 0,
            "question_truncated": False,
            "warnings": ["language_pending_corpus_detection", "supervision_pending"],
        },
    }


def normalize_contextbench_overlay(
    record: dict[str, Any], swebench_ids: set[str], *, subset: str
) -> dict[str, Any] | None:
    """只接受能由 original_inst_id 精确命中 SWE-bench 的 ContextBench overlay。"""

    source_id = str(record.get("original_inst_id") or "").strip()
    if source_id not in swebench_ids:
        return None
    raw_sha256 = hashlib.sha256(stable_json_dumps(record).encode("utf-8")).hexdigest()
    return {
        "dataset": "contextbench",
        "subset": subset,
        "source_id": source_id,
        "variant_id": str(record.get("instance_id") or "").strip(),
        "revision": SOURCE_DATASET_REVISIONS["contextbench"],
        "license": SOURCE_LICENSES["contextbench"],
        "trust_tier": "strong",
        "raw_record_sha256": raw_sha256,
        "gold_context": record.get("gold_context") or [],
        "raw_record": record,
    }


def normalize_swe_explore_overlay(
    record: dict[str, Any], swebench_ids: set[str]
) -> dict[str, Any] | None:
    """只接受 instance_id 与 SWE-bench 完全一致的 SWE-Explore 记录。"""

    source_id = str(record.get("instance_id") or "").strip()
    if source_id not in swebench_ids:
        return None
    raw_sha256 = hashlib.sha256(stable_json_dumps(record).encode("utf-8")).hexdigest()
    return {
        "dataset": "swe_explore",
        "subset": str(record.get("dataset") or "verified"),
        "source_id": source_id,
        "revision": SOURCE_DATASET_REVISIONS["swe_explore"],
        "license": SOURCE_LICENSES["swe_explore"],
        "trust_tier": "observed",
        "raw_record_sha256": raw_sha256,
        "raw_record": record,
    }


def iter_swebench_records(raw_root: Path):
    """按 train/dev/test 顺序流式读取冻结 SWE-bench Parquet。"""

    import pyarrow.parquet as pq

    for split in ("train", "dev", "test"):
        paths = sorted(raw_root.glob(f"{split}-*.parquet"))
        if not paths:
            continue
        for path in paths:
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(batch_size=256):
                for record in batch.to_pylist():
                    yield split, record


def open_state_database(path: Path) -> sqlite3.Connection:
    """创建或打开唯一可恢复 SQLite 状态库。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS source_records (
            source_record_id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            subset TEXT,
            source_id TEXT NOT NULL,
            raw_sha256 TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS canonical_tasks (
            task_id TEXT PRIMARY KEY,
            task_group_id TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            upstream_split TEXT NOT NULL,
            final_split TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS task_aliases (
            dataset TEXT NOT NULL,
            source_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            raw_sha256 TEXT NOT NULL,
            PRIMARY KEY (dataset, source_id, raw_sha256)
        );
        CREATE TABLE IF NOT EXISTS supervision (
            task_id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trajectories (
            trajectory_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS split_assignments (
            task_id TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            trainable INTEGER NOT NULL,
            frozen INTEGER NOT NULL,
            reason TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS snapshots (
            snapshot_id TEXT PRIMARY KEY,
            repo TEXT NOT NULL,
            base_commit TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS file_versions (
            file_version_id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS snapshot_file_memberships (
            snapshot_id TEXT NOT NULL,
            path TEXT NOT NULL,
            file_version_id TEXT NOT NULL,
            PRIMARY KEY (snapshot_id, path)
        );
        CREATE TABLE IF NOT EXISTS evidence_units (
            evidence_id TEXT PRIMARY KEY,
            file_version_id TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS obligations (
            obligation_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS witness_groups (
            group_id TEXT PRIMARY KEY,
            obligation_id TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS policy_states (
            state_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS candidate_actions (
            action_key TEXT PRIMARY KEY,
            state_id TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS teacher_cache (
            input_sha256 TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS conflicts (
            conflict_id TEXT PRIMARY KEY,
            severity TEXT NOT NULL,
            reason TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS build_phases (
            phase_name TEXT PRIMARY KEY,
            phase_version TEXT NOT NULL,
            input_fingerprint TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            failed_at TEXT,
            processed_count INTEGER NOT NULL DEFAULT 0,
            output_row_count INTEGER NOT NULL DEFAULT 0,
            error_summary TEXT,
            resumable INTEGER NOT NULL DEFAULT 1
        );
        """
    )
    connection.commit()
    return connection


def build_source_state(
    raw_root: Path,
    state_path: Path,
    *,
    expected_raw_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """规范化唯一 SWE-bench 任务，并把可对齐来源保存为 overlay。"""

    import pyarrow.parquet as pq

    swebench_root = raw_root / "swebench"
    contextbench_full = raw_root / "contextbench" / "full.parquet"
    swe_explore_path = raw_root / "swe_explore" / "bench.final.public.jsonl"
    if not swebench_root.is_dir():
        raise FileNotFoundError(f"缺少 SWE-bench 目录：{swebench_root}")
    if not contextbench_full.is_file():
        raise FileNotFoundError(f"缺少 ContextBench 主表：{contextbench_full}")
    if not swe_explore_path.is_file():
        raise FileNotFoundError(f"缺少 SWE-Explore 主表：{swe_explore_path}")

    connection = open_state_database(state_path)
    raw_counts = {"train": 0, "dev": 0, "test": 0}
    task_id_by_source: dict[str, str] = {}
    contextbench_ids: set[str] = set()
    swe_explore_ids: set[str] = set()
    try:
        connection.execute("BEGIN IMMEDIATE")
        for table in (
            "source_records",
            "canonical_tasks",
            "task_aliases",
            "supervision",
            "trajectories",
            "split_assignments",
            "snapshots",
            "file_versions",
            "snapshot_file_memberships",
            "evidence_units",
            "obligations",
            "witness_groups",
            "policy_states",
            "candidate_actions",
            "teacher_cache",
            "conflicts",
            "build_phases",
        ):
            connection.execute(f'DELETE FROM "{table}"')

        for upstream_split, record in iter_swebench_records(swebench_root):
            raw_counts[upstream_split] += 1
            task = normalize_swebench_task(record, upstream_split)
            source_id = task["provenance"][0]["source_id"]
            if source_id in task_id_by_source:
                raise ValueError(f"SWE-bench instance_id 重复：{source_id}")
            task_id_by_source[source_id] = task["task_id"]
            raw_sha256 = task["provenance"][0]["raw_record_sha256"]
            source_record_id = stable_id(
                "source", "swebench", upstream_split, source_id, raw_sha256
            )
            connection.execute(
                "INSERT INTO source_records VALUES (?, ?, ?, ?, ?, ?)",
                (
                    source_record_id,
                    "swebench",
                    upstream_split,
                    source_id,
                    raw_sha256,
                    stable_json_dumps(record),
                ),
            )
            state_task = copy.deepcopy(task)
            state_task["supervision"]["gold_patch"] = None
            state_task["supervision"]["test_patch"] = None
            connection.execute(
                "INSERT INTO canonical_tasks VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    task["task_id"],
                    task["task_group_id"],
                    task["snapshot_id"],
                    upstream_split,
                    task["split_info"]["split"],
                    "normalized",
                    stable_json_dumps(state_task),
                ),
            )
            connection.execute(
                "INSERT INTO task_aliases VALUES (?, ?, ?, ?)",
                ("swebench", source_id, task["task_id"], raw_sha256),
            )
            connection.execute(
                "INSERT INTO split_assignments VALUES (?, ?, ?, ?, ?)",
                (
                    task["task_id"],
                    task["split_info"]["split"],
                    int(task["split_info"]["trainable"]),
                    1,
                    task["split_info"]["split_reason"],
                ),
            )
            connection.execute(
                "INSERT OR IGNORE INTO snapshots VALUES (?, ?, ?, ?, ?)",
                (
                    task["snapshot_id"],
                    task["input"]["repo"],
                    task["input"]["base_commit"],
                    "pending",
                    stable_json_dumps(
                        {
                            "snapshot_id": task["snapshot_id"],
                            "repo": task["input"]["repo"],
                            "base_commit": task["input"]["base_commit"],
                        }
                    ),
                ),
            )

        if expected_raw_counts is not None and raw_counts != expected_raw_counts:
            raise ValueError(
                f"SWE-bench 原始 split 数量不匹配：actual={raw_counts}, "
                f"expected={expected_raw_counts}"
            )

        swebench_ids = set(task_id_by_source)
        context_file = pq.ParquetFile(contextbench_full)
        for batch in context_file.iter_batches(batch_size=128):
            for record in batch.to_pylist():
                overlay = normalize_contextbench_overlay(record, swebench_ids, subset="full")
                if overlay is None:
                    continue
                source_id = overlay["source_id"]
                contextbench_ids.add(source_id)
                raw_sha256 = overlay["raw_record_sha256"]
                connection.execute(
                    "INSERT OR IGNORE INTO source_records VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        stable_id("source", "contextbench", "full", source_id, raw_sha256),
                        "contextbench",
                        "full",
                        source_id,
                        raw_sha256,
                        stable_json_dumps(record),
                    ),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO task_aliases VALUES (?, ?, ?, ?)",
                    ("contextbench", source_id, task_id_by_source[source_id], raw_sha256),
                )

        with swe_explore_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"SWE-Explore JSONL 第 {line_number} 行无法解析。"
                    ) from error
                overlay = normalize_swe_explore_overlay(record, swebench_ids)
                if overlay is None:
                    continue
                source_id = overlay["source_id"]
                swe_explore_ids.add(source_id)
                raw_sha256 = overlay["raw_record_sha256"]
                subset = overlay["subset"]
                connection.execute(
                    "INSERT OR IGNORE INTO source_records VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        stable_id("source", "swe_explore", subset, source_id, raw_sha256),
                        "swe_explore",
                        subset,
                        source_id,
                        raw_sha256,
                        stable_json_dumps(record),
                    ),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO task_aliases VALUES (?, ?, ?, ?)",
                    ("swe_explore", source_id, task_id_by_source[source_id], raw_sha256),
                )

        if expected_raw_counts is not None:
            if len(contextbench_ids) != 351:
                raise ValueError(
                    f"ContextBench 严格对齐任务数应为 351，实际为 {len(contextbench_ids)}。"
                )
            if len(swe_explore_ids) != 451:
                raise ValueError(
                    f"SWE-Explore 严格对齐任务数应为 451，实际为 {len(swe_explore_ids)}。"
                )

        report = {
            "raw_swebench_counts": raw_counts,
            "canonical_task_count": len(task_id_by_source),
            "contextbench_aligned_task_count": len(contextbench_ids),
            "swe_explore_aligned_task_count": len(swe_explore_ids),
            "overlay_task_union_count": len(contextbench_ids | swe_explore_ids),
            "overlay_task_intersection_count": len(contextbench_ids & swe_explore_ids),
        }
        fingerprint = hashlib.sha256(stable_json_dumps(report).encode("utf-8")).hexdigest()
        connection.execute(
            "INSERT INTO build_phases "
            "(phase_name, phase_version, input_fingerprint, completed_at, "
            "processed_count, output_row_count, resumable) "
            "VALUES (?, ?, ?, datetime('now'), ?, ?, 1)",
            (
                "split",
                "1.0.0",
                fingerprint,
                sum(raw_counts.values()),
                len(task_id_by_source),
            ),
        )
        connection.commit()
        return report
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def audit_source_state(
    state_path: Path,
    *,
    expected_raw_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """从持久化状态独立重算来源、任务和 overlay 数量。"""

    if not state_path.is_file():
        raise FileNotFoundError(f"状态库不存在：{state_path}")
    connection = sqlite3.connect(state_path)
    try:
        raw_counts = {"train": 0, "dev": 0, "test": 0}
        for split, count in connection.execute(
            "SELECT upstream_split, COUNT(*) FROM canonical_tasks GROUP BY upstream_split"
        ):
            if split not in raw_counts:
                raise ValueError(f"状态库包含未知 upstream split：{split}")
            raw_counts[split] = count
        canonical_count = connection.execute(
            "SELECT COUNT(*) FROM canonical_tasks"
        ).fetchone()[0]
        contextbench_ids = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT source_id FROM task_aliases WHERE dataset='contextbench'"
            )
        }
        swe_explore_ids = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT source_id FROM task_aliases WHERE dataset='swe_explore'"
            )
        }
        duplicate_tasks = connection.execute(
            "SELECT COUNT(*) - COUNT(DISTINCT task_id) FROM canonical_tasks"
        ).fetchone()[0]
        split_rows = connection.execute(
            "SELECT COUNT(*) FROM split_assignments"
        ).fetchone()[0]
    finally:
        connection.close()
    if duplicate_tasks:
        raise ValueError(f"状态库存在 {duplicate_tasks} 个重复 task_id。")
    if split_rows != canonical_count:
        raise ValueError(
            f"split assignment 数量不完整：tasks={canonical_count}, assignments={split_rows}"
        )
    if expected_raw_counts is not None:
        if raw_counts != expected_raw_counts:
            raise ValueError(
                f"SWE-bench 状态数量不匹配：actual={raw_counts}, expected={expected_raw_counts}"
            )
        if len(contextbench_ids) != 351 or len(swe_explore_ids) != 451:
            raise ValueError(
                "严格 overlay 数量不匹配："
                f"contextbench={len(contextbench_ids)}, swe_explore={len(swe_explore_ids)}"
            )
    return {
        "raw_swebench_counts": raw_counts,
        "canonical_task_count": canonical_count,
        "contextbench_aligned_task_count": len(contextbench_ids),
        "swe_explore_aligned_task_count": len(swe_explore_ids),
        "overlay_task_union_count": len(contextbench_ids | swe_explore_ids),
        "overlay_task_intersection_count": len(contextbench_ids & swe_explore_ids),
    }


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    raw_root: Path = Path("data/raw"),
    state_path: Path = Path("data/.build/unified_swe_v1.sqlite3"),
    expected_raw_counts: dict[str, int] | None = EXPECTED_RAW_SWEBENCH_COUNTS,
    output: Any = sys.stdout,
) -> int:
    """执行当前已实现阶段；来源阶段的构建与审计使用同一稳定报告。"""

    args = build_parser().parse_args(argv)
    if args.self_test:
        return _run_contract_tests()
    if args.release and args.format != "parquet":
        raise ValueError("正式发布必须使用 --format parquet。")
    if args.clean_state:
        raise ValueError("--clean-state 仅能在正式发布及文件哈希复核完成后使用。")

    target_phase = args.through_phase or BUILD_PHASES[-1]
    source_phase_index = BUILD_PHASES.index("split")
    target_phase_index = BUILD_PHASES.index(target_phase)
    if target_phase_index < source_phase_index:
        raise ValueError("sources、normalize、identity 与 split 是不可拆分的原子来源阶段。")

    if args.audit_only:
        report = audit_source_state(
            state_path,
            expected_raw_counts=expected_raw_counts,
        )
    else:
        if expected_raw_counts is not None:
            ensure_frozen_sources(raw_root)
        report = build_source_state(
            raw_root,
            state_path,
            expected_raw_counts=expected_raw_counts,
        )
        audited = audit_source_state(
            state_path,
            expected_raw_counts=expected_raw_counts,
        )
        if audited != report:
            raise ValueError(f"来源阶段构建报告与独立审计不一致：build={report}, audit={audited}")

    output.write(stable_json_dumps(report) + "\n")
    if target_phase_index > source_phase_index:
        raise ValueError(f"阶段 {target_phase!r} 尚未实现；当前可执行到 'split'。")
    return 0


class ContractTests(unittest.TestCase):
    """锁定主设计文档中已经确认的不可变构建契约。"""

    def test_release_file_contract(self) -> None:
        self.assertIn("RELEASE_FILES", globals())
        self.assertEqual(
            globals().get("RELEASE_FILES"),
            (
                "train.parquet",
                "validation.parquet",
                "benchmark.parquet",
                "repository_corpus.parquet",
                "manifest.json",
            ),
        )

    def test_split_count_contract(self) -> None:
        self.assertIn("EXPECTED_SPLIT_COUNTS", globals())
        self.assertEqual(
            globals().get("EXPECTED_SPLIT_COUNTS"),
            {"train": 18_336, "validation": 223, "benchmark": 2_294},
        )

    def test_retrieval_contract(self) -> None:
        self.assertEqual(
            globals().get("RETRIEVAL_CHANNELS"),
            ("bm25_content", "path_name", "symbol", "structure"),
        )
        self.assertEqual(globals().get("RRF_K"), 64)
        self.assertEqual(globals().get("CHANNEL_DEPTH"), 64)
        self.assertEqual(globals().get("FINAL_DEPTH"), 64)

    def test_token_and_evidence_budget_contract(self) -> None:
        self.assertEqual(globals().get("MODEL_MAX_LENGTH"), 4_096)
        self.assertEqual(globals().get("QUESTION_MAX_TOKENS"), 2_048)
        self.assertEqual(globals().get("SCOREABLE_UNIT_MAX_TOKENS"), 1_024)
        self.assertEqual(globals().get("EVIDENCE_TOKEN_BUDGET"), 32_768)
        self.assertEqual(globals().get("SELECTED_EVIDENCE_UNIT_CAP"), 64)

    def test_teacher_packet_contract(self) -> None:
        self.assertEqual(
            globals().get("EXPECTED_TEACHER_PACKETS"),
            {"train_main": 12_000, "validation": 1_784, "train_rare": 1_216},
        )

    def test_source_license_contract(self) -> None:
        self.assertEqual(
            globals().get("SOURCE_LICENSES"),
            {"swebench": "MIT", "contextbench": "Apache-2.0", "swe_explore": "MIT"},
        )

    def test_stable_json_and_id_contract(self) -> None:
        self.assertIn("stable_json_dumps", globals())
        self.assertIn("stable_id", globals())
        if "stable_json_dumps" not in globals() or "stable_id" not in globals():
            return
        self.assertEqual(
            stable_json_dumps({"z": 1, "a": ["中文", 2]}),
            '{"a":["中文",2],"z":1}',
        )
        first = stable_id("task", "django/django", "123", length=24)
        second = stable_id("task", "django/django", "123", length=24)
        changed = stable_id("task", "django/django", "124", length=24)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first), len("task_") + 24)

    def test_cli_contract(self) -> None:
        self.assertIn("build_parser", globals())
        if "build_parser" not in globals():
            return
        parser = build_parser()
        default_args = parser.parse_args([])
        self.assertEqual(default_args.format, "jsonl")
        self.assertFalse(default_args.release)
        self.assertFalse(default_args.audit_only)
        release_args = parser.parse_args(["--format", "parquet", "--release"])
        self.assertEqual(release_args.format, "parquet")
        self.assertTrue(release_args.release)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--format", "csv"])


class SourceNormalizationTests(unittest.TestCase):
    """锁定 SWE-bench 基准身份和 overlay 的防泄漏边界。"""

    def test_swebench_task_keeps_gold_out_of_input(self) -> None:
        self.assertIn("normalize_swebench_task", globals())
        if "normalize_swebench_task" not in globals():
            return
        record = {
            "repo": "django/django",
            "instance_id": "django__django-123",
            "base_commit": "a" * 40,
            "problem_statement": "Fix QuerySet behavior",
            "hints_text": "use the public API",
            "patch": "diff --git a/a.py b/a.py",
            "test_patch": "diff --git a/tests.py b/tests.py",
            "created_at": "2024-01-02T03:04:05Z",
            "version": "1.0",
        }
        task = normalize_swebench_task(record, "train")
        self.assertEqual(task["input"]["repo"], "django/django")
        self.assertEqual(task["input"]["problem_statement"], record["problem_statement"])
        self.assertEqual(task["input"]["hints"], ["use the public API"])
        self.assertNotIn("patch", task["input"])
        self.assertNotIn("test_patch", task["input"])
        self.assertEqual(task["supervision"]["gold_patch"], record["patch"])
        self.assertEqual(task["supervision"]["test_patch"], record["test_patch"])
        self.assertEqual(task["split_info"]["split"], "train")
        self.assertTrue(task["split_info"]["trainable"])

    def test_contextbench_uses_original_instance_id_only_as_overlay(self) -> None:
        self.assertIn("normalize_contextbench_overlay", globals())
        if "normalize_contextbench_overlay" not in globals():
            return
        swe_ids = {"django__django-123"}
        record = {
            "instance_id": "SWE-PolyBench__python__bugfix__abc",
            "original_inst_id": "django__django-123",
            "repo": "django/django",
            "base_commit": "a" * 40,
            "gold_context": [{"file": "a.py", "start_line": 1, "end_line": 2, "content": "x"}],
        }
        overlay = normalize_contextbench_overlay(record, swe_ids, subset="full")
        self.assertEqual(overlay["source_id"], "django__django-123")
        self.assertEqual(overlay["dataset"], "contextbench")
        self.assertNotIn("task_id", overlay)
        self.assertIsNone(
            normalize_contextbench_overlay(
                {**record, "original_inst_id": "not_in_swebench"}, swe_ids, subset="full"
            )
        )

    def test_swe_explore_requires_exact_swebench_instance_id(self) -> None:
        self.assertIn("normalize_swe_explore_overlay", globals())
        if "normalize_swe_explore_overlay" not in globals():
            return
        swe_ids = {"django__django-123"}
        accepted = normalize_swe_explore_overlay(
            {"instance_id": "django__django-123", "meta": {"num_read_core": 99}}, swe_ids
        )
        self.assertEqual(accepted["source_id"], "django__django-123")
        self.assertIsNone(
            normalize_swe_explore_overlay({"instance_id": "foreign__task-1"}, swe_ids)
        )

    def test_upstream_split_mapping_is_frozen(self) -> None:
        self.assertIn("map_upstream_split", globals())
        if "map_upstream_split" not in globals():
            return
        self.assertEqual(map_upstream_split("train"), "train")
        self.assertEqual(map_upstream_split("dev"), "validation")
        self.assertEqual(map_upstream_split("test"), "benchmark")
        with self.assertRaises(ValueError):
            map_upstream_split("random")

    def test_swebench_parquet_loader_preserves_all_rows(self) -> None:
        self.assertIn("iter_swebench_records", globals())
        if "iter_swebench_records" not in globals():
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {
                    "repo": "django/django",
                    "instance_id": "django__django-1",
                    "base_commit": "a" * 40,
                    "problem_statement": "one",
                    "patch": "p1",
                    "test_patch": "t1",
                    "hints_text": "",
                    "created_at": None,
                    "version": "1",
                },
                {
                    "repo": "django/django",
                    "instance_id": "django__django-2",
                    "base_commit": "b" * 40,
                    "problem_statement": "two",
                    "patch": "p2",
                    "test_patch": "t2",
                    "hints_text": "[]",
                    "created_at": None,
                    "version": "1",
                },
            ]
            pq.write_table(pa.Table.from_pylist(rows), root / "train-00000-of-00001.parquet")
            loaded = list(iter_swebench_records(root))
        self.assertEqual([item[0] for item in loaded], ["train", "train"])
        self.assertEqual([item[1]["instance_id"] for item in loaded], ["django__django-1", "django__django-2"])

    def test_sqlite_state_contains_required_tables(self) -> None:
        self.assertIn("open_state_database", globals())
        if "open_state_database" not in globals():
            return
        with tempfile.TemporaryDirectory() as tmp:
            connection = open_state_database(Path(tmp) / "state.sqlite3")
            try:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            finally:
                connection.close()
        self.assertTrue(set(REQUIRED_STATE_TABLES).issubset(tables))

    def test_source_phase_creates_only_swebench_tasks(self) -> None:
        self.assertIn("build_source_state", globals())
        if "build_source_state" not in globals():
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        def swe_row(instance_id: str, commit: str) -> dict[str, Any]:
            return {
                "repo": "django/django",
                "instance_id": instance_id,
                "base_commit": commit,
                "problem_statement": instance_id,
                "patch": "diff --git a/a.py b/a.py",
                "test_patch": "",
                "hints_text": "",
                "created_at": None,
                "version": "1",
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw"
            swe_root = root / "swebench"
            context_root = root / "contextbench"
            explore_root = root / "swe_explore"
            swe_root.mkdir(parents=True)
            context_root.mkdir()
            explore_root.mkdir()
            fixtures = {
                "train": swe_row("django__django-1", "a" * 40),
                "dev": swe_row("django__django-2", "b" * 40),
                "test": swe_row("django__django-3", "c" * 40),
            }
            for split, row in fixtures.items():
                pq.write_table(
                    pa.Table.from_pylist([row]),
                    swe_root / f"{split}-00000-of-00001.parquet",
                )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "instance_id": "poly-1",
                            "original_inst_id": "django__django-1",
                            "repo": "django/django",
                            "base_commit": "a" * 40,
                            "gold_context": "[]",
                        },
                        {
                            "instance_id": "poly-foreign",
                            "original_inst_id": "foreign__task-1",
                            "repo": "foreign/repo",
                            "base_commit": "d" * 40,
                            "gold_context": "[]",
                        },
                    ]
                ),
                context_root / "full.parquet",
            )
            with (explore_root / "bench.final.public.jsonl").open("w", encoding="utf-8") as file:
                file.write(stable_json_dumps({"instance_id": "django__django-3"}) + "\n")
                file.write(stable_json_dumps({"instance_id": "foreign__task-2"}) + "\n")
            state_path = Path(tmp) / "state.sqlite3"
            report = build_source_state(root, state_path, expected_raw_counts=None)
            self.assertIn("audit_source_state", globals())
            audited = (
                audit_source_state(state_path, expected_raw_counts=None)
                if "audit_source_state" in globals()
                else None
            )
            self.assertIn("run_cli", globals())
            cli_output = io.StringIO()
            cli_status = (
                run_cli(
                    ["--through-phase", "split", "--audit-only"],
                    raw_root=root,
                    state_path=state_path,
                    expected_raw_counts=None,
                    output=cli_output,
                )
                if "run_cli" in globals()
                else None
            )
            connection = sqlite3.connect(state_path)
            try:
                task_count = connection.execute("SELECT COUNT(*) FROM canonical_tasks").fetchone()[0]
                alias_datasets = dict(
                    connection.execute(
                        "SELECT dataset, COUNT(*) FROM task_aliases GROUP BY dataset"
                    ).fetchall()
                )
            finally:
                connection.close()
        self.assertEqual(task_count, 3)
        self.assertEqual(report["raw_swebench_counts"], {"train": 1, "dev": 1, "test": 1})
        self.assertEqual(report["contextbench_aligned_task_count"], 1)
        self.assertEqual(report["swe_explore_aligned_task_count"], 1)
        self.assertEqual(report["overlay_task_union_count"], 2)
        self.assertEqual(alias_datasets, {"contextbench": 1, "swe_explore": 1, "swebench": 3})
        self.assertEqual(audited, report)
        self.assertEqual(cli_status, 0)
        self.assertEqual(json.loads(cli_output.getvalue()), audited)

    def test_file_hash_is_streamed_and_stable(self) -> None:
        self.assertIn("sha256_file", globals())
        if "sha256_file" not in globals():
            return
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.bin"
            path.write_bytes(b"abc")
            digest = sha256_file(path, chunk_size=2)
        self.assertEqual(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )

    def test_missing_frozen_source_is_downloaded_before_validation(self) -> None:
        self.assertIn("ensure_frozen_sources", globals())
        if "ensure_frozen_sources" not in globals():
            return
        payload = b"official frozen bytes\n"
        expected_hash = hashlib.sha256(payload).hexdigest()

        def fake_download(_url: str, path: Path, _newline_mode: str | None) -> int:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path.write_bytes(payload)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                mock.patch.dict(
                    FROZEN_SOURCE_FILES,
                    {"fixture": {"source.bin": expected_hash}},
                    clear=True,
                ),
                mock.patch.dict(
                    SOURCE_DATASET_REVISIONS,
                    {
                        "fixture": "sha256:"
                        + hashlib.sha256(
                            b"data/raw/fixture/source.bin\0"
                            + expected_hash.encode("ascii")
                            + b"\n"
                        ).hexdigest()
                    },
                    clear=True,
                ),
                mock.patch.dict(
                    SOURCE_FILE_URLS,
                    {("fixture", "source.bin"): ("https://example.test/source.bin", None)},
                    clear=True,
                ),
                mock.patch(
                    f"{__name__}._download_file",
                    side_effect=fake_download,
                ) as downloader,
            ):
                fingerprints = ensure_frozen_sources(root)
            self.assertTrue((root / "fixture" / "source.bin").is_file())
            self.assertIn("fixture", fingerprints)
            downloader.assert_called_once()


def _run_contract_tests() -> int:
    argv = [sys.argv[0], *(arg for arg in sys.argv[1:] if arg != "--self-test")]
    program = unittest.main(argv=argv, exit=False)
    return 0 if program.result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run_cli(sys.argv[1:]))
