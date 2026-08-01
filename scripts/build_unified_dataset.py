#!/usr/bin/env python3
"""Unified SWE Dataset 单脚本构建器。"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import unittest
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


def _run_contract_tests() -> int:
    argv = [sys.argv[0], *(arg for arg in sys.argv[1:] if arg != "--self-test")]
    program = unittest.main(argv=argv, exit=False)
    return 0 if program.result.wasSuccessful() else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_run_contract_tests())
    build_parser().parse_args()
    raise SystemExit("数据阶段尚未实现；当前仅完成构建契约和 CLI。")
