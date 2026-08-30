#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strong-Teacher Markdown Prompt Refresher v1.5 (Cache-First / Single-Task)

目标：
    只重排已经导出的 Strong-Teacher Markdown 提示词，使 DeepSeek API 在“1 条 task / 1 次请求”模式下
    尽可能命中 Context Cache。

核心原则：
    DeepSeek 上下文缓存要求从第 0 token 开始具有相同前缀。因此，本工具把所有固定内容集中到文件开头：

        固定 Header / Protocol / Output Contract
        ------------------------------------------------  <- CACHEABLE FIXED PREFIX END
        TASK CONTEXT（从这里开始才允许 task_id / Issue / Candidate / Gold 等动态内容）

    这样直接把整个 .md 原样作为一个 user message 发送，也能让前面的大段固定 Prompt 形成一致前缀。
    后续若 API dispatcher 愿意进一步拆分，可把固定前缀作为 system message、TASK CONTEXT 作为 user message；
    本工具放置了稳定 marker，便于程序无歧义切分。

严格不做：
    - 不读取 / 修改 V2.10 dataset；
    - 不打开 build SQLite；
    - 不运行 Candidate Builder；
    - 不生成 requests.jsonl / merge_context.jsonl / tasks.jsonl / report；
    - 不修改 Candidate Pool / Issue / Original Supervision / Gold Hints；
    - 不修改 task_id；
    - 不修改文件名；
    - 不改变 Strong-Teacher 语义规则本身，只改变静态 / 动态块的排列位置和固定外壳措辞。

输入要求：
    - 每个 Markdown 恰好 1 个 TASK；
    - 当前文件是 v1.4 风格，至少包含：
        [UNIFIED STRONG-TEACHER PROTOCOL]
        [TASK CONTEXT]
        [UNIFIED STRONG-TEACHER OUTPUT]
    - 若结构不符合预期，Hard Fail，不猜测改写。

PowerShell：

    # 先检查，不写盘
    python scripts/refresh_strong_teacher_md_v1_5_cache_first.py `
      --root data/upstream/external_supervision/strong_teacher_v1_3_all `
      --dry-run

    # 正式修改
    python scripts/refresh_strong_teacher_md_v1_5_cache_first.py `
      --root data/upstream/external_supervision/strong_teacher_v1_3_all
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError as exc:
    raise RuntimeError("缺少 tqdm。请执行：python -m pip install -U tqdm") from exc


SCRIPT_VERSION = "1.5.0"
PROMPT_CONTRACT_VERSION = "cache-first-single-task-v1.5"

DEFAULT_ROOT = Path("data/upstream/external_supervision/strong_teacher_v1_3_all")
DEFAULT_SPLITS = ("train", "validation", "benchmark")

PROTOCOL_MARKER = "[UNIFIED STRONG-TEACHER PROTOCOL]"
TASK_CONTEXT_MARKER = "[TASK CONTEXT]"
OUTPUT_MARKER = "[UNIFIED STRONG-TEACHER OUTPUT]"
CACHE_END_MARKER = "[CACHEABLE FIXED PREFIX END]"
OLD_DYNAMIC_ID_PREFIX = "当前 task_id 必须原样复制为："

STATIC_HEADER = """# Unified Strong-Teacher Single Task

本文件恰好包含 1 个 TASK。
最终只返回一个 JSON array，且数组中恰好 1 个 object；不要 Markdown、不要代码块、不要额外解释。
所有解释/审计字段必须使用简体中文；机器字段和枚举保持规定的英文值。

重要：本文件前半部分是所有任务共用的固定 Strong-Teacher 指令；真正的 task-specific 内容从后面的任务上下文区域开始。必须完整阅读该区域，再生成结果。
""".strip()

# 固定 Output Contract 的尾注。旧版这里写死了当前 task_id，导致每个文件尾部不同；
# 新版改成静态规则，并依赖 TASK CONTEXT 中已有的 [TASK] 字段提供真实 task_id。
STATIC_TASK_ID_RULE = "当前 task_id 必须从后续任务上下文中的 [TASK] 字段原样复制；不得改写、截断或猜测。"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_single_task_id(task_context: str, path: Path) -> str:
    """只从动态 TASK CONTEXT 中读取 task_id，避免依赖旧 header。"""
    match = re.search(r"^\[TASK\]\s*\n([^\n]+)\s*$", task_context, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"{path}: TASK CONTEXT 中未找到标准 [TASK] + task_id")
    task_id = match.group(1).strip()
    if not task_id.startswith("task_"):
        raise ValueError(f"{path}: 非法 task_id：{task_id!r}")
    if len(re.findall(r"^\[TASK\]\s*$", task_context, flags=re.MULTILINE)) != 1:
        raise ValueError(f"{path}: 每个 MD 必须恰好 1 个 [TASK]")
    return task_id


def _strip_old_dynamic_task_id_from_output(output_contract: str, task_id: str, path: Path) -> str:
    """
    删除旧 Output Contract 尾部唯一的动态 task_id 行，替换成固定规则。

    这是提高缓存命中率的关键之一：固定 Output Contract 中不能残留任何 task-specific token。
    """
    pattern = re.compile(
        r"\n*当前 task_id 必须原样复制为：\s*" + re.escape(task_id) + r"\s*\Z",
        flags=re.MULTILINE,
    )
    stripped, count = pattern.subn("\n\n" + STATIC_TASK_ID_RULE + "\n", output_contract)
    if count != 1:
        raise ValueError(
            f"{path}: Output Contract 中旧动态 task_id 行替换次数={count}，期望=1"
        )
    return stripped.strip()


def transform_one(text: str, path: Path) -> tuple[str, dict[str, str | int]]:
    """
    v1.4 -> v1.5 Cache-First。

    关键保证：TASK CONTEXT 整块逐字符保持不变，只把它移动到所有固定规则之后。
    因此 Issue / Candidate Pool / Gold Hints / Original Supervision 等业务内容不会被修改。
    """
    if CACHE_END_MARKER in text:
        # 已是 v1.5：幂等返回，但仍由 validate_new_format() 做完整校验。
        return text, {"already_new": 1}

    protocol_pos = text.find(PROTOCOL_MARKER)
    task_pos = text.find(TASK_CONTEXT_MARKER)
    output_pos = text.find(OUTPUT_MARKER)
    if min(protocol_pos, task_pos, output_pos) < 0:
        raise ValueError(f"{path}: 缺少 v1.4 标准 marker")
    if not (protocol_pos < task_pos < output_pos):
        raise ValueError(
            f"{path}: marker 顺序异常；期望 PROTOCOL < TASK CONTEXT < OUTPUT"
        )

    # 固定 Protocol：从 protocol marker 到 TASK CONTEXT 之前。
    protocol_block = text[protocol_pos:task_pos].strip()

    # 动态 TASK CONTEXT：从 [TASK CONTEXT] 到 [UNIFIED OUTPUT] 之前。
    # 这是受保护区域，必须 byte-for-byte（这里按 Python str）保持一致。
    task_context = text[task_pos:output_pos].strip()
    task_id = _extract_single_task_id(task_context, path)

    # 固定 Output Contract：旧版位于 task-specific 内容之后；现在移到前面。
    output_contract = text[output_pos:].strip()
    output_contract = _strip_old_dynamic_task_id_from_output(output_contract, task_id, path)

    # Header 中不再出现 “TASK 1 — task_id”。所有动态 token 都推迟到 TASK CONTEXT。
    new_text = (
        STATIC_HEADER
        + "\n\n"
        + protocol_block
        + "\n\n"
        + output_contract
        + "\n\n"
        + "从下一行开始均为 task-specific 内容。不要将不同 TASK 的 Candidate Number 或语义相互混用。"
        + "\n"
        + CACHE_END_MARKER
        + "\n\n"
        + task_context
        + "\n"
    )

    stats: dict[str, str | int] = {
        "already_new": 0,
        "task_id": task_id,
        "protected_task_context_sha256": sha256_text(task_context),
        "fixed_prefix_chars": new_text.find(TASK_CONTEXT_MARKER),
        "total_chars": len(new_text),
    }
    return new_text, stats


def validate_new_format(path: Path, original: str, patched: str) -> dict[str, str | int | float]:
    """严格验证 Cache-First 合同。任何不确定情况直接拒绝写盘。"""
    if patched.count(PROTOCOL_MARKER) != 1:
        raise ValueError(f"{path}: PROTOCOL marker 数量不是 1")
    if patched.count(TASK_CONTEXT_MARKER) != 1:
        raise ValueError(f"{path}: TASK CONTEXT marker 数量不是 1")
    if patched.count(OUTPUT_MARKER) != 1:
        raise ValueError(f"{path}: OUTPUT marker 数量不是 1")
    if patched.count(CACHE_END_MARKER) != 1:
        raise ValueError(f"{path}: CACHE END marker 数量不是 1")

    p_protocol = patched.find(PROTOCOL_MARKER)
    p_output = patched.find(OUTPUT_MARKER)
    p_cache_end = patched.find(CACHE_END_MARKER)
    p_task = patched.find(TASK_CONTEXT_MARKER)
    if not (p_protocol < p_output < p_cache_end < p_task):
        raise ValueError(
            f"{path}: 新格式 marker 顺序异常；期望 PROTOCOL < OUTPUT < CACHE_END < TASK_CONTEXT"
        )

    task_context = patched[p_task:].strip()
    task_id = _extract_single_task_id(task_context, path)

    # 最关键的缓存约束：task_id 不得在固定前缀中出现。
    fixed_prefix = patched[:p_task]
    if task_id in fixed_prefix:
        raise ValueError(f"{path}: task_id 泄漏到 cacheable fixed prefix，缓存优化失败")
    if re.search(r"^TASK\s+\d+\s+—\s+task_", fixed_prefix, flags=re.MULTILINE):
        raise ValueError(f"{path}: 旧动态 TASK header 仍存在于固定前缀")
    if OLD_DYNAMIC_ID_PREFIX in fixed_prefix:
        raise ValueError(f"{path}: 固定前缀仍含旧动态 task_id 提示")

    # 保护 TASK CONTEXT：对 v1.4 输入，比较原始动态块；对已经 v1.5 的输入则自校验。
    if CACHE_END_MARKER not in original:
        old_task_pos = original.find(TASK_CONTEXT_MARKER)
        old_output_pos = original.find(OUTPUT_MARKER)
        if old_task_pos < 0 or old_output_pos < 0 or old_task_pos >= old_output_pos:
            raise ValueError(f"{path}: 无法提取原始 TASK CONTEXT 做保护校验")
        old_task_context = original[old_task_pos:old_output_pos].strip()
        if old_task_context != task_context:
            raise ValueError(
                f"{path}: TASK CONTEXT 发生变化；为防止污染 supervision，拒绝写盘"
            )
        old_hash = sha256_text(old_task_context)
    else:
        old_hash = sha256_text(task_context)

    fixed_chars = p_task
    total_chars = len(patched)
    return {
        "task_id": task_id,
        "task_context_sha256": old_hash,
        "fixed_prefix_chars": fixed_chars,
        "total_chars": total_chars,
        "fixed_prefix_char_ratio": fixed_chars / total_chars if total_chars else 0.0,
    }


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def collect_md_files(root: Path, splits: list[str]) -> list[Path]:
    result: list[Path] = []
    for split in splits:
        md_dir = root / split / "md"
        if not md_dir.is_dir():
            raise FileNotFoundError(f"缺少目录：{md_dir}")
        result.extend(sorted(md_dir.glob("*.md")))
    return result


def run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    paths = collect_md_files(root, list(args.splits))
    if not paths:
        raise ValueError(f"没有找到 Markdown：{root}")

    changed = 0
    already_new = 0
    failed: list[tuple[Path, str]] = []
    fixed_prefix_chars_total = 0
    total_chars_total = 0

    for path in tqdm(paths, desc="Cache-first Strong-Teacher MD", unit="md", dynamic_ncols=True):
        try:
            original = path.read_text(encoding="utf-8-sig")
            patched, stats = transform_one(original, path)
            validation = validate_new_format(path, original, patched)

            fixed_prefix_chars_total += int(validation["fixed_prefix_chars"])
            total_chars_total += int(validation["total_chars"])

            if int(stats.get("already_new", 0)) == 1:
                already_new += 1
                continue

            if patched == original:
                raise ValueError("预期需要升级，但内容没有变化")

            if not args.dry_run:
                atomic_write_text(path, patched)
            changed += 1
        except Exception as exc:
            failed.append((path, f"{type(exc).__name__}: {exc}"))

    ratio = (
        fixed_prefix_chars_total / total_chars_total
        if total_chars_total > 0
        else 0.0
    )

    print()
    print(f"script_version={SCRIPT_VERSION}")
    print(f"prompt_contract={PROMPT_CONTRACT_VERSION}")
    print(f"root={root}")
    print(f"scanned_md={len(paths)}")
    print(f"changed_md={changed}")
    print(f"already_new_md={already_new}")
    print(f"failed_md={len(failed)}")
    print(f"dry_run={bool(args.dry_run)}")
    print("persistent_sidecar_written=0")
    print(f"aggregate_fixed_prefix_char_ratio={ratio:.4f}")

    if failed:
        print("\n失败文件：")
        for path, message in failed[:100]:
            print(f"- {path}: {message}")
        if len(failed) > 100:
            print(f"... 其余 {len(failed) - 100} 个失败文件省略")
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "将 v1.4 单任务 Strong-Teacher Markdown 重排为 Cache-First 格式；"
            "只改 Prompt 布局，不改 TASK CONTEXT，不生成 sidecar。"
        )
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION} / {PROMPT_CONTRACT_VERSION}",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="默认 data/upstream/external_supervision/strong_teacher_v1_3_all",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(DEFAULT_SPLITS),
        default=list(DEFAULT_SPLITS),
        help="默认 train validation benchmark。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只验证和统计，不写盘。",
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
