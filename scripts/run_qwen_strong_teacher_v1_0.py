#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen Offline Strong-Teacher Runner
==================================

设计目标
--------
1. 1 个 Markdown TASK = 1 次 API 请求。
2. 输入来自 frozen Strong-Teacher MD；不修改输入文件。
3. 输出写入:
       data/.external_supervision/result/{train,validation,benchmark}/<same_filename>.md
4. 已存在且 size > 0 的结果文件：绝不覆盖，直接 skip。
5. 已存在且 size == 0 的结果文件：允许重新生成；只有 API + 校验全部成功后才原子覆盖。
6. 最终结果必须是纯 JSON array，且只包含当前 task_id 的一个 object。
7. Candidate Number 必须绑定当前 TASK Candidate Pool；越界或引用不存在 Candidate 时 hard fail。
8. 只保存最终 answer content，不保存 reasoning_content。
9. Qwen3.7-Max 05-17 为 only-thinking 模型。为了最大思考，不设置 thinking_budget；
   官方默认值即模型最大思维链长度。
10. 支持 cache-first MD 的 [CACHEABLE FIXED PREFIX END] marker：
    marker 前作为 system，marker 后作为 user。对不支持 cache 的模型不影响正确性；
    后续切换到支持 cache 的模型时可直接复用。

依赖
----
    pip install -U openai tqdm python-dotenv

环境变量
--------
    QWEN_API_KEY=...
    QWEN_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    QWEN_MODEL=qwen3.7-max-2026-05-17

注意：
- 变量名是 QWEN_API_URL，不是 QWEN_API_UEL。
- .env 中 URL 直接写裸 URL，不要写 Markdown 的 [text](url)。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        OpenAI,
        RateLimitError,
    )
except ImportError as exc:
    raise SystemExit(
        "缺少 openai SDK。请执行：pip install -U openai tqdm python-dotenv"
    ) from exc

try:
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "缺少 tqdm。请执行：pip install -U openai tqdm python-dotenv"
    ) from exc


SCRIPT_VERSION = "1.0.0"
CACHE_MARKER = "[CACHEABLE FIXED PREFIX END]"
DEFAULT_INPUT_ROOT = Path("data/.external_supervision/strong_teacher_v1_3_all")
DEFAULT_RESULT_ROOT = Path("data/.external_supervision/result")
DEFAULT_USAGE_LOG = Path(
    "data/.external_supervision/.run_logs/qwen_strong_teacher_usage.jsonl"
)
SPLITS = ("train", "validation", "benchmark")

# 输出 JSON 中所有 Candidate 引用都必须来自当前 TASK。
CANDIDATE_REF_FIELDS = (
    "supporting_candidates",
    "candidate_numbers",
)

# 七个固定 canonical slots。
CANONICAL_SLOTS = (
    "fault_location",
    "fault_logic",
    "dependency_context",
    "state_flow",
    "behavior_constraint",
    "repair_scope",
    "validation_constraint",
)

ENUMS = {
    "applicability": {"required", "not_required", "uncertain"},
    "question_coverage": {
        "sufficient",
        "partial",
        "none",
        "uncertain",
        "not_applicable",
    },
    "repository_need": {
        "required",
        "helpful",
        "not_needed",
        "uncertain",
        "not_applicable",
    },
    "candidate_pool_status": {
        "sufficient",
        "insufficient",
        "uncertain",
        "not_needed",
    },
}


@dataclass
class UsageRecord:
    timestamp_utc: str
    split: str
    filename: str
    task_id: str
    model: str
    status: str
    attempt: int
    elapsed_seconds: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    output_chars: Optional[int] = None
    error: Optional[str] = None


@dataclass
class RunStats:
    scanned: int = 0
    selected: int = 0
    skipped_nonempty: int = 0
    zero_byte_retries: int = 0
    succeeded: int = 0
    failed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_environment(env_file: Optional[Path]) -> None:
    """加载 .env；若未安装 python-dotenv，则仅使用系统环境变量。"""
    if load_dotenv is None:
        return
    if env_file is not None:
        load_dotenv(dotenv_path=env_file, override=False)
    else:
        load_dotenv(override=False)


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"缺少环境变量：{name}")
    return value


def normalize_base_url(url: str) -> str:
    """
    防止用户把 Markdown 链接形式误放到 .env:
        [https://...](https://...)
    这种情况直接 hard fail，而不是静默修复，避免把错误配置带入长任务。
    """
    if url.startswith("[") or "](" in url:
        raise SystemExit(
            "QWEN_API_URL 看起来是 Markdown 链接。"
            "请改成裸 URL，例如：https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    return url.rstrip("/")


def discover_tasks(input_root: Path, splits: Iterable[str]) -> list[tuple[str, Path]]:
    tasks: list[tuple[str, Path]] = []
    for split in splits:
        md_dir = input_root / split / "md"
        if not md_dir.is_dir():
            print(f"[WARN] 输入目录不存在，跳过：{md_dir}", file=sys.stderr)
            continue
        for path in sorted(md_dir.glob("*.md")):
            if path.is_file():
                tasks.append((split, path))
    return tasks


def extract_task_id(md_text: str) -> str:
    """
    task_id 必须在 MD 中唯一、明确。
    支持 cache-first 后的:
        [TASK]
        task_xxx
    也兼容旧格式:
        TASK 1 — task_xxx
    """
    patterns = (
        r"(?m)^\[TASK\]\s*\n(?P<id>task_[A-Za-z0-9]+)\s*$",
        r"(?m)^TASK\s+\d+\s+[—-]\s+(?P<id>task_[A-Za-z0-9]+)\s*$",
        r"(?m)^当前 task_id 必须原样复制为：(?P<id>task_[A-Za-z0-9]+)\s*$",
    )
    found: list[str] = []
    for pattern in patterns:
        found.extend(m.group("id") for m in re.finditer(pattern, md_text))

    unique = sorted(set(found))
    if len(unique) != 1:
        raise ValueError(f"无法唯一确定 task_id，检测到：{unique}")
    return unique[0]


def extract_candidate_numbers(md_text: str) -> set[int]:
    """
    只从 [CANDIDATE N] 标题提取合法 Candidate Number。
    不能用任意正文数字，否则会把行号、版本号、Gold hunk 行号误当 Candidate。
    """
    nums = {
        int(x)
        for x in re.findall(r"(?m)^\[CANDIDATE\s+(\d+)\]", md_text)
    }
    if not nums:
        raise ValueError("Candidate Evidence Pool 中未检测到任何 [CANDIDATE N]")
    return nums


def split_prompt(md_text: str, mode: str) -> list[dict[str, str]]:
    """
    split:
      marker 前 = system，marker 后 = user。
      这样固定协议和 task-specific context 在消息层面严格分离。

    whole:
      整份 MD 作为单个 user message，用于和旧实验做 A/B 对照。
    """
    if mode == "whole":
        return [{"role": "user", "content": md_text}]

    if CACHE_MARKER not in md_text:
        # 没有 marker 时不猜边界，退回 whole，避免错误切割业务内容。
        return [{"role": "user", "content": md_text}]

    prefix, suffix = md_text.split(CACHE_MARKER, 1)
    prefix = prefix.rstrip()
    suffix = suffix.lstrip()

    if not prefix or not suffix:
        raise ValueError(f"{CACHE_MARKER} 前后内容不完整")

    return [
        {"role": "system", "content": prefix},
        {"role": "user", "content": suffix},
    ]


def strip_accidental_fence(text: str) -> str:
    """
    Teacher 协议要求纯 JSON。这里仅容忍最常见的机械代码围栏，
    不做任何语义修复、不补 task_id、不改 Candidate Number。
    """
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def ensure_int_list(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} 必须是 list")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in value):
        raise ValueError(f"{field_name} 只能包含整数 Candidate Number")
    return value


def validate_candidate_refs(
    values: Iterable[int],
    legal_candidates: set[int],
    where: str,
) -> None:
    illegal = sorted(set(values) - legal_candidates)
    if illegal:
        raise ValueError(f"{where} 引用了不存在的 Candidate Number：{illegal}")


def validate_slot(
    slot_name: str,
    slot: Any,
    legal_candidates: set[int],
) -> None:
    if not isinstance(slot, dict):
        raise ValueError(f"slot {slot_name} 必须是 object")

    required_fields = {
        "applicability",
        "question_coverage",
        "repository_need",
        "candidate_pool_status",
        "sufficient_witness_groups",
        "supporting_candidates",
        "reason",
    }
    missing = required_fields - set(slot)
    if missing:
        raise ValueError(f"slot {slot_name} 缺字段：{sorted(missing)}")

    for field, allowed in ENUMS.items():
        value = slot.get(field)
        if value not in allowed:
            raise ValueError(
                f"slot {slot_name}.{field} 非法枚举：{value!r}；"
                f"允许值={sorted(allowed)}"
            )

    if not isinstance(slot["reason"], str) or not slot["reason"].strip():
        raise ValueError(f"slot {slot_name}.reason 必须是非空字符串")

    groups = slot["sufficient_witness_groups"]
    if not isinstance(groups, list):
        raise ValueError(
            f"slot {slot_name}.sufficient_witness_groups 必须是 OR-of-AND list"
        )

    flattened: list[int] = []
    for i, group in enumerate(groups):
        ints = ensure_int_list(
            group,
            f"slot {slot_name}.sufficient_witness_groups[{i}]",
        )
        if not ints:
            raise ValueError(
                f"slot {slot_name}.sufficient_witness_groups[{i}] 不能是空 AND group"
            )
        if len(ints) != len(set(ints)):
            raise ValueError(
                f"slot {slot_name}.sufficient_witness_groups[{i}] 内 Candidate 重复"
            )
        flattened.extend(ints)

    validate_candidate_refs(
        flattened,
        legal_candidates,
        f"slot {slot_name}.sufficient_witness_groups",
    )

    supporting = ensure_int_list(
        slot["supporting_candidates"],
        f"slot {slot_name}.supporting_candidates",
    )
    validate_candidate_refs(
        supporting,
        legal_candidates,
        f"slot {slot_name}.supporting_candidates",
    )

    # 结构一致性硬约束：repository 不需要时不能声称 mandatory witness。
    if slot["repository_need"] in {"not_needed", "not_applicable", "helpful"} and groups:
        raise ValueError(
            f"slot {slot_name}: repository_need={slot['repository_need']} "
            "却存在 sufficient_witness_groups"
        )

    if slot["repository_need"] == "required":
        status = slot["candidate_pool_status"]
        if status == "not_needed":
            raise ValueError(
                f"slot {slot_name}: repository_need=required，"
                "candidate_pool_status 不能是 not_needed"
            )
        if status == "sufficient" and not groups:
            raise ValueError(
                f"slot {slot_name}: repository_need=required + "
                "candidate_pool_status=sufficient，但 Witness 为空"
            )
        if status in {"insufficient", "uncertain"} and groups:
            raise ValueError(
                f"slot {slot_name}: candidate_pool_status={status} "
                "却存在 sufficient_witness_groups"
            )


def validate_output(
    raw_content: str,
    expected_task_id: str,
    legal_candidates: set[int],
) -> str:
    """
    返回 canonical compact JSON string。
    这里只做机械规范化（空白/缩进），绝不改 Teacher 的语义标签。
    """
    clean = strip_accidental_fence(raw_content)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"模型输出不是合法 JSON：line={exc.lineno}, col={exc.colno}, {exc.msg}"
        ) from exc

    if not isinstance(data, list):
        raise ValueError("顶层必须是 JSON array")
    if len(data) != 1:
        raise ValueError(f"单任务请求必须恰好返回 1 个 object，实际={len(data)}")

    obj = data[0]
    if not isinstance(obj, dict):
        raise ValueError("JSON array 的唯一元素必须是 object")

    if obj.get("task_id") != expected_task_id:
        raise ValueError(
            f"task_id 不匹配：expected={expected_task_id}, got={obj.get('task_id')!r}"
        )

    slots = obj.get("slots")
    if not isinstance(slots, dict):
        raise ValueError("缺少 slots object")

    missing_slots = set(CANONICAL_SLOTS) - set(slots)
    extra_slots = set(slots) - set(CANONICAL_SLOTS)
    if missing_slots or extra_slots:
        raise ValueError(
            f"slots 不符合 canonical 7 dimensions；"
            f"missing={sorted(missing_slots)}, extra={sorted(extra_slots)}"
        )

    for slot_name in CANONICAL_SLOTS:
        validate_slot(slot_name, slots[slot_name], legal_candidates)

    findings = obj.get("additional_findings")
    if not isinstance(findings, list):
        raise ValueError("additional_findings 必须是 list")
    for i, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise ValueError(f"additional_findings[{i}] 必须是 object")
        for field in ("description", "candidate_numbers", "reason"):
            if field not in finding:
                raise ValueError(f"additional_findings[{i}] 缺字段 {field}")
        nums = ensure_int_list(
            finding["candidate_numbers"],
            f"additional_findings[{i}].candidate_numbers",
        )
        validate_candidate_refs(
            nums,
            legal_candidates,
            f"additional_findings[{i}].candidate_numbers",
        )

    uncertainties = obj.get("uncertainties")
    if not isinstance(uncertainties, list) or any(
        not isinstance(x, str) for x in uncertainties
    ):
        raise ValueError("uncertainties 必须是字符串 list")

    if not isinstance(obj.get("overall_assessment"), str):
        raise ValueError("overall_assessment 必须是字符串")

    # 中文解释字段是否真的为中文属于语义/语言质量问题，
    # 这里不靠脆弱 regex 自动改写；后续可由 Critic/人工抽检处理。

    # 保持可人工审计的两空格缩进，并确保文件以换行结束。
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def get_reasoning_tokens(usage: Any) -> Optional[int]:
    if usage is None:
        return None

    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        value = getattr(details, "reasoning_tokens", None)
        if isinstance(value, int):
            return value

    # DashScope/OpenAI 兼容实现有时可能通过 output_tokens_details 暴露。
    details = getattr(usage, "output_tokens_details", None)
    if details is not None:
        value = getattr(details, "reasoning_tokens", None)
        if isinstance(value, int):
            return value

    return None


def collect_stream_response(stream: Any) -> tuple[str, Any]:
    """
    流式调用可避免超长深度思考时长时间无响应。
    reasoning_content 只计数/丢弃，不写入监督文件。
    """
    answer_parts: list[str] = []
    usage = None

    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue

        delta = choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            answer_parts.append(content)

        # 故意不保存 reasoning_content。
        # 强教师监督只落最终结构化结论，避免把私有/冗长推理链写入数据集。
        _ = getattr(delta, "reasoning_content", None)

    return "".join(answer_parts), usage


def append_jsonl(path: Path, record: UsageRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(record), ensure_ascii=False)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
        f.flush()


def atomic_write_text(path: Path, content: str) -> None:
    """
    原子写入：
    - 不提前 truncate 目标文件；
    - 只有完整内容已 fsync 到临时文件后才 os.replace；
    - 因此 API/解析失败不会破坏已有 0-byte 占位或其它状态。
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def is_quota_or_billing_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    needles = (
        "quota",
        "insufficient balance",
        "insufficient_balance",
        "balance not enough",
        "free quota",
        "account arrears",
        "arrears",
    )
    return any(x in text for x in needles)


def should_retry(exc: BaseException) -> bool:
    if is_quota_or_billing_error(exc):
        return False
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        return status in {408, 409, 429, 500, 502, 503, 504}
    return False


def call_teacher(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
) -> tuple[str, Any]:
    """
    qwen3.7-max-2026-05-17 是 only-thinking：
    - 不传 enable_thinking；
    - 不传 thinking_budget；
    - thinking_budget 默认即模型最大思维链长度。
    """
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    return collect_stream_response(stream)


def usage_int(usage: Any, field: str) -> Optional[int]:
    if usage is None:
        return None
    value = getattr(usage, field, None)
    return value if isinstance(value, int) else None


def make_output_path(
    result_root: Path,
    split: str,
    input_md: Path,
) -> Path:
    return result_root / split / input_md.name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen Strong-Teacher runner：1 task/request，非空结果绝不覆盖。"
    )
    p.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Strong-Teacher export root，默认：{DEFAULT_INPUT_ROOT}",
    )
    p.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help=f"结果根目录，默认：{DEFAULT_RESULT_ROOT}",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="要运行的 split。",
    )
    p.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="env 文件；默认 .env。",
    )
    p.add_argument(
        "--usage-log",
        type=Path,
        default=DEFAULT_USAGE_LOG,
        help=f"usage/error JSONL，默认：{DEFAULT_USAGE_LOG}",
    )
    p.add_argument(
        "--message-mode",
        choices=("split", "whole"),
        default="split",
        help="split=按 cache marker 分 system/user；whole=整份 MD 作为 user。",
    )
    p.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="最多实际请求多少条；0=不限。用于先小规模试跑。",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="每个 task 最大尝试次数，默认 3。",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="OpenAI client timeout（秒），默认 1800。",
    )
    p.add_argument(
        "--sleep-between",
        type=float,
        default=0.2,
        help="成功请求之间最少等待秒数，默认 0.2。",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="随机打乱待运行任务。默认按 split/文件名顺序。",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260812,
        help="--shuffle 的随机种子。",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只扫描并报告 skip / zero-byte / pending，不调用 API。",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="任一 task 最终失败后立即停止；默认记录失败并继续。",
    )
    p.add_argument(
        "--version",
        action="version",
        version=SCRIPT_VERSION,
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_environment(args.env_file)

    api_key = require_env("QWEN_API_KEY")
    base_url = normalize_base_url(require_env("QWEN_API_URL"))
    model = require_env("QWEN_MODEL")

    tasks = discover_tasks(args.input_root, args.splits)
    if not tasks:
        print("没有发现任何输入 MD。", file=sys.stderr)
        return 2

    stats = RunStats(scanned=len(tasks))

    pending: list[tuple[str, Path, Path]] = []
    for split, input_md in tasks:
        output_md = make_output_path(args.result_root, split, input_md)

        if output_md.exists():
            size = output_md.stat().st_size
            if size > 0:
                # 用户明确要求：之前已经跑过的非空结果绝不覆盖。
                stats.skipped_nonempty += 1
                continue

            # 只有严格的 0 byte 才允许覆盖。
            # whitespace-only 但 size>0 的文件依然属于“非空”，不会碰。
            stats.zero_byte_retries += 1

        pending.append((split, input_md, output_md))

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(pending)

    if args.max_tasks > 0:
        pending = pending[: args.max_tasks]

    stats.selected = len(pending)

    print(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "model": model,
                "input_root": str(args.input_root),
                "result_root": str(args.result_root),
                "scanned": stats.scanned,
                "skipped_nonempty": stats.skipped_nonempty,
                "zero_byte_retries": stats.zero_byte_retries,
                "selected_for_request": stats.selected,
                "message_mode": args.message_mode,
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.dry_run or not pending:
        return 0

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=args.timeout,
        # SDK 自带 retries 与我们自己的 task-level retry 叠加会让次数失控，
        # 因此关闭 SDK 自动重试，统一由下面的语义感知 retry 管理。
        max_retries=0,
    )

    progress = tqdm(
        pending,
        total=len(pending),
        desc="Qwen Strong-Teacher",
        unit="task",
        dynamic_ncols=True,
    )

    for split, input_md, output_md in progress:
        task_id = "<unknown>"
        legal_candidates: set[int] = set()

        try:
            md_text = input_md.read_text(encoding="utf-8")
            if not md_text.strip():
                raise ValueError("输入 MD 为空")

            task_id = extract_task_id(md_text)
            legal_candidates = extract_candidate_numbers(md_text)
            messages = split_prompt(md_text, args.message_mode)
        except Exception as exc:
            stats.failed += 1
            append_jsonl(
                args.usage_log,
                UsageRecord(
                    timestamp_utc=utc_now_iso(),
                    split=split,
                    filename=input_md.name,
                    task_id=task_id,
                    model=model,
                    status="PREPARE_FAILED",
                    attempt=0,
                    elapsed_seconds=0.0,
                    error=str(exc),
                ),
            )
            tqdm.write(f"[PREPARE_FAILED] {input_md}: {exc}")
            if args.stop_on_error:
                return 1
            continue

        success = False
        last_error: Optional[BaseException] = None

        for attempt in range(1, args.max_retries + 1):
            started = time.perf_counter()
            usage = None
            raw_content = ""

            try:
                raw_content, usage = call_teacher(
                    client=client,
                    model=model,
                    messages=messages,
                )

                if not raw_content.strip():
                    raise ValueError("模型最终 content 为空")

                canonical_json = validate_output(
                    raw_content=raw_content,
                    expected_task_id=task_id,
                    legal_candidates=legal_candidates,
                )

                # 直到这里才写结果。非空旧文件此前已被过滤，0-byte 可原子覆盖。
                atomic_write_text(output_md, canonical_json)

                elapsed = time.perf_counter() - started
                prompt_tokens = usage_int(usage, "prompt_tokens")
                completion_tokens = usage_int(usage, "completion_tokens")
                total_tokens = usage_int(usage, "total_tokens")
                reasoning_tokens = get_reasoning_tokens(usage)

                stats.succeeded += 1
                stats.prompt_tokens += prompt_tokens or 0
                stats.completion_tokens += completion_tokens or 0
                stats.total_tokens += total_tokens or 0
                stats.reasoning_tokens += reasoning_tokens or 0

                append_jsonl(
                    args.usage_log,
                    UsageRecord(
                        timestamp_utc=utc_now_iso(),
                        split=split,
                        filename=input_md.name,
                        task_id=task_id,
                        model=model,
                        status="SUCCESS",
                        attempt=attempt,
                        elapsed_seconds=round(elapsed, 3),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        reasoning_tokens=reasoning_tokens,
                        output_chars=len(canonical_json),
                    ),
                )

                progress.set_postfix(
                    ok=stats.succeeded,
                    fail=stats.failed,
                    tok=stats.total_tokens,
                    reason=stats.reasoning_tokens,
                )

                success = True
                time.sleep(max(args.sleep_between, 0.0))
                break

            except Exception as exc:
                elapsed = time.perf_counter() - started
                last_error = exc

                prompt_tokens = usage_int(usage, "prompt_tokens")
                completion_tokens = usage_int(usage, "completion_tokens")
                total_tokens = usage_int(usage, "total_tokens")
                reasoning_tokens = get_reasoning_tokens(usage)

                append_jsonl(
                    args.usage_log,
                    UsageRecord(
                        timestamp_utc=utc_now_iso(),
                        split=split,
                        filename=input_md.name,
                        task_id=task_id,
                        model=model,
                        status="ATTEMPT_FAILED",
                        attempt=attempt,
                        elapsed_seconds=round(elapsed, 3),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        reasoning_tokens=reasoning_tokens,
                        output_chars=len(raw_content) if raw_content else 0,
                        error=str(exc),
                    ),
                )

                retryable = should_retry(exc)

                # JSON/schema/task_id/Candidate-binding 失败也允许重试：
                # 这是模型输出质量失败，而不是 API 网络失败。
                validation_failure = isinstance(exc, ValueError)
                may_retry = (
                    attempt < args.max_retries
                    and (retryable or validation_failure)
                    and not is_quota_or_billing_error(exc)
                )

                if may_retry:
                    # 指数退避 + jitter。429/5xx 不高频轰炸接口。
                    delay = min(2 ** (attempt - 1), 20) + random.random()
                    tqdm.write(
                        f"[RETRY {attempt}/{args.max_retries}] "
                        f"{task_id} | {type(exc).__name__}: {exc} | "
                        f"sleep={delay:.1f}s"
                    )
                    time.sleep(delay)
                    continue

                break

        if not success:
            stats.failed += 1
            error_text = str(last_error) if last_error else "unknown failure"
            tqdm.write(f"[FAILED] {task_id} | {error_text}")

            # 配额/余额错误直接停止，避免无意义地把剩余 2 万任务全部打一遍失败请求。
            if last_error is not None and is_quota_or_billing_error(last_error):
                tqdm.write("[STOP] 检测到 quota/balance 类错误，停止后续请求。")
                break

            if args.stop_on_error:
                return 1

    print(
        "\n" +
        json.dumps(
            {
                "finished": True,
                "model": model,
                "scanned": stats.scanned,
                "selected": stats.selected,
                "skipped_nonempty": stats.skipped_nonempty,
                "zero_byte_retries": stats.zero_byte_retries,
                "succeeded": stats.succeeded,
                "failed": stats.failed,
                "usage": {
                    "prompt_tokens": stats.prompt_tokens,
                    "completion_tokens": stats.completion_tokens,
                    "reasoning_tokens": stats.reasoning_tokens,
                    "total_tokens": stats.total_tokens,
                },
                "usage_log": str(args.usage_log),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
