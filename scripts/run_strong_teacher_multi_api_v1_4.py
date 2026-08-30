#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-API Offline Strong-Teacher Runner
=======================================

版本: 1.4.0

核心契约
--------
1. 1 个 Markdown TASK = 1 次模型请求（失败重试除外）。
2. 输入来自 frozen Strong-Teacher MD；绝不修改输入文件。
3. 输出写入:
       data/upstream/external_supervision/result/{train,validation,benchmark}/<same_filename>.md
4. 已存在且 size > 0 的结果文件：绝不覆盖，直接 skip。
5. 已存在且 size == 0 的结果文件：允许重新生成；只有 API + 校验全部成功后才原子覆盖。
6. 最终结果必须是纯 JSON array，且只包含当前 task_id 的一个 object。
7. Candidate Number 必须绑定当前 TASK Candidate Pool；引用不存在 Candidate 时 hard fail。
8. 只保存最终 answer content，不保存 reasoning_content。
9. 支持 Qwen 与 TokenRhythm(LIN) 两种 OpenAI-compatible provider。
10. LIN 自动扫描 lin_API_KEY / lin_API_KEY_N（任意数字后缀）组成 key pool，并支持多线程并发。
11. 某个 LIN key quota/balance 用尽时仅禁用该 key；其余 key 继续工作。
12. 429 / 网络错误 / 5xx 会重试，并在可用 key 之间轮换。
13. 支持 cache-first MD 的 [CACHEABLE FIXED PREFIX END] marker：
    marker 前作为 system，marker 后作为 user。

依赖
----
    pip install -U openai tqdm python-dotenv

环境变量
--------

Qwen:
    QWEN_API_KEY=...
    QWEN_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
    QWEN_MODEL=qwen3.7-max-2026-05-17

TokenRhythm / LIN:
    lin_API_KEY=...
    lin_API_KEY_1=...
    lin_API_KEY_2=...
    lin_API_KEY_3=...
    # ...可继续任意数字后缀，编号允许不连续。
    lin_API_URL=https://tokenrhythm.studio/v1/chat/completions
    LIN_MODEL=deepseek-v4-flash-0731

兼容大写形式:
    LIN_API_KEY
    LIN_API_KEY_1
    LIN_API_KEY_2
    LIN_API_KEY_3
    LIN_API_URL

说明
----
- TokenRhythm 官方 OpenAI-compatible SDK base_url 是 https://tokenrhythm.studio/v1。
  如果 .env 填的是完整 /v1/chat/completions，本脚本会自动截为 /v1。
- 如果 URL 被误写成 Markdown 链接:
      [https://...](https://...)
  本脚本会自动提取真实 URL。
- 对 LIN 默认传 reasoning_effort="max"。
  如果该具体模型/路由不支持此参数，请显式使用:
      --lin-reasoning-effort none
  本脚本不会在 400 后静默降级思考强度。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


SCRIPT_VERSION = "1.4.0"
CACHE_MARKER = "[CACHEABLE FIXED PREFIX END]"
DEFAULT_INPUT_ROOT = Path("data/upstream/external_supervision/strong_teacher_v1_3_all")
DEFAULT_RESULT_ROOT = Path("data/upstream/external_supervision/result")
DEFAULT_USAGE_LOG = Path(
    "data/upstream/external_supervision/.run_logs/strong_teacher_multi_api_usage.jsonl"
)
SPLITS = ("train", "validation", "benchmark")

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


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    model: str
    base_url: str
    api_keys: tuple[str, ...]
    reasoning_effort: Optional[str]


@dataclass
class UsageRecord:
    timestamp_utc: str
    provider: str
    key_slot: Optional[int]
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
class TaskResult:
    split: str
    filename: str
    task_id: str
    success: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    error: Optional[str] = None
    all_keys_exhausted: bool = False


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


class KeyPool:
    """
    多 key 严格租约池（strict per-key single concurrency）。

    并发契约：
    - 一个 API key 同一时刻最多只能被一个请求持有；
    - acquire_key() 获取 key 时会把该 slot 标记为 in_use；
    - 所有可用 key 都在使用中时，额外 worker 会等待，而不是复用正在工作的 key；
    - release() 后该 key 才能被下一个请求使用；
    - quota/balance 明确耗尽时 disable()，该 key 永久退出本次进程；
    - 429/网络错误不会永久禁用 key，只在当前 attempt 结束后释放并参与后续轮换；
    - 所有 key 都 disabled 时 acquire_key() 返回 None。

    因此：
        4 个有效 key + --concurrency 20
    实际最多仍只有 4 个 API 请求同时进行，并且严格一 key 一请求。
    """

    def __init__(self, keys: Iterable[str]):
        self._keys = list(keys)
        if not self._keys:
            raise ValueError("KeyPool 至少需要一个 API key")

        self._enabled = [True] * len(self._keys)
        self._in_use = [False] * len(self._keys)
        self._cursor = 0

        # Condition 同时负责状态互斥和“等待空闲 key”通知。
        self._cond = threading.Condition()

    @property
    def size(self) -> int:
        return len(self._keys)

    def active_count(self) -> int:
        """仍未因 quota/balance 被禁用的 key 数量。"""
        with self._cond:
            return sum(self._enabled)

    def in_use_count(self) -> int:
        """当前正在被请求占用的 key 数量。"""
        with self._cond:
            return sum(
                enabled and in_use
                for enabled, in_use in zip(self._enabled, self._in_use)
            )

    def acquire_key(self) -> Optional[tuple[int, str]]:
        """
        阻塞获取一个“enabled 且 not in_use”的 key。

        返回:
            (key_slot, api_key)
        或:
            None —— 所有 key 都已被永久禁用。

        这里不能在所有 key 正忙时返回 None；
        “正忙”不是“耗尽”，必须等待 release()。
        """
        with self._cond:
            n = len(self._keys)

            while True:
                if not any(self._enabled):
                    return None

                for _ in range(n):
                    idx = self._cursor % n
                    self._cursor = (self._cursor + 1) % n

                    if self._enabled[idx] and not self._in_use[idx]:
                        self._in_use[idx] = True
                        return idx, self._keys[idx]

                # 至少还有 enabled key，但它们当前都在工作。
                # 等待任一请求 release()/disable() 后重新检查。
                self._cond.wait()

    def release(self, idx: int) -> None:
        """
        释放一个 key 的当前租约。
        即使该 key 在请求过程中被 disable，也必须 release，
        以保证等待线程被唤醒并重新判断池状态。
        """
        with self._cond:
            if 0 <= idx < len(self._in_use):
                self._in_use[idx] = False
            self._cond.notify_all()

    def disable(self, idx: int) -> None:
        """永久禁用当前进程中的某个 key（仅 quota/balance 类错误使用）。"""
        with self._cond:
            if 0 <= idx < len(self._enabled):
                self._enabled[idx] = False
            self._cond.notify_all()

    def status(self) -> list[dict[str, Any]]:
        with self._cond:
            return [
                {
                    "slot": i,
                    "enabled": self._enabled[i],
                    "in_use": self._in_use[i],
                }
                for i in range(len(self._keys))
            ]


_thread_local = threading.local()
_log_lock = threading.Lock()
_print_lock = threading.Lock()


def safe_tqdm_write(text: str) -> None:
    with _print_lock:
        tqdm.write(text)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_environment(env_file: Optional[Path]) -> None:
    if load_dotenv is None:
        return
    if env_file is not None:
        load_dotenv(dotenv_path=env_file, override=False)
    else:
        load_dotenv(override=False)


def getenv_first(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def require_env_any(*names: str) -> str:
    value = getenv_first(*names)
    if value is None:
        raise SystemExit(f"缺少环境变量之一：{', '.join(names)}")
    return value


def discover_lin_api_keys() -> tuple[str, ...]:
    """
    自动发现所有 LIN API Key。

    支持的变量名：
        lin_API_KEY
        lin_API_KEY_1
        lin_API_KEY_2
        lin_API_KEY_3
        ...

    同时兼容全大写形式 ``LIN_API_KEY`` / ``LIN_API_KEY_N``。

    设计契约：
    1. 数字后缀没有上限，也不要求连续；只有 ``lin_API_KEY_3`` 也可以启动。
    2. 主 key（无后缀）排序在最前；其后按数字后缀升序进入 key pool。
    3. 空值自动忽略，因此 .env 中可以把失效 key 注释掉或留空。
    4. 如果多个变量意外保存了完全相同的 key，只保留一次，避免同一个额度被
       伪装成多个独立 key slot。
    5. 只接受 ``lin_API_KEY`` 或 ``lin_API_KEY_<纯数字>``；类似
       ``lin_API_KEY_backup`` 不会被误识别。
    """
    pattern = re.compile(r"^lin_API_KEY(?:_(\d+))?$", re.IGNORECASE)
    discovered: list[tuple[int, str, str]] = []

    for env_name, raw_value in os.environ.items():
        match = pattern.fullmatch(env_name)
        if match is None:
            continue

        value = raw_value.strip()
        if not value:
            continue

        # 无后缀主 key 使用 -1，确保排在 _1、_2、_3... 之前。
        suffix = -1 if match.group(1) is None else int(match.group(1))
        discovered.append((suffix, env_name.lower(), value))

    # 先按数字槽位，再按变量名排序，保证不同平台上的结果可复现。
    discovered.sort(key=lambda item: (item[0], item[1]))

    keys: list[str] = []
    seen_values: set[str] = set()
    for _suffix, _env_name, value in discovered:
        if value in seen_values:
            continue
        seen_values.add(value)
        keys.append(value)

    if not keys:
        raise SystemExit(
            "未发现任何 LIN API key。请在 .env 中设置 "
            "lin_API_KEY 或 lin_API_KEY_<数字>（例如 lin_API_KEY_3）。"
        )

    return tuple(keys)


def normalize_url_value(raw: str) -> str:
    """
    兼容两种输入:
      https://example/v1
      [https://example/v1](https://example/v1)
    """
    value = raw.strip()

    md = re.fullmatch(r"\[([^\]]+)\]\((https?://[^)]+)\)", value)
    if md:
        value = md.group(2)

    if not re.match(r"^https?://", value, flags=re.I):
        raise SystemExit(f"API URL 非法：{raw!r}")

    return value.rstrip("/")


def normalize_openai_base_url(raw: str) -> str:
    """
    OpenAI SDK 需要 base_url，而不是完整 chat/completions endpoint。

    例如:
      https://tokenrhythm.studio/v1/chat/completions
        -> https://tokenrhythm.studio/v1

      https://dashscope.aliyuncs.com/compatible-mode/v1
        -> 原样保留
    """
    value = normalize_url_value(raw)

    for suffix in ("/chat/completions", "/completions"):
        if value.endswith(suffix):
            value = value[: -len(suffix)].rstrip("/")
            break

    return value


def load_provider_config(args: argparse.Namespace) -> ProviderConfig:
    if args.provider == "qwen":
        api_key = require_env_any("QWEN_API_KEY")
        api_url = require_env_any("QWEN_API_URL")
        model = require_env_any("QWEN_MODEL")
        return ProviderConfig(
            name="qwen",
            model=model,
            base_url=normalize_openai_base_url(api_url),
            api_keys=(api_key,),
            reasoning_effort=None,
        )

    if args.provider == "lin":
        # 自动扫描任意数量的 lin_API_KEY / lin_API_KEY_N。
        # 因此新增 key 时只需要改 .env，不再需要修改 Python 脚本。
        keys = discover_lin_api_keys()

        api_url = require_env_any("lin_API_URL", "LIN_API_URL")
        model = require_env_any("LIN_MODEL", "lin_MODEL")

        reasoning_effort: Optional[str]
        if args.lin_reasoning_effort == "none":
            reasoning_effort = None
        else:
            reasoning_effort = args.lin_reasoning_effort

        return ProviderConfig(
            name="lin",
            model=model,
            base_url=normalize_openai_base_url(api_url),
            api_keys=keys,
            reasoning_effort=reasoning_effort,
        )

    raise AssertionError(f"未知 provider: {args.provider}")


def get_openai_client(
    provider: ProviderConfig,
    key_slot: int,
    api_key: str,
    timeout: float,
) -> OpenAI:
    """
    每个线程缓存自己的 client。
    OpenAI client 内部 HTTP connection pool 不跨线程共享，减少并发争用。
    """
    cache = getattr(_thread_local, "clients", None)
    if cache is None:
        cache = {}
        _thread_local.clients = cache

    cache_key = (provider.name, key_slot, provider.base_url, timeout)
    client = cache.get(cache_key)
    if client is None:
        client = OpenAI(
            api_key=api_key,
            base_url=provider.base_url,
            timeout=timeout,
            # task-level retry 统一由本脚本管理，避免 SDK retry 与外层叠加。
            max_retries=0,
        )
        cache[cache_key] = client
    return client


def discover_tasks(
    input_root: Path,
    splits: Iterable[str],
) -> list[tuple[str, Path]]:
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
    nums = {
        int(x)
        for x in re.findall(r"(?m)^\[CANDIDATE\s+(\d+)\]", md_text)
    }
    if not nums:
        raise ValueError("Candidate Evidence Pool 中未检测到任何 [CANDIDATE N]")
    return nums


def split_prompt(md_text: str, mode: str) -> list[dict[str, str]]:
    if mode == "whole":
        return [{"role": "user", "content": md_text}]

    if CACHE_MARKER not in md_text:
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

    if slot["repository_need"] in {"not_needed", "not_applicable", "helpful"} and groups:
        raise ValueError(
            f"slot {slot_name}: repository_need={slot['repository_need']} "
            "却存在 sufficient_witness_groups"
        )

    if slot["repository_need"] == "required":
        status = slot["candidate_pool_status"]
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

    # Mechanical normalization（机械归一化）：
    # uncertainties 的业务 Schema 是“字符串数组”。部分模型会机械性地输出：
    #   null
    #   ""
    #   "某个不确定性"
    # 这些形式都能无歧义地规范化，而不改变 Witness、AND/OR、slot 或语义判断：
    #   null / "" -> []
    #   "text"    -> ["text"]
    #
    # 重要边界：
    # - object / number / mixed-type list 不做猜测，仍然 hard fail；
    # - 不把 dict 自动抽取 description/reason，因为那会变成语义修复；
    # - 该归一化仅用于避免为纯格式偏差重新消耗一次大模型请求。
    uncertainties = obj.get("uncertainties")
    if uncertainties is None:
        obj["uncertainties"] = []
    elif isinstance(uncertainties, str):
        normalized_uncertainty = uncertainties.strip()
        obj["uncertainties"] = (
            [normalized_uncertainty] if normalized_uncertainty else []
        )
    elif isinstance(uncertainties, list):
        if any(not isinstance(x, str) for x in uncertainties):
            raise ValueError(
                "uncertainties 必须是字符串 list；"
                "仅 null / 空字符串 / 单个字符串允许机械归一化"
            )
    else:
        raise ValueError(
            "uncertainties 必须是字符串 list；"
            "仅 null / 空字符串 / 单个字符串允许机械归一化"
        )

    if not isinstance(obj.get("overall_assessment"), str):
        raise ValueError("overall_assessment 必须是字符串")

    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


def get_reasoning_tokens(usage: Any) -> Optional[int]:
    if usage is None:
        return None

    for attr in ("completion_tokens_details", "output_tokens_details"):
        details = getattr(usage, attr, None)
        if details is not None:
            value = getattr(details, "reasoning_tokens", None)
            if isinstance(value, int):
                return value
    return None


def usage_int(usage: Any, field: str) -> Optional[int]:
    if usage is None:
        return None
    value = getattr(usage, field, None)
    return value if isinstance(value, int) else None


def collect_stream_response(stream: Any) -> tuple[str, Any]:
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

        # reasoning_content 只消费，不落盘。
        _ = getattr(delta, "reasoning_content", None)

    return "".join(answer_parts), usage


def append_jsonl(path: Path, record: UsageRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(record), ensure_ascii=False)
    with _log_lock:
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
            f.flush()


def atomic_write_text(path: Path, content: str) -> None:
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
        "余额不足",
        "额度不足",
    )
    return any(x in text for x in needles)


def should_retry_transport(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        return status in {408, 409, 429, 500, 502, 503, 504}
    return False


def call_teacher(
    client: OpenAI,
    provider: ProviderConfig,
    messages: list[dict[str, str]],
) -> tuple[str, Any]:
    kwargs: dict[str, Any] = {
        "model": provider.model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    if provider.name == "lin" and provider.reasoning_effort is not None:
        # 使用 extra_body，避免 OpenAI SDK 版本差异阻止透传扩展字段。
        kwargs["extra_body"] = {
            "reasoning_effort": provider.reasoning_effort,
        }

    stream = client.chat.completions.create(**kwargs)
    return collect_stream_response(stream)


def make_output_path(
    result_root: Path,
    split: str,
    input_md: Path,
) -> Path:
    return result_root / split / input_md.name


def process_one_task(
    split: str,
    input_md: Path,
    output_md: Path,
    provider: ProviderConfig,
    key_pool: KeyPool,
    args: argparse.Namespace,
) -> TaskResult:
    task_id = "<unknown>"

    try:
        md_text = input_md.read_text(encoding="utf-8")
        if not md_text.strip():
            raise ValueError("输入 MD 为空")

        task_id = extract_task_id(md_text)
        legal_candidates = extract_candidate_numbers(md_text)
        messages = split_prompt(md_text, args.message_mode)
    except Exception as exc:
        append_jsonl(
            args.usage_log,
            UsageRecord(
                timestamp_utc=utc_now_iso(),
                provider=provider.name,
                key_slot=None,
                split=split,
                filename=input_md.name,
                task_id=task_id,
                model=provider.model,
                status="PREPARE_FAILED",
                attempt=0,
                elapsed_seconds=0.0,
                error=str(exc),
            ),
        )
        return TaskResult(
            split=split,
            filename=input_md.name,
            task_id=task_id,
            success=False,
            error=f"PREPARE_FAILED: {exc}",
        )

    prompt_total = 0
    completion_total = 0
    total_total = 0
    reasoning_total = 0
    last_error: Optional[BaseException] = None

    for attempt in range(1, args.max_retries + 1):
        key_item = key_pool.acquire_key()
        if key_item is None:
            return TaskResult(
                split=split,
                filename=input_md.name,
                task_id=task_id,
                success=False,
                prompt_tokens=prompt_total,
                completion_tokens=completion_total,
                total_tokens=total_total,
                reasoning_tokens=reasoning_total,
                error="所有 API key 均已因 quota/balance 被禁用",
                all_keys_exhausted=True,
            )

        key_slot, api_key = key_item
        client = get_openai_client(
            provider=provider,
            key_slot=key_slot,
            api_key=api_key,
            timeout=args.timeout,
        )

        started = time.perf_counter()
        usage = None
        raw_content = ""

        try:
            raw_content, usage = call_teacher(
                client=client,
                provider=provider,
                messages=messages,
            )

            if not raw_content.strip():
                raise ValueError("模型最终 content 为空")

            canonical_json = validate_output(
                raw_content=raw_content,
                expected_task_id=task_id,
                legal_candidates=legal_candidates,
            )

            # 启动时已过滤非空文件；并发中每个 input filename 唯一。
            # 仍再次防御：若期间被其他进程生成了非空结果，则绝不覆盖。
            if output_md.exists() and output_md.stat().st_size > 0:
                return TaskResult(
                    split=split,
                    filename=input_md.name,
                    task_id=task_id,
                    success=True,
                    prompt_tokens=prompt_total,
                    completion_tokens=completion_total,
                    total_tokens=total_total,
                    reasoning_tokens=reasoning_total,
                    error="SKIPPED_RACE_NONEMPTY",
                )

            atomic_write_text(output_md, canonical_json)

            elapsed = time.perf_counter() - started
            p = usage_int(usage, "prompt_tokens") or 0
            c = usage_int(usage, "completion_tokens") or 0
            t = usage_int(usage, "total_tokens") or 0
            r = get_reasoning_tokens(usage) or 0

            prompt_total += p
            completion_total += c
            total_total += t
            reasoning_total += r

            append_jsonl(
                args.usage_log,
                UsageRecord(
                    timestamp_utc=utc_now_iso(),
                    provider=provider.name,
                    key_slot=key_slot,
                    split=split,
                    filename=input_md.name,
                    task_id=task_id,
                    model=provider.model,
                    status="SUCCESS",
                    attempt=attempt,
                    elapsed_seconds=round(elapsed, 3),
                    prompt_tokens=p or None,
                    completion_tokens=c or None,
                    total_tokens=t or None,
                    reasoning_tokens=r or None,
                    output_chars=len(canonical_json),
                ),
            )

            if args.sleep_between > 0:
                time.sleep(args.sleep_between)

            return TaskResult(
                split=split,
                filename=input_md.name,
                task_id=task_id,
                success=True,
                prompt_tokens=prompt_total,
                completion_tokens=completion_total,
                total_tokens=total_total,
                reasoning_tokens=reasoning_total,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - started
            last_error = exc

            p = usage_int(usage, "prompt_tokens") or 0
            c = usage_int(usage, "completion_tokens") or 0
            t = usage_int(usage, "total_tokens") or 0
            r = get_reasoning_tokens(usage) or 0

            prompt_total += p
            completion_total += c
            total_total += t
            reasoning_total += r

            quota_error = is_quota_or_billing_error(exc)
            if quota_error:
                key_pool.disable(key_slot)

            append_jsonl(
                args.usage_log,
                UsageRecord(
                    timestamp_utc=utc_now_iso(),
                    provider=provider.name,
                    key_slot=key_slot,
                    split=split,
                    filename=input_md.name,
                    task_id=task_id,
                    model=provider.model,
                    status="ATTEMPT_FAILED",
                    attempt=attempt,
                    elapsed_seconds=round(elapsed, 3),
                    prompt_tokens=p or None,
                    completion_tokens=c or None,
                    total_tokens=t or None,
                    reasoning_tokens=r or None,
                    output_chars=len(raw_content) if raw_content else 0,
                    error=str(exc),
                ),
            )

            # 输出 JSON/schema/task_id/Candidate binding 失败属于模型机械失败，可重试。
            validation_failure = isinstance(exc, ValueError)
            transport_retry = should_retry_transport(exc)

            active_keys = key_pool.active_count()
            may_retry = (
                attempt < args.max_retries
                and active_keys > 0
                and (quota_error or transport_retry or validation_failure)
            )

            if may_retry:
                delay = min(2 ** (attempt - 1), 20) + random.random()
                safe_tqdm_write(
                    f"[RETRY {attempt}/{args.max_retries}] "
                    f"{task_id} | provider={provider.name} | key#{key_slot} | "
                    f"{type(exc).__name__}: {exc} | "
                    f"active_keys={active_keys} | sleep={delay:.1f}s"
                )
                time.sleep(delay)
                continue

            break

        finally:
            # 严格 per-key single concurrency 的关键：
            # 无论 SUCCESS、JSON 校验失败、429、5xx、quota、异常 return，
            # 当前 attempt 持有的 key 都必须释放。
            # disable() 只改变 enabled 状态；release() 负责清除 in_use 并唤醒等待 worker。
            key_pool.release(key_slot)

    return TaskResult(
        split=split,
        filename=input_md.name,
        task_id=task_id,
        success=False,
        prompt_tokens=prompt_total,
        completion_tokens=completion_total,
        total_tokens=total_total,
        reasoning_tokens=reasoning_total,
        error=str(last_error) if last_error else "unknown failure",
        all_keys_exhausted=(key_pool.active_count() == 0),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Strong-Teacher multi-API runner："
            "1 task/request，非空结果绝不覆盖，0B 可重跑，支持 LIN 多 key 并发。"
        )
    )
    p.add_argument(
        "--provider",
        choices=("qwen", "lin"),
        default="lin",
        help="调用 provider。默认 lin。",
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
        "--concurrency",
        type=int,
        default=0,
        help=(
            "并发 worker 数。0=自动：LIN 使用 API key 数量，Qwen 使用 1。"
            "worker 数可大于 key 数，但每个 API key 同时最多 1 个请求；"
            "因此实际 API 并发上限=min(worker 数, 可用 key 数)。"
        ),
    )
    p.add_argument(
        "--lin-reasoning-effort",
        choices=("max", "high", "medium", "low", "none"),
        default=os.getenv("LIN_REASONING_EFFORT", "max"),
        help=(
            'LIN reasoning_effort，默认 "max"。'
            '"none" 表示不发送该扩展字段。'
        ),
    )
    p.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="最多实际请求多少条；0=不限。",
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
        help="单请求 timeout（秒），默认 1800。",
    )
    p.add_argument(
        "--sleep-between",
        type=float,
        default=0.0,
        help="每个 worker 成功请求后等待秒数；默认 0。",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="随机打乱待运行任务。",
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
        help="只扫描并报告，不调用 API。",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="出现一个最终失败 task 后停止继续提交新任务。",
    )
    p.add_argument(
        "--version",
        action="version",
        version=SCRIPT_VERSION,
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.concurrency < 0:
        raise SystemExit("--concurrency 不能小于 0")
    if args.max_retries < 1:
        raise SystemExit("--max-retries 必须 >= 1")

    load_environment(args.env_file)
    provider = load_provider_config(args)
    key_pool = KeyPool(provider.api_keys)

    concurrency = args.concurrency
    if concurrency == 0:
        concurrency = key_pool.size if provider.name == "lin" else 1
    concurrency = max(1, concurrency)

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
                stats.skipped_nonempty += 1
                continue
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
                "provider": provider.name,
                "model": provider.model,
                "base_url": provider.base_url,
                "api_key_count": key_pool.size,
                "concurrency": concurrency,
                "reasoning_effort": provider.reasoning_effort,
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

    stop_event = threading.Event()

    # 不一次性把 2 万个任务全部 submit 到 Future 队列。
    # 使用一个有限 inflight window，便于 stop-on-error / key exhaustion 后及时停止。
    max_inflight = max(concurrency * 2, concurrency)
    pending_iter = iter(pending)
    futures: dict[Any, tuple[str, Path, Path]] = {}

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        if stop_event.is_set():
            return False
        try:
            split, input_md, output_md = next(pending_iter)
        except StopIteration:
            return False

        fut = executor.submit(
            process_one_task,
            split,
            input_md,
            output_md,
            provider,
            key_pool,
            args,
        )
        futures[fut] = (split, input_md, output_md)
        return True

    progress = tqdm(
        total=len(pending),
        desc=f"{provider.name}:{provider.model}",
        unit="task",
        dynamic_ncols=True,
    )

    completed = 0
    with ThreadPoolExecutor(
        max_workers=concurrency,
        thread_name_prefix=f"teacher-{provider.name}",
    ) as executor:
        for _ in range(min(max_inflight, len(pending))):
            if not submit_next(executor):
                break

        while futures:
            # as_completed 针对当前 snapshot，处理至少一个完成结果后再补任务。
            snapshot = list(futures.keys())
            first_done = None
            for fut in as_completed(snapshot):
                first_done = fut
                break

            if first_done is None:
                break

            futures.pop(first_done, None)

            try:
                result: TaskResult = first_done.result()
            except Exception as exc:
                # 理论上 worker 内部已经捕获；这里是最后的线程级安全网。
                result = TaskResult(
                    split="<unknown>",
                    filename="<unknown>",
                    task_id="<unknown>",
                    success=False,
                    error=f"WORKER_CRASH: {exc}",
                )

            completed += 1
            progress.update(1)

            stats.prompt_tokens += result.prompt_tokens
            stats.completion_tokens += result.completion_tokens
            stats.total_tokens += result.total_tokens
            stats.reasoning_tokens += result.reasoning_tokens

            if result.success:
                stats.succeeded += 1
            else:
                stats.failed += 1
                safe_tqdm_write(
                    f"[FAILED] {result.task_id} | {result.filename} | {result.error}"
                )

                if result.all_keys_exhausted:
                    safe_tqdm_write(
                        "[STOP] 所有可用 API key 均已额度/余额耗尽；停止提交新任务。"
                    )
                    stop_event.set()
                elif args.stop_on_error:
                    stop_event.set()

            progress.set_postfix(
                ok=stats.succeeded,
                fail=stats.failed,
                active_keys=key_pool.active_count(),
                tok=stats.total_tokens,
                reason=stats.reasoning_tokens,
            )

            if not stop_event.is_set():
                submit_next(executor)

        # stop_event 后，不再新增；已经在 futures 中/运行中的请求仍完成，
        # 避免粗暴中断导致半请求和不确定计费状态。
        if stop_event.is_set():
            while futures:
                snapshot = list(futures.keys())
                first_done = None
                for fut in as_completed(snapshot):
                    first_done = fut
                    break
                if first_done is None:
                    break

                futures.pop(first_done, None)
                try:
                    result = first_done.result()
                except Exception as exc:
                    result = TaskResult(
                        split="<unknown>",
                        filename="<unknown>",
                        task_id="<unknown>",
                        success=False,
                        error=f"WORKER_CRASH: {exc}",
                    )

                completed += 1
                progress.update(1)

                stats.prompt_tokens += result.prompt_tokens
                stats.completion_tokens += result.completion_tokens
                stats.total_tokens += result.total_tokens
                stats.reasoning_tokens += result.reasoning_tokens

                if result.success:
                    stats.succeeded += 1
                else:
                    stats.failed += 1

    progress.close()

    not_submitted = len(pending) - completed

    print(
        "\n"
        + json.dumps(
            {
                "finished": True,
                "provider": provider.name,
                "model": provider.model,
                "api_key_status": key_pool.status(),
                "concurrency": concurrency,
                "scanned": stats.scanned,
                "selected": stats.selected,
                "completed_requests": completed,
                "not_submitted_after_stop": max(not_submitted, 0),
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
