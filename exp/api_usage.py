"""按任务隔离记录 OpenAI-compatible API 调用成本。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping


_CURRENT_USAGE: ContextVar[dict[str, int] | None] = ContextVar(
    "evidence_agent_api_usage",
    default=None,
)


def _usage_value(usage: Any, *names: str) -> int:
    for name in names:
        value = usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, None)
        if value is not None:
            return int(value)
    return 0


def record_api_usage(
    usage: Any = None,
    *,
    calls: int = 1,
    scope: str = "api",
) -> None:
    """把一次 API 调用按在线推理或离线索引归入当前任务。"""

    bucket = _CURRENT_USAGE.get()
    if bucket is None:
        return
    prompt = _usage_value(usage, "prompt_tokens", "input_tokens") if usage else 0
    completion = (
        _usage_value(usage, "completion_tokens", "output_tokens") if usage else 0
    )
    total = _usage_value(usage, "total_tokens") if usage else 0
    bucket[f"{scope}_prompt_tokens"] += prompt
    bucket[f"{scope}_completion_tokens"] += completion
    bucket[f"{scope}_total_tokens"] += total or prompt + completion
    bucket[f"{scope}_calls"] += calls


@contextmanager
def capture_api_usage() -> Iterator[dict[str, int]]:
    """为一个任务建立独立的 API usage 容器。"""

    bucket = {
        "api_prompt_tokens": 0,
        "api_completion_tokens": 0,
        "api_total_tokens": 0,
        "api_calls": 0,
        "index_api_prompt_tokens": 0,
        "index_api_completion_tokens": 0,
        "index_api_total_tokens": 0,
        "index_api_calls": 0,
    }
    token = _CURRENT_USAGE.set(bucket)
    try:
        yield bucket
    finally:
        _CURRENT_USAGE.reset(token)
