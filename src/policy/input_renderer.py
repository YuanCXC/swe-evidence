"""严格复现 Evidence Policy 训练时的文本输入契约。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


MODEL_MAX_LENGTH = 4096
QUESTION_MAX_TOKENS = 2048


def render_unit(record: Mapping[str, Any]) -> str:
    """渲染一个候选 Evidence Unit 的完整正文。"""

    symbol = record.get("symbol")
    symbol_line = f"\n[SYMBOL] {symbol}" if symbol else ""
    return (
        f"[PATH] {record['path']}\n[TYPE] {record['unit_type']}\n"
        f"[LINES] {record['start_line']}-{record['end_line']}"
        f"{symbol_line}\n[CONTENT]\n{record['content']}"
    )


def render_metadata(record: Mapping[str, Any]) -> str:
    """渲染当前 K 中一个 Evidence 的元数据。"""

    return (
        f"[EVIDENCE_META] id={record['evidence_id']} path={record.get('path')} "
        f"type={record.get('unit_type')} symbol={record.get('symbol')} "
        f"lines={record.get('start_line')}-{record.get('end_line')}"
    )


def build_question(task_input: Mapping[str, Any]) -> str:
    """按训练契约拼接 Problem Statement 与 hints。"""

    pieces = [str(task_input.get("problem_statement") or "")]
    hints = task_input.get("hints")
    if isinstance(hints, str) and hints.strip():
        pieces.append(hints)
    elif isinstance(hints, Sequence) and not isinstance(hints, str):
        pieces.extend(str(item) for item in hints if str(item).strip())
    return "\n".join(piece for piece in pieces if piece.strip())


def encode_ids(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    """在显式长度审计时关闭 Tokenizer 的原生长度警告。"""

    old_max = tokenizer.model_max_length
    try:
        tokenizer.model_max_length = max(int(old_max), 10**12)
        return list(
            tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                truncation=False,
            )
        )
    finally:
        tokenizer.model_max_length = old_max


def token_length(tokenizer: Any, text: str, *, add_special_tokens: bool = True) -> int:
    """计算与模型 forward 一致的真实 Token 数。"""

    return len(encode_ids(tokenizer, text, add_special_tokens=add_special_tokens))


def truncate_question_view(
    text: str,
    tokenizer: Any,
    max_tokens: int = QUESTION_MAX_TOKENS,
) -> str:
    """按训练时的首尾保留规则截断超长问题。"""

    token_ids = encode_ids(tokenizer, text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    marker = "[TRUNCATED_MIDDLE]"
    marker_tokens = encode_ids(tokenizer, marker, add_special_tokens=False)
    available = max_tokens - len(marker_tokens)
    head_count = min(1536, int(available * 0.75))
    tail_count = available - head_count
    head = tokenizer.decode(token_ids[:head_count], skip_special_tokens=True)
    tail = tokenizer.decode(token_ids[-tail_count:], skip_special_tokens=True)
    result = f"{head}\n{marker}\n{tail}".strip()
    result_ids = encode_ids(tokenizer, result, add_special_tokens=False)
    if len(result_ids) > max_tokens:
        result = tokenizer.decode(
            result_ids[:max_tokens],
            skip_special_tokens=True,
        ).strip()
    return result


def render_action_text(
    *,
    question_view: str,
    current_evidence: Sequence[Mapping[str, Any]],
    body_ids: Sequence[str],
    action: Mapping[str, Any],
) -> str:
    """渲染一个完整的 (q, K, A) Cross-Encoder 输入。"""

    state_metadata = "\n".join(render_metadata(unit) for unit in current_evidence)
    if not state_metadata:
        state_metadata = "[EMPTY]"
    body_set = set(map(str, body_ids))
    state_body = "\n\n".join(
        f"[STATE BODY] evidence_id={unit['evidence_id']}\n{unit['content']}"
        for unit in current_evidence
        if str(unit["evidence_id"]) in body_set
    )
    if not state_body:
        state_body = "[NONE]"
    candidate = "\n\n".join(render_unit(unit) for unit in action["evidence"])
    if not candidate:
        candidate = "[STOP]"
    return (
        f"[QUESTION]\n{question_view}\n\n"
        f"[CURRENT EVIDENCE METADATA]\n{state_metadata}\n\n"
        f"[CURRENT EVIDENCE BODY]\n{state_body}\n\n"
        f"[CANDIDATE ACTION]\n{candidate}"
    )


class PolicyInputRenderer:
    """为统一动作集合生成可评分模型输入。"""

    def __init__(
        self,
        tokenizer: Any,
        *,
        model_max_length: int = MODEL_MAX_LENGTH,
    ) -> None:
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length

    def render_actions(
        self,
        *,
        task_input: Mapping[str, Any],
        current_evidence: Sequence[Mapping[str, Any]],
        actions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """渲染并删除违反冻结 4096 Token 契约的动作。"""

        question = build_question(task_input)
        question_view = truncate_question_view(question, self.tokenizer)
        rendered = []
        for action in actions:
            body_ids: list[str] = []
            for unit in reversed(current_evidence):
                trial_ids = [str(unit["evidence_id"]), *body_ids]
                trial_text = render_action_text(
                    question_view=question_view,
                    current_evidence=current_evidence,
                    body_ids=trial_ids,
                    action=action,
                )
                if (
                    token_length(
                        self.tokenizer,
                        trial_text,
                        add_special_tokens=True,
                    )
                    <= self.model_max_length
                ):
                    body_ids = trial_ids
            text = render_action_text(
                question_view=question_view,
                current_evidence=current_evidence,
                body_ids=body_ids,
                action=action,
            )
            count = token_length(self.tokenizer, text, add_special_tokens=True)
            if count <= self.model_max_length:
                rendered.append(
                    {
                        "action": dict(action),
                        "text": text,
                        "model_input_token_count": count,
                        "rendered_state_body_evidence_ids": list(body_ids),
                    }
                )
        return rendered
