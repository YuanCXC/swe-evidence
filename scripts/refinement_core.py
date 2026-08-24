# -*- coding: utf-8 -*-
"""
LLM-assisted Supervision Refinement Core
（大模型辅助监督修正核心）

建议位置：
    scripts/refinement_core.py

本文件属于 Data Processing（数据处理），不是 src/ 实验核心。

职责：
    Original Supervision（原始监督）
        -> LLM Proposal（大模型修正建议）
        -> Deterministic Verification（确定性校验）
        -> Refined Sidecar（修正旁路结果）

原则：
1. 不修改冻结的 V2.10 Parquet / Manifest。
2. LLM 只能引用真实 Candidate Evidence ID（候选证据 ID）。
3. LLM 负责“提议”，程序负责“是否合法”。
4. 显式支持 Question-satisfied Obligation（已由问题描述满足的要求）。
5. 程序会根据 Teacher 输出的 witness graph 计算确定性最小 Certificate；Teacher 自报 Certificate 只作为审计输入。若它缺失、过多或不最小，程序只按 Teacher 自己已经明确给出的 OR-of-AND witness graph 做机械规范化，不会替 Teacher 猜 obligation、AND/OR 或 Evidence 语义。

为什么需要 satisfied_by_question（已由问题描述满足）：
    某些 behavior_constraint（行为约束）或 validation_constraint（验证约束）
    已经在 Issue（问题描述）中清楚给出。

    它们在语义上仍然重要，但不应该强迫 Retriever（检索器）
    再从代码仓库找一段 Evidence（证据）重复证明。

因此 refined obligation 采用：

    required_for_sufficiency
        这个语义要求是否必须理解；

    satisfied_by_question
        是否已经由 q（问题描述）满足；

    retrieval_required
        是否仍需要 repository Evidence。

并定义：

    retrieval_required
      = applicable
        AND required_for_sufficiency
        AND NOT satisfied_by_question

为了兼容现有 V2.10 Certificate 计算：
    mandatory = retrieval_required
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ALLOWED_OBLIGATION_TYPES = (
    "fault_location",
    "fault_logic",
    "dependency_context",
    "state_flow",
    "behavior_constraint",
    "repair_scope",
    "validation_constraint",
)

ALLOWED_ASSESSMENTS = {
    "keep",
    "refine",
    "candidate_pool_insufficient",
    "uncertain",
}

ALLOWED_STOP_ASSESSMENTS = {
    "original_stop_correct",
    "too_early",
    "too_late",
    "uncertain",
}


def stable_id(*parts: object, prefix: str) -> str:
    """生成可复现稳定 ID。"""
    payload = "\0".join(str(x) for x in parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:24]}"


def build_question(task_input: Mapping[str, Any]) -> str:
    """构造与现有语义评估一致的 q = problem_statement + hints。"""
    statement = task_input.get("problem_statement")
    if not isinstance(statement, str) or not statement.strip():
        raise ValueError("task.input.problem_statement 必须为非空字符串")

    parts = [statement.strip()]
    hints = task_input.get("hints")

    if hints is None:
        pass
    elif isinstance(hints, str):
        if hints.strip():
            parts.append(hints.strip())
    elif isinstance(hints, (list, tuple)):
        for item in hints:
            if not isinstance(item, str):
                raise ValueError("input.hints 数组元素必须是字符串")
            if item.strip():
                parts.append(item.strip())
    else:
        raise ValueError("input.hints 必须是 string/list/null")

    return "\n".join(parts)


def choose_policy_state(
    supervision: Mapping[str, Any],
    state_type: str,
) -> Mapping[str, Any] | None:
    """
    稳定选择一个 state_type。

    initial（初始状态）：
        Evidence 更少者优先。

    decision_boundary / complete：
        Evidence 更多者优先。
    """
    states = [
        s
        for s in (supervision.get("policy_states") or [])
        if str(s.get("state_type") or "") == state_type
    ]
    if not states:
        return None

    def n(s: Mapping[str, Any]) -> int:
        return len(s.get("evidence_ids") or [])

    if state_type == "initial":
        return min(states, key=lambda s: (n(s), str(s.get("state_id") or "")))

    return max(states, key=lambda s: (n(s), str(s.get("state_id") or "")))


def original_certificate(supervision: Mapping[str, Any]) -> tuple[str, ...]:
    """读取冻结 V2.10 Complete（完成状态）的 Evidence 集。"""
    state = choose_policy_state(supervision, "complete")
    if state is None:
        return ()
    return tuple(str(x) for x in (state.get("evidence_ids") or []))


def original_boundary(supervision: Mapping[str, Any]) -> tuple[str, ...]:
    """读取冻结 V2.10 Decision Boundary（边界状态）的 Evidence 集。"""
    state = choose_policy_state(supervision, "decision_boundary")
    if state is None:
        return ()
    return tuple(str(x) for x in (state.get("evidence_ids") or []))


@dataclass
class CandidateEvidenceStats:
    """Candidate Pool（候选池）中的 Evidence 来源统计。"""

    evidence_id: str
    in_original_witness: bool = False
    in_original_certificate: bool = False
    in_original_boundary: bool = False
    min_online_rank: int | None = None
    max_online_score: float | None = None
    candidate_sources: set[str] | None = None
    action_labels: set[str] | None = None
    covered_obligation_ids: set[str] | None = None

    def __post_init__(self) -> None:
        if self.candidate_sources is None:
            self.candidate_sources = set()
        if self.action_labels is None:
            self.action_labels = set()
        if self.covered_obligation_ids is None:
            self.covered_obligation_ids = set()

    def ranking_key(self) -> tuple[Any, ...]:
        original_priority = 0 if (
            self.in_original_witness
            or self.in_original_certificate
            or self.in_original_boundary
        ) else 1
        online_priority = 0 if self.min_online_rank is not None else 1
        rank = self.min_online_rank if self.min_online_rank is not None else 2**31 - 1
        score = self.max_online_score if self.max_online_score is not None else float("-inf")
        return (original_priority, online_priority, rank, -score, self.evidence_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "in_original_witness": self.in_original_witness,
            "in_original_certificate": self.in_original_certificate,
            "in_original_boundary": self.in_original_boundary,
            "min_online_rank": self.min_online_rank,
            "max_online_score": self.max_online_score,
            "candidate_sources": sorted(self.candidate_sources or []),
            "action_labels": sorted(self.action_labels or []),
            "covered_obligation_ids": sorted(self.covered_obligation_ids or []),
        }


def collect_candidate_evidence_stats(
    supervision: Mapping[str, Any],
) -> dict[str, CandidateEvidenceStats]:
    """
    从当前 task 的真实发布结构提取 Candidate Evidence（候选证据）：

    1. 原始 Obligation Witness（证据要求支撑证据）；
    2. 原始 Boundary / Complete；
    3. 所有 policy_states.candidate_actions。

    Candidate 只是“可供 LLM 选择的真实证据”，不是标签真值。
    """
    stats: dict[str, CandidateEvidenceStats] = {}

    def get(evidence_id: str) -> CandidateEvidenceStats:
        evidence_id = str(evidence_id)
        if evidence_id not in stats:
            stats[evidence_id] = CandidateEvidenceStats(evidence_id=evidence_id)
        return stats[evidence_id]

    for obligation in (supervision.get("obligations") or []):
        for group in (obligation.get("witness_groups") or []):
            for eid in (group.get("evidence_ids") or []):
                get(str(eid)).in_original_witness = True

    for eid in original_boundary(supervision):
        get(eid).in_original_boundary = True

    for eid in original_certificate(supervision):
        get(eid).in_original_certificate = True

    for state in (supervision.get("policy_states") or []):
        for action in (state.get("candidate_actions") or []):
            for eid in (action.get("evidence_ids") or []):
                item = get(str(eid))

                rank = action.get("online_retrieval_rank")
                if rank is not None:
                    rank = int(rank)
                    if item.min_online_rank is None or rank < item.min_online_rank:
                        item.min_online_rank = rank

                score = action.get("online_retrieval_score")
                if score is not None:
                    score = float(score)
                    if item.max_online_score is None or score > item.max_online_score:
                        item.max_online_score = score

                for source in (action.get("candidate_sources") or []):
                    item.candidate_sources.add(str(source))

                label = str(action.get("action_label") or "")
                if label:
                    item.action_labels.add(label)

                for oid in (action.get("covered_obligation_ids") or []):
                    item.covered_obligation_ids.add(str(oid))

    return stats


def select_candidate_evidence_ids(
    supervision: Mapping[str, Any],
    *,
    limit: int,
    available_evidence_ids: set[str] | None = None,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """
    选择 Candidate Pool（候选证据池）。

    关键区别：

    1. Original Supervision Evidence（原始监督依赖证据）：
         - original witness（原始支撑证据）
         - decision boundary（边界状态）
         - complete certificate（完成证书）

       这些证据如果在 Evidence Cache（证据缓存）中不存在，
       属于真正的数据完整性错误，必须 hard fail（硬失败）。

    2. Ordinary Policy Candidate（普通策略候选）：

       它们只是扩展候选，不是冻结标签。
       当前训练 Evidence Cache 不保证把所有普通候选全部物化。

       如果普通候选缺失：
         - 跳过；
         - 继续用后续可用候选补位；
         - 在 metadata 中记录；
         - 不把整个 task 判为 preparation failure（准备失败）。

    3. available_evidence_ids=None：

       不做 Cache availability（缓存可用性）过滤，
       兼容旧调用行为。

    4. 可用普通候选不足时：

       返回 underfilled candidate pool（未填满候选池），
       不编造 Evidence，也不静默把缺失候选当成存在。
       Teacher 后续可返回 candidate_pool_insufficient（候选池不足）。
    """
    if limit < 1:
        raise ValueError("candidate limit 必须 >= 1")

    stats = collect_candidate_evidence_stats(supervision)
    ordered = sorted(stats.values(), key=lambda item: item.ranking_key())

    # ------------------------------------------------------------------
    # Original Witness（原始支撑证据）需要再区分两类：
    #
    # A. Trajectory-critical Evidence（轨迹关键证据）
    #    当前 Boundary / Complete 真正使用的 Evidence。
    #
    #    这些 Evidence 缺失会导致我们连“当前被审计的轨迹”都无法还原，
    #    因此必须 hard fail（硬失败）。
    #
    # B. Alternative Witness Evidence（备用支撑证据）
    #    它存在于某个原始 witness group 中，
    #    但没有进入当前 Boundary / Complete。
    #
    #    这种 Evidence 只是历史监督给出的“另一种可能支撑方式”。
    #    如果当前训练 Evidence Cache 没有物化它：
    #
    #        - 不能把整个 task 判为失败；
    #        - 记录为 soft missing（软缺失）；
    #        - Teacher 不允许重新选择它；
    #        - 仍可审计当前 Boundary -> Complete 轨迹。
    #
    # 这是本轮 audit 发现的真实数据合同：
    # 当前 4 个失败 task 缺的全部都是这一类备用 witness。
    # ------------------------------------------------------------------

    hard_required = [
        item
        for item in ordered
        if (
            item.in_original_certificate
            or item.in_original_boundary
        )
    ]

    hard_required_ids = {
        item.evidence_id
        for item in hard_required
    }

    original_witness_ids = {
        item.evidence_id
        for item in ordered
        if item.in_original_witness
    }

    if available_evidence_ids is None:
        available_set = {
            item.evidence_id
            for item in ordered
        }
        cache_filter_applied = False
    else:
        available_set = set(
            map(
                str,
                available_evidence_ids,
            )
        )
        cache_filter_applied = True

    hard_missing_ids = sorted(
        hard_required_ids
        - available_set
    )

    if hard_missing_ids:
        details = [
            stats[evidence_id].to_dict()
            for evidence_id in hard_missing_ids
        ]

        raise ValueError(
            "Evidence Cache 缺少 Boundary/Complete Evidence "
            "（当前审计轨迹的关键证据），不能安全 refinement："
            f"{json.dumps(details, ensure_ascii=False)}"
        )

    alternative_witness_ids = (
        original_witness_ids
        - hard_required_ids
    )

    missing_alternative_witness_ids = sorted(
        alternative_witness_ids
        - available_set
    )

    unavailable_ids = {
        item.evidence_id
        for item in ordered
        if item.evidence_id
        not in available_set
    }

    # 普通 Policy Candidate：
    # 既不属于当前 Boundary/Complete，
    # 也不是历史 alternative witness。
    missing_ordinary_candidate_ids = sorted(
        unavailable_ids
        - hard_required_ids
        - alternative_witness_ids
    )

    eligible = [
        item
        for item in ordered
        if item.evidence_id
        in available_set
    ]

    # ------------------------------------------------------------------
    # 所有“当前 Cache 中真实存在的原始 witness”都优先保留。
    #
    # 原因：
    # Teacher 的任务之一就是审计旧 witness 是否合理。
    # 既然某个旧 witness 的正文已经可用，就不应该为了 48 条上限把它丢掉。
    #
    # 这意味着：
    # 如果 available original witness 本身 > limit，
    # 实际 candidate count 可以超过 limit。
    # ------------------------------------------------------------------

    preferred_original_ids = {
        item.evidence_id
        for item in eligible
        if (
            item.in_original_witness
            or item.in_original_certificate
            or item.in_original_boundary
        )
    }

    target = max(
        limit,
        len(
            preferred_original_ids
        ),
    )

    selected_ids = set(
        preferred_original_ids
    )

    # 然后按稳定 ranking 顺序用其它 Cache-present candidate 补位。
    for item in eligible:
        if len(selected_ids) >= target:
            break

        selected_ids.add(
            item.evidence_id
        )

    selected = tuple(
        item.evidence_id
        for item in ordered
        if item.evidence_id
        in selected_ids
    )

    if not selected:
        raise ValueError(
            "Candidate Pool（候选池）为空；无法进行 supervision refinement"
        )

    return selected, {
        "available_candidate_count": len(
            ordered
        ),
        "cache_filter_applied": (
            cache_filter_applied
        ),
        "cache_present_candidate_count": len(
            eligible
        ),
        "cache_missing_candidate_count": len(
            unavailable_ids
        ),

        # 真正的硬错误只允许出现在 Boundary / Complete。
        "cache_missing_trajectory_critical_count": len(
            hard_missing_ids
        ),

        # 备用 witness 缺失：
        # 允许继续，但必须透明报告。
        "cache_missing_alternative_witness_count": len(
            missing_alternative_witness_ids
        ),
        "cache_missing_alternative_witness_ids": (
            missing_alternative_witness_ids
        ),
        "cache_missing_alternative_witness_candidates": [
            stats[evidence_id].to_dict()
            for evidence_id in (
                missing_alternative_witness_ids
            )
        ],

        # 普通扩展候选缺失：
        # 同样允许继续。
        "cache_missing_ordinary_candidate_count": len(
            missing_ordinary_candidate_ids
        ),
        "cache_missing_ordinary_candidate_ids": (
            missing_ordinary_candidate_ids
        ),

        # 兼容上一版 report 消费方：
        # optional = alternative witness + ordinary candidate
        "cache_missing_optional_candidate_count": (
            len(
                missing_alternative_witness_ids
            )
            + len(
                missing_ordinary_candidate_ids
            )
        ),
        "cache_missing_forced_candidate_count": len(
            hard_missing_ids
        ),

        "requested_limit": (
            limit
        ),
        "effective_target_count": (
            target
        ),
        "available_original_witness_candidate_count": sum(
            bool(
                item.in_original_witness
            )
            for item in eligible
        ),
        "trajectory_critical_candidate_count": len(
            hard_required
        ),
        "selected_candidate_count": len(
            selected
        ),
        "candidate_pool_underfilled": (
            len(selected)
            < target
        ),
        "candidate_pool_underfilled_by": max(
            0,
            target
            - len(selected),
        ),
        "selected_candidates": [
            stats[evidence_id].to_dict()
            for evidence_id in selected
        ],
    }


def strip_json_fence(text: str) -> str:
    """仅去掉最外层 ```json fence，不做模糊 JSON 修复。"""
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return value


def parse_refinement_proposal(text: str) -> dict[str, Any]:
    """严格解析 Teacher 的 JSON object。"""
    try:
        value = json.loads(strip_json_fence(text))
    except json.JSONDecodeError as exc:
        raise ValueError("Refinement Teacher 输出不是合法 JSON") from exc

    if not isinstance(value, dict):
        raise ValueError("Refinement Teacher 顶层必须是 JSON object")
    return value


def _str(value: Any, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} 必须是 string")
    value = value.strip()
    if not value and not allow_empty:
        raise ValueError(f"{name} 不能为空")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} 必须是 bool")
    return value


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence 必须是 [0,1] 数字")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence 必须在 [0,1]")
    return value


def _evidence_ids(
    values: Any,
    *,
    allowed: set[str],
    name: str,
) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{name} 必须是 list")
    ids = tuple(dict.fromkeys(_str(x, name) for x in values))
    unknown = sorted(set(ids) - allowed)
    if unknown:
        raise ValueError(f"{name} 引用了候选池之外 Evidence ID：{unknown}")
    return ids


def evidence_state_metrics(
    evidence_ids: Sequence[str],
    obligations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    与 V2.10 相同的 witness 逻辑：

        Obligation = group1 OR group2 OR ...
        Group = evidenceA AND evidenceB AND ...
    """
    selected = set(map(str, evidence_ids))
    applicable = [x for x in obligations if x.get("applicable")]
    mandatory = [x for x in applicable if x.get("mandatory")]
    completed: list[str] = []

    for obligation in applicable:
        groups = [
            set(map(str, g.get("evidence_ids") or []))
            for g in (obligation.get("witness_groups") or [])
            if g.get("evidence_ids")
        ]
        if any(group <= selected for group in groups):
            completed.append(str(obligation["obligation_id"]))

    if mandatory:
        mids = {str(x["obligation_id"]) for x in mandatory}
        completion_score = len(mids & set(completed)) / len(mids)
    else:
        completion_score = None

    return {
        "completed_obligation_ids": sorted(completed),
        "completion_score": completion_score,
        "mandatory_obligation_count": len(mandatory),
    }


def minimum_sufficient_certificate(
    obligations: Sequence[Mapping[str, Any]],
    token_costs: Mapping[str, int],
) -> list[str]:
    """
    按 V2.10 风格重新计算 Verified Minimal Certificate（已校验最小证书）。

    优先：
        1. 总 token cost 小；
        2. Evidence 数少；
        3. ID 稳定排序。

    组合 <=4096 时穷举；过大时使用与 V2.10 同类贪心 fallback。
    """
    mandatory = [
        o for o in obligations
        if o.get("applicable") and o.get("mandatory")
    ]

    if not mandatory:
        return []

    choices = [
        [
            (
                str(g["group_id"]),
                tuple(sorted(set(map(str, g.get("evidence_ids") or [])))),
            )
            for g in (o.get("witness_groups") or [])
            if g.get("evidence_ids")
        ]
        for o in mandatory
    ]

    if any(not groups for groups in choices):
        return []

    def key(items: Sequence[tuple[str, tuple[str, ...]]]) -> tuple[Any, ...]:
        evidence = sorted({
            eid for _gid, group in items for eid in group
        })
        return (
            sum(int(token_costs.get(eid, 2**30)) for eid in evidence),
            len(evidence),
            tuple(evidence),
            tuple(gid for gid, _group in items),
        )

    combinations = 1
    for groups in choices:
        combinations *= len(groups)

    if combinations <= 4096:
        selected = min(itertools.product(*choices), key=key)
    else:
        selected_list = []
        acquired: set[str] = set()
        for groups in choices:
            best = min(
                groups,
                key=lambda item: (
                    sum(
                        int(token_costs.get(eid, 2**30))
                        for eid in (set(item[1]) - acquired)
                    ),
                    len(set(item[1]) - acquired),
                    item[0],
                ),
            )
            selected_list.append(best)
            acquired.update(best[1])
        selected = tuple(selected_list)

    return sorted({
        eid for _gid, group in selected for eid in group
    })


def _normalize_nullable_source_id(
    value: Any,
) -> tuple[str | None, str | None]:
    """
    只对 source_obligation_id 做“无语义风险”的空值规范化。

    接受：
        None
        ""
        "null"
        "NULL"
        "none"
        "None"

    统一为：
        None

    返回：
        (normalized_value, normalization_reason)

    重要：
        这不是语义修复。
        程序不会：
            - 猜 obligation type；
            - 改 witness group；
            - 改 certificate；
            - 把错误 ID 自动映射到某个真实 ID。

        只修复常见 JSON 空值格式问题。
    """
    if value is None:
        return None, None

    if not isinstance(value, str):
        raise ValueError(
            "source_obligation_id 必须是 string 或 null"
        )

    text = value.strip()

    if text.lower() in {
        "",
        "null",
        "none",
    }:
        return (
            None,
            (
                "normalized source_obligation_id "
                f"{value!r} -> null"
            ),
        )

    return text, None


def normalize_refinement_proposal_format(
    proposal: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """
    Deterministic Format Normalization（确定性格式规范化）。

    当前只规范化：
        refined_obligations[*].source_obligation_id

    原因：
        Teacher 有时语义正确，但把 JSON null 写成字符串 "null"。

    这种问题可以安全地由程序机械修正，
    不需要第二次 LLM 调用。

    不做任何语义推断。
    """
    normalized = json.loads(
        json.dumps(
            proposal,
            ensure_ascii=False,
        )
    )

    events: list[
        dict[str, Any]
    ] = []

    raw_obligations = (
        normalized.get(
            "refined_obligations"
        )
    )

    if isinstance(
        raw_obligations,
        list,
    ):
        for index, raw in enumerate(
            raw_obligations
        ):
            if not isinstance(
                raw,
                dict,
            ):
                continue

            if (
                "source_obligation_id"
                not in raw
            ):
                continue

            old_value = raw.get(
                "source_obligation_id"
            )

            (
                new_value,
                reason,
            ) = _normalize_nullable_source_id(
                old_value
            )

            raw[
                "source_obligation_id"
            ] = new_value

            if reason is not None:
                events.append(
                    {
                        "field": (
                            "refined_obligations"
                            f"[{index}]"
                            ".source_obligation_id"
                        ),
                        "old_value": old_value,
                        "new_value": None,
                        "reason": reason,
                    }
                )

    return (
        normalized,
        events,
    )


def _unsatisfied_mandatory_obligation_ids(
    *,
    evidence_ids: Sequence[str],
    obligations: Sequence[
        Mapping[str, Any]
    ],
) -> list[str]:
    """
    返回某个 Evidence Set 尚未满足的 mandatory obligation IDs。

    这里只执行已有 OR-of-AND 结构：
        obligation 满足
        <=> 至少一个 witness_group 的全部 evidence_ids 都已出现。

    不进行语义推断。
    """
    evidence_set = set(
        map(
            str,
            evidence_ids,
        )
    )

    unsatisfied: list[
        str
    ] = []

    for obligation in obligations:
        if not bool(
            obligation.get(
                "mandatory"
            )
        ):
            continue

        groups = (
            obligation.get(
                "witness_groups"
            )
            or []
        )

        satisfied = False

        for group in groups:
            group_ids = set(
                map(
                    str,
                    (
                        group.get(
                            "evidence_ids"
                        )
                        or []
                    ),
                )
            )

            if (
                group_ids
                and group_ids.issubset(
                    evidence_set
                )
            ):
                satisfied = True
                break

        if not satisfied:
            unsatisfied.append(
                str(
                    obligation.get(
                        "obligation_id"
                    )
                    or obligation.get(
                        "type"
                    )
                    or "<unknown>"
                )
            )

    return unsatisfied


def _copy_original_obligations(
    supervision: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return json.loads(json.dumps(
        supervision.get("obligations") or [],
        ensure_ascii=False,
    ))


def _normalize_refined_obligations(
    *,
    task_id: str,
    raw_obligations: Any,
    original_obligations: Sequence[Mapping[str, Any]],
    allowed_evidence_ids: set[str],
    confidence: float,
) -> list[dict[str, Any]]:
    """验证并规范化 Teacher 提出的完整 refined obligation graph。"""
    if not isinstance(raw_obligations, list):
        raise ValueError("refined_obligations 必须是 list")

    original_by_id = {
        str(x.get("obligation_id")): x
        for x in original_obligations
        if x.get("obligation_id")
    }

    result: list[dict[str, Any]] = []
    seen_types: set[str] = set()

    # 一个 Original Obligation 最多映射到一个 refined obligation。
    # 如果想新增不同 type，必须 source_obligation_id=null。
    seen_source_ids: set[str] = set()

    for i, raw in enumerate(raw_obligations):
        if not isinstance(raw, dict):
            raise ValueError("refined_obligations 元素必须是 object")

        otype = _str(raw.get("type"), f"refined_obligations[{i}].type")
        if otype not in ALLOWED_OBLIGATION_TYPES:
            raise ValueError(f"非法 obligation type：{otype}")

        (
            source_id,
            _source_id_normalization_reason,
        ) = _normalize_nullable_source_id(
            raw.get(
                "source_obligation_id"
            )
        )

        # 先检查 source identity，再检查 type 唯一性。
        #
        # 这样如果 Teacher 重复复用了同一个 Original Obligation，
        # 错误信息会直接指出 source_obligation_id，而不是被
        # “同 type 重复”提前遮住。
        if source_id is not None:
            if source_id not in original_by_id:
                raise ValueError(
                    f"source_obligation_id 不存在：{source_id}"
                )

            if source_id in seen_source_ids:
                raise ValueError(
                    "同一个 source_obligation_id 不允许在 "
                    f"refined_obligations 中重复使用：{source_id}"
                )

            seen_source_ids.add(
                source_id
            )

            old_type = str(
                original_by_id[
                    source_id
                ].get(
                    "type"
                )
                or ""
            )

            if old_type != otype:
                raise ValueError(
                    "已有 obligation 不允许改变 type："
                    f"source_obligation_id={source_id}, "
                    f"{old_type} -> {otype}。"
                    "如果要新增不同 type，"
                    "必须 source_obligation_id=null"
                )

        if otype in seen_types:
            raise ValueError(
                f"同一 refined obligation type 重复：{otype}"
            )

        seen_types.add(
            otype
        )

        description = _str(raw.get("description"), f"{otype}.description")
        applicable = _bool(raw.get("applicable"), f"{otype}.applicable")
        required = _bool(
            raw.get("required_for_sufficiency"),
            f"{otype}.required_for_sufficiency",
        )
        question_satisfied = _bool(
            raw.get("satisfied_by_question"),
            f"{otype}.satisfied_by_question",
        )

        retrieval_required = bool(
            applicable and required and not question_satisfied
        )

        raw_groups = raw.get("witness_groups")
        if not isinstance(raw_groups, list):
            raise ValueError(f"{otype}.witness_groups 必须是 list")

        groups: list[dict[str, Any]] = []
        for j, group in enumerate(raw_groups):
            if not isinstance(group, dict):
                raise ValueError(f"{otype}.witness_groups[{j}] 必须是 object")

            ids = _evidence_ids(
                group.get("evidence_ids"),
                allowed=allowed_evidence_ids,
                name=f"{otype}.witness_groups[{j}].evidence_ids",
            )
            if not ids:
                raise ValueError(f"{otype} 存在空 witness group")

            reason = _str(
                group.get("reason"),
                f"{otype}.witness_groups[{j}].reason",
            )

            groups.append({
                "group_id": stable_id(
                    task_id,
                    otype,
                    *sorted(ids),
                    prefix="ref_witness",
                ),
                "logic": "AND",
                "evidence_ids": list(ids),
                "source": "llm_refinement",
                "confidence": confidence,
                "reason": reason,
                "annotation_ids": [],
            })

        # 已由问题描述满足，则不再要求 repository witness。
        if question_satisfied:
            groups = []

        if retrieval_required and not groups:
            raise ValueError(
                f"{otype} retrieval_required=True 但没有 witness_groups"
            )

        obligation_id = (
            source_id
            if source_id is not None
            else stable_id(
                task_id,
                otype,
                description,
                prefix="ref_obligation",
            )
        )

        result.append({
            "obligation_id": obligation_id,
            "source_obligation_id": source_id,
            "type": otype,
            "description": description,
            "applicable": applicable,
            "required_for_sufficiency": required,
            "satisfied_by_question": question_satisfied,
            "retrieval_required": retrieval_required,
            "mandatory": retrieval_required,
            "construction_method": "llm_refinement_verified",
            "confidence": confidence,
            "witness_groups": groups,
            "annotation_ids": [],
        })

    return result


def verify_and_finalize_refinement(
    *,
    task_id: str,
    supervision: Mapping[str, Any],
    candidate_evidence_ids: Sequence[str],
    existing_evidence_ids: set[str],
    token_costs: Mapping[str, int],
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    """
    对 LLM Proposal 做 Deterministic Verification（确定性校验）。

    非法引用直接抛错；
    候选池不足则返回 needs_more_candidates；
    合法 proposal 最终由程序重新计算最小 certificate。
    """
    # ------------------------------------------------------------------
    # Proposal Format Normalization（提案格式规范化）
    #
    # 只修 JSON 层的无语义风险问题，例如：
    #     "source_obligation_id": "null"
    #         ->
    #     "source_obligation_id": null
    #
    # AND/OR、type、Witness、Certificate 均不会自动修改。
    # ------------------------------------------------------------------

    (
        normalized_proposal,
        proposal_format_normalizations,
    ) = normalize_refinement_proposal_format(
        proposal
    )

    proposal = normalized_proposal

    candidate_set = set(map(str, candidate_evidence_ids))
    if not candidate_set:
        raise ValueError("candidate evidence pool 为空")

    missing_cache = sorted(candidate_set - existing_evidence_ids)
    if missing_cache:
        raise ValueError(f"候选池 Evidence Cache 缺失：{missing_cache}")

    assessment = _str(proposal.get("assessment"), "assessment")
    if assessment not in ALLOWED_ASSESSMENTS:
        raise ValueError(f"非法 assessment={assessment}")

    stop_assessment = _str(proposal.get("stop_assessment"), "stop_assessment")
    if stop_assessment not in ALLOWED_STOP_ASSESSMENTS:
        raise ValueError(f"非法 stop_assessment={stop_assessment}")

    confidence = _confidence(proposal.get("confidence"))
    rationale = _str(proposal.get("rationale"), "rationale")
    missing_requests = proposal.get("missing_candidate_requests") or []
    if not isinstance(missing_requests, list):
        raise ValueError("missing_candidate_requests 必须是 list")

    # ------------------------------------------------------------------
    # Assessment / Missing Candidate Contract（判断与缺失候选合同）
    #
    # candidate_pool_insufficient：
    #     必须明确指出缺什么。
    #
    # 其它 assessment：
    #     不允许一边声称当前候选足够形成答案，
    #     一边又要求“缺少关键候选”。
    #
    # 这种矛盾不是 API 错误，而是 Teacher Proposal（教师提案）
    # 需要 reconciliation（重新对齐）。
    # ------------------------------------------------------------------

    if (
        assessment == "candidate_pool_insufficient"
        and not missing_requests
    ):
        return {
            "task_id": task_id,
            "verification_status": "needs_reconciliation",
            "assessment": assessment,
            "stop_assessment": stop_assessment,
            "confidence": confidence,
            "rationale": rationale,
            "missing_candidate_requests": missing_requests,
            "reconciliation_reason": (
                "assessment=candidate_pool_insufficient "
                "requires non-empty missing_candidate_requests"
            ),
            "refined_obligations": None,
            "teacher_proposed_certificate_evidence_ids": None,
            "verified_minimal_certificate_evidence_ids": None,
            "added_evidence_ids": [],
            "removed_evidence_ids": [],
        }

    if (
        assessment != "candidate_pool_insufficient"
        and missing_requests
    ):
        return {
            "task_id": task_id,
            "verification_status": "needs_reconciliation",
            "assessment": assessment,
            "stop_assessment": stop_assessment,
            "confidence": confidence,
            "rationale": rationale,
            "missing_candidate_requests": missing_requests,
            "reconciliation_reason": (
                "non-empty missing_candidate_requests requires "
                "assessment=candidate_pool_insufficient"
            ),
            "refined_obligations": None,
            "teacher_proposed_certificate_evidence_ids": (
                proposal.get("proposed_certificate_evidence_ids")
            ),
            "verified_minimal_certificate_evidence_ids": None,
            "added_evidence_ids": [],
            "removed_evidence_ids": [],
        }

    old_obligations = _copy_original_obligations(supervision)
    old_certificate = list(original_certificate(supervision))
    old_boundary = list(original_boundary(supervision))

    common = {
        "task_id": task_id,
        "proposal_format_normalizations": (
            proposal_format_normalizations
        ),
        "assessment": assessment,
        "stop_assessment": stop_assessment,
        "confidence": confidence,
        "rationale": rationale,
        "missing_candidate_requests": missing_requests,
        "original_boundary_evidence_ids": old_boundary,
        "original_certificate_evidence_ids": old_certificate,
    }

    if assessment == "candidate_pool_insufficient":
        return {
            **common,
            "verification_status": "needs_more_candidates",
            "refined_obligations": None,
            "teacher_proposed_certificate_evidence_ids": None,
            "verified_minimal_certificate_evidence_ids": None,
            "added_evidence_ids": [],
            "removed_evidence_ids": [],
        }

    if assessment == "uncertain":
        return {
            **common,
            "verification_status": "uncertain",
            "refined_obligations": None,
            "teacher_proposed_certificate_evidence_ids": None,
            "verified_minimal_certificate_evidence_ids": None,
            "added_evidence_ids": [],
            "removed_evidence_ids": [],
        }

    if assessment == "keep":
        # KEEP（保留）意味着：
        #   - 不允许偷偷重写 obligation graph；
        #   - proposed certificate 必须就是原 certificate。
        raw_refined = proposal.get("refined_obligations") or []

        if raw_refined:
            return {
                **common,
                "verification_status": "needs_reconciliation",
                "reconciliation_reason": (
                    "assessment=keep requires empty refined_obligations"
                ),
                "refined_obligations": None,
                "teacher_proposed_certificate_evidence_ids": (
                    proposal.get("proposed_certificate_evidence_ids")
                ),
                "verified_minimal_certificate_evidence_ids": None,
                "added_evidence_ids": [],
                "removed_evidence_ids": [],
            }

        refined_obligations = old_obligations

    else:
        raw_refined = proposal.get("refined_obligations")

        if not isinstance(raw_refined, list) or not raw_refined:
            return {
                **common,
                "verification_status": "needs_reconciliation",
                "reconciliation_reason": (
                    "assessment=refine requires non-empty refined_obligations"
                ),
                "refined_obligations": None,
                "teacher_proposed_certificate_evidence_ids": (
                    proposal.get("proposed_certificate_evidence_ids")
                ),
                "verified_minimal_certificate_evidence_ids": None,
                "added_evidence_ids": [],
                "removed_evidence_ids": [],
            }

        refined_obligations = _normalize_refined_obligations(
            task_id=task_id,
            raw_obligations=raw_refined,
            original_obligations=old_obligations,
            allowed_evidence_ids=candidate_set,
            confidence=confidence,
        )

    proposed = _evidence_ids(
        proposal.get("proposed_certificate_evidence_ids") or [],
        allowed=candidate_set,
        name="proposed_certificate_evidence_ids",
    )

    # KEEP 必须真的是原 certificate，不允许字段名称说 keep，
    # 实际 certificate 又发生变化。
    if (
        assessment == "keep"
        and set(proposed) != set(old_certificate)
    ):
        return {
            **common,
            "verification_status": "needs_reconciliation",
            "reconciliation_reason": (
                "assessment=keep requires proposed certificate "
                "to equal original certificate"
            ),
            "refined_obligations": refined_obligations,
            "teacher_proposed_certificate_evidence_ids": list(proposed),
            "verified_minimal_certificate_evidence_ids": old_certificate,
            "added_evidence_ids": [],
            "removed_evidence_ids": [],
        }

    verified = (
        list(old_certificate)
        if assessment == "keep"
        else minimum_sufficient_certificate(
            refined_obligations,
            token_costs,
        )
    )

    verified_metrics = evidence_state_metrics(
        verified,
        refined_obligations,
    )

    mandatory_count = int(verified_metrics["mandatory_obligation_count"])
    completion_score = verified_metrics["completion_score"]

    if mandatory_count > 0 and (
        completion_score is None
        or abs(float(completion_score) - 1.0) > 1e-9
    ):
        raise ValueError("无法从 refined obligations 构造完整 certificate")

    proposed_metrics = evidence_state_metrics(
        proposed,
        refined_obligations,
    )

    proposed_complete = (
        proposed_metrics["mandatory_obligation_count"] == 0
        or (
            proposed_metrics["completion_score"] is not None
            and abs(float(proposed_metrics["completion_score"]) - 1.0) <= 1e-9
        )
    )

    old_set = set(old_certificate)
    new_set = set(verified)
    proposed_set = set(proposed)

    # ------------------------------------------------------------------
    # Certificate Consistency Gate（证书一致性门）
    #
    # Teacher 提出的 certificate 必须：
    #
    #   1. 完整满足 refined obligation graph；
    #   2. 与同一 graph 的 deterministic minimal certificate 完全一致。
    #
    # 如果不一致：
    #   不再像 v1 那样“程序偷偷压缩以后仍标 accepted”。
    #
    # 因为这恰恰可能意味着：
    #   Teacher 在自然语言里认为 A+B+C 必须一起出现，
    #   但却把它们错误编码成多个 OR witness groups。
    #
    # 这种样本必须重新让 Teacher 对齐结构，而不是生成训练标签。
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Certificate Canonicalization（证书机械规范化）
    #
    # v1.7 的原则：
    #
    # Teacher 已经负责决定：
    #   - obligation 是否必要；
    #   - satisfied_by_question；
    #   - witness_groups 的 OR-of-AND 结构；
    #   - 每个 group 里哪些 Evidence 必须共同出现。
    #
    # 一旦这些语义结构已经明确，
    # “从该图计算一个 deterministic minimal certificate”
    # 就是纯机械执行，而不是新的语义判断。
    #
    # 因此：
    #   - Teacher certificate 不完整；
    #   - Teacher certificate 完整但包含冗余 Evidence；
    #
    # 不再直接判 needs_reconciliation。
    #
    # 程序统一使用 verified minimal certificate 作为最终证书，
    # 并完整记录 Teacher 原始集合和规范化原因。
    #
    # 仍然不会自动修：
    #   - 错误 Evidence ID；
    #   - obligation type；
    #   - witness_groups；
    #   - AND/OR；
    #   - assessment；
    #   - candidate_pool_insufficient 的语义判断。
    # ------------------------------------------------------------------

    unsatisfied_obligation_ids = (
        []
        if proposed_complete
        else _unsatisfied_mandatory_obligation_ids(
            evidence_ids=(
                proposed
            ),
            obligations=(
                refined_obligations
            ),
        )
    )

    certificate_normalized = bool(
        not proposed_complete
        or proposed_set != new_set
    )

    if not proposed_complete:
        certificate_normalization_reason = (
            "teacher_certificate_incomplete_for_own_witness_graph"
        )
    elif proposed_set != new_set:
        certificate_normalization_reason = (
            "teacher_certificate_nonminimal_or_alternative_for_own_witness_graph"
        )
    else:
        certificate_normalization_reason = None

    return {
        **common,
        "verification_status": "accepted",

        # Teacher 原始输出，永远保留。
        "teacher_proposed_certificate_evidence_ids": list(
            proposed
        ),
        "teacher_proposed_certificate_is_complete": (
            proposed_complete
        ),
        "teacher_proposed_unsatisfied_obligation_ids": (
            unsatisfied_obligation_ids
        ),
        "teacher_certificate_matches_verified_minimal": (
            proposed_set
            == new_set
        ),

        # v1.7 新增：明确记录机械规范化。
        "certificate_normalized_from_teacher_graph": (
            certificate_normalized
        ),
        "certificate_normalization_reason": (
            certificate_normalization_reason
        ),

        # 真正进入 refined supervision 的 Certificate。
        "verified_minimal_certificate_evidence_ids": (
            verified
        ),
        "verified_certificate_completion_score": (
            completion_score
        ),

        "refined_obligations": (
            refined_obligations
        ),
        "retrieval_required_mandatory_count": (
            mandatory_count
        ),
        "question_satisfied_required_count": sum(
            bool(
                o.get(
                    "applicable"
                )
            )
            and bool(
                o.get(
                    "required_for_sufficiency"
                )
            )
            and bool(
                o.get(
                    "satisfied_by_question"
                )
            )
            for o in refined_obligations
        ),

        "added_evidence_ids": sorted(
            new_set
            - old_set
        ),
        "removed_evidence_ids": sorted(
            old_set
            - new_set
        ),
        "certificate_size_before": len(
            old_certificate
        ),
        "certificate_size_after": len(
            verified
        ),
    }



def refinement_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """汇总一批 sidecar refinement 结果。"""
    status_counts: defaultdict[str, int] = defaultdict(int)
    before: list[int] = []
    after: list[int] = []
    added = 0
    removed = 0
    q_satisfied = 0

    for record in records:
        status_counts[str(record.get("verification_status") or "unknown")] += 1
        added += len(record.get("added_evidence_ids") or [])
        removed += len(record.get("removed_evidence_ids") or [])
        if record.get("certificate_size_before") is not None:
            before.append(int(record["certificate_size_before"]))
        if record.get("certificate_size_after") is not None:
            after.append(int(record["certificate_size_after"]))
        q_satisfied += int(record.get("question_satisfied_required_count") or 0)

    def avg(values: Sequence[int]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "record_count": len(records),
        "verification_status_counts": dict(sorted(status_counts.items())),
        "total_added_evidence_count": added,
        "total_removed_evidence_count": removed,
        "mean_certificate_size_before": avg(before),
        "mean_certificate_size_after": avg(after),
        "question_satisfied_required_count": q_satisfied,
    }


def _self_check() -> None:
    """
    toy case：
        原证书 [A, B]
        其中 behavior_constraint 已由 q 满足
        refined 证书应缩短为 [A]
    """
    supervision = {
        "obligations": [
            {
                "obligation_id": "ob_fault",
                "type": "fault_location",
                "description": "Locate fault.",
                "applicable": True,
                "mandatory": True,
                "witness_groups": [{"group_id": "g1", "evidence_ids": ["A"]}],
            },
            {
                "obligation_id": "ob_behavior",
                "type": "behavior_constraint",
                "description": "Expected behavior.",
                "applicable": True,
                "mandatory": True,
                "witness_groups": [{"group_id": "g2", "evidence_ids": ["B"]}],
            },
        ],
        "policy_states": [
            {
                "state_id": "boundary",
                "state_type": "decision_boundary",
                "evidence_ids": ["A"],
                "candidate_actions": [],
            },
            {
                "state_id": "complete",
                "state_type": "complete",
                "evidence_ids": ["A", "B"],
                "candidate_actions": [],
            },
        ],
    }

    proposal = {
        "assessment": "refine",
        "stop_assessment": "too_late",
        "confidence": 0.9,
        "rationale": "Expected behavior is already explicit in q.",
        "missing_candidate_requests": [],
        "refined_obligations": [
            {
                "source_obligation_id": "ob_fault",
                "type": "fault_location",
                "description": "Locate fault.",
                "applicable": True,
                "required_for_sufficiency": True,
                "satisfied_by_question": False,
                "witness_groups": [
                    {"evidence_ids": ["A"], "reason": "A locates the fault."}
                ],
            },
            {
                "source_obligation_id": "ob_behavior",
                "type": "behavior_constraint",
                "description": "Expected behavior.",
                "applicable": True,
                "required_for_sufficiency": True,
                "satisfied_by_question": True,
                "witness_groups": [],
            },
        ],
        "proposed_certificate_evidence_ids": ["A"],
    }

    result = verify_and_finalize_refinement(
        task_id="task1",
        supervision=supervision,
        candidate_evidence_ids=["A", "B"],
        existing_evidence_ids={"A", "B"},
        token_costs={"A": 10, "B": 20},
        proposal=proposal,
    )

    assert result["verification_status"] == "accepted"
    assert result["verified_minimal_certificate_evidence_ids"] == ["A"]
    assert result["removed_evidence_ids"] == ["B"]
    assert result["question_satisfied_required_count"] == 1

    # --------------------------------------------------------------
    # JSON "null" 字符串应安全规范化为真正 null。
    # --------------------------------------------------------------

    null_proposal = {
        "assessment": "refine",
        "stop_assessment": "too_early",
        "confidence": 0.9,
        "rationale": "Add fault logic.",
        "missing_candidate_requests": [],
        "refined_obligations": [
            {
                "source_obligation_id": "ob_fault",
                "type": "fault_location",
                "description": "Locate fault.",
                "applicable": True,
                "required_for_sufficiency": True,
                "satisfied_by_question": False,
                "witness_groups": [
                    {
                        "evidence_ids": ["A"],
                        "reason": "A locates fault.",
                    }
                ],
            },
            {
                "source_obligation_id": "null",
                "type": "fault_logic",
                "description": "Explain fault logic.",
                "applicable": True,
                "required_for_sufficiency": True,
                "satisfied_by_question": False,
                "witness_groups": [
                    {
                        "evidence_ids": ["A"],
                        "reason": "A also shows the faulty condition.",
                    }
                ],
            },
        ],
        "proposed_certificate_evidence_ids": ["A"],
    }

    null_result = verify_and_finalize_refinement(
        task_id="task_null",
        supervision=supervision,
        candidate_evidence_ids=["A", "B"],
        existing_evidence_ids={"A", "B"},
        token_costs={"A": 10, "B": 20},
        proposal=null_proposal,
    )

    assert null_result["verification_status"] == "accepted"
    assert len(
        null_result[
            "proposal_format_normalizations"
        ]
    ) == 1
    assert (
        null_result[
            "refined_obligations"
        ][1][
            "source_obligation_id"
        ]
        is None
    )

    # --------------------------------------------------------------
    # Certificate Canonicalization：
    # Teacher graph 明确 A AND B，但 certificate 只写 A。
    # 程序可以机械补成 A+B，因为这不涉及任何新的语义推断。
    # --------------------------------------------------------------

    certificate_supervision = {
        "obligations": [
            {
                "obligation_id": "ob_fault",
                "type": "fault_location",
                "description": "Locate fault.",
                "applicable": True,
                "required_for_sufficiency": True,
                "satisfied_by_question": False,
                "mandatory": True,
                "witness_groups": [
                    {
                        "evidence_ids": ["A"],
                    }
                ],
            }
        ],
        "policy_states": [
            {
                "state_type": "complete",
                "current_evidence_ids": ["A"],
            }
        ],
    }

    incomplete_certificate_proposal = {
        "assessment": "refine",
        "stop_assessment": "too_early",
        "confidence": 0.9,
        "rationale": "A and B are jointly required.",
        "missing_candidate_requests": [],
        "refined_obligations": [
            {
                "source_obligation_id": "ob_fault",
                "type": "fault_location",
                "description": "Locate the full fault context.",
                "applicable": True,
                "required_for_sufficiency": True,
                "satisfied_by_question": False,
                "witness_groups": [
                    {
                        "evidence_ids": ["A", "B"],
                        "reason": "A and B are jointly required.",
                    }
                ],
            }
        ],
        "proposed_certificate_evidence_ids": ["A"],
    }

    incomplete_certificate_result = (
        verify_and_finalize_refinement(
            task_id="task_incomplete_certificate",
            supervision=certificate_supervision,
            candidate_evidence_ids=["A", "B", "C"],
            existing_evidence_ids={"A", "B", "C"},
            token_costs={
                "A": 10,
                "B": 10,
                "C": 10,
            },
            proposal=(
                incomplete_certificate_proposal
            ),
        )
    )

    assert (
        incomplete_certificate_result[
            "verification_status"
        ]
        == "accepted"
    )

    assert (
        incomplete_certificate_result[
            "certificate_normalized_from_teacher_graph"
        ]
        is True
    )

    assert set(
        incomplete_certificate_result[
            "verified_minimal_certificate_evidence_ids"
        ]
    ) == {
        "A",
        "B",
    }

    assert (
        incomplete_certificate_result[
            "teacher_proposed_unsatisfied_obligation_ids"
        ]
    )

    # Teacher graph 只需要 A，但 certificate 写 A+C。
    # C 应机械去除。
    nonminimal_certificate_proposal = json.loads(
        json.dumps(
            incomplete_certificate_proposal
        )
    )

    nonminimal_certificate_proposal[
        "refined_obligations"
    ][0][
        "witness_groups"
    ][0][
        "evidence_ids"
    ] = [
        "A",
    ]

    nonminimal_certificate_proposal[
        "proposed_certificate_evidence_ids"
    ] = [
        "A",
        "C",
    ]

    nonminimal_certificate_result = (
        verify_and_finalize_refinement(
            task_id="task_nonminimal_certificate",
            supervision=certificate_supervision,
            candidate_evidence_ids=["A", "B", "C"],
            existing_evidence_ids={"A", "B", "C"},
            token_costs={
                "A": 10,
                "B": 10,
                "C": 10,
            },
            proposal=(
                nonminimal_certificate_proposal
            ),
        )
    )

    assert (
        nonminimal_certificate_result[
            "verification_status"
        ]
        == "accepted"
    )

    assert (
        nonminimal_certificate_result[
            "verified_minimal_certificate_evidence_ids"
        ]
        == ["A"]
    )

    assert (
        nonminimal_certificate_result[
            "certificate_normalized_from_teacher_graph"
        ]
        is True
    )

    # --------------------------------------------------------------
    # Existing Obligation type 不允许改变。
    # --------------------------------------------------------------

    bad_type = json.loads(
        json.dumps(
            null_proposal
        )
    )

    bad_type[
        "refined_obligations"
    ][0][
        "type"
    ] = "fault_logic"

    try:
        verify_and_finalize_refinement(
            task_id="task_bad_type",
            supervision=supervision,
            candidate_evidence_ids=["A", "B"],
            existing_evidence_ids={"A", "B"},
            token_costs={"A": 10, "B": 20},
            proposal=bad_type,
        )
    except ValueError as exc:
        assert (
            "不允许改变 type"
            in str(exc)
        )
    else:
        raise AssertionError(
            "type mutation should fail"
        )

    # --------------------------------------------------------------
    # 同一个 source_obligation_id 不允许重复映射。
    # --------------------------------------------------------------

    duplicate_source = {
        "assessment": "refine",
        "stop_assessment": "too_early",
        "confidence": 0.9,
        "rationale": "duplicate source id test",
        "missing_candidate_requests": [],
        "refined_obligations": [
            {
                "source_obligation_id": "ob_fault",
                "type": "fault_location",
                "description": "Locate fault 1.",
                "applicable": True,
                "required_for_sufficiency": True,
                "satisfied_by_question": False,
                "witness_groups": [
                    {
                        "evidence_ids": ["A"],
                        "reason": "A",
                    }
                ],
            },
            {
                "source_obligation_id": "ob_fault",
                "type": "fault_location",
                "description": "Locate fault 2.",
                "applicable": True,
                "required_for_sufficiency": True,
                "satisfied_by_question": False,
                "witness_groups": [
                    {
                        "evidence_ids": ["A"],
                        "reason": "A",
                    }
                ],
            },
        ],
        "proposed_certificate_evidence_ids": ["A"],
    }

    try:
        verify_and_finalize_refinement(
            task_id="task_duplicate_source",
            supervision=supervision,
            candidate_evidence_ids=["A", "B"],
            existing_evidence_ids={"A", "B"},
            token_costs={"A": 10, "B": 20},
            proposal=duplicate_source,
        )
    except ValueError as exc:
        assert (
            "source_obligation_id"
            in str(exc)
        )
    else:
        raise AssertionError(
            "duplicate source_obligation_id should fail"
        )


if __name__ == "__main__":
    _self_check()
    print("scripts/refinement_core.py self-check: passed")
