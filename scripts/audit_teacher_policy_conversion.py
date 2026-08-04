#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读审计：教师监督如何转化为 certificate、policy boundary 和候选动作监督。"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_DB = Path("data/.build/unified_swe_v1.sqlite3")
DEFAULT_OUT_DIR = Path("data/.build/audit_teacher_policy_conversion")
INF_COST = 2**30
SQL_BATCH = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "只读分析 selected teacher packets、最终 obligation/witness、"
            "最小充分 certificate、decision_boundary 和 action 标签。"
        )
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="构建 SQLite 路径")
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="审计结果输出目录"
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation"),
        default="all",
        help="只分析指定教师 split；默认 train+validation",
    )
    return parser.parse_args()


def open_readonly_database(path: Path) -> sqlite3.Connection:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到 SQLite：{path}")
    uri = path.as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def load_json(text: str) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("payload_json 顶层必须是 object")
    return value


def batched(values: Sequence[str], size: int = SQL_BATCH) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def distribution(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(values)
    return {str(key): int(counts[key]) for key in sorted(counts, key=lambda x: str(x))}


def percentile(values: Sequence[int], q: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return int(ordered[index])


def summarize_numeric(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "max": max(values),
        "mean": round(sum(values) / len(values), 4),
    }


def applicable_mandatory_obligations(
    obligations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        item
        for item in obligations
        if item.get("applicable") is True and item.get("mandatory") is True
    ]


def evidence_state_metrics(
    evidence_ids: Sequence[str] | set[str],
    obligations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    selected = set(map(str, evidence_ids))
    applicable = [item for item in obligations if item.get("applicable") is True]
    mandatory = [item for item in applicable if item.get("mandatory") is True]
    completed: list[str] = []
    progress_values: list[float] = []

    for obligation in applicable:
        groups = [
            set(map(str, group.get("evidence_ids") or []))
            for group in obligation.get("witness_groups") or []
            if group.get("evidence_ids")
        ]
        if any(group <= selected for group in groups):
            completed.append(str(obligation["obligation_id"]))
        progress_values.append(
            max((len(selected & group) / len(group) for group in groups), default=0.0)
        )

    if mandatory:
        mandatory_ids = {str(item["obligation_id"]) for item in mandatory}
        completion_score: float | None = len(mandatory_ids & set(completed)) / len(mandatory_ids)
    else:
        completion_score = None

    progress_score = (
        sum(progress_values) / len(progress_values) if progress_values else None
    )
    return {
        "completed_obligation_ids": sorted(completed),
        "completion_score": completion_score,
        "progress_score": progress_score,
    }


def minimum_sufficient_certificate(
    obligations: Sequence[dict[str, Any]], token_costs: dict[str, int]
) -> list[str]:
    """与 build_unified_dataset.py 的 _minimum_sufficient_certificate 保持同一规则。"""

    mandatory = applicable_mandatory_obligations(obligations)
    choices = [
        [
            (
                str(group.get("group_id") or ""),
                tuple(sorted(set(map(str, group.get("evidence_ids") or [])))),
            )
            for group in obligation.get("witness_groups") or []
            if group.get("evidence_ids")
        ]
        for obligation in mandatory
    ]
    if not choices or any(not groups for groups in choices):
        return []

    def choice_key(items: Sequence[tuple[str, tuple[str, ...]]]) -> tuple[Any, ...]:
        evidence = sorted(
            {evidence_id for _group_id, group in items for evidence_id in group}
        )
        return (
            sum(int(token_costs.get(evidence_id, INF_COST)) for evidence_id in evidence),
            len(evidence),
            tuple(evidence),
            tuple(group_id for group_id, _group in items),
        )

    combination_count = 1
    for groups in choices:
        combination_count *= len(groups)

    if combination_count <= 4096:
        selected = min(itertools.product(*choices), key=choice_key)
    else:
        selected_list: list[tuple[str, tuple[str, ...]]] = []
        acquired: set[str] = set()
        for groups in choices:
            best = min(
                groups,
                key=lambda item: (
                    sum(
                        int(token_costs.get(evidence_id, INF_COST))
                        for evidence_id in set(item[1]) - acquired
                    ),
                    len(set(item[1]) - acquired),
                    item[0],
                ),
            )
            selected_list.append(best)
            acquired.update(best[1])
        selected = tuple(selected_list)

    return sorted(
        {evidence_id for _group_id, group in selected for evidence_id in group}
    )


def expected_natural_boundary(
    certificate: Sequence[str],
    obligations: Sequence[dict[str, Any]],
    token_costs: dict[str, int],
) -> dict[str, Any] | None:
    """复现 gold_prefix decision_boundary 的选择，不构造 controlled corruption。"""

    if len(certificate) <= 1:
        return None
    candidates: list[dict[str, Any]] = []
    certificate = list(map(str, certificate))
    for removed in certificate:
        evidence = [item for item in certificate if item != removed]
        metrics = evidence_state_metrics(evidence, obligations)
        completion = metrics["completion_score"]
        if completion is not None and completion < 1.0:
            candidates.append(
                {
                    "evidence_ids": sorted(set(evidence)),
                    "removed_evidence_id": removed,
                    **metrics,
                }
            )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            -float(item["completion_score"]),
            -float(item["progress_score"] or 0.0),
            sum(token_costs.get(evidence_id, 0) for evidence_id in item["evidence_ids"]),
            tuple(item["evidence_ids"]),
        ),
    )


def strip_teacher_witnesses(supervision: dict[str, Any]) -> dict[str, Any]:
    """构造“去掉 teacher witness”的反事实近似，不声称恢复精确教师前快照。"""

    obligations: list[dict[str, Any]] = []
    for obligation in supervision.get("obligations") or []:
        groups = [
            dict(group)
            for group in obligation.get("witness_groups") or []
            if str(group.get("source") or "") != "teacher"
        ]
        if not groups:
            # 纯教师创建的 obligation 或最终仅剩教师 witness，反事实中删除。
            continue
        item = dict(obligation)
        item["witness_groups"] = groups
        obligations.append(item)
    return {**supervision, "obligations": obligations}


def load_selected_teacher_packets(
    connection: sqlite3.Connection, split_filter: str
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    packet_split_counts: Counter[str] = Counter()
    obligation_type_counts: Counter[str] = Counter()
    packet_obligation_counts: list[int] = []
    packet_witness_counts: list[int] = []

    for row in connection.execute(
        "SELECT status, payload_json FROM teacher_cache "
        "WHERE status='teacher_verified' ORDER BY input_sha256"
    ):
        payload = load_json(row["payload_json"])
        if payload.get("selected_for_training") is not True:
            continue
        if payload.get("teacher_loss_mask") is not True:
            continue
        split = str(payload.get("split") or "")
        if split_filter != "all" and split != split_filter:
            continue
        task_id = str(payload.get("task_id") or "")
        if not task_id:
            continue
        by_task[task_id].append(payload)
        packet_split_counts[split] += 1

        training = payload.get("training_output") or {}
        obligations = training.get("obligations") or []
        packet_obligation_counts.append(len(obligations))
        witness_count = 0
        for obligation in obligations:
            obligation_type_counts[str(obligation.get("type") or "unknown")] += 1
            witness_count += len(obligation.get("witness_groups") or [])
        packet_witness_counts.append(witness_count)

    unique_task_split_counts: Counter[str] = Counter()
    packets_per_task: list[int] = []
    for task_id, packets in by_task.items():
        packets_per_task.append(len(packets))
        splits = {str(packet.get("split") or "") for packet in packets}
        for split in splits:
            unique_task_split_counts[split] += 1

    return dict(by_task), {
        "selected_packet_count": sum(packet_split_counts.values()),
        "selected_packet_counts_by_split": dict(sorted(packet_split_counts.items())),
        "selected_unique_task_count": len(by_task),
        "selected_unique_task_counts_by_split": dict(sorted(unique_task_split_counts.items())),
        "packets_per_task": summarize_numeric(packets_per_task),
        "packets_per_task_distribution": distribution(packets_per_task),
        "teacher_output_obligation_types": dict(sorted(obligation_type_counts.items())),
        "teacher_output_obligations_per_packet": summarize_numeric(packet_obligation_counts),
        "teacher_output_witness_groups_per_packet": summarize_numeric(packet_witness_counts),
    }


def load_split_by_task(connection: sqlite3.Connection) -> dict[str, str]:
    return {
        str(row["task_id"]): str(row["final_split"])
        for row in connection.execute("SELECT task_id, final_split FROM canonical_tasks")
    }


def load_supervision_for_tasks(
    connection: sqlite3.Connection, task_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for batch in batched(sorted(task_ids)):
        placeholders = ",".join("?" for _ in batch)
        query = f"SELECT task_id, payload_json FROM supervision WHERE task_id IN ({placeholders})"
        for row in connection.execute(query, batch):
            result[str(row["task_id"])] = load_json(row["payload_json"])
    return result


def load_token_costs(
    connection: sqlite3.Connection, evidence_ids: Sequence[str]
) -> dict[str, int]:
    costs: dict[str, int] = {}
    for batch in batched(sorted(set(evidence_ids))):
        placeholders = ",".join("?" for _ in batch)
        query = (
            "SELECT evidence_id, rendered_token_count FROM evidence_units "
            f"WHERE evidence_id IN ({placeholders})"
        )
        for row in connection.execute(query, batch):
            costs[str(row["evidence_id"])] = int(row["rendered_token_count"] or 0)
    return costs


def load_policy_states(
    connection: sqlite3.Connection,
    selected_task_ids: set[str],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    state_by_id: dict[str, dict[str, Any]] = {}
    global_types: Counter[str] = Counter()
    selected_types: Counter[str] = Counter()
    nonselected_types: Counter[str] = Counter()

    for row in connection.execute(
        "SELECT state_id, task_id, payload_json FROM policy_states ORDER BY task_id, state_id"
    ):
        task_id = str(row["task_id"])
        payload = load_json(row["payload_json"])
        state_id = str(row["state_id"])
        payload["state_id"] = state_id
        payload["task_id"] = task_id
        state_type = str(payload.get("state_type") or "unknown")
        global_types[state_type] += 1
        if task_id in selected_task_ids:
            selected_types[state_type] += 1
            by_task[task_id].append(payload)
            state_by_id[state_id] = payload
        else:
            nonselected_types[state_type] += 1

    return dict(by_task), state_by_id, {
        "global_state_type_counts": dict(sorted(global_types.items())),
        "selected_teacher_state_type_counts": dict(sorted(selected_types.items())),
        "nonselected_state_type_counts": dict(sorted(nonselected_types.items())),
    }


def load_actions_for_states(
    connection: sqlite3.Connection,
    state_ids: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for batch in batched(sorted(set(state_ids))):
        placeholders = ",".join("?" for _ in batch)
        query = (
            "SELECT state_id, payload_json FROM candidate_actions "
            f"WHERE state_id IN ({placeholders}) ORDER BY state_id, action_key"
        )
        for row in connection.execute(query, batch):
            by_state[str(row["state_id"])].append(load_json(row["payload_json"]))
    return dict(by_state)


def action_summary(actions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    known_positive = [
        action
        for action in actions
        if action.get("action_loss_mask") is True and action.get("action_label") == "positive"
    ]
    known_negative = [
        action
        for action in actions
        if action.get("action_loss_mask") is True and action.get("action_label") == "negative"
    ]
    unknown = [action for action in actions if action.get("action_loss_mask") is not True]
    positive_nonstop = [
        action for action in known_positive if action.get("action_type") != "stop"
    ]
    positive_online = [
        action
        for action in positive_nonstop
        if action.get("candidate_scope") == "online"
    ]
    positive_injected = [
        action
        for action in positive_nonstop
        if action.get("candidate_scope") == "offline_injected"
    ]
    return {
        "action_count": len(actions),
        "known_positive_count": len(known_positive),
        "known_negative_count": len(known_negative),
        "unknown_count": len(unknown),
        "positive_nonstop_count": len(positive_nonstop),
        "positive_online_count": len(positive_online),
        "positive_offline_injected_count": len(positive_injected),
        "has_positive_nonstop": bool(positive_nonstop),
        "has_online_positive": bool(positive_online),
        "has_offline_injected_positive": bool(positive_injected),
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    connection = open_readonly_database(args.db)
    try:
        teacher_by_task, teacher_report = load_selected_teacher_packets(
            connection, args.split
        )
        selected_task_ids = set(teacher_by_task)
        if not selected_task_ids:
            raise RuntimeError("没有找到 selected_for_training=true 的 teacher_verified 包。")

        split_by_task = load_split_by_task(connection)
        supervision_by_task = load_supervision_for_tasks(
            connection, sorted(selected_task_ids)
        )
        missing_supervision = sorted(selected_task_ids - set(supervision_by_task))
        if missing_supervision:
            raise RuntimeError(
                f"selected teacher task 缺少 supervision：count={len(missing_supervision)}, "
                f"first={missing_supervision[:3]}"
            )

        all_witness_ids: set[str] = set()
        for supervision in supervision_by_task.values():
            for obligation in supervision.get("obligations") or []:
                for group in obligation.get("witness_groups") or []:
                    all_witness_ids.update(map(str, group.get("evidence_ids") or []))
        token_costs = load_token_costs(connection, sorted(all_witness_ids))

        states_by_task, state_by_id, policy_type_report = load_policy_states(
            connection, selected_task_ids
        )
        actions_by_state = load_actions_for_states(connection, list(state_by_id))

        task_rows: list[dict[str, Any]] = []
        final_certificate_sizes: list[int] = []
        approx_certificate_sizes: list[int] = []
        final_mandatory_counts: list[int] = []
        final_obligation_counts: list[int] = []
        final_witness_counts: list[int] = []
        teacher_group_counts: list[int] = []

        counters: Counter[str] = Counter()
        boundary_sources: Counter[str] = Counter()
        final_certificate_size_distribution: Counter[int] = Counter()
        approx_certificate_size_distribution: Counter[int] = Counter()
        final_and_group_size_distribution: Counter[int] = Counter()
        teacher_and_group_size_distribution: Counter[int] = Counter()

        selected_noncomplete_state_count = 0
        selected_noncomplete_with_online_positive = 0
        selected_noncomplete_with_offline_positive = 0
        selected_noncomplete_offline_only_positive = 0
        selected_noncomplete_without_positive = 0
        selected_ranking_active_state_count = 0

        for task_id in sorted(selected_task_ids):
            supervision = supervision_by_task[task_id]
            obligations = list(supervision.get("obligations") or [])
            approx_supervision = strip_teacher_witnesses(supervision)
            approx_obligations = list(approx_supervision.get("obligations") or [])

            final_certificate = minimum_sufficient_certificate(obligations, token_costs)
            approx_certificate = minimum_sufficient_certificate(
                approx_obligations, token_costs
            )
            expected_boundary = expected_natural_boundary(
                final_certificate, obligations, token_costs
            )
            approx_expected_boundary = expected_natural_boundary(
                approx_certificate, approx_obligations, token_costs
            )

            states = states_by_task.get(task_id, [])
            state_types = [str(state.get("state_type") or "unknown") for state in states]
            boundaries = [state for state in states if state.get("state_type") == "decision_boundary"]
            complete_states = [state for state in states if state.get("state_type") == "complete"]
            initial_states = [state for state in states if state.get("state_type") == "initial"]

            for boundary in boundaries:
                boundary_sources[str(boundary.get("label_source") or "unknown")] += 1

            if boundaries:
                counters["tasks_with_actual_boundary"] += 1
            if expected_boundary is not None:
                counters["tasks_with_expected_natural_boundary"] += 1
            if approx_expected_boundary is not None:
                counters["tasks_with_approx_no_teacher_natural_boundary"] += 1
            if expected_boundary is not None and not boundaries:
                counters["expected_natural_boundary_missing"] += 1
            if boundaries and expected_boundary is None:
                # 这类通常应为 controlled_corruption。
                counters["boundary_without_natural_certificate_boundary"] += 1

            actual_gold_prefix = [
                state
                for state in boundaries
                if state.get("label_source") == "gold_prefix"
            ]
            actual_controlled = [
                state
                for state in boundaries
                if state.get("label_source") == "controlled_corruption"
            ]
            counters["actual_gold_prefix_boundary_count"] += len(actual_gold_prefix)
            counters["actual_controlled_corruption_boundary_count"] += len(actual_controlled)

            if expected_boundary is not None and actual_gold_prefix:
                expected_ids = set(expected_boundary["evidence_ids"])
                if any(set(state.get("evidence_ids") or []) == expected_ids for state in actual_gold_prefix):
                    counters["expected_natural_boundary_matched"] += 1
                else:
                    counters["expected_natural_boundary_mismatched"] += 1

            complete_matches_certificate = False
            if len(complete_states) == 1:
                complete_matches_certificate = (
                    set(map(str, complete_states[0].get("evidence_ids") or []))
                    == set(final_certificate)
                )
            if complete_matches_certificate:
                counters["complete_state_matches_recomputed_certificate"] += 1
            else:
                counters["complete_state_certificate_mismatch"] += 1

            teacher_groups = [
                group
                for obligation in obligations
                for group in obligation.get("witness_groups") or []
                if str(group.get("source") or "") == "teacher"
            ]
            all_groups = [
                group
                for obligation in obligations
                for group in obligation.get("witness_groups") or []
                if group.get("evidence_ids")
            ]
            for group in all_groups:
                final_and_group_size_distribution[len(group.get("evidence_ids") or [])] += 1
            for group in teacher_groups:
                teacher_and_group_size_distribution[len(group.get("evidence_ids") or [])] += 1

            mandatory_count = len(applicable_mandatory_obligations(obligations))
            teacher_only_obligation_count = sum(
                str(obligation.get("construction_method") or "") == "teacher_rule_verified"
                for obligation in obligations
            )

            final_obligation_counts.append(len(obligations))
            final_mandatory_counts.append(mandatory_count)
            final_witness_counts.append(len(all_groups))
            teacher_group_counts.append(len(teacher_groups))
            final_certificate_sizes.append(len(final_certificate))
            approx_certificate_sizes.append(len(approx_certificate))
            final_certificate_size_distribution[len(final_certificate)] += 1
            approx_certificate_size_distribution[len(approx_certificate)] += 1

            if final_certificate != approx_certificate:
                counters["certificate_changed_vs_no_teacher_witness_approx"] += 1
            if len(final_certificate) > len(approx_certificate):
                counters["certificate_size_increased_vs_no_teacher_witness_approx"] += 1
            if expected_boundary is not None and approx_expected_boundary is None:
                counters["natural_boundary_added_vs_no_teacher_witness_approx"] += 1

            initial_action_stats = {
                "has_online_positive": False,
                "has_offline_injected_positive": False,
                "positive_nonstop_count": 0,
            }
            boundary_action_stats = {
                "has_online_positive": False,
                "has_offline_injected_positive": False,
                "positive_nonstop_count": 0,
            }

            for state in states:
                actions = actions_by_state.get(str(state["state_id"]), [])
                summary = action_summary(actions)
                if state.get("ranking_loss_mask") is True:
                    selected_ranking_active_state_count += 1
                if state.get("state_type") != "complete":
                    selected_noncomplete_state_count += 1
                    if summary["has_online_positive"]:
                        selected_noncomplete_with_online_positive += 1
                    if summary["has_offline_injected_positive"]:
                        selected_noncomplete_with_offline_positive += 1
                    if summary["has_offline_injected_positive"] and not summary["has_online_positive"]:
                        selected_noncomplete_offline_only_positive += 1
                    if not summary["has_positive_nonstop"]:
                        selected_noncomplete_without_positive += 1
                if state.get("state_type") == "initial":
                    initial_action_stats = summary
                elif state.get("state_type") == "decision_boundary":
                    boundary_action_stats = summary

            task_rows.append(
                {
                    "task_id": task_id,
                    "split": split_by_task.get(task_id, ""),
                    "selected_teacher_packet_count": len(teacher_by_task[task_id]),
                    "final_supervision_level": str(supervision.get("level") or ""),
                    "final_obligation_count": len(obligations),
                    "final_mandatory_obligation_count": mandatory_count,
                    "teacher_only_obligation_count": teacher_only_obligation_count,
                    "final_witness_group_count": len(all_groups),
                    "teacher_witness_group_count": len(teacher_groups),
                    "final_certificate_size": len(final_certificate),
                    "approx_no_teacher_certificate_size": len(approx_certificate),
                    "certificate_changed_vs_no_teacher_approx": final_certificate != approx_certificate,
                    "expected_natural_boundary": expected_boundary is not None,
                    "approx_no_teacher_expected_natural_boundary": approx_expected_boundary is not None,
                    "actual_boundary_count": len(boundaries),
                    "actual_boundary_sources": ";".join(
                        sorted(str(state.get("label_source") or "unknown") for state in boundaries)
                    ),
                    "complete_matches_recomputed_certificate": complete_matches_certificate,
                    "state_types": ";".join(state_types),
                    "initial_stop_label": str(initial_states[0].get("stop_label") if initial_states else ""),
                    "complete_stop_label": str(complete_states[0].get("stop_label") if complete_states else ""),
                    "initial_positive_nonstop_count": initial_action_stats["positive_nonstop_count"],
                    "initial_has_online_positive": initial_action_stats["has_online_positive"],
                    "initial_has_offline_injected_positive": initial_action_stats["has_offline_injected_positive"],
                    "boundary_positive_nonstop_count": boundary_action_stats["positive_nonstop_count"],
                    "boundary_has_online_positive": boundary_action_stats["has_online_positive"],
                    "boundary_has_offline_injected_positive": boundary_action_stats["has_offline_injected_positive"],
                }
            )

        selected_unique_tasks = len(selected_task_ids)
        actual_boundary_tasks = counters["tasks_with_actual_boundary"]
        expected_boundary_tasks = counters["tasks_with_expected_natural_boundary"]

        report = {
            "database": str(args.db.resolve()),
            "mode": "read_only",
            "scope": args.split,
            "important_note": (
                "当前 supervision 已被 policy 阶段写回为教师合并后的最终版本。"
                "因此 approx_no_teacher_* 仅通过删除 source=teacher 的 witness group 构造反事实近似，"
                "不是精确的教师前历史快照。"
            ),
            "teacher": teacher_report,
            "final_supervision_on_selected_teacher_tasks": {
                "obligation_count_per_task": summarize_numeric(final_obligation_counts),
                "mandatory_obligation_count_per_task": summarize_numeric(final_mandatory_counts),
                "witness_group_count_per_task": summarize_numeric(final_witness_counts),
                "teacher_witness_group_count_per_task": summarize_numeric(teacher_group_counts),
                "and_witness_group_size_distribution": {
                    str(k): int(v) for k, v in sorted(final_and_group_size_distribution.items())
                },
                "teacher_and_witness_group_size_distribution": {
                    str(k): int(v) for k, v in sorted(teacher_and_group_size_distribution.items())
                },
            },
            "certificate": {
                "final_certificate_size_distribution": {
                    str(k): int(v) for k, v in sorted(final_certificate_size_distribution.items())
                },
                "final_certificate_size_summary": summarize_numeric(final_certificate_sizes),
                "approx_no_teacher_certificate_size_distribution": {
                    str(k): int(v) for k, v in sorted(approx_certificate_size_distribution.items())
                },
                "approx_no_teacher_certificate_size_summary": summarize_numeric(approx_certificate_sizes),
                "certificate_changed_vs_no_teacher_witness_approx_task_count": int(
                    counters["certificate_changed_vs_no_teacher_witness_approx"]
                ),
                "certificate_size_increased_vs_no_teacher_witness_approx_task_count": int(
                    counters["certificate_size_increased_vs_no_teacher_witness_approx"]
                ),
            },
            "policy_state_conversion": {
                **policy_type_report,
                "selected_teacher_task_count": selected_unique_tasks,
                "selected_teacher_tasks_with_actual_boundary": int(actual_boundary_tasks),
                "selected_teacher_actual_boundary_task_rate": round(
                    actual_boundary_tasks / selected_unique_tasks, 6
                ) if selected_unique_tasks else None,
                "selected_teacher_tasks_with_expected_natural_boundary": int(expected_boundary_tasks),
                "selected_teacher_expected_natural_boundary_task_rate": round(
                    expected_boundary_tasks / selected_unique_tasks, 6
                ) if selected_unique_tasks else None,
                "selected_teacher_tasks_with_approx_no_teacher_natural_boundary": int(
                    counters["tasks_with_approx_no_teacher_natural_boundary"]
                ),
                "natural_boundary_added_vs_no_teacher_witness_approx_task_count": int(
                    counters["natural_boundary_added_vs_no_teacher_witness_approx"]
                ),
                "expected_natural_boundary_missing_task_count": int(
                    counters["expected_natural_boundary_missing"]
                ),
                "expected_natural_boundary_matched_task_count": int(
                    counters["expected_natural_boundary_matched"]
                ),
                "expected_natural_boundary_mismatched_task_count": int(
                    counters["expected_natural_boundary_mismatched"]
                ),
                "boundary_without_natural_certificate_boundary_task_count": int(
                    counters["boundary_without_natural_certificate_boundary"]
                ),
                "boundary_label_source_counts": dict(sorted(boundary_sources.items())),
                "complete_state_matches_recomputed_certificate_task_count": int(
                    counters["complete_state_matches_recomputed_certificate"]
                ),
                "complete_state_certificate_mismatch_task_count": int(
                    counters["complete_state_certificate_mismatch"]
                ),
            },
            "action_supervision_on_selected_teacher_states": {
                "ranking_active_state_count": selected_ranking_active_state_count,
                "noncomplete_state_count": selected_noncomplete_state_count,
                "noncomplete_states_with_online_positive": selected_noncomplete_with_online_positive,
                "noncomplete_states_with_online_positive_rate": round(
                    selected_noncomplete_with_online_positive / selected_noncomplete_state_count, 6
                ) if selected_noncomplete_state_count else None,
                "noncomplete_states_with_offline_injected_positive": selected_noncomplete_with_offline_positive,
                "noncomplete_states_with_offline_injected_positive_rate": round(
                    selected_noncomplete_with_offline_positive / selected_noncomplete_state_count, 6
                ) if selected_noncomplete_state_count else None,
                "noncomplete_states_with_offline_only_positive": selected_noncomplete_offline_only_positive,
                "noncomplete_states_with_offline_only_positive_rate": round(
                    selected_noncomplete_offline_only_positive / selected_noncomplete_state_count, 6
                ) if selected_noncomplete_state_count else None,
                "noncomplete_states_without_known_positive": selected_noncomplete_without_positive,
            },
            "consistency_counters": {key: int(value) for key, value in sorted(counters.items())},
            "outputs": {
                "task_csv": str((args.out_dir / "teacher_policy_tasks.csv").resolve()),
                "report_json": str((args.out_dir / "report.json").resolve()),
            },
        }

        csv_path = args.out_dir / "teacher_policy_tasks.csv"
        if task_rows:
            with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(task_rows[0].keys()))
                writer.writeheader()
                writer.writerows(task_rows)

        report_path = args.out_dir / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
