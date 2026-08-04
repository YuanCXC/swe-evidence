#!/usr/bin/env python3
"""审计 Unified SWE Dataset 的 Retriever / Policy 覆盖率。

只读分析 data/.build/unified_swe_v1.sqlite3，不修改任何构建状态。
重点回答：
1. 非 complete state 的正确动作有多少真正来自 online Retriever。
2. online positive single 的 Recall@1/5/10/20/64。
3. pair 正动作是否能由 online single Top-K 组成。
4. offline-only positive 是在“文件级候选 universe”被丢掉，
   还是文件已进入候选但具体 Evidence Unit / pair 没被召回。
5. bm25_content/path_name/symbol/structure 对正确 online single 的贡献。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_DB = Path("data/.build/unified_swe_v1.sqlite3")
DEFAULT_OUTPUT_DIR = Path("data/.build/audit_retriever_policy_coverage")
DEFAULT_KS = (1, 5, 10, 20, 64)


def stable_json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def open_readonly_database(path: Path) -> sqlite3.Connection:
    """以 SQLite mode=ro 打开状态库，防止审计脚本意外写入。"""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"找不到构建状态库：{resolved}")
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def percentile(values: Sequence[int | float], q: float) -> float | int | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def numeric_summary(values: Sequence[int | float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "max": max(values),
        "mean": round(sum(values) / len(values), 6),
    }


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def load_task_splits(
    connection: sqlite3.Connection,
    allowed_splits: set[str] | None,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in connection.execute(
        "SELECT task_id, final_split FROM canonical_tasks "
        "WHERE status='normalized' ORDER BY task_id"
    ):
        split = str(row["final_split"])
        if allowed_splits is not None and split not in allowed_splits:
            continue
        result[str(row["task_id"])] = split
    return result


def load_state_metadata(
    connection: sqlite3.Connection,
    task_splits: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """只加载 4 万级 policy state 元数据，不加载 action。"""

    states: dict[str, dict[str, Any]] = {}
    for row in connection.execute(
        "SELECT state_id, task_id, payload_json "
        "FROM policy_states ORDER BY state_id"
    ):
        task_id = str(row["task_id"])
        if task_id not in task_splits:
            continue
        payload = json.loads(row["payload_json"])
        states[str(row["state_id"])] = {
            "state_id": str(row["state_id"]),
            "task_id": task_id,
            "split": task_splits[task_id],
            "state_type": str(payload.get("state_type") or ""),
            "step": int(payload.get("step") or 0),
            "evidence_ids": tuple(map(str, payload.get("evidence_ids") or [])),
            "completion_score": payload.get("completion_score"),
            "progress_score": payload.get("progress_score"),
            "ranking_loss_mask": bool(payload.get("ranking_loss_mask")),
            "stop_label": str(payload.get("stop_label") or ""),
            "stop_loss_mask": bool(payload.get("stop_loss_mask")),
            "candidate_pool_stats": payload.get("candidate_pool_stats") or {},
        }
    return states


def new_action_accumulator() -> dict[str, Any]:
    return {
        "positive_nonstop_count": 0,
        "negative_nonstop_count": 0,
        "known_nonstop_count": 0,
        "online_positive_action_count": 0,
        "offline_positive_action_count": 0,
        "online_positive_single_count": 0,
        "online_positive_pair_count": 0,
        "offline_positive_single_count": 0,
        "offline_positive_pair_count": 0,
        "best_positive_single_rank": None,
        "online_single_count": 0,
        "offline_injected_single_count": 0,
        "online_pair_count": 0,
        "offline_pair_count": 0,
        "scoreable_action_count": 0,
        "loss_active_action_count": 0,
        "online_single_ids": set(),
        "online_single_rank_by_id": {},
        "offline_positive_actions": [],
        "positive_pair_actions": [],
        "positive_online_single_sources": [],
    }


def update_action_accumulator(
    accumulator: dict[str, Any],
    action: dict[str, Any],
) -> None:
    action_type = str(action.get("action_type") or "")
    scope = str(action.get("candidate_scope") or "")
    label = str(action.get("action_label") or "")
    loss_mask = bool(action.get("action_loss_mask"))
    scoreable = bool(action.get("scoreable"))
    evidence_ids = tuple(map(str, action.get("evidence_ids") or []))

    if scoreable:
        accumulator["scoreable_action_count"] += 1
    if loss_mask:
        accumulator["loss_active_action_count"] += 1

    if action_type == "single":
        if scope == "online":
            accumulator["online_single_count"] += 1
            if evidence_ids:
                evidence_id = evidence_ids[0]
                accumulator["online_single_ids"].add(evidence_id)
                rank = action.get("online_retrieval_rank")
                if rank is not None:
                    rank = int(rank)
                    accumulator["online_single_rank_by_id"][evidence_id] = rank
        elif scope == "offline_injected":
            accumulator["offline_injected_single_count"] += 1
    elif action_type == "pair":
        if scope == "online":
            accumulator["online_pair_count"] += 1
        elif scope == "offline_injected":
            accumulator["offline_pair_count"] += 1

    if action_type == "stop":
        return

    if loss_mask:
        accumulator["known_nonstop_count"] += 1
        if label == "negative":
            accumulator["negative_nonstop_count"] += 1

    if not (loss_mask and label == "positive"):
        return

    accumulator["positive_nonstop_count"] += 1

    if scope == "online":
        accumulator["online_positive_action_count"] += 1
    elif scope == "offline_injected":
        accumulator["offline_positive_action_count"] += 1
        accumulator["offline_positive_actions"].append(
            {
                "action_type": action_type,
                "evidence_ids": evidence_ids,
            }
        )

    if action_type == "single":
        if scope == "online":
            accumulator["online_positive_single_count"] += 1
            rank = action.get("online_retrieval_rank")
            if rank is not None:
                rank = int(rank)
                best = accumulator["best_positive_single_rank"]
                if best is None or rank < best:
                    accumulator["best_positive_single_rank"] = rank
            accumulator["positive_online_single_sources"].append(
                tuple(map(str, action.get("candidate_sources") or []))
            )
        elif scope == "offline_injected":
            accumulator["offline_positive_single_count"] += 1
    elif action_type == "pair":
        accumulator["positive_pair_actions"].append(
            {
                "scope": scope,
                "evidence_ids": evidence_ids,
            }
        )
        if scope == "online":
            accumulator["online_positive_pair_count"] += 1
        elif scope == "offline_injected":
            accumulator["offline_positive_pair_count"] += 1


def finalize_state_record(
    state_meta: dict[str, Any],
    accumulator: dict[str, Any],
    ks: Sequence[int],
) -> dict[str, Any]:
    online_single_rank_by_id: dict[str, int] = accumulator["online_single_rank_by_id"]
    positive_pairs = accumulator["positive_pair_actions"]

    pair_realizable_at_k: dict[str, bool] = {}
    for k in ks:
        pair_realizable_at_k[str(k)] = any(
            len(pair["evidence_ids"]) == 2
            and all(
                online_single_rank_by_id.get(evidence_id, 2**31 - 1) <= k
                for evidence_id in pair["evidence_ids"]
            )
            for pair in positive_pairs
        )

    best_positive_rank = accumulator["best_positive_single_rank"]
    positive_single_recall_at_k = {
        str(k): bool(best_positive_rank is not None and best_positive_rank <= k)
        for k in ks
    }

    positive_online_channels: set[str] = set()
    for sources in accumulator["positive_online_single_sources"]:
        positive_online_channels.update(sources)

    return {
        **state_meta,
        "positive_nonstop_count": accumulator["positive_nonstop_count"],
        "negative_nonstop_count": accumulator["negative_nonstop_count"],
        "known_nonstop_count": accumulator["known_nonstop_count"],
        "online_positive_action_count": accumulator["online_positive_action_count"],
        "offline_positive_action_count": accumulator["offline_positive_action_count"],
        "online_positive_single_count": accumulator["online_positive_single_count"],
        "online_positive_pair_count": accumulator["online_positive_pair_count"],
        "offline_positive_single_count": accumulator["offline_positive_single_count"],
        "offline_positive_pair_count": accumulator["offline_positive_pair_count"],
        "best_positive_single_rank": best_positive_rank,
        "positive_single_recall_at_k": positive_single_recall_at_k,
        "positive_pair_realizable_from_online_singles_at_k": pair_realizable_at_k,
        "online_single_count": accumulator["online_single_count"],
        "offline_injected_single_count": accumulator["offline_injected_single_count"],
        "online_pair_count": accumulator["online_pair_count"],
        "offline_pair_count": accumulator["offline_pair_count"],
        "scoreable_action_count": accumulator["scoreable_action_count"],
        "loss_active_action_count": accumulator["loss_active_action_count"],
        "positive_online_channels": sorted(positive_online_channels),
        "_online_single_ids": set(accumulator["online_single_ids"]),
        "_offline_positive_actions": list(accumulator["offline_positive_actions"]),
    }


def stream_state_action_records(
    connection: sqlite3.Connection,
    states: dict[str, dict[str, Any]],
    ks: Sequence[int],
) -> list[dict[str, Any]]:
    """按 state_id 流式扫描 288 万 action，只保留 state 级汇总。"""

    records: list[dict[str, Any]] = []
    current_state_id: str | None = None
    current_accumulator: dict[str, Any] | None = None

    cursor = connection.execute(
        "SELECT state_id, payload_json FROM candidate_actions "
        "ORDER BY state_id, action_key"
    )

    processed_actions = 0
    for row in cursor:
        state_id = str(row["state_id"])
        if state_id not in states:
            continue

        if current_state_id != state_id:
            if current_state_id is not None and current_accumulator is not None:
                records.append(
                    finalize_state_record(
                        states[current_state_id],
                        current_accumulator,
                        ks,
                    )
                )
            current_state_id = state_id
            current_accumulator = new_action_accumulator()

        assert current_accumulator is not None
        update_action_accumulator(
            current_accumulator,
            json.loads(row["payload_json"]),
        )
        processed_actions += 1
        if processed_actions == 1 or processed_actions % 250_000 == 0:
            print(
                f"audit-actions: {processed_actions}",
                flush=True,
            )

    if current_state_id is not None and current_accumulator is not None:
        records.append(
            finalize_state_record(
                states[current_state_id],
                current_accumulator,
                ks,
            )
        )

    state_ids_with_actions = {record["state_id"] for record in records}
    missing = sorted(set(states) - state_ids_with_actions)
    if missing:
        raise ValueError(
            f"有 {len(missing)} 个 policy state 没有 candidate action，"
            f"first={missing[0]}"
        )

    return records


def iter_chunks(values: Sequence[str], size: int = 800) -> Iterable[list[str]]:
    for offset in range(0, len(values), size):
        yield list(values[offset : offset + size])


def load_evidence_file_mapping(
    connection: sqlite3.Connection,
    evidence_ids: set[str],
) -> dict[str, str]:
    """只查询 miss-state 涉及的 Evidence ID，不加载 2549 万 Evidence Unit。"""

    result: dict[str, str] = {}
    ordered = sorted(evidence_ids)
    for index, chunk in enumerate(iter_chunks(ordered), 1):
        placeholders = ",".join("?" for _ in chunk)
        for row in connection.execute(
            f"SELECT evidence_id, file_version_id FROM evidence_units "
            f"WHERE evidence_id IN ({placeholders})",
            chunk,
        ):
            result[str(row["evidence_id"])] = str(row["file_version_id"])
        if index == 1 or index % 100 == 0:
            print(
                f"audit-evidence-map: {min(index * 800, len(ordered))}/{len(ordered)}",
                flush=True,
            )
    missing = sorted(evidence_ids - set(result))
    if missing:
        raise ValueError(
            f"有 {len(missing)} 个审计 Evidence ID 无法映射到 file_version，"
            f"first={missing[0]}"
        )
    return result


def classify_offline_only_misses(
    records: list[dict[str, Any]],
    connection: sqlite3.Connection,
) -> None:
    """把 offline-only positive 进一步拆成文件 gate / unit / pair 生成问题。"""

    target_records = [
        record
        for record in records
        if record["state_type"] != "complete"
        and record["positive_nonstop_count"] > 0
        and record["online_positive_action_count"] == 0
        and record["_offline_positive_actions"]
    ]

    evidence_ids: set[str] = set()
    for record in target_records:
        evidence_ids.update(record["_online_single_ids"])
        for action in record["_offline_positive_actions"]:
            evidence_ids.update(action["evidence_ids"])

    mapping = load_evidence_file_mapping(connection, evidence_ids) if evidence_ids else {}

    for record in records:
        record["offline_only_miss_class"] = None
        record["positive_files_absent_from_online_candidate_files"] = 0
        record["positive_files_present_in_online_candidate_files"] = 0
        record["positive_pair_generation_miss"] = False

    for record in target_records:
        online_single_ids = set(record["_online_single_ids"])
        online_files = {
            mapping[evidence_id]
            for evidence_id in online_single_ids
            if evidence_id in mapping
        }
        positive_ids = {
            evidence_id
            for action in record["_offline_positive_actions"]
            for evidence_id in action["evidence_ids"]
        }
        positive_files = {mapping[evidence_id] for evidence_id in positive_ids}
        absent_files = positive_files - online_files
        present_files = positive_files & online_files

        pair_generation_miss = any(
            action["action_type"] == "pair"
            and len(action["evidence_ids"]) == 2
            and set(action["evidence_ids"]) <= online_single_ids
            for action in record["_offline_positive_actions"]
        )

        if pair_generation_miss:
            miss_class = "pair_generation_miss"
        elif positive_files and not present_files:
            miss_class = "file_universe_miss"
        elif present_files:
            miss_class = "within_file_or_unit_retrieval_miss"
        else:
            miss_class = "unclassified"

        record["offline_only_miss_class"] = miss_class
        record["positive_files_absent_from_online_candidate_files"] = len(absent_files)
        record["positive_files_present_in_online_candidate_files"] = len(present_files)
        record["positive_pair_generation_miss"] = pair_generation_miss


def aggregate_records(
    records: Sequence[dict[str, Any]],
    ks: Sequence[int],
) -> dict[str, Any]:
    noncomplete = [record for record in records if record["state_type"] != "complete"]
    known_positive = [
        record for record in noncomplete if record["positive_nonstop_count"] > 0
    ]

    result: dict[str, Any] = {
        "state_count": len(records),
        "noncomplete_state_count": len(noncomplete),
        "ranking_active_state_count": sum(
            bool(record["ranking_loss_mask"]) for record in records
        ),
        "noncomplete_states_with_known_positive": len(known_positive),
        "noncomplete_states_without_known_positive": sum(
            record["positive_nonstop_count"] == 0 for record in noncomplete
        ),
        "noncomplete_states_with_online_positive_action": sum(
            record["online_positive_action_count"] > 0 for record in known_positive
        ),
        "noncomplete_states_with_offline_only_positive": sum(
            record["online_positive_action_count"] == 0
            and record["offline_positive_action_count"] > 0
            for record in known_positive
        ),
        "noncomplete_states_with_any_offline_injected_positive": sum(
            record["offline_positive_action_count"] > 0 for record in known_positive
        ),
        "noncomplete_states_with_online_positive_pair": sum(
            record["online_positive_pair_count"] > 0 for record in known_positive
        ),
        "noncomplete_states_with_online_positive_single": sum(
            record["online_positive_single_count"] > 0 for record in known_positive
        ),
        "online_single_count_per_noncomplete_state": numeric_summary(
            [record["online_single_count"] for record in noncomplete]
        ),
        "offline_injected_single_count_per_noncomplete_state": numeric_summary(
            [record["offline_injected_single_count"] for record in noncomplete]
        ),
        "pair_count_per_noncomplete_state": numeric_summary(
            [
                record["online_pair_count"] + record["offline_pair_count"]
                for record in noncomplete
            ]
        ),
        "loss_active_action_count_per_state": numeric_summary(
            [record["loss_active_action_count"] for record in records]
        ),
        "offline_only_miss_class_counts": dict(
            sorted(
                Counter(
                    record["offline_only_miss_class"]
                    for record in known_positive
                    if record["offline_only_miss_class"] is not None
                ).items()
            )
        ),
    }

    denominator = len(known_positive)
    result["online_positive_action_coverage_rate"] = safe_rate(
        result["noncomplete_states_with_online_positive_action"], denominator
    )
    result["offline_only_positive_rate"] = safe_rate(
        result["noncomplete_states_with_offline_only_positive"], denominator
    )
    result["any_offline_injected_positive_rate"] = safe_rate(
        result["noncomplete_states_with_any_offline_injected_positive"], denominator
    )

    recall = {}
    pair_realizable = {}
    for k in ks:
        key = str(k)
        recall[key] = {
            "state_hit_count": sum(
                bool(record["positive_single_recall_at_k"][key])
                for record in known_positive
            ),
            "state_hit_rate": safe_rate(
                sum(
                    bool(record["positive_single_recall_at_k"][key])
                    for record in known_positive
                ),
                denominator,
            ),
        }
        pair_realizable[key] = {
            "state_hit_count": sum(
                bool(record["positive_pair_realizable_from_online_singles_at_k"][key])
                for record in known_positive
            ),
            "state_hit_rate": safe_rate(
                sum(
                    bool(record["positive_pair_realizable_from_online_singles_at_k"][key])
                    for record in known_positive
                ),
                denominator,
            ),
        }
    result["positive_online_single_recall_at_k"] = recall
    result["positive_pair_realizable_from_online_singles_at_k"] = pair_realizable

    source_state_counts = Counter()
    for record in known_positive:
        for source in record["positive_online_channels"]:
            source_state_counts[source] += 1
    result["positive_online_single_channel_state_hits"] = {
        source: {
            "state_hit_count": count,
            "state_hit_rate": safe_rate(count, denominator),
        }
        for source, count in sorted(source_state_counts.items())
    }

    return result


def build_grouped_report(
    records: Sequence[dict[str, Any]],
    ks: Sequence[int],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "overall": aggregate_records(records, ks),
        "by_split": {},
        "by_state_type": {},
        "by_split_and_state_type": {},
    }

    splits = sorted({str(record["split"]) for record in records})
    state_types = sorted({str(record["state_type"]) for record in records})

    for split in splits:
        subset = [record for record in records if record["split"] == split]
        report["by_split"][split] = aggregate_records(subset, ks)

    for state_type in state_types:
        subset = [
            record for record in records if record["state_type"] == state_type
        ]
        report["by_state_type"][state_type] = aggregate_records(subset, ks)

    for split in splits:
        report["by_split_and_state_type"][split] = {}
        for state_type in state_types:
            subset = [
                record
                for record in records
                if record["split"] == split
                and record["state_type"] == state_type
            ]
            if subset:
                report["by_split_and_state_type"][split][state_type] = (
                    aggregate_records(subset, ks)
                )

    return report


def write_missed_states_csv(
    path: Path,
    records: Sequence[dict[str, Any]],
) -> int:
    rows = [
        record
        for record in records
        if record["state_type"] != "complete"
        and record["positive_nonstop_count"] > 0
        and record["online_positive_action_count"] == 0
    ]
    rows.sort(
        key=lambda record: (
            record["split"],
            record["state_type"],
            record["task_id"],
            record["state_id"],
        )
    )

    fieldnames = [
        "task_id",
        "split",
        "state_id",
        "state_type",
        "step",
        "completion_score",
        "progress_score",
        "ranking_loss_mask",
        "positive_nonstop_count",
        "offline_positive_action_count",
        "offline_positive_single_count",
        "offline_positive_pair_count",
        "online_single_count",
        "offline_injected_single_count",
        "online_pair_count",
        "offline_pair_count",
        "offline_only_miss_class",
        "positive_files_absent_from_online_candidate_files",
        "positive_files_present_in_online_candidate_files",
        "positive_pair_generation_miss",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in rows:
            writer.writerow({field: record.get(field) for field in fieldnames})
    return len(rows)


def strip_internal_fields(records: Sequence[dict[str, Any]]) -> None:
    for record in records:
        record.pop("_online_single_ids", None)
        record.pop("_offline_positive_actions", None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读审计 Unified SWE Dataset 的 Retriever / Policy 覆盖率。"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="构建状态 SQLite，默认 data/.build/unified_swe_v1.sqlite3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="报告输出目录",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "validation", "benchmark"),
        help="仅审计指定 split；可重复传入。默认审计全部。",
    )
    parser.add_argument(
        "--k",
        type=int,
        action="append",
        help="Recall@K；可重复传入。默认 1,5,10,20,64。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ks = tuple(sorted(set(args.k or DEFAULT_KS)))
    if any(k <= 0 for k in ks):
        raise ValueError("所有 K 必须为正整数。")

    allowed_splits = set(args.split) if args.split else None
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = open_readonly_database(args.db)
    try:
        task_splits = load_task_splits(connection, allowed_splits)
        states = load_state_metadata(connection, task_splits)
        print(
            f"audit: tasks={len(task_splits)}, states={len(states)}, ks={list(ks)}",
            flush=True,
        )

        records = stream_state_action_records(connection, states, ks)
        classify_offline_only_misses(records, connection)

        grouped = build_grouped_report(records, ks)

        miss_csv = output_dir / "offline_only_missed_states.csv"
        missed_row_count = write_missed_states_csv(miss_csv, records)

        report = {
            "mode": "read_only",
            "database": str(args.db.resolve()),
            "scope": sorted(allowed_splits) if allowed_splits else "all",
            "definitions": {
                "retrieval_state": "state_type != complete",
                "known_positive_state": (
                    "至少一个非 STOP action 满足 action_label=positive 且 "
                    "action_loss_mask=true"
                ),
                "online_positive_action_coverage": (
                    "known-positive state 中，至少存在一个 candidate_scope=online "
                    "的正 single 或正 pair"
                ),
                "positive_online_single_recall_at_k": (
                    "known-positive state 中，至少一个正 single 的 "
                    "candidate_scope=online 且 online_retrieval_rank<=K。"
                    "分母包含 pair-only 正状态，因此这是严格 state-level 指标。"
                ),
                "positive_pair_realizable_from_online_singles_at_k": (
                    "至少一个正 pair 的两个 Evidence Unit 都分别出现在 "
                    "online single Top-K。它衡量 pair 组件是否已召回，不代表 "
                    "pair generator 实际生成了该 pair。"
                ),
                "file_universe_miss": (
                    "offline-only positive 所需 Evidence 所属 file_version "
                    "均未出现在该 state 的任何 online single candidate 中。"
                ),
                "within_file_or_unit_retrieval_miss": (
                    "至少一个正确 Evidence 的 file_version 已出现在 online "
                    "single candidate 文件集合中，但正确 action 仍未 online 命中。"
                ),
                "pair_generation_miss": (
                    "offline 正 pair 的两个 Evidence ID 都已经作为 online single "
                    "出现，但该正 pair 本身没有成为 online positive action。"
                ),
            },
            "counts": {
                "task_count": len(task_splits),
                "state_count": len(records),
                "offline_only_missed_state_csv_rows": missed_row_count,
            },
            "coverage": grouped,
            "outputs": {
                "report_json": str(output_dir / "report.json"),
                "offline_only_missed_states_csv": str(miss_csv),
            },
        }

        report_path = output_dir / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        strip_internal_fields(records)

        print(
            "\n=== 关键结果 ===",
            flush=True,
        )
        overall = grouped["overall"]
        print(
            "known-positive noncomplete states: "
            f"{overall['noncomplete_states_with_known_positive']}",
            flush=True,
        )
        print(
            "online positive action coverage: "
            f"{overall['online_positive_action_coverage_rate']}",
            flush=True,
        )
        print(
            "offline-only positive rate: "
            f"{overall['offline_only_positive_rate']}",
            flush=True,
        )
        for k in ks:
            metric = overall["positive_online_single_recall_at_k"][str(k)]
            print(
                f"positive online single Recall@{k}: "
                f"{metric['state_hit_rate']}",
                flush=True,
            )
        print(
            "miss classes: "
            + stable_json_dumps(overall["offline_only_miss_class_counts"]),
            flush=True,
        )
        print(f"report: {report_path}", flush=True)
        print(f"miss csv: {miss_csv}", flush=True)
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
