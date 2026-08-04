#!/usr/bin/env python3
"""逐层重放当前 Unified SWE Retriever，定位 offline-only positive 在哪一层丢失。

只读：
- data/.build/unified_swe_v1.sqlite3
- 上一步生成的 offline_only_missed_states.csv
- scripts/build_unified_dataset.py

不会修改 SQLite，不会重建 corpus，不会调用教师 API。
本脚本直接导入当前 build_unified_dataset.py 中的真实 Retriever 函数，避免复制逻辑导致口径漂移。

输出：
- report.json
- state_replay.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_DB = Path("data/.build/unified_swe_v1.sqlite3")
DEFAULT_BUILDER = Path("scripts/build_unified_dataset.py")
DEFAULT_MISS_CSV = Path(
    "data/.build/audit_retriever_policy_coverage/offline_only_missed_states.csv"
)
DEFAULT_OUT_DIR = Path("data/.build/audit_retriever_pipeline_replay")
SQL_BATCH = 700


def batched(values: Sequence[str], size: int = SQL_BATCH) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def open_readonly_database(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"找不到 SQLite：{resolved}")
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def load_builder(path: Path) -> Any:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"找不到构建脚本：{resolved}")
    spec = importlib.util.spec_from_file_location(
        "unified_swe_builder_for_audit", resolved
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载构建脚本：{resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "ONLINE_FILE_CAP",
        "ONLINE_UNIT_UNIVERSE_CAP",
        "CHANNEL_DEPTH",
        "FINAL_DEPTH",
        "RRF_K",
        "select_online_file_memberships",
        "policy_records_from_file_payload",
        "_load_policy_evidence_universe",
        "build_policy_structural_edges",
        "retrieve_online_channels",
        "reciprocal_rank_fusion",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"构建脚本缺少审计所需对象：{missing}")
    return module


def read_miss_rows(
    path: Path,
    *,
    splits: set[str] | None,
    state_types: set[str] | None,
) -> list[dict[str, str]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"找不到 miss CSV：{resolved}")
    rows: list[dict[str, str]] = []
    with resolved.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            split = str(row.get("split") or "")
            state_type = str(row.get("state_type") or "")
            if splits is not None and split not in splits:
                continue
            if state_types is not None and state_type not in state_types:
                continue
            state_id = str(row.get("state_id") or "")
            task_id = str(row.get("task_id") or "")
            if state_id and task_id:
                rows.append(
                    {
                        "task_id": task_id,
                        "state_id": state_id,
                        "split": split,
                        "state_type": state_type,
                    }
                )
    rows.sort(
        key=lambda item: (
            item["task_id"],
            item["state_type"],
            item["state_id"],
        )
    )
    return rows


def load_state_payloads(
    connection: sqlite3.Connection,
    state_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for chunk in batched(sorted(set(state_ids))):
        placeholders = ",".join("?" for _ in chunk)
        for row in connection.execute(
            "SELECT state_id, payload_json FROM policy_states "
            f"WHERE state_id IN ({placeholders})",
            chunk,
        ):
            result[str(row["state_id"])] = json.loads(row["payload_json"])
    return result


def load_task_payloads(
    connection: sqlite3.Connection,
    task_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for chunk in batched(sorted(set(task_ids))):
        placeholders = ",".join("?" for _ in chunk)
        for row in connection.execute(
            "SELECT task_id, snapshot_id, payload_json FROM canonical_tasks "
            f"WHERE task_id IN ({placeholders})",
            chunk,
        ):
            payload = json.loads(row["payload_json"])
            result[str(row["task_id"])] = {
                "snapshot_id": str(row["snapshot_id"]),
                "payload": payload,
            }
    return result


def load_supervision_payloads(
    connection: sqlite3.Connection,
    task_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for chunk in batched(sorted(set(task_ids))):
        placeholders = ",".join("?" for _ in chunk)
        for row in connection.execute(
            "SELECT task_id, payload_json FROM supervision "
            f"WHERE task_id IN ({placeholders})",
            chunk,
        ):
            result[str(row["task_id"])] = json.loads(row["payload_json"])
    return result


def load_offline_positive_single_ids(
    connection: sqlite3.Connection,
    state_ids: Sequence[str],
) -> dict[str, set[str]]:
    """读取上一步判定为 offline-only state 的正 single Evidence ID。"""

    result: dict[str, set[str]] = defaultdict(set)
    for chunk in batched(sorted(set(state_ids))):
        placeholders = ",".join("?" for _ in chunk)
        for row in connection.execute(
            "SELECT state_id, payload_json FROM candidate_actions "
            f"WHERE state_id IN ({placeholders})",
            chunk,
        ):
            action = json.loads(row["payload_json"])
            if (
                action.get("action_type") == "single"
                and action.get("candidate_scope") == "offline_injected"
                and action.get("action_label") == "positive"
                and action.get("action_loss_mask") is True
            ):
                evidence_ids = list(map(str, action.get("evidence_ids") or []))
                if len(evidence_ids) == 1:
                    result[str(row["state_id"])].add(evidence_ids[0])
    return dict(result)


def load_evidence_file_versions(
    connection: sqlite3.Connection,
    evidence_ids: Sequence[str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for chunk in batched(sorted(set(evidence_ids))):
        placeholders = ",".join("?" for _ in chunk)
        for row in connection.execute(
            "SELECT evidence_id, file_version_id FROM evidence_units "
            f"WHERE evidence_id IN ({placeholders})",
            chunk,
        ):
            result[str(row["evidence_id"])] = str(row["file_version_id"])
    return result


def task_question(task_payload: dict[str, Any]) -> str:
    task_input = task_payload.get("input") or {}
    return "\n".join(
        [
            str(task_input.get("problem_statement") or ""),
            *[
                str(hint)
                for hint in task_input.get("hints") or []
                if str(hint).strip()
            ],
        ]
    )


def supervision_witness_ids(supervision: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(evidence_id)
            for obligation in supervision.get("obligations") or []
            for group in obligation.get("witness_groups") or []
            for evidence_id in group.get("evidence_ids") or []
        }
    )


def selected_memberships_and_pretrim_ids(
    connection: sqlite3.Connection,
    builder: Any,
    *,
    snapshot_id: str,
    question: str,
) -> tuple[set[str], set[str]]:
    """重放 Top-32 文件 gate，并返回 gate 后、4096 截断前的 Evidence IDs。"""

    membership_rows = connection.execute(
        "SELECT path, file_version_id FROM snapshot_file_memberships "
        "WHERE snapshot_id=? ORDER BY path",
        (snapshot_id,),
    ).fetchall()
    selected = builder.select_online_file_memberships(
        question,
        [dict(row) for row in membership_rows],
        cap=int(builder.ONLINE_FILE_CAP),
    )
    selected_file_ids = {
        str(item["file_version_id"]) for item in selected
    }

    pretrim_ids: set[str] = set()
    if not selected_file_ids:
        return selected_file_ids, pretrim_ids

    selected_file_list = sorted(selected_file_ids)
    for chunk in batched(selected_file_list):
        placeholders = ",".join("?" for _ in chunk)
        for row in connection.execute(
            "SELECT payload_json FROM file_versions "
            f"WHERE file_version_id IN ({placeholders})",
            chunk,
        ):
            file_record = json.loads(row["payload_json"])
            for record in builder.policy_records_from_file_payload(file_record):
                pretrim_ids.add(str(record["evidence_id"]))
    return selected_file_ids, pretrim_ids


def deepest_stage(
    *,
    positive_file_selected: bool,
    positive_pretrim: bool,
    positive_base: bool,
    positive_structure_expanded: bool,
    positive_visible: bool,
    positive_channel: bool,
    positive_fused: bool,
) -> str:
    """返回最深的成功阶段；最终 fused 成功理论上应与 offline-only 输入冲突。"""

    if positive_fused:
        return "replay_mismatch_final_fused_hit"
    if positive_channel:
        return "rrf_final64_miss"
    if positive_visible:
        return "channel_top64_miss"
    if positive_structure_expanded:
        return "visible_set_inconsistency"
    if positive_base:
        return "visible_set_inconsistency"
    if positive_pretrim:
        return "unit_universe_4096_miss"
    if positive_file_selected:
        return "selected_file_missing_positive_unit"
    return "path_file_top32_miss"


def replay_task(
    connection: sqlite3.Connection,
    builder: Any,
    task_id: str,
    rows: Sequence[dict[str, str]],
    *,
    task_meta: dict[str, Any],
    supervision: dict[str, Any],
    state_payloads: dict[str, dict[str, Any]],
    positive_ids_by_state: dict[str, set[str]],
    evidence_file_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    snapshot_id = str(task_meta["snapshot_id"])
    question = task_question(task_meta["payload"])
    witness_ids = supervision_witness_ids(supervision)

    selected_file_ids, pretrim_ids = selected_memberships_and_pretrim_ids(
        connection,
        builder,
        snapshot_id=snapshot_id,
        question=question,
    )

    # 直接调用当前构建器真实 universe 函数，得到 4096 截断后的 base IDs，
    # 同时补齐监督 witness，保证 structural_edges 与正式 policy 构建一致。
    evidence_by_id, base_online_ids_list = builder._load_policy_evidence_universe(
        connection,
        snapshot_id=snapshot_id,
        question=question,
        witness_evidence_ids=witness_ids,
    )
    base_online_ids = set(map(str, base_online_ids_list))
    structural_edges = builder.build_policy_structural_edges(evidence_by_id)

    outputs: list[dict[str, Any]] = []
    for row in rows:
        state_id = row["state_id"]
        state = state_payloads[state_id]
        positives = set(positive_ids_by_state.get(state_id) or ())
        if not positives:
            outputs.append(
                {
                    **row,
                    "classification": "missing_offline_positive_single",
                    "positive_single_count": 0,
                }
            )
            continue

        positive_files = {
            evidence_file_by_id[evidence_id]
            for evidence_id in positives
            if evidence_id in evidence_file_by_id
        }

        selected = set(map(str, state.get("evidence_ids") or []))
        expanded_ids = {
            str(target)
            for source in selected
            for target in structural_edges.get(source, ())
            if str(target) in evidence_by_id
        }
        visible_ids_list = list(
            dict.fromkeys(
                [
                    *base_online_ids_list,
                    *sorted(expanded_ids),
                ]
            )
        )
        visible_ids = set(visible_ids_list)
        visible_records = [
            evidence_by_id[evidence_id]
            for evidence_id in visible_ids_list
            if evidence_id in evidence_by_id
        ]

        channels = builder.retrieve_online_channels(
            question,
            visible_records,
            state_evidence_ids=state.get("evidence_ids") or [],
            structural_edges=structural_edges,
            channel_depth=int(builder.CHANNEL_DEPTH),
        )
        channel_union = {
            str(evidence_id)
            for values in channels.values()
            for evidence_id in values
        }
        fused = builder.reciprocal_rank_fusion(
            channels,
            depth=int(builder.FINAL_DEPTH),
            rrf_k=int(builder.RRF_K),
        )
        fused_ids = {str(item["evidence_id"]) for item in fused}

        positive_file_selected = bool(positive_files & selected_file_ids)
        positive_pretrim = bool(positives & pretrim_ids)
        positive_base = bool(positives & base_online_ids)
        positive_structure_expanded = bool(positives & expanded_ids)
        positive_visible = bool(positives & visible_ids)
        positive_channel = bool(positives & channel_union)
        positive_fused = bool(positives & fused_ids)

        hit_channels = sorted(
            channel
            for channel, ids in channels.items()
            if positives & set(map(str, ids))
        )

        classification = deepest_stage(
            positive_file_selected=positive_file_selected,
            positive_pretrim=positive_pretrim,
            positive_base=positive_base,
            positive_structure_expanded=positive_structure_expanded,
            positive_visible=positive_visible,
            positive_channel=positive_channel,
            positive_fused=positive_fused,
        )

        outputs.append(
            {
                **row,
                "classification": classification,
                "positive_single_count": len(positives),
                "positive_file_count": len(positive_files),
                "positive_file_selected_top32": positive_file_selected,
                "positive_unit_present_before_4096_trim": positive_pretrim,
                "positive_unit_present_in_base_after_4096_trim": positive_base,
                "positive_unit_added_by_structure": positive_structure_expanded,
                "positive_unit_visible_before_channels": positive_visible,
                "positive_unit_in_any_channel_top64": positive_channel,
                "positive_unit_in_final_rrf64": positive_fused,
                "positive_hit_channels": "|".join(hit_channels),
                "selected_file_count": len(selected_file_ids),
                "pretrim_unit_count": len(pretrim_ids),
                "base_online_unit_count": len(base_online_ids),
                "structure_expanded_unit_count": len(expanded_ids),
                "visible_unit_count": len(visible_ids),
                "channel_union_count": len(channel_union),
                "fused_count": len(fused_ids),
            }
        )
    return outputs


def rate(count: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(count / denominator, 6)


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    classification = Counter(str(row.get("classification")) for row in rows)

    stage_fields = (
        "positive_file_selected_top32",
        "positive_unit_present_before_4096_trim",
        "positive_unit_present_in_base_after_4096_trim",
        "positive_unit_added_by_structure",
        "positive_unit_visible_before_channels",
        "positive_unit_in_any_channel_top64",
        "positive_unit_in_final_rrf64",
    )
    stage_report: dict[str, Any] = {}
    for field in stage_fields:
        hit = sum(row.get(field) is True for row in rows)
        stage_report[field] = {
            "hit_count": hit,
            "hit_rate": rate(hit, total),
        }

    channel_hits = Counter()
    for row in rows:
        for channel in str(row.get("positive_hit_channels") or "").split("|"):
            if channel:
                channel_hits[channel] += 1

    return {
        "state_count": total,
        "classification_counts": dict(sorted(classification.items())),
        "classification_rates": {
            key: rate(value, total)
            for key, value in sorted(classification.items())
        },
        "pipeline_stage_survival": stage_report,
        "positive_channel_hits": {
            channel: {
                "state_hit_count": count,
                "state_hit_rate": rate(count, total),
            }
            for channel, count in sorted(channel_hits.items())
        },
    }


def build_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "overall": aggregate(rows),
        "by_split": {},
        "by_state_type": {},
        "by_split_and_state_type": {},
    }
    splits = sorted({str(row["split"]) for row in rows})
    state_types = sorted({str(row["state_type"]) for row in rows})

    for split in splits:
        subset = [row for row in rows if row["split"] == split]
        report["by_split"][split] = aggregate(subset)

    for state_type in state_types:
        subset = [row for row in rows if row["state_type"] == state_type]
        report["by_state_type"][state_type] = aggregate(subset)

    for split in splits:
        report["by_split_and_state_type"][split] = {}
        for state_type in state_types:
            subset = [
                row
                for row in rows
                if row["split"] == split and row["state_type"] == state_type
            ]
            if subset:
                report["by_split_and_state_type"][split][state_type] = aggregate(
                    subset
                )
    return report


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="逐层重放当前 Retriever，定位 offline-only positive 丢失阶段。"
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--builder", type=Path, default=DEFAULT_BUILDER)
    parser.add_argument("--miss-csv", type=Path, default=DEFAULT_MISS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "validation", "benchmark"),
        help="只审计指定 split，可重复。默认全部。",
    )
    parser.add_argument(
        "--state-type",
        action="append",
        choices=("initial", "decision_boundary"),
        help="只审计指定 state type，可重复。默认全部 offline-only state。",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=None,
        help="只用于快速试跑；正式结果不要设置。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_states is not None and args.max_states <= 0:
        raise ValueError("--max-states 必须大于 0。")

    builder = load_builder(args.builder)
    connection = open_readonly_database(args.db)
    try:
        miss_rows = read_miss_rows(
            args.miss_csv,
            splits=set(args.split) if args.split else None,
            state_types=set(args.state_type) if args.state_type else None,
        )
        if args.max_states is not None:
            miss_rows = miss_rows[: args.max_states]
        if not miss_rows:
            raise RuntimeError("筛选后没有 offline-only miss state。")

        state_ids = [row["state_id"] for row in miss_rows]
        task_ids = sorted({row["task_id"] for row in miss_rows})

        state_payloads = load_state_payloads(connection, state_ids)
        task_payloads = load_task_payloads(connection, task_ids)
        supervision_payloads = load_supervision_payloads(connection, task_ids)
        positive_ids_by_state = load_offline_positive_single_ids(
            connection, state_ids
        )
        all_positive_ids = sorted(
            {
                evidence_id
                for ids in positive_ids_by_state.values()
                for evidence_id in ids
            }
        )
        evidence_file_by_id = load_evidence_file_versions(
            connection, all_positive_ids
        )

        missing_states = sorted(set(state_ids) - set(state_payloads))
        missing_tasks = sorted(set(task_ids) - set(task_payloads))
        missing_supervision = sorted(set(task_ids) - set(supervision_payloads))
        missing_positive_states = sorted(
            set(state_ids) - set(positive_ids_by_state)
        )
        if missing_states or missing_tasks or missing_supervision:
            raise RuntimeError(
                "输入引用不完整："
                f"missing_states={len(missing_states)}, "
                f"missing_tasks={len(missing_tasks)}, "
                f"missing_supervision={len(missing_supervision)}"
            )
        if missing_positive_states:
            print(
                f"warning: {len(missing_positive_states)} 个 state "
                "没有 offline positive single，将单独标记。",
                flush=True,
            )

        rows_by_task: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in miss_rows:
            rows_by_task[row["task_id"]].append(row)

        replay_rows: list[dict[str, Any]] = []
        total_tasks = len(rows_by_task)
        for index, task_id in enumerate(sorted(rows_by_task), 1):
            replay_rows.extend(
                replay_task(
                    connection,
                    builder,
                    task_id,
                    rows_by_task[task_id],
                    task_meta=task_payloads[task_id],
                    supervision=supervision_payloads[task_id],
                    state_payloads=state_payloads,
                    positive_ids_by_state=positive_ids_by_state,
                    evidence_file_by_id=evidence_file_by_id,
                )
            )
            if index == 1 or index % 250 == 0 or index == total_tasks:
                print(
                    f"replay: {index}/{total_tasks} tasks, "
                    f"{len(replay_rows)}/{len(miss_rows)} states",
                    flush=True,
                )

        report_body = build_report(replay_rows)
        out_dir = args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "state_replay.csv"
        report_path = out_dir / "report.json"
        write_csv(csv_path, replay_rows)

        report = {
            "mode": "read_only_exact_builder_replay",
            "database": str(args.db.resolve()),
            "builder": str(args.builder.resolve()),
            "miss_csv": str(args.miss_csv.resolve()),
            "scope": {
                "splits": sorted(set(args.split)) if args.split else "all",
                "state_types": (
                    sorted(set(args.state_type))
                    if args.state_type
                    else "all_noncomplete_offline_only"
                ),
                "max_states": args.max_states,
            },
            "builder_constants": {
                "ONLINE_FILE_CAP": int(builder.ONLINE_FILE_CAP),
                "ONLINE_UNIT_UNIVERSE_CAP": int(builder.ONLINE_UNIT_UNIVERSE_CAP),
                "CHANNEL_DEPTH": int(builder.CHANNEL_DEPTH),
                "FINAL_DEPTH": int(builder.FINAL_DEPTH),
                "RRF_K": int(builder.RRF_K),
            },
            "classification_semantics": {
                "path_file_top32_miss": (
                    "所有正确 positive single 的 file_version 都未通过 "
                    "select_online_file_memberships Top-32。"
                ),
                "selected_file_missing_positive_unit": (
                    "正确文件进入 Top-32，但对应 positive Evidence Unit "
                    "在 4096 截断前的 file payload online records 中不存在。"
                ),
                "unit_universe_4096_miss": (
                    "positive unit 在 Top-32 文件展开结果中存在，但被 "
                    "ONLINE_UNIT_UNIVERSE_CAP=4096 截掉。"
                ),
                "channel_top64_miss": (
                    "positive unit 已进入当前 state 的 visible set，但没有进入 "
                    "bm25/path/symbol/structure 任一通道 Top-64。"
                ),
                "rrf_final64_miss": (
                    "positive unit 已进入至少一个通道 Top-64，但被最终 RRF Top-64 淘汰。"
                ),
                "replay_mismatch_final_fused_hit": (
                    "精确重放时 positive 进入最终 RRF64，但旧 action 仍被标为 "
                    "offline-only；正常情况下应为 0，需要单独调查。"
                ),
            },
            "coverage": report_body,
            "outputs": {
                "report_json": str(report_path),
                "state_replay_csv": str(csv_path),
            },
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print("\n=== 关键结果 ===", flush=True)
        overall = report_body["overall"]
        print(
            "classification_counts="
            + json.dumps(
                overall["classification_counts"],
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        print(
            "pipeline_stage_survival="
            + json.dumps(
                overall["pipeline_stage_survival"],
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        print(f"report: {report_path}", flush=True)
        print(f"csv: {csv_path}", flush=True)
        return 0
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
