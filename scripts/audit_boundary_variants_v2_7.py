#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2.7 decision-boundary retrieval ablation（只读）。

目的
----
在不修改 unified_swe_v1.sqlite3 的前提下，只重放 decision_boundary states，
比较四种 Evidence Unit 排序策略：

A. static_head
   V2.6 当前策略：
   task 级 q-only BM25/path/symbol 静态排名
   + state-dependent structure(K)
   + channel-head protected RRF

B. dynamic_head
   每个 boundary state 基于当前 visible(q, K) 重新计算 BM25/path/symbol
   + structure(K)
   + channel-head protected RRF

C. dynamic_pure
   与 B 相同，但关闭 channel head reserve，使用纯 RRF Top-64

D. static_pure
   与 A 相同，但关闭 channel head reserve。
   这是诊断项，用于把“动态重排收益”和“head reserve 影响”拆开。

重要定义
--------
本脚本中的“正 Evidence / target file”不是 SWE-bench 原始字段。
SWE-bench 原始记录提供 problem_statement / repo / base_commit / patch / test_patch 等；
当前构建器把 patch/test supervision、公开 gold context、teacher obligations 等离线监督
映射为 Evidence Unit / positive action。

因此：
- positive Evidence = 当前 V2.6 policy 中 action_label=positive 且 action_loss_mask=true
  的非 STOP action 所引用的 Evidence Unit。
- derived target file = 上述 positive Evidence Unit 所属 file_version。
- 这些只用于离线审计，绝不会作为在线 Retriever 输入。

该脚本采用“固定正目标”ablation：
使用当前 V2.6 已物化且 scoreable 的 positive single/pair 作为所有 A/B/C/D 的共同目标。
这样可以只测 Retriever 排序变化，不把 supervision/rendering 变化混进实验。

运行示例
--------
python scripts/audit_boundary_variants_v2_7.py `
  --db data/.build/unified_swe_v1.sqlite3 `
  --fts data/.build/retriever_v2_2_fts.sqlite3 `
  --output-dir data/.build/audit_boundary_variants_v2_7
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


K_VALUES = (1, 5, 10, 20, 64)
VARIANTS = (
    "A_static_head",
    "B_dynamic_head",
    "C_dynamic_pure",
    "D_static_pure",
)


def stable_json_dump(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )


def open_readonly_sqlite(path: Path) -> sqlite3.Connection:
    """以 SQLite mode=ro + query_only 打开数据库。"""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    connection = sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-131072")
    return connection


def load_builder(script_path: Path) -> Any:
    """加载当前 V2.6 builder，直接复用真实 Retriever 实现。"""

    resolved = script_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"找不到 V2.6 builder：{resolved}\n"
            "请把 audit 脚本与 build_unified_dataset_v2_6.py 一起放在 scripts/。"
        )

    spec = importlib.util.spec_from_file_location(
        "build_unified_dataset_v2_6_for_boundary_audit",
        resolved,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 builder：{resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_question(task_payload: dict[str, Any]) -> str:
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


def positive_targets(
    actions: Sequence[dict[str, Any]],
) -> tuple[set[str], set[tuple[str, str]]]:
    """固定目标：当前 V2.6 已 scoreable 的正 single / pair。"""

    positive_singles: set[str] = set()
    positive_pairs: set[tuple[str, str]] = set()

    for action in actions:
        if action.get("action_type") == "stop":
            continue
        if action.get("action_label") != "positive":
            continue
        if not bool(action.get("action_loss_mask")):
            continue

        evidence_ids = tuple(map(str, action.get("evidence_ids") or []))
        if action.get("action_type") == "single" and len(evidence_ids) == 1:
            positive_singles.add(evidence_ids[0])
        elif action.get("action_type") == "pair" and len(evidence_ids) == 2:
            positive_pairs.add(tuple(sorted(evidence_ids)))

    return positive_singles, positive_pairs


def generate_online_pairs(
    online_ids: Sequence[str],
    structural_edges: dict[str, Sequence[str]],
    cap: int,
) -> list[tuple[str, str]]:
    """复制 V2.6 build_task_policy_states 的在线 pair generator。"""

    result: list[tuple[str, str]] = []
    candidate_set = set(map(str, online_ids))
    for source in map(str, online_ids):
        for target in structural_edges.get(source, ()):
            target = str(target)
            pair = tuple(sorted((source, target)))
            if pair[0] == pair[1] or not set(pair) <= candidate_set:
                continue
            if pair not in result:
                result.append(pair)
            if len(result) >= cap:
                return result
    return result


def evaluate_variant(
    *,
    fused: Sequence[dict[str, Any]],
    channels: dict[str, Sequence[str]],
    positive_singles: set[str],
    positive_pairs: set[tuple[str, str]],
    structural_edges: dict[str, Sequence[str]],
    pair_cap: int,
) -> dict[str, Any]:
    online_ids = [str(item["evidence_id"]) for item in fused]
    rank_by_id = {
        str(item["evidence_id"]): int(item["online_retrieval_rank"])
        for item in fused
    }
    online_pairs = set(generate_online_pairs(online_ids, structural_edges, pair_cap))

    single_ranks = sorted(
        rank_by_id[evidence_id]
        for evidence_id in positive_singles
        if evidence_id in rank_by_id
    )
    best_positive_rank = single_ranks[0] if single_ranks else None

    online_positive_single = bool(single_ranks)
    online_positive_pair = bool(positive_pairs & online_pairs)
    online_positive_action = online_positive_single or online_positive_pair

    recall = {
        str(k): bool(best_positive_rank is not None and best_positive_rank <= k)
        for k in K_VALUES
    }
    pair_realizable = {}
    for k in K_VALUES:
        top_k = set(online_ids[:k])
        pair_realizable[str(k)] = any(
            set(pair) <= top_k for pair in positive_pairs
        )

    channel_hits = {
        channel: bool(positive_singles & set(map(str, ranking)))
        for channel, ranking in channels.items()
    }

    return {
        "online_single_count": len(online_ids),
        "online_positive_action": online_positive_action,
        "online_positive_single": online_positive_single,
        "online_positive_pair": online_positive_pair,
        "best_positive_single_rank": best_positive_rank,
        "positive_single_recall_at_k": recall,
        "positive_pair_realizable_from_online_singles_at_k": pair_realizable,
        "positive_single_channel_hits": channel_hits,
        "online_ids": online_ids,
        "online_pairs": sorted(online_pairs),
    }


def persisted_result(
    actions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """从 V2.6 SQLite 中读取当前实际物化结果，作为 A replay 的一致性基准。"""

    positive_online_single_ranks: list[int] = []
    positive_online_pair = False

    for action in actions:
        if action.get("action_type") == "stop":
            continue
        if action.get("action_label") != "positive":
            continue
        if not bool(action.get("action_loss_mask")):
            continue
        if action.get("candidate_scope") != "online":
            continue

        if action.get("action_type") == "single":
            rank = action.get("online_retrieval_rank")
            if rank is not None:
                positive_online_single_ranks.append(int(rank))
        elif action.get("action_type") == "pair":
            positive_online_pair = True

    best_rank = min(positive_online_single_ranks, default=None)
    return {
        "online_positive_action": bool(
            positive_online_single_ranks or positive_online_pair
        ),
        "online_positive_single": bool(positive_online_single_ranks),
        "online_positive_pair": bool(positive_online_pair),
        "best_positive_single_rank": best_rank,
        "positive_single_recall_at_k": {
            str(k): bool(best_rank is not None and best_rank <= k)
            for k in K_VALUES
        },
    }


def aggregate_rows(
    rows: Sequence[dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    state_count = len(rows)
    if state_count == 0:
        return {
            "state_count": 0,
            "online_positive_action_hit_count": 0,
            "online_positive_action_coverage_rate": None,
        }

    action_hits = sum(bool(row["variants"][variant]["online_positive_action"]) for row in rows)
    single_hits = sum(bool(row["variants"][variant]["online_positive_single"]) for row in rows)
    pair_hits = sum(bool(row["variants"][variant]["online_positive_pair"]) for row in rows)

    recall = {}
    pair_realizable = {}
    for k in K_VALUES:
        key = str(k)
        recall[key] = {
            "state_hit_count": sum(
                bool(row["variants"][variant]["positive_single_recall_at_k"][key])
                for row in rows
            ),
        }
        recall[key]["state_hit_rate"] = recall[key]["state_hit_count"] / state_count

        pair_realizable[key] = {
            "state_hit_count": sum(
                bool(
                    row["variants"][variant][
                        "positive_pair_realizable_from_online_singles_at_k"
                    ][key]
                )
                for row in rows
            ),
        }
        pair_realizable[key]["state_hit_rate"] = (
            pair_realizable[key]["state_hit_count"] / state_count
        )

    channel_names = set()
    for row in rows:
        channel_names.update(
            row["variants"][variant]["positive_single_channel_hits"].keys()
        )
    channel_hits = {}
    for channel in sorted(channel_names):
        count = sum(
            bool(
                row["variants"][variant]["positive_single_channel_hits"].get(
                    channel, False
                )
            )
            for row in rows
        )
        channel_hits[channel] = {
            "state_hit_count": count,
            "state_hit_rate": count / state_count,
        }

    return {
        "state_count": state_count,
        "online_positive_action_hit_count": action_hits,
        "online_positive_action_coverage_rate": action_hits / state_count,
        "online_positive_single_hit_count": single_hits,
        "online_positive_single_hit_rate": single_hits / state_count,
        "online_positive_pair_hit_count": pair_hits,
        "online_positive_pair_hit_rate": pair_hits / state_count,
        "mean_online_single_count": (
            sum(row["variants"][variant]["online_single_count"] for row in rows)
            / state_count
        ),
        "positive_single_recall_at_k": recall,
        "positive_pair_realizable_from_online_singles_at_k": pair_realizable,
        "positive_single_channel_state_hits": channel_hits,
    }


def aggregate_persisted(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    state_count = len(rows)
    action_hits = sum(bool(row["persisted"]["online_positive_action"]) for row in rows)
    single_hits = sum(bool(row["persisted"]["online_positive_single"]) for row in rows)
    pair_hits = sum(bool(row["persisted"]["online_positive_pair"]) for row in rows)
    recall = {}
    for k in K_VALUES:
        key = str(k)
        count = sum(
            bool(row["persisted"]["positive_single_recall_at_k"][key])
            for row in rows
        )
        recall[key] = {
            "state_hit_count": count,
            "state_hit_rate": count / state_count if state_count else None,
        }
    return {
        "state_count": state_count,
        "online_positive_action_hit_count": action_hits,
        "online_positive_action_coverage_rate": (
            action_hits / state_count if state_count else None
        ),
        "online_positive_single_hit_count": single_hits,
        "online_positive_pair_hit_count": pair_hits,
        "positive_single_recall_at_k": recall,
    }


def write_state_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "task_id",
        "state_id",
        "split",
        "selected_evidence_count",
        "positive_single_count",
        "positive_pair_count",
        "derived_target_file_count",
        "persisted_hit",
        "persisted_best_positive_rank",
    ]
    for variant in VARIANTS:
        fieldnames.extend(
            [
                f"{variant}_hit",
                f"{variant}_best_positive_rank",
                f"{variant}_online_single_count",
                f"{variant}_bm25_hit",
                f"{variant}_path_hit",
                f"{variant}_symbol_hit",
                f"{variant}_structure_hit",
            ]
        )

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {
                "task_id": row["task_id"],
                "state_id": row["state_id"],
                "split": row["split"],
                "selected_evidence_count": len(row["selected_evidence_ids"]),
                "positive_single_count": len(row["positive_single_ids"]),
                "positive_pair_count": len(row["positive_pairs"]),
                "derived_target_file_count": len(row["derived_target_file_ids"]),
                "persisted_hit": int(row["persisted"]["online_positive_action"]),
                "persisted_best_positive_rank": (
                    row["persisted"]["best_positive_single_rank"]
                    if row["persisted"]["best_positive_single_rank"] is not None
                    else ""
                ),
            }
            for variant in VARIANTS:
                result = row["variants"][variant]
                hits = result["positive_single_channel_hits"]
                out.update(
                    {
                        f"{variant}_hit": int(result["online_positive_action"]),
                        f"{variant}_best_positive_rank": (
                            result["best_positive_single_rank"]
                            if result["best_positive_single_rank"] is not None
                            else ""
                        ),
                        f"{variant}_online_single_count": result["online_single_count"],
                        f"{variant}_bm25_hit": int(hits.get("bm25_content", False)),
                        f"{variant}_path_hit": int(hits.get("path_name", False)),
                        f"{variant}_symbol_hit": int(hits.get("symbol", False)),
                        f"{variant}_structure_hit": int(hits.get("structure", False)),
                    }
                )
            writer.writerow(out)


def load_boundary_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    try:
        return connection.execute(
            """
            SELECT
                p.state_id,
                p.task_id,
                p.payload_json AS state_json,
                c.snapshot_id,
                c.final_split,
                c.payload_json AS task_json,
                s.payload_json AS supervision_json
            FROM policy_states AS p
            JOIN canonical_tasks AS c ON c.task_id = p.task_id
            JOIN supervision AS s ON s.task_id = p.task_id
            WHERE json_extract(p.payload_json, '$.state_type') = 'decision_boundary'
            ORDER BY p.task_id, p.state_id
            """
        ).fetchall()
    except sqlite3.OperationalError as error:
        if "json" not in str(error).lower():
            raise
        # 极少数 SQLite 构建若没有 JSON1，则退化为 Python 过滤。
        rows = connection.execute(
            """
            SELECT
                p.state_id,
                p.task_id,
                p.payload_json AS state_json,
                c.snapshot_id,
                c.final_split,
                c.payload_json AS task_json,
                s.payload_json AS supervision_json
            FROM policy_states AS p
            JOIN canonical_tasks AS c ON c.task_id = p.task_id
            JOIN supervision AS s ON s.task_id = p.task_id
            ORDER BY p.task_id, p.state_id
            """
        ).fetchall()
        return [
            row
            for row in rows
            if json.loads(row["state_json"]).get("state_type")
            == "decision_boundary"
        ]


def load_boundary_actions(
    connection: sqlite3.Connection,
    state_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    actions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    state_list = sorted(state_ids)
    for offset in range(0, len(state_list), 700):
        chunk = state_list[offset : offset + 700]
        placeholders = ",".join("?" for _ in chunk)
        for row in connection.execute(
            f"SELECT state_id, payload_json FROM candidate_actions "
            f"WHERE state_id IN ({placeholders}) ORDER BY state_id, action_key",
            chunk,
        ):
            actions[str(row["state_id"])].append(json.loads(row["payload_json"]))
    return actions


def lookup_target_file_ids(
    connection: sqlite3.Connection,
    positive_ids: set[str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    ids = sorted(positive_ids)
    for offset in range(0, len(ids), 700):
        chunk = ids[offset : offset + 700]
        placeholders = ",".join("?" for _ in chunk)
        for row in connection.execute(
            f"SELECT evidence_id, file_version_id FROM evidence_units "
            f"WHERE evidence_id IN ({placeholders})",
            chunk,
        ):
            result[str(row["evidence_id"])] = str(row["file_version_id"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="只读重放 V2.6 decision-boundary A/B/C/D retrieval variants。"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/.build/unified_swe_v1.sqlite3"),
    )
    parser.add_argument(
        "--fts",
        type=Path,
        default=Path("data/.build/retriever_v2_2_fts.sqlite3"),
    )
    parser.add_argument(
        "--builder",
        type=Path,
        default=Path(__file__).with_name("build_unified_dataset_v2_6.py"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/.build/audit_boundary_variants_v2_7"),
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=None,
        help="仅调试时限制 boundary state 数；正式实验不要设置。",
    )
    args = parser.parse_args()

    if args.max_states is not None and args.max_states <= 0:
        raise ValueError("--max-states 必须为正整数。")

    builder = load_builder(args.builder)
    required_symbols = (
        "_load_policy_evidence_universe",
        "build_policy_structural_edges",
        "precompute_task_query_channel_rankings",
        "task_query_channels_for_state",
        "retrieve_online_channels",
        "reciprocal_rank_fusion",
        "CHANNEL_HEAD_RESERVE",
        "FINAL_DEPTH",
        "RRF_K",
        "REGULAR_PAIR_CAP",
    )
    missing = [name for name in required_symbols if not hasattr(builder, name)]
    if missing:
        raise RuntimeError(f"V2.6 builder 缺少所需符号：{missing}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    connection = open_readonly_sqlite(args.db)
    fts_connection = open_readonly_sqlite(args.fts)

    started = time.perf_counter()
    try:
        boundary_rows = load_boundary_rows(connection)
        if args.max_states is not None:
            boundary_rows = boundary_rows[: args.max_states]

        if not boundary_rows:
            raise ValueError("当前 DB 没有 decision_boundary state。")

        actions_by_state = load_boundary_actions(
            connection,
            {str(row["state_id"]) for row in boundary_rows},
        )

        print(
            f"boundary-ablation-v2.7: states={len(boundary_rows)}, "
            f"db={args.db.resolve()}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "boundary-ablation-v2.7: read-only; no SQLite mutation",
            file=sys.stderr,
            flush=True,
        )

        task_cache: dict[str, dict[str, Any]] = {}
        output_rows: list[dict[str, Any]] = []
        universe_seconds = 0.0
        ranking_seconds = 0.0

        for index, row in enumerate(boundary_rows, 1):
            task_id = str(row["task_id"])
            state_id = str(row["state_id"])
            state = json.loads(row["state_json"])
            actions = actions_by_state.get(state_id, [])
            positive_singles, positive_pairs = positive_targets(actions)
            if not positive_singles and not positive_pairs:
                raise ValueError(
                    f"boundary state 缺少固定正目标：state_id={state_id}"
                )

            task_cached = task_cache.get(task_id)
            if task_cached is None:
                task_payload = json.loads(row["task_json"])
                supervision = json.loads(row["supervision_json"])
                question = build_question(task_payload)
                witness_ids = supervision_witness_ids(supervision)

                stage_started = time.perf_counter()
                evidence_by_id, online_evidence_ids = (
                    builder._load_policy_evidence_universe(
                        connection,
                        snapshot_id=str(row["snapshot_id"]),
                        question=question,
                        witness_evidence_ids=witness_ids,
                        repo_cache_index=None,
                        fts_connection=fts_connection,
                    )
                )
                structural_edges = builder.build_policy_structural_edges(
                    evidence_by_id
                )
                static_rankings = builder.precompute_task_query_channel_rankings(
                    question,
                    list(evidence_by_id.values()),
                )
                universe_seconds += time.perf_counter() - stage_started

                task_cached = {
                    "question": question,
                    "evidence_by_id": evidence_by_id,
                    "online_evidence_ids": [
                        str(item) for item in online_evidence_ids
                    ],
                    "structural_edges": structural_edges,
                    "static_rankings": static_rankings,
                }
                task_cache[task_id] = task_cached

            question = task_cached["question"]
            evidence_by_id = task_cached["evidence_by_id"]
            base_ids = task_cached["online_evidence_ids"]
            structural_edges = task_cached["structural_edges"]
            static_rankings = task_cached["static_rankings"]

            selected_ids = set(map(str, state.get("evidence_ids") or []))
            expanded_ids = {
                str(target)
                for source in selected_ids
                for target in structural_edges.get(source, ())
                if str(target) in evidence_by_id
            }
            visible_ids = list(
                dict.fromkeys([*base_ids, *sorted(expanded_ids)])
            )
            visible_set = set(visible_ids)
            visible_records = [
                evidence_by_id[evidence_id]
                for evidence_id in visible_ids
                if evidence_id in evidence_by_id
            ]

            stage_started = time.perf_counter()

            static_channels = builder.task_query_channels_for_state(
                precomputed_rankings=static_rankings,
                visible_ids=visible_set,
                selected_ids=selected_ids,
                structural_edges=structural_edges,
                channel_depth=builder.CHANNEL_DEPTH,
            )
            dynamic_channels = builder.retrieve_online_channels(
                question,
                visible_records,
                state_evidence_ids=sorted(selected_ids),
                structural_edges=structural_edges,
                channel_depth=builder.CHANNEL_DEPTH,
            )

            variant_specs = {
                "A_static_head": (
                    static_channels,
                    int(builder.CHANNEL_HEAD_RESERVE),
                ),
                "B_dynamic_head": (
                    dynamic_channels,
                    int(builder.CHANNEL_HEAD_RESERVE),
                ),
                "C_dynamic_pure": (dynamic_channels, 0),
                "D_static_pure": (static_channels, 0),
            }

            variant_results: dict[str, dict[str, Any]] = {}
            for variant, (channels, head_reserve) in variant_specs.items():
                fused = builder.reciprocal_rank_fusion(
                    channels,
                    depth=int(builder.FINAL_DEPTH),
                    rrf_k=int(builder.RRF_K),
                    channel_head_reserve=head_reserve,
                )
                variant_results[variant] = evaluate_variant(
                    fused=fused,
                    channels=channels,
                    positive_singles=positive_singles,
                    positive_pairs=positive_pairs,
                    structural_edges=structural_edges,
                    pair_cap=int(builder.REGULAR_PAIR_CAP),
                )

            ranking_seconds += time.perf_counter() - stage_started

            all_positive_ids = set(positive_singles)
            for pair in positive_pairs:
                all_positive_ids.update(pair)
            file_by_positive = lookup_target_file_ids(
                connection,
                all_positive_ids,
            )
            derived_target_file_ids = sorted(set(file_by_positive.values()))

            output_rows.append(
                {
                    "task_id": task_id,
                    "state_id": state_id,
                    "split": str(row["final_split"]),
                    "selected_evidence_ids": sorted(selected_ids),
                    "positive_single_ids": sorted(positive_singles),
                    "positive_pairs": [
                        list(pair) for pair in sorted(positive_pairs)
                    ],
                    "derived_target_file_ids": derived_target_file_ids,
                    "persisted": persisted_result(actions),
                    "variants": variant_results,
                }
            )

            if index % 25 == 0 or index == len(boundary_rows):
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"boundary-ablation-v2.7: {index}/{len(boundary_rows)} "
                    f"states, rate={index / elapsed:.2f} state/s, "
                    f"universe={universe_seconds:.1f}s, "
                    f"ranking={ranking_seconds:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )

        persisted_overall = aggregate_persisted(output_rows)
        aggregates: dict[str, Any] = {
            "overall": {
                "persisted_v2_6": persisted_overall,
                "variants": {
                    variant: aggregate_rows(output_rows, variant)
                    for variant in VARIANTS
                },
            },
            "by_split": {},
        }

        for split in sorted({row["split"] for row in output_rows}):
            split_rows = [row for row in output_rows if row["split"] == split]
            aggregates["by_split"][split] = {
                "persisted_v2_6": aggregate_persisted(split_rows),
                "variants": {
                    variant: aggregate_rows(split_rows, variant)
                    for variant in VARIANTS
                },
            }

        # A 必须尽可能复现持久化的 V2.6 命中。
        # 若不一致，说明 builder/DB 版本或固定目标定义不匹配，此时 B/C/D 不应直接用于版本决策。
        persisted_hits = persisted_overall["online_positive_action_hit_count"]
        a_hits = aggregates["overall"]["variants"]["A_static_head"][
            "online_positive_action_hit_count"
        ]
        per_state_match_count = sum(
            row["persisted"]["online_positive_action"]
            == row["variants"]["A_static_head"]["online_positive_action"]
            for row in output_rows
        )
        replay_consistency = {
            "persisted_online_positive_action_hit_count": persisted_hits,
            "A_static_head_replay_hit_count": a_hits,
            "hit_count_delta": a_hits - persisted_hits,
            "per_state_hit_match_count": per_state_match_count,
            "per_state_hit_match_rate": per_state_match_count / len(output_rows),
            "strict_ok": (
                a_hits == persisted_hits
                and per_state_match_count == len(output_rows)
            ),
        }

        report = {
            "mode": "read_only_boundary_ablation",
            "database": str(args.db.resolve()),
            "fts_sidecar": str(args.fts.resolve()),
            "builder": str(args.builder.resolve()),
            "builder_retriever_version": getattr(
                builder, "RETRIEVER_VERSION", None
            ),
            "definitions": {
                "raw_correct_file_field_exists": False,
                "positive_target_definition": (
                    "固定使用当前 V2.6 candidate_actions 中 "
                    "action_label=positive、action_loss_mask=true 的非 STOP "
                    "single/pair Evidence ID。"
                ),
                "derived_target_file_definition": (
                    "positive target Evidence Unit 所属的 file_version；"
                    "这是离线 supervision 派生概念，不是 SWE-bench 原始字段，"
                    "也不作为 Retriever 输入。"
                ),
                "A_static_head": (
                    "V2.6 当前 task-static q-only channels + "
                    "state-dependent structure(K) + head-protected RRF"
                ),
                "B_dynamic_head": (
                    "boundary state 重新计算 visible(q,K) "
                    "BM25/path/symbol + structure(K) + head-protected RRF"
                ),
                "C_dynamic_pure": (
                    "与 B 相同，但 channel_head_reserve=0，纯 RRF"
                ),
                "D_static_pure": (
                    "与 A 相同，但 channel_head_reserve=0；诊断 head reserve 影响"
                ),
            },
            "counts": {
                "boundary_state_count": len(output_rows),
                "unique_task_count": len(
                    {row["task_id"] for row in output_rows}
                ),
            },
            "timing_seconds": {
                "total": time.perf_counter() - started,
                "universe_and_task_static_precompute": universe_seconds,
                "boundary_variant_ranking": ranking_seconds,
            },
            "replay_consistency": replay_consistency,
            "results": aggregates,
        }

        report_path = output_dir / "report.json"
        state_csv_path = output_dir / "per_state.csv"
        report_path.write_text(
            stable_json_dump(report) + "\n",
            encoding="utf-8",
        )
        write_state_csv(state_csv_path, output_rows)

        print(stable_json_dump({
            "replay_consistency": replay_consistency,
            "overall": {
                variant: {
                    "coverage": aggregates["overall"]["variants"][variant][
                        "online_positive_action_coverage_rate"
                    ],
                    "recall_at_64": aggregates["overall"]["variants"][variant][
                        "positive_single_recall_at_k"
                    ]["64"]["state_hit_rate"],
                }
                for variant in VARIANTS
            },
            "outputs": {
                "report_json": str(report_path),
                "per_state_csv": str(state_csv_path),
            },
        }))

        if not replay_consistency["strict_ok"]:
            print(
                "WARNING: A_static_head 未严格复现当前 V2.6 persisted boundary hit。"
                "请先检查 builder 与 DB 是否为同一 V2.6 版本；"
                "在一致性问题解决前，不要据 B/C/D 做最终版本决策。",
                file=sys.stderr,
                flush=True,
            )
            return 2

        return 0
    finally:
        fts_connection.close()
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
