#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2.9 decision-boundary clean-structure audit（只读）。

目标
----
验证当前 policy structure channel 是否被 offline witness / 子集邻接污染，并比较：

A_persisted_v26
    当前数据库中已物化的 V2.6 boundary 结果。

B_clean_subset
    - q-only base universe 仍使用 V2.6 online_evidence_ids；
    - 当前 K Evidence 允许作为已获取 state 节点；
    - offline witness 不参与 online graph；
    - adjacency 使用完整 file_version 中的真实 scoreable unit 顺序；
    - 但 structure 邻居只有已经存在于 base online universe 的节点才可见。
    用于测“只去掉 witness leakage + subset-adjacency shortcut”后的真实覆盖。

C_clean_expand
    与 B 相同，但允许 structure(K) 从 pre-fix file_version 中把真实 1-hop
    parent/child/adjacent Evidence Unit 拉入候选，即使它原本不在 lexical
    base universe 中。
    这是推荐的在线 state-aware structure semantics。

D_clean_expand_pure
    与 C 相同，但关闭 channel head reserve，作为诊断项。

注意
----
- 不修改 working DB。
- positive target 只用于离线 evaluation。
- offline witness 绝不进入 B/C/D 的 online candidate graph。
- 当前 K 是 state 已经获取的证据，因此即使它不是当前 q-only base retrieval
  的成员，也允许作为 structure expansion 的 source。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sqlite3
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Sequence


VARIANTS = ("B_clean_subset", "C_clean_expand", "D_clean_expand_pure")
K_VALUES = (1, 5, 10, 20, 64)


def open_ro(path: Path) -> sqlite3.Connection:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-131072")
    return conn


def load_builder(path: Path) -> Any:
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("v26_for_clean_structure", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def question_from_task(task: dict[str, Any]) -> str:
    task_input = task.get("input") or {}
    return "\n".join(
        [
            str(task_input.get("problem_statement") or ""),
            *[
                str(h)
                for h in task_input.get("hints") or []
                if str(h).strip()
            ],
        ]
    )


def supervision_witness_ids(supervision: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(eid)
            for obligation in supervision.get("obligations") or []
            for group in obligation.get("witness_groups") or []
            for eid in group.get("evidence_ids") or []
        }
    )


def positive_singles(actions: Sequence[dict[str, Any]]) -> set[str]:
    return {
        str((action.get("evidence_ids") or [None])[0])
        for action in actions
        if action.get("action_type") == "single"
        and action.get("action_label") == "positive"
        and bool(action.get("action_loss_mask"))
        and len(action.get("evidence_ids") or []) == 1
    }


def load_boundary_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            p.state_id,
            p.task_id,
            p.payload_json AS state_json,
            c.snapshot_id,
            c.final_split,
            c.payload_json AS task_json,
            s.payload_json AS supervision_json
        FROM policy_states p
        JOIN canonical_tasks c ON c.task_id=p.task_id
        JOIN supervision s ON s.task_id=p.task_id
        WHERE json_extract(p.payload_json, '$.state_type')='decision_boundary'
        ORDER BY p.task_id,p.state_id
        """
    ).fetchall()


def load_actions(
    conn: sqlite3.Connection, state_ids: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ids = list(state_ids)
    for offset in range(0, len(ids), 700):
        chunk = ids[offset : offset + 700]
        placeholders = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"SELECT state_id,payload_json FROM candidate_actions "
            f"WHERE state_id IN ({placeholders}) ORDER BY state_id,action_key",
            chunk,
        ):
            result[str(row["state_id"])].append(json.loads(row["payload_json"]))
    return result


def persisted_hit(actions: Sequence[dict[str, Any]]) -> bool:
    return any(
        action.get("action_type") != "stop"
        and action.get("action_label") == "positive"
        and bool(action.get("action_loss_mask"))
        and action.get("candidate_scope") == "online"
        for action in actions
    )


def load_record_by_evidence_id(
    builder: Any,
    conn: sqlite3.Connection,
    evidence_id: str,
) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT file_version_id FROM evidence_units "
        "WHERE evidence_id=? AND scoreable=1",
        (evidence_id,),
    ).fetchone()
    if row is None:
        return None
    file_id = str(row["file_version_id"])
    records = builder._load_cached_policy_records(conn, [file_id]).get(file_id, [])
    for record in records:
        if str(record["evidence_id"]) == evidence_id:
            return dict(record)
    return None


def build_canonical_file_graph(
    records: Sequence[dict[str, Any]],
) -> dict[str, set[str]]:
    """完整 file_version 上的真实 parent/child + 真实相邻 scoreable unit。"""

    graph: dict[str, set[str]] = {
        str(record["evidence_id"]): set() for record in records
    }
    by_id = {str(record["evidence_id"]): record for record in records}

    for record in records:
        eid = str(record["evidence_id"])
        parent = record.get("parent_evidence_id")
        if parent is not None and str(parent) in by_id:
            parent = str(parent)
            graph[eid].add(parent)
            graph[parent].add(eid)

    ordered = sorted(
        records,
        key=lambda record: (
            int(record.get("start_line") or 0),
            int(record.get("end_line") or 0),
            str(record["evidence_id"]),
        ),
    )
    for left, right in zip(ordered, ordered[1:]):
        a = str(left["evidence_id"])
        b = str(right["evidence_id"])
        graph[a].add(b)
        graph[b].add(a)

    return graph


def canonical_graph_for_k(
    builder: Any,
    conn: sqlite3.Connection,
    selected_ids: set[str],
) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]]]:
    """只加载 K 所在文件，构造真实 canonical structure graph。"""

    selected_file_ids: set[str] = set()
    selected_records: dict[str, dict[str, Any]] = {}

    for eid in sorted(selected_ids):
        record = load_record_by_evidence_id(builder, conn, eid)
        if record is None:
            continue
        selected_records[eid] = record
        selected_file_ids.add(str(record["file_version_id"]))

    records_by_file = builder._load_cached_policy_records(
        conn, sorted(selected_file_ids)
    )

    graph: dict[str, set[str]] = defaultdict(set)
    all_records: dict[str, dict[str, Any]] = {}
    for file_id, records in records_by_file.items():
        file_graph = build_canonical_file_graph(records)
        for source, targets in file_graph.items():
            graph[source].update(targets)
        for record in records:
            all_records[str(record["evidence_id"])] = dict(record)

    return graph, all_records


def structure_channel(
    selected_ids: set[str],
    canonical_graph: dict[str, set[str]],
    allowed_ids: set[str],
    depth: int,
) -> list[str]:
    counts: dict[str, int] = defaultdict(int)
    for source in selected_ids:
        for target in canonical_graph.get(source, set()):
            if target in allowed_ids and target not in selected_ids:
                counts[target] += 1
    return sorted(
        counts,
        key=lambda eid: (-counts[eid], eid),
    )[:depth]


def state_channels(
    builder: Any,
    *,
    question: str,
    base_records: list[dict[str, Any]],
    selected_ids: set[str],
    canonical_graph: dict[str, set[str]],
    canonical_records: dict[str, dict[str, Any]],
    allow_expand: bool,
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    """构造不含 offline witness 的 online-visible records + channels。"""

    visible_by_id = {
        str(record["evidence_id"]): dict(record) for record in base_records
    }

    # 当前 K 是已获取 evidence，允许作为 state 节点，但本身不作为 candidate。
    for eid in selected_ids:
        if eid not in visible_by_id:
            record = canonical_records.get(eid)
            if record is not None:
                visible_by_id[eid] = dict(record)

    if allow_expand:
        for source in selected_ids:
            for target in canonical_graph.get(source, set()):
                if target not in visible_by_id:
                    record = canonical_records.get(target)
                    if record is not None:
                        visible_by_id[target] = dict(record)

    visible_records = list(visible_by_id.values())

    # q-only 三通道直接在当前 honest visible records 上算。
    dynamic = builder.retrieve_online_channels(
        question,
        visible_records,
        state_evidence_ids=(),
        structural_edges=None,
        channel_depth=int(builder.CHANNEL_DEPTH),
    )

    allowed = set(visible_by_id) - selected_ids
    dynamic["structure"] = structure_channel(
        selected_ids,
        canonical_graph,
        allowed,
        int(builder.CHANNEL_DEPTH),
    )
    return dynamic, visible_by_id


def fuse(
    builder: Any,
    channels: dict[str, list[str]],
    *,
    head_reserve: int,
) -> list[str]:
    return [
        str(item["evidence_id"])
        for item in builder.reciprocal_rank_fusion(
            channels,
            depth=int(builder.FINAL_DEPTH),
            rrf_k=int(builder.RRF_K),
            channel_head_reserve=head_reserve,
        )
    ]


def rank_of_positive(positive: set[str], ids: Sequence[str]) -> int | None:
    for index, eid in enumerate(ids, 1):
        if eid in positive:
            return index
    return None


def aggregate(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    n = len(rows)
    hits = sum(bool(row[f"{variant}_hit"]) for row in rows)
    out = {
        "state_count": n,
        "hit_count": hits,
        "coverage": hits / n if n else None,
        "structure_positive_hit_count": sum(
            bool(row[f"{variant}_structure_hit"]) for row in rows
        ),
        "structure_positive_hit_rate": (
            sum(bool(row[f"{variant}_structure_hit"]) for row in rows) / n
            if n else None
        ),
    }
    for k in K_VALUES:
        count = sum(
            row[f"{variant}_best_positive_rank"] is not None
            and int(row[f"{variant}_best_positive_rank"]) <= k
            for row in rows
        )
        out[f"recall_at_{k}"] = count / n if n else None
        out[f"recall_at_{k}_count"] = count
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
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
        default=Path("data/.build/audit_boundary_clean_structure_v2_9"),
    )
    parser.add_argument("--max-states", type=int, default=None)
    args = parser.parse_args()

    builder = load_builder(args.builder)
    db = open_ro(args.db)
    fts = open_ro(args.fts)
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    try:
        boundary = load_boundary_rows(db)
        if args.max_states:
            boundary = boundary[: args.max_states]
        actions_by_state = load_actions(
            db, [str(row["state_id"]) for row in boundary]
        )

        rows_out: list[dict[str, Any]] = []

        for index, row in enumerate(boundary, 1):
            state = json.loads(row["state_json"])
            task = json.loads(row["task_json"])
            supervision = json.loads(row["supervision_json"])
            question = question_from_task(task)
            selected_ids = set(map(str, state.get("evidence_ids") or []))
            actions = actions_by_state[str(row["state_id"])]
            positives = positive_singles(actions)

            # 只使用 V2.6 的 honest online base IDs；offline witness records 从 graph 中剔除。
            evidence_all, online_ids = builder._load_policy_evidence_universe(
                db,
                snapshot_id=str(row["snapshot_id"]),
                question=question,
                witness_evidence_ids=supervision_witness_ids(supervision),
                repo_cache_index=None,
                fts_connection=fts,
            )
            base_records = [
                dict(evidence_all[eid])
                for eid in online_ids
                if eid in evidence_all
            ]

            canonical_graph, canonical_records = canonical_graph_for_k(
                builder, db, selected_ids
            )

            subset_channels, subset_visible = state_channels(
                builder,
                question=question,
                base_records=base_records,
                selected_ids=selected_ids,
                canonical_graph=canonical_graph,
                canonical_records=canonical_records,
                allow_expand=False,
            )
            expand_channels, expand_visible = state_channels(
                builder,
                question=question,
                base_records=base_records,
                selected_ids=selected_ids,
                canonical_graph=canonical_graph,
                canonical_records=canonical_records,
                allow_expand=True,
            )

            variants = {
                "B_clean_subset": (
                    subset_channels,
                    int(builder.CHANNEL_HEAD_RESERVE),
                ),
                "C_clean_expand": (
                    expand_channels,
                    int(builder.CHANNEL_HEAD_RESERVE),
                ),
                "D_clean_expand_pure": (expand_channels, 0),
            }

            result: dict[str, Any] = {
                "task_id": str(row["task_id"]),
                "state_id": str(row["state_id"]),
                "split": str(row["final_split"]),
                "positive_single_count": len(positives),
                "selected_k_count": len(selected_ids),
                "persisted_v26_hit": persisted_hit(actions),
                "base_online_count": len(base_records),
                "canonical_k_file_record_count": len(canonical_records),
                "clean_expand_added_count": len(expand_visible) - len(subset_visible),
            }

            for variant, (channels, head_reserve) in variants.items():
                final_ids = fuse(
                    builder, channels, head_reserve=head_reserve
                )
                best_rank = rank_of_positive(positives, final_ids)
                structure_hit = bool(
                    positives & set(channels.get("structure") or [])
                )
                result[f"{variant}_hit"] = best_rank is not None
                result[f"{variant}_best_positive_rank"] = best_rank
                result[f"{variant}_structure_hit"] = structure_hit

            rows_out.append(result)

            if index % 25 == 0 or index == len(boundary):
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"boundary-clean-structure-v2.9: {index}/{len(boundary)}, "
                    f"rate={index/elapsed:.2f} state/s",
                    file=sys.stderr,
                    flush=True,
                )

        persisted_count = sum(bool(row["persisted_v26_hit"]) for row in rows_out)
        report = {
            "mode": "read_only_clean_structure_audit",
            "definitions": {
                "offline_witness_in_online_graph": False,
                "canonical_adjacency": (
                    "完整 file_version scoreable Evidence Unit 顺序中的真实前后邻接，"
                    "不是当前候选子集中的相邻。"
                ),
                "B_clean_subset": (
                    "V2.6 lexical base + K source；canonical structure 只能命中已在 base 中的真实邻居。"
                ),
                "C_clean_expand": (
                    "B + canonical 1-hop parent/child/adjacent 可以从 pre-fix repo 拉入 base 外 Evidence。"
                ),
                "D_clean_expand_pure": "C + pure RRF（head reserve=0）。",
            },
            "counts": {
                "boundary_states": len(rows_out),
                "persisted_v26_hit_count": persisted_count,
                "persisted_v26_coverage": persisted_count / len(rows_out),
            },
            "results": {
                variant: aggregate(rows_out, variant)
                for variant in VARIANTS
            },
            "by_split": {
                split: {
                    variant: aggregate(
                        [row for row in rows_out if row["split"] == split],
                        variant,
                    )
                    for variant in VARIANTS
                }
                for split in sorted({row["split"] for row in rows_out})
            },
            "timing_seconds": time.perf_counter() - started,
        }

        report_path = outdir / "report.json"
        csv_path = outdir / "per_state.csv"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)

        print(
            json.dumps(
                {
                    "counts": report["counts"],
                    "results": report["results"],
                    "outputs": {
                        "report_json": str(report_path),
                        "per_state_csv": str(csv_path),
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        fts.close()
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
