#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2.8 boundary universe/trim 只读诊断。

本脚本用于回答一个具体问题：

    V2.6 decision-boundary coverage 从 V1 的约 76.26% 降到 63.85%，
    究竟发生在：
      1) 文件选择，
      2) 4096 Evidence Unit universe trim，
      3) structure(K) 可达性，
      4) channel Top-64，
      5) 最终 fusion
    哪一层？

同时构造一个 V1-like 对照：
    path-only Top32
    + 4096 unit cap
    + dynamic BM25/path/symbol/structure(K)
    + pure RRF Top64

它不是依赖旧 V1 policy 表，而是用当前冻结的 corpus/supervision 和当前
decision_boundary K 状态，只读重放旧 Retriever 逻辑。

重要：
- 不修改 unified_swe_v1.sqlite3。
- “目标文件”是 positive Evidence Unit 所属 file_version 的监督派生概念，
  不是 SWE-bench 原始字段，也不进入 Retriever 输入。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


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


def load_module(path: Path) -> Any:
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("v26_builder_for_v28_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_question(task_payload: dict[str, Any]) -> str:
    task_input = task_payload.get("input") or {}
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


def witness_ids(supervision: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(eid)
            for obligation in supervision.get("obligations") or []
            for group in obligation.get("witness_groups") or []
            for eid in group.get("evidence_ids") or []
        }
    )


def positive_targets(actions: Sequence[dict[str, Any]]) -> tuple[set[str], set[tuple[str, str]]]:
    singles: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for action in actions:
        if action.get("action_type") == "stop":
            continue
        if action.get("action_label") != "positive":
            continue
        if not bool(action.get("action_loss_mask")):
            continue
        ids = tuple(map(str, action.get("evidence_ids") or []))
        if action.get("action_type") == "single" and len(ids) == 1:
            singles.add(ids[0])
        elif action.get("action_type") == "pair" and len(ids) == 2:
            pairs.add(tuple(sorted(ids)))
    return singles, pairs


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
        ORDER BY p.task_id, p.state_id
        """
    ).fetchall()


def load_actions(
    conn: sqlite3.Connection, state_ids: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ids = list(state_ids)
    for off in range(0, len(ids), 700):
        chunk = ids[off : off + 700]
        placeholders = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"SELECT state_id,payload_json FROM candidate_actions "
            f"WHERE state_id IN ({placeholders}) ORDER BY state_id,action_key",
            chunk,
        ):
            result[str(row["state_id"])].append(json.loads(row["payload_json"]))
    return result


def evidence_file_map(
    conn: sqlite3.Connection, evidence_ids: set[str]
) -> dict[str, str]:
    result: dict[str, str] = {}
    ids = sorted(evidence_ids)
    for off in range(0, len(ids), 700):
        chunk = ids[off : off + 700]
        if not chunk:
            continue
        placeholders = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"SELECT evidence_id,file_version_id FROM evidence_units "
            f"WHERE evidence_id IN ({placeholders})",
            chunk,
        ):
            result[str(row["evidence_id"])] = str(row["file_version_id"])
    return result


def add_witness_records(
    builder: Any,
    conn: sqlite3.Connection,
    evidence_by_id: dict[str, dict[str, Any]],
    ids: Sequence[str],
) -> None:
    """把 witness 作为 offline-only records 补齐；不加入 online_ids。"""

    missing = sorted(set(map(str, ids)) - set(evidence_by_id))
    if not missing:
        return

    unit_file: dict[str, str] = {}
    for off in range(0, len(missing), 700):
        chunk = missing[off : off + 700]
        placeholders = ",".join("?" for _ in chunk)
        for row in conn.execute(
            f"SELECT evidence_id,file_version_id FROM evidence_units "
            f"WHERE evidence_id IN ({placeholders}) AND scoreable=1",
            chunk,
        ):
            unit_file[str(row["evidence_id"])] = str(row["file_version_id"])

    records_by_file = builder._load_cached_policy_records(
        conn, sorted(set(unit_file.values()))
    )
    maps = {
        file_id: {str(r["evidence_id"]): r for r in records}
        for file_id, records in records_by_file.items()
    }
    for eid, file_id in unit_file.items():
        record = maps.get(file_id, {}).get(eid)
        if record is not None:
            evidence_by_id[eid] = dict(record)


def annotate_records_for_memberships(
    builder: Any,
    conn: sqlite3.Connection,
    selected_memberships: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    file_ids = list(
        dict.fromkeys(str(item["file_version_id"]) for item in selected_memberships)
    )
    records_by_file = builder._load_cached_policy_records(conn, file_ids)
    result: list[dict[str, Any]] = []

    for membership in selected_memberships:
        file_id = str(membership["file_version_id"])
        for base in records_by_file.get(file_id, []):
            record = dict(base)
            record["_file_candidate_sources"] = list(
                membership.get("candidate_file_sources") or []
            )
            record["_grep_hit_lines"] = list(
                membership.get("content_hit_lines") or []
            )
            record["_grep_matched_terms"] = list(
                membership.get("content_matched_terms") or []
            )
            result.append(record)
    return result


def trim_online_records(
    builder: Any,
    question: str,
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """复制 V2.6 _load_policy_evidence_universe 的 4096 trim key。"""

    records = list(records)
    if len(records) <= int(builder.ONLINE_UNIT_UNIVERSE_CAP):
        return records

    query_terms = set(builder._retrieval_terms(question))

    def key(record: dict[str, Any]) -> tuple[Any, ...]:
        hit_lines = [int(x) for x in record.get("_grep_hit_lines") or []]
        start = int(record.get("start_line") or 0)
        end = int(record.get("end_line") or start)
        direct_hit = any(start <= line <= end for line in hit_lines)
        if hit_lines:
            distance = min(
                0 if start <= line <= end else min(abs(line - start), abs(line - end))
                for line in hit_lines
            )
        else:
            distance = 2**31 - 1

        sources = set(map(str, record.get("_file_candidate_sources") or []))
        searchable = " ".join(
            [
                str(record.get("path") or ""),
                str(record.get("symbol") or ""),
                str(record.get("content") or ""),
            ]
        ).lower()
        overlap = sum(term in searchable for term in query_terms)
        digest = hashlib.sha256(
            f"{question}\0{record['evidence_id']}".encode("utf-8")
        ).hexdigest()
        return (
            -int(direct_hit),
            -int(bool({"content_fts_file", "git_grep_content"} & sources)),
            distance,
            -overlap,
            -int("path_name_file" in sources),
            digest,
            str(record["evidence_id"]),
        )

    return sorted(records, key=key)[: int(builder.ONLINE_UNIT_UNIVERSE_CAP)]


def visible_for_state(
    base_online_ids: Sequence[str],
    selected_ids: set[str],
    graph: dict[str, Sequence[str]],
    evidence_by_id: dict[str, dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    expanded = {
        str(target)
        for source in selected_ids
        for target in graph.get(source, ())
        if str(target) in evidence_by_id
    }
    visible_ids = list(dict.fromkeys([*map(str, base_online_ids), *sorted(expanded)]))
    records = [
        evidence_by_id[eid]
        for eid in visible_ids
        if eid in evidence_by_id
    ]
    return visible_ids, records


def fusion_result(
    builder: Any,
    question: str,
    selected_ids: set[str],
    evidence_by_id: dict[str, dict[str, Any]],
    online_ids: Sequence[str],
    *,
    head_reserve: int,
) -> tuple[dict[str, list[str]], list[str]]:
    graph = builder.build_policy_structural_edges(evidence_by_id)
    _, visible_records = visible_for_state(
        online_ids, selected_ids, graph, evidence_by_id
    )
    channels = builder.retrieve_online_channels(
        question,
        visible_records,
        state_evidence_ids=sorted(selected_ids),
        structural_edges=graph,
        channel_depth=int(builder.CHANNEL_DEPTH),
    )
    fused = builder.reciprocal_rank_fusion(
        channels,
        depth=int(builder.FINAL_DEPTH),
        rrf_k=int(builder.RRF_K),
        channel_head_reserve=head_reserve,
    )
    return channels, [str(item["evidence_id"]) for item in fused]


def any_positive_single_hit(positive: set[str], ids: Sequence[str]) -> bool:
    return bool(positive & set(map(str, ids)))


def any_channel_hit(
    positive: set[str], channels: dict[str, Sequence[str]]
) -> bool:
    return any(positive & set(map(str, ranking)) for ranking in channels.values())


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)

    def count(field: str) -> int:
        return sum(bool(row[field]) for row in rows)

    stage_counts = defaultdict(int)
    for row in rows:
        stage_counts[row["v26_miss_stage"]] += 1

    return {
        "state_count": n,
        "v26_persisted_hit_count": count("v26_persisted_hit"),
        "v26_persisted_coverage": count("v26_persisted_hit") / n,
        "v1_like_hit_count": count("v1_like_hit"),
        "v1_like_coverage": count("v1_like_hit") / n,
        "v26_positive_file_selected_count": count("v26_positive_file_selected"),
        "path32_positive_file_selected_count": count("path32_positive_file_selected"),
        "v26_positive_pretrim_count": count("v26_positive_pretrim"),
        "v26_positive_posttrim_online_count": count("v26_positive_posttrim_online"),
        "path32_positive_posttrim_online_count": count("path32_positive_posttrim_online"),
        "v26_structure_hit_count": count("v26_structure_hit"),
        "path32_structure_hit_count": count("path32_structure_hit"),
        "v26_any_channel_hit_count": count("v26_any_channel_hit"),
        "path32_any_channel_hit_count": count("path32_any_channel_hit"),
        "v26_miss_stage_counts": dict(sorted(stage_counts.items())),
        "path32_recovers_v26_miss_count": sum(
            (not row["v26_persisted_hit"]) and row["v1_like_hit"]
            for row in rows
        ),
        "v26_recovers_path32_miss_count": sum(
            row["v26_persisted_hit"] and (not row["v1_like_hit"])
            for row in rows
        ),
    }


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
        default=Path("data/.build/audit_boundary_universe_v2_8"),
    )
    parser.add_argument("--max-states", type=int, default=None)
    args = parser.parse_args()

    builder = load_module(args.builder)
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
            db, [str(r["state_id"]) for r in boundary]
        )

        output_rows: list[dict[str, Any]] = []

        for idx, row in enumerate(boundary, 1):
            task_id = str(row["task_id"])
            state_id = str(row["state_id"])
            state = json.loads(row["state_json"])
            task = json.loads(row["task_json"])
            supervision = json.loads(row["supervision_json"])
            question = task_question(task)
            selected_k = set(map(str, state.get("evidence_ids") or []))
            witnesses = witness_ids(supervision)

            positives, positive_pairs = positive_targets(
                actions_by_state.get(state_id, [])
            )
            positive_all = set(positives)
            for pair in positive_pairs:
                positive_all.update(pair)
            file_by_eid = evidence_file_map(db, positive_all)
            positive_files = set(file_by_eid.values())

            # 当前 V2.6 exact universe。
            v26_evidence, v26_online = builder._load_policy_evidence_universe(
                db,
                snapshot_id=str(row["snapshot_id"]),
                question=question,
                witness_evidence_ids=witnesses,
                repo_cache_index=None,
                fts_connection=fts,
            )
            v26_graph = builder.build_policy_structural_edges(v26_evidence)
            v26_visible_ids, v26_visible_records = visible_for_state(
                v26_online, selected_k, v26_graph, v26_evidence
            )
            v26_channels = builder.task_query_channels_for_state(
                precomputed_rankings=builder.precompute_task_query_channel_rankings(
                    question, list(v26_evidence.values())
                ),
                visible_ids=set(v26_visible_ids),
                selected_ids=selected_k,
                structural_edges=v26_graph,
                channel_depth=int(builder.CHANNEL_DEPTH),
            )
            v26_fused = builder.reciprocal_rank_fusion(
                v26_channels,
                depth=int(builder.FINAL_DEPTH),
                rrf_k=int(builder.RRF_K),
                channel_head_reserve=int(builder.CHANNEL_HEAD_RESERVE),
            )
            v26_final_ids = [str(x["evidence_id"]) for x in v26_fused]

            # 重新取 membership + FTS file candidates，以得到 exact selected file stage。
            memberships = [
                dict(x)
                for x in db.execute(
                    "SELECT path,file_version_id FROM snapshot_file_memberships "
                    "WHERE snapshot_id=? ORDER BY path",
                    (str(row["snapshot_id"]),),
                ).fetchall()
            ]
            repo_row = db.execute(
                "SELECT repo FROM snapshots WHERE snapshot_id=?",
                (str(row["snapshot_id"]),),
            ).fetchone()
            repo = str(repo_row["repo"])
            membership_ids = {str(x["file_version_id"]) for x in memberships}
            content_candidates = builder.query_policy_file_fts(
                fts,
                repo=repo,
                question=question,
                membership_file_ids=membership_ids,
                cap=int(builder.CONTENT_FILE_CAP),
            )
            v26_selected_memberships = builder.select_online_file_memberships(
                question,
                memberships,
                content_candidates=content_candidates,
                cap=int(builder.ONLINE_FILE_CAP),
                path_cap=int(builder.PATH_FILE_CAP),
                content_cap=int(builder.CONTENT_FILE_CAP),
            )
            v26_selected_files = {
                str(x["file_version_id"]) for x in v26_selected_memberships
            }
            v26_pretrim_records = annotate_records_for_memberships(
                builder, db, v26_selected_memberships
            )
            v26_pretrim_ids = {
                str(x["evidence_id"]) for x in v26_pretrim_records
            }

            # V1-like：只取 path Top32，按同一 4096 trim 逻辑，然后 dynamic+pure RRF。
            path_memberships = builder.select_online_file_memberships(
                question,
                memberships,
                content_candidates=(),
                content_hits=(),
                cap=int(builder.PATH_FILE_CAP),
                path_cap=int(builder.PATH_FILE_CAP),
                content_cap=0,
            )
            path_files = {
                str(x["file_version_id"]) for x in path_memberships
            }
            path_pretrim = annotate_records_for_memberships(
                builder, db, path_memberships
            )
            path_trimmed = trim_online_records(builder, question, path_pretrim)
            path_online_ids = [str(x["evidence_id"]) for x in path_trimmed]
            path_evidence = {
                str(x["evidence_id"]): dict(x) for x in path_trimmed
            }
            add_witness_records(builder, db, path_evidence, witnesses)
            path_graph = builder.build_policy_structural_edges(path_evidence)
            path_visible_ids, path_visible_records = visible_for_state(
                path_online_ids, selected_k, path_graph, path_evidence
            )
            path_channels = builder.retrieve_online_channels(
                question,
                path_visible_records,
                state_evidence_ids=sorted(selected_k),
                structural_edges=path_graph,
                channel_depth=int(builder.CHANNEL_DEPTH),
            )
            path_fused = builder.reciprocal_rank_fusion(
                path_channels,
                depth=int(builder.FINAL_DEPTH),
                rrf_k=int(builder.RRF_K),
                channel_head_reserve=0,
            )
            path_final_ids = [str(x["evidence_id"]) for x in path_fused]

            persisted_hit = any(
                a.get("action_label") == "positive"
                and bool(a.get("action_loss_mask"))
                and a.get("candidate_scope") == "online"
                and a.get("action_type") != "stop"
                for a in actions_by_state.get(state_id, [])
            )

            v26_file_selected = bool(positive_files & v26_selected_files)
            path_file_selected = bool(positive_files & path_files)
            v26_pretrim_hit = bool(positive_all & v26_pretrim_ids)
            v26_posttrim_hit = any_positive_single_hit(positives, v26_online)
            path_posttrim_hit = any_positive_single_hit(positives, path_online_ids)
            v26_structure_hit = any_positive_single_hit(
                positives, v26_channels.get("structure", [])
            )
            path_structure_hit = any_positive_single_hit(
                positives, path_channels.get("structure", [])
            )
            v26_channel_hit = any_channel_hit(positives, v26_channels)
            path_channel_hit = any_channel_hit(positives, path_channels)
            v26_final_hit = any_positive_single_hit(positives, v26_final_ids)
            path_final_hit = any_positive_single_hit(positives, path_final_ids)

            # 对 V2.6 miss 做第一失败阶段分类。
            if persisted_hit:
                miss_stage = "hit"
            elif not v26_file_selected:
                miss_stage = "file_selection"
            elif not v26_pretrim_hit:
                miss_stage = "positive_unit_not_in_selected_file_records"
            elif not v26_posttrim_hit and not v26_structure_hit:
                miss_stage = "unit_trim_or_structure_reachability"
            elif not v26_channel_hit:
                miss_stage = "channel_top64"
            elif not v26_final_hit:
                miss_stage = "fusion_top64"
            else:
                # 可能是 pair/action materialization 差异。
                miss_stage = "action_materialization_or_pair"

            output_rows.append(
                {
                    "task_id": task_id,
                    "state_id": state_id,
                    "split": str(row["final_split"]),
                    "positive_single_count": len(positives),
                    "positive_pair_count": len(positive_pairs),
                    "positive_file_count": len(positive_files),
                    "v26_persisted_hit": persisted_hit,
                    "v26_replay_final_single_hit": v26_final_hit,
                    "v1_like_hit": path_final_hit,
                    "v26_positive_file_selected": v26_file_selected,
                    "path32_positive_file_selected": path_file_selected,
                    "v26_positive_pretrim": v26_pretrim_hit,
                    "v26_positive_posttrim_online": v26_posttrim_hit,
                    "path32_positive_posttrim_online": path_posttrim_hit,
                    "v26_structure_hit": v26_structure_hit,
                    "path32_structure_hit": path_structure_hit,
                    "v26_any_channel_hit": v26_channel_hit,
                    "path32_any_channel_hit": path_channel_hit,
                    "v26_miss_stage": miss_stage,
                    "v26_selected_file_count": len(v26_selected_files),
                    "path32_selected_file_count": len(path_files),
                    "v26_pretrim_unit_count": len(v26_pretrim_records),
                    "v26_posttrim_online_unit_count": len(v26_online),
                    "path32_pretrim_unit_count": len(path_pretrim),
                    "path32_posttrim_online_unit_count": len(path_online_ids),
                }
            )

            if idx % 25 == 0 or idx == len(boundary):
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"boundary-universe-v2.8: {idx}/{len(boundary)}, "
                    f"rate={idx/elapsed:.2f} state/s",
                    file=sys.stderr,
                    flush=True,
                )

        # persisted vs replay sanity
        replay_match = sum(
            row["v26_persisted_hit"] == row["v26_replay_final_single_hit"]
            for row in output_rows
        )

        report = {
            "mode": "read_only_boundary_universe_stage_audit",
            "definitions": {
                "derived_target_file": (
                    "positive Evidence Unit 所属 file_version；"
                    "是离线 supervision 派生目标，不是 SWE-bench 原始字段。"
                ),
                "v1_like": (
                    "path-only Top32 + 4096 unit cap + dynamic four channels "
                    "+ pure RRF Top64"
                ),
                "v26": (
                    "path Top32 + FTS content Top64 file union + 4096 unit cap "
                    "+ V2.6 static q-only channels + structure(K) + head RRF"
                ),
            },
            "counts": {
                "boundary_states": len(output_rows),
                "unique_tasks": len({r["task_id"] for r in output_rows}),
            },
            "v26_replay_consistency": {
                "per_state_hit_match_count": replay_match,
                "per_state_hit_match_rate": replay_match / len(output_rows),
                "strict_ok": replay_match == len(output_rows),
            },
            "overall": summarize(output_rows),
            "by_split": {
                split: summarize(
                    [r for r in output_rows if r["split"] == split]
                )
                for split in sorted({r["split"] for r in output_rows})
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

        fieldnames = list(output_rows[0].keys())
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

        print(
            json.dumps(
                {
                    "v26_replay_consistency": report["v26_replay_consistency"],
                    "overall": report["overall"],
                    "outputs": {
                        "report_json": str(report_path),
                        "per_state_csv": str(csv_path),
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0 if report["v26_replay_consistency"]["strict_ok"] else 2

    finally:
        fts.close()
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
