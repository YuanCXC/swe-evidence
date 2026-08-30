#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_integrated_training_dataset_v1_3.py

独立审计 Strong-Teacher 派生 V2.10 数据，重点验证 Teacher obligation 改动之后
所有下游 Policy 依赖都已经重新生成，而不是继续沿用旧 trajectory/policy labels。

默认：
  base:    data/upstream/unified_swe_dataset_v2_10/
  derived: data/upstream/unified_swe_dataset_v2_10_teacher_v1/
  db:      data/.build/unified_swe_v1.sqlite3

检查：
- task/split/immutable model-visible fields 不变；
- 顶层 source trajectories 不变；
- mechanical Teacher 不擅自提升 level/recommended_weight；
- changed obligations => policy_states 必须 rebuilt；
- unchanged/no-Teacher => 旧 policy_states 必须保持；
- state/action/STOP/action_label/scoreability 结构合法；
- witness/action evidence_id 全部存在；
- witness 不得同时仍是 hard negative；
- witness 必须属于当前 task 的 pre-fix snapshot；
- benchmark/trainability 等 split 信息保持不变；
- semantic review 未完成时 training_ready 必须仍为 false。

建议放：scripts/audit_integrated_training_dataset_v1_3.py
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCRIPT_VERSION = "1.3.0"
SPLITS = ("train", "validation", "benchmark")
EXPECTED = {"train": 18347, "validation": 223, "benchmark": 2294}
IMMUTABLE_FIELDS = (
    "schema_version",
    "task_id",
    "task_group_id",
    "snapshot_id",
    "input",
    "provenance",
    "trajectories",
    "evaluation",
    "split_info",
    "quality",
)


def stable_json(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(chunk), b""):
            h.update(part)
    return h.hexdigest()


def require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("缺少 pyarrow") from exc
    return pq


def resolve_split(root: Path, split: str, *, base: bool) -> Path:
    candidates = (
        (root / f"{split}_v2_10.parquet", root / f"{split}.parquet")
        if base
        else (root / f"{split}.parquet", root / f"{split}_v2_10.parquet")
    )
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"找不到 {split} parquet: {root}")


def collect_witness_ids(supervision: Mapping[str, Any]) -> set[str]:
    return {
        str(eid)
        for ob in (supervision.get("obligations") or [])
        if isinstance(ob, Mapping)
        for group in (ob.get("witness_groups") or [])
        if isinstance(group, Mapping)
        for eid in (group.get("evidence_ids") or [])
        if str(eid).strip()
    }


def all_action_ids(supervision: Mapping[str, Any]) -> set[str]:
    return {
        str(eid)
        for state in (supervision.get("policy_states") or [])
        if isinstance(state, Mapping)
        for action in (state.get("candidate_actions") or [])
        if isinstance(action, Mapping)
        for eid in (action.get("evidence_ids") or [])
        if str(eid).strip()
    }


def _mandatory_obligation_count(supervision: Mapping[str, Any]) -> int:
    return sum(
        1
        for ob in (supervision.get("obligations") or [])
        if isinstance(ob, Mapping)
        and ob.get("applicable") is True
        and ob.get("mandatory") is True
    )


def _completion_for_state(
    evidence_ids: Sequence[str],
    supervision: Mapping[str, Any],
) -> float | None:
    selected = set(map(str, evidence_ids))
    mandatory = [
        ob
        for ob in (supervision.get("obligations") or [])
        if isinstance(ob, Mapping)
        and ob.get("applicable") is True
        and ob.get("mandatory") is True
    ]
    if not mandatory:
        # 与冻结 V2.10 builder 的 evidence_state_metrics 契约一致：
        # 没有 mandatory repository obligation 时 completion_score=None，
        # 不能仅凭 state_type="complete" 推导 STOP positive。
        return None
    completed = 0
    for ob in mandatory:
        groups = [
            set(map(str, g.get("evidence_ids") or []))
            for g in (ob.get("witness_groups") or [])
            if isinstance(g, Mapping) and g.get("evidence_ids")
        ]
        if any(group <= selected for group in groups):
            completed += 1
    return completed / len(mandatory)


def state_action_errors(task_id: str, supervision: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    states = supervision.get("policy_states") or []
    if not isinstance(states, list) or not (2 <= len(states) <= 3):
        return [f"POLICY_STATE_COUNT:{task_id}:{len(states) if isinstance(states, list) else 'non-list'}"]

    mandatory_count = _mandatory_obligation_count(supervision)
    types = [str(s.get("state_type") or "") for s in states if isinstance(s, Mapping)]
    if types.count("initial") != 1:
        errors.append(f"INITIAL_STATE_COUNT:{task_id}:{types.count('initial')}")
    if types.count("complete") != 1:
        errors.append(f"COMPLETE_STATE_COUNT:{task_id}:{types.count('complete')}")
    if types.count("decision_boundary") > 1:
        errors.append(f"BOUNDARY_STATE_COUNT:{task_id}:{types.count('decision_boundary')}")

    seen_state_ids: set[str] = set()
    for state in states:
        if not isinstance(state, Mapping):
            errors.append(f"STATE_NOT_OBJECT:{task_id}")
            continue
        state_id = str(state.get("state_id") or "")
        if not state_id or state_id in seen_state_ids:
            errors.append(f"STATE_ID_INVALID:{task_id}:{state_id}")
        seen_state_ids.add(state_id)
        stype = str(state.get("state_type") or "")
        evidence_ids = [str(x) for x in (state.get("evidence_ids") or [])]
        if len(evidence_ids) != len(set(evidence_ids)):
            errors.append(f"STATE_DUP_EVIDENCE:{task_id}:{state_id}")
        if stype == "initial" and evidence_ids:
            errors.append(f"INITIAL_NOT_EMPTY:{task_id}:{state_id}")

        stored_completion = state.get("completion_score")
        recomputed_completion = _completion_for_state(evidence_ids, supervision)
        if recomputed_completion is None:
            if stored_completion is not None:
                errors.append(
                    f"COMPLETION_SHOULD_BE_NULL:{task_id}:{state_id}:{stored_completion}"
                )
        else:
            if stored_completion is None or abs(float(stored_completion) - recomputed_completion) > 1e-5:
                errors.append(
                    f"COMPLETION_MISMATCH:{task_id}:{state_id}:{stored_completion}:{recomputed_completion}"
                )

        stop_label = str(state.get("stop_label") or "")
        stop_loss_mask = bool(state.get("stop_loss_mask"))

        # V2.10 真正的 STOP 契约由 completion_score 决定，而不是 state_type 名称：
        # - completion=None（没有 mandatory obligations）=> STOP unknown / mask false
        # - completion=1 => STOP positive
        # - 其它 completion => STOP 不得 positive
        if recomputed_completion is None:
            if stop_label != "unknown":
                errors.append(f"NO_MANDATORY_STOP_NOT_UNKNOWN:{task_id}:{state_id}:{stop_label}")
            if stop_loss_mask:
                errors.append(f"NO_MANDATORY_STOP_MASKED_IN:{task_id}:{state_id}")
        elif recomputed_completion >= 1.0 - 1e-9:
            if stop_label != "positive":
                errors.append(f"SUFFICIENT_STOP_NOT_POSITIVE:{task_id}:{state_id}:{stop_label}")
            if not stop_loss_mask:
                errors.append(f"SUFFICIENT_STOP_MASKED:{task_id}:{state_id}")
        elif stop_label == "positive":
            errors.append(f"INSUFFICIENT_STOP_POSITIVE:{task_id}:{state_id}")

        if stype == "complete" and mandatory_count > 0:
            if recomputed_completion is None or recomputed_completion < 1.0 - 1e-9:
                errors.append(
                    f"COMPLETE_NOT_SUFFICIENT:{task_id}:{state_id}:{recomputed_completion}"
                )

        actions = state.get("candidate_actions") or []
        if not isinstance(actions, list) or not actions:
            errors.append(f"EMPTY_ACTIONS:{task_id}:{state_id}")
            continue
        stop_count = 0
        stop_action: Mapping[str, Any] | None = None
        action_ids: set[str] = set()
        known_labels: list[str] = []
        for action in actions:
            if not isinstance(action, Mapping):
                errors.append(f"ACTION_NOT_OBJECT:{task_id}:{state_id}")
                continue
            aid = str(action.get("action_id") or "")
            if not aid or aid in action_ids:
                errors.append(f"ACTION_ID_INVALID:{task_id}:{state_id}:{aid}")
            action_ids.add(aid)
            typ = str(action.get("action_type") or "")
            ids = [str(x) for x in (action.get("evidence_ids") or [])]
            if len(ids) != len(set(ids)):
                errors.append(f"ACTION_DUP_EVIDENCE:{task_id}:{aid}")
            if typ == "stop":
                stop_count += 1
                stop_action = action
                if ids:
                    errors.append(f"STOP_HAS_EVIDENCE:{task_id}:{aid}")
            elif typ == "single":
                if len(ids) != 1:
                    errors.append(f"SINGLE_ARITY:{task_id}:{aid}:{len(ids)}")
            elif typ == "pair":
                if len(ids) != 2:
                    errors.append(f"PAIR_ARITY:{task_id}:{aid}:{len(ids)}")
            else:
                errors.append(f"ACTION_TYPE:{task_id}:{aid}:{typ}")

            label = str(action.get("action_label") or "")
            scoreable = action.get("scoreable") is True
            loss_mask = action.get("action_loss_mask") is True
            if loss_mask and not scoreable:
                errors.append(f"LOSS_ACTIVE_UNSCOREABLE:{task_id}:{aid}")
            if label == "positive":
                if not scoreable:
                    errors.append(f"POSITIVE_UNSCOREABLE:{task_id}:{aid}")
                if not loss_mask:
                    errors.append(f"POSITIVE_MASKED:{task_id}:{aid}")
            if loss_mask:
                known_labels.append(label)

        if stop_count != 1:
            errors.append(f"STOP_ACTION_COUNT:{task_id}:{state_id}:{stop_count}")
        elif stop_action is not None:
            if str(stop_action.get("action_label") or "") != stop_label:
                errors.append(
                    f"STOP_LABEL_MISMATCH:{task_id}:{state_id}:"
                    f"state={stop_label}:action={stop_action.get('action_label')}"
                )
            if bool(stop_action.get("action_loss_mask")) != stop_loss_mask:
                errors.append(
                    f"STOP_MASK_MISMATCH:{task_id}:{state_id}:"
                    f"state={stop_loss_mask}:action={bool(stop_action.get('action_loss_mask'))}"
                )

        expected_ranking_mask = (
            any(label == "positive" for label in known_labels)
            and any(label == "negative" for label in known_labels)
        )
        if bool(state.get("ranking_loss_mask")) != expected_ranking_mask:
            errors.append(
                f"RANKING_MASK_MISMATCH:{task_id}:{state_id}:"
                f"stored={bool(state.get('ranking_loss_mask'))}:expected={expected_ranking_mask}"
            )
    return errors

def sqlite_ro(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def validate_global_evidence(conn: sqlite3.Connection, ids: set[str]) -> list[str]:
    found: set[str] = set()
    vals = sorted(ids)
    for off in range(0, len(vals), 800):
        chunk = vals[off:off+800]
        if not chunk:
            continue
        ph = ",".join("?" for _ in chunk)
        found.update(str(r[0]) for r in conn.execute(
            f"SELECT evidence_id FROM evidence_units WHERE evidence_id IN ({ph})", chunk
        ))
    return sorted(set(vals) - found)


def validate_snapshot_pairs(
    conn: sqlite3.Connection,
    by_snapshot: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    bad: list[dict[str, Any]] = []
    for idx, (snapshot_id, ids) in enumerate(by_snapshot.items(), 1):
        found: set[str] = set()
        vals = sorted(ids)
        for off in range(0, len(vals), 500):
            chunk = vals[off:off+500]
            ph = ",".join("?" for _ in chunk)
            sql = (
                "SELECT DISTINCT e.evidence_id FROM evidence_units e "
                "JOIN snapshot_file_memberships m ON m.file_version_id=e.file_version_id "
                f"WHERE m.snapshot_id=? AND e.evidence_id IN ({ph})"
            )
            found.update(str(r[0]) for r in conn.execute(sql, [snapshot_id, *chunk]))
        missing = sorted(set(vals) - found)
        if missing:
            bad.append({"snapshot_id": snapshot_id, "missing": missing})
        if idx % 2000 == 0:
            print(f"[audit-integrated] snapshot membership {idx:,}/{len(by_snapshot):,}", flush=True)
    return bad


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=Path, default=Path("data/upstream/unified_swe_dataset_v2_10"))
    p.add_argument("--derived-dir", type=Path, default=Path("data/upstream/unified_swe_dataset_v2_10_teacher_v1"))
    p.add_argument("--build-db", type=Path, default=Path("data/.build/unified_swe_v1.sqlite3"))
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def self_test() -> int:
    supervision = {
        "obligations": [{"applicable": True, "mandatory": True, "obligation_id": "o1", "witness_groups": [{"evidence_ids": ["a", "b"]}]}],
        "hard_negative_evidence_ids": [],
        "policy_states": [
            {
                "state_id": "i", "state_type": "initial", "evidence_ids": [],
                "completion_score": 0.0, "stop_label": "negative", "stop_loss_mask": True,
                "ranking_loss_mask": True,
                "candidate_actions": [
                    {"action_id": "a", "action_type": "single", "evidence_ids": ["a"], "action_label": "positive", "scoreable": True, "action_loss_mask": True},
                    {"action_id": "s0", "action_type": "stop", "evidence_ids": [], "action_label": "negative", "scoreable": True, "action_loss_mask": True},
                ],
            },
            {
                "state_id": "c", "state_type": "complete", "evidence_ids": ["a", "b"],
                "completion_score": 1.0, "stop_label": "positive", "stop_loss_mask": True,
                "ranking_loss_mask": False,
                "candidate_actions": [
                    {"action_id": "s1", "action_type": "stop", "evidence_ids": [], "action_label": "positive", "scoreable": True, "action_loss_mask": True}
                ],
            },
        ],
    }
    assert collect_witness_ids(supervision) == {"a", "b"}
    assert not state_action_errors("t", supervision)

    no_mandatory = {
        "obligations": [],
        "hard_negative_evidence_ids": [],
        "policy_states": [
            {
                "state_id": "i0", "state_type": "initial", "evidence_ids": [],
                "completion_score": None, "stop_label": "unknown", "stop_loss_mask": False,
                "ranking_loss_mask": False,
                "candidate_actions": [
                    {"action_id": "s0", "action_type": "stop", "evidence_ids": [],
                     "action_label": "unknown", "scoreable": True, "action_loss_mask": False}
                ],
            },
            {
                "state_id": "c0", "state_type": "complete", "evidence_ids": [],
                "completion_score": None, "stop_label": "unknown", "stop_loss_mask": False,
                "ranking_loss_mask": False,
                "candidate_actions": [
                    {"action_id": "s1", "action_type": "stop", "evidence_ids": [],
                     "action_label": "unknown", "scoreable": True, "action_loss_mask": False}
                ],
            },
        ],
    }
    assert not state_action_errors("n", no_mandatory)
    print("SELF_TEST_OK")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    root = Path.cwd().resolve()
    base_dir = (root / args.base_dir).resolve() if not args.base_dir.is_absolute() else args.base_dir.resolve()
    derived_dir = (root / args.derived_dir).resolve() if not args.derived_dir.is_absolute() else args.derived_dir.resolve()
    db_path = (root / args.build_db).resolve() if not args.build_db.is_absolute() else args.build_db.resolve()
    report_path = (
        (derived_dir / "integrity_audit.json") if args.report is None
        else ((root / args.report).resolve() if not args.report.is_absolute() else args.report.resolve())
    )
    pq = require_pyarrow()
    manifest_path = derived_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"缺 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    errors: list[str] = []
    counters = Counter()
    all_evidence_ids: set[str] = set()
    witness_by_snapshot: dict[str, set[str]] = defaultdict(set)
    total = 0

    for split in SPLITS:
        base_path = resolve_split(base_dir, split, base=True)
        derived_path = resolve_split(derived_dir, split, base=False)
        bpf = pq.ParquetFile(base_path)
        dpf = pq.ParquetFile(derived_path)
        if bpf.metadata.num_rows != EXPECTED[split] or dpf.metadata.num_rows != EXPECTED[split]:
            errors.append(
                f"ROW_COUNT:{split}:base={bpf.metadata.num_rows}:derived={dpf.metadata.num_rows}:expected={EXPECTED[split]}"
            )
        biter = bpf.iter_batches(batch_size=args.batch_size)
        diter = dpf.iter_batches(batch_size=args.batch_size)
        split_count = 0
        while True:
            try:
                bb = next(biter)
            except StopIteration:
                bb = None
            try:
                db = next(diter)
            except StopIteration:
                db = None
            if bb is None and db is None:
                break
            if bb is None or db is None:
                errors.append(f"BATCH_LENGTH_MISMATCH:{split}")
                break
            brows = bb.to_pylist()
            drows = db.to_pylist()
            if len(brows) != len(drows):
                errors.append(f"BATCH_ROW_MISMATCH:{split}:{len(brows)}:{len(drows)}")
                break
            for base, der in zip(brows, drows):
                split_count += 1
                total += 1
                task_id = str(base.get("task_id") or "")
                if task_id != str(der.get("task_id") or ""):
                    errors.append(f"TASK_ORDER_OR_ID:{split}:{task_id}:{der.get('task_id')}")
                    continue
                for field in IMMUTABLE_FIELDS:
                    if stable_json(base.get(field)) != stable_json(der.get(field)):
                        errors.append(f"IMMUTABLE_CHANGED:{task_id}:{field}")

                bsuper = base.get("supervision") or {}
                dsuper = der.get("supervision") or {}
                status = str(der.get("strong_teacher_status") or "")
                rebuild_required = bool(der.get("strong_teacher_policy_rebuild_required"))
                rebuilt = bool(der.get("strong_teacher_policy_rebuilt"))
                counters[f"status:{status}"] += 1

                # Mechanical-only Teacher 不能自动升级 supervision strength/weight.
                if status == "INCLUDED":
                    if str(bsuper.get("level") or "") != str(dsuper.get("level") or ""):
                        errors.append(f"MECHANICAL_LEVEL_PROMOTED:{task_id}")
                    if float(bsuper.get("recommended_weight") or 0.0) != float(dsuper.get("recommended_weight") or 0.0):
                        errors.append(f"MECHANICAL_WEIGHT_CHANGED:{task_id}")

                rollback_reason = str(der.get("strong_teacher_exclusion_reason") or "")
                scoreability_rollback = rollback_reason == "POLICY_ROLLBACK_SCOREABILITY"

                # experiment_eligible=false 表示该 task 只保留用于 provenance/audit，
                # 不得进入 train/validation/benchmark 或聚合实验指标。
                # 兼容旧数据：字段不存在时默认 eligible=True。
                experiment_eligible = der.get("experiment_eligible") is not False
                if not experiment_eligible:
                    counters["experiment_excluded_tasks"] += 1
                    counters[f"experiment_excluded:{split}"] += 1
                    reason = str(der.get("experiment_exclusion_reason") or "")
                    if not reason:
                        errors.append(f"EXPERIMENT_EXCLUDED_WITHOUT_REASON:{task_id}")

                if _mandatory_obligation_count(dsuper) == 0:
                    counters["tasks_without_mandatory_obligations"] += 1

                if status in {"EXCLUDED", "NOT_IN_QUESTION_TREE"}:
                    if stable_json(bsuper) != stable_json(dsuper):
                        errors.append(f"BASE_SUPERVISION_CHANGED_WITHOUT_TEACHER:{task_id}:{status}")
                elif status == "INCLUDED":
                    if scoreability_rollback:
                        counters["teacher_policy_scoreability_rollback"] += 1
                        if stable_json(bsuper) != stable_json(dsuper):
                            errors.append(f"SCOREABILITY_ROLLBACK_NOT_BASE:{task_id}")
                        if rebuild_required or rebuilt:
                            errors.append(f"SCOREABILITY_ROLLBACK_FLAGS:{task_id}:{rebuild_required}:{rebuilt}")
                    elif rebuild_required:
                        counters["rebuild_required"] += 1
                        if not rebuilt:
                            errors.append(f"REBUILD_REQUIRED_NOT_DONE:{task_id}")
                        if stable_json(bsuper.get("obligations") or []) == stable_json(dsuper.get("obligations") or []):
                            errors.append(f"REBUILD_FLAG_WITHOUT_OBLIGATION_CHANGE:{task_id}")
                    else:
                        if rebuilt:
                            errors.append(f"UNNEEDED_REBUILD:{task_id}")
                        if stable_json(bsuper.get("policy_states") or []) != stable_json(dsuper.get("policy_states") or []):
                            errors.append(f"POLICY_CHANGED_WITHOUT_OBLIGATION_CHANGE:{task_id}")

                    witnesses = collect_witness_ids(dsuper)
                    hard_neg = set(map(str, dsuper.get("hard_negative_evidence_ids") or []))
                    conflict = sorted(witnesses & hard_neg)
                    if conflict:
                        errors.append(f"WITNESS_HARD_NEGATIVE_CONFLICT:{task_id}:{conflict[:10]}")
                    snapshot_id = str(der.get("snapshot_id") or "")
                    witness_by_snapshot[snapshot_id].update(witnesses)
                    all_evidence_ids.update(witnesses)

                # 只有 experiment_eligible=true 的 task 才必须满足训练/评测 Policy gate。
                # 被显式排除的 task 仍保留 supervision/policy_states 供 provenance/audit，
                # 但不因 scoreability 等训练门禁导致整个数据集审计失败。
                if experiment_eligible:
                    serr = state_action_errors(task_id, dsuper)
                    if serr:
                        errors.extend(serr[:50])
                        counters["tasks_with_policy_errors"] += 1
                else:
                    counters["experiment_excluded_policy_checks_skipped"] += 1
                all_evidence_ids.update(all_action_ids(dsuper))

                if args.progress_every > 0 and total % args.progress_every == 0:
                    print(
                        f"[audit-integrated] {total:,} tasks | errors={len(errors):,} | rebuild={counters['rebuild_required']:,}",
                        flush=True,
                    )
        if split_count != EXPECTED[split]:
            errors.append(f"SPLIT_COUNT:{split}:{split_count}:{EXPECTED[split]}")

    conn = sqlite_ro(db_path)
    try:
        missing_global = validate_global_evidence(conn, all_evidence_ids)
        if missing_global:
            errors.append(f"MISSING_EVIDENCE_IDS:{len(missing_global)}:{missing_global[:20]}")
        snapshot_bad = validate_snapshot_pairs(conn, witness_by_snapshot)
        if snapshot_bad:
            for item in snapshot_bad[:100]:
                errors.append(
                    f"WITNESS_OUTSIDE_SNAPSHOT:{item['snapshot_id']}:{item['missing'][:20]}"
                )
    finally:
        conn.close()

    integration = manifest.get("integration") or {}
    eligibility = integration.get("experiment_eligibility") or {}
    if eligibility:
        manifest_excluded = int(eligibility.get("excluded_task_count") or 0)
        actual_excluded = int(counters.get("experiment_excluded_tasks") or 0)
        if manifest_excluded != actual_excluded:
            errors.append(
                f"MANIFEST_EXPERIMENT_EXCLUDED_COUNT:{manifest_excluded}:{actual_excluded}"
            )
        by_split = eligibility.get("excluded_by_split") or {}
        for split in SPLITS:
            expected_ex = int(by_split.get(split) or 0)
            actual_ex = int(counters.get(f"experiment_excluded:{split}") or 0)
            if expected_ex != actual_ex:
                errors.append(
                    f"MANIFEST_EXPERIMENT_EXCLUDED_SPLIT:{split}:{expected_ex}:{actual_ex}"
                )

    if bool(integration.get("semantic_review_complete")):
        counters["semantic_review_complete_manifest"] += 1
    if bool(integration.get("training_ready")):
        errors.append("TRAINING_READY_TRUE_BEFORE_SEMANTIC_REVIEW")
    expected_rebuilt = int(integration.get("policy_rebuild_required_task_count") or 0)
    actual_rebuilt = counters["rebuild_required"]
    if expected_rebuilt != actual_rebuilt:
        errors.append(f"MANIFEST_REBUILD_COUNT:{expected_rebuilt}:{actual_rebuilt}")

    report = {
        "audit_version": SCRIPT_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "base_dir": str(base_dir),
        "derived_dir": str(derived_dir),
        "task_count": total,
        "counters": dict(counters),
        "referenced_evidence_id_count": len(all_evidence_ids),
        "snapshot_with_teacher_witness_count": len(witness_by_snapshot),
        "error_count": len(errors),
        "errors": errors,
        "semantic_review_complete": False,
        "training_ready": False,
        "experiment_excluded_task_count": int(counters.get("experiment_excluded_tasks") or 0),
        "derived_files": {
            f"{s}.parquet": {"sha256": sha256_file(resolve_split(derived_dir, s, base=False))}
            for s in SPLITS
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, report_path)

    # Derived manifest 可更新 mechanical audit 状态，但不改变 semantic/training gate。
    if not errors:
        manifest["audit_status"] = "mechanical_integrity_passed"
        manifest["mechanical_integrity_audit"] = {
            "report": report_path.name,
            "report_sha256": sha256_file(report_path),
            "audit_version": SCRIPT_VERSION,
            "error_count": 0,
        }
        manifest.setdefault("integration", {})["training_ready"] = False
        manifest["integration"]["semantic_review_complete"] = False
        mt = manifest_path.with_suffix(".json.tmp")
        mt.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(mt, manifest_path)

    print(json.dumps({
        "status": report["status"],
        "task_count": total,
        "rebuild_required": counters["rebuild_required"],
        "error_count": len(errors),
        "report": str(report_path),
        "training_ready": False,
    }, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[audit-integrated] interrupted", file=sys.stderr)
        raise SystemExit(130)
