#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_integrated_training_dataset_v1_0.py

把冻结的 Unified SWE Dataset V2.10 与 Strong-Teacher mechanical freeze 合并为
新的派生数据集，最终输出放在普通 data/ 目录下，而不是 .external_supervision 下。

默认输入：
    data/upstream/unified_swe_dataset_v2_10/

Strong-Teacher freeze 自动按下列顺序寻找：
    1) --teacher-parquet 显式指定
    2) data/upstream/strong_teacher_mechanical_v1_0.parquet
    3) data/upstream/external_supervision/frozen/strong_teacher_mechanical_v1_0/
       strong_teacher_mechanical_v1_0.parquet

默认输出：
    data/upstream/unified_swe_dataset_v2_10_teacher_v1/
      train.parquet
      validation.parquet
      benchmark.parquet
      manifest.json

核心设计：
1. 永不修改 data/upstream/unified_swe_dataset_v2_10/。
2. 保留全部 20,864 个 V2.10 task；没有 Teacher 的 task 保留原 supervision。
3. Strong-Teacher Candidate Number 只作为接口；最终 Witness 绑定稳定 evidence_id。
4. OR-of-AND 保持：组内 AND、组间 OR，不拍平。
5. Teacher 对 canonical slot 的覆盖采用“保守覆盖”：
   - 明确 not_required：移除该 slot 的 repository obligation；
   - required + repository_need=required + pool=sufficient + 非空 witness：
     用 Teacher witness 替换该 slot 的原 obligation；
   - required + question 已 sufficient 且 repository 不 required：
     移除该 slot 的 repository obligation；
   - uncertain / insufficient / 其它不完备情况：不覆盖原 obligation。
6. 只要 effective obligations 发生变化，就清空该 task 的旧 policy_states，避免把
   与旧 obligation 对应的陈旧 Single/Pair/STOP 标签误当成新监督。
7. 因此本脚本完成的是“最终 supervision integration”，不是 policy rebuild。
   manifest 会明确 policy_rebuild_required=true；下一步应基于新 obligations 重建 policy。
8. 不复制 7GB+ repository_corpus；manifest 直接引用冻结 V2.10 corpus 文件。
9. 87 个 Teacher 排除样本和 276 个未进入 question tree 的任务都保留 V2.10 原监督。

建议项目位置：
    scripts/build_integrated_training_dataset_v1_0.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_VERSION = "1.0.0"
OUTPUT_DATASET_NAME = "unified_swe_dataset_v2_10_teacher_v1"
OUTPUT_DATASET_VERSION = "2.10.0+strong_teacher_mechanical_v1"
EXPECTED_BASE_VERSION = "2.10.0"
SPLITS = ("train", "validation", "benchmark")
EXPECTED_SPLIT_COUNTS = {
    "train": 18_347,
    "validation": 223,
    "benchmark": 2_294,
}
EXPECTED_TOTAL = sum(EXPECTED_SPLIT_COUNTS.values())
CANONICAL_SLOTS = (
    "fault_location",
    "fault_logic",
    "dependency_context",
    "state_flow",
    "behavior_constraint",
    "repair_scope",
    "validation_constraint",
)
CANONICAL_SLOT_SET = set(CANONICAL_SLOTS)


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "缺少 pyarrow。请在当前项目 Python 环境安装：python -m pip install -U pyarrow"
        ) from exc
    return pa, pq


def resolve_manifest_path(dataset_dir: Path) -> Path:
    for candidate in (
        dataset_dir / "manifest_v2_10.json",
        dataset_dir / "manifest.json",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"找不到 V2.10 manifest：{dataset_dir}")


def resolve_split_path(dataset_dir: Path, split: str) -> Path:
    for candidate in (
        dataset_dir / f"{split}_v2_10.parquet",
        dataset_dir / f"{split}.parquet",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"找不到 split parquet：split={split}, dir={dataset_dir}")


def resolve_corpus_path(dataset_dir: Path) -> Path:
    for candidate in (
        dataset_dir / "repository_corpus_v2_10.parquet",
        dataset_dir / "repository_corpus.parquet",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"找不到 repository corpus：{dataset_dir}")


def resolve_teacher_parquet(explicit: Path | None, project_root: Path) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            project_root / "data" / "strong_teacher_mechanical_v1_0.parquet",
            project_root
            / "data"
            / ".external_supervision"
            / "frozen"
            / "strong_teacher_mechanical_v1_0"
            / "strong_teacher_mechanical_v1_0.parquet",
        ]
    )
    for path in candidates:
        p = path if path.is_absolute() else (project_root / path)
        if p.is_file():
            return p.resolve()
    checked = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"找不到 Strong-Teacher frozen parquet。已检查：\n{checked}")


def load_base_manifest(dataset_dir: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = resolve_manifest_path(dataset_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = str(manifest.get("dataset_version") or manifest.get("version") or "")
    if version != EXPECTED_BASE_VERSION:
        raise ValueError(
            f"只接受冻结 V2.10：expected={EXPECTED_BASE_VERSION}, actual={version!r}"
        )
    if str(manifest.get("audit_status") or "") != "passed":
        raise ValueError(
            "基础数据集必须 audit_status=passed；"
            f"actual={manifest.get('audit_status')!r}"
        )
    return manifest_path, manifest


def parse_json_object(text: str, where: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where} JSON 解析失败：{exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{where} 必须是 JSON object")
    return obj


def parse_json_list(text: str, where: str) -> list[Any]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where} JSON 解析失败：{exc}") from exc
    if not isinstance(obj, list):
        raise ValueError(f"{where} 必须是 JSON array")
    return obj


def normalize_binding(binding_json: str, task_id: str) -> dict[int, dict[str, str]]:
    raw = parse_json_object(binding_json, f"{task_id}.candidate_binding_json")
    out: dict[int, dict[str, str]] = {}
    for key, value in raw.items():
        try:
            number = int(key)
        except Exception as exc:
            raise ValueError(f"{task_id}: 非法 Candidate Number key={key!r}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{task_id}: Candidate {number} binding 不是 object")
        evidence_id = str(value.get("evidence_id") or "").strip()
        if not evidence_id:
            raise ValueError(f"{task_id}: Candidate {number} 缺 evidence_id")
        out[number] = {
            "evidence_id": evidence_id,
            "path": str(value.get("path") or ""),
            "symbol": str(value.get("symbol") or ""),
        }
    return out


def map_candidate_group(
    group: Sequence[int],
    bindings: Mapping[int, Mapping[str, str]],
    *,
    task_id: str,
    slot_name: str,
) -> list[str]:
    if not group:
        raise ValueError(f"{task_id}.{slot_name}: 空 AND group")
    numbers = [int(x) for x in group]
    if len(numbers) != len(set(numbers)):
        raise ValueError(f"{task_id}.{slot_name}: AND group Candidate 重复={numbers}")
    evidence_ids: list[str] = []
    for number in sorted(numbers):
        record = bindings.get(number)
        if record is None:
            raise ValueError(
                f"{task_id}.{slot_name}: Candidate {number} 不在 frozen binding 中"
            )
        evidence_ids.append(str(record["evidence_id"]))
    return evidence_ids


def original_obligations_by_type(
    supervision: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obligation in supervision.get("obligations") or []:
        if isinstance(obligation, dict):
            typ = str(obligation.get("type") or "")
            if typ:
                out[typ].append(obligation)
    return out


def normalize_witness_groups_for_slot(
    *,
    task_id: str,
    slot_name: str,
    slot: Mapping[str, Any],
    bindings: Mapping[int, Mapping[str, str]],
    annotation_id: str,
    teacher_confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    """返回 (semantic_groups, obligation_groups, referenced_ids)。"""
    raw_groups = slot.get("sufficient_witness_groups") or []
    if not isinstance(raw_groups, list):
        raise ValueError(f"{task_id}.{slot_name}.sufficient_witness_groups 不是 list")

    semantic_groups: list[dict[str, Any]] = []
    obligation_groups: list[dict[str, Any]] = []
    referenced: set[str] = set()
    seen: set[tuple[str, ...]] = set()

    for index, raw_group in enumerate(raw_groups, 1):
        if not isinstance(raw_group, list):
            raise ValueError(f"{task_id}.{slot_name}.group[{index}] 不是 list")
        candidate_numbers = [int(x) for x in raw_group]
        evidence_ids = map_candidate_group(
            candidate_numbers,
            bindings,
            task_id=task_id,
            slot_name=slot_name,
        )
        key = tuple(evidence_ids)
        if key in seen:
            continue
        seen.add(key)
        referenced.update(evidence_ids)

        group_id = deterministic_id(
            "stwg",
            task_id,
            slot_name,
            *evidence_ids,
        )
        semantic_groups.append(
            {
                "group_id": group_id,
                "candidate_numbers": sorted(candidate_numbers),
                "evidence_ids": evidence_ids,
            }
        )
        obligation_groups.append(
            {
                "group_id": group_id,
                "evidence_ids": evidence_ids,
                "logic": "AND",
                "source": "strong_teacher_v1_3",
                "confidence": float(teacher_confidence),
                "annotation_ids": [annotation_id],
            }
        )

    return semantic_groups, obligation_groups, referenced


def classify_slot_override(slot: Mapping[str, Any], has_groups: bool) -> str:
    """
    返回：
      repository  -> 用 Teacher repository witness 覆盖原 slot obligation
      question    -> Question 已足够，移除原 repository obligation
      not_required-> slot 不需要，移除原 repository obligation
      unresolved  -> 不覆盖原 supervision
    """
    applicability = str(slot.get("applicability") or "")
    qcov = str(slot.get("question_coverage") or "")
    repo_need = str(slot.get("repository_need") or "")
    pool = str(slot.get("candidate_pool_status") or "")

    if applicability == "not_required":
        return "not_required"
    if applicability != "required":
        return "unresolved"

    if repo_need == "required":
        if pool == "sufficient" and has_groups:
            return "repository"
        return "unresolved"

    if repo_need in {"not_needed", "not_applicable", "helpful"}:
        if qcov == "sufficient":
            return "question"
        return "unresolved"

    return "unresolved"


def build_teacher_effective_supervision(
    *,
    task_id: str,
    base_supervision: Mapping[str, Any],
    answer: Mapping[str, Any],
    bindings: Mapping[int, Mapping[str, str]],
    canonical_answer_sha256: str,
    teacher_confidence: float,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[str],
    list[str],
    set[str],
    bool,
]:
    """
    返回：
      effective_supervision
      semantic_slots
      overridden_slots
      blockers
      referenced_evidence_ids
      obligations_changed
    """
    if str(answer.get("task_id") or "") != task_id:
        raise ValueError(
            f"Teacher answer task_id 错位：expected={task_id}, got={answer.get('task_id')!r}"
        )
    slots = answer.get("slots")
    if not isinstance(slots, dict):
        raise ValueError(f"{task_id}: answer.slots 不是 object")
    if set(slots) != CANONICAL_SLOT_SET:
        missing = sorted(CANONICAL_SLOT_SET - set(slots))
        extra = sorted(set(slots) - CANONICAL_SLOT_SET)
        raise ValueError(f"{task_id}: slots 不完整，missing={missing}, extra={extra}")

    effective = copy.deepcopy(dict(base_supervision))
    base_obligations = [
        copy.deepcopy(item)
        for item in (base_supervision.get("obligations") or [])
        if isinstance(item, dict)
    ]
    by_type = original_obligations_by_type(base_supervision)

    annotation_id = deterministic_id(
        "annotation",
        task_id,
        "strong_teacher_mechanical_v1_0",
        canonical_answer_sha256,
    )
    annotation = {
        "annotation_id": annotation_id,
        "source": "strong_teacher_mechanical_v1_0",
        "source_record_ids": [],
        "teacher_model": None,
        "prompt_version": "strong_teacher_v1_3",
        # 这里只通过机械审计，尚不把它伪装成完整语义 rule verification。
        "rule_verified": False,
        "input_sha256": canonical_answer_sha256,
    }

    semantic_slots: list[dict[str, Any]] = []
    overridden_slots: list[str] = []
    blockers: list[str] = []
    referenced_ids: set[str] = set()
    replacement_by_type: dict[str, list[dict[str, Any]]] = {}

    for slot_name in CANONICAL_SLOTS:
        slot = slots[slot_name]
        if not isinstance(slot, dict):
            raise ValueError(f"{task_id}.{slot_name}: slot 不是 object")

        semantic_groups, obligation_groups, refs = normalize_witness_groups_for_slot(
            task_id=task_id,
            slot_name=slot_name,
            slot=slot,
            bindings=bindings,
            annotation_id=annotation_id,
            teacher_confidence=teacher_confidence,
        )
        referenced_ids.update(refs)

        supporting_numbers = slot.get("supporting_candidates") or []
        if not isinstance(supporting_numbers, list):
            raise ValueError(f"{task_id}.{slot_name}.supporting_candidates 不是 list")
        supporting_evidence_ids: list[str] = []
        for number in sorted(set(int(x) for x in supporting_numbers)):
            record = bindings.get(number)
            if record is None:
                raise ValueError(f"{task_id}.{slot_name}: supporting Candidate {number} 不存在")
            evidence_id = str(record["evidence_id"])
            supporting_evidence_ids.append(evidence_id)
            referenced_ids.add(evidence_id)

        override = classify_slot_override(slot, bool(obligation_groups))
        source_obligations = by_type.get(slot_name, [])
        source_obligation_id = (
            str(source_obligations[0].get("obligation_id") or "")
            if len(source_obligations) == 1
            else ""
        )

        if override == "repository":
            obligation_id = source_obligation_id or deterministic_id(
                "obligation", task_id, slot_name, "strong_teacher_v1_3"
            )
            description = (
                str(source_obligations[0].get("description") or "")
                if source_obligations
                else ""
            ) or slot_name
            replacement_by_type[slot_name] = [
                {
                    "obligation_id": obligation_id,
                    "type": slot_name,
                    "description": description,
                    "applicable": True,
                    "mandatory": True,
                    "confidence": float(teacher_confidence),
                    "construction_method": "strong_teacher_mechanical_v1_0",
                    "witness_groups": obligation_groups,
                    "annotation_ids": [annotation_id],
                }
            ]
            overridden_slots.append(slot_name)
        elif override in {"question", "not_required"}:
            replacement_by_type[slot_name] = []
            overridden_slots.append(slot_name)
        else:
            if str(slot.get("applicability") or "") == "uncertain":
                blockers.append(f"{slot_name}:applicability_uncertain")
            elif str(slot.get("repository_need") or "") == "required":
                blockers.append(
                    f"{slot_name}:repository_required_but_"
                    f"{slot.get('candidate_pool_status') or 'unknown'}"
                )
            elif str(slot.get("repository_need") or "") == "uncertain":
                blockers.append(f"{slot_name}:repository_need_uncertain")
            elif (
                str(slot.get("applicability") or "") == "required"
                and str(slot.get("question_coverage") or "") != "sufficient"
            ):
                blockers.append(f"{slot_name}:teacher_slot_unresolved")

        semantic_slots.append(
            {
                "type": slot_name,
                "applicability": str(slot.get("applicability") or ""),
                "question_coverage": str(slot.get("question_coverage") or ""),
                "repository_need": str(slot.get("repository_need") or ""),
                "candidate_pool_status": str(slot.get("candidate_pool_status") or ""),
                "override_decision": override,
                "witness_groups": semantic_groups,
                "supporting_evidence_ids": supporting_evidence_ids,
                "reason": str(slot.get("reason") or ""),
                "source_obligation_ids": [
                    str(item.get("obligation_id") or "") for item in source_obligations
                ],
            }
        )

    new_obligations: list[dict[str, Any]] = []
    for obligation in base_obligations:
        typ = str(obligation.get("type") or "")
        if typ in replacement_by_type:
            continue
        new_obligations.append(obligation)
    for slot_name in CANONICAL_SLOTS:
        new_obligations.extend(replacement_by_type.get(slot_name, []))

    # 稳定顺序，便于可复现 hash / diff。
    new_obligations.sort(key=lambda x: (str(x.get("type") or ""), str(x.get("obligation_id") or "")))
    old_obligations_for_compare = copy.deepcopy(base_obligations)
    old_obligations_for_compare.sort(
        key=lambda x: (str(x.get("type") or ""), str(x.get("obligation_id") or ""))
    )
    obligations_changed = stable_json(new_obligations) != stable_json(old_obligations_for_compare)

    effective["obligations"] = new_obligations

    provenance = [
        copy.deepcopy(item)
        for item in (base_supervision.get("label_provenance") or [])
        if isinstance(item, dict)
    ]
    if overridden_slots:
        existing_ids = {str(item.get("annotation_id") or "") for item in provenance}
        if annotation_id not in existing_ids:
            provenance.append(annotation)
        provenance.sort(key=lambda x: str(x.get("annotation_id") or ""))
    effective["label_provenance"] = provenance

    if overridden_slots:
        effective["level"] = "strong"
        effective["recommended_weight"] = max(
            1.0,
            float(effective.get("recommended_weight") or 0.0),
        )

    training_targets = list(effective.get("training_targets") or [])
    if any(
        len(group.get("evidence_ids") or []) > 1
        for obligation in new_obligations
        for group in obligation.get("witness_groups") or []
    ):
        if "interaction_classification" not in training_targets:
            training_targets.append("interaction_classification")
    effective["training_targets"] = training_targets

    # 关键：只要 obligations 变了，旧 policy_states 就必须失效。
    if obligations_changed:
        effective["policy_states"] = []

    return (
        effective,
        semantic_slots,
        sorted(set(overridden_slots)),
        sorted(set(blockers)),
        referenced_ids,
        obligations_changed,
    )


def load_teacher_rows(teacher_parquet: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    pa, pq = require_pyarrow()
    del pa
    table = pq.read_table(teacher_parquet)
    required = {
        "task_id",
        "split",
        "included",
        "exclusion_reason",
        "audit_status",
        "risk_score",
        "risk_flags_json",
        "answer_json",
        "candidate_binding_json",
        "question_sha256",
        "source_result_sha256",
        "canonical_answer_sha256",
    }
    missing = sorted(required - set(table.schema.names))
    if missing:
        raise ValueError(f"Teacher freeze parquet 缺列：{missing}")

    by_task: dict[str, dict[str, Any]] = {}
    stats = Counter()
    split_stats: dict[str, Counter] = {s: Counter() for s in SPLITS}
    for row in table.to_pylist():
        task_id = str(row.get("task_id") or "")
        split = str(row.get("split") or "")
        if not task_id:
            raise ValueError("Teacher freeze 存在空 task_id")
        if split not in SPLITS:
            raise ValueError(f"Teacher freeze split 非法：{task_id} -> {split!r}")
        if task_id in by_task:
            raise ValueError(f"Teacher freeze task_id 重复：{task_id}")
        by_task[task_id] = row
        status = "included" if bool(row.get("included")) else "excluded"
        stats[status] += 1
        split_stats[split][status] += 1

    report = {
        "row_count": len(by_task),
        "included": stats["included"],
        "excluded": stats["excluded"],
        "split_counts": {s: dict(split_stats[s]) for s in SPLITS},
    }
    return by_task, report


def validate_evidence_ids_in_db(build_db: Path, evidence_ids: set[str]) -> set[str]:
    if not evidence_ids:
        return set()
    if not build_db.is_file():
        raise FileNotFoundError(f"Evidence ID 校验需要 working DB：{build_db}")
    conn = sqlite3.connect(f"file:{build_db.resolve()}?mode=ro", uri=True)
    try:
        found: set[str] = set()
        ids = sorted(evidence_ids)
        for offset in range(0, len(ids), 800):
            chunk = ids[offset : offset + 800]
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT evidence_id FROM evidence_units WHERE evidence_id IN ({placeholders})"
            found.update(str(row[0]) for row in conn.execute(query, chunk))
        return set(ids) - found
    finally:
        conn.close()


def extension_fields(pa):
    return [
        pa.field("strong_teacher_status", pa.string()),
        pa.field("strong_teacher_audit_status", pa.string()),
        pa.field("strong_teacher_risk_score", pa.int32()),
        pa.field("strong_teacher_risk_flags_json", pa.large_string()),
        pa.field("strong_teacher_exclusion_reason", pa.string()),
        pa.field("strong_teacher_overridden_slots_json", pa.large_string()),
        pa.field("strong_teacher_blockers_json", pa.large_string()),
        pa.field("strong_teacher_semantic_slots_json", pa.large_string()),
        pa.field("strong_teacher_question_sha256", pa.string()),
        pa.field("strong_teacher_answer_sha256", pa.string()),
        pa.field("strong_teacher_policy_rebuild_required", pa.bool_()),
    ]


def build_default_teacher_metadata(status: str) -> dict[str, Any]:
    return {
        "strong_teacher_status": status,
        "strong_teacher_audit_status": "",
        "strong_teacher_risk_score": 0,
        "strong_teacher_risk_flags_json": "[]",
        "strong_teacher_exclusion_reason": "",
        "strong_teacher_overridden_slots_json": "[]",
        "strong_teacher_blockers_json": "[]",
        "strong_teacher_semantic_slots_json": "[]",
        "strong_teacher_question_sha256": "",
        "strong_teacher_answer_sha256": "",
        "strong_teacher_policy_rebuild_required": False,
    }


def build_output_row(
    *,
    base_row: dict[str, Any],
    split: str,
    teacher_row: dict[str, Any] | None,
    teacher_confidence: float,
    all_referenced_ids: set[str],
) -> tuple[dict[str, Any], bool, str]:
    task_id = str(base_row.get("task_id") or "")
    if not task_id:
        raise ValueError(f"{split}: base row 存在空 task_id")

    row = copy.deepcopy(base_row)
    if teacher_row is None:
        row.update(build_default_teacher_metadata("NOT_IN_QUESTION_TREE"))
        return row, False, "NOT_IN_QUESTION_TREE"

    teacher_split = str(teacher_row.get("split") or "")
    if teacher_split != split:
        raise ValueError(
            f"Teacher split 错位：task={task_id}, base={split}, teacher={teacher_split}"
        )

    if not bool(teacher_row.get("included")):
        meta = build_default_teacher_metadata("EXCLUDED")
        meta.update(
            {
                "strong_teacher_audit_status": str(teacher_row.get("audit_status") or ""),
                "strong_teacher_risk_score": int(teacher_row.get("risk_score") or 0),
                "strong_teacher_risk_flags_json": str(teacher_row.get("risk_flags_json") or "[]"),
                "strong_teacher_exclusion_reason": str(teacher_row.get("exclusion_reason") or ""),
                "strong_teacher_question_sha256": str(teacher_row.get("question_sha256") or ""),
                "strong_teacher_answer_sha256": str(teacher_row.get("canonical_answer_sha256") or ""),
            }
        )
        row.update(meta)
        return row, False, "EXCLUDED"

    answer = parse_json_object(str(teacher_row.get("answer_json") or ""), f"{task_id}.answer_json")
    bindings = normalize_binding(str(teacher_row.get("candidate_binding_json") or ""), task_id)
    canonical_sha = str(teacher_row.get("canonical_answer_sha256") or "")
    if not canonical_sha:
        canonical_sha = sha256_bytes(stable_json(answer).encode("utf-8"))

    base_supervision = row.get("supervision") or {}
    if not isinstance(base_supervision, dict):
        raise ValueError(f"{task_id}: base supervision 不是 object")

    (
        effective_supervision,
        semantic_slots,
        overridden_slots,
        blockers,
        referenced_ids,
        obligations_changed,
    ) = build_teacher_effective_supervision(
        task_id=task_id,
        base_supervision=base_supervision,
        answer=answer,
        bindings=bindings,
        canonical_answer_sha256=canonical_sha,
        teacher_confidence=teacher_confidence,
    )
    all_referenced_ids.update(referenced_ids)
    row["supervision"] = effective_supervision
    row.update(
        {
            "strong_teacher_status": "INCLUDED",
            "strong_teacher_audit_status": str(teacher_row.get("audit_status") or ""),
            "strong_teacher_risk_score": int(teacher_row.get("risk_score") or 0),
            "strong_teacher_risk_flags_json": str(teacher_row.get("risk_flags_json") or "[]"),
            "strong_teacher_exclusion_reason": "",
            "strong_teacher_overridden_slots_json": stable_json(overridden_slots),
            "strong_teacher_blockers_json": stable_json(blockers),
            "strong_teacher_semantic_slots_json": stable_json(semantic_slots),
            "strong_teacher_question_sha256": str(teacher_row.get("question_sha256") or ""),
            "strong_teacher_answer_sha256": canonical_sha,
            "strong_teacher_policy_rebuild_required": bool(obligations_changed),
        }
    )
    return row, obligations_changed, "INCLUDED"


def write_split(
    *,
    source_path: Path,
    output_path: Path,
    split: str,
    teacher_rows: Mapping[str, dict[str, Any]],
    teacher_confidence: float,
    batch_size: int,
    progress_every: int,
    all_referenced_ids: set[str],
) -> dict[str, Any]:
    pa, pq = require_pyarrow()
    parquet = pq.ParquetFile(source_path)
    base_schema = parquet.schema_arrow
    if "task_id" not in base_schema.names or "supervision" not in base_schema.names:
        raise ValueError(f"基础 parquet 缺 task_id/supervision：{source_path}")

    duplicate_fields = [field.name for field in extension_fields(pa) if field.name in base_schema.names]
    if duplicate_fields:
        raise ValueError(f"基础数据已经存在 Strong-Teacher 扩展列：{duplicate_fields}")

    output_schema = pa.schema([*base_schema, *extension_fields(pa)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(output_path, output_schema, compression="zstd")

    count = 0
    status_counts = Counter()
    changed_count = 0
    seen_task_ids: set[str] = set()
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            rows = batch.to_pylist()
            output_rows: list[dict[str, Any]] = []
            for base_row in rows:
                task_id = str(base_row.get("task_id") or "")
                if task_id in seen_task_ids:
                    raise ValueError(f"{split}: task_id 重复：{task_id}")
                seen_task_ids.add(task_id)
                teacher_row = teacher_rows.get(task_id)
                out_row, changed, status = build_output_row(
                    base_row=base_row,
                    split=split,
                    teacher_row=teacher_row,
                    teacher_confidence=teacher_confidence,
                    all_referenced_ids=all_referenced_ids,
                )
                output_rows.append(out_row)
                count += 1
                status_counts[status] += 1
                changed_count += int(changed)

                if progress_every > 0 and count % progress_every == 0:
                    print(
                        f"[integrate] {split}: {count:,} rows | "
                        f"teacher={status_counts['INCLUDED']:,} | "
                        f"policy_rebuild={changed_count:,}",
                        flush=True,
                    )

            table = pa.Table.from_pylist(output_rows, schema=output_schema)
            writer.write_table(table)
    finally:
        writer.close()

    expected = EXPECTED_SPLIT_COUNTS[split]
    if count != expected:
        raise ValueError(f"{split} 行数错误：expected={expected}, actual={count}")

    return {
        "rows": count,
        "status_counts": dict(status_counts),
        "policy_rebuild_required_tasks": changed_count,
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
    }


def self_test() -> int:
    base = {
        "level": "support",
        "training_targets": ["evidence_ranking"],
        "recommended_weight": 0.5,
        "obligations": [
            {
                "obligation_id": "obl_old",
                "type": "fault_logic",
                "description": "old",
                "applicable": True,
                "mandatory": True,
                "confidence": 0.8,
                "construction_method": "deterministic_rule",
                "witness_groups": [
                    {
                        "group_id": "wg_old",
                        "evidence_ids": ["ev_old"],
                        "logic": "AND",
                        "source": "patch",
                        "confidence": 0.8,
                        "annotation_ids": [],
                    }
                ],
                "annotation_ids": [],
            }
        ],
        "policy_states": [{"state_id": "old"}],
        "label_provenance": [],
    }
    answer = {
        "task_id": "task_x",
        "overall_assessment": "ok",
        "slots": {},
        "additional_findings": [],
        "uncertainties": [],
    }
    for slot_name in CANONICAL_SLOTS:
        answer["slots"][slot_name] = {
            "applicability": "not_required",
            "question_coverage": "not_applicable",
            "repository_need": "not_applicable",
            "candidate_pool_status": "not_needed",
            "sufficient_witness_groups": [],
            "supporting_candidates": [],
            "reason": "",
        }
    answer["slots"]["fault_logic"] = {
        "applicability": "required",
        "question_coverage": "partial",
        "repository_need": "required",
        "candidate_pool_status": "sufficient",
        "sufficient_witness_groups": [[1, 2]],
        "supporting_candidates": [1],
        "reason": "joint",
    }
    bindings = {
        1: {"evidence_id": "ev_a", "path": "a.py", "symbol": "a"},
        2: {"evidence_id": "ev_b", "path": "b.py", "symbol": "b"},
    }
    effective, slots, overridden, blockers, refs, changed = build_teacher_effective_supervision(
        task_id="task_x",
        base_supervision=base,
        answer=answer,
        bindings=bindings,
        canonical_answer_sha256="a" * 64,
        teacher_confidence=0.9,
    )
    assert changed is True
    assert effective["policy_states"] == []
    assert set(refs) == {"ev_a", "ev_b"}
    fault = [x for x in effective["obligations"] if x["type"] == "fault_logic"]
    assert len(fault) == 1
    assert fault[0]["witness_groups"][0]["evidence_ids"] == ["ev_a", "ev_b"]
    assert "fault_logic" in overridden
    assert len(slots) == 7
    assert isinstance(blockers, list)
    print("SELF_TEST_OK")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Integrate frozen Strong-Teacher supervision into a derived V2.10 dataset."
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/upstream/unified_swe_dataset_v2_10"),
    )
    p.add_argument(
        "--teacher-parquet",
        type=Path,
        default=None,
        help="默认自动查找 data/upstream/strong_teacher_mechanical_v1_0.parquet 或 hidden freeze。",
    )
    p.add_argument(
        "--teacher-manifest",
        type=Path,
        default=None,
        help="可选：freeze_manifest.json；若存在会记录 hash。",
    )
    p.add_argument(
        "--build-db",
        type=Path,
        default=Path("data/.build/unified_swe_v1.sqlite3"),
        help="用于验证 Teacher 引用的 stable evidence_id 是否真实存在。",
    )
    p.add_argument(
        "--skip-evidence-id-db-check",
        action="store_true",
        help="不建议。仅在 working DB 不可用时跳过 Evidence ID existence gate。",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/upstream/unified_swe_dataset_v2_10_teacher_v1"),
    )
    p.add_argument("--teacher-confidence", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--progress-every", type=int, default=500)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if not (0.0 <= args.teacher_confidence <= 1.0):
        raise SystemExit("--teacher-confidence 必须在 [0,1]")
    if args.batch_size < 1:
        raise SystemExit("--batch-size 必须 >= 1")

    project_root = Path.cwd().resolve()
    dataset_dir = (project_root / args.dataset_dir).resolve() if not args.dataset_dir.is_absolute() else args.dataset_dir.resolve()
    output_dir = (project_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    build_db = (project_root / args.build_db).resolve() if not args.build_db.is_absolute() else args.build_db.resolve()
    teacher_parquet = resolve_teacher_parquet(args.teacher_parquet, project_root)

    manifest_path, base_manifest = load_base_manifest(dataset_dir)
    corpus_path = resolve_corpus_path(dataset_dir)
    teacher_rows, teacher_report = load_teacher_rows(teacher_parquet)

    # 预扫描基础 task_id -> split，只读 task_id 列，不展开大 supervision。
    _, pq = require_pyarrow()
    base_task_owner: dict[str, str] = {}
    for split in SPLITS:
        source = resolve_split_path(dataset_dir, split)
        ids = pq.read_table(source, columns=["task_id"])["task_id"].to_pylist()
        if len(ids) != EXPECTED_SPLIT_COUNTS[split]:
            raise SystemExit(
                f"基础 split 行数不符：{split}, expected={EXPECTED_SPLIT_COUNTS[split]}, actual={len(ids)}"
            )
        for raw in ids:
            task_id = str(raw or "")
            if not task_id:
                raise SystemExit(f"{split}: 空 task_id")
            if task_id in base_task_owner:
                raise SystemExit(
                    f"基础 task_id 跨 split/重复：{task_id}, "
                    f"{base_task_owner[task_id]} vs {split}"
                )
            base_task_owner[task_id] = split

    if len(base_task_owner) != EXPECTED_TOTAL:
        raise SystemExit(
            f"基础 task 总数错误：expected={EXPECTED_TOTAL}, actual={len(base_task_owner)}"
        )

    foreign_teacher = sorted(set(teacher_rows) - set(base_task_owner))
    if foreign_teacher:
        raise SystemExit(
            f"Teacher freeze 存在不属于 V2.10 的 task：count={len(foreign_teacher)}, first={foreign_teacher[:10]}"
        )
    for task_id, trow in teacher_rows.items():
        if str(trow.get("split") or "") != base_task_owner[task_id]:
            raise SystemExit(
                f"Teacher split 错位：{task_id}, "
                f"teacher={trow.get('split')}, base={base_task_owner[task_id]}"
            )

    if output_dir.exists() and not args.overwrite:
        raise SystemExit(
            f"输出目录已存在，拒绝覆盖：{output_dir}\n"
            "确认要重建时显式使用 --overwrite。"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{OUTPUT_DATASET_NAME}.staging.",
            dir=str(output_dir.parent),
        )
    )

    all_referenced_ids: set[str] = set()
    split_reports: dict[str, Any] = {}
    try:
        for split in SPLITS:
            source = resolve_split_path(dataset_dir, split)
            target = staging / f"{split}.parquet"
            split_reports[split] = write_split(
                source_path=source,
                output_path=target,
                split=split,
                teacher_rows=teacher_rows,
                teacher_confidence=float(args.teacher_confidence),
                batch_size=int(args.batch_size),
                progress_every=int(args.progress_every),
                all_referenced_ids=all_referenced_ids,
            )

        missing_evidence_ids: set[str] = set()
        evidence_check = "skipped"
        if not args.skip_evidence_id_db_check:
            missing_evidence_ids = validate_evidence_ids_in_db(build_db, all_referenced_ids)
            evidence_check = "passed" if not missing_evidence_ids else "failed"
            if missing_evidence_ids:
                raise RuntimeError(
                    "Teacher 引用了 V2.10 working DB 中不存在的 evidence_id："
                    f"count={len(missing_evidence_ids)}, first={sorted(missing_evidence_ids)[:20]}"
                )

        total_rows = sum(int(r["rows"]) for r in split_reports.values())
        total_rebuild = sum(
            int(r["policy_rebuild_required_tasks"]) for r in split_reports.values()
        )
        aggregate_status = Counter()
        for report in split_reports.values():
            aggregate_status.update(report["status_counts"])

        teacher_manifest_path: Path | None = None
        if args.teacher_manifest is not None:
            teacher_manifest_path = (
                args.teacher_manifest.resolve()
                if args.teacher_manifest.is_absolute()
                else (project_root / args.teacher_manifest).resolve()
            )
        else:
            auto = teacher_parquet.parent / "freeze_manifest.json"
            if auto.is_file():
                teacher_manifest_path = auto

        manifest = {
            "dataset_name": OUTPUT_DATASET_NAME,
            "dataset_version": OUTPUT_DATASET_VERSION,
            "schema_version": str(base_manifest.get("schema_version") or "1.0"),
            "script_version": SCRIPT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "audit_status": "passed",
            "base_dataset": {
                "path": str(dataset_dir),
                "dataset_version": EXPECTED_BASE_VERSION,
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "overwritten": False,
            },
            "strong_teacher": {
                "parquet": str(teacher_parquet),
                "parquet_sha256": sha256_file(teacher_parquet),
                "freeze_manifest": str(teacher_manifest_path) if teacher_manifest_path else None,
                "freeze_manifest_sha256": (
                    sha256_file(teacher_manifest_path) if teacher_manifest_path else None
                ),
                **teacher_report,
                "status_counts_in_final_dataset": dict(aggregate_status),
                "semantic_review_complete": False,
                "mechanical_freeze": True,
            },
            "task_counts": {
                "train": EXPECTED_SPLIT_COUNTS["train"],
                "validation": EXPECTED_SPLIT_COUNTS["validation"],
                "benchmark": EXPECTED_SPLIT_COUNTS["benchmark"],
                "total": total_rows,
            },
            "files": {
                f"{split}.parquet": {
                    "sha256": split_reports[split]["sha256"],
                    "bytes": split_reports[split]["bytes"],
                    "rows": split_reports[split]["rows"],
                }
                for split in SPLITS
            },
            "repository_corpus": {
                "path": str(corpus_path),
                "sha256": sha256_file(corpus_path),
                "copied_into_output": False,
            },
            "integration": {
                "canonical_slots": list(CANONICAL_SLOTS),
                "teacher_candidate_number_as_primary_identity": False,
                "stable_evidence_id_binding": True,
                "or_of_and_preserved": True,
                "supporting_candidates_promoted_to_witness": False,
                "conservative_slot_override": True,
                "teacher_confidence": float(args.teacher_confidence),
                "referenced_evidence_id_count": len(all_referenced_ids),
                "evidence_id_db_check": evidence_check,
                "missing_evidence_id_count": len(missing_evidence_ids),
                "policy_rebuild_required": total_rebuild > 0,
                "policy_rebuild_required_task_count": total_rebuild,
                "training_ready": total_rebuild == 0,
                "note": (
                    "Strong-Teacher 改变 obligation 的 task 已清空旧 policy_states。"
                    "在 policy rebuild 完成前，不得把这些 task 的旧 V2.10 policy labels 当成"
                    "Teacher-refined 训练监督。"
                ),
            },
        }

        manifest_path_out = staging / "manifest.json"
        manifest_path_out.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # 输出最终目录时只留下 3 个 split + 1 个 manifest，不复制 corpus。
        if output_dir.exists():
            if not args.overwrite:
                raise RuntimeError(f"输出目录突然出现：{output_dir}")
            backup = output_dir.with_name(output_dir.name + ".old")
            if backup.exists():
                shutil.rmtree(backup)
            output_dir.replace(backup)
            try:
                staging.replace(output_dir)
            except Exception:
                backup.replace(output_dir)
                raise
            shutil.rmtree(backup)
        else:
            staging.replace(output_dir)

        result = {
            "status": "OK",
            "output_dir": str(output_dir),
            "rows": total_rows,
            "teacher_rows": teacher_report["row_count"],
            "teacher_included": aggregate_status["INCLUDED"],
            "teacher_excluded": aggregate_status["EXCLUDED"],
            "teacher_not_in_question_tree": aggregate_status["NOT_IN_QUESTION_TREE"],
            "policy_rebuild_required_tasks": total_rebuild,
            "training_ready": total_rebuild == 0,
            "files_created": 4,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[integrate] interrupted", file=sys.stderr)
        raise SystemExit(130)
