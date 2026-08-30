#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrate_strong_teacher_supervision_v1_0.py

把 Strong-Teacher 最终答案机械绑定到冻结 V2.10 task / Evidence ID，输出可审计的
Supervision Overlay Dataset（监督侧车数据集）。

设计边界：
1. 绝不修改 data/upstream/unified_swe_dataset_v2_10/。
2. Candidate Number 只作为 Teacher 接口；最终 Witness 一律解析为稳定 evidence_id。
3. OR-of-AND 原样保持：组内 AND、组间 OR，不拍平、不猜测语义。
4. supporting_candidates 只做 supporting_evidence_ids 记录，绝不自动提升为 sufficient witness。
5. 只有 repository_need=required 且 candidate_pool_status=sufficient 的 Witness
   才生成 repository obligation。
6. Issue 已满足 / repository 不需要的 required slot 只记录为 semantic slot，
   不制造空 Witness 的 repository obligation。
7. REVIEW / FAIL 不进入训练；PASS / SOFT PASS 可进入。
   若没有人工 review CSV，默认只生成 overlay，不自动 training_eligible；
   可显式 --accept-unreviewed-clean 放宽。
8. benchmark 永远 training_eligible=false。
9. 默认要求 Frozen V2.10 全 20,864 task 与 question/result 完整覆盖；
   --allow-incomplete 仅用于先生成诊断报告。

建议项目位置：
    scripts/integrate_strong_teacher_supervision_v1_0.py

默认输入：
    data/upstream/unified_swe_dataset_v2_10/
    data/upstream/external_supervision/strong_teacher_v1_3_all/
    data/upstream/external_supervision/result/
    data/upstream/external_supervision/.audit/strong_teacher_audit/audit_summary.json
    data/upstream/external_supervision/.audit/strong_teacher_audit/per_answer_status.csv

默认输出：
    data/upstream/external_supervision/integrated_strong_teacher_v1_0/
      train_strong_teacher_overlay.parquet
      validation_strong_teacher_overlay.parquet
      benchmark_strong_teacher_overlay.parquet
      integration_issues.csv
      integration_manifest.json

人工 review CSV（可选）格式：
    task_id,label
    task_xxx,PASS
    task_yyy,SOFT PASS
    task_zzz,REVIEW

也兼容列名：question_task_id + review_label。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


SCRIPT_VERSION = "1.0.0"
EXPECTED_DATASET_VERSION = "2.10.0"
EXPECTED_SPLIT_COUNTS = {
    "train": 18_347,
    "validation": 223,
    "benchmark": 2_294,
}
EXPECTED_TOTAL = sum(EXPECTED_SPLIT_COUNTS.values())
SPLITS = tuple(EXPECTED_SPLIT_COUNTS)

CANONICAL_SLOTS = (
    "fault_location",
    "fault_logic",
    "dependency_context",
    "state_flow",
    "behavior_constraint",
    "repair_scope",
    "validation_constraint",
)

ENUMS = {
    "applicability": {"required", "not_required", "uncertain"},
    "question_coverage": {
        "sufficient",
        "partial",
        "none",
        "uncertain",
        "not_applicable",
    },
    "repository_need": {
        "required",
        "helpful",
        "not_needed",
        "uncertain",
        "not_applicable",
    },
    "candidate_pool_status": {
        "sufficient",
        "insufficient",
        "uncertain",
        "not_needed",
    },
}

ACCEPTED_REVIEW_LABELS = {"PASS", "SOFT PASS", "SOFT_PASS"}
BLOCKED_REVIEW_LABELS = {"REVIEW", "FAIL"}


@dataclass(frozen=True)
class QuestionRecord:
    split: str
    task_id: str
    filename: str
    path: Path
    sha256: str
    candidate_map: dict[int, dict[str, str]]


@dataclass(frozen=True)
class IntegrationIssue:
    severity: str  # HARD_ERROR / BLOCKED / INFO
    code: str
    split: str
    task_id: str
    detail: str


# -----------------------------------------------------------------------------
# 基础工具
# -----------------------------------------------------------------------------

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_accidental_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def ensure_int_list(value: Any, where: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{where} 必须是 list")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in value):
        raise ValueError(f"{where} 必须只包含整数 Candidate Number")
    return list(value)


def canonicalize_candidate_group(group: Sequence[int]) -> list[int]:
    """只做顺序规范化，不改变 AND 语义。"""
    return sorted(map(int, group))


def deterministic_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def add_issue(
    issues: list[IntegrationIssue],
    severity: str,
    code: str,
    split: str,
    task_id: str,
    detail: str,
) -> None:
    issues.append(IntegrationIssue(severity, code, split, task_id, detail))


# -----------------------------------------------------------------------------
# V2.10 数据集读取
# -----------------------------------------------------------------------------

def resolve_manifest_path(dataset_dir: Path) -> Path:
    for path in (
        dataset_dir / "manifest_v2_10.json",
        dataset_dir / "manifest.json",
    ):
        if path.is_file():
            return path
    raise FileNotFoundError(f"找不到 V2.10 manifest：{dataset_dir}")


def load_and_validate_manifest(dataset_dir: Path) -> tuple[dict[str, Any], Path]:
    path = resolve_manifest_path(dataset_dir)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if str(manifest.get("dataset_version") or "") != EXPECTED_DATASET_VERSION:
        raise ValueError(
            "数据集版本不匹配："
            f"expected={EXPECTED_DATASET_VERSION}, actual={manifest.get('dataset_version')!r}"
        )
    if str(manifest.get("audit_status") or "") != "passed":
        raise ValueError(
            "冻结数据集 manifest.audit_status 必须为 passed；"
            f"actual={manifest.get('audit_status')!r}"
        )
    split_counts = manifest.get("split_counts")
    if split_counts is not None:
        normalized = {k: int(v) for k, v in split_counts.items() if k in SPLITS}
        if normalized != EXPECTED_SPLIT_COUNTS:
            raise ValueError(
                f"manifest split_counts 与 V2.10 冻结计数不一致：{normalized}"
            )
    return manifest, path


def resolve_split_parquet(dataset_dir: Path, split: str) -> Path:
    for path in (
        dataset_dir / f"{split}_v2_10.parquet",
        dataset_dir / f"{split}.parquet",
    ):
        if path.is_file():
            return path
    raise FileNotFoundError(f"找不到 split parquet：split={split}, dir={dataset_dir}")


def load_dataset_tasks(
    dataset_dir: Path,
    splits: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("缺少 pyarrow。请执行：python -m pip install -U pyarrow") from exc

    out: dict[str, dict[str, dict[str, Any]]] = {}
    seen_global: dict[str, str] = {}

    for split in splits:
        path = resolve_split_parquet(dataset_dir, split)
        table = pq.read_table(path, columns=["task_id", "supervision"])
        rows = table.to_pylist()
        if len(rows) != EXPECTED_SPLIT_COUNTS[split]:
            raise ValueError(
                f"V2.10 split 行数错误：split={split}, "
                f"expected={EXPECTED_SPLIT_COUNTS[split]}, actual={len(rows)}"
            )

        split_map: dict[str, dict[str, Any]] = {}
        for row in rows:
            task_id = str(row.get("task_id") or "")
            if not task_id:
                raise ValueError(f"{split} 中存在空 task_id")
            if task_id in split_map:
                raise ValueError(f"{split} 中 task_id 重复：{task_id}")
            previous = seen_global.get(task_id)
            if previous is not None:
                raise ValueError(
                    f"task_id 跨 split 重复：{task_id}, {previous} vs {split}"
                )
            seen_global[task_id] = split
            split_map[task_id] = {
                "task_id": task_id,
                "supervision": row.get("supervision") or {},
            }
        out[split] = split_map

    total = sum(len(v) for v in out.values())
    if set(splits) == set(SPLITS) and total != EXPECTED_TOTAL:
        raise ValueError(f"V2.10 task 总数错误：expected={EXPECTED_TOTAL}, actual={total}")
    return out


# -----------------------------------------------------------------------------
# Question / Candidate 解析
# -----------------------------------------------------------------------------

def extract_question_task_id(text: str) -> str:
    patterns = (
        r"(?m)^\[TASK\]\s*\n(?P<id>task_[A-Za-z0-9]+)\s*$",
        r"(?m)^TASK\s+\d+\s+[—-]\s+(?P<id>task_[A-Za-z0-9]+)\s*$",
        r"(?m)^当前 task_id 必须原样复制为：(?P<id>task_[A-Za-z0-9]+)\s*$",
    )
    found: list[str] = []
    for pattern in patterns:
        found.extend(m.group("id") for m in re.finditer(pattern, text))
    unique = sorted(set(found))
    if len(unique) != 1:
        raise ValueError(f"无法唯一确定题目 task_id：{unique}")
    return unique[0]


def parse_header_value(header: str, key: str) -> str:
    # Candidate header 约定为 id=... | path=... | symbol=... | ...
    m = re.search(rf"(?:^|\|)\s*{re.escape(key)}=([^|]+?)(?=\s*\||$)", header)
    return m.group(1).strip() if m else ""


def extract_candidate_map(text: str) -> dict[int, dict[str, str]]:
    candidate_map: dict[int, dict[str, str]] = {}
    evidence_to_number: dict[str, int] = {}

    for m in re.finditer(
        r"(?m)^\[CANDIDATE\s+(\d+)\](?P<header>.*)$",
        text,
    ):
        number = int(m.group(1))
        if number in candidate_map:
            raise ValueError(f"Candidate Number 重复：{number}")
        header = m.group("header")
        evidence_id = parse_header_value(header, "id")
        path = parse_header_value(header, "path")
        symbol = parse_header_value(header, "symbol")
        if not evidence_id:
            raise ValueError(
                f"Candidate {number} header 缺少稳定 id=...，无法绑定 evidence_id"
            )
        old = evidence_to_number.get(evidence_id)
        if old is not None:
            raise ValueError(
                f"同一 evidence_id 被多个 Candidate Number 引用："
                f"{evidence_id}, {old}, {number}"
            )
        evidence_to_number[evidence_id] = number
        candidate_map[number] = {
            "evidence_id": evidence_id,
            "path": path,
            "symbol": symbol,
        }

    if not candidate_map:
        raise ValueError("题目中未检测到任何 [CANDIDATE N]")
    return candidate_map


def discover_questions(
    input_root: Path,
    splits: Sequence[str],
) -> tuple[dict[str, dict[str, QuestionRecord]], list[IntegrationIssue]]:
    by_split: dict[str, dict[str, QuestionRecord]] = {s: {} for s in splits}
    issues: list[IntegrationIssue] = []
    global_owner: dict[str, str] = {}

    for split in splits:
        preferred = input_root / split / "md"
        base = preferred if preferred.exists() else input_root / split
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.md")):
            raw = path.read_bytes()
            try:
                text = raw.decode("utf-8")
                if not text.strip():
                    raise ValueError("question 文件为空")
                task_id = extract_question_task_id(text)
                candidate_map = extract_candidate_map(text)
                filename_ids = re.findall(r"task_[A-Za-z0-9]+", path.stem)
                if filename_ids and task_id not in filename_ids:
                    raise ValueError(
                        f"filename task_id={filename_ids} 与 question task_id={task_id} 不一致"
                    )
            except Exception as exc:
                add_issue(
                    issues, "HARD_ERROR", "QUESTION_PARSE_ERROR", split,
                    "<unknown>", f"{path}: {type(exc).__name__}: {exc}"
                )
                continue

            if task_id in by_split[split]:
                add_issue(
                    issues, "HARD_ERROR", "DUPLICATE_QUESTION_TASK", split,
                    task_id, f"重复 question：{by_split[split][task_id].path} ; {path}"
                )
                continue
            previous_split = global_owner.get(task_id)
            if previous_split is not None and previous_split != split:
                add_issue(
                    issues, "HARD_ERROR", "QUESTION_CROSS_SPLIT", split,
                    task_id, f"同一 task question 出现在 {previous_split} 与 {split}"
                )
                continue
            global_owner[task_id] = split
            by_split[split][task_id] = QuestionRecord(
                split=split,
                task_id=task_id,
                filename=path.name,
                path=path,
                sha256=sha256_bytes(raw),
                candidate_map=candidate_map,
            )

    return by_split, issues


# -----------------------------------------------------------------------------
# Answer Schema 校验
# -----------------------------------------------------------------------------

def parse_answer_file(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    if not raw:
        raise ValueError("answer 文件 0 byte")
    text = raw.decode("utf-8")
    data = json.loads(strip_accidental_fence(text))
    if not isinstance(data, list):
        raise ValueError("答案顶层不是 JSON array")
    if len(data) != 1:
        raise ValueError(f"答案顶层 array 必须恰好 1 个 object，实际={len(data)}")
    if not isinstance(data[0], dict):
        raise ValueError("答案 array 唯一元素不是 object")
    return data[0], sha256_bytes(raw)


def validate_candidate_refs(
    values: Iterable[int],
    legal: set[int],
    where: str,
) -> None:
    illegal = sorted(set(values) - legal)
    if illegal:
        raise ValueError(f"{where} 引用了不存在的 Candidate Number：{illegal}")


def validate_slot_schema(
    slot_name: str,
    slot: Any,
    legal_candidates: set[int],
) -> None:
    if not isinstance(slot, dict):
        raise ValueError(f"slot {slot_name} 不是 object")

    required_fields = {
        "applicability",
        "question_coverage",
        "repository_need",
        "candidate_pool_status",
        "sufficient_witness_groups",
        "supporting_candidates",
        "reason",
    }
    missing = sorted(required_fields - set(slot))
    if missing:
        raise ValueError(f"slot {slot_name} 缺字段：{missing}")

    for field, allowed in ENUMS.items():
        if slot.get(field) not in allowed:
            raise ValueError(
                f"slot {slot_name}.{field} 非法枚举={slot.get(field)!r}"
            )

    reason = slot.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"slot {slot_name}.reason 必须是非空字符串")

    groups = slot.get("sufficient_witness_groups")
    if not isinstance(groups, list):
        raise ValueError(f"slot {slot_name}.sufficient_witness_groups 必须是 list")

    canonical_groups: list[tuple[int, ...]] = []
    for i, group in enumerate(groups):
        values = ensure_int_list(group, f"{slot_name}.groups[{i}]")
        if not values:
            raise ValueError(f"{slot_name}.groups[{i}] 是空 AND group")
        if len(values) != len(set(values)):
            raise ValueError(f"{slot_name}.groups[{i}] 内 Candidate 重复")
        validate_candidate_refs(values, legal_candidates, f"{slot_name}.groups[{i}]")
        canonical_groups.append(tuple(sorted(values)))

    if len(canonical_groups) != len(set(canonical_groups)):
        raise ValueError(f"slot {slot_name} 存在重复 OR group")

    # Superset Elimination 是机械可证明的协议约束。
    sets = [set(x) for x in canonical_groups]
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            if i != j and b < a:
                raise ValueError(
                    f"slot {slot_name} 存在非最小 superset group："
                    f"{canonical_groups[i]} > {canonical_groups[j]}"
                )

    supporting = ensure_int_list(
        slot.get("supporting_candidates"),
        f"{slot_name}.supporting_candidates",
    )
    validate_candidate_refs(
        supporting, legal_candidates, f"{slot_name}.supporting_candidates"
    )

    repo_need = slot["repository_need"]
    pool_status = slot["candidate_pool_status"]

    if repo_need in {"helpful", "not_needed", "not_applicable"} and groups:
        raise ValueError(
            f"slot {slot_name}: repository_need={repo_need} 却存在 sufficient witness"
        )
    if repo_need == "required":
        if pool_status == "sufficient" and not groups:
            raise ValueError(
                f"slot {slot_name}: required+sufficient 但 Witness 为空"
            )
        if pool_status in {"insufficient", "uncertain"} and groups:
            raise ValueError(
                f"slot {slot_name}: pool_status={pool_status} 却存在 Witness"
            )


def validate_answer_schema(
    obj: dict[str, Any],
    expected_task_id: str,
    legal_candidates: set[int],
) -> None:
    if obj.get("task_id") != expected_task_id:
        raise ValueError(
            f"答案 task_id 不匹配：expected={expected_task_id}, got={obj.get('task_id')!r}"
        )
    if not isinstance(obj.get("overall_assessment"), str):
        raise ValueError("overall_assessment 不是字符串")

    slots = obj.get("slots")
    if not isinstance(slots, dict):
        raise ValueError("缺少 slots object")
    missing = sorted(set(CANONICAL_SLOTS) - set(slots))
    extra = sorted(set(slots) - set(CANONICAL_SLOTS))
    if missing or extra:
        raise ValueError(f"7 slots 不匹配：missing={missing}, extra={extra}")

    for slot_name in CANONICAL_SLOTS:
        validate_slot_schema(slot_name, slots[slot_name], legal_candidates)

    findings = obj.get("additional_findings")
    if not isinstance(findings, list):
        raise ValueError("additional_findings 不是 list")
    for i, finding in enumerate(findings):
        if not isinstance(finding, dict):
            raise ValueError(f"additional_findings[{i}] 不是 object")
        for field in ("description", "candidate_numbers", "reason"):
            if field not in finding:
                raise ValueError(f"additional_findings[{i}] 缺字段 {field}")
        nums = ensure_int_list(
            finding["candidate_numbers"],
            f"additional_findings[{i}].candidate_numbers",
        )
        validate_candidate_refs(nums, legal_candidates, f"additional_findings[{i}]")

    uncertainties = obj.get("uncertainties")
    if not isinstance(uncertainties, list) or any(
        not isinstance(x, str) for x in uncertainties
    ):
        raise ValueError("uncertainties 不是字符串 list")


# -----------------------------------------------------------------------------
# Audit / Human Review gate
# -----------------------------------------------------------------------------

def enforce_audit_gate(audit_summary: Path, skip: bool) -> dict[str, Any] | None:
    if skip:
        return None
    if not audit_summary.is_file():
        raise FileNotFoundError(
            "缺少 Strong-Teacher audit_summary.json。"
            "请先跑 audit_strong_teacher_results_v1_0.py，"
            "或仅诊断时显式使用 --skip-audit-gate。"
        )
    summary = json.loads(audit_summary.read_text(encoding="utf-8"))
    hard = int((summary.get("issue_severity_counts") or {}).get("HARD_ERROR", 0))
    if hard != 0:
        raise ValueError(
            f"Strong-Teacher mechanical audit 尚有 HARD_ERROR={hard}，禁止正式整合。"
        )
    return summary


def load_per_answer_hard_errors(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    out: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            task_id = str(
                row.get("question_task_id") or row.get("task_id") or ""
            ).strip()
            if not task_id:
                continue
            try:
                count = int(row.get("hard_error_count") or 0)
            except ValueError:
                count = 0
            out[task_id] = max(out.get(task_id, 0), count)
    return out


def normalize_review_label(value: str) -> str:
    label = re.sub(r"\s+", " ", value.strip().upper().replace("_", " "))
    if label == "SOFTPASS":
        label = "SOFT PASS"
    return label


def load_review_labels(path: Optional[Path]) -> dict[str, str]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"review labels CSV 不存在：{path}")
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            task_id = str(
                row.get("task_id") or row.get("question_task_id") or ""
            ).strip()
            raw_label = str(row.get("label") or row.get("review_label") or "").strip()
            if not task_id or not raw_label:
                continue
            label = normalize_review_label(raw_label)
            if label not in {"PASS", "SOFT PASS", "REVIEW", "FAIL"}:
                raise ValueError(
                    f"review label 非法：task={task_id}, label={raw_label!r}"
                )
            old = out.get(task_id)
            if old is not None and old != label:
                raise ValueError(
                    f"同一 task 有冲突 review label：{task_id}, {old} vs {label}"
                )
            out[task_id] = label
    return out


# -----------------------------------------------------------------------------
# Strong-Teacher -> stable Evidence ID overlay
# -----------------------------------------------------------------------------

def map_candidate_numbers(
    numbers: Sequence[int],
    candidate_map: Mapping[int, Mapping[str, str]],
) -> list[str]:
    out: list[str] = []
    for number in numbers:
        record = candidate_map.get(int(number))
        if record is None:
            raise ValueError(f"Candidate {number} 不存在")
        evidence_id = str(record.get("evidence_id") or "")
        if not evidence_id:
            raise ValueError(f"Candidate {number} 缺 evidence_id")
        out.append(evidence_id)
    return out


def original_obligations_by_type(
    supervision: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    obligations = supervision.get("obligations") or []
    if not isinstance(obligations, list):
        return out
    for obligation in obligations:
        if not isinstance(obligation, dict):
            continue
        slot_type = str(obligation.get("type") or "")
        if slot_type:
            out[slot_type].append(obligation)
    return out


def convert_answer_to_overlay(
    *,
    task_id: str,
    answer: Mapping[str, Any],
    question: QuestionRecord,
    original_supervision: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],  # semantic slots
    list[dict[str, Any]],  # repository obligations
    list[dict[str, Any]],  # additional findings mapped
    list[str],             # blockers
    set[str],              # all referenced stable evidence ids
]:
    source_by_type = original_obligations_by_type(original_supervision)
    semantic_slots: list[dict[str, Any]] = []
    repository_obligations: list[dict[str, Any]] = []
    mapped_findings: list[dict[str, Any]] = []
    blockers: list[str] = []
    referenced_evidence_ids: set[str] = set()

    slots = answer["slots"]

    for slot_name in CANONICAL_SLOTS:
        slot = dict(slots[slot_name])
        candidate_groups: list[list[int]] = [
            canonicalize_candidate_group(group)
            for group in slot["sufficient_witness_groups"]
        ]
        witness_groups: list[dict[str, Any]] = []

        for group_index, group in enumerate(candidate_groups, start=1):
            evidence_ids = map_candidate_numbers(group, question.candidate_map)
            referenced_evidence_ids.update(evidence_ids)
            witness_groups.append(
                {
                    "group_id": deterministic_id(
                        "stg",
                        task_id,
                        slot_name,
                        str(group_index),
                        *evidence_ids,
                    ),
                    "evidence_ids": evidence_ids,
                    "candidate_numbers": group,
                }
            )

        supporting_numbers = sorted(set(map(int, slot["supporting_candidates"])))
        supporting_evidence_ids = map_candidate_numbers(
            supporting_numbers, question.candidate_map
        )
        referenced_evidence_ids.update(supporting_evidence_ids)

        originals = source_by_type.get(slot_name, [])
        source_obligation_id = None
        source_description = ""
        if len(originals) == 1:
            source_obligation_id = str(originals[0].get("obligation_id") or "") or None
            source_description = str(originals[0].get("description") or "")

        applicability = slot["applicability"]
        question_coverage = slot["question_coverage"]
        repository_need = slot["repository_need"]
        pool_status = slot["candidate_pool_status"]

        evidence_requirement = "none"
        satisfied_by_question = False

        if applicability == "uncertain":
            blockers.append(f"{slot_name}:applicability_uncertain")
            evidence_requirement = "uncertain"
        elif applicability == "not_required":
            evidence_requirement = "not_required"
        else:
            # applicability == required
            if repository_need == "required":
                evidence_requirement = "repository_required"
                if pool_status == "sufficient" and witness_groups:
                    obligation_id = deterministic_id("sto", task_id, slot_name)
                    repository_obligations.append(
                        {
                            "obligation_id": obligation_id,
                            "source_obligation_id": source_obligation_id,
                            "type": slot_name,
                            "description": source_description or slot_name,
                            "applicable": True,
                            "mandatory": True,
                            "satisfied_by_question": False,
                            "witness_groups": [
                                {
                                    "group_id": group["group_id"],
                                    "evidence_ids": group["evidence_ids"],
                                    "source": "strong_teacher_v1_3",
                                    "rationale": str(slot["reason"]),
                                }
                                for group in witness_groups
                            ],
                        }
                    )
                else:
                    blockers.append(
                        f"{slot_name}:repository_required_but_{pool_status}"
                    )
            elif repository_need in {"not_needed", "not_applicable", "helpful"}:
                # 对 Evidence Policy 来说，不需要产生 repository obligation。
                # 但 required slot 若 Question 本身不 sufficient，则无法机械证明已满足。
                evidence_requirement = "question_or_nonrequired_repo"
                if question_coverage == "sufficient":
                    satisfied_by_question = True
                else:
                    blockers.append(
                        f"{slot_name}:required_without_repository_witness_"
                        f"and_question_coverage_{question_coverage}"
                    )
            elif repository_need == "uncertain":
                evidence_requirement = "uncertain"
                blockers.append(f"{slot_name}:repository_need_uncertain")
            else:
                blockers.append(f"{slot_name}:unhandled_repository_need_{repository_need}")

        semantic_slots.append(
            {
                "type": slot_name,
                "applicability": applicability,
                "question_coverage": question_coverage,
                "repository_need": repository_need,
                "candidate_pool_status": pool_status,
                "evidence_requirement": evidence_requirement,
                "satisfied_by_question": satisfied_by_question,
                "witness_groups": witness_groups,
                "supporting_evidence_ids": supporting_evidence_ids,
                "supporting_candidate_numbers": supporting_numbers,
                "reason": str(slot["reason"]),
                "source_obligation_id": source_obligation_id,
            }
        )

    for finding in answer.get("additional_findings") or []:
        nums = sorted(set(map(int, finding["candidate_numbers"])))
        evidence_ids = map_candidate_numbers(nums, question.candidate_map)
        referenced_evidence_ids.update(evidence_ids)
        mapped_findings.append(
            {
                "description": str(finding["description"]),
                "candidate_numbers": nums,
                "evidence_ids": evidence_ids,
                "reason": str(finding["reason"]),
            }
        )

    return (
        semantic_slots,
        repository_obligations,
        mapped_findings,
        sorted(set(blockers)),
        referenced_evidence_ids,
    )


# -----------------------------------------------------------------------------
# Evidence ID existence validation
# -----------------------------------------------------------------------------

def validate_evidence_ids_exist(
    build_db: Path,
    evidence_ids: set[str],
) -> set[str]:
    if not evidence_ids:
        return set()
    if not build_db.is_file():
        raise FileNotFoundError(f"缺少 working SQLite：{build_db}")

    connection = sqlite3.connect(f"file:{build_db.resolve()}?mode=ro", uri=True)
    try:
        found: set[str] = set()
        ids = sorted(evidence_ids)
        for offset in range(0, len(ids), 800):
            chunk = ids[offset : offset + 800]
            placeholders = ",".join("?" for _ in chunk)
            query = f"SELECT evidence_id FROM evidence_units WHERE evidence_id IN ({placeholders})"
            found.update(str(row[0]) for row in connection.execute(query, chunk))
        return set(ids) - found
    finally:
        connection.close()


# -----------------------------------------------------------------------------
# 输出
# -----------------------------------------------------------------------------

def overlay_arrow_schema() -> Any:
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError("缺少 pyarrow。请执行：python -m pip install -U pyarrow") from exc

    return pa.schema(
        [
            ("task_id", pa.string()),
            ("split", pa.string()),
            ("integration_status", pa.string()),
            ("question_file", pa.string()),
            ("result_file", pa.string()),
            ("review_label", pa.string()),
            ("accepted_semantic", pa.bool_()),
            ("supervision_complete", pa.bool_()),
            ("training_eligible", pa.bool_()),
            ("validation_eligible", pa.bool_()),
            ("benchmark_evaluation_eligible", pa.bool_()),
            ("overall_assessment", pa.string()),
            ("question_sha256", pa.string()),
            ("answer_sha256", pa.string()),
            ("candidate_count", pa.int32()),
            ("witness_evidence_count", pa.int32()),
            ("repository_obligation_count", pa.int16()),
            ("blockers_json", pa.string()),
            ("candidate_binding_json", pa.string()),
            ("semantic_slots_json", pa.string()),
            ("repository_obligations_json", pa.string()),
            ("additional_findings_json", pa.string()),
            ("uncertainties_json", pa.string()),
            ("raw_answer_json", pa.string()),
            ("source_supervision_obligations_json", pa.string()),
        ]
    )


def write_overlay_parquet(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("缺少 pyarrow。请执行：python -m pip install -U pyarrow") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=overlay_arrow_schema())
    pq.write_table(table, path, compression="zstd")
    return {
        "file": path.name,
        "row_count": len(rows),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_issues_csv(path: Path, issues: Sequence[IntegrationIssue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["severity", "code", "split", "task_id", "detail"],
        )
        writer.writeheader()
        for item in issues:
            writer.writerow(asdict(item))


# -----------------------------------------------------------------------------
# 主整合逻辑
# -----------------------------------------------------------------------------

def integrate(args: argparse.Namespace) -> int:
    dataset_dir = args.dataset_dir.resolve()
    input_root = args.input_root.resolve()
    result_root = args.result_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest, manifest_path = load_and_validate_manifest(dataset_dir)
    audit_summary_obj = enforce_audit_gate(args.audit_summary, args.skip_audit_gate)
    per_answer_hard = load_per_answer_hard_errors(args.per_answer_status)
    review_labels = load_review_labels(args.review_labels)

    dataset_tasks = load_dataset_tasks(dataset_dir, args.splits)
    questions_by_split, discovery_issues = discover_questions(input_root, args.splits)
    issues: list[IntegrationIssue] = list(discovery_issues)

    # 用于发现“question 放错 split”。
    question_global: dict[str, tuple[str, QuestionRecord]] = {}
    for split, mapping in questions_by_split.items():
        for task_id, question in mapping.items():
            question_global[task_id] = (split, question)

    rows_by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in args.splits}
    task_referenced_evidence: dict[str, set[str]] = defaultdict(set)

    for split in args.splits:
        for task_id, dataset_record in dataset_tasks[split].items():
            original_supervision = dataset_record["supervision"] or {}
            question = questions_by_split.get(split, {}).get(task_id)

            base_row: dict[str, Any] = {
                "task_id": task_id,
                "split": split,
                "integration_status": "",
                "question_file": "",
                "result_file": "",
                "review_label": review_labels.get(task_id, "UNREVIEWED"),
                "accepted_semantic": False,
                "supervision_complete": False,
                "training_eligible": False,
                "validation_eligible": False,
                "benchmark_evaluation_eligible": False,
                "overall_assessment": "",
                "question_sha256": "",
                "answer_sha256": "",
                "candidate_count": 0,
                "witness_evidence_count": 0,
                "repository_obligation_count": 0,
                "blockers_json": "[]",
                "candidate_binding_json": "{}",
                "semantic_slots_json": "[]",
                "repository_obligations_json": "[]",
                "additional_findings_json": "[]",
                "uncertainties_json": "[]",
                "raw_answer_json": "{}",
                "source_supervision_obligations_json": stable_json(
                    original_supervision.get("obligations") or []
                ),
            }

            if question is None:
                wrong = question_global.get(task_id)
                if wrong is not None:
                    code = "QUESTION_MISPLACED_SPLIT"
                    detail = f"question 实际位于 split={wrong[0]}"
                else:
                    code = "MISSING_QUESTION"
                    detail = "Frozen V2.10 task 在 Strong-Teacher question tree 中不存在"
                add_issue(issues, "HARD_ERROR", code, split, task_id, detail)
                base_row["integration_status"] = code
                base_row["blockers_json"] = stable_json([code])
                rows_by_split[split].append(base_row)
                continue

            base_row["question_file"] = str(question.path)
            base_row["question_sha256"] = question.sha256
            base_row["candidate_count"] = len(question.candidate_map)
            base_row["candidate_binding_json"] = stable_json(question.candidate_map)

            result_path = result_root / split / question.filename
            base_row["result_file"] = str(result_path)
            if not result_path.is_file():
                add_issue(
                    issues, "HARD_ERROR", "MISSING_RESULT", split, task_id,
                    f"缺少答案：{result_path}",
                )
                base_row["integration_status"] = "MISSING_RESULT"
                base_row["blockers_json"] = stable_json(["MISSING_RESULT"])
                rows_by_split[split].append(base_row)
                continue

            try:
                answer, answer_sha256 = parse_answer_file(result_path)
                validate_answer_schema(
                    answer,
                    expected_task_id=task_id,
                    legal_candidates=set(question.candidate_map),
                )
                if per_answer_hard.get(task_id, 0) > 0:
                    raise ValueError(
                        f"per_answer_status.csv hard_error_count={per_answer_hard[task_id]}"
                    )

                (
                    semantic_slots,
                    repository_obligations,
                    mapped_findings,
                    blockers,
                    referenced_ids,
                ) = convert_answer_to_overlay(
                    task_id=task_id,
                    answer=answer,
                    question=question,
                    original_supervision=original_supervision,
                )
            except Exception as exc:
                add_issue(
                    issues, "HARD_ERROR", "ANSWER_INTEGRATION_ERROR", split, task_id,
                    f"{type(exc).__name__}: {exc}",
                )
                base_row["integration_status"] = "ANSWER_INTEGRATION_ERROR"
                base_row["blockers_json"] = stable_json(
                    [f"ANSWER_INTEGRATION_ERROR:{type(exc).__name__}:{exc}"]
                )
                rows_by_split[split].append(base_row)
                continue

            task_referenced_evidence[task_id].update(referenced_ids)

            review_label = review_labels.get(task_id, "UNREVIEWED")
            if review_label in {"PASS", "SOFT PASS"}:
                accepted_semantic = True
            elif review_label in BLOCKED_REVIEW_LABELS:
                accepted_semantic = False
                blockers = sorted(set([*blockers, f"human_review_{review_label.lower()}"]))
            else:
                accepted_semantic = bool(args.accept_unreviewed_clean)
                if not accepted_semantic:
                    blockers = sorted(set([*blockers, "human_review_missing"]))

            supervision_complete = not any(
                b
                for b in blockers
                if not b.startswith("human_review_")
            )
            # Human review 是正式训练 gate；complete 表示结构/语义槽位足够构造 STOP 监督。
            eligible_core = accepted_semantic and supervision_complete

            base_row.update(
                {
                    "integration_status": "READY" if eligible_core else "BLOCKED",
                    "review_label": review_label,
                    "accepted_semantic": accepted_semantic,
                    "supervision_complete": supervision_complete,
                    "training_eligible": bool(eligible_core and split == "train"),
                    "validation_eligible": bool(eligible_core and split == "validation"),
                    "benchmark_evaluation_eligible": bool(
                        eligible_core and split == "benchmark"
                    ),
                    "overall_assessment": str(answer.get("overall_assessment") or ""),
                    "answer_sha256": answer_sha256,
                    "witness_evidence_count": len(referenced_ids),
                    "repository_obligation_count": len(repository_obligations),
                    "blockers_json": stable_json(blockers),
                    "semantic_slots_json": stable_json(semantic_slots),
                    "repository_obligations_json": stable_json(repository_obligations),
                    "additional_findings_json": stable_json(mapped_findings),
                    "uncertainties_json": stable_json(answer.get("uncertainties") or []),
                    "raw_answer_json": stable_json(answer),
                }
            )
            rows_by_split[split].append(base_row)

    # 全局 Evidence ID existence gate。
    all_referenced = set().union(*task_referenced_evidence.values()) if task_referenced_evidence else set()
    missing_evidence_ids = validate_evidence_ids_exist(args.build_db, all_referenced)

    if missing_evidence_ids:
        missing_set = set(missing_evidence_ids)
        for split, rows in rows_by_split.items():
            for row in rows:
                task_id = row["task_id"]
                bad = sorted(task_referenced_evidence.get(task_id, set()) & missing_set)
                if not bad:
                    continue
                blockers = json.loads(row["blockers_json"])
                blockers.append(f"missing_evidence_ids:{bad[:8]}")
                row["blockers_json"] = stable_json(sorted(set(blockers)))
                row["integration_status"] = "BLOCKED"
                row["supervision_complete"] = False
                row["training_eligible"] = False
                row["validation_eligible"] = False
                row["benchmark_evaluation_eligible"] = False
                add_issue(
                    issues, "HARD_ERROR", "EVIDENCE_ID_NOT_FOUND", split, task_id,
                    f"Witness/supporting Evidence ID 不在 V2.10 evidence_units：{bad[:20]}",
                )

    # 输出 sidecar parquet。
    file_reports: dict[str, Any] = {}
    for split in args.splits:
        name = f"{split}_strong_teacher_overlay.parquet"
        file_reports[name] = write_overlay_parquet(output_root / name, rows_by_split[split])

    issues_path = output_root / "integration_issues.csv"
    write_issues_csv(issues_path, issues)

    status_counts = Counter()
    review_counts = Counter()
    split_summary: dict[str, dict[str, Any]] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            status_counts[row["integration_status"]] += 1
            review_counts[row["review_label"]] += 1
        split_summary[split] = {
            "dataset_tasks": len(dataset_tasks[split]),
            "questions": len(questions_by_split.get(split, {})),
            "overlay_rows": len(rows),
            "ready": sum(r["integration_status"] == "READY" for r in rows),
            "blocked": sum(r["integration_status"] != "READY" for r in rows),
            "training_eligible": sum(bool(r["training_eligible"]) for r in rows),
            "validation_eligible": sum(bool(r["validation_eligible"]) for r in rows),
            "benchmark_evaluation_eligible": sum(
                bool(r["benchmark_evaluation_eligible"]) for r in rows
            ),
        }

    issue_counts = Counter((i.severity, i.code) for i in issues)
    hard_error_count = sum(1 for i in issues if i.severity == "HARD_ERROR")
    coverage_complete = all(
        split_summary[s]["questions"] == EXPECTED_SPLIT_COUNTS[s]
        and split_summary[s]["overlay_rows"] == EXPECTED_SPLIT_COUNTS[s]
        for s in args.splits
    )

    manifest_out = {
        "artifact": "strong_teacher_supervision_overlay",
        "script_version": SCRIPT_VERSION,
        "source_dataset_version": EXPECTED_DATASET_VERSION,
        "source_dataset_dir": str(dataset_dir),
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_question_root": str(input_root),
        "source_result_root": str(result_root),
        "source_build_db": str(args.build_db.resolve()),
        "audit_summary": str(args.audit_summary.resolve()),
        "audit_gate_skipped": bool(args.skip_audit_gate),
        "audit_summary_loaded": audit_summary_obj is not None,
        "review_labels": str(args.review_labels.resolve()) if args.review_labels else None,
        "accept_unreviewed_clean": bool(args.accept_unreviewed_clean),
        "split_summary": split_summary,
        "status_counts": dict(status_counts),
        "review_label_counts": dict(review_counts),
        "issue_counts": {
            f"{severity}:{code}": count
            for (severity, code), count in sorted(issue_counts.items())
        },
        "hard_error_count": hard_error_count,
        "coverage_complete": coverage_complete,
        "referenced_evidence_id_count": len(all_referenced),
        "missing_evidence_id_count": len(missing_evidence_ids),
        "files": file_reports,
        "integration_contract": {
            "v2_10_overwritten": False,
            "candidate_number_persisted_as_primary_identity": False,
            "stable_evidence_id_binding": True,
            "or_of_and_preserved": True,
            "supporting_candidates_promoted_to_witness": False,
            "benchmark_training_eligible": False,
            "policy_states_rebuilt": False,
            "note": (
                "本产物冻结 semantic slot + repository obligation overlay；"
                "正式训练前需基于 repository_obligations 重建派生 policy states/actions，"
                "不得继续使用与旧 obligations 对应的陈旧 policy labels。"
            ),
        },
    }

    manifest_out_path = output_root / "integration_manifest.json"
    manifest_out_path.write_text(
        json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest_out, ensure_ascii=False, indent=2))

    if hard_error_count and not args.allow_incomplete:
        return 2
    if not coverage_complete and not args.allow_incomplete:
        return 3
    if missing_evidence_ids and not args.allow_incomplete:
        return 4
    return 0


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

def self_test() -> int:
    question_text = """[TASK]\ntask_abc123\n\n[CANDIDATE 1] id=ev_a | path=pkg/a.py | symbol=foo | type=function\nbody\n\n[CANDIDATE 2] id=ev_b | path=pkg/b.py | symbol=bar | type=function\nbody\n"""
    task_id = extract_question_task_id(question_text)
    assert task_id == "task_abc123"
    cmap = extract_candidate_map(question_text)
    assert cmap[1]["evidence_id"] == "ev_a"
    assert cmap[2]["path"] == "pkg/b.py"

    answer = {
        "task_id": "task_abc123",
        "overall_assessment": "ok",
        "slots": {},
        "additional_findings": [],
        "uncertainties": [],
    }
    for slot in CANONICAL_SLOTS:
        answer["slots"][slot] = {
            "applicability": "not_required",
            "question_coverage": "not_applicable",
            "repository_need": "not_applicable",
            "candidate_pool_status": "not_needed",
            "sufficient_witness_groups": [],
            "supporting_candidates": [],
            "reason": "not needed",
        }
    answer["slots"]["fault_logic"] = {
        "applicability": "required",
        "question_coverage": "partial",
        "repository_need": "required",
        "candidate_pool_status": "sufficient",
        "sufficient_witness_groups": [[1, 2]],
        "supporting_candidates": [1],
        "reason": "A and B are jointly sufficient",
    }
    validate_answer_schema(answer, task_id, set(cmap))
    q = QuestionRecord(
        split="train",
        task_id=task_id,
        filename="task_abc123.md",
        path=Path("task_abc123.md"),
        sha256="x" * 64,
        candidate_map=cmap,
    )
    slots, obligations, findings, blockers, refs = convert_answer_to_overlay(
        task_id=task_id,
        answer=answer,
        question=q,
        original_supervision={"obligations": []},
    )
    assert len(slots) == 7
    assert len(obligations) == 1
    assert obligations[0]["witness_groups"][0]["evidence_ids"] == ["ev_a", "ev_b"]
    assert blockers == []
    assert refs == {"ev_a", "ev_b"}
    assert findings == []
    print("SELF_TEST_OK")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Bind Strong-Teacher Candidate Numbers to stable V2.10 evidence_ids and "
            "emit a frozen supervision overlay without mutating V2.10."
        )
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/upstream/unified_swe_dataset_v2_10"),
    )
    p.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/upstream/external_supervision/strong_teacher_v1_3_all"),
    )
    p.add_argument(
        "--result-root",
        type=Path,
        default=Path("data/upstream/external_supervision/result"),
    )
    p.add_argument(
        "--build-db",
        type=Path,
        default=Path("data/.build/unified_swe_v1.sqlite3"),
    )
    p.add_argument(
        "--audit-summary",
        type=Path,
        default=Path(
            "data/upstream/external_supervision/.audit/strong_teacher_audit/audit_summary.json"
        ),
    )
    p.add_argument(
        "--per-answer-status",
        type=Path,
        default=Path(
            "data/upstream/external_supervision/.audit/strong_teacher_audit/per_answer_status.csv"
        ),
    )
    p.add_argument(
        "--review-labels",
        type=Path,
        default=None,
        help="人工语义复核 CSV；PASS/SOFT PASS 可用，REVIEW/FAIL 阻断。",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/upstream/external_supervision/integrated_strong_teacher_v1_0"
        ),
    )
    p.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
    )
    p.add_argument(
        "--accept-unreviewed-clean",
        action="store_true",
        help=(
            "没有人工 PASS/SOFT PASS 标签时，也允许机械 clean 样本进入 eligible。"
            "默认关闭，避免把未人工审查样本直接送入训练。"
        ),
    )
    p.add_argument(
        "--skip-audit-gate",
        action="store_true",
        help="仅诊断时跳过 audit_summary HARD_ERROR=0 gate。",
    )
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "即使缺 question/result/evidence 也写报告并返回 0。"
            "仅用于诊断；正式冻结不要使用。"
        ),
    )
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    return integrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
