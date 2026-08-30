#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repair Strong-Teacher result files that failed audit only because of JSON parsing.

Design goals
------------
1. Touch only rows selected from audit_issues.csv where:
   severity == HARD_ERROR
   code == ANSWER_VALIDATE_FAILED
   detail starts with JSONDecodeError
2. Never invent semantic content.
3. Recover only when exactly one unique schema-valid answer can be extracted from
   the existing file and bound to the current question/candidate pool.
4. Default is dry-run. --apply makes an atomic in-place replacement and writes a
   backup first.
5. Missing results are intentionally out of scope.

Typical use
-----------
python scripts/repair_strong_teacher_invalid_json_v1_0.py ^
  --input-root data/upstream/external_supervision/strong_teacher_v1_3_all ^
  --result-root data/upstream/external_supervision/result ^
  --issues-csv data/upstream/external_supervision/.audit/strong_teacher_audit/audit_issues.csv

Then inspect the report and apply:
python scripts/repair_strong_teacher_invalid_json_v1_0.py ... --apply
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

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
        "sufficient", "partial", "none", "uncertain", "not_applicable"
    },
    "repository_need": {
        "required", "helpful", "not_needed", "uncertain", "not_applicable"
    },
    "candidate_pool_status": {
        "sufficient", "insufficient", "uncertain", "not_needed"
    },
}


@dataclass
class RepairRow:
    split: str
    filename: str
    task_id: str
    source_path: str
    original_error: str
    status: str
    valid_candidate_count: int
    extraction_methods: str
    normalization_notes: str
    backup_path: str
    output_sha256: str
    detail: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Safely recover Strong-Teacher result files with JSONDecodeError."
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
        "--issues-csv",
        type=Path,
        default=Path(
            "data/upstream/external_supervision/.audit/strong_teacher_audit/audit_issues.csv"
        ),
    )
    p.add_argument(
        "--report-root",
        type=Path,
        default=Path(
            "data/upstream/external_supervision/.repair/strong_teacher_invalid_json_v1_0"
        ),
    )
    p.add_argument(
        "--backup-root",
        type=Path,
        default=Path(
            "data/upstream/external_supervision/.repair_backups/strong_teacher_invalid_json_v1_0"
        ),
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually replace recoverable result files after writing backups. Default: dry-run.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N target files. Default 10; 0 disables.",
    )
    return p.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def filename_task_ids(filename: str) -> list[str]:
    return re.findall(r"task_[A-Za-z0-9]+", Path(filename).stem)


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


def extract_candidates(text: str) -> set[int]:
    nums = {
        int(m.group(1))
        for m in re.finditer(r"(?m)^\[CANDIDATE\s+(\d+)\](?:.*)$", text)
    }
    if not nums:
        raise ValueError("题目中未检测到任何 [CANDIDATE N]")
    return nums


def find_unique_file(base: Path, split: str, filename: str, question: bool) -> Path:
    if question:
        preferred = base / split / "md" / filename
        fallback = base / split / filename
        if preferred.is_file():
            return preferred
        if fallback.is_file():
            return fallback
        roots = [base / split / "md", base / split]
    else:
        direct = base / split / filename
        if direct.is_file():
            return direct
        roots = [base / split]

    matches: list[Path] = []
    for root in roots:
        if root.exists():
            matches.extend(p for p in root.rglob(filename) if p.is_file())
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise FileNotFoundError(
            f"无法唯一定位 {'question' if question else 'result'}: "
            f"split={split}, filename={filename}, matches={unique}"
        )
    return unique[0]


def safe_task_id_alias(answer_task_id: Any, expected_task_id: str, filename: str) -> bool:
    if not isinstance(answer_task_id, str) or answer_task_id == expected_task_id:
        return False
    stem = Path(filename).stem
    ids = filename_task_ids(filename)
    if expected_task_id not in ids:
        return False
    if answer_task_id == stem:
        return True
    m = re.match(r"^(task_\d+)(?:_|$)", stem)
    return bool(m and answer_task_id == m.group(1))


def ensure_int_list(value: Any, where: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{where} 必须是 list")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in value):
        raise ValueError(f"{where} 必须只包含整数 Candidate Number")
    return value


def validate_candidate_refs(values: Iterable[int], legal: set[int], where: str) -> None:
    illegal = sorted(set(values) - legal)
    if illegal:
        raise ValueError(f"{where} 引用了不存在的 Candidate Number: {illegal}")


def sanitize_additional_findings(obj: dict[str, Any], legal: set[int]) -> list[str]:
    notes: list[str] = []
    findings = obj.get("additional_findings")
    if not isinstance(findings, list):
        obj["additional_findings"] = []
        return ["additional_findings 非 list，已保守置空"]

    kept: list[dict[str, Any]] = []
    for i, f in enumerate(findings):
        reason: str | None = None
        if not isinstance(f, dict):
            reason = "不是 object"
        else:
            missing = [k for k in ("description", "candidate_numbers", "reason") if k not in f]
            if missing:
                reason = f"缺字段 {missing}"
            elif not isinstance(f.get("description"), str):
                reason = "description 不是字符串"
            elif not isinstance(f.get("reason"), str):
                reason = "reason 不是字符串"
            else:
                try:
                    nums = ensure_int_list(
                        f.get("candidate_numbers"),
                        f"additional_findings[{i}].candidate_numbers",
                    )
                    validate_candidate_refs(nums, legal, f"additional_findings[{i}]")
                except Exception as exc:
                    reason = str(exc)
        if reason is None:
            kept.append(f)
        else:
            notes.append(f"drop additional_findings[{i}]: {reason}")
    if len(kept) != len(findings):
        obj["additional_findings"] = kept
    return notes


def validate_slot(name: str, slot: Any, legal: set[int]) -> None:
    if not isinstance(slot, dict):
        raise ValueError(f"slot {name} 不是 object")
    required = {
        "applicability", "question_coverage", "repository_need",
        "candidate_pool_status", "sufficient_witness_groups",
        "supporting_candidates", "reason",
    }
    missing = sorted(required - set(slot))
    if missing:
        raise ValueError(f"slot {name} 缺字段: {missing}")

    for field, allowed in ENUMS.items():
        if slot.get(field) not in allowed:
            raise ValueError(f"slot {name}.{field} 非法枚举={slot.get(field)!r}")

    if not isinstance(slot.get("reason"), str):
        raise ValueError(f"slot {name}.reason 必须是字符串")

    groups = slot["sufficient_witness_groups"]
    if not isinstance(groups, list):
        raise ValueError(f"slot {name}.sufficient_witness_groups 不是 list")

    canonical: list[tuple[int, ...]] = []
    for i, g in enumerate(groups):
        ints = ensure_int_list(g, f"{name}.groups[{i}]")
        if not ints:
            raise ValueError(f"{name}.groups[{i}] 是空 AND group")
        if len(ints) != len(set(ints)):
            raise ValueError(f"{name}.groups[{i}] 内 Candidate 重复")
        validate_candidate_refs(ints, legal, f"{name}.groups[{i}]")
        canonical.append(tuple(sorted(ints)))

    if len(canonical) != len(set(canonical)):
        raise ValueError(f"{name}: DUPLICATE_OR_GROUP")
    sets = [set(x) for x in canonical]
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            if i != j and b < a:
                raise ValueError(
                    f"{name}: NONMINIMAL_SUPERSET_GROUP:{canonical[i]}>{canonical[j]}"
                )

    supporting = ensure_int_list(slot["supporting_candidates"], f"{name}.supporting_candidates")
    validate_candidate_refs(supporting, legal, f"{name}.supporting_candidates")

    repo_need = slot["repository_need"]
    status = slot["candidate_pool_status"]
    if repo_need in {"not_needed", "not_applicable"} and groups:
        raise ValueError(f"{name}: repository_need={repo_need} 却存在 Witness")
    if repo_need == "required":
        if status == "sufficient" and not groups:
            raise ValueError(f"{name}: required+sufficient 但 Witness 为空")
        if status == "insufficient" and groups:
            raise ValueError(f"{name}: candidate_pool_status=insufficient 却存在 Witness")
    # helpful+Witness / uncertain+Witness / empty reason intentionally do not block here.


def validate_answer(obj: dict[str, Any], expected_task_id: str, legal: set[int]) -> None:
    if obj.get("task_id") != expected_task_id:
        raise ValueError(
            f"答案 task_id 不匹配 expected={expected_task_id}, got={obj.get('task_id')!r}"
        )
    if not isinstance(obj.get("overall_assessment"), str):
        raise ValueError("overall_assessment 不是字符串")
    slots = obj.get("slots")
    if not isinstance(slots, dict):
        raise ValueError("缺少 slots object")
    missing_slots = sorted(set(CANONICAL_SLOTS) - set(slots))
    extra_slots = sorted(set(slots) - set(CANONICAL_SLOTS))
    if missing_slots or extra_slots:
        raise ValueError(f"7 slots 不匹配 missing={missing_slots}, extra={extra_slots}")
    for name in CANONICAL_SLOTS:
        validate_slot(name, slots[name], legal)
    uncertainties = obj.get("uncertainties")
    if not isinstance(uncertainties, list) or any(not isinstance(x, str) for x in uncertainties):
        raise ValueError("uncertainties 不是字符串 list")


def object_from_value(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0], "array[1]"
    if isinstance(value, dict):
        return value, "single-object-wrapped"
    return None, None


def strip_simple_outer_fence(text: str) -> str:
    s = text.lstrip("\ufeff").strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if lines and lines[0].strip().lower() in {"```", "```json"}:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_embedded_values(text: str) -> list[tuple[Any, str]]:
    """Harvest JSON values without assuming the whole file is valid JSON."""
    decoder = json.JSONDecoder()
    variants: list[tuple[str, str]] = [(text.lstrip("\ufeff"), "raw")]
    stripped = strip_simple_outer_fence(text)
    if stripped != variants[0][0]:
        variants.append((stripped, "outer-fence-stripped"))

    harvested: list[tuple[Any, str]] = []
    seen_spans: set[tuple[str, int, int]] = set()

    for s, variant_name in variants:
        # Full document decode first.
        try:
            value = json.loads(s)
            harvested.append((value, f"{variant_name}:full"))
            if isinstance(value, str):
                try:
                    harvested.append((json.loads(value), f"{variant_name}:full-json-string"))
                except Exception:
                    pass
        except Exception:
            pass

        # Search for embedded JSON arrays/objects. This is safe only because every
        # harvested object must later pass task_id + candidate + 7-slot validation.
        for i, ch in enumerate(s):
            if ch not in "[{\"":
                continue
            try:
                value, end = decoder.raw_decode(s, i)
            except Exception:
                continue
            key = (variant_name, i, end)
            if key in seen_spans:
                continue
            seen_spans.add(key)
            harvested.append((value, f"{variant_name}:embedded@{i}:{end}"))
            if isinstance(value, str):
                inner = value.strip()
                if inner.startswith(("[", "{")):
                    try:
                        harvested.append((json.loads(inner), f"{variant_name}:embedded-json-string@{i}:{end}"))
                    except Exception:
                        pass
    return harvested


def normalize_and_validate_candidate(
    obj: dict[str, Any],
    expected_task_id: str,
    filename: str,
    legal: set[int],
) -> tuple[dict[str, Any], list[str]]:
    out = copy.deepcopy(obj)
    notes: list[str] = []

    if out.get("task_id") != expected_task_id:
        if safe_task_id_alias(out.get("task_id"), expected_task_id, filename):
            notes.append(f"task_id alias {out.get('task_id')!r} -> {expected_task_id!r}")
            out["task_id"] = expected_task_id
        else:
            raise ValueError(
                f"task_id 不匹配 expected={expected_task_id}, got={out.get('task_id')!r}"
            )

    notes.extend(sanitize_additional_findings(out, legal))
    validate_answer(out, expected_task_id, legal)
    return out, notes


def canonical_obj(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def recover_unique_answer(
    text: str,
    expected_task_id: str,
    filename: str,
    legal: set[int],
) -> tuple[dict[str, Any] | None, list[str], list[str], str]:
    valid: dict[str, tuple[dict[str, Any], set[str], set[str]]] = {}
    failures: list[str] = []

    for value, method in parse_embedded_values(text):
        obj, shape = object_from_value(value)
        if obj is None:
            continue
        try:
            normalized, notes = normalize_and_validate_candidate(
                obj, expected_task_id, filename, legal
            )
        except Exception as exc:
            if len(failures) < 12:
                failures.append(f"{method}: {type(exc).__name__}: {exc}")
            continue

        key = canonical_obj(normalized)
        if key not in valid:
            valid[key] = (normalized, set(), set())
        valid[key][1].add(f"{method}/{shape}")
        valid[key][2].update(notes)

    if len(valid) == 1:
        obj, methods, notes = next(iter(valid.values()))
        return obj, sorted(methods), sorted(notes), ""
    if len(valid) == 0:
        detail = "没有找到唯一 schema-valid JSON answer"
        if failures:
            detail += "；样例失败=" + " | ".join(failures[:5])
        return None, [], [], detail

    previews = []
    for _, (obj, methods, _) in list(valid.items())[:3]:
        previews.append(
            f"task_id={obj.get('task_id')}, methods={sorted(methods)}"
        )
    return None, [], [], f"找到 {len(valid)} 个不同的 schema-valid answers，拒绝猜测；" + " | ".join(previews)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
        raise


def load_targets(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    targets = [
        r for r in rows
        if r.get("severity") == "HARD_ERROR"
        and r.get("code") == "ANSWER_VALIDATE_FAILED"
        and (r.get("detail") or "").startswith("JSONDecodeError:")
    ]
    # Deduplicate by split+filename. Current dataset should yield exactly 77.
    uniq: dict[tuple[str, str], dict[str, str]] = {}
    for r in targets:
        uniq[(r.get("split", ""), r.get("filename", ""))] = r
    return [uniq[k] for k in sorted(uniq)]


def main() -> int:
    args = parse_args()
    targets = load_targets(args.issues_csv)
    args.report_root.mkdir(parents=True, exist_ok=True)

    print(f"[repair] targets={len(targets)} | mode={'APPLY' if args.apply else 'DRY_RUN'}")
    print(f"[repair] issues_csv={args.issues_csv}")

    rows: list[RepairRow] = []
    counts: Counter[str] = Counter()

    for idx, t in enumerate(targets, 1):
        split = t.get("split", "")
        filename = t.get("filename", "")
        csv_task_id = t.get("task_id", "")
        original_error = t.get("detail", "")

        source_path = ""
        backup_path = ""
        output_sha = ""
        try:
            q_path = find_unique_file(args.input_root, split, filename, question=True)
            r_path = find_unique_file(args.result_root, split, filename, question=False)
            source_path = str(r_path)

            q_text = q_path.read_text(encoding="utf-8")
            expected_task_id = extract_question_task_id(q_text)
            legal = extract_candidates(q_text)
            if csv_task_id and csv_task_id != expected_task_id:
                raise ValueError(
                    f"audit CSV task_id={csv_task_id} 与 question task_id={expected_task_id} 不一致"
                )

            raw = r_path.read_text(encoding="utf-8")
            obj, methods, notes, detail = recover_unique_answer(
                raw, expected_task_id, filename, legal
            )

            if obj is None:
                status = "UNRECOVERABLE"
                counts[status] += 1
                rows.append(
                    RepairRow(
                        split, filename, expected_task_id, source_path,
                        original_error, status, 0, "", "", "", "", detail,
                    )
                )
            else:
                payload = (json.dumps([obj], ensure_ascii=False, indent=2) + "\n").encode("utf-8")
                output_sha = sha256_bytes(payload)
                status = "RECOVERABLE"

                if args.apply:
                    backup = args.backup_root / split / filename
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    if backup.exists():
                        # Never overwrite an existing backup with possibly already-repaired content.
                        original_bytes = r_path.read_bytes()
                        if backup.read_bytes() != original_bytes:
                            raise FileExistsError(
                                f"backup 已存在且内容不同，拒绝覆盖: {backup}"
                            )
                    else:
                        shutil.copy2(r_path, backup)
                    backup_path = str(backup)
                    atomic_write(r_path, payload)
                    status = "REPAIRED"

                counts[status] += 1
                rows.append(
                    RepairRow(
                        split, filename, expected_task_id, source_path,
                        original_error, status, 1,
                        " | ".join(methods),
                        " | ".join(notes),
                        backup_path, output_sha,
                        "唯一 schema-valid answer 已提取；输出规范化为纯 JSON array。",
                    )
                )

        except Exception as exc:
            status = "ERROR"
            counts[status] += 1
            rows.append(
                RepairRow(
                    split, filename, csv_task_id, source_path,
                    original_error, status, 0, "", "", backup_path, output_sha,
                    f"{type(exc).__name__}: {exc}",
                )
            )

        if args.progress_every and idx % args.progress_every == 0:
            print(f"[repair] {idx}/{len(targets)} | {dict(counts)}")

    write_csv(
        args.report_root / "repair_report.csv",
        [asdict(r) for r in rows],
        [
            "split", "filename", "task_id", "source_path", "original_error",
            "status", "valid_candidate_count", "extraction_methods",
            "normalization_notes", "backup_path", "output_sha256", "detail",
        ],
    )

    unresolved = [r for r in rows if r.status in {"UNRECOVERABLE", "ERROR"}]
    write_csv(
        args.report_root / "unrecoverable_tasks.csv",
        [asdict(r) for r in unresolved],
        [
            "split", "filename", "task_id", "source_path", "original_error",
            "status", "valid_candidate_count", "extraction_methods",
            "normalization_notes", "backup_path", "output_sha256", "detail",
        ],
    )

    summary = {
        "repair_version": "1.0",
        "mode": "apply" if args.apply else "dry_run",
        "target_count": len(targets),
        "status_counts": dict(counts),
        "writes_source_files": bool(args.apply),
        "backup_root": str(args.backup_root) if args.apply else None,
        "report_root": str(args.report_root),
        "policy": {
            "semantic_guessing": False,
            "requires_unique_schema_valid_answer": True,
            "task_id_and_candidate_binding_required": True,
            "missing_results_out_of_scope": True,
        },
    }
    (args.report_root / "repair_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Dry-run: 0 if all targets recoverable; Apply: 0 if all repaired.
    bad = len(unresolved)
    return 2 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
