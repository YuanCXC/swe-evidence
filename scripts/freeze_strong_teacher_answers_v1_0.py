#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Freeze mechanically-audited Strong-Teacher answers into an immutable sidecar snapshot.

Current project policy (2026-08-16):
- 20,501 mechanically usable answers are INCLUDED.
- 80 missing results are explicitly EXCLUDED.
- 7 unrecoverable invalid results are explicitly EXCLUDED.
- This is a MECHANICAL FREEZE, not the final semantic PASS/SOFT-PASS freeze.
- Frozen V2.10 is never modified.

The freeze canonicalizes only the same safe recoveries already allowed by audit v1.0.2:
- strict JSON salvage (only unambiguous wrapper/fence cases),
- provable task_id alias normalization,
- malformed additional_findings drop,
then re-validates schema and Candidate references.

For every included task it writes:
- canonical answer JSON,
- stable Candidate Number -> evidence_id binding,
- source/canonical SHA256 provenance.

The output directory is created through a staging directory and atomically renamed.
Existing freeze directories are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SPLITS = ("train", "validation", "benchmark")
DEFAULT_EXPECTED_QUESTIONS = 20588
DEFAULT_EXPECTED_RESULTS = 20508
DEFAULT_EXPECTED_INCLUDED = 20501
DEFAULT_EXPECTED_EXCLUDED = 87
ALLOWED_EXCLUDED_CODES = {"MISSING_RESULT_IGNORED"}
ALLOWED_HARD_CODES = {"ANSWER_VALIDATE_FAILED"}


@dataclass(frozen=True)
class TaskRow:
    split: str
    filename: str
    task_id: str
    status: str
    question_path: Path | None
    result_path: Path | None
    risk_score: int
    risk_flag_count: int


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def load_audit_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"audit script 不存在: {path}")
    spec = importlib.util.spec_from_file_location("strong_teacher_audit_v102", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 audit script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = [
        "parse_answer_with_safe_salvage",
        "safe_task_id_alias",
        "sanitize_additional_findings",
        "validate_answer_schema",
        "extract_question_task_id",
        "filename_task_ids",
        "extract_candidates",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"audit script 缺少所需接口: {missing}")
    return module


def resolve_path(raw: str, project_root: Path) -> Path | None:
    if not raw:
        return None
    # CSV may contain Windows backslashes even when parsed elsewhere.
    p = Path(raw.replace("\\", os.sep))
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def discover_question_path(input_root: Path, split: str, filename: str) -> Path:
    p = input_root / split / "md" / filename
    if p.exists():
        return p
    p2 = input_root / split / filename
    if p2.exists():
        return p2
    hits = list((input_root / split).rglob(filename)) if (input_root / split).exists() else []
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"无法唯一定位 question: split={split}, filename={filename}, hits={len(hits)}")


def discover_result_path(result_root: Path, split: str, filename: str) -> Path:
    p = result_root / split / filename
    if p.exists():
        return p
    hits = list((result_root / split).rglob(filename)) if (result_root / split).exists() else []
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"无法唯一定位 result: split={split}, filename={filename}, hits={len(hits)}")


def parse_header_value(header: str, key: str) -> str:
    m = re.search(rf"(?:^|\|)\s*{re.escape(key)}=([^|]+?)(?=\s*\||$)", header)
    return m.group(1).strip() if m else ""


def extract_candidate_bindings(text: str) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    seen_evidence: dict[str, int] = {}
    for m in re.finditer(r"(?m)^\[CANDIDATE\s+(\d+)\](?P<header>.*)$", text):
        n = int(m.group(1))
        if n in out:
            raise ValueError(f"Candidate Number 重复: {n}")
        header = m.group("header")
        eid = parse_header_value(header, "id")
        if not eid:
            raise ValueError(f"Candidate {n} 缺稳定 id=...，不能冻结 binding")
        if eid in seen_evidence:
            raise ValueError(f"evidence_id 重复绑定: {eid} -> {seen_evidence[eid]}, {n}")
        seen_evidence[eid] = n
        out[n] = {
            "evidence_id": eid,
            "path": parse_header_value(header, "path"),
            "symbol": parse_header_value(header, "symbol"),
        }
    if not out:
        raise ValueError("题目中未检测到 Candidate binding")
    return out


def canonicalize_answer(
    audit_mod,
    result_text: str,
    expected_task_id: str,
    filename: str,
    known_task_ids: set[str],
    legal_candidates: set[int],
) -> tuple[dict[str, Any], list[str]]:
    obj, parse_warnings = audit_mod.parse_answer_with_safe_salvage(result_text)
    notes = list(parse_warnings)

    answer_task_id = obj.get("task_id")
    if answer_task_id != expected_task_id:
        if audit_mod.safe_task_id_alias(answer_task_id, expected_task_id, filename, known_task_ids):
            obj["task_id"] = expected_task_id
            notes.append("TASK_ID_SAFE_NORMALIZED")
        else:
            raise ValueError(
                f"answer.task_id 无法安全归一化: expected={expected_task_id!r}, got={answer_task_id!r}"
            )

    dropped, details = audit_mod.sanitize_additional_findings(obj, legal_candidates)
    if dropped:
        notes.append(f"ADDITIONAL_FINDINGS_SANITIZED:{dropped}")
        notes.extend(f"ADDITIONAL_FINDINGS_DETAIL:{x}" for x in details)

    # Raises on any remaining mechanical invalidity.
    audit_mod.validate_answer_schema(obj, expected_task_id, legal_candidates)
    return obj, notes


def build_aggregate_hash(file_rows: list[dict[str, str]]) -> str:
    h = hashlib.sha256()
    for row in sorted(file_rows, key=lambda x: x["relative_path"]):
        line = f'{row["relative_path"]}\t{row["sha256"]}\n'.encode("utf-8")
        h.update(line)
    return h.hexdigest()


def choose_audit_script(project_root: Path, explicit: str | None) -> Path:
    if explicit:
        return resolve_path(explicit, project_root)  # type: ignore[return-value]
    candidates = [
        project_root / "scripts" / "audit_strong_teacher_results_v1_0_2.py",
        project_root / "scripts" / "audit_strong_teacher_results_v1_0.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("找不到 audit v1.0.2 脚本，请用 --audit-script 指定")


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze mechanically-audited Strong-Teacher answers.")
    ap.add_argument("--input-root", default=r"data\.external_supervision\strong_teacher_v1_3_all")
    ap.add_argument("--result-root", default=r"data\.external_supervision\result")
    ap.add_argument("--audit-root", default=r"data\.external_supervision\.audit\strong_teacher_audit")
    ap.add_argument(
        "--output-root",
        default=r"data\.external_supervision\frozen\strong_teacher_answers_mechanical_v1_0",
    )
    ap.add_argument("--audit-script", default=None)
    ap.add_argument("--expected-questions", type=int, default=DEFAULT_EXPECTED_QUESTIONS)
    ap.add_argument("--expected-results", type=int, default=DEFAULT_EXPECTED_RESULTS)
    ap.add_argument("--expected-included", type=int, default=DEFAULT_EXPECTED_INCLUDED)
    ap.add_argument("--expected-excluded", type=int, default=DEFAULT_EXPECTED_EXCLUDED)
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        # Tiny deterministic checks independent of project files.
        q = """[TASK]\ntask_abc123\n[CANDIDATE 1] | id=ev_a | path=a.py | symbol=f\n[CANDIDATE 2] | id=ev_b | path=b.py | symbol=g\n"""
        b = extract_candidate_bindings(q)
        assert b[1]["evidence_id"] == "ev_a" and b[2]["path"] == "b.py"
        rows = [
            {"relative_path": "b", "sha256": "2"},
            {"relative_path": "a", "sha256": "1"},
        ]
        assert build_aggregate_hash(rows) == build_aggregate_hash(list(reversed(rows)))
        print("SELF_TEST_OK")
        return 0

    project_root = Path.cwd().resolve()
    input_root = resolve_path(args.input_root, project_root)
    result_root = resolve_path(args.result_root, project_root)
    audit_root = resolve_path(args.audit_root, project_root)
    output_root = resolve_path(args.output_root, project_root)
    audit_script = choose_audit_script(project_root, args.audit_script)
    assert input_root and result_root and audit_root and output_root

    if output_root.exists():
        raise SystemExit(f"冻结目录已存在，拒绝覆盖: {output_root}")

    summary_path = audit_root / "audit_summary.json"
    issues_path = audit_root / "audit_issues.csv"
    status_path = audit_root / "per_answer_status.csv"
    for p in (summary_path, issues_path, status_path):
        if not p.exists():
            raise SystemExit(f"缺少审计文件: {p}")

    summary = read_json(summary_path)
    if str(summary.get("audit_version")) != "1.0.2":
        raise SystemExit(f"要求 audit_version=1.0.2，实际={summary.get('audit_version')!r}")
    if not summary.get("safe_recovery_policy", {}).get("allow_missing_results"):
        raise SystemExit("audit_summary 未启用 allow_missing_results，拒绝冻结")

    expected_summary = {
        "question_file_count": args.expected_questions,
        "result_file_count": args.expected_results,
        "per_answer_count": args.expected_results,
    }
    for key, expected in expected_summary.items():
        actual = int(summary.get(key, -1))
        if expected >= 0 and actual != expected:
            raise SystemExit(f"{key} 数量不符合冻结预期: expected={expected}, actual={actual}")

    issues = read_csv_rows(issues_path)
    statuses = read_csv_rows(status_path)

    excluded_map: dict[tuple[str, str], dict[str, Any]] = {}
    hard_codes: set[str] = set()
    excluded_codes: set[str] = set()
    for row in issues:
        sev = row.get("severity", "")
        code = row.get("code", "")
        key = (row.get("split", ""), row.get("filename", ""))
        if sev == "EXCLUDED":
            excluded_codes.add(code)
            e = excluded_map.setdefault(key, {
                "split": key[0], "filename": key[1], "task_id": row.get("task_id", ""),
                "reasons": [], "severity": "EXCLUDED",
            })
            e["reasons"].append(code)
        elif sev == "HARD_ERROR":
            hard_codes.add(code)
            e = excluded_map.setdefault(key, {
                "split": key[0], "filename": key[1], "task_id": row.get("task_id", ""),
                "reasons": [], "severity": "HARD_ERROR",
            })
            e["reasons"].append(code)
            e["severity"] = "HARD_ERROR"

    unknown_excluded = excluded_codes - ALLOWED_EXCLUDED_CODES
    unknown_hard = hard_codes - ALLOWED_HARD_CODES
    if unknown_excluded:
        raise SystemExit(f"存在未批准的 EXCLUDED code，拒绝冻结: {sorted(unknown_excluded)}")
    if unknown_hard:
        raise SystemExit(f"存在未批准的 HARD_ERROR code，拒绝冻结: {sorted(unknown_hard)}")

    # Build included rows from per-answer status. HARD_ERROR rows are excluded.
    included: list[TaskRow] = []
    status_keys: set[tuple[str, str]] = set()
    known_task_ids: set[str] = set()
    for row in statuses:
        split = row.get("split", "")
        filename = row.get("filename", "")
        key = (split, filename)
        if key in status_keys:
            raise SystemExit(f"per_answer_status 重复 key: {key}")
        status_keys.add(key)
        task_id = row.get("question_task_id", "")
        if task_id:
            known_task_ids.add(task_id)
        if row.get("status") == "HARD_ERROR":
            # Must be explicitly present in hard-error issue set.
            if key not in excluded_map:
                raise SystemExit(f"HARD_ERROR answer 未出现在 exclusion map: {key}")
            continue
        qpath = discover_question_path(input_root, split, filename)
        rpath = discover_result_path(result_root, split, filename)
        included.append(TaskRow(
            split=split,
            filename=filename,
            task_id=task_id,
            status=row.get("status", ""),
            question_path=qpath,
            result_path=rpath,
            risk_score=int(row.get("risk_score") or 0),
            risk_flag_count=int(row.get("risk_flag_count") or 0),
        ))

    included_keys = {(x.split, x.filename) for x in included}
    overlap = included_keys & set(excluded_map)
    if overlap:
        raise SystemExit(f"included/excluded 有重叠，拒绝冻结: {sorted(overlap)[:10]}")

    if len(included) != args.expected_included:
        raise SystemExit(f"included 数不符合预期: expected={args.expected_included}, actual={len(included)}")
    if len(excluded_map) != args.expected_excluded:
        raise SystemExit(f"excluded 数不符合预期: expected={args.expected_excluded}, actual={len(excluded_map)}")
    if len(included) + len(excluded_map) != args.expected_questions:
        raise SystemExit(
            "question coverage 不闭合: "
            f"included={len(included)} + excluded={len(excluded_map)} != questions={args.expected_questions}"
        )

    audit_mod = load_audit_module(audit_script)

    # Atomic staging under the same parent, so rename is atomic on the filesystem.
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output_root.name + ".staging.", dir=str(output_root.parent)))
    print(f"[freeze] staging={staging}")
    print(f"[freeze] included={len(included)}, excluded={len(excluded_map)}")

    included_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    file_hash_rows: list[dict[str, str]] = []
    normalization_counts: dict[str, int] = {}

    try:
        for idx, row in enumerate(included, start=1):
            assert row.question_path is not None and row.result_path is not None
            q_bytes = row.question_path.read_bytes()
            r_bytes = row.result_path.read_bytes()
            q_text = q_bytes.decode("utf-8-sig")
            r_text = r_bytes.decode("utf-8-sig")

            q_task_id = audit_mod.extract_question_task_id(q_text)
            if q_task_id != row.task_id:
                raise ValueError(
                    f"question task_id 与 audit status 不一致: {row.filename}: {q_task_id} != {row.task_id}"
                )
            legal_candidates, _ = audit_mod.extract_candidates(q_text)
            bindings = extract_candidate_bindings(q_text)
            if set(bindings) != set(legal_candidates):
                raise ValueError(f"Candidate binding 集合与 audit Candidate 集合不一致: {row.filename}")

            obj, notes = canonicalize_answer(
                audit_mod,
                r_text,
                expected_task_id=row.task_id,
                filename=row.filename,
                known_task_ids=known_task_ids,
                legal_candidates=legal_candidates,
            )
            for note in notes:
                code = note.split(":", 1)[0]
                normalization_counts[code] = normalization_counts.get(code, 0) + 1

            canonical_bytes = (
                json.dumps([obj], ensure_ascii=False, indent=2) + "\n"
            ).encode("utf-8")

            answer_rel = Path("answers") / row.split / row.filename
            binding_rel = Path("bindings") / row.split / (Path(row.filename).stem + ".json")
            meta_rel = Path("provenance") / row.split / (Path(row.filename).stem + ".json")

            answer_out = staging / answer_rel
            answer_out.parent.mkdir(parents=True, exist_ok=True)
            answer_out.write_bytes(canonical_bytes)

            binding_obj = {
                "task_id": row.task_id,
                "split": row.split,
                "filename": row.filename,
                "question_sha256": sha256_bytes(q_bytes),
                "candidate_count": len(bindings),
                "candidates": {str(n): bindings[n] for n in sorted(bindings)},
            }
            write_json(staging / binding_rel, binding_obj)

            meta_obj = {
                "task_id": row.task_id,
                "split": row.split,
                "filename": row.filename,
                "audit_status": row.status,
                "risk_score": row.risk_score,
                "risk_flag_count": row.risk_flag_count,
                "source_question_path": str(row.question_path),
                "source_result_path": str(row.result_path),
                "source_question_sha256": sha256_bytes(q_bytes),
                "source_result_sha256": sha256_bytes(r_bytes),
                "canonical_answer_sha256": sha256_bytes(canonical_bytes),
                "safe_normalizations": notes,
            }
            write_json(staging / meta_rel, meta_obj)

            for rel in (answer_rel, binding_rel, meta_rel):
                p = staging / rel
                file_hash_rows.append({"relative_path": rel.as_posix(), "sha256": sha256_file(p)})

            included_rows.append({
                "split": row.split,
                "filename": row.filename,
                "task_id": row.task_id,
                "audit_status": row.status,
                "risk_score": row.risk_score,
                "risk_flag_count": row.risk_flag_count,
                "answer_relative_path": answer_rel.as_posix(),
                "binding_relative_path": binding_rel.as_posix(),
                "source_result_sha256": sha256_bytes(r_bytes),
                "canonical_answer_sha256": sha256_bytes(canonical_bytes),
                "normalization_codes": ";".join(sorted({x.split(':', 1)[0] for x in notes})),
            })

            if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(included)):
                print(f"[freeze] canonicalized {idx:,}/{len(included):,}")

        for key, e in sorted(excluded_map.items()):
            split, filename = key
            result_exists = False
            source_result_sha = ""
            try:
                rpath = discover_result_path(result_root, split, filename)
                result_exists = rpath.exists()
                if result_exists:
                    source_result_sha = sha256_file(rpath)
            except FileNotFoundError:
                pass
            excluded_rows.append({
                "split": split,
                "filename": filename,
                "task_id": e.get("task_id", ""),
                "severity": e.get("severity", ""),
                "reasons": ";".join(sorted(set(e.get("reasons", [])))),
                "result_exists": str(result_exists).lower(),
                "source_result_sha256": source_result_sha,
            })

        # Freeze exact audit evidence used for the decision.
        audit_dst = staging / "audit"
        audit_dst.mkdir(parents=True, exist_ok=True)
        for name in (
            "audit_summary.json",
            "audit_issues.csv",
            "per_answer_status.csv",
            "semantic_review_queue.csv",
            "random_low_risk_sample.csv",
        ):
            src = audit_root / name
            if src.exists():
                dst = audit_dst / name
                shutil.copy2(src, dst)
                rel = dst.relative_to(staging)
                file_hash_rows.append({"relative_path": rel.as_posix(), "sha256": sha256_file(dst)})

        write_csv_rows(
            staging / "included_tasks.csv",
            included_rows,
            [
                "split", "filename", "task_id", "audit_status", "risk_score", "risk_flag_count",
                "answer_relative_path", "binding_relative_path", "source_result_sha256",
                "canonical_answer_sha256", "normalization_codes",
            ],
        )
        write_csv_rows(
            staging / "excluded_tasks.csv",
            excluded_rows,
            ["split", "filename", "task_id", "severity", "reasons", "result_exists", "source_result_sha256"],
        )
        file_hash_rows.append({
            "relative_path": "included_tasks.csv",
            "sha256": sha256_file(staging / "included_tasks.csv"),
        })
        file_hash_rows.append({
            "relative_path": "excluded_tasks.csv",
            "sha256": sha256_file(staging / "excluded_tasks.csv"),
        })

        aggregate_sha = build_aggregate_hash(file_hash_rows)
        manifest = {
            "freeze_version": "1.0",
            "freeze_type": "mechanical_strong_teacher_answer_freeze",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "input_root": str(input_root),
                "result_root": str(result_root),
                "audit_root": str(audit_root),
                "audit_script": str(audit_script),
                "audit_script_sha256": sha256_file(audit_script),
                "audit_version": summary.get("audit_version"),
            },
            "counts": {
                "question_file_count": args.expected_questions,
                "result_file_count": args.expected_results,
                "included_mechanically_usable": len(included_rows),
                "excluded_total": len(excluded_rows),
                "excluded_missing_result": sum("MISSING_RESULT_IGNORED" in x["reasons"] for x in excluded_rows),
                "excluded_unrecoverable_invalid": sum("ANSWER_VALIDATE_FAILED" in x["reasons"] for x in excluded_rows),
            },
            "exclusion_policy": {
                "allowed_excluded_codes": sorted(ALLOWED_EXCLUDED_CODES),
                "allowed_hard_codes": sorted(ALLOWED_HARD_CODES),
                "missing_results_are_not_backfilled": True,
                "unrecoverable_invalid_results_are_not_semantically_guessed": True,
            },
            "canonicalization_policy": {
                "strict_json_salvage": True,
                "task_id_alias_normalization": True,
                "malformed_additional_findings_drop": True,
                "source_files_modified": False,
                "normalization_counts": dict(sorted(normalization_counts.items())),
            },
            "semantic_status": {
                "final_semantic_freeze": False,
                "note": "This snapshot freezes mechanically usable Teacher answers. PASS/SOFT PASS/REVIEW/FAIL semantic labels must be applied as a later overlay without mutating this snapshot.",
            },
            "integrity": {
                "file_hash_entry_count": len(file_hash_rows),
                "aggregate_sha256": aggregate_sha,
            },
        }
        write_json(staging / "freeze_manifest.json", manifest)
        file_hash_rows.append({
            "relative_path": "freeze_manifest.json",
            "sha256": sha256_file(staging / "freeze_manifest.json"),
        })
        write_csv_rows(staging / "files.sha256.csv", file_hash_rows, ["relative_path", "sha256"])

        # Final staging consistency checks.
        if len(included_rows) != args.expected_included or len(excluded_rows) != args.expected_excluded:
            raise RuntimeError("冻结 staging 最终数量检查失败")
        if output_root.exists():
            raise RuntimeError(f"冻结目标在构建期间被创建，拒绝覆盖: {output_root}")

        staging.rename(output_root)
        print(json.dumps({
            "freeze_status": "OK",
            "output_root": str(output_root),
            "included": len(included_rows),
            "excluded": len(excluded_rows),
            "aggregate_sha256": aggregate_sha,
            "semantic_freeze": False,
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        # Keep the failed staging directory for forensic inspection, but make it obvious.
        failed = staging.with_name(staging.name + ".FAILED")
        try:
            if not failed.exists():
                staging.rename(failed)
                print(f"[freeze] 失败 staging 已保留: {failed}", file=sys.stderr)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
