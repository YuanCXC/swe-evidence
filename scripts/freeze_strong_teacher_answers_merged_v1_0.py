#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
freeze_strong_teacher_answers_merged_v1_0.py

将当前 Strong-Teacher 机械审计结果冻结成“单一 Parquet + 单一 Manifest”。
不再为每个 task 生成 answers/bindings/provenance 小文件。

当前冻结口径（默认值，可通过 CLI 覆盖）：
- Strong-Teacher question tree: 20,588
- result files: 20,508
- mechanically usable answers: 20,501
- explicitly excluded: 87
  - 80 MISSING_RESULT_IGNORED
  - 7 ANSWER_VALIDATE_FAILED (unrecoverable)

重要边界：
1. 不修改 input-root / result-root / V2.10。
2. 只复用 audit v1.0.2 已允许的机械无歧义恢复：
   - strict JSON salvage
   - provable task_id alias normalization
   - malformed additional_findings drop
3. Candidate Number 只作为 Teacher 接口；冻结表中同时保存 Candidate Number -> stable evidence_id binding。
4. OR-of-AND 不在本脚本中重新解释；answer_json 按已验证 canonical JSON 保存。
5. 这是 mechanical freeze，不等于最终 semantic PASS/SOFT PASS freeze。

最终输出目录只包含：
    strong_teacher_mechanical_v1_0.parquet
    freeze_manifest.json

建议位置：
    scripts/freeze_strong_teacher_answers_merged_v1_0.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SPLITS = ("train", "validation", "benchmark")
SPLIT_ORDER = {s: i for i, s in enumerate(SPLITS)}

SCRIPT_VERSION = "1.0.0"
FREEZE_VERSION = "mechanical_v1_0"

DEFAULT_EXPECTED_QUESTIONS = 20_588
DEFAULT_EXPECTED_RESULTS = 20_508
DEFAULT_EXPECTED_INCLUDED = 20_501
DEFAULT_EXPECTED_EXCLUDED = 87

ALLOWED_EXCLUDED_CODES = {"MISSING_RESULT_IGNORED"}
ALLOWED_HARD_CODES = {"ANSWER_VALIDATE_FAILED"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def resolve_path(raw: str | None, project_root: Path) -> Path | None:
    if raw is None:
        return None
    p = Path(raw.replace("\\", os.sep))
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def load_audit_module(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"audit script 不存在: {path}")
    spec = importlib.util.spec_from_file_location("strong_teacher_audit_v102_for_freeze", path)
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
        "extract_candidates",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"audit script 缺少接口: {missing}")
    return module


def choose_audit_script(project_root: Path, explicit: str | None) -> Path:
    if explicit:
        p = resolve_path(explicit, project_root)
        assert p is not None
        return p
    for p in (
        project_root / "scripts" / "audit_strong_teacher_results_v1_0_2.py",
        project_root / "scripts" / "audit_strong_teacher_results_v1_0.py",
    ):
        if p.exists():
            return p.resolve()
    raise FileNotFoundError("找不到 audit v1.0.2；请用 --audit-script 显式指定")


def discover_question_path(input_root: Path, split: str, filename: str) -> Path:
    candidates = [
        input_root / split / "md" / filename,
        input_root / split / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    root = input_root / split
    hits = list(root.rglob(filename)) if root.exists() else []
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(
        f"无法唯一定位 question: split={split}, filename={filename}, hits={len(hits)}"
    )


def discover_result_path(result_root: Path, split: str, filename: str) -> Path:
    p = result_root / split / filename
    if p.exists():
        return p
    root = result_root / split
    hits = list(root.rglob(filename)) if root.exists() else []
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(
        f"无法唯一定位 result: split={split}, filename={filename}, hits={len(hits)}"
    )


def parse_header_value(header: str, key: str) -> str:
    m = re.search(rf"(?:^|\|)\s*{re.escape(key)}=([^|]+?)(?=\s*\||$)", header)
    return m.group(1).strip() if m else ""


def extract_candidate_bindings(text: str) -> dict[int, dict[str, str]]:
    """从 Strong-Teacher question Markdown 中读取稳定 Candidate binding。"""
    out: dict[int, dict[str, str]] = {}
    seen_eids: dict[str, int] = {}

    for m in re.finditer(r"(?m)^\[CANDIDATE\s+(\d+)\](?P<header>.*)$", text):
        number = int(m.group(1))
        header = m.group("header")
        if number in out:
            raise ValueError(f"Candidate Number 重复: {number}")

        evidence_id = parse_header_value(header, "id")
        if not evidence_id:
            raise ValueError(f"Candidate {number} 缺少稳定 id=...，不能冻结")
        if evidence_id in seen_eids:
            raise ValueError(
                f"evidence_id 重复绑定: {evidence_id} -> {seen_eids[evidence_id]}, {number}"
            )
        seen_eids[evidence_id] = number
        out[number] = {
            "evidence_id": evidence_id,
            "path": parse_header_value(header, "path"),
            "symbol": parse_header_value(header, "symbol"),
        }

    if not out:
        raise ValueError("题目未解析到任何 [CANDIDATE N] binding")
    return out


def canonicalize_answer(
    audit_mod,
    result_text: str,
    expected_task_id: str,
    filename: str,
    known_task_ids: set[str],
    legal_candidates: set[int],
) -> tuple[dict[str, Any], list[str]]:
    obj, warnings = audit_mod.parse_answer_with_safe_salvage(result_text)
    notes = list(warnings)

    got_task_id = obj.get("task_id")
    if got_task_id != expected_task_id:
        if audit_mod.safe_task_id_alias(
            got_task_id, expected_task_id, filename, known_task_ids
        ):
            obj["task_id"] = expected_task_id
            notes.append("TASK_ID_SAFE_NORMALIZED")
        else:
            raise ValueError(
                "answer.task_id 无法安全归一化: "
                f"expected={expected_task_id!r}, got={got_task_id!r}"
            )

    dropped, details = audit_mod.sanitize_additional_findings(obj, legal_candidates)
    if dropped:
        notes.append(f"ADDITIONAL_FINDINGS_SANITIZED:{dropped}")
        notes.extend(f"ADDITIONAL_FINDINGS_DETAIL:{x}" for x in details)

    audit_mod.validate_answer_schema(obj, expected_task_id, legal_candidates)
    return obj, notes


def require_pyarrow():
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "缺少 pyarrow。请在当前环境安装：pip install pyarrow\n"
            "V2.10 本身也是 parquet，因此正式冻结建议使用同一 Python 环境。"
        ) from exc
    return pa, pq


def build_schema(pa):
    return pa.schema([
        ("task_id", pa.string()),
        ("split", pa.string()),
        ("filename", pa.string()),
        ("included", pa.bool_()),
        ("exclusion_reason", pa.string()),
        ("audit_status", pa.string()),
        ("risk_score", pa.int32()),
        ("risk_flag_count", pa.int32()),
        ("risk_flags_json", pa.large_string()),
        ("answer_json", pa.large_string()),
        ("candidate_binding_json", pa.large_string()),
        ("safe_normalizations_json", pa.large_string()),
        ("question_sha256", pa.string()),
        ("source_result_sha256", pa.string()),
        ("canonical_answer_sha256", pa.string()),
    ])


def self_test() -> int:
    question = (
        "[TASK]\n"
        "task_abc\n"
        "[CANDIDATE 1] | id=ev_a | path=a.py | symbol=f\n"
        "[CANDIDATE 2] | id=ev_b | path=b.py | symbol=g\n"
    )
    bindings = extract_candidate_bindings(question)
    assert bindings[1]["evidence_id"] == "ev_a"
    assert bindings[2]["path"] == "b.py"
    assert sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()
    print("SELF_TEST_OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Freeze Strong-Teacher answers into ONE parquet + ONE manifest."
    )
    ap.add_argument(
        "--input-root",
        default=r"data\.external_supervision\strong_teacher_v1_3_all",
    )
    ap.add_argument(
        "--result-root",
        default=r"data\.external_supervision\result",
    )
    ap.add_argument(
        "--audit-root",
        default=r"data\.external_supervision\.audit\strong_teacher_audit",
    )
    ap.add_argument(
        "--output-root",
        default=r"data\.external_supervision\frozen\strong_teacher_mechanical_v1_0",
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
        return self_test()

    pa, pq = require_pyarrow()

    project_root = Path.cwd().resolve()
    input_root = resolve_path(args.input_root, project_root)
    result_root = resolve_path(args.result_root, project_root)
    audit_root = resolve_path(args.audit_root, project_root)
    output_root = resolve_path(args.output_root, project_root)
    audit_script = choose_audit_script(project_root, args.audit_script)
    assert input_root and result_root and audit_root and output_root

    summary_path = audit_root / "audit_summary.json"
    issues_path = audit_root / "audit_issues.csv"
    status_path = audit_root / "per_answer_status.csv"
    for p in (summary_path, issues_path, status_path):
        if not p.exists():
            raise SystemExit(f"缺少审计文件: {p}")

    summary = read_json(summary_path)
    if str(summary.get("audit_version")) != "1.0.2":
        raise SystemExit(
            f"冻结要求 audit_version=1.0.2，实际={summary.get('audit_version')!r}"
        )
    if not summary.get("safe_recovery_policy", {}).get("allow_missing_results"):
        raise SystemExit("audit 没有启用 --allow-missing-results，拒绝冻结")

    expected_summary = {
        "question_file_count": args.expected_questions,
        "result_file_count": args.expected_results,
        "per_answer_count": args.expected_results,
    }
    for key, expected in expected_summary.items():
        actual = int(summary.get(key, -1))
        if expected >= 0 and actual != expected:
            raise SystemExit(
                f"{key} 不符合冻结预期: expected={expected}, actual={actual}"
            )

    issues = read_csv_rows(issues_path)
    statuses = read_csv_rows(status_path)

    # task-level risk flags / exclusions
    risk_flags: dict[tuple[str, str], list[str]] = defaultdict(list)
    excluded: dict[tuple[str, str], dict[str, Any]] = {}
    hard_codes: set[str] = set()
    excluded_codes: set[str] = set()

    for row in issues:
        key = (row.get("split", ""), row.get("filename", ""))
        sev = row.get("severity", "")
        code = row.get("code", "")
        if sev == "RISK_FLAG":
            risk_flags[key].append(code)
        elif sev == "EXCLUDED":
            excluded_codes.add(code)
            ent = excluded.setdefault(key, {
                "task_id": row.get("task_id", ""),
                "reasons": [],
                "severity": "EXCLUDED",
            })
            ent["reasons"].append(code)
        elif sev == "HARD_ERROR":
            hard_codes.add(code)
            ent = excluded.setdefault(key, {
                "task_id": row.get("task_id", ""),
                "reasons": [],
                "severity": "HARD_ERROR",
            })
            ent["reasons"].append(code)
            ent["severity"] = "HARD_ERROR"

    unknown_excluded = excluded_codes - ALLOWED_EXCLUDED_CODES
    unknown_hard = hard_codes - ALLOWED_HARD_CODES
    if unknown_excluded:
        raise SystemExit(
            f"发现未批准 EXCLUDED code，拒绝冻结: {sorted(unknown_excluded)}"
        )
    if unknown_hard:
        raise SystemExit(
            f"发现未批准 HARD_ERROR code，拒绝冻结: {sorted(unknown_hard)}"
        )

    # per_answer_status maps all existing results, including the 7 unrecoverable.
    status_map: dict[tuple[str, str], dict[str, str]] = {}
    known_task_ids: set[str] = set()
    for row in statuses:
        key = (row.get("split", ""), row.get("filename", ""))
        if key in status_map:
            raise SystemExit(f"per_answer_status 重复: {key}")
        status_map[key] = row
        tid = row.get("question_task_id", "")
        if tid:
            known_task_ids.add(tid)

    # Missing results are absent from per_answer_status, but their task_id is in audit_issues.
    for ent in excluded.values():
        tid = str(ent.get("task_id") or "")
        if tid:
            known_task_ids.add(tid)

    all_keys = set(status_map) | set(excluded)
    if len(all_keys) != args.expected_questions:
        raise SystemExit(
            "question coverage 不闭合: "
            f"unique task files={len(all_keys)}, expected={args.expected_questions}"
        )
    if len(excluded) != args.expected_excluded:
        raise SystemExit(
            f"excluded 数不符合预期: expected={args.expected_excluded}, actual={len(excluded)}"
        )

    included_count = sum(1 for key in status_map if key not in excluded)
    if included_count != args.expected_included:
        raise SystemExit(
            f"included 数不符合预期: expected={args.expected_included}, actual={included_count}"
        )

    audit_mod = load_audit_module(audit_script)

    rows: list[dict[str, Any]] = []
    normalization_counts: dict[str, int] = defaultdict(int)

    ordered_keys = sorted(all_keys, key=lambda x: (SPLIT_ORDER.get(x[0], 99), x[1]))
    print(
        f"[freeze] tasks={len(ordered_keys):,} | included={included_count:,} | "
        f"excluded={len(excluded):,}"
    )

    for idx, (split, filename) in enumerate(ordered_keys, start=1):
        status_row = status_map.get((split, filename))
        excluded_row = excluded.get((split, filename))

        if status_row:
            task_id = status_row.get("question_task_id", "")
            audit_status = status_row.get("status", "")
            risk_score = int(status_row.get("risk_score") or 0)
            risk_flag_count = int(status_row.get("risk_flag_count") or 0)
        else:
            task_id = str((excluded_row or {}).get("task_id") or "")
            audit_status = "EXCLUDED"
            risk_score = 0
            risk_flag_count = 0

        if not task_id:
            raise RuntimeError(f"task_id 为空: split={split}, filename={filename}")

        qpath = discover_question_path(input_root, split, filename)
        q_bytes = qpath.read_bytes()
        q_text = q_bytes.decode("utf-8-sig")
        q_task_id = audit_mod.extract_question_task_id(q_text)
        if q_task_id != task_id:
            raise RuntimeError(
                f"question task_id 不一致: {filename}: audit={task_id}, question={q_task_id}"
            )

        legal_candidates, _ = audit_mod.extract_candidates(q_text)
        bindings = extract_candidate_bindings(q_text)
        if set(bindings) != set(legal_candidates):
            raise RuntimeError(
                f"Candidate binding 与 audit Candidate 集合不一致: {split}/{filename}"
            )

        included = excluded_row is None
        exclusion_reason = ""
        answer_json = ""
        source_result_sha = ""
        canonical_answer_sha = ""
        normalization_notes: list[str] = []

        if included:
            rpath = discover_result_path(result_root, split, filename)
            r_bytes = rpath.read_bytes()
            r_text = r_bytes.decode("utf-8-sig")
            source_result_sha = sha256_bytes(r_bytes)

            obj, normalization_notes = canonicalize_answer(
                audit_mod,
                result_text=r_text,
                expected_task_id=task_id,
                filename=filename,
                known_task_ids=known_task_ids,
                legal_candidates=legal_candidates,
            )
            # Canonical storage is one answer object, not an unnecessary outer one-element array.
            answer_json = stable_json(obj)
            canonical_answer_sha = sha256_bytes(answer_json.encode("utf-8"))
            for note in normalization_notes:
                normalization_counts[note.split(":", 1)[0]] += 1
        else:
            exclusion_reason = ";".join(
                sorted(set((excluded_row or {}).get("reasons", [])))
            )
            # For the 7 invalid existing results, freeze source hash for provenance only.
            try:
                rpath = discover_result_path(result_root, split, filename)
                if rpath.exists():
                    source_result_sha = sha256_file(rpath)
            except FileNotFoundError:
                pass

        unique_risk_flags = sorted(set(risk_flags.get((split, filename), [])))
        rows.append({
            "task_id": task_id,
            "split": split,
            "filename": filename,
            "included": included,
            "exclusion_reason": exclusion_reason,
            "audit_status": audit_status,
            "risk_score": risk_score,
            "risk_flag_count": risk_flag_count,
            "risk_flags_json": stable_json(unique_risk_flags),
            "answer_json": answer_json,
            "candidate_binding_json": stable_json({
                str(n): bindings[n] for n in sorted(bindings)
            }),
            "safe_normalizations_json": stable_json(normalization_notes),
            "question_sha256": sha256_bytes(q_bytes),
            "source_result_sha256": source_result_sha,
            "canonical_answer_sha256": canonical_answer_sha,
        })

        if args.progress_every > 0 and (
            idx % args.progress_every == 0 or idx == len(ordered_keys)
        ):
            print(f"[freeze] processed {idx:,}/{len(ordered_keys):,}")

    # Final in-memory consistency checks before writing anything final.
    actual_included = sum(bool(r["included"]) for r in rows)
    actual_excluded = len(rows) - actual_included
    if len(rows) != args.expected_questions:
        raise RuntimeError("最终 row_count 与 expected questions 不一致")
    if actual_included != args.expected_included:
        raise RuntimeError(
            f"最终 included 不一致: {actual_included} != {args.expected_included}"
        )
    if actual_excluded != args.expected_excluded:
        raise RuntimeError(
            f"最终 excluded 不一致: {actual_excluded} != {args.expected_excluded}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    parquet_path = output_root / "strong_teacher_mechanical_v1_0.parquet"
    manifest_path = output_root / "freeze_manifest.json"
    if parquet_path.exists() or manifest_path.exists():
        raise SystemExit(
            "冻结目标已存在，拒绝覆盖。若要产生新冻结版本，请换 --output-root。\n"
            f"parquet={parquet_path}\nmanifest={manifest_path}"
        )

    schema = build_schema(pa)
    table = pa.Table.from_pylist(rows, schema=schema)

    # Same-directory temp files + os.replace => file-level atomic publish.
    fd, tmp_parquet_raw = tempfile.mkstemp(
        prefix=".strong_teacher_mechanical_v1_0.", suffix=".parquet.tmp", dir=str(output_root)
    )
    os.close(fd)
    tmp_parquet = Path(tmp_parquet_raw)
    tmp_manifest = output_root / ".freeze_manifest.json.tmp"

    try:
        pq.write_table(
            table,
            tmp_parquet,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        parquet_sha = sha256_file(tmp_parquet)

        manifest = {
            "freeze_version": FREEZE_VERSION,
            "script_version": SCRIPT_VERSION,
            "freeze_type": "mechanical_strong_teacher_merged_parquet",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "semantic_freeze": False,
            "source": {
                "input_root": str(input_root),
                "result_root": str(result_root),
                "audit_root": str(audit_root),
                "audit_script": str(audit_script),
                "audit_script_sha256": sha256_file(audit_script),
                "audit_summary_sha256": sha256_file(summary_path),
                "audit_issues_sha256": sha256_file(issues_path),
                "per_answer_status_sha256": sha256_file(status_path),
                "audit_version": summary.get("audit_version"),
            },
            "counts": {
                "question_count": len(rows),
                "source_result_count": args.expected_results,
                "included_mechanically_usable": actual_included,
                "excluded_total": actual_excluded,
                "excluded_missing_result": sum(
                    "MISSING_RESULT_IGNORED" in r["exclusion_reason"] for r in rows
                ),
                "excluded_unrecoverable_invalid": sum(
                    "ANSWER_VALIDATE_FAILED" in r["exclusion_reason"] for r in rows
                ),
            },
            "policy": {
                "source_files_modified": False,
                "strict_json_salvage": True,
                "provable_task_id_alias_normalization": True,
                "malformed_additional_findings_drop": True,
                "missing_results_backfilled": False,
                "unrecoverable_invalid_semantically_guessed": False,
                "allowed_excluded_codes": sorted(ALLOWED_EXCLUDED_CODES),
                "allowed_hard_codes": sorted(ALLOWED_HARD_CODES),
                "normalization_counts": dict(sorted(normalization_counts.items())),
            },
            "parquet": {
                "filename": parquet_path.name,
                "sha256": parquet_sha,
                "row_count": table.num_rows,
                "column_count": table.num_columns,
                "compression": "zstd",
                "schema": [
                    {"name": field.name, "type": str(field.type)} for field in schema
                ],
            },
            "next_stage": {
                "note": (
                    "人工语义审查标签应作为后续 overlay/derived dataset 使用，"
                    "不要修改本 mechanical freeze。"
                )
            },
        }
        write_json(tmp_manifest, manifest)

        os.replace(tmp_parquet, parquet_path)
        os.replace(tmp_manifest, manifest_path)

        print(json.dumps({
            "freeze_status": "OK",
            "output_root": str(output_root),
            "parquet": str(parquet_path),
            "manifest": str(manifest_path),
            "rows": len(rows),
            "included": actual_included,
            "excluded": actual_excluded,
            "parquet_sha256": parquet_sha,
            "files_created": 2,
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        for p in (tmp_parquet, tmp_manifest):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
