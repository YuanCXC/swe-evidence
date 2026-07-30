#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按需读取候选源码文件，抽取修复前 Evidence Units。

不读取配置文件，不扫描整个仓库，不创建 worktree。

输入：
    data/processed/evidence_anchors/candidate_files.jsonl
    data/registry/git_snapshots.jsonl

输出：
    data/processed/evidence_units/evidence_units.jsonl
    data/processed/evidence_units/file_records.jsonl
    data/processed/evidence_units/extraction_failures.jsonl
    data/processed/evidence_units/extraction_report.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

SCRIPT_VERSION = "1.0.0"
DEFAULT_WINDOW_RADIUS = 20
DEFAULT_MAX_FILE_BYTES = 2 * 1024 * 1024
DEFAULT_MAX_FILE_LINES = 50_000
DEFAULT_MAX_UNIT_LINES = 400
PYTHON_SUFFIXES = {".py", ".pyi"}
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico",
    ".pdf", ".zip", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".jar", ".war", ".class", ".pyc", ".pyo", ".exe", ".dll",
    ".so", ".dylib", ".a", ".o", ".woff", ".woff2", ".ttf", ".otf",
}

DECLARATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("class", re.compile(r"^\s*(?:public\s+|private\s+|protected\s+|internal\s+|abstract\s+|final\s+|sealed\s+|static\s+)*class\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
    ("interface", re.compile(r"^\s*(?:public\s+|private\s+|protected\s+|internal\s+)*interface\s+([A-Za-z_][A-Za-z0-9_]*)\b")),
    ("function", re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")),
    ("function", re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")),
    ("function", re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\(")),
    ("function", re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_!?=]*)\b")),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行 JSON 无效：{exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是 JSON object")
            records.append(record)
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def run_git_show(project_root: Path, git_dir: Path, commit: str, file_path: str) -> subprocess.CompletedProcess[bytes]:
    environment = os.environ.copy()
    environment.setdefault("GIT_TERMINAL_PROMPT", "0")
    return subprocess.run(
        ["git", "--git-dir", str(git_dir), "show", f"{commit}:{file_path}"],
        cwd=str(project_root), env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=300, check=False,
    )


def decode_text(data: bytes) -> tuple[str | None, str]:
    if b"\x00" in data:
        return None, "binary_nul_byte"
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "cp1252", "latin-1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return None, "decode_failed"


def language_from_path(file_path: str) -> str:
    suffix = PurePosixPath(file_path).suffix.lower()
    mapping = {
        ".py": "python", ".pyi": "python", ".js": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "typescript", ".java": "java", ".kt": "kotlin",
        ".go": "go", ".rs": "rust", ".c": "c", ".h": "c_or_cpp_header",
        ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp",
        ".cs": "csharp", ".rb": "ruby", ".php": "php", ".swift": "swift",
        ".scala": "scala", ".sh": "shell", ".sql": "sql", ".json": "json",
        ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".xml": "xml",
        ".md": "markdown", ".rst": "rst",
    }
    return mapping.get(suffix, "text")


def normalize_range(start: Any, end: Any, total_lines: int) -> tuple[int, int] | None:
    try:
        start_line = int(start)
    except (TypeError, ValueError):
        return None
    try:
        end_line = int(end) if end is not None else start_line
    except (TypeError, ValueError):
        end_line = start_line
    start_line = max(1, min(start_line, total_lines))
    end_line = max(start_line, min(end_line, total_lines))
    return start_line, end_line


def slice_lines(lines: list[str], start_line: int, end_line: int) -> str:
    return "".join(lines[start_line - 1:end_line])


def make_unit(candidate: dict[str, Any], *, unit_type: str, start_line: int, end_line: int,
              symbol: str | None, language: str, content: str,
              extraction_method: str, confidence: float) -> dict[str, Any]:
    identity = {
        "canonical_instance_id": candidate["canonical_instance_id"],
        "snapshot_id": candidate["snapshot_id"],
        "file_path": candidate["file_path"],
        "unit_type": unit_type,
        "start_line": start_line,
        "end_line": end_line,
        "symbol": symbol,
        "content_sha256": sha256_text(content),
    }
    return {
        "evidence_unit_id": "eu-" + sha256_text(stable_json(identity))[:24],
        "canonical_instance_id": candidate["canonical_instance_id"],
        "snapshot_id": candidate["snapshot_id"],
        "repo": candidate["repo"],
        "resolved_commit": candidate["resolved_commit"],
        "file_path": candidate["file_path"],
        "unit_type": unit_type,
        "start_line": start_line,
        "end_line": end_line,
        "symbol": symbol,
        "language": language,
        "content": content,
        "content_sha256": sha256_text(content),
        "line_count": end_line - start_line + 1,
        "anchor_ids": sorted(set(candidate.get("anchor_ids", []))),
        "evidence_roles": sorted(set(candidate.get("evidence_roles", []))),
        "provenances": sorted(set(candidate.get("provenances", []))),
        "extraction_method": extraction_method,
        "confidence": confidence,
        "code_origin": "pre_fix_base_commit",
        "visibility": "offline_supervision",
    }


def extract_line_windows(candidate: dict[str, Any], lines: list[str], radius: int, max_unit_lines: int) -> list[dict[str, Any]]:
    ranges: list[tuple[int, int]] = []
    for item in candidate.get("line_ranges", []):
        normalized = normalize_range(item.get("start_line"), item.get("end_line"), len(lines))
        if normalized is None:
            continue
        start_line, end_line = normalized
        window_start = max(1, start_line - radius)
        window_end = min(len(lines), end_line + radius)
        if window_end - window_start + 1 > max_unit_lines:
            window_end = window_start + max_unit_lines - 1
        ranges.append((window_start, window_end))
    if not ranges:
        ranges.append((1, min(len(lines), min(120, max_unit_lines))))

    ranges.sort()
    merged: list[list[int]] = []
    for start_line, end_line in ranges:
        if not merged or start_line > merged[-1][1] + 1:
            merged.append([start_line, end_line])
        else:
            merged[-1][1] = max(merged[-1][1], end_line)

    language = language_from_path(candidate["file_path"])
    return [
        make_unit(
            candidate,
            unit_type="line_window",
            start_line=start_line,
            end_line=end_line,
            symbol=None,
            language=language,
            content=slice_lines(lines, start_line, end_line),
            extraction_method="anchored_line_window",
            confidence=0.95,
        )
        for start_line, end_line in merged
    ]


def extract_python_units(candidate: dict[str, Any], text: str, lines: list[str], max_unit_lines: int) -> tuple[list[dict[str, Any]], str | None]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [], f"SyntaxError: {exc}"

    anchor_ranges = [
        normalized
        for item in candidate.get("line_ranges", [])
        if (normalized := normalize_range(item.get("start_line"), item.get("end_line"), len(lines))) is not None
    ]
    if not anchor_ranges:
        return [], None

    units: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start_line = int(getattr(node, "lineno", 1))
        end_line = int(getattr(node, "end_lineno", start_line) or start_line)
        overlaps = [r for r in anchor_ranges if not (r[1] < start_line or r[0] > end_line)]
        if not overlaps:
            continue

        unit_start, unit_end = start_line, end_line
        if unit_end - unit_start + 1 > max_unit_lines:
            anchor_start = min(item[0] for item in overlaps)
            anchor_end = max(item[1] for item in overlaps)
            extra = max(0, max_unit_lines - (anchor_end - anchor_start + 1))
            unit_start = max(start_line, anchor_start - extra // 2)
            unit_end = min(end_line, unit_start + max_unit_lines - 1)

        unit_type = "class" if isinstance(node, ast.ClassDef) else "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
        units.append(
            make_unit(
                candidate,
                unit_type=unit_type,
                start_line=unit_start,
                end_line=unit_end,
                symbol=getattr(node, "name", None),
                language="python",
                content=slice_lines(lines, unit_start, unit_end),
                extraction_method="python_ast",
                confidence=1.0,
            )
        )
    return units, None


def find_declaration(line: str) -> tuple[str, str] | None:
    for unit_type, pattern in DECLARATION_PATTERNS:
        match = pattern.search(line)
        if match:
            return unit_type, match.group(1)
    return None


def extract_heuristic_units(candidate: dict[str, Any], lines: list[str], radius: int) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for item in candidate.get("line_ranges", []):
        normalized = normalize_range(item.get("start_line"), item.get("end_line"), len(lines))
        if normalized is None:
            continue
        anchor_start, anchor_end = normalized
        declaration = None
        for line_number in range(anchor_start, max(0, anchor_start - 60), -1):
            found = find_declaration(lines[line_number - 1])
            if found:
                declaration = (line_number, found[0], found[1])
                break
        if declaration is None:
            continue
        declaration_line, unit_type, symbol = declaration
        if (declaration_line, symbol) in seen:
            continue
        seen.add((declaration_line, symbol))
        unit_start = max(1, declaration_line - 2)
        unit_end = min(len(lines), max(anchor_end, declaration_line) + radius)
        units.append(
            make_unit(
                candidate,
                unit_type=unit_type,
                start_line=unit_start,
                end_line=unit_end,
                symbol=symbol,
                language=language_from_path(candidate["file_path"]),
                content=slice_lines(lines, unit_start, unit_end),
                extraction_method="declaration_heuristic",
                confidence=0.65,
            )
        )
    return units


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按需读取候选源码并抽取修复前 Evidence Units。")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--window-radius", type=int, default=DEFAULT_WINDOW_RADIUS)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--max-file-lines", type=int, default=DEFAULT_MAX_FILE_LINES)
    parser.add_argument("--max-unit-lines", type=int, default=DEFAULT_MAX_UNIT_LINES)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    candidate_path = project_root / "data" / "processed" / "evidence_anchors" / "candidate_files.jsonl"
    snapshot_path = project_root / "data" / "registry" / "git_snapshots.jsonl"
    output_root = project_root / "data" / "processed" / "evidence_units"
    units_path = output_root / "evidence_units.jsonl"
    files_path = output_root / "file_records.jsonl"
    failures_path = output_root / "extraction_failures.jsonl"
    report_path = output_root / "extraction_report.json"

    if not candidate_path.exists():
        print(f"[错误] 候选文件清单不存在：{candidate_path}", file=sys.stderr)
        return 2
    if not snapshot_path.exists():
        print(f"[错误] Git 快照索引不存在：{snapshot_path}", file=sys.stderr)
        return 2
    if any(path.exists() for path in (units_path, files_path, failures_path, report_path)) and not args.overwrite:
        print("[错误] 输出已存在，重建时添加 --overwrite。", file=sys.stderr)
        return 2
    for option_name, value in {
        "--window-radius": args.window_radius,
        "--max-file-bytes": args.max_file_bytes,
        "--max-file-lines": args.max_file_lines,
        "--max-unit-lines": args.max_unit_lines,
    }.items():
        if value <= 0:
            print(f"[错误] {option_name} 必须大于 0。", file=sys.stderr)
            return 2

    candidates = read_jsonl(candidate_path)
    snapshots = {
        str(record.get("snapshot_id")): record
        for record in read_jsonl(snapshot_path)
        if record.get("status") == "passed"
    }
    candidates.sort(key=lambda item: (
        str(item.get("canonical_instance_id", "")),
        -float(item.get("priority_score", 0.0)),
        str(item.get("file_path", "")),
    ))
    if args.max_files is not None:
        if args.max_files <= 0:
            print("[错误] --max-files 必须大于 0。", file=sys.stderr)
            return 2
        candidates = candidates[:args.max_files]

    all_units: list[dict[str, Any]] = []
    file_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates, start=1):
        snapshot_id = str(candidate.get("snapshot_id", ""))
        snapshot = snapshots.get(snapshot_id)
        file_path = str(candidate.get("file_path", ""))
        instance_id = str(candidate.get("canonical_instance_id", ""))
        print(f"[{index}/{len(candidates)}] {instance_id}: {file_path}")

        if snapshot is None:
            failures.append({"canonical_instance_id": instance_id, "snapshot_id": snapshot_id, "file_path": file_path, "reason": "snapshot_not_found"})
            continue
        if PurePosixPath(file_path).suffix.lower() in BINARY_SUFFIXES:
            failures.append({"canonical_instance_id": instance_id, "snapshot_id": snapshot_id, "file_path": file_path, "reason": "binary_suffix"})
            continue

        git_dir = Path(str(snapshot["git_dir"]))
        if not git_dir.is_absolute():
            git_dir = project_root / git_dir
        commit = str(snapshot["resolved_commit"])
        result = run_git_show(project_root, git_dir, commit, file_path)
        if result.returncode != 0:
            failures.append({
                "canonical_instance_id": instance_id,
                "snapshot_id": snapshot_id,
                "file_path": file_path,
                "reason": "git_show_failed",
                "error": result.stderr.decode("utf-8", errors="replace").strip(),
            })
            continue

        raw_bytes = result.stdout
        if len(raw_bytes) > args.max_file_bytes:
            failures.append({"canonical_instance_id": instance_id, "snapshot_id": snapshot_id, "file_path": file_path, "reason": "file_too_large", "file_size_bytes": len(raw_bytes)})
            continue
        text, encoding = decode_text(raw_bytes)
        if text is None:
            failures.append({"canonical_instance_id": instance_id, "snapshot_id": snapshot_id, "file_path": file_path, "reason": encoding})
            continue
        lines = text.splitlines(keepends=True) or [""]
        if len(lines) > args.max_file_lines:
            failures.append({"canonical_instance_id": instance_id, "snapshot_id": snapshot_id, "file_path": file_path, "reason": "too_many_lines", "line_count": len(lines)})
            continue

        units = extract_line_windows(candidate, lines, args.window_radius, args.max_unit_lines)
        parser_error = None
        if PurePosixPath(file_path).suffix.lower() in PYTHON_SUFFIXES:
            python_units, parser_error = extract_python_units(candidate, text, lines, args.max_unit_lines)
            units.extend(python_units)
        else:
            units.extend(extract_heuristic_units(candidate, lines, args.window_radius))

        unique_units = {unit["evidence_unit_id"]: unit for unit in units}
        units = list(unique_units.values())
        all_units.extend(units)
        file_records.append({
            "file_record_id": "file-" + sha256_text(f"{snapshot_id}|{file_path}")[:24],
            "canonical_instance_id": instance_id,
            "snapshot_id": snapshot_id,
            "repo": candidate.get("repo"),
            "resolved_commit": commit,
            "file_path": file_path,
            "language": language_from_path(file_path),
            "encoding": encoding,
            "file_size_bytes": len(raw_bytes),
            "file_sha256": sha256_bytes(raw_bytes),
            "line_count": len(lines),
            "evidence_unit_count": len(units),
            "parser_error": parser_error,
            "source_mode": "git_show_base_commit",
        })
        if parser_error:
            failures.append({
                "canonical_instance_id": instance_id,
                "snapshot_id": snapshot_id,
                "file_path": file_path,
                "reason": "language_parser_failed",
                "parser": "python_ast",
                "error": parser_error,
                "fallback": "line_window_units_kept",
            })

    all_units.sort(key=lambda item: (item["canonical_instance_id"], item["file_path"], item["start_line"], item["end_line"], item["unit_type"]))
    file_records.sort(key=lambda item: (item["canonical_instance_id"], item["file_path"]))
    write_jsonl(units_path, all_units)
    write_jsonl(files_path, file_records)
    write_jsonl(failures_path, failures)

    report = {
        "schema_version": "1.0",
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": utc_now(),
        "status": "passed" if all_units and file_records else "failed",
        "candidate_file_count": len(candidates),
        "processed_file_count": len(file_records),
        "processed_instance_count": len({record["canonical_instance_id"] for record in file_records}),
        "evidence_unit_count": len(all_units),
        "failure_count": len(failures),
        "unit_type_counts": dict(sorted(Counter(unit["unit_type"] for unit in all_units).items())),
        "language_counts": dict(sorted(Counter(record["language"] for record in file_records).items())),
        "extraction_method_counts": dict(sorted(Counter(unit["extraction_method"] for unit in all_units).items())),
        "limits": {
            "window_radius": args.window_radius,
            "max_file_bytes": args.max_file_bytes,
            "max_file_lines": args.max_file_lines,
            "max_unit_lines": args.max_unit_lines,
        },
        "network_behavior": "仅对候选文件执行 git show；partial clone 可能按需获取这些文件 blob，不读取无关文件。",
        "outputs": {
            "evidence_units": units_path.relative_to(project_root).as_posix(),
            "file_records": files_path.relative_to(project_root).as_posix(),
            "extraction_failures": failures_path.relative_to(project_root).as_posix(),
        },
    }
    write_json(report_path, report)
    print(json.dumps({
        "status": report["status"],
        "candidate_file_count": len(candidates),
        "processed_file_count": len(file_records),
        "processed_instance_count": report["processed_instance_count"],
        "evidence_unit_count": len(all_units),
        "failure_count": len(failures),
        "output_directory": str(output_root),
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
