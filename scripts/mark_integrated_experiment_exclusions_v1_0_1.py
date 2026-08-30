#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mark_integrated_experiment_exclusions_v1_0_1.py

把 integrated dataset 中因 Policy 正动作不可评分而不适合任何实验用途的 task
标记为 experiment_eligible=false，而不是回退到旧 V2.10 supervision，也不是物理删行。

默认输入：
  data/upstream/unified_swe_dataset_v2_10_teacher_v1/
  data/upstream/unified_swe_dataset_v2_10_teacher_v1/integrity_audit.json

默认只从审计错误中提取：
  POSITIVE_UNSCOREABLE
  POSITIVE_MASKED
  LOSS_ACTIVE_UNSCOREABLE

输出：原地原子重写 train/validation/benchmark parquet，并更新 manifest.json。
物理 task 行数保持不变；下游训练/验证/benchmark 必须过滤 experiment_eligible == true。

用法：
  # dry-run
  python scripts/mark_integrated_experiment_exclusions_v1_0_1.py

  # apply
  python scripts/mark_integrated_experiment_exclusions_v1_0_1.py --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import gc
import time
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_VERSION = "1.0.1"
SPLITS = ("train", "validation", "benchmark")
TARGET_CODES = {
    "POSITIVE_UNSCOREABLE",
    "POSITIVE_MASKED",
    "LOSS_ACTIVE_UNSCOREABLE",
}
REASON = "POLICY_POSITIVE_UNSCOREABLE"


def require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("需要 pyarrow: pip install pyarrow") from exc
    return pa, pq


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--derived-dir",
        type=Path,
        default=Path("data/upstream/unified_swe_dataset_v2_10_teacher_v1"),
    )
    p.add_argument("--audit-json", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def parse_audit_targets(audit: dict[str, Any]) -> tuple[set[str], dict[str, set[str]]]:
    targets: set[str] = set()
    codes_by_task: dict[str, set[str]] = defaultdict(set)
    errors = audit.get("errors") or []
    if not isinstance(errors, list):
        raise ValueError("audit.errors 必须是 list")

    for item in errors:
        text = str(item)
        parts = text.split(":", 2)
        if len(parts) < 2:
            continue
        code, task_id = parts[0], parts[1]
        if code not in TARGET_CODES:
            continue
        if not task_id.startswith("task_"):
            raise ValueError(f"异常 task_id: {text}")
        targets.add(task_id)
        codes_by_task[task_id].add(code)
    return targets, codes_by_task


def _close_parquet_file(pf: Any) -> None:
    """Close pyarrow ParquetFile across pyarrow versions."""
    if pf is None:
        return
    close = getattr(pf, "close", None)
    if not callable(close):
        return
    try:
        close(force=True)
    except TypeError:
        close()


def _windows_safe_replace(src: Path, dst: Path, *, attempts: int = 6) -> None:
    """Replace dst after releasing Python/Arrow handles; retry transient Windows locks."""
    last: Exception | None = None
    for i in range(attempts):
        gc.collect()
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last = exc
            if i + 1 < attempts:
                time.sleep(0.25 * (i + 1))
    raise PermissionError(
        f"无法替换 {dst}。Python/pyarrow 句柄已显式关闭；如果仍报 WinError 5，"
        "请关闭 VS Code/IDE 的 Parquet 预览、Python/Jupyter 进程、资源管理器预览窗格或其他占用该文件的程序后重试。"
        f" 临时文件保留在: {src}"
    ) from last


def locate_targets(derived_dir: Path, targets: set[str]) -> tuple[dict[str, str], dict[str, int]]:
    _, pq = require_pyarrow()
    found: dict[str, str] = {}
    counts = Counter()
    for split in SPLITS:
        path = derived_dir / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        pf = None
        try:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(columns=["task_id"], batch_size=4096):
                for row in batch.to_pylist():
                    tid = str(row.get("task_id") or "")
                    if tid in targets:
                        if tid in found:
                            raise ValueError(f"task 重复出现在多个 split: {tid}")
                        found[tid] = split
                        counts[split] += 1
        finally:
            _close_parquet_file(pf)
            pf = None
            gc.collect()
    missing = sorted(targets - set(found))
    if missing:
        raise ValueError(f"审计目标在 derived dataset 中不存在: {missing}")
    return found, dict(counts)


def add_or_get_schema(base_schema, pa):
    fields = {
        "experiment_eligible": pa.bool_(),
        "experiment_exclusion_reason": pa.string(),
        "experiment_exclusion_source": pa.string(),
        "experiment_exclusion_details_json": pa.large_string(),
    }
    schema = base_schema
    for name, typ in fields.items():
        if name not in schema.names:
            schema = schema.append(pa.field(name, typ))
    return schema


def rewrite_split(
    *,
    path: Path,
    split: str,
    targets: set[str],
    codes_by_task: dict[str, set[str]],
    audit_version: str,
    batch_size: int,
) -> tuple[int, int, str, int]:
    pa, pq = require_pyarrow()
    pf = pq.ParquetFile(path)
    out_schema = add_or_get_schema(pf.schema_arrow, pa)

    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    writer = pq.ParquetWriter(tmp_path, out_schema, compression="zstd")
    rows = 0
    excluded = 0
    write_ok = False
    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            out_rows = []
            for row in batch.to_pylist():
                tid = str(row.get("task_id") or "")
                rows += 1
                if tid in targets:
                    excluded += 1
                    row["experiment_eligible"] = False
                    row["experiment_exclusion_reason"] = REASON
                    row["experiment_exclusion_source"] = f"integrity_audit:{audit_version}"
                    row["experiment_exclusion_details_json"] = stable_json({
                        "task_id": tid,
                        "split": split,
                        "issue_codes": sorted(codes_by_task.get(tid) or []),
                        "policy": (
                            "Retain task and Teacher-derived supervision for provenance/audit only; "
                            "exclude from training, validation/model-selection, benchmark evaluation, "
                            "and aggregate experiment metrics."
                        ),
                    })
                else:
                    if "experiment_eligible" not in row or row.get("experiment_eligible") is None:
                        row["experiment_eligible"] = True
                    row.setdefault("experiment_exclusion_reason", "")
                    row.setdefault("experiment_exclusion_source", "")
                    row.setdefault("experiment_exclusion_details_json", "{}")
                out_rows.append(row)
            writer.write_table(pa.Table.from_pylist(out_rows, schema=out_schema))
        write_ok = True
    finally:
        # Windows cannot replace an open Parquet file. Explicitly release both
        # Arrow writer and reader before os.replace().
        try:
            writer.close()
        finally:
            writer = None
            _close_parquet_file(pf)
            pf = None
            gc.collect()

    if not write_ok:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(f"写入临时 parquet 失败: {tmp_path}")

    _windows_safe_replace(tmp_path, path)
    return rows, excluded, sha256_file(path), path.stat().st_size


def update_manifest(
    manifest_path: Path,
    *,
    targets: set[str],
    found: dict[str, str],
    codes_by_task: dict[str, set[str]],
    audit_path: Path,
    audit_version: str,
    split_meta: dict[str, dict[str, Any]],
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_split = Counter(found.values())
    reason_counts = Counter({REASON: len(targets)})

    for split, meta in split_meta.items():
        file_key = f"{split}.parquet"
        if file_key in (manifest.get("files") or {}):
            manifest["files"][file_key]["sha256"] = meta["sha256"]
            manifest["files"][file_key]["bytes"] = meta["bytes"]
            manifest["files"][file_key]["rows"] = meta["rows"]

    integration = manifest.setdefault("integration", {})
    integration["experiment_eligibility"] = {
        "field": "experiment_eligible",
        "eligible_value": True,
        "excluded_task_count": len(targets),
        "excluded_by_split": {s: int(by_split.get(s, 0)) for s in SPLITS},
        "reason_counts": dict(reason_counts),
        "excluded_task_ids": sorted(targets),
        "audit_source": str(audit_path),
        "audit_version": audit_version,
        "rule": (
            "All training, validation/model-selection, benchmark evaluation, and aggregate metrics "
            "must filter experiment_eligible == true. Excluded rows remain only for provenance/audit."
        ),
        "old_supervision_fallback": False,
        "physical_row_deletion": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    integration["training_ready"] = False
    manifest["audit_status"] = "pending_mechanical_integrity_reaudit_after_exclusion_marking"

    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, manifest_path)


def self_test() -> int:
    audit = {
        "errors": [
            "POSITIVE_UNSCOREABLE:task_a:action_x",
            "POSITIVE_MASKED:task_a:action_x",
            "COMPLETE_STOP_NOT_POSITIVE:task_b:state_x:unknown",
            "LOSS_ACTIVE_UNSCOREABLE:task_c:action_y",
        ]
    }
    targets, codes = parse_audit_targets(audit)
    assert targets == {"task_a", "task_c"}
    assert codes["task_a"] == {"POSITIVE_UNSCOREABLE", "POSITIVE_MASKED"}
    assert codes["task_c"] == {"LOSS_ACTIVE_UNSCOREABLE"}
    print("SELF_TEST_OK")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()

    root = Path.cwd().resolve()
    derived_dir = args.derived_dir.resolve() if args.derived_dir.is_absolute() else (root / args.derived_dir).resolve()
    audit_path = (
        (derived_dir / "integrity_audit.json")
        if args.audit_json is None
        else (args.audit_json.resolve() if args.audit_json.is_absolute() else (root / args.audit_json).resolve())
    )
    manifest_path = derived_dir / "manifest.json"

    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_version = str(audit.get("audit_version") or "unknown")
    targets, codes_by_task = parse_audit_targets(audit)
    if not targets:
        raise SystemExit("审计中没有找到需要 experiment exclusion 的 task。")

    found, by_split = locate_targets(derived_dir, targets)
    result = {
        "mode": "APPLY" if args.apply else "DRY_RUN",
        "target_count": len(targets),
        "targets_by_split": {s: int(by_split.get(s, 0)) for s in SPLITS},
        "target_task_ids": sorted(targets),
        "reason": REASON,
        "old_supervision_fallback": False,
        "physical_row_deletion": False,
    }

    if not args.apply:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    split_meta: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = derived_dir / f"{split}.parquet"
        rows, excluded, sha, size = rewrite_split(
            path=path,
            split=split,
            targets=targets,
            codes_by_task=codes_by_task,
            audit_version=audit_version,
            batch_size=args.batch_size,
        )
        split_meta[split] = {
            "rows": rows,
            "excluded": excluded,
            "sha256": sha,
            "bytes": size,
        }

    if sum(int(m["excluded"]) for m in split_meta.values()) != len(targets):
        raise RuntimeError("实际标记数量与目标数量不一致")

    update_manifest(
        manifest_path,
        targets=targets,
        found=found,
        codes_by_task=codes_by_task,
        audit_path=audit_path,
        audit_version=audit_version,
        split_meta=split_meta,
    )

    result["status"] = "OK"
    result["files_rewritten"] = 3
    result["manifest_updated"] = True
    result["physical_rows_preserved"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[mark-exclusions] interrupted", file=sys.stderr)
        raise SystemExit(130)
