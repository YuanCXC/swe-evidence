#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the Evidence Policy Cross-Encoder from the final four-file bundle.

Normal data dependency (and only data dependency):
    data/evidence_agent_final_v1/
      - tasks.parquet
      - policy_evidence.parquet
      - manifest.json

repository_runtime.sqlite3 is NOT used by offline training.

Policy:
    s_A = f_theta(q, K, A)
    A in {[u], [u,v], STOP}

Training objective:
    multi-positive listwise softmax

For a rankable state with sampled active candidates C and positives P:
    L = logsumexp(scores(C)) - logsumexp(scores(P))

Only actions satisfying all of the following enter the loss:
    state.ranking_loss_mask == True
    action.action_loss_mask == True
    action.scoreable == True
    action.action_label in {positive, negative}

The script builds one disposable SQLite lookup cache next to checkpoints from
policy_evidence.parquet. The cache is an execution accelerator, not a source of
truth and not a dataset dependency.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import ctypes

try:
    libc = ctypes.CDLL(None)
    PR_SET_DUMPABLE = 4
    libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
except Exception:
    pass


SCRIPT_VERSION = "1.5.3"
DEFAULT_BUNDLE_DIR = Path("data/evidence_agent_final_v1")
DEFAULT_OUTPUT_DIR = Path("models/evidence_policy_v1_0")
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MODEL_MAX_LENGTH = 4096
QUESTION_MAX_TOKENS = 2048
SUPPORTED_SPLITS = ("train", "validation", "benchmark")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def make_progress(
    iterable, *, total=None, desc="", unit="it", enabled=True, initial=0
):
    """tqdm progress with a plain-iterator fallback."""
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(
            iterable,
            total=total,
            initial=initial,
            desc=desc,
            unit=unit,
            dynamic_ncols=True,
            mininterval=0.5,
            leave=True,
        )
    except ImportError:
        log(f"[progress] {desc}: tqdm 未安装，退化为阶段日志")
        return iterable


def progress_update(bar, n=1):
    if hasattr(bar, "update"):
        bar.update(n)


def progress_close(bar):
    if hasattr(bar, "close"):
        bar.close()


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG state so a recovery resume continues the same dropout stream."""
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    import torch

    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("torch_cpu") is not None:
        torch.set_rng_state(state["torch_cpu"])
    if state.get("numpy") is not None:
        try:
            import numpy as np
            np.random.set_state(state["numpy"])
        except ImportError:
            pass
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        saved_cuda_states = list(state["torch_cuda"])
        for device_index, cuda_state in enumerate(
            saved_cuda_states[: torch.cuda.device_count()]
        ):
            torch.cuda.set_rng_state(cuda_state, device=device_index)


def torch_load_full(path: Path, *, map_location: str | None = None) -> Any:
    """Load trainer state objects across torch versions (including weights_only defaults)."""
    import torch

    kwargs: dict[str, Any] = {}
    if map_location is not None:
        kwargs["map_location"] = map_location
    try:
        return torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".tmp.", dir=str(path.parent)
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


# ---------------------------------------------------------------------------
# Final bundle contract
# ---------------------------------------------------------------------------

def load_bundle(bundle_dir: Path, *, verify_hashes: bool) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    paths = {
        "manifest": bundle_dir / "manifest.json",
        "tasks": bundle_dir / "tasks.parquet",
        "evidence": bundle_dir / "policy_evidence.parquet",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Bundle 缺少 {name}: {path}")

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if manifest.get("training_ready") is not True:
        raise ValueError("manifest.training_ready != true，拒绝训练")
    if manifest.get("semantic_review_complete") is not True:
        raise ValueError("manifest.semantic_review_complete != true，拒绝训练")
    if manifest.get("integrity_audit_passed") is not True:
        raise ValueError("manifest.integrity_audit_passed != true，拒绝训练")
    if str(manifest.get("status") or "").upper() != "FROZEN":
        raise ValueError(f"Bundle 不是 FROZEN: status={manifest.get('status')!r}")

    contracts = manifest.get("contracts") or {}
    if contracts.get("benchmark_for_training_or_tuning") is not False:
        raise ValueError("manifest 未明确禁止 benchmark 用于 training/tuning")
    if contracts.get("policy_evidence_is_self_contained_for_offline_training") is not True:
        raise ValueError("manifest 未声明 policy_evidence 可用于自包含离线训练")

    files = manifest.get("files") or {}
    if verify_hashes:
        for filename, path_key in (
            ("tasks.parquet", "tasks"),
            ("policy_evidence.parquet", "evidence"),
        ):
            expected = str((files.get(filename) or {}).get("sha256") or "")
            if not expected:
                raise ValueError(f"manifest 缺少 {filename} sha256")
            log(f"[train] hashing {filename} ...")
            actual = sha256_file(paths[path_key])
            if actual != expected:
                raise ValueError(
                    f"{filename} SHA256 不一致: expected={expected}, actual={actual}"
                )

    return {
        "bundle_dir": bundle_dir,
        "manifest_path": paths["manifest"],
        "tasks_path": paths["tasks"],
        "evidence_path": paths["evidence"],
        "manifest": manifest,
        "manifest_sha256": sha256_file(paths["manifest"]),
    }


# ---------------------------------------------------------------------------
# Evidence rendering: copied from the frozen V2.10 input contract
# ---------------------------------------------------------------------------

def render_unit(
    path: str,
    unit_type: str,
    symbol: str | None,
    start: int,
    end: int,
    text: str,
) -> str:
    symbol_line = f"\n[SYMBOL] {symbol}" if symbol else ""
    return (
        f"[PATH] {path}\n[TYPE] {unit_type}\n[LINES] {start}-{end}"
        f"{symbol_line}\n[CONTENT]\n{text}"
    )


def render_metadata(record: Mapping[str, Any]) -> str:
    return (
        f"[EVIDENCE_META] id={record['evidence_id']} path={record.get('path')} "
        f"type={record.get('unit_type')} symbol={record.get('symbol')} "
        f"lines={record.get('start_line')}-{record.get('end_line')}"
    )


def build_question(task_input: Mapping[str, Any]) -> str:
    pieces = [str(task_input.get("problem_statement") or "")]
    hints = task_input.get("hints")
    if isinstance(hints, str):
        if hints.strip():
            pieces.append(hints)
    elif isinstance(hints, Sequence):
        pieces.extend(str(x) for x in hints if str(x).strip())
    return "\n".join(piece for piece in pieces if piece.strip())


def encode_ids_without_length_warning(
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool,
) -> list[int]:
    """Encode without the Hugging Face over-model-length warning.

    This helper is ONLY for length inspection / deterministic middle truncation.
    It does not change the frozen 4096 model-input contract and never feeds an
    over-length sequence to the model.  Some SWE problem statements can be very
    large (100k+ tokenizer tokens); calling tokenizer.encode() directly makes
    Transformers warn against the tokenizer's native max length before we have
    had a chance to truncate the question view.
    """
    old_max = getattr(tokenizer, "model_max_length", None)
    # PreTrainedTokenizerBase emits the warning from model_max_length.  Raising
    # it temporarily suppresses only that warning; truncation is explicitly off.
    try:
        tokenizer.model_max_length = max(
            int(old_max) if isinstance(old_max, int) else 0,
            10**12,
        )
        try:
            ids = tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                truncation=False,
            )
        except TypeError:
            # Tiny/self-test tokenizers may not expose the truncation keyword.
            ids = tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
            )
        return list(ids)
    finally:
        if old_max is not None:
            tokenizer.model_max_length = old_max


def token_length_without_warning(
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool = True,
) -> int:
    return len(
        encode_ids_without_length_warning(
            tokenizer, text, add_special_tokens=add_special_tokens
        )
    )


def truncate_question_view(
    text: str, tokenizer: Any, max_tokens: int = QUESTION_MAX_TOKENS
) -> str:
    # Intentionally inspect the full question, but suppress the misleading
    # tokenizer-native max-length warning.  The returned view is <= 2048 tokens.
    token_ids = encode_ids_without_length_warning(
        tokenizer, text, add_special_tokens=False
    )
    if len(token_ids) <= max_tokens:
        return text
    marker = "[TRUNCATED_MIDDLE]"
    marker_tokens = encode_ids_without_length_warning(
        tokenizer, marker, add_special_tokens=False
    )
    available = max(0, max_tokens - len(marker_tokens))
    head_count = min(1536, int(available * 0.75))
    tail_count = max(0, available - head_count)
    head = tokenizer.decode(token_ids[:head_count], skip_special_tokens=True)
    tail = (
        tokenizer.decode(token_ids[-tail_count:], skip_special_tokens=True)
        if tail_count
        else ""
    )
    result = f"{head}\n{marker}\n{tail}".strip()

    # Decode/re-encode can shift token counts slightly for some tokenizers.  The
    # final question view must still obey the frozen question budget.
    result_ids = encode_ids_without_length_warning(
        tokenizer, result, add_special_tokens=False
    )
    if len(result_ids) > max_tokens:
        result = tokenizer.decode(
            result_ids[:max_tokens], skip_special_tokens=True
        ).strip()
    return result


def render_action_text(
    *,
    question_view: str,
    state: Mapping[str, Any],
    action: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> str:
    state_ids = list(map(str, state.get("evidence_ids") or []))
    state_metadata = "\n".join(
        str(evidence[evidence_id]["metadata"]) for evidence_id in state_ids
    ) or "[EMPTY]"

    body_ids = list(
        map(str, action.get("rendered_state_body_evidence_ids") or [])
    )
    state_body = "\n\n".join(
        f"[STATE BODY] evidence_id={evidence_id}\n"
        f"{evidence[evidence_id]['content']}"
        for evidence_id in body_ids
    ) or "[NONE]"

    candidate_ids = list(map(str, action.get("evidence_ids") or []))
    if candidate_ids:
        candidate = "\n\n".join(
            str(evidence[evidence_id]["rendered_body"])
            for evidence_id in candidate_ids
        )
    else:
        candidate = "[STOP]"

    return (
        f"[QUESTION]\n{question_view}\n\n"
        f"[CURRENT EVIDENCE METADATA]\n{state_metadata}\n\n"
        f"[CURRENT EVIDENCE BODY]\n{state_body}\n\n"
        f"[CANDIDATE ACTION]\n{candidate}"
    )


# ---------------------------------------------------------------------------
# Disposable read-optimized Evidence cache built from policy_evidence.parquet
# ---------------------------------------------------------------------------

def _remove_sqlite_family(path: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = Path(str(path) + suffix)
        with contextlib.suppress(FileNotFoundError):
            candidate.unlink()


def cache_fingerprint(bundle: dict[str, Any]) -> str:
    report = (bundle["manifest"].get("files") or {}).get(
        "policy_evidence.parquet"
    ) or {}
    source_hash = str(report.get("sha256") or "")
    payload = {
        "cache_schema": 1,
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "policy_evidence_sha256": source_hash,
        "policy_evidence_rows": report.get("rows"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def existing_cache_is_valid(path: Path, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        conn = sqlite3.connect(path)
        row = conn.execute(
            "SELECT value FROM meta WHERE key='fingerprint'"
        ).fetchone()
        done = conn.execute(
            "SELECT value FROM meta WHERE key='completed'"
        ).fetchone()
        conn.close()
        return (
            row is not None
            and json.loads(row[0]) == fingerprint
            and done is not None
            and json.loads(done[0]) is True
        )
    except Exception:
        return False


def build_evidence_cache(
    *,
    source_parquet: Path,
    cache_path: Path,
    fingerprint: str,
    rebuild: bool,
    show_progress: bool = True,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    if not rebuild and existing_cache_is_valid(cache_path, fingerprint):
        # Fingerprint + completed=true already prove this cache was built
        # from the current policy_evidence.parquet. Avoid a multi-GB
        # SQLite COUNT(*) scan during recovery.
        count = int(pq.ParquetFile(source_parquet).metadata.num_rows)
        return {"status": "reused", "rows": count, "path": str(cache_path)}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _remove_sqlite_family(cache_path)
    tmp = cache_path.with_name(cache_path.name + f".tmp.{os.getpid()}")
    _remove_sqlite_family(tmp)

    started = time.perf_counter()
    conn = sqlite3.connect(tmp)
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-262144")
        conn.executescript(
            """
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE evidence (
                evidence_id TEXT PRIMARY KEY,
                file_version_id TEXT NOT NULL,
                path TEXT NOT NULL,
                unit_type TEXT NOT NULL,
                symbol TEXT,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                rendered_body TEXT NOT NULL,
                metadata TEXT NOT NULL,
                rendered_token_count INTEGER NOT NULL
            ) WITHOUT ROWID;
            """
        )
        conn.execute(
            "INSERT INTO meta VALUES('fingerprint', ?)",
            (json.dumps(fingerprint),),
        )
        conn.execute(
            "INSERT INTO meta VALUES('completed', 'false')"
        )

        pf = pq.ParquetFile(source_parquet)
        cols = set(pf.schema_arrow.names)
        required = {
            "evidence_id",
            "file_version_id",
            "path",
            "unit_type",
            "start_line",
            "end_line",
            "content",
            "rendered_token_count",
        }
        missing = sorted(required - cols)
        if missing:
            raise ValueError(f"policy_evidence.parquet 缺少列: {missing}")

        columns = [
            "evidence_id",
            "file_version_id",
            "path",
            "unit_type",
            "start_line",
            "end_line",
            "content",
            "rendered_token_count",
        ]
        if "qualified_name" in cols:
            columns.append("qualified_name")
        if "symbol" in cols:
            columns.append("symbol")

        inserted = 0
        buffer: list[tuple[Any, ...]] = []

        def flush() -> None:
            nonlocal buffer
            if not buffer:
                return
            conn.executemany(
                """
                INSERT INTO evidence(
                    evidence_id,file_version_id,path,unit_type,symbol,
                    start_line,end_line,content,rendered_body,metadata,
                    rendered_token_count
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                buffer,
            )
            buffer = []

        rg_iter = make_progress(
            range(pf.num_row_groups),
            total=pf.num_row_groups,
            desc="cache:evidence",
            unit="rg",
            enabled=show_progress,
        )
        for rg in rg_iter:
            table = pf.read_row_group(rg, columns=columns, use_threads=True)
            for record in table.to_pylist():
                evidence_id = str(record.get("evidence_id") or "")
                file_version_id = str(record.get("file_version_id") or "")
                path = str(record.get("path") or "")
                unit_type = str(record.get("unit_type") or "code_block")
                start_line = int(record.get("start_line") or 1)
                end_line = int(record.get("end_line") or start_line)
                content = str(record.get("content") or "")
                symbol_raw = record.get("qualified_name") or record.get("symbol")
                symbol = None if symbol_raw is None else str(symbol_raw)
                rendered_tokens = int(record.get("rendered_token_count") or 0)
                if not evidence_id or not path:
                    raise ValueError(
                        f"policy_evidence 非法行: evidence_id={evidence_id!r}, path={path!r}"
                    )
                meta_record = {
                    "evidence_id": evidence_id,
                    "path": path,
                    "unit_type": unit_type,
                    "symbol": symbol,
                    "start_line": start_line,
                    "end_line": end_line,
                }
                buffer.append(
                    (
                        evidence_id,
                        file_version_id,
                        path,
                        unit_type,
                        symbol,
                        start_line,
                        end_line,
                        content,
                        render_unit(
                            path, unit_type, symbol, start_line, end_line, content
                        ),
                        render_metadata(meta_record),
                        rendered_tokens,
                    )
                )
                inserted += 1
                if len(buffer) >= 10_000:
                    flush()
                    conn.commit()

        flush()
        count = int(conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0])
        if count != inserted:
            raise RuntimeError(
                f"Evidence cache PK 去重/缺失: inserted={inserted}, rows={count}"
            )
        conn.execute(
            "UPDATE meta SET value='true' WHERE key='completed'"
        )
        conn.commit()
        quick = str(conn.execute("PRAGMA quick_check").fetchone()[0])
        if quick.lower() != "ok":
            raise RuntimeError(f"Evidence cache quick_check failed: {quick}")
    except BaseException:
        conn.rollback()
        conn.close()
        _remove_sqlite_family(tmp)
        raise
    else:
        conn.close()
        os.replace(tmp, cache_path)

    return {
        "status": "built",
        "rows": inserted,
        "path": str(cache_path),
        "seconds": round(time.perf_counter() - started, 3),
    }


def open_ro_cache(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    conn = sqlite3.connect(resolved.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA cache_size=-131072")
    return conn


class EvidenceCache:
    def __init__(self, path: Path, expected_fingerprint: str):
        self.connection = open_ro_cache(path)
        meta = {
            str(row["key"]): json.loads(row["value"])
            for row in self.connection.execute("SELECT key,value FROM meta")
        }
        if meta.get("completed") is not True:
            raise ValueError("Evidence lookup cache 未完整构建")
        if meta.get("fingerprint") != expected_fingerprint:
            raise ValueError("Evidence lookup cache fingerprint 不匹配")

    def close(self) -> None:
        self.connection.close()

    def get_many(self, evidence_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        ids = list(dict.fromkeys(map(str, evidence_ids)))
        result: dict[str, dict[str, Any]] = {}
        for offset in range(0, len(ids), 700):
            chunk = ids[offset : offset + 700]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            for row in self.connection.execute(
                f"SELECT * FROM evidence WHERE evidence_id IN ({placeholders})",
                chunk,
            ):
                result[str(row["evidence_id"])] = {
                    "evidence_id": str(row["evidence_id"]),
                    "file_version_id": str(row["file_version_id"]),
                    "path": str(row["path"]),
                    "unit_type": str(row["unit_type"]),
                    "symbol": row["symbol"],
                    "start_line": int(row["start_line"]),
                    "end_line": int(row["end_line"]),
                    "content": str(row["content"]),
                    "rendered_body": str(row["rendered_body"]),
                    "metadata": str(row["metadata"]),
                    "rendered_token_count": int(row["rendered_token_count"]),
                }
        missing = sorted(set(ids) - set(result))
        if missing:
            raise KeyError(
                f"policy_evidence 缺少 {len(missing)} 个引用, first={missing[:5]}"
            )
        return result


# ---------------------------------------------------------------------------
# tasks.parquet streaming and eligibility gate
# ---------------------------------------------------------------------------

def _row_groups_for_split(pf: Any, split: str) -> list[int]:
    """Return row groups that may contain ``split`` using Parquet min/max stats.

    The final bundle is physically written in train -> validation -> benchmark order.
    Older code reopened the Parquet for each split and scanned every preceding row
    group before yielding the first matching task, so ``audit:validation`` could sit
    at 0/223 while silently reading all train task payloads.  This helper performs
    safe row-group pruning: only a row group proven (by split column statistics) to
    contain a different single split is skipped.  If statistics are absent or
    ambiguous, the row group is retained, preserving correctness.
    """
    metadata = pf.metadata
    split_col_index = None
    if metadata is not None and metadata.num_row_groups:
        first_rg = metadata.row_group(0)
        for i in range(first_rg.num_columns):
            try:
                path = str(first_rg.column(i).path_in_schema)
            except Exception:
                path = ""
            if path == "split":
                split_col_index = i
                break

    if split_col_index is None:
        return list(range(pf.num_row_groups))

    selected: list[int] = []
    for rg_index in range(pf.num_row_groups):
        try:
            stats = metadata.row_group(rg_index).column(split_col_index).statistics
        except Exception:
            stats = None
        if stats is None or not getattr(stats, "has_min_max", False):
            selected.append(rg_index)
            continue

        def norm(value: Any) -> str:
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)

        try:
            min_v = norm(stats.min)
            max_v = norm(stats.max)
        except Exception:
            selected.append(rg_index)
            continue

        # Safe pruning only when the row group is known to contain one split and
        # that split differs from the requested one. Mixed/unknown groups remain.
        if min_v == max_v and min_v != split:
            continue
        selected.append(rg_index)
    return selected


def iter_task_rows(
    tasks_path: Path,
    *,
    split: str,
    eligible_only: bool = True,
    shuffle_row_groups: bool = False,
    seed: int = 0,
) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(tasks_path)
    required = {"task_id", "split", "experiment_eligible", "input", "supervision"}
    missing = sorted(required - set(pf.schema_arrow.names))
    if missing:
        raise ValueError(f"tasks.parquet 缺少列: {missing}")

    order = _row_groups_for_split(pf, split)
    if shuffle_row_groups:
        random.Random(seed).shuffle(order)

    columns = ["task_id", "split", "experiment_eligible", "input", "supervision"]
    for rg in order:
        rows = pf.read_row_group(rg, columns=columns, use_threads=True).to_pylist()
        if shuffle_row_groups:
            random.Random(seed ^ (rg * 0x9E3779B1)).shuffle(rows)
        for row in rows:
            if str(row.get("split") or "") != split:
                continue
            if eligible_only and row.get("experiment_eligible") is not True:
                continue
            yield row


def active_actions(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    if state.get("ranking_loss_mask") is not True:
        return []
    return [
        dict(action)
        for action in state.get("candidate_actions") or []
        if action.get("scoreable") is True
        and action.get("action_loss_mask") is True
        and action.get("action_label") in {"positive", "negative"}
    ]


def action_sort_key(action: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        0 if action.get("candidate_scope") == "online" else 1,
        int(action["online_retrieval_rank"])
        if action.get("online_retrieval_rank") is not None
        else 2**31 - 1,
        0 if action.get("action_type") == "single" else 1,
        tuple(map(str, action.get("evidence_ids") or [])),
        str(action.get("action_id") or ""),
    )


def sample_candidate_actions(
    state: Mapping[str, Any],
    *,
    max_candidates: int,
    rng: random.Random,
    pair_negative_quota: int,
) -> list[dict[str, Any]]:
    active = active_actions(state)
    positives = [a for a in active if a.get("action_label") == "positive"]
    negatives = [a for a in active if a.get("action_label") == "negative"]
    if not positives or not negatives:
        return []

    # max_candidates <= 0 means use every active candidate.
    if max_candidates <= 0 or len(active) <= max_candidates:
        return sorted(active, key=action_sort_key)

    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()

    def add(action: dict[str, Any]) -> None:
        action_id = str(action.get("action_id") or "")
        if action_id not in chosen_ids:
            chosen.append(action)
            chosen_ids.add(action_id)

    # Never discard a supervised positive.
    for action in sorted(positives, key=action_sort_key):
        add(action)

    # STOP is part of the same score space, so keep the active STOP control action.
    for action in negatives:
        if action.get("action_type") == "stop":
            add(action)

    budget = max(max_candidates, len(chosen) + 1)

    pair_negatives = sorted(
        [a for a in negatives if a.get("action_type") == "pair"],
        key=action_sort_key,
    )
    pair_slots = min(
        pair_negative_quota,
        max(0, budget - len(chosen) - 1),
        len(pair_negatives),
    )
    for action in pair_negatives[:pair_slots]:
        add(action)

    hard_online_singles = sorted(
        [
            a
            for a in negatives
            if a.get("candidate_scope") == "online"
            and a.get("action_type") == "single"
        ],
        key=action_sort_key,
    )
    for action in hard_online_singles:
        if len(chosen) >= budget:
            break
        add(action)

    remaining = [
        a for a in negatives if str(a.get("action_id") or "") not in chosen_ids
    ]
    rng.shuffle(remaining)
    for action in remaining:
        if len(chosen) >= budget:
            break
        add(action)

    return chosen


@dataclass
class StateExample:
    task_id: str
    state_id: str
    state_type: str
    recommended_weight: float
    state_confidence: float
    texts: list[str]
    positive_mask: list[bool]
    positive_scope_weights: list[float]
    action_types: list[str]
    action_scopes: list[str]
    action_ids: list[str]
    stop_index: int | None


class ExampleBuilder:
    def __init__(
        self,
        *,
        cache: EvidenceCache,
        tokenizer: Any,
        max_candidates: int,
        pair_negative_quota: int,
        offline_positive_weight: float,
        verify_token_counts: bool,
        verify_limit: int,
    ):
        self.cache = cache
        self.tokenizer = tokenizer
        self.max_candidates = max_candidates
        self.pair_negative_quota = pair_negative_quota
        self.offline_positive_weight = offline_positive_weight
        self.verify_token_counts = verify_token_counts
        self.verify_limit = verify_limit
        self.verified_action_count = 0

    def build(
        self,
        row: Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        rng: random.Random,
    ) -> StateExample | None:
        actions = sample_candidate_actions(
            state,
            max_candidates=self.max_candidates,
            rng=rng,
            pair_negative_quota=self.pair_negative_quota,
        )
        if not actions:
            return None

        required_ids: set[str] = set(map(str, state.get("evidence_ids") or []))
        for action in actions:
            required_ids.update(map(str, action.get("evidence_ids") or []))
            required_ids.update(
                map(str, action.get("rendered_state_body_evidence_ids") or [])
            )
        evidence = self.cache.get_many(sorted(required_ids))

        question = build_question(row.get("input") or {})
        question_view = truncate_question_view(question, self.tokenizer)

        texts: list[str] = []
        positive_mask: list[bool] = []
        positive_scope_weights: list[float] = []
        action_types: list[str] = []
        action_scopes: list[str] = []
        action_ids: list[str] = []
        stop_index: int | None = None

        for index, action in enumerate(actions):
            text = render_action_text(
                question_view=question_view,
                state=state,
                action=action,
                evidence=evidence,
            )

            expected_raw = action.get("model_input_token_count")
            if expected_raw is None:
                raise ValueError(
                    "active action 缺少 model_input_token_count: "
                    f"task={row.get('task_id')}, state={state.get('state_id')}, "
                    f"action={action.get('action_id')}"
                )
            expected = int(expected_raw)
            if expected > MODEL_MAX_LENGTH:
                raise ValueError(
                    "active/scoreable action 违反冻结 4096 contract: "
                    f"task={row.get('task_id')}, state={state.get('state_id')}, "
                    f"action={action.get('action_id')}, frozen_tokens={expected}"
                )

            should_verify = (
                self.verify_token_counts
                and self.verified_action_count < self.verify_limit
            )
            if should_verify:
                actual = token_length_without_warning(
                    self.tokenizer, text, add_special_tokens=True
                )
                if actual != expected:
                    raise ValueError(
                        "冻结输入重建 token count 不一致: "
                        f"task={row.get('task_id')}, state={state.get('state_id')}, "
                        f"action={action.get('action_id')}, expected={expected}, actual={actual}"
                    )
                if actual > MODEL_MAX_LENGTH:
                    raise ValueError(
                        "重建输入违反冻结 4096 contract: "
                        f"task={row.get('task_id')}, state={state.get('state_id')}, "
                        f"action={action.get('action_id')}, actual_tokens={actual}"
                    )
                self.verified_action_count += 1

            texts.append(text)
            is_positive = action.get("action_label") == "positive"
            positive_mask.append(is_positive)
            scope = str(action.get("candidate_scope") or "")
            positive_scope_weights.append(
                self.offline_positive_weight
                if is_positive and scope == "offline_injected"
                else 1.0
            )
            action_types.append(str(action.get("action_type") or ""))
            action_scopes.append(scope)
            action_ids.append(str(action.get("action_id") or ""))
            if action.get("action_type") == "stop":
                stop_index = index

        supervision = row.get("supervision") or {}
        weight_raw = supervision.get("recommended_weight")
        recommended_weight = 1.0 if weight_raw is None else float(weight_raw)
        confidence_raw = state.get("confidence")
        state_confidence = 1.0 if confidence_raw is None else float(confidence_raw)

        return StateExample(
            task_id=str(row.get("task_id") or ""),
            state_id=str(state.get("state_id") or ""),
            state_type=str(state.get("state_type") or "unknown"),
            recommended_weight=max(0.0, recommended_weight),
            state_confidence=max(0.0, state_confidence),
            texts=texts,
            positive_mask=positive_mask,
            positive_scope_weights=positive_scope_weights,
            action_types=action_types,
            action_scopes=action_scopes,
            action_ids=action_ids,
            stop_index=stop_index,
        )


def iter_state_examples(
    tasks_path: Path,
    *,
    split: str,
    builder: ExampleBuilder,
    epoch: int,
    seed: int,
    boundary_repeat: int,
    shuffle: bool,
    max_states: int | None,
    skip_states: int = 0,
) -> Iterator[StateExample]:
    """Yield deterministic state examples, optionally skipping a completed prefix.

    skip_states is used by epoch-internal recovery. The skipped prefix is counted
    using the same rankability predicate as ExampleBuilder, but is skipped before
    evidence lookup/token rendering, so resuming late in an epoch is cheap.
    """
    if skip_states < 0:
        raise ValueError("skip_states must be >= 0")
    if max_states is not None and skip_states >= max_states:
        return

    yielded_after_skip = 0
    logical_trainable_seen = 0
    row_seed = seed + epoch * 1_000_003
    for row_index, row in enumerate(
        iter_task_rows(
            tasks_path,
            split=split,
            eligible_only=True,
            shuffle_row_groups=shuffle,
            seed=row_seed,
        )
    ):
        states = list((row.get("supervision") or {}).get("policy_states") or [])
        if shuffle:
            random.Random(row_seed ^ (row_index * 0x85EBCA6B)).shuffle(states)
        for state_index, state in enumerate(states):
            if state.get("ranking_loss_mask") is not True:
                continue
            active = active_actions(state)
            if not any(a.get("action_label") == "positive" for a in active):
                continue
            if not any(a.get("action_label") == "negative" for a in active):
                continue
            repeats = (
                max(1, boundary_repeat)
                if state.get("state_type") == "decision_boundary"
                else 1
            )
            for repeat in range(repeats):
                if max_states is not None and logical_trainable_seen >= max_states:
                    return
                if logical_trainable_seen < skip_states:
                    logical_trainable_seen += 1
                    continue

                rng = random.Random(
                    row_seed
                    ^ (row_index * 0x9E3779B1)
                    ^ (state_index * 0xC2B2AE35)
                    ^ repeat
                )
                example = builder.build(row, state, rng=rng)
                if example is None:
                    # This should be unreachable because the active positive/negative
                    # predicate above matches sample_candidate_actions' gate.
                    continue
                yield example
                logical_trainable_seen += 1
                yielded_after_skip += 1
                if max_states is not None and logical_trainable_seen >= max_states:
                    return


def audit_task_supervision(
    tasks_path: Path,
    *,
    split_totals: Mapping[str, int] | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    report: dict[str, Any] = {"splits": {}}
    split_totals = dict(split_totals or {})
    for split in ("train", "validation", "benchmark"):
        c = Counter()
        if show_progress:
            try:
                import pyarrow.parquet as pq
                _pf_diag = pq.ParquetFile(tasks_path)
                _selected_rgs = _row_groups_for_split(_pf_diag, split)
                log(
                    f"[audit] {split}: reading {len(_selected_rgs)}/{_pf_diag.num_row_groups} "
                    "Parquet row groups (split pushdown)"
                )
                close_fn = getattr(_pf_diag, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass
        rows = iter_task_rows(
            tasks_path, split=split, eligible_only=False, shuffle_row_groups=False
        )
        bar = make_progress(
            rows,
            total=split_totals.get(split),
            desc=f"audit:{split}",
            unit="task",
            enabled=show_progress,
        )
        try:
            for row in bar:
                c["tasks"] += 1
                if row.get("experiment_eligible") is True:
                    c["eligible_tasks"] += 1
                else:
                    c["excluded_tasks"] += 1
                    continue
                for state in (row.get("supervision") or {}).get("policy_states") or []:
                    c["states"] += 1
                    if state.get("ranking_loss_mask") is not True:
                        c["ranking_mask_false"] += 1
                        continue
                    c["rankable_states"] += 1
                    actions = active_actions(state)
                    pos = sum(a.get("action_label") == "positive" for a in actions)
                    neg = sum(a.get("action_label") == "negative" for a in actions)
                    c["active_actions"] += len(actions)
                    c["active_positive"] += pos
                    c["active_negative"] += neg
                    if not pos:
                        c["rankable_without_positive"] += 1
                    if not neg:
                        c["rankable_without_negative"] += 1
                    if pos and neg:
                        c[f"trainable_state_type:{str(state.get('state_type') or 'unknown')}"] += 1
        finally:
            progress_close(bar)
        report["splits"][split] = dict(c)
    train = report["splits"]["train"]
    val = report["splits"]["validation"]
    if train.get("rankable_without_positive", 0):
        raise ValueError("train 存在 rankable state 无 active positive")
    if val.get("rankable_without_positive", 0):
        raise ValueError("validation 存在 rankable state 无 active positive")
    return report


def count_trainable_states(tasks_path: Path, split: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in iter_task_rows(
        tasks_path, split=split, eligible_only=True, shuffle_row_groups=False
    ):
        for state in (row.get("supervision") or {}).get("policy_states") or []:
            if state.get("ranking_loss_mask") is not True:
                continue
            actions = active_actions(state)
            if not any(a.get("action_label") == "positive" for a in actions):
                continue
            if not any(a.get("action_label") == "negative" for a in actions):
                continue
            counts[str(state.get("state_type") or "unknown")] += 1
    return counts


# ---------------------------------------------------------------------------
# Model and objective
# ---------------------------------------------------------------------------

def resolve_precision(requested: str, device: Any) -> str:
    import torch

    requested = requested.lower()
    if requested != "auto":
        if requested in {"fp16", "bf16"} and device.type != "cuda":
            raise ValueError(f"{requested} 仅允许 CUDA")
        if requested == "bf16" and device.type == "cuda":
            if not torch.cuda.is_bf16_supported():
                raise ValueError("当前 CUDA GPU 不支持 bf16")
        return requested
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return "bf16"
        return "fp16"
    return "fp32"


def autocast_context(precision: str, device: Any) -> Any:
    import torch

    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def model_scores(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    device: Any,
    precision: str,
    candidate_microbatch: int,
) -> Any:
    import torch

    score_chunks = []
    for offset in range(0, len(texts), candidate_microbatch):
        chunk = list(texts[offset : offset + candidate_microbatch])

        # Preflight BEFORE tokenizer(..., return_tensors=...) so a malformed
        # reconstructed input can never reach the model tokenizer/forward path.
        lengths = [
            token_length_without_warning(
                tokenizer, text, add_special_tokens=True
            )
            for text in chunk
        ]
        max_length = max(lengths, default=0)
        if max_length > MODEL_MAX_LENGTH:
            bad = [
                (offset + i, length)
                for i, length in enumerate(lengths)
                if length > MODEL_MAX_LENGTH
            ]
            raise ValueError(
                "模型输入在 forward 前超过冻结 4096 contract: "
                f"max={max_length}>{MODEL_MAX_LENGTH}, offending={bad[:8]}"
            )

        encoded = tokenizer(
            chunk,
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        encoded_max_length = int(encoded["input_ids"].shape[1])
        if encoded_max_length != max_length:
            raise ValueError(
                "token preflight 与 tensorized token length 不一致: "
                f"preflight={max_length}, tensorized={encoded_max_length}"
            )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with autocast_context(precision, device):
            output = model(**encoded)
            logits = output.logits
            if logits.ndim == 2 and logits.shape[-1] == 1:
                scores = logits[:, 0]
            elif logits.ndim == 1:
                scores = logits
            else:
                raise ValueError(
                    "Policy model 必须每 action 输出单 logit, "
                    f"shape={tuple(logits.shape)}"
                )
        score_chunks.append(scores)
    return torch.cat(score_chunks, dim=0)


def listwise_loss(
    scores: Any,
    example: StateExample,
    *,
    use_state_confidence: bool,
) -> Any:
    import torch

    positive_mask = torch.tensor(
        example.positive_mask, dtype=torch.bool, device=scores.device
    )
    if not bool(positive_mask.any()):
        raise ValueError("state 没有 positive action")
    scope_weights = torch.tensor(
        example.positive_scope_weights,
        dtype=scores.dtype,
        device=scores.device,
    )
    positive_scores = scores[positive_mask]
    positive_weights = scope_weights[positive_mask].clamp_min(1e-8)
    numerator = torch.logsumexp(
        positive_scores + torch.log(positive_weights), dim=0
    )
    denominator = torch.logsumexp(scores, dim=0)
    loss = denominator - numerator
    weight = max(0.0, float(example.recommended_weight))
    if use_state_confidence:
        weight *= max(0.0, float(example.state_confidence))
    return loss * weight


def state_metrics(scores: Any, example: StateExample) -> dict[str, Any]:
    import torch

    order = torch.argsort(scores, descending=True).tolist()
    positive_indices = {
        i for i, is_positive in enumerate(example.positive_mask) if is_positive
    }
    first_positive_rank = next(
        rank for rank, index in enumerate(order, 1) if index in positive_indices
    )
    top_index = order[0]
    stop_should_win = (
        example.stop_index is not None and example.stop_index in positive_indices
    )
    stop_did_win = (
        example.stop_index is not None and top_index == example.stop_index
    )
    return {
        "hit1": int(top_index in positive_indices),
        "rr": 1.0 / first_positive_rank,
        "stop_correct": int(stop_should_win == stop_did_win),
        "top_scope": example.action_scopes[top_index],
        "top_type": example.action_types[top_index],
        "positive_online": int(
            any(
                p and scope == "online"
                for p, scope in zip(example.positive_mask, example.action_scopes)
            )
        ),
        "positive_offline_injected": int(
            any(
                p and scope == "offline_injected"
                for p, scope in zip(example.positive_mask, example.action_scopes)
            )
        ),
    }


class MetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.loss_sum = 0.0
        self.hit1 = 0
        self.rr_sum = 0.0
        self.stop_correct = 0
        self.by_type: dict[str, dict[str, float]] = defaultdict(
            lambda: {"count": 0.0, "loss": 0.0, "hit1": 0.0, "rr": 0.0, "stop": 0.0}
        )
        self.scope = Counter()
        self.top_action_type = Counter()

    def add(self, *, state_type: str, loss: float, metrics: Mapping[str, Any]) -> None:
        self.count += 1
        self.loss_sum += float(loss)
        self.hit1 += int(metrics["hit1"])
        self.rr_sum += float(metrics["rr"])
        self.stop_correct += int(metrics["stop_correct"])
        bucket = self.by_type[state_type]
        bucket["count"] += 1
        bucket["loss"] += float(loss)
        bucket["hit1"] += int(metrics["hit1"])
        bucket["rr"] += float(metrics["rr"])
        bucket["stop"] += int(metrics["stop_correct"])
        self.scope["positive_online_states"] += int(metrics["positive_online"])
        self.scope["positive_offline_injected_states"] += int(
            metrics["positive_offline_injected"]
        )
        self.top_action_type[str(metrics["top_type"])] += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "loss_sum": self.loss_sum,
            "hit1": self.hit1,
            "rr_sum": self.rr_sum,
            "stop_correct": self.stop_correct,
            "by_type": {k: dict(v) for k, v in self.by_type.items()},
            "scope": dict(self.scope),
            "top_action_type": dict(self.top_action_type),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.count = int(state.get("count", 0))
        self.loss_sum = float(state.get("loss_sum", 0.0))
        self.hit1 = int(state.get("hit1", 0))
        self.rr_sum = float(state.get("rr_sum", 0.0))
        self.stop_correct = int(state.get("stop_correct", 0))
        self.by_type = defaultdict(
            lambda: {"count": 0.0, "loss": 0.0, "hit1": 0.0, "rr": 0.0, "stop": 0.0}
        )
        for key, bucket in (state.get("by_type") or {}).items():
            self.by_type[str(key)] = {
                "count": float(bucket.get("count", 0.0)),
                "loss": float(bucket.get("loss", 0.0)),
                "hit1": float(bucket.get("hit1", 0.0)),
                "rr": float(bucket.get("rr", 0.0)),
                "stop": float(bucket.get("stop", 0.0)),
            }
        self.scope = Counter({str(k): int(v) for k, v in (state.get("scope") or {}).items()})
        self.top_action_type = Counter(
            {str(k): int(v) for k, v in (state.get("top_action_type") or {}).items()}
        )

    def report(self) -> dict[str, Any]:
        by_type = {}
        for name, bucket in sorted(self.by_type.items()):
            n = int(bucket["count"])
            by_type[name] = {
                "count": n,
                "mean_loss": bucket["loss"] / n if n else None,
                "hit_at_1": bucket["hit1"] / n if n else None,
                "mrr": bucket["rr"] / n if n else None,
                "stop_accuracy": bucket["stop"] / n if n else None,
            }
        return {
            "state_count": self.count,
            "mean_loss": self.loss_sum / self.count if self.count else None,
            "hit_at_1": self.hit1 / self.count if self.count else None,
            "mrr": self.rr_sum / self.count if self.count else None,
            "stop_accuracy": self.stop_correct / self.count if self.count else None,
            "by_state_type": by_type,
            "scope": dict(self.scope),
            "top_action_type_counts": dict(self.top_action_type),
        }



def reference_listwise_backward(
    model: Any,
    tokenizer: Any,
    example: StateExample,
    *,
    device: Any,
    precision: str,
    candidate_microbatch: int,
    grad_accum_steps: int,
    use_state_confidence: bool,
    scaler: Any | None,
) -> tuple[Any, Any]:
    """Original listwise backward path.

    All candidate forward graphs for one state are retained until the complete
    listwise loss is formed, then one backward call propagates through the full
    score vector. This preserves the pre-1.5.1 training semantics and is the
    default path in 1.5.3.

    candidate_microbatch only controls how many candidate texts enter each
    model forward. It does NOT change the fact that all candidate autograd
    graphs remain alive until listwise backward completes.
    """
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")

    scores = model_scores(
        model,
        tokenizer,
        example.texts,
        device=device,
        precision=precision,
        candidate_microbatch=candidate_microbatch,
    )
    loss = listwise_loss(
        scores,
        example,
        use_state_confidence=use_state_confidence,
    )
    scaled_loss = loss / grad_accum_steps
    if scaler is not None:
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()
    return scores, loss

def streaming_listwise_backward(
    model: Any,
    tokenizer: Any,
    example: Any,
    *,
    device: Any,
    precision: str,
    candidate_microbatch: int,
    grad_accum_steps: int,
    use_state_confidence: bool,
    scaler: Any | None,
    verify_replay: bool = False,
    replay_atol: float = 1e-5,
) -> tuple[Any, Any]:
    """Backprop a listwise state while keeping only one candidate chunk graph alive.

    Pass 1 runs all candidate chunks under no_grad to obtain the exact score vector
    used by the listwise objective. Pass 2 replays the same chunks with the same RNG
    stream and immediately backpropagates dL/dscore for each chunk, so Transformer
    autograd/checkpoint graphs do not accumulate across all candidates.

    The outer RNG stream is restored to the state produced by pass 1, so the extra
    replay does not advance training randomness. For fp16 GradScaler, dL/dscore is
    computed from the scaled loss so existing unscale_/step logic remains unchanged.

    When verify_replay is enabled, pass-2 logits are compared against the pass-1
    logits used to construct dL/dscore. Any non-finite value or mismatch larger than
    replay_atol aborts immediately with task/state/action diagnostics. This is a
    diagnostic invariant check and does not alter the objective or gradients.
    """
    import torch

    if candidate_microbatch < 1:
        raise ValueError("candidate_microbatch must be >= 1")
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")

    chunks = [
        list(example.texts[offset : offset + candidate_microbatch])
        for offset in range(0, len(example.texts), candidate_microbatch)
    ]
    if not chunks:
        raise ValueError("training example has no candidate texts")

    # Save the RNG state before pass 1. Pass 1 is the canonical forward stream.
    cpu_rng_before = torch.get_rng_state()
    cuda_index = None
    cuda_rng_before = None
    if device.type == "cuda":
        cuda_index = device.index
        if cuda_index is None:
            cuda_index = torch.cuda.current_device()
        cuda_rng_before = torch.cuda.get_rng_state(cuda_index)

    # Pass 1: compute the complete listwise score vector without retaining model graphs.
    detached_score_chunks = []
    with torch.no_grad():
        for chunk in chunks:
            chunk_scores = model_scores(
                model,
                tokenizer,
                chunk,
                device=device,
                precision=precision,
                candidate_microbatch=candidate_microbatch,
            )
            detached_score_chunks.append(chunk_scores.detach())

    scores = torch.cat(detached_score_chunks, dim=0)
    if int(scores.numel()) != len(example.texts):
        raise RuntimeError(
            "streaming candidate score count mismatch: "
            f"scores={int(scores.numel())}, texts={len(example.texts)}"
        )

    # Build the tiny listwise graph only over the score vector and obtain dL/dscore.
    score_leaf = scores.detach().clone().requires_grad_(True)
    loss = listwise_loss(
        score_leaf,
        example,
        use_state_confidence=use_state_confidence,
    )
    scaled_loss = loss / grad_accum_steps
    score_objective = scaler.scale(scaled_loss) if scaler is not None else scaled_loss
    (score_grads,) = torch.autograd.grad(score_objective, score_leaf)
    score_grads = score_grads.detach()

    # Pass 2: replay the same RNG stream and backprop each candidate chunk immediately.
    # fork_rng captures the post-pass-1 state on entry and restores it on exit, so the
    # extra replay is invisible to the global training RNG sequence.
    fork_devices = [cuda_index] if cuda_index is not None else []
    with torch.random.fork_rng(devices=fork_devices, enabled=True):
        torch.set_rng_state(cpu_rng_before)
        if cuda_index is not None and cuda_rng_before is not None:
            torch.cuda.set_rng_state(cuda_rng_before, cuda_index)

        cursor = 0
        replay_score_chunks = [] if verify_replay else None
        for chunk in chunks:
            chunk_scores = model_scores(
                model,
                tokenizer,
                chunk,
                device=device,
                precision=precision,
                candidate_microbatch=candidate_microbatch,
            )
            next_cursor = cursor + len(chunk)
            grad_chunk = score_grads[cursor:next_cursor]
            if tuple(grad_chunk.shape) != tuple(chunk_scores.shape):
                raise RuntimeError(
                    "streaming candidate gradient shape mismatch: "
                    f"grad={tuple(grad_chunk.shape)}, score={tuple(chunk_scores.shape)}"
                )

            if replay_score_chunks is not None:
                replay_score_chunks.append(chunk_scores.detach())

            torch.autograd.backward(chunk_scores, grad_tensors=grad_chunk)
            cursor = next_cursor

        if cursor != len(example.texts):
            raise RuntimeError(
                "streaming candidate replay count mismatch: "
                f"replayed={cursor}, texts={len(example.texts)}"
            )

        if replay_score_chunks is not None:
            replay_scores = torch.cat(replay_score_chunks, dim=0)
            if tuple(replay_scores.shape) != tuple(scores.shape):
                raise RuntimeError(
                    "two-pass score replay shape mismatch: "
                    f"pass1={tuple(scores.shape)}, pass2={tuple(replay_scores.shape)}"
                )

            pass1_float = scores.detach().float()
            pass2_float = replay_scores.detach().float()
            if not bool(torch.isfinite(pass1_float).all()):
                raise RuntimeError(
                    "two-pass score replay found non-finite pass-1 logits: "
                    f"task={example.task_id}, state={example.state_id}, "
                    f"state_type={example.state_type}"
                )
            if not bool(torch.isfinite(pass2_float).all()):
                raise RuntimeError(
                    "two-pass score replay found non-finite pass-2 logits: "
                    f"task={example.task_id}, state={example.state_id}, "
                    f"state_type={example.state_type}"
                )

            replay_diff = (pass2_float - pass1_float).abs()
            max_replay_diff = (
                float(replay_diff.max().item()) if replay_diff.numel() else 0.0
            )
            if max_replay_diff > replay_atol:
                worst_local = int(torch.argmax(replay_diff).item())
                action_id = (
                    example.action_ids[worst_local]
                    if worst_local < len(example.action_ids)
                    else "<out-of-range>"
                )
                raise RuntimeError(
                    "two-pass score replay mismatch: "
                    f"task={example.task_id}, state={example.state_id}, "
                    f"state_type={example.state_type}, candidate_index={worst_local}, "
                    f"action_id={action_id}, "
                    f"pass1={float(pass1_float[worst_local].item()):.9g}, "
                    f"pass2={float(pass2_float[worst_local].item()):.9g}, "
                    f"max_abs_diff={max_replay_diff:.9g}, atol={replay_atol:.9g}"
                )

    return scores, loss

def evaluate_validation(
    *,
    model: Any,
    tokenizer: Any,
    cache: EvidenceCache,
    tasks_path: Path,
    max_candidates: int,
    pair_negative_quota: int,
    offline_positive_weight: float,
    device: Any,
    precision: str,
    candidate_microbatch: int,
    seed: int,
    max_states: int | None,
    expected_states: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    import torch

    builder = ExampleBuilder(
        cache=cache,
        tokenizer=tokenizer,
        max_candidates=max_candidates,
        pair_negative_quota=pair_negative_quota,
        offline_positive_weight=offline_positive_weight,
        verify_token_counts=False,
        verify_limit=0,
    )
    metrics = MetricAccumulator()
    model.eval()
    with torch.inference_mode():
        val_iter = iter_state_examples(
            tasks_path,
            split="validation",
            builder=builder,
            epoch=0,
            seed=seed + 97_531,
            boundary_repeat=1,
            shuffle=False,
            max_states=max_states,
        )
        val_total = expected_states
        if max_states is not None:
            val_total = min(val_total, max_states) if val_total is not None else max_states
        val_bar = make_progress(
            val_iter, total=val_total, desc="validation", unit="state", enabled=show_progress
        )
        for example in val_bar:
            scores = model_scores(
                model,
                tokenizer,
                example.texts,
                device=device,
                precision=precision,
                candidate_microbatch=candidate_microbatch,
            )
            loss = listwise_loss(
                scores, example, use_state_confidence=False
            )
            metrics.add(
                state_type=example.state_type,
                loss=float(loss.detach().float().cpu()),
                metrics=state_metrics(scores, example),
            )
    return metrics.report()


# ---------------------------------------------------------------------------
# Model loading/checkpointing
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, Path | str]:
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    if args.resume_from is not None:
        source: Path | str = resolve_resume_dir(args.resume_from)
        revision = None
    elif args.model_dir is not None:
        source = args.model_dir.resolve()
        revision = None
    else:
        source = args.model_name
        revision = args.model_revision or None

    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=revision,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    config = AutoConfig.from_pretrained(
        source,
        revision=revision,
        local_files_only=args.local_files_only,
    )
    config.num_labels = 1
    config.problem_type = "regression"
    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        revision=revision,
        config=config,
        local_files_only=args.local_files_only,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model, tokenizer, source


def save_checkpoint(
    *,
    output_dir: Path,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    trainer_state: Mapping[str, Any],
    scaler: Any | None = None,
    rng_state: Mapping[str, Any] | None = None,
) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), output_dir / "scheduler.pt")
    if scaler is not None:
        torch.save(scaler.state_dict(), output_dir / "scaler.pt")
    if rng_state is not None:
        torch.save(dict(rng_state), output_dir / "rng_state.pt")
    atomic_write_json(output_dir / "trainer_state.json", dict(trainer_state))


def save_checkpoint_atomic(
    *,
    target_dir: Path,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    trainer_state: Mapping[str, Any],
    scaler: Any | None = None,
    rng_state: Mapping[str, Any] | None = None,
) -> None:
    """Write a checkpoint to staging, then rotate it into place on the same FS."""
    staging = target_dir.with_name(target_dir.name + ".staging")
    previous = target_dir.with_name(target_dir.name + ".previous")
    if staging.exists():
        shutil.rmtree(staging)
    save_checkpoint(
        output_dir=staging,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state=trainer_state,
        scaler=scaler,
        rng_state=rng_state,
    )
    if previous.exists():
        shutil.rmtree(previous)
    if target_dir.exists():
        os.replace(target_dir, previous)
    os.replace(staging, target_dir)
    if previous.exists():
        shutil.rmtree(previous)


def resolve_resume_dir(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_dir():
        return resolved
    previous = resolved.with_name(resolved.name + ".previous")
    if previous.is_dir():
        log(f"[resume] requested checkpoint missing; using previous: {previous}")
        return previous
    raise FileNotFoundError(f"resume checkpoint not found: {resolved}")


def replace_checkpoint_dir(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test() -> int:
    class TinyTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            ids = list(range(len(text.split())))
            if add_special_tokens:
                ids = [101, *ids, 102]
            return ids

        def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
            return " ".join(f"t{x}" for x in ids)

    record = {
        "evidence_id": "ev_a",
        "path": "src/a.py",
        "unit_type": "function",
        "symbol": "f",
        "start_line": 1,
        "end_line": 2,
        "content": "def f():\n    pass",
    }
    assert render_metadata(record).startswith("[EVIDENCE_META] id=ev_a")
    body = render_unit("src/a.py", "function", "f", 1, 2, record["content"])
    evidence = {
        "ev_a": {
            **record,
            "metadata": render_metadata(record),
            "rendered_body": body,
        }
    }
    state = {
        "state_id": "s1",
        "evidence_ids": [],
        "ranking_loss_mask": True,
        "candidate_actions": [
            {
                "action_id": "p",
                "action_type": "single",
                "evidence_ids": ["ev_a"],
                "candidate_scope": "online",
                "scoreable": True,
                "action_loss_mask": True,
                "action_label": "positive",
                "rendered_state_body_evidence_ids": [],
            },
            {
                "action_id": "n",
                "action_type": "stop",
                "evidence_ids": [],
                "candidate_scope": "stop",
                "scoreable": True,
                "action_loss_mask": True,
                "action_label": "negative",
                "rendered_state_body_evidence_ids": [],
            },
        ],
    }
    chosen = sample_candidate_actions(
        state, max_candidates=8, rng=random.Random(1), pair_negative_quota=2
    )
    assert {a["action_id"] for a in chosen} == {"p", "n"}
    text = render_action_text(
        question_view="bug", state=state, action=chosen[0], evidence=evidence
    )
    assert "[QUESTION]\nbug" in text
    assert "[CANDIDATE ACTION]" in text
    assert truncate_question_view("a b c", TinyTokenizer(), 10) == "a b c"
    masked_state = {**state, "ranking_loss_mask": False}
    assert active_actions(masked_state) == []
    m = MetricAccumulator()
    m.add(
        state_type="initial",
        loss=1.25,
        metrics={
            "hit1": 1, "rr": 1.0, "stop_correct": 0,
            "positive_online": 1, "positive_offline_injected": 0, "top_type": "single"
        },
    )
    m2 = MetricAccumulator()
    m2.load_state_dict(m.state_dict())
    assert m2.report() == m.report()
    print("SELF_TEST_OK")
    return 0


# ---------------------------------------------------------------------------
# CLI and training loop
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Evidence Policy only from data/evidence_agent_final_v1."
    )
    p.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument(
        "--checkpoint-every-opt-steps",
        type=int,
        default=500,
        help="每 N 个 optimizer step 覆盖保存一次 epoch 内 recovery；0 表示关闭",
    )
    p.add_argument("--local-files-only", action="store_true")

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-candidates", type=int, default=12)
    p.add_argument("--pair-negative-quota", type=int, default=3)
    p.add_argument("--candidate-microbatch", type=int, default=1)
    p.add_argument(
        "--candidate-backward",
        choices=["reference", "two_pass_streaming"],
        default="reference",
        help=(
            "每个 state 的 backward 实现。reference=原始一次 listwise backward "
            "（1.5.3 默认、正式训练使用）；two_pass_streaming=实验性诊断路径"
        ),
    )
    p.add_argument(
        "--verify-streaming-replay-states",
        type=int,
        default=0,
        help=(
            "诊断 two-pass streaming：对本次进程最先处理的 N 个训练 state "
            "比较 pass-1/pass-2 logits；0 表示关闭"
        ),
    )
    p.add_argument(
        "--streaming-replay-atol",
        type=float,
        default=1e-5,
        help="two-pass logits 一致性检查的最大绝对误差阈值，默认 1e-5",
    )
    p.add_argument("--boundary-repeat", type=int, default=1)
    p.add_argument("--offline-positive-weight", type=float, default=1.0)
    p.add_argument("--use-state-confidence", action="store_true")
    p.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    p.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max-train-states", type=int, default=None)
    p.add_argument("--max-val-states", type=int, default=None)
    p.add_argument("--verify-render-count-actions", type=int, default=64)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--early-stopping-patience", type=int, default=2)
    p.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示阶段/训练/验证进度条；用 --no-progress 关闭",
    )
    p.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="实时写出单条 train/loss 曲线；用 --no-tensorboard 关闭",
    )
    p.add_argument(
        "--loss-ema-beta",
        type=float,
        default=0.98,
        help="TensorBoard train/loss 的 EMA 平滑系数，默认 0.98",
    )

    p.add_argument("--verify-bundle-hashes", action="store_true")
    p.add_argument("--rebuild-evidence-cache", action="store_true")
    p.add_argument("--prepare-cache-only", action="store_true")
    p.add_argument("--audit-data-only", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs 必须 >= 1")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps 必须 >= 1")
    if args.candidate_microbatch < 1:
        raise ValueError("--candidate-microbatch 必须 >= 1")
    if args.verify_streaming_replay_states < 0:
        raise ValueError("--verify-streaming-replay-states 必须 >= 0")
    if args.streaming_replay_atol < 0.0:
        raise ValueError("--streaming-replay-atol 必须 >= 0")
    if (
        args.candidate_backward != "two_pass_streaming"
        and args.verify_streaming_replay_states > 0
    ):
        raise ValueError(
            "--verify-streaming-replay-states 仅允许与 "
            "--candidate-backward two_pass_streaming 一起使用"
        )
    if not (0.0 <= args.loss_ema_beta < 1.0):
        raise ValueError("--loss-ema-beta 必须满足 0 <= beta < 1")
    if args.max_candidates == 1:
        raise ValueError("--max-candidates=1 无法进行 ranking；设 >=2 或 <=0 表示全候选")
    if not (0.0 <= args.offline_positive_weight <= 1.0):
        raise ValueError("--offline-positive-weight 必须位于 [0,1]")
    if not (0.0 <= args.warmup_ratio < 1.0):
        raise ValueError("--warmup-ratio 必须位于 [0,1)")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience 必须 >= 0")
    if args.checkpoint_every_opt_steps < 0:
        raise ValueError("--checkpoint-every-opt-steps 必须 >= 0")


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    validate_args(args)
    seed_everything(args.seed)

    log("[phase 1/8] validating final bundle manifest ...")
    bundle = load_bundle(
        args.bundle_dir, verify_hashes=args.verify_bundle_hashes
    )
    manifest = bundle["manifest"]
    log(
        "[train] bundle ready: "
        + json.dumps(
            {
                "dataset_name": manifest.get("dataset_name"),
                "dataset_version": manifest.get("dataset_version"),
                "task_counts": manifest.get("task_counts"),
                "training_ready": manifest.get("training_ready"),
            },
            ensure_ascii=False,
        )
    )

    # audit/cache-only modes intentionally do not download a model. Normal training
    # downloads/loads the backbone first, then performs the full task scan.
    model = tokenizer = model_source = None
    if not args.audit_data_only and not args.prepare_cache_only:
        log("[phase 2/8] model download/load (CPU) ...")
        try:
            import torch
            from transformers import get_linear_schedule_with_warmup
        except ImportError as exc:
            raise RuntimeError(
                "训练需要 torch + transformers + pyarrow。"
            ) from exc
        model, tokenizer, model_source = load_model_and_tokenizer(args)
        log(f"[phase 2/8] model ready: {model_source}")

    task_counts = manifest.get("task_counts") or {}
    split_totals = {
        split: int(task_counts.get(split, 0) or 0)
        for split in ("train", "validation", "benchmark")
    }
    log("[phase 3/8] auditing task supervision ...")
    audit = audit_task_supervision(
        bundle["tasks_path"], split_totals=split_totals, show_progress=args.progress
    )
    print(json.dumps({"data_audit": audit}, ensure_ascii=False, indent=2))
    if args.audit_data_only:
        return 0

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "evidence_lookup.sqlite3"
    fingerprint = cache_fingerprint(bundle)
    log("[phase 4/8] preparing evidence lookup cache ...")
    cache_report = build_evidence_cache(
        source_parquet=bundle["evidence_path"],
        cache_path=cache_path,
        fingerprint=fingerprint,
        rebuild=args.rebuild_evidence_cache,
        show_progress=args.progress,
    )
    log("[phase 4/8] evidence cache: " + json.dumps(cache_report, ensure_ascii=False))
    if args.prepare_cache_only:
        print(json.dumps(cache_report, ensure_ascii=False, indent=2))
        return 0

    assert model is not None and tokenizer is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[phase 5/8] moving model to {device} ...")
    model.to(device)
    precision = resolve_precision(args.precision, device)

    runtime_summary = {
        "device": str(device),
        "precision": precision,
        "model_source": str(model_source),
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "max_candidates": args.max_candidates,
        "candidate_microbatch": args.candidate_microbatch,
        "gradient_checkpointing": args.gradient_checkpointing,
        "candidate_backward": args.candidate_backward,
        "streaming_replay_verify_states": args.verify_streaming_replay_states,
        "streaming_replay_atol": args.streaming_replay_atol,
        "bundle_manifest_sha256": bundle["manifest_sha256"],
    }
    log("[train] runtime: " + json.dumps(runtime_summary, ensure_ascii=False))
    if args.candidate_backward == "two_pass_streaming":
        log(
            "[train][warning] two_pass_streaming 当前仅用于诊断；"
            "已观察到其参数梯度与 reference backward 存在数值差异，"
            "不要用于正式模型训练。"
        )

    cache = EvidenceCache(cache_path, fingerprint)
    tb_writer = None
    try:
        # Strictly verify a small sample against frozen model_input_token_count.
        if args.verify_render_count_actions > 0:
            verifier = ExampleBuilder(
                cache=cache,
                tokenizer=tokenizer,
                max_candidates=args.max_candidates,
                pair_negative_quota=args.pair_negative_quota,
                offline_positive_weight=args.offline_positive_weight,
                verify_token_counts=True,
                verify_limit=args.verify_render_count_actions,
            )
            for _ in iter_state_examples(
                bundle["tasks_path"],
                split="train",
                builder=verifier,
                epoch=0,
                seed=args.seed,
                boundary_repeat=1,
                shuffle=False,
                max_states=max(128, args.verify_render_count_actions),
            ):
                if verifier.verified_action_count >= args.verify_render_count_actions:
                    break
            if verifier.verified_action_count == 0:
                raise ValueError("未完成任何冻结 model-input token contract 验证")
            log(
                f"[train] render-contract passed: actions={verifier.verified_action_count}"
            )

        def counts_from_audit(split: str) -> Counter[str]:
            src = audit["splits"][split]
            return Counter({
                str(k).split(":", 1)[1]: int(v)
                for k, v in src.items()
                if str(k).startswith("trainable_state_type:")
            })

        train_counts = counts_from_audit("train")
        validation_counts = counts_from_audit("validation")
        base_states = sum(train_counts.values())
        states_per_epoch = (
            base_states
            + train_counts.get("decision_boundary", 0)
            * max(0, args.boundary_repeat - 1)
        )
        if args.max_train_states is not None:
            states_per_epoch = min(states_per_epoch, args.max_train_states)
        if states_per_epoch <= 0:
            raise ValueError("没有可训练 state")

        optimizer_steps_per_epoch = math.ceil(
            states_per_epoch / args.grad_accum_steps
        )
        total_optimizer_steps = max(
            1, optimizer_steps_per_epoch * args.epochs
        )
        warmup_steps = int(total_optimizer_steps * args.warmup_ratio)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )

        start_epoch = 0
        resume_state_in_epoch = 0
        resume_epoch_metrics_state: Mapping[str, Any] | None = None
        resume_sample_scope: Mapping[str, Any] | None = None
        resume_epoch_elapsed_seconds = 0.0
        resume_dir: Path | None = None
        resume_saved: dict[str, Any] = {}
        global_state_step = 0
        global_optimizer_step = 0
        best_mrr = -1.0
        no_improve_epochs = 0
        loss_ema: float | None = None

        resume_contract = {
            "bundle_manifest_sha256": bundle["manifest_sha256"],
            "epochs": args.epochs,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "grad_accum_steps": args.grad_accum_steps,
            "max_grad_norm": args.max_grad_norm,
            "max_candidates": args.max_candidates,
            "pair_negative_quota": args.pair_negative_quota,
            "candidate_microbatch": args.candidate_microbatch,
            "boundary_repeat": args.boundary_repeat,
            "offline_positive_weight": args.offline_positive_weight,
            "use_state_confidence": args.use_state_confidence,
            "precision": args.precision,
            "gradient_checkpointing": args.gradient_checkpointing,
            "max_train_states": args.max_train_states,
        }

        if args.resume_from is not None:
            resume_dir = resolve_resume_dir(args.resume_from)
            optimizer_state = resume_dir / "optimizer.pt"
            scheduler_state = resume_dir / "scheduler.pt"
            trainer_state = resume_dir / "trainer_state.json"
            if not trainer_state.is_file():
                raise FileNotFoundError(f"missing trainer_state.json: {resume_dir}")
            resume_saved = json.loads(trainer_state.read_text(encoding="utf-8"))

            saved_contract = resume_saved.get("resume_contract")
            if saved_contract is not None:
                mismatches = {
                    key: {"saved": saved_contract.get(key), "current": value}
                    for key, value in resume_contract.items()
                    if saved_contract.get(key) != value
                }
                if mismatches:
                    raise ValueError(
                        "resume contract mismatch; use the same training arguments: "
                        + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
                    )
            saved_manifest = resume_saved.get("bundle_manifest_sha256")
            if saved_manifest and saved_manifest != bundle["manifest_sha256"]:
                raise ValueError("resume checkpoint belongs to a different dataset bundle")

            if optimizer_state.is_file():
                optimizer.load_state_dict(torch_load_full(optimizer_state, map_location="cpu"))
            if scheduler_state.is_file():
                scheduler.load_state_dict(torch_load_full(scheduler_state, map_location="cpu"))

            checkpoint_type = str(resume_saved.get("checkpoint_type") or "epoch")
            epoch_complete = bool(resume_saved.get("epoch_complete", checkpoint_type != "recovery"))
            saved_epoch = int(resume_saved.get("epoch", -1))
            if checkpoint_type == "recovery" and not epoch_complete:
                start_epoch = saved_epoch
                resume_state_in_epoch = int(resume_saved.get("epoch_state_count", 0))
                resume_epoch_metrics_state = resume_saved.get("train_metrics_state")
                resume_sample_scope = resume_saved.get("train_sample_scope")
                resume_epoch_elapsed_seconds = float(
                    resume_saved.get("epoch_elapsed_seconds", 0.0)
                )
                if resume_state_in_epoch <= 0:
                    raise ValueError("recovery checkpoint has invalid epoch_state_count")
            else:
                start_epoch = saved_epoch + 1

            global_state_step = int(resume_saved.get("global_state_step", 0))
            global_optimizer_step = int(resume_saved.get("global_optimizer_step", 0))
            best_mrr = float(resume_saved.get("best_validation_mrr", -1.0))
            no_improve_epochs = int(resume_saved.get("no_improve_epochs", 0))
            saved_loss_ema = resume_saved.get("loss_ema")
            if saved_loss_ema is not None:
                loss_ema = float(saved_loss_ema)
            log(
                f"[resume] checkpoint={resume_dir} type={checkpoint_type} "
                f"epoch={start_epoch+1} state_in_epoch={resume_state_in_epoch:,} "
                f"global_opt={global_optimizer_step:,}"
            )

        training_config = {
            "script_version": SCRIPT_VERSION,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "bundle_manifest_sha256": bundle["manifest_sha256"],
            "bundle_dataset_version": manifest.get("dataset_version"),
            "trainable_state_counts": dict(train_counts),
            "validation_state_counts": dict(validation_counts),
            "states_per_epoch": states_per_epoch,
            "total_optimizer_steps": total_optimizer_steps,
            "warmup_steps": warmup_steps,
            "runtime": runtime_summary,
            "loss": "multi_positive_listwise_softmax",
            "experiment_filter": "experiment_eligible == true",
            "benchmark_used": False,
            "checkpoint_every_opt_steps": args.checkpoint_every_opt_steps,
            "recovery_checkpoint": str(output_dir / "recovery"),
        }
        atomic_write_json(output_dir / "training_config.json", training_config)
        metrics_path = output_dir / "metrics.jsonl"

        if args.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise RuntimeError(
                    "实时 loss 曲线需要 tensorboard：pip install tensorboard"
                ) from exc
            tb_dir = output_dir / "tensorboard"
            tb_writer = SummaryWriter(log_dir=str(tb_dir), flush_secs=5)
            log(f"[train] TensorBoard loss curve: {tb_dir}")
            log(
                f"[train] view with: tensorboard --logdir \"{tb_dir}\" --bind_all"
            )

        scaler = None
        if precision == "fp16" and device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda")

        if resume_dir is not None:
            scaler_state = resume_dir / "scaler.pt"
            if scaler is not None and scaler_state.is_file():
                scaler.load_state_dict(torch_load_full(scaler_state, map_location="cpu"))
            rng_state_path = resume_dir / "rng_state.pt"
            if rng_state_path.is_file():
                restore_rng_state(torch_load_full(rng_state_path, map_location="cpu"))

        model.train()
        optimizer.zero_grad(set_to_none=True)

        streaming_replay_states_checked = 0
        streaming_replay_verification_announced = False

        log("[phase 6/8] training epochs ...")
        recovery_dir = output_dir / "recovery"
        for epoch in range(start_epoch, args.epochs):
            epoch_started = time.perf_counter()
            resume_this_epoch = resume_state_in_epoch if epoch == start_epoch else 0
            elapsed_before_resume = (
                resume_epoch_elapsed_seconds if epoch == start_epoch else 0.0
            )
            train_metrics = MetricAccumulator()
            if resume_this_epoch and resume_epoch_metrics_state is not None:
                train_metrics.load_state_dict(resume_epoch_metrics_state)
            sample_scope = Counter()
            if resume_this_epoch and resume_sample_scope is not None:
                sample_scope.update(
                    {str(k): int(v) for k, v in resume_sample_scope.items()}
                )
            builder = ExampleBuilder(
                cache=cache,
                tokenizer=tokenizer,
                max_candidates=args.max_candidates,
                pair_negative_quota=args.pair_negative_quota,
                offline_positive_weight=args.offline_positive_weight,
                verify_token_counts=False,
                verify_limit=0,
            )

            # Recovery checkpoints are written only immediately after optimizer.step(),
            # therefore no partial accumulation bucket has to be reconstructed.
            accumulated = 0
            accumulation_loss_sum = 0.0
            accumulation_loss_count = 0
            epoch_state_count = resume_this_epoch
            if resume_this_epoch:
                log(
                    f"[resume] epoch {epoch+1}: skipping completed prefix "
                    f"{resume_this_epoch:,}/{states_per_epoch:,} states"
                )
            train_iter = iter_state_examples(
                bundle["tasks_path"],
                split="train",
                builder=builder,
                epoch=epoch,
                seed=args.seed,
                boundary_repeat=args.boundary_repeat,
                shuffle=True,
                max_states=args.max_train_states,
                skip_states=resume_this_epoch,
            )
            train_bar = make_progress(
                train_iter,
                total=states_per_epoch,
                initial=resume_this_epoch,
                desc=f"train epoch {epoch+1}/{args.epochs}",
                unit="state",
                enabled=args.progress,
            )
            for example in train_bar:
                global_state_step += 1
                epoch_state_count += 1
                accumulated += 1
                sample_scope["states"] += 1
                sample_scope["candidate_actions"] += len(example.action_ids)
                sample_scope["positive_online_actions"] += sum(
                    p and scope == "online"
                    for p, scope in zip(example.positive_mask, example.action_scopes)
                )
                sample_scope["positive_offline_injected_actions"] += sum(
                    p and scope == "offline_injected"
                    for p, scope in zip(example.positive_mask, example.action_scopes)
                )

                if args.candidate_backward == "reference":
                    scores, loss = reference_listwise_backward(
                        model,
                        tokenizer,
                        example,
                        device=device,
                        precision=precision,
                        candidate_microbatch=args.candidate_microbatch,
                        grad_accum_steps=args.grad_accum_steps,
                        use_state_confidence=args.use_state_confidence,
                        scaler=scaler,
                    )
                else:
                    verify_streaming_replay = (
                        streaming_replay_states_checked
                        < args.verify_streaming_replay_states
                    )
                    scores, loss = streaming_listwise_backward(
                        model,
                        tokenizer,
                        example,
                        device=device,
                        precision=precision,
                        candidate_microbatch=args.candidate_microbatch,
                        grad_accum_steps=args.grad_accum_steps,
                        use_state_confidence=args.use_state_confidence,
                        scaler=scaler,
                        verify_replay=verify_streaming_replay,
                        replay_atol=args.streaming_replay_atol,
                    )
                    if verify_streaming_replay:
                        streaming_replay_states_checked += 1
                        if (
                            streaming_replay_states_checked
                            == args.verify_streaming_replay_states
                            and not streaming_replay_verification_announced
                        ):
                            log(
                                "[train] two-pass streaming replay verification passed: "
                                f"states={streaming_replay_states_checked}, "
                                f"atol={args.streaming_replay_atol:g}"
                            )
                            streaming_replay_verification_announced = True

                loss_value = float(loss.detach().float().cpu())
                train_metrics.add(
                    state_type=example.state_type,
                    loss=loss_value,
                    metrics=state_metrics(scores.detach(), example),
                )
                accumulation_loss_sum += loss_value
                accumulation_loss_count += 1
                if hasattr(train_bar, "set_postfix") and epoch_state_count % 10 == 0:
                    train_bar.set_postfix(
                        loss=f"{train_metrics.loss_sum/max(1, train_metrics.count):.4f}",
                        opt=global_optimizer_step,
                    )

                if accumulated >= args.grad_accum_steps:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated = 0
                    global_optimizer_step += 1
                    if accumulation_loss_count > 0:
                        optimizer_step_loss = accumulation_loss_sum / accumulation_loss_count
                        if loss_ema is None:
                            loss_ema = optimizer_step_loss
                        else:
                            beta = args.loss_ema_beta
                            loss_ema = beta * loss_ema + (1.0 - beta) * optimizer_step_loss
                        if tb_writer is not None:
                            tb_writer.add_scalar(
                                "train/loss", loss_ema, global_optimizer_step
                            )
                        accumulation_loss_sum = 0.0
                        accumulation_loss_count = 0

                    if (
                        args.checkpoint_every_opt_steps > 0
                        and global_optimizer_step % args.checkpoint_every_opt_steps == 0
                        and epoch_state_count < states_per_epoch
                    ):
                        if tb_writer is not None:
                            tb_writer.flush()
                        log(
                            f"[checkpoint] recovery: epoch={epoch+1}/{args.epochs} "
                            f"state={epoch_state_count:,}/{states_per_epoch:,} "
                            f"opt_step={global_optimizer_step:,}"
                        )
                        save_checkpoint_atomic(
                            target_dir=recovery_dir,
                            model=model,
                            tokenizer=tokenizer,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            rng_state=capture_rng_state(),
                            trainer_state={
                                "checkpoint_type": "recovery",
                                "epoch_complete": False,
                                "epoch": epoch,
                                "epoch_state_count": epoch_state_count,
                                "global_state_step": global_state_step,
                                "global_optimizer_step": global_optimizer_step,
                                "best_validation_mrr": best_mrr,
                                "no_improve_epochs": no_improve_epochs,
                                "loss_ema": loss_ema,
                                "train_metrics_state": train_metrics.state_dict(),
                                "train_sample_scope": dict(sample_scope),
                                "epoch_elapsed_seconds": (
                                    elapsed_before_resume
                                    + time.perf_counter()
                                    - epoch_started
                                ),
                                "bundle_manifest_sha256": bundle["manifest_sha256"],
                                "resume_contract": resume_contract,
                            },
                        )

                if args.log_every > 0 and epoch_state_count % args.log_every == 0:
                    elapsed = max(
                        elapsed_before_resume + time.perf_counter() - epoch_started, 1e-9
                    )
                    log(
                        f"[train] epoch={epoch+1}/{args.epochs} "
                        f"state={epoch_state_count:,}/{states_per_epoch:,} "
                        f"opt_step={global_optimizer_step:,} "
                        f"rate={epoch_state_count/elapsed:.2f} state/s "
                        f"loss={train_metrics.loss_sum/max(1,train_metrics.count):.4f}"
                    )

            # Flush the final partial gradient accumulation bucket.
            if accumulated > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_optimizer_step += 1
                if accumulation_loss_count > 0:
                    optimizer_step_loss = accumulation_loss_sum / accumulation_loss_count
                    if loss_ema is None:
                        loss_ema = optimizer_step_loss
                    else:
                        beta = args.loss_ema_beta
                        loss_ema = beta * loss_ema + (1.0 - beta) * optimizer_step_loss
                    if tb_writer is not None:
                        tb_writer.add_scalar(
                            "train/loss", loss_ema, global_optimizer_step
                        )
                    accumulation_loss_sum = 0.0
                    accumulation_loss_count = 0

            if tb_writer is not None:
                tb_writer.flush()

            # Also checkpoint the fully trained epoch before validation. If validation
            # or the subsequent save crashes, resume can skip the whole train prefix
            # and restart directly from validation preparation.
            if args.checkpoint_every_opt_steps > 0:
                log(
                    f"[checkpoint] pre-validation recovery: epoch={epoch+1}/{args.epochs} "
                    f"state={epoch_state_count:,}/{states_per_epoch:,} "
                    f"opt_step={global_optimizer_step:,}"
                )
                save_checkpoint_atomic(
                    target_dir=recovery_dir,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    rng_state=capture_rng_state(),
                    trainer_state={
                        "checkpoint_type": "recovery",
                        "epoch_complete": False,
                        "epoch": epoch,
                        "epoch_state_count": epoch_state_count,
                        "global_state_step": global_state_step,
                        "global_optimizer_step": global_optimizer_step,
                        "best_validation_mrr": best_mrr,
                        "no_improve_epochs": no_improve_epochs,
                        "loss_ema": loss_ema,
                        "train_metrics_state": train_metrics.state_dict(),
                        "train_sample_scope": dict(sample_scope),
                        "epoch_elapsed_seconds": (
                            elapsed_before_resume
                            + time.perf_counter()
                            - epoch_started
                        ),
                        "bundle_manifest_sha256": bundle["manifest_sha256"],
                        "resume_contract": resume_contract,
                    },
                )

            log(f"[phase 7/8] validation after epoch {epoch+1} ...")
            validation = evaluate_validation(
                model=model,
                tokenizer=tokenizer,
                cache=cache,
                tasks_path=bundle["tasks_path"],
                max_candidates=max(args.max_candidates, 12),
                pair_negative_quota=args.pair_negative_quota,
                offline_positive_weight=args.offline_positive_weight,
                device=device,
                precision=precision,
                candidate_microbatch=args.candidate_microbatch,
                seed=args.seed,
                max_states=args.max_val_states,
                expected_states=sum(validation_counts.values()),
                show_progress=args.progress,
            )
            train_report = train_metrics.report()
            current_mrr = float(validation.get("mrr") or 0.0)
            improved = current_mrr > best_mrr
            if improved:
                best_mrr = current_mrr
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            record = {
                "epoch": epoch + 1,
                "global_state_step": global_state_step,
                "global_optimizer_step": global_optimizer_step,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "train": train_report,
                "train_sample_scope": dict(sample_scope),
                "validation": validation,
                "best_validation_mrr": best_mrr,
                "improved": improved,
                "no_improve_epochs": no_improve_epochs,
                "seconds": round(
                    elapsed_before_resume + time.perf_counter() - epoch_started, 3
                ),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

            # Keep only two named checkpoints: last and best.
            log(f"[phase 8/8] saving checkpoint for epoch {epoch+1} ...")
            last_dir = output_dir / "last"
            save_checkpoint_atomic(
                target_dir=last_dir,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                rng_state=capture_rng_state(),
                trainer_state={
                    "checkpoint_type": "epoch",
                    "epoch_complete": True,
                    "epoch": epoch,
                    "epoch_state_count": states_per_epoch,
                    "global_state_step": global_state_step,
                    "global_optimizer_step": global_optimizer_step,
                    "best_validation_mrr": best_mrr,
                    "no_improve_epochs": no_improve_epochs,
                    "loss_ema": loss_ema,
                    "validation": validation,
                    "bundle_manifest_sha256": bundle["manifest_sha256"],
                    "resume_contract": resume_contract,
                },
            )
            # The completed epoch checkpoint supersedes any in-epoch recovery.
            for stale_recovery in (
                recovery_dir,
                recovery_dir.with_name(recovery_dir.name + ".previous"),
                recovery_dir.with_name(recovery_dir.name + ".staging"),
            ):
                if stale_recovery.exists():
                    shutil.rmtree(stale_recovery)
            if improved:
                best_dir = output_dir / "best"
                replace_checkpoint_dir(last_dir, best_dir)

            print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
            model.train()

            if (
                args.early_stopping_patience > 0
                and no_improve_epochs >= args.early_stopping_patience
            ):
                log(
                    f"[train] early stop: no validation MRR improvement for "
                    f"{no_improve_epochs} epoch(s)"
                )
                break

        final = {
            "status": "OK",
            "output_dir": str(output_dir),
            "best_checkpoint": str(output_dir / "best"),
            "last_checkpoint": str(output_dir / "last"),
            "recovery_checkpoint": str(output_dir / "recovery"),
            "best_validation_mrr": best_mrr,
            "global_state_step": global_state_step,
            "global_optimizer_step": global_optimizer_step,
            "benchmark_used": False,
            "bundle_manifest_sha256": bundle["manifest_sha256"],
            "tensorboard_log_dir": str(output_dir / "tensorboard") if args.tensorboard else None,
        }
        atomic_write_json(output_dir / "training_result.json", final)
        print(json.dumps(final, ensure_ascii=False, indent=2))
        return 0
    finally:
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
        cache.close()


if __name__ == "__main__":
    raise SystemExit(main())
