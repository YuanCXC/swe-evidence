#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External Supervision Merge v1
（外部监督结果合并器 v1）

建议位置：
    scripts/external_supervision_merge.py

========================================================================
一、定位
========================================================================

本脚本补齐 External Supervision Bridge（外部监督桥接）目前缺少的最后一段：

    Stage 1 normalized results
            +
    Stage 2 normalized results
            +
    冻结 V2.10 的机器上下文
            ↓
    v1.9.2.1 Two-Stage merge（两阶段合并）
            ↓
    Program-built Supervision（程序构造监督）
            ↓
    Core v1.7 Deterministic Verification（确定性验证）
            ↓
    VERIFIED / NEEDS_MORE / BLOCKED

重要：
    - 本脚本不调用任何 LLM API；
    - 不修改冻结 V2.10；
    - 不复制一套新的 supervision 语义；
    - Candidate Number -> Evidence ID、KEEP/REFINE、Certificate、STOP、
      Core verification 全部复用 v1.9.2.1 runner 中已经锁定的正式函数。

========================================================================
二、为什么需要 prepare-context
========================================================================

旧的 refinement_requests.jsonl 是“Teacher 请求审计文件”。
它保存 Prompt / candidate diagnostics，但没有完整保存：

    supervision
    candidate_records
    token_costs

因此不能安全地从 Prompt 文本反解析再构造最终监督。

prepare-context 会：
    1. 只读加载冻结 V2.10；
    2. 对 requests 中的 task_id 重新运行同一个 prepare_task_payload；
    3. 重新得到 candidate_records / supervision / token_costs；
    4. 将重新得到的 Candidate Evidence ID 顺序，与旧 request 中
       candidate_diagnostics.selected_evidence_ids 做逐项比较；
    5. 任何不一致立即 hard fail。

这个 Candidate Identity Lock（候选身份锁）是当前 20 条旧 request 能安全续接的关键。
它避免“配置变化后 Candidate #5 已经不是原来的 Evidence，却仍拿旧网页模型结果合并”的灾难。

========================================================================
三、最终状态
========================================================================

VERIFIED
    Stage 1 / Stage 2 均确定；Programmatic Supervision 构造成功；
    Core v1.7 verification_status == accepted。

NEEDS_MORE
    至少一个 repository_required slot 在 Stage 2 明确返回 insufficient。
    含义是当前 Candidate Pool 缺少必要 pre-fix context。
    不进入训练。

BLOCKED
    包括但不限于：
    - Stage 1 uncertain；
    - Stage 2 uncertain；
    - Stage 2 缺失；
    - Schema / Candidate Number / Candidate Identity 不合法；
    - Programmatic construction exception；
    - Core verification 未 accepted。

training_eligible 在本脚本中始终为 false。
外部 Teacher 结果仍然只是监督修正材料，不会因为 Core accepted 自动提升为训练数据。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

try:
    from tqdm import tqdm
except ImportError as exc:
    raise RuntimeError(
        "缺少 tqdm（进度条依赖）。请执行：python -m pip install -U tqdm"
    ) from exc


SCRIPT_VERSION = "1.0.0"
EXPECTED_RUNNER_VERSION = "1.9.2.1"
EXPECTED_DATASET_VERSION = "2.10.0"

SLOT_TYPES = (
    "fault_location",
    "fault_logic",
    "dependency_context",
    "state_flow",
    "behavior_constraint",
    "repair_scope",
    "validation_constraint",
)

STAGE1_DECISIONS = {
    "repository_required",
    "question_satisfied",
    "not_required",
    "uncertain",
}

STAGE2_STATUSES = {
    "select",
    "insufficient",
    "uncertain",
}

FINAL_STATUSES = {
    "VERIFIED",
    "NEEDS_MORE",
    "BLOCKED",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """严格读取 JSONL；空行允许，非 object 不允许。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} 第 {line_number} 行不是合法 JSON：{exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path} 第 {line_number} 行必须是 JSON object"
                )
            rows.append(value)
    return rows


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    外部网页模型常返回 JSON array；validator 常输出 JSONL。
    这里同时支持两种形式，但不做任何语义修复。
    """
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    if text.startswith("["):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} JSON array 无效：{exc}") from exc
        if not isinstance(value, list):
            raise ValueError(f"{path} 必须是 JSON array 或 JSONL")
        rows = []
        for index, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"{path} array 第 {index} 项必须是 object")
            rows.append(item)
        return rows

    return read_jsonl(path)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Sidecar 使用临时文件 + replace，避免中途失败留下半文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_runner(path: Path) -> ModuleType:
    """
    动态加载当前项目里的 v1.9.2.1 runner。

    为什么不是复制 build_programmatic_refinement_v1_9_2：
        一旦复制，后续修 bug 很容易出现两套实现漂移。
        当前设计要求 program-owned semantics 只有一个来源。
    """
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        "external_merge_refinement_runner",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 runner：{path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    actual_version = str(getattr(module, "RUNNER_VERSION", ""))
    if actual_version != EXPECTED_RUNNER_VERSION:
        raise RuntimeError(
            "External merge 只允许复用已校准的 v1.9.2.1 runner："
            f"actual={actual_version!r}, expected={EXPECTED_RUNNER_VERSION!r}"
        )

    required_functions = (
        "build_parser",
        "load_and_validate_manifest",
        "resolve_split_path",
        "iter_task_rows",
        "prepare_task_payload",
        "build_two_stage_final_consensus",
        "build_programmatic_refinement_v1_9_2",
    )
    missing = [name for name in required_functions if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "v1.9.2.1 runner 缺少 merge 所需函数：" + ", ".join(missing)
        )

    return module


def unique_by_task_id(rows: Sequence[Mapping[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            raise ValueError(f"{source} 存在缺少 task_id 的记录")
        if task_id in result:
            raise ValueError(f"{source} task_id 重复：{task_id}")
        result[task_id] = dict(row)
    return result


def request_candidate_ids(request: Mapping[str, Any]) -> list[str]:
    """
    从旧 refinement_requests.jsonl 的 diagnostics 读取当时真正展示给 Teacher 的顺序。

    Candidate Number 是位置语义，所以不仅集合必须相同，顺序也必须完全相同。
    """
    diagnostics = request.get("candidate_diagnostics") or {}
    ids = diagnostics.get("selected_evidence_ids")
    if ids is None:
        ids = diagnostics.get("teacher_display_evidence_ids")

    if not isinstance(ids, list) or not ids:
        raise ValueError(
            f"task={request.get('task_id')} 的 request 缺少 "
            "candidate_diagnostics.selected_evidence_ids；"
            "无法建立 Candidate Number 身份锁"
        )

    normalized = [str(item) for item in ids]
    if len(normalized) != len(set(normalized)):
        raise ValueError(
            f"task={request.get('task_id')} 的 request Candidate ID 有重复"
        )
    return normalized


def validate_stage1_row(row: Mapping[str, Any]) -> dict[str, Any]:
    task_id = str(row.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("Stage 1 记录缺少 task_id")

    slots = row.get("slots")
    if not isinstance(slots, dict):
        raise ValueError(f"Stage 1 task={task_id} 缺少 slots object")

    actual_keys = set(map(str, slots))
    expected_keys = set(SLOT_TYPES)
    if actual_keys != expected_keys:
        raise ValueError(
            f"Stage 1 task={task_id} slots 必须恰好为固定 7 槽；"
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )

    normalized_slots: dict[str, Any] = {}
    for slot_type in SLOT_TYPES:
        value = slots[slot_type]
        if not isinstance(value, dict):
            raise ValueError(
                f"Stage 1 task={task_id} slot={slot_type} 必须是 object"
            )
        decision = str(value.get("decision") or "").strip()
        if decision not in STAGE1_DECISIONS:
            raise ValueError(
                f"Stage 1 task={task_id} slot={slot_type} decision 非法：{decision!r}"
            )
        normalized_slots[slot_type] = {
            "decision": decision,
            "reason": str(value.get("reason") or "").strip(),
        }

    return {
        "task_id": task_id,
        "slots": normalized_slots,
    }


def canonicalize_witness_groups(
    groups: Any,
    *,
    task_id: str,
    slot_type: str,
    candidate_count: int,
) -> list[list[int]]:
    """
    只做机械 canonicalization（规范化）：
        - 每个 AND group 内去重并升序；
        - OR groups 去重并按 tuple 排序。

    不允许做的事：
        - 把 [[2],[5]] 猜成 [[2,5]]；
        - 删除“看起来冗余”的 OR alternative；
        - 改写 AND / OR 语义。
    """
    if not isinstance(groups, list):
        raise ValueError(
            f"Stage 2 task={task_id} slot={slot_type} witness_groups 必须是 list"
        )

    canonical: set[tuple[int, ...]] = set()
    for group_index, group in enumerate(groups, start=1):
        if not isinstance(group, list) or not group:
            raise ValueError(
                f"Stage 2 task={task_id} slot={slot_type} "
                f"第 {group_index} 个 AND group 必须是非空 list"
            )

        numbers: list[int] = []
        for raw in group:
            # bool 是 int 的子类，必须显式拒绝 True/False。
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ValueError(
                    f"Stage 2 task={task_id} slot={slot_type} "
                    f"Candidate Number 必须是整数：{raw!r}"
                )
            if raw < 1 or raw > candidate_count:
                raise ValueError(
                    f"Stage 2 task={task_id} slot={slot_type} "
                    f"Candidate Number 越界：{raw}, range=1..{candidate_count}"
                )
            numbers.append(raw)

        canonical.add(tuple(sorted(set(numbers))))

    return [list(group) for group in sorted(canonical)]


def validate_stage2_row(
    row: Mapping[str, Any],
    *,
    candidate_count_by_task: Mapping[str, int],
) -> dict[str, Any]:
    task_id = str(row.get("task_id") or "").strip()
    slot_type = str(row.get("slot_type") or "").strip()
    status = str(row.get("status") or "").strip()

    if not task_id:
        raise ValueError("Stage 2 记录缺少 task_id")
    if slot_type not in SLOT_TYPES:
        raise ValueError(
            f"Stage 2 task={task_id} slot_type 非法：{slot_type!r}"
        )
    if status not in STAGE2_STATUSES:
        raise ValueError(
            f"Stage 2 task={task_id} slot={slot_type} status 非法：{status!r}"
        )
    if task_id not in candidate_count_by_task:
        raise ValueError(f"Stage 2 出现未知 task_id：{task_id}")

    candidate_count = int(candidate_count_by_task[task_id])
    raw_groups = row.get("witness_groups")
    if raw_groups is None:
        raw_groups = []

    if status == "select":
        groups = canonicalize_witness_groups(
            raw_groups,
            task_id=task_id,
            slot_type=slot_type,
            candidate_count=candidate_count,
        )
        if not groups:
            raise ValueError(
                f"Stage 2 task={task_id} slot={slot_type} status=select "
                "必须有非空 witness_groups"
            )
    else:
        if raw_groups != []:
            raise ValueError(
                f"Stage 2 task={task_id} slot={slot_type} status={status} "
                "时 witness_groups 必须为 []；程序不能替模型猜语义"
            )
        groups = []

    return {
        "task_id": task_id,
        "slot_type": slot_type,
        "status": status,
        "witness_groups": groups,
        "reason": str(row.get("reason") or "").strip(),
    }


def stage2_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (str(row["task_id"]), str(row["slot_type"]))


def build_requirement_consensus(stage1: Mapping[str, Any]) -> dict[str, Any]:
    """把 External Stage 1 转为 v1.9.2.1 merge 函数需要的协议形状。"""
    return {
        "status": "agreed",
        "stage": "external_requirement_decision",
        "slot_results": {
            slot_type: {
                "decision": stage1["slots"][slot_type]["decision"],
                "reason": stage1["slots"][slot_type].get("reason") or "",
            }
            for slot_type in SLOT_TYPES
        },
        "independent_semantic_review": False,
    }


def build_witness_consensus_map(
    *,
    task_id: str,
    stage1: Mapping[str, Any],
    stage2_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """只为 repository_required slot 构造 Stage 2 merge 输入。"""
    result: dict[str, Any] = {}
    for slot_type in SLOT_TYPES:
        if stage1["slots"][slot_type]["decision"] != "repository_required":
            continue
        row = stage2_by_key.get((task_id, slot_type))
        if row is None:
            result[slot_type] = {
                "status": "blocked",
                "selection_status": "uncertain",
                "witness_groups": [],
                "reason": "missing_external_stage2_result",
            }
            continue
        result[slot_type] = {
            "status": "agreed",
            "selection_status": row["status"],
            "witness_groups": list(row.get("witness_groups") or []),
            "reason": row.get("reason") or "",
        }
    return result


def classify_before_core(
    *,
    stage1: Mapping[str, Any],
    stage2_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """
    在 Core 前先区分 NEEDS_MORE 与 BLOCKED。

    v1.9.2.1 的 merge 函数统一把 insufficient / uncertain 归为 blocked；
    但当前 External Bridge 的研究协议明确要求三分：
        VERIFIED / NEEDS_MORE / BLOCKED。
    所以这里只做“状态分类”，不会改变任何 supervision 语义。
    """
    task_id = str(stage1["task_id"])
    blockers: list[dict[str, Any]] = []
    insufficient: list[dict[str, Any]] = []

    for slot_type in SLOT_TYPES:
        decision = stage1["slots"][slot_type]["decision"]
        if decision == "uncertain":
            blockers.append({
                "stage": "stage1",
                "slot_type": slot_type,
                "reason": "requirement_uncertain",
            })
            continue

        if decision != "repository_required":
            continue

        row = stage2_by_key.get((task_id, slot_type))
        if row is None:
            blockers.append({
                "stage": "stage2",
                "slot_type": slot_type,
                "reason": "missing_stage2_result",
            })
            continue

        status = row["status"]
        if status == "insufficient":
            insufficient.append({
                "stage": "stage2",
                "slot_type": slot_type,
                "reason": row.get("reason") or "candidate_pool_insufficient",
            })
        elif status == "uncertain":
            blockers.append({
                "stage": "stage2",
                "slot_type": slot_type,
                "reason": row.get("reason") or "witness_uncertain",
            })

    # uncertain / missing 的语义优先级高于 NEEDS_MORE：
    # 有未决项时，不能声称“唯一问题就是候选不足”。
    if blockers:
        return "BLOCKED", blockers + insufficient
    if insufficient:
        return "NEEDS_MORE", insufficient
    return None, []


def prepare_context(args: argparse.Namespace) -> int:
    runner = load_runner(args.runner)

    requests = read_jsonl(args.requests.resolve())
    request_by_task = unique_by_task_id(requests, source="requests")
    requested_task_ids = list(request_by_task)
    requested_set = set(requested_task_ids)
    if not requested_task_ids:
        raise ValueError("requests 为空")

    dataset_dir = args.dataset_dir.resolve()
    manifest_path, manifest = runner.load_and_validate_manifest(dataset_dir)
    if str(manifest.get("dataset_version") or "") != EXPECTED_DATASET_VERSION:
        raise ValueError("只允许冻结 V2.10")

    split_path = runner.resolve_split_path(dataset_dir, args.split)

    # ------------------------------------------------------------------
    # Runner defaults 是当前 v1.9.2.1 的单一配置来源。
    # bridge 自己不复制那些默认值；只有用户显式传 override 才覆盖。
    # ------------------------------------------------------------------
    runner_defaults = runner.build_parser().parse_args([])

    def choose(name: str) -> Any:
        explicit = getattr(args, name)
        return getattr(runner_defaults, name) if explicit is None else explicit

    candidate_config = runner.CandidateBuilderConfig(
        candidate_limit=choose("candidate_limit"),
        max_per_file=choose("candidate_max_per_file"),
        test_quota=choose("candidate_test_quota"),
        doc_quota=choose("candidate_doc_quota"),
        resource_quota=choose("candidate_resource_quota"),
        low_value_quota=choose("candidate_low_value_quota"),
        overlap_threshold=choose("candidate_overlap_threshold"),
        gold_units_per_hunk=choose("gold_units_per_hunk"),
        max_gold_units_per_file=choose("max_gold_units_per_file"),
        issue_symbol_units_per_symbol=choose("issue_symbol_units_per_symbol"),
        issue_symbol_policy_path_limit=choose("issue_symbol_policy_path_limit"),
    )
    candidate_config.validate()

    reference_mode = args.reference_mode or runner_defaults.reference_mode
    max_prompt_chars = args.max_prompt_chars or runner_defaults.max_prompt_chars
    max_teacher_question_chars = (
        args.max_teacher_question_chars
        or runner_defaults.max_teacher_question_chars
    )

    # 只扫描目标 split；不会加载整个数据集到内存。
    rows_by_task: dict[str, dict[str, Any]] = {}
    for row in tqdm(
        runner.iter_task_rows(split_path),
        desc="Scan requested tasks（扫描目标任务）",
        unit="task",
        dynamic_ncols=True,
    ):
        task_id = str(row.get("task_id") or "")
        if task_id in requested_set:
            rows_by_task[task_id] = row
            if len(rows_by_task) == len(requested_set):
                break

    missing_tasks = sorted(requested_set - set(rows_by_task))
    if missing_tasks:
        raise ValueError(
            f"split={args.split} 中找不到 {len(missing_tasks)} 个 request task："
            f"{missing_tasks[:10]}"
        )

    cache_path = (args.evidence_cache or runner_defaults.evidence_cache).resolve()
    build_db_path = (args.build_db or runner_defaults.build_db).resolve()
    cache = runner.EvidenceCache(cache_path)
    build_store = runner.BuildEvidenceStore(build_db_path)

    output_rows: list[dict[str, Any]] = []
    try:
        for task_id in tqdm(
            requested_task_ids,
            total=len(requested_task_ids),
            desc="Prepare merge context（准备合并上下文）",
            unit="task",
            dynamic_ncols=True,
        ):
            row = rows_by_task[task_id]
            item = runner.prepare_task_payload(
                task_row=row,
                cache=cache,
                build_store=build_store,
                candidate_config=candidate_config,
                reference_mode=reference_mode,
                max_prompt_chars=max_prompt_chars,
                max_teacher_question_chars=max_teacher_question_chars,
            )

            candidate_records = list(item.get("candidate_records") or [])
            if not candidate_records:
                raise RuntimeError(
                    f"task={task_id} prepare_task_payload 未返回 candidate_records"
                )

            rebuilt_ids = [
                str(record.get("evidence_id") or "")
                for record in candidate_records
            ]
            if any(not evidence_id for evidence_id in rebuilt_ids):
                raise RuntimeError(f"task={task_id} candidate_records 存在空 evidence_id")

            # prepare_task_payload 若已经显式给 candidate_ids，应与 records 完全一致。
            item_ids = [str(x) for x in (item.get("candidate_ids") or rebuilt_ids)]
            if item_ids != rebuilt_ids:
                raise RuntimeError(
                    f"task={task_id} runner 内部 candidate_ids 与 candidate_records 顺序不一致"
                )

            expected_ids = request_candidate_ids(request_by_task[task_id])
            if rebuilt_ids != expected_ids:
                first_diff = None
                for index, (old, new) in enumerate(
                    zip(expected_ids, rebuilt_ids),
                    start=1,
                ):
                    if old != new:
                        first_diff = {
                            "candidate_number": index,
                            "request_evidence_id": old,
                            "rebuilt_evidence_id": new,
                        }
                        break
                if first_diff is None and len(expected_ids) != len(rebuilt_ids):
                    first_diff = {
                        "candidate_number": min(len(expected_ids), len(rebuilt_ids)) + 1,
                        "request_count": len(expected_ids),
                        "rebuilt_count": len(rebuilt_ids),
                    }
                raise RuntimeError(
                    "Candidate Identity Lock FAILED："
                    f"task={task_id}, first_diff={stable_json(first_diff)}。"
                    "不要继续合并旧 Stage 2 Candidate Number；"
                    "请检查当时运行参数或重新导出该任务。"
                )

            supervision = item.get("supervision") or row.get("supervision") or {}
            if not isinstance(supervision, dict) or not supervision:
                raise RuntimeError(f"task={task_id} 缺少 supervision")

            token_costs = item.get("token_costs")
            if not isinstance(token_costs, dict):
                token_costs = {
                    evidence_id: int(record.get("rendered_token_count") or 2**30)
                    for evidence_id, record in zip(rebuilt_ids, candidate_records)
                }
            else:
                token_costs = {str(k): int(v) for k, v in token_costs.items()}

            # existing_evidence_ids 沿用 v1.9.2.1 的当前语义：
            # Teacher 最终只能引用已绑定的 Candidate Evidence IDs。
            output_rows.append({
                "task_id": task_id,
                "split": args.split,
                "runner_version": str(runner.RUNNER_VERSION),
                "dataset_version": str(manifest.get("dataset_version") or ""),
                "manifest": str(manifest_path.resolve()),
                "supervision": supervision,
                "candidate_records": candidate_records,
                "candidate_ids": rebuilt_ids,
                "token_costs": token_costs,
                "existing_evidence_ids": rebuilt_ids,
                "has_boundary": bool(item.get("has_boundary")),
                "candidate_metadata": item.get("candidate_metadata") or {},
                "offline_gold_reference_used": bool(
                    item.get("offline_gold_reference_used")
                ),
                "identity_lock": {
                    "status": "passed",
                    "candidate_count": len(rebuilt_ids),
                    "source": "refinement_requests.candidate_diagnostics.selected_evidence_ids",
                },
            })
    finally:
        cache.close()
        build_store.close()

    output_path = args.output.resolve()
    atomic_write_jsonl(output_path, output_rows)

    report = {
        "script_version": SCRIPT_VERSION,
        "operation": "prepare-context",
        "created_at": utc_now(),
        "runner_version": str(runner.RUNNER_VERSION),
        "dataset_version": str(manifest.get("dataset_version") or ""),
        "split": args.split,
        "request_task_count": len(requested_task_ids),
        "context_task_count": len(output_rows),
        "candidate_identity_lock_passed_count": len(output_rows),
        "output": str(output_path),
        "training_eligible": False,
    }
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_refinement(args: argparse.Namespace) -> int:
    runner = load_runner(args.runner)

    contexts = read_jsonl(args.context.resolve())
    context_by_task = unique_by_task_id(contexts, source="context")
    if not context_by_task:
        raise ValueError("context 为空")

    candidate_count_by_task = {
        task_id: len(row.get("candidate_ids") or [])
        for task_id, row in context_by_task.items()
    }
    for task_id, count in candidate_count_by_task.items():
        if count < 1:
            raise ValueError(f"context task={task_id} candidate_ids 为空")

    stage1_rows_raw = read_json_or_jsonl(args.stage1_results.resolve())
    stage1_rows = [validate_stage1_row(row) for row in stage1_rows_raw]
    stage1_by_task = unique_by_task_id(stage1_rows, source="Stage 1 results")

    # 结果必须覆盖 context 中每个 task；多余 task 也 hard fail。
    context_tasks = set(context_by_task)
    stage1_tasks = set(stage1_by_task)
    if stage1_tasks != context_tasks:
        raise ValueError(
            "Stage 1 与 context task 集不一致："
            f"missing={sorted(context_tasks - stage1_tasks)[:10]}, "
            f"extra={sorted(stage1_tasks - context_tasks)[:10]}"
        )

    stage2_rows_raw = read_json_or_jsonl(args.stage2_results.resolve())
    stage2_rows = [
        validate_stage2_row(
            row,
            candidate_count_by_task=candidate_count_by_task,
        )
        for row in stage2_rows_raw
    ]

    stage2_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in stage2_rows:
        key = stage2_key(row)
        if key in stage2_by_key:
            raise ValueError(
                f"Stage 2 task/slot 重复：task={key[0]}, slot={key[1]}"
            )
        stage2_by_key[key] = row

    # Stage 2 只能回答 Stage 1 的 repository_required slot。
    expected_stage2_keys = {
        (task_id, slot_type)
        for task_id, stage1 in stage1_by_task.items()
        for slot_type in SLOT_TYPES
        if stage1["slots"][slot_type]["decision"] == "repository_required"
    }
    extra_stage2_keys = set(stage2_by_key) - expected_stage2_keys
    if extra_stage2_keys:
        raise ValueError(
            "Stage 2 包含并非 repository_required 的额外 task/slot："
            + stable_json(sorted(extra_stage2_keys)[:20])
        )

    output_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    verification_counts: Counter[str] = Counter()
    assessment_counts: Counter[str] = Counter()
    stage1_decisions: dict[str, Counter[str]] = {
        slot: Counter() for slot in SLOT_TYPES
    }
    stage2_status_counts: Counter[str] = Counter()

    for row in stage2_rows:
        stage2_status_counts[row["status"]] += 1

    for task_id in tqdm(
        context_by_task,
        total=len(context_by_task),
        desc="Quality gate（质量门）",
        unit="task",
        dynamic_ncols=True,
    ):
        context = context_by_task[task_id]
        stage1 = stage1_by_task[task_id]

        for slot_type in SLOT_TYPES:
            stage1_decisions[slot_type][
                stage1["slots"][slot_type]["decision"]
            ] += 1

        pre_status, reasons = classify_before_core(
            stage1=stage1,
            stage2_by_key=stage2_by_key,
        )

        requirement_consensus = build_requirement_consensus(stage1)
        witness_consensus = build_witness_consensus_map(
            task_id=task_id,
            stage1=stage1,
            stage2_by_key=stage2_by_key,
        )
        final_consensus = runner.build_two_stage_final_consensus(
            requirement_consensus=requirement_consensus,
            witness_consensus_by_slot=witness_consensus,
        )

        base_record: dict[str, Any] = {
            "task_id": task_id,
            "external_teacher_protocol": {
                "protocol_version": "external-supervision-bridge-v1",
                "teacher_source": "human-selected external web model",
                "stage1_candidate_blind": True,
                "stage1_allowed_decisions": sorted(STAGE1_DECISIONS),
                "stage2_allowed_statuses": sorted(STAGE2_STATUSES),
                "candidate_number_binding": "identity-locked to pre-fix Candidate Evidence IDs",
                "independent_semantic_review": False,
            },
            "stage1_result": stage1,
            "stage2_results": [
                stage2_by_key[key]
                for key in sorted(expected_stage2_keys)
                if key[0] == task_id and key in stage2_by_key
            ],
            "two_stage_final_consensus": final_consensus,
            # Sidecar 必须保留原监督与 Gold 使用审计信息。
            # 这里记录的是冻结 V2.10 中的原 supervision，不会回写原数据集。
            "original_supervision": context.get("supervision") or {},
            "candidate_ids": list(context.get("candidate_ids") or []),
            "offline_gold_reference_used": bool(
                context.get("offline_gold_reference_used")
            ),
            "candidate_identity_lock": context.get("identity_lock") or {},
            "training_eligible": False,
        }

        if pre_status is not None:
            status_counts[pre_status] += 1
            record = {
                **base_record,
                "final_status": pre_status,
                "status_reasons": reasons,
                "proposal": None,
                "verification": None,
                "supervision_verified": False,
            }
            output_rows.append(record)
            if pre_status == "BLOCKED":
                error_rows.append({
                    "task_id": task_id,
                    "stage": "external_quality_gate",
                    "error_type": "BlockedExternalDecision",
                    "error": stable_json(reasons),
                })
            continue

        # 到这里所有 Stage 1/2 语义都已经明确，应当由正式 v1.9.2.1
        # Programmatic Builder + Core v1.7 接管；bridge 不自行拼 obligation。
        try:
            proposal, verification, construction_error = (
                runner.build_programmatic_refinement_v1_9_2(
                    task_id=task_id,
                    supervision=context["supervision"],
                    candidate_records=context["candidate_records"],
                    candidate_ids=context["candidate_ids"],
                    existing_evidence_ids=set(
                        map(str, context.get("existing_evidence_ids") or context["candidate_ids"])
                    ),
                    token_costs={
                        str(k): int(v)
                        for k, v in (context.get("token_costs") or {}).items()
                    },
                    final_consensus=final_consensus,
                )
            )
        except Exception as exc:
            proposal = None
            verification = None
            construction_error = {
                "stage": "programmatic_supervision_construction",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        if construction_error is not None:
            status = "BLOCKED"
            status_counts[status] += 1
            error_rows.append({
                "task_id": task_id,
                **construction_error,
            })
            output_rows.append({
                **base_record,
                "final_status": status,
                "status_reasons": [construction_error],
                "proposal": proposal,
                "verification": verification,
                "supervision_verified": False,
            })
            continue

        if verification is None:
            status = "BLOCKED"
            status_counts[status] += 1
            reason = {
                "stage": "core_verification",
                "error_type": "MissingVerification",
                "error": "Programmatic builder did not return verification",
            }
            error_rows.append({"task_id": task_id, **reason})
            output_rows.append({
                **base_record,
                "final_status": status,
                "status_reasons": [reason],
                "proposal": proposal,
                "verification": None,
                "supervision_verified": False,
            })
            continue

        verification_status = str(
            verification.get("verification_status") or "unknown"
        )
        verification_counts[verification_status] += 1
        assessment = str(
            verification.get("assessment")
            or (proposal or {}).get("assessment")
            or "unknown"
        )
        assessment_counts[assessment] += 1

        if verification_status == "accepted":
            status = "VERIFIED"
            verified = True
            reasons = []
        else:
            status = "BLOCKED"
            verified = False
            reasons = [{
                "stage": "core_verification",
                "reason": "verification_not_accepted",
                "verification_status": verification_status,
            }]
            error_rows.append({
                "task_id": task_id,
                "stage": "core_verification",
                "error_type": "VerificationRejected",
                "error": stable_json(reasons[0]),
            })

        status_counts[status] += 1
        output_rows.append({
            **base_record,
            "final_status": status,
            "status_reasons": reasons,
            "proposal": proposal,
            "verification": verification,
            "supervision_verified": verified,
            "training_eligible": False,
        })

    if set(status_counts) - FINAL_STATUSES:
        raise AssertionError("内部 final status 非法")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    refinement_path = output_dir / "external_refinement.jsonl"
    errors_path = output_dir / "external_refinement_errors.jsonl"
    report_path = output_dir / "external_refinement_report.json"

    atomic_write_jsonl(refinement_path, output_rows)
    atomic_write_jsonl(errors_path, error_rows)

    report = {
        "script_version": SCRIPT_VERSION,
        "operation": "build-refinement",
        "created_at": utc_now(),
        "runner_version": str(runner.RUNNER_VERSION),
        "task_count": len(context_by_task),
        "stage1_task_count": len(stage1_by_task),
        "stage2_target_count": len(expected_stage2_keys),
        "stage2_returned_count": len(stage2_by_key),
        "stage2_missing_count": len(expected_stage2_keys - set(stage2_by_key)),
        "final_status_counts": dict(sorted(status_counts.items())),
        "verification_status_counts": dict(sorted(verification_counts.items())),
        "assessment_counts": dict(sorted(assessment_counts.items())),
        "stage1_decision_counts_by_slot": {
            slot: dict(sorted(counter.items()))
            for slot, counter in stage1_decisions.items()
        },
        "stage2_status_counts": dict(sorted(stage2_status_counts.items())),
        "supervision_verified_count": int(status_counts.get("VERIFIED", 0)),
        "needs_more_count": int(status_counts.get("NEEDS_MORE", 0)),
        "blocked_count": int(status_counts.get("BLOCKED", 0)),
        "training_eligible_count": 0,
        "training_promotion_policy": "disabled",
        "outputs": {
            "refinement": str(refinement_path),
            "errors": str(errors_path),
            "report": str(report_path),
        },
        "scientific_contract": {
            "core_accepted_is_not_semantic_truth": True,
            "external_teacher_output_is_not_direct_training_label": True,
            "candidate_number_or_of_and_preserved": True,
            "v2_10_modified": False,
        },
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def add_candidate_override_args(parser: argparse.ArgumentParser) -> None:
    """
    这些参数默认 None：None 表示继承 v1.9.2.1 runner 自己的默认值。
    只有当旧 20 条当时显式用了非默认参数时才需要传。

    即使传错，Candidate Identity Lock 也会在写 context 前阻断。
    """
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--candidate-max-per-file", type=int, default=None)
    parser.add_argument("--candidate-test-quota", type=int, default=None)
    parser.add_argument("--candidate-doc-quota", type=int, default=None)
    parser.add_argument("--candidate-resource-quota", type=int, default=None)
    parser.add_argument("--candidate-low-value-quota", type=int, default=None)
    parser.add_argument("--candidate-overlap-threshold", type=float, default=None)
    parser.add_argument("--gold-units-per-hunk", type=int, choices=[1, 2], default=None)
    parser.add_argument("--max-gold-units-per-file", type=int, default=None)
    parser.add_argument("--issue-symbol-units-per-symbol", type=int, default=None)
    parser.add_argument("--issue-symbol-policy-path-limit", type=int, default=None)
    parser.add_argument("--max-prompt-chars", type=int, default=None)
    parser.add_argument("--max-teacher-question-chars", type=int, default=None)
    parser.add_argument("--reference-mode", choices=["gold", "none"], default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge external Stage-1/Stage-2 supervision decisions back into "
            "the frozen v1.9.2.1 programmatic refinement + Core v1.7 verifier."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-context",
        help="从冻结 V2.10 重建机器上下文，并执行 Candidate Identity Lock。",
    )
    prepare.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/refine_supervision_with_llm.py"),
        help="必须是 RUNNER_VERSION=1.9.2.1 的当前 runner。",
    )
    prepare.add_argument("--requests", type=Path, required=True)
    prepare.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/unified_swe_dataset_v2_10"),
    )
    prepare.add_argument(
        "--evidence-cache",
        type=Path,
        default=None,
        help="默认继承 v1.9.2.1 runner。",
    )
    prepare.add_argument(
        "--build-db",
        type=Path,
        default=None,
        help="默认继承 v1.9.2.1 runner。",
    )
    prepare.add_argument(
        "--split",
        choices=["train", "validation", "benchmark"],
        default="validation",
    )
    prepare.add_argument(
        "--output",
        type=Path,
        default=Path("data/.external_supervision/merge_context.jsonl"),
    )
    add_candidate_override_args(prepare)
    prepare.set_defaults(func=prepare_context)

    build = subparsers.add_parser(
        "build-refinement",
        help="合并 Stage1/Stage2，并调用 v1.9.2.1 + Core v1.7。",
    )
    build.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/refine_supervision_with_llm.py"),
    )
    build.add_argument("--context", type=Path, required=True)
    build.add_argument("--stage1-results", type=Path, required=True)
    build.add_argument("--stage2-results", type=Path, required=True)
    build.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/.external_supervision/refinement"),
    )
    build.set_defaults(func=build_refinement)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
