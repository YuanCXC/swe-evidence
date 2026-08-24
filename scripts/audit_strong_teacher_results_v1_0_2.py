#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strong-Teacher 结果离线审计器 v1.0.1（兼容 v1.0 报告结构）。

目标：
1. 100% 扫描文件放错 split / 找不到对应题目 / 0B / 重复文件。
2. 校验“题目 task_id == 答案 task_id == 文件对应关系”。
3. 校验 JSON Schema、7 个 canonical slots、枚举、Candidate Number 绑定。
4. 检查 OR-of-AND 可机械发现的非最小/重复结构。
5. 基于高风险模式生成 semantic review queue（语义人工复核队列）。
6. 不修改任何题目或答案文件，只在内存中执行可证明安全的规范化/降级，并生成报告。

默认目录契约：
  input:
    data/.external_supervision/strong_teacher_v1_3_all/{split}/md/*.md
  result:
    data/.external_supervision/result/{split}/*.md

报告默认写入：
  data/.external_supervision/.audit/strong_teacher_audit/

重要边界：
- HARD_ERROR 只用于可以机械证明的异常。
- RISK_FLAG 不是“答案错误”，只是优先人工复核。
- Gold Change Hints 只用于离线风险提示，不会据此自动判答案失败。
- v1.0.1 将“helpful+Witness / 空 reason / uncertain pool+Witness”从 HARD_ERROR 降为 RISK_FLAG。
- v1.0.1 可对可证明属于当前文件的错误 task_id 做内存规范化；绝不写回 result。
- v1.0.1 会保守丢弃 malformed additional_findings（非核心监督），并记录风险。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

SPLITS = ("train", "validation", "benchmark")

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

RISK_WEIGHTS = {
    "CAUSAL_MULTI_AND": 4,
    "REPAIR_MULTI_OR": 3,
    "LARGE_AND": 3,
    "QUESTION_SUFFICIENT_BUT_REPO_REQUIRED": 2,
    "WITNESS_SATURATION": 2,
    "SAME_SINGLETON_SATURATION": 2,
    "UNCERTAINTY_CAUSAL_CONTRADICTION": 3,
    "REPAIR_GOLD_PATH_DIVERGENCE": 3,
    "POOL_STATUS_CROSS_SLOT_TENSION": 2,
    "MANY_SUPPORTING": 1,
    "HELPFUL_WITH_WITNESS": 1,
    "EMPTY_REASON": 1,
    "UNCERTAIN_POOL_WITH_WITNESS": 2,
    "TASK_ID_SAFE_NORMALIZED": 1,
    "ADDITIONAL_FINDINGS_SANITIZED": 1,
    "JSON_SAFE_SALVAGED": 1,
}

@dataclass
class AuditIssue:
    severity: str       # HARD_ERROR / RISK_FLAG / INFO
    code: str
    split: str
    filename: str
    task_id: str
    detail: str
    risk_points: int = 0

@dataclass
class FileRecord:
    split: str
    filename: str
    path: Path
    task_id: Optional[str] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit Strong-Teacher question/result alignment and answer quality risks."
    )
    p.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/.external_supervision/strong_teacher_v1_3_all"),
    )
    p.add_argument(
        "--result-root",
        type=Path,
        default=Path("data/.external_supervision/result"),
    )
    p.add_argument(
        "--report-root",
        type=Path,
        default=Path("data/.external_supervision/.audit/strong_teacher_audit"),
    )
    p.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
    )
    p.add_argument(
        "--review-top",
        type=int,
        default=300,
        help="输出 risk score 最高的前 N 条到 semantic_review_queue.csv；默认 300。",
    )
    p.add_argument(
        "--random-low-sample",
        type=int,
        default=200,
        help="从无 HARD_ERROR 且 risk score 较低的答案随机抽样 N 条；默认 200。",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260813,
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="每处理 N 个文件打印一次进度；0 表示关闭。默认 500。",
    )
    p.add_argument(
        "--allow-missing-results",
        action="store_true",
        help=(
            "允许 question 没有 result。此时记为 EXCLUDED/MISSING_RESULT_IGNORED，"
            "不计入 HARD_ERROR，也不进入 per-answer 样本。"
        ),
    )
    return p.parse_args()


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


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


def filename_task_ids(filename: str) -> list[str]:
    return re.findall(r"task_[A-Za-z0-9]+", Path(filename).stem)


def extract_candidates(text: str) -> tuple[set[int], dict[int, str]]:
    nums: set[int] = set()
    paths: dict[int, str] = {}

    for m in re.finditer(
        r"(?m)^\[CANDIDATE\s+(\d+)\](?P<header>.*)$",
        text,
    ):
        n = int(m.group(1))
        nums.add(n)
        header = m.group("header")
        pm = re.search(r"\|\s*path=([^|]+?)(?:\s*\||$)", header)
        if pm:
            paths[n] = pm.group(1).strip()

    if not nums:
        raise ValueError("题目中未检测到任何 [CANDIDATE N]")
    return nums, paths


def extract_gold_changed_files(text: str) -> list[str]:
    marker = "[GOLD CHANGE HINTS - OFFLINE ONLY, NOT EVIDENCE]"
    idx = text.find(marker)
    if idx < 0:
        return []
    tail = text[idx + len(marker):]

    m = re.search(
        r'"changed_files"\s*:\s*(\[[\s\S]*?\])\s*,\s*"hunk_headers"',
        tail,
    )
    if not m:
        return []
    try:
        arr = json.loads(m.group(1))
    except Exception:
        return []
    return [x for x in arr if isinstance(x, str)]


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


def _validate_top_level_answer(data: Any) -> dict[str, Any]:
    if not isinstance(data, list):
        raise ValueError("顶层不是 JSON array")
    if len(data) != 1:
        raise ValueError(f"顶层 array 必须恰好 1 个 object，实际 {len(data)}")
    if not isinstance(data[0], dict):
        raise ValueError("array 唯一元素不是 object")
    return data[0]


def parse_answer_with_safe_salvage(text: str) -> tuple[dict[str, Any], list[str]]:
    """
    解析答案。仅做可证明安全的 JSON salvage：
    - 正常纯 JSON；
    - 外层 accidental ``` / ```json fence；
    - 在一个完整 JSON 值之后只剩空白或纯 fence 标记。

    不接受：第二个 JSON、自然语言后缀、未知前后缀。
    返回 (answer_object, warning_codes)。
    """
    warnings: list[str] = []
    s = strip_accidental_fence(text)

    try:
        data = json.loads(s)
        return _validate_top_level_answer(data), warnings
    except json.JSONDecodeError as original_exc:
        decoder = json.JSONDecoder()
        try:
            data, end = decoder.raw_decode(s)
        except Exception:
            raise original_exc

        tail = s[end:].strip()
        # 只允许剩余纯 Markdown fence；任何文本、第二个 JSON 都拒绝。
        if tail and not re.fullmatch(r"(?:```(?:json)?\s*)+", tail, flags=re.IGNORECASE):
            raise original_exc

        obj = _validate_top_level_answer(data)
        warnings.append("JSON_SAFE_SALVAGED")
        return obj, warnings


def parse_answer(text: str) -> dict[str, Any]:
    # 保留旧接口，供外部 import 兼容。
    obj, _ = parse_answer_with_safe_salvage(text)
    return obj


def safe_task_id_alias(
    answer_task_id: Any,
    expected_task_id: str,
    filename: str,
    known_task_ids: set[str],
) -> bool:
    """
    仅在能从当前 filename 机械证明 alias 属于当前题时返回 True。

    允许：
      1) answer task_id == 当前文件 stem；
      2) answer task_id == 当前文件开头的序号 token（如 task_012194），
         且 expected_task_id 同时明确出现在 filename 中。

    若 answer_task_id 本身是另一个真实题目的 task_id，则绝不 normalize。
    """
    if not isinstance(answer_task_id, str) or not expected_task_id:
        return False
    if answer_task_id == expected_task_id:
        return False
    if answer_task_id in known_task_ids:
        return False

    stem = Path(filename).stem
    ids = filename_task_ids(filename)
    if expected_task_id not in ids:
        return False

    if answer_task_id == stem:
        return True

    m = re.match(r"^(task_\d+)(?:_|$)", stem)
    if m and answer_task_id == m.group(1):
        return True

    return False


def sanitize_additional_findings(
    obj: dict[str, Any],
    legal_candidates: set[int],
) -> tuple[int, list[str]]:
    """
    additional_findings 是非核心附加信息。对 malformed item 采取保守丢弃，
    不据此否定 7-slot 核心监督。绝不推断/补写缺失语义。
    返回 (dropped_count, details)。
    """
    findings = obj.get("additional_findings")
    details: list[str] = []

    if not isinstance(findings, list):
        obj["additional_findings"] = []
        return 1, ["additional_findings 不是 list，已在内存中置空"]

    kept: list[dict[str, Any]] = []
    dropped = 0
    for i, f in enumerate(findings):
        reason = None
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
                    validate_candidate_refs(
                        nums, legal_candidates, f"additional_findings[{i}]"
                    )
                except Exception as exc:
                    reason = str(exc)

        if reason is not None:
            dropped += 1
            details.append(f"[{i}] {reason}")
        else:
            kept.append(f)

    if dropped:
        obj["additional_findings"] = kept
    return dropped, details


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


def canonical_group(group: list[int]) -> tuple[int, ...]:
    return tuple(sorted(group))


def validate_slot(
    slot_name: str,
    slot: Any,
    legal_candidates: set[int],
) -> list[str]:
    """
    返回可机械证明但不一定需要阻断的 canonicalization warnings。
    真正 schema/binding 错误直接 raise ValueError。
    """
    warnings: list[str] = []

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
        raise ValueError(f"slot {slot_name} 缺字段: {missing}")

    for field, allowed in ENUMS.items():
        if slot.get(field) not in allowed:
            raise ValueError(
                f"slot {slot_name}.{field} 非法枚举={slot.get(field)!r}"
            )

    if not isinstance(slot["reason"], str):
        raise ValueError(f"slot {slot_name}.reason 必须是字符串")
    if not slot["reason"].strip():
        warnings.append("EMPTY_REASON")

    groups = slot["sufficient_witness_groups"]
    if not isinstance(groups, list):
        raise ValueError(f"slot {slot_name}.sufficient_witness_groups 不是 list")

    canonical_groups: list[tuple[int, ...]] = []
    flat: list[int] = []

    for i, g in enumerate(groups):
        ints = ensure_int_list(g, f"{slot_name}.groups[{i}]")
        if not ints:
            raise ValueError(f"{slot_name}.groups[{i}] 是空 AND group")
        if len(ints) != len(set(ints)):
            raise ValueError(f"{slot_name}.groups[{i}] 内 Candidate 重复")
        validate_candidate_refs(ints, legal_candidates, f"{slot_name}.groups[{i}]")
        flat.extend(ints)
        canonical_groups.append(canonical_group(ints))

    # OR alternatives 完全重复，是可机械证明的非规范化。
    if len(canonical_groups) != len(set(canonical_groups)):
        warnings.append("DUPLICATE_OR_GROUP")

    # 若一个 sufficient group 是另一个 sufficient group 的真超集，
    # 根据协议 Superset Elimination 可机械证明它不是 minimal。
    sets = [set(x) for x in canonical_groups]
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            if i != j and b < a:
                warnings.append(
                    f"NONMINIMAL_SUPERSET_GROUP:{canonical_groups[i]}>{canonical_groups[j]}"
                )

    supporting = ensure_int_list(
        slot["supporting_candidates"],
        f"{slot_name}.supporting_candidates",
    )
    validate_candidate_refs(
        supporting,
        legal_candidates,
        f"{slot_name}.supporting_candidates",
    )
    if len(supporting) != len(set(supporting)):
        warnings.append("DUPLICATE_SUPPORTING_CANDIDATE")

    repo_need = slot["repository_need"]
    status = slot["candidate_pool_status"]

    # helpful 表示仓库证据可提供额外支持，但不是必需；允许存在 Witness，
    # 只作为语义一致性风险，不再阻断整个答案。
    if repo_need == "helpful" and groups:
        warnings.append("HELPFUL_WITH_WITNESS")
    elif repo_need in {"not_needed", "not_applicable"} and groups:
        raise ValueError(
            f"{slot_name}: repository_need={repo_need} 却存在 sufficient_witness_groups"
        )

    if repo_need == "required":
        if status == "sufficient" and not groups:
            raise ValueError(
                f"{slot_name}: required+sufficient 但 Witness 为空"
            )
        if status == "insufficient" and groups:
            raise ValueError(
                f"{slot_name}: candidate_pool_status=insufficient 却存在 Witness"
            )
        if status == "uncertain" and groups:
            warnings.append("UNCERTAIN_POOL_WITH_WITNESS")

    # 按协议，repository Evidence 非 required 时 candidate_pool_status 应 not_needed。
    if repo_need != "required" and status != "not_needed":
        warnings.append(
            f"POOL_STATUS_WITHOUT_REQUIRED_REPO:{repo_need}+{status}"
        )

    return warnings


def validate_answer_schema(
    obj: dict[str, Any],
    expected_task_id: str,
    legal_candidates: set[int],
) -> list[tuple[str, str]]:
    """
    返回 (slot/code, detail) 的机械 warning。
    """
    warnings: list[tuple[str, str]] = []

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
        raise ValueError(
            f"7 slots 不匹配 missing={missing_slots}, extra={extra_slots}"
        )

    for slot_name in CANONICAL_SLOTS:
        for w in validate_slot(slot_name, slots[slot_name], legal_candidates):
            warnings.append((slot_name, w))

    uncertainties = obj.get("uncertainties")
    if not isinstance(uncertainties, list) or any(
        not isinstance(x, str) for x in uncertainties
    ):
        raise ValueError("uncertainties 不是字符串 list")

    return warnings


def discover_questions(input_root: Path, splits: list[str]) -> list[FileRecord]:
    out: list[FileRecord] = []
    for split in splits:
        preferred = input_root / split / "md"
        base = preferred if preferred.exists() else input_root / split
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.md")):
            # 如果 fallback 扫到其他嵌套工件，可按需扩展；当前只保留 md。
            out.append(FileRecord(split=split, filename=p.name, path=p))
    return out


def discover_results(result_root: Path, splits: list[str]) -> list[FileRecord]:
    out: list[FileRecord] = []
    for split in splits:
        base = result_root / split
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.md")):
            out.append(FileRecord(split=split, filename=p.name, path=p))
    return out


def issue(
    issues: list[AuditIssue],
    severity: str,
    code: str,
    split: str,
    filename: str,
    task_id: str,
    detail: str,
    risk_points: int = 0,
) -> None:
    issues.append(
        AuditIssue(
            severity=severity,
            code=code,
            split=split,
            filename=filename,
            task_id=task_id,
            detail=detail,
            risk_points=risk_points,
        )
    )


def compute_semantic_risks(
    obj: dict[str, Any],
    split: str,
    filename: str,
    task_id: str,
    candidate_paths: dict[int, str],
    gold_changed_files: list[str],
) -> list[AuditIssue]:
    out: list[AuditIssue] = []
    slots = obj["slots"]

    # 1. fault_logic/state_flow 多成员 AND：历史上最容易出现执行路径误拼。
    for slot_name in ("fault_logic", "state_flow"):
        groups = slots[slot_name]["sufficient_witness_groups"]
        for g in groups:
            if len(g) >= 2:
                issue(
                    out, "RISK_FLAG", "CAUSAL_MULTI_AND",
                    split, filename, task_id,
                    f"{slot_name} 存在多成员 AND={g}；需人工确认每个成员都在真实执行路径且不可删除。",
                    RISK_WEIGHTS["CAUSAL_MULTI_AND"],
                )
            if len(g) >= 4:
                issue(
                    out, "RISK_FLAG", "LARGE_AND",
                    split, filename, task_id,
                    f"{slot_name} AND group 成员数={len(g)}，高概率存在 supporting evidence 混入。",
                    RISK_WEIGHTS["LARGE_AND"],
                )

    # 2. repair_scope 多 OR：可能把两个不同修复面误写成可互换。
    repair_groups = slots["repair_scope"]["sufficient_witness_groups"]
    if len(repair_groups) >= 2:
        issue(
            out, "RISK_FLAG", "REPAIR_MULTI_OR",
            split, filename, task_id,
            f"repair_scope 有 {len(repair_groups)} 个 OR alternatives；需确认每个 alternative 真能独立支持完整修复范围。",
            RISK_WEIGHTS["REPAIR_MULTI_OR"],
        )

    # 3. Issue 已 sufficient，但仍强制 repository required：不是错误，只是 Question-Sufficiency Gate 风险。
    for slot_name, slot in slots.items():
        if (
            slot["question_coverage"] == "sufficient"
            and slot["repository_need"] == "required"
        ):
            issue(
                out, "RISK_FLAG", "QUESTION_SUFFICIENT_BUT_REPO_REQUIRED",
                split, filename, task_id,
                f"{slot_name}: question_coverage=sufficient 但 repository_need=required；检查是否把相关仓库证据误升级为必需。",
                RISK_WEIGHTS["QUESTION_SUFFICIENT_BUT_REPO_REQUIRED"],
            )

    # 4. Witness saturation：同一 Candidate 横跨很多 required slots。
    witness_slot_count: Counter[int] = Counter()
    singleton_by_slot: dict[str, Optional[int]] = {}
    for slot_name, slot in slots.items():
        if slot["repository_need"] != "required":
            continue
        seen_here: set[int] = set()
        for g in slot["sufficient_witness_groups"]:
            seen_here.update(g)
        for n in seen_here:
            witness_slot_count[n] += 1

        groups = slot["sufficient_witness_groups"]
        if len(groups) == 1 and len(groups[0]) == 1:
            singleton_by_slot[slot_name] = groups[0][0]
        else:
            singleton_by_slot[slot_name] = None

    for n, count in witness_slot_count.items():
        if count >= 4:
            issue(
                out, "RISK_FLAG", "WITNESS_SATURATION",
                split, filename, task_id,
                f"Candidate {n} 被用于 {count} 个 repository-required slots；可能完全合理，也可能是 shortcut。",
                RISK_WEIGHTS["WITNESS_SATURATION"],
            )

    singleton_values = [x for x in singleton_by_slot.values() if x is not None]
    if len(singleton_values) >= 3:
        common = Counter(singleton_values).most_common(1)[0]
        if common[1] >= 3:
            issue(
                out, "RISK_FLAG", "SAME_SINGLETON_SATURATION",
                split, filename, task_id,
                f"Candidate {common[0]} 作为 singleton Witness 覆盖 {common[1]} 个 required slots；抽查是否确实一段代码独立覆盖多维语义。",
                RISK_WEIGHTS["SAME_SINGLETON_SATURATION"],
            )

    # 5. uncertainties 与完整因果充分性之间的张力。
    uncertainty_text = " ".join(obj.get("uncertainties", []))
    causal_unknown = bool(
        re.search(
            r"(根因|机制|传播|执行路径|具体原因|无法确定|不能确定|不确定|缺少|unknown|unclear)",
            uncertainty_text,
            flags=re.I,
        )
    )
    if causal_unknown:
        for slot_name in ("fault_logic", "state_flow"):
            s = slots[slot_name]
            if (
                s["repository_need"] == "required"
                and s["candidate_pool_status"] == "sufficient"
                and s["sufficient_witness_groups"]
            ):
                issue(
                    out, "RISK_FLAG", "UNCERTAINTY_CAUSAL_CONTRADICTION",
                    split, filename, task_id,
                    f"uncertainties 提到根因/机制仍不确定，但 {slot_name} 被标为 sufficient；需确认两者是否指同一机制。",
                    RISK_WEIGHTS["UNCERTAINTY_CAUSAL_CONTRADICTION"],
                )

    # 6. fault_logic/state_flow 不足，但 repair_scope 却完整 sufficient：可能是合理的局部修复，也值得看。
    fault_status = slots["fault_logic"]["candidate_pool_status"]
    repair_status = slots["repair_scope"]["candidate_pool_status"]
    if fault_status in {"insufficient", "uncertain"} and repair_status == "sufficient":
        issue(
            out, "RISK_FLAG", "POOL_STATUS_CROSS_SLOT_TENSION",
            split, filename, task_id,
            f"fault_logic={fault_status}，但 repair_scope=sufficient；检查是否在根因未知时过度确定修复面。",
            RISK_WEIGHTS["POOL_STATUS_CROSS_SLOT_TENSION"],
        )

    # 7. supporting_candidates 特别多：通常意味着候选池相关性较散。
    for slot_name, slot in slots.items():
        if len(slot["supporting_candidates"]) >= 6:
            issue(
                out, "RISK_FLAG", "MANY_SUPPORTING",
                split, filename, task_id,
                f"{slot_name} supporting_candidates={len(slot['supporting_candidates'])}；可能有轻度证据堆叠。",
                RISK_WEIGHTS["MANY_SUPPORTING"],
            )

    # 8. repair_scope Witness 与 Gold changed_files 完全不重合。
    # 这是离线 review trigger，绝不是自动语义失败：Gold 不能作为最终 Witness。
    if gold_changed_files and repair_groups:
        witness_nums = sorted({n for g in repair_groups for n in g})
        witness_paths = {
            candidate_paths[n]
            for n in witness_nums
            if n in candidate_paths
        }
        gold = set(gold_changed_files)
        if witness_paths and witness_paths.isdisjoint(gold):
            issue(
                out, "RISK_FLAG", "REPAIR_GOLD_PATH_DIVERGENCE",
                split, filename, task_id,
                "repair_scope Witness 路径与 Gold changed_files 完全不重合；"
                f"Witness={sorted(witness_paths)}, Gold={sorted(gold)}。仅作为离线复核提示。",
                RISK_WEIGHTS["REPAIR_GOLD_PATH_DIVERGENCE"],
            )

    return out


def main() -> int:
    args = parse_args()
    report_root: Path = args.report_root
    report_root.mkdir(parents=True, exist_ok=True)

    print(f"[audit] input_root={args.input_root}")
    print(f"[audit] result_root={args.result_root}")
    print(f"[audit] report_root={report_root}")
    print("[audit] 正在扫描 question/result 文件树...")
    questions = discover_questions(args.input_root, args.splits)
    results = discover_results(args.result_root, args.splits)
    print(f"[audit] 扫描完成: questions={len(questions)}, results={len(results)}")

    issues: list[AuditIssue] = []

    # -------------------- Build question index --------------------
    q_by_split_name: dict[tuple[str, str], FileRecord] = {}
    q_by_filename: dict[str, list[FileRecord]] = defaultdict(list)
    q_by_task: dict[str, list[FileRecord]] = defaultdict(list)

    q_candidate_cache: dict[Path, tuple[set[int], dict[int, str]]] = {}
    q_gold_cache: dict[Path, list[str]] = {}

    print("[audit] 正在解析题目 task_id / Candidate / Gold hints...")
    for q_idx, q in enumerate(questions, 1):
        key = (q.split, q.filename)
        if key in q_by_split_name:
            issue(
                issues, "HARD_ERROR", "DUPLICATE_QUESTION_PATH_KEY",
                q.split, q.filename, "",
                f"同一 split+filename 出现多个题目文件：{q_by_split_name[key].path} / {q.path}",
            )
        q_by_split_name[key] = q
        q_by_filename[q.filename].append(q)

        try:
            text = q.path.read_text(encoding="utf-8")
            q.task_id = extract_question_task_id(text)
            q_by_task[q.task_id].append(q)
            q_candidate_cache[q.path] = extract_candidates(text)
            q_gold_cache[q.path] = extract_gold_changed_files(text)
        except Exception as exc:
            issue(
                issues, "HARD_ERROR", "QUESTION_PARSE_FAILED",
                q.split, q.filename, q.task_id or "",
                f"{q.path}: {type(exc).__name__}: {exc}",
            )

        if args.progress_every and q_idx % args.progress_every == 0:
            print(f"[audit] questions {q_idx:,}/{len(questions):,} ({q.split})")

    # 同一个 task_id 对应多个题目文件通常不应该发生。
    for task_id, recs in q_by_task.items():
        unique_paths = {str(r.path) for r in recs}
        if len(unique_paths) > 1:
            for r in recs:
                issue(
                    issues, "HARD_ERROR", "DUPLICATE_QUESTION_TASK_ID",
                    r.split, r.filename, task_id,
                    f"task_id={task_id} 对应多个题目：{sorted(unique_paths)}",
                )

    # -------------------- Result path checks --------------------
    r_by_split_name: dict[tuple[str, str], list[FileRecord]] = defaultdict(list)
    r_by_filename: dict[str, list[FileRecord]] = defaultdict(list)
    for r in results:
        r_by_split_name[(r.split, r.filename)].append(r)
        r_by_filename[r.filename].append(r)

    for key, recs in r_by_split_name.items():
        if len(recs) > 1:
            for r in recs:
                issue(
                    issues, "HARD_ERROR", "DUPLICATE_RESULT_FILE",
                    r.split, r.filename, "",
                    f"同一 split+filename 存在多个答案文件：{[str(x.path) for x in recs]}",
                )

    # 预期每个题目都应该有同 split、同 filename 的答案。
    for key, q in q_by_split_name.items():
        if key not in r_by_split_name:
            # 如果同名答案在别的 split，则这是“放错地方”；否则是真缺失。
            elsewhere = r_by_filename.get(q.filename, [])
            if elsewhere:
                issue(
                    issues, "HARD_ERROR", "RESULT_MISPLACED_SPLIT",
                    q.split, q.filename, q.task_id or "",
                    f"预期答案在 {q.split}，但同名结果出现在："
                    + ", ".join(str(x.path) for x in elsewhere),
                )
            else:
                if args.allow_missing_results:
                    issue(
                        issues, "EXCLUDED", "MISSING_RESULT_IGNORED",
                        q.split, q.filename, q.task_id or "",
                        f"没有找到对应答案；按 --allow-missing-results 明确排除，不作为 HARD_ERROR；预期 key={key}",
                    )
                else:
                    issue(
                        issues, "HARD_ERROR", "MISSING_RESULT",
                        q.split, q.filename, q.task_id or "",
                        f"没有找到对应答案；预期 key={key}",
                    )

    # 结果若同 split 同名找不到题目：orphan / wrong split。
    for r in results:
        key = (r.split, r.filename)
        if key not in q_by_split_name:
            same_name_q = q_by_filename.get(r.filename, [])
            if same_name_q:
                issue(
                    issues, "HARD_ERROR", "ORPHAN_RESULT_WRONG_SPLIT",
                    r.split, r.filename, "",
                    "结果文件所在 split 与题目不一致；题目实际在："
                    + ", ".join(f"{q.split}:{q.path}" for q in same_name_q),
                )
            else:
                issue(
                    issues, "HARD_ERROR", "ORPHAN_RESULT_NO_QUESTION",
                    r.split, r.filename, "",
                    f"结果找不到任何同名题目：{r.path}",
                )

    # -------------------- Per-answer deep checks --------------------
    per_answer_rows: list[dict[str, Any]] = []
    known_task_ids = set(q_by_task)
    print("[audit] 正在逐条解析并校验 Strong-Teacher 答案...")

    for r_idx, r in enumerate(results, 1):
        row = {
            "split": r.split,
            "filename": r.filename,
            "result_path": str(r.path),
            "question_path": "",
            "question_task_id": "",
            "answer_task_id": "",
            "effective_answer_task_id": "",
            "safe_normalization_count": 0,
            "hard_error_count": 0,
            "risk_score": 0,
            "risk_flag_count": 0,
            "status": "",
        }

        before_issue_count = len(issues)
        q = q_by_split_name.get((r.split, r.filename))

        if not r.path.exists():
            continue
        if r.path.stat().st_size == 0:
            issue(
                issues, "HARD_ERROR", "ZERO_BYTE_RESULT",
                r.split, r.filename, "",
                f"答案文件是 0 byte：{r.path}",
            )
            row["status"] = "HARD_ERROR"
            per_answer_rows.append(row)
            continue

        if q is None:
            row["status"] = "HARD_ERROR"
            per_answer_rows.append(row)
            continue

        row["question_path"] = str(q.path)
        row["question_task_id"] = q.task_id or ""

        # 文件名自身是否包含题目 task_id。
        ids_in_name = filename_task_ids(r.filename)
        if q.task_id and q.task_id not in ids_in_name:
            issue(
                issues, "HARD_ERROR", "FILENAME_TASK_ID_MISMATCH",
                r.split, r.filename, q.task_id,
                f"文件名 task ids={ids_in_name}，但题目 task_id={q.task_id}",
            )

        try:
            answer_text = r.path.read_text(encoding="utf-8")
            obj, parse_warnings = parse_answer_with_safe_salvage(answer_text)
            for warning in parse_warnings:
                issue(
                    issues, "RISK_FLAG", warning,
                    r.split, r.filename, q.task_id or "",
                    "答案 JSON 通过严格安全 salvage 解析；原文件未被修改。",
                    RISK_WEIGHTS.get(warning, 1),
                )
                row["safe_normalization_count"] += 1

            answer_task = obj.get("task_id")
            row["answer_task_id"] = answer_task if isinstance(answer_task, str) else ""

            # task_id mismatch 分两类：
            # 1) 可由当前 filename 机械证明只是 alias -> 内存 normalize + RISK；
            # 2) 其它 mismatch -> HARD_ERROR，但仍在临时对象中改成 expected 继续做其余 schema 检查，
            #    避免同一 task_id 问题再重复产生 ANSWER_VALIDATE_FAILED。
            if isinstance(answer_task, str) and q.task_id and answer_task != q.task_id:
                if safe_task_id_alias(
                    answer_task, q.task_id, r.filename, known_task_ids
                ):
                    issue(
                        issues, "RISK_FLAG", "TASK_ID_SAFE_NORMALIZED",
                        r.split, r.filename, q.task_id,
                        f"answer.task_id={answer_task!r} 可由当前 filename 机械证明为当前题 alias；"
                        f"审计时临时规范化为 {q.task_id!r}，原文件未修改。",
                        RISK_WEIGHTS["TASK_ID_SAFE_NORMALIZED"],
                    )
                    obj["task_id"] = q.task_id
                    row["safe_normalization_count"] += 1
                else:
                    actual_q = q_by_task.get(answer_task, [])
                    detail = f"当前题目={q.task_id}, 答案={answer_task}。"
                    if actual_q:
                        detail += "该答案 task_id 对应题目：" + ", ".join(
                            f"{x.split}:{x.path}" for x in actual_q
                        )
                    issue(
                        issues, "HARD_ERROR", "ANSWER_WRONG_TASK_ID",
                        r.split, r.filename, q.task_id, detail,
                    )
                    # 仅用于继续审计其它字段；不会写回。
                    obj["task_id"] = q.task_id

            row["effective_answer_task_id"] = (
                obj.get("task_id") if isinstance(obj.get("task_id"), str) else ""
            )

            legal, candidate_paths = q_candidate_cache[q.path]

            dropped_findings, finding_details = sanitize_additional_findings(
                obj, legal
            )
            if dropped_findings:
                preview = "; ".join(finding_details[:5])
                if len(finding_details) > 5:
                    preview += f"; ... 共 {len(finding_details)} 项"
                issue(
                    issues, "RISK_FLAG", "ADDITIONAL_FINDINGS_SANITIZED",
                    r.split, r.filename, q.task_id or "",
                    f"保守丢弃 malformed additional_findings={dropped_findings}；{preview}",
                    RISK_WEIGHTS["ADDITIONAL_FINDINGS_SANITIZED"],
                )
                row["safe_normalization_count"] += dropped_findings

            schema_warnings = validate_answer_schema(
                obj=obj,
                expected_task_id=q.task_id or "",
                legal_candidates=legal,
            )

            for slot_name, warning in schema_warnings:
                code = warning.split(":")[0]
                # Superset / duplicate group 是协议上可机械证明的结构异常，按 HARD_ERROR；
                # 其它状态张力作为 RISK。
                if (
                    warning.startswith("NONMINIMAL_SUPERSET_GROUP")
                    or warning == "DUPLICATE_OR_GROUP"
                ):
                    issue(
                        issues, "HARD_ERROR", code,
                        r.split, r.filename, q.task_id or "",
                        f"{slot_name}: {warning}",
                    )
                else:
                    issue(
                        issues, "RISK_FLAG", code,
                        r.split, r.filename, q.task_id or "",
                        f"{slot_name}: {warning}",
                        RISK_WEIGHTS.get(code, 1),
                    )

            semantic_risks = compute_semantic_risks(
                obj=obj,
                split=r.split,
                filename=r.filename,
                task_id=q.task_id or "",
                candidate_paths=candidate_paths,
                gold_changed_files=q_gold_cache.get(q.path, []),
            )
            issues.extend(semantic_risks)

        except Exception as exc:
            issue(
                issues, "HARD_ERROR", "ANSWER_VALIDATE_FAILED",
                r.split, r.filename, q.task_id or "",
                f"{type(exc).__name__}: {exc}",
            )

        # 汇总当前文件的 issues。
        file_issues = [
            x for x in issues[before_issue_count:]
            if x.split == r.split and x.filename == r.filename
        ]
        hard = [x for x in file_issues if x.severity == "HARD_ERROR"]
        risks = [x for x in file_issues if x.severity == "RISK_FLAG"]

        row["hard_error_count"] = len(hard)
        row["risk_flag_count"] = len(risks)
        row["risk_score"] = sum(x.risk_points for x in risks)
        if hard:
            row["status"] = "HARD_ERROR"
        elif row["risk_score"] >= 6:
            row["status"] = "HIGH_RISK_REVIEW"
        elif row["risk_score"] >= 2:
            row["status"] = "MEDIUM_RISK_REVIEW"
        else:
            row["status"] = "LOW_RISK"
        per_answer_rows.append(row)

        if args.progress_every and r_idx % args.progress_every == 0:
            print(
                f"[audit] answers {r_idx:,}/{len(results):,} ({r.split}) "
                f"| issues={len(issues):,}"
            )

    # -------------------- Reports --------------------
    issue_rows = [asdict(x) for x in issues]
    write_csv(
        report_root / "audit_issues.csv",
        issue_rows,
        [
            "severity", "code", "split", "filename", "task_id",
            "detail", "risk_points",
        ],
    )

    write_csv(
        report_root / "per_answer_status.csv",
        per_answer_rows,
        [
            "split", "filename", "question_task_id", "answer_task_id",
            "effective_answer_task_id", "safe_normalization_count",
            "status", "hard_error_count", "risk_score", "risk_flag_count",
            "question_path", "result_path",
        ],
    )

    # Semantic review queue：排除 hard error 后按 risk score 降序。
    clean_rows = [
        r for r in per_answer_rows
        if r["hard_error_count"] == 0
    ]
    review_rows = sorted(
        clean_rows,
        key=lambda x: (-int(x["risk_score"]), x["split"], x["filename"]),
    )[: max(0, args.review_top)]

    write_csv(
        report_root / "semantic_review_queue.csv",
        review_rows,
        [
            "split", "filename", "question_task_id", "answer_task_id",
            "effective_answer_task_id", "safe_normalization_count",
            "status", "risk_score", "risk_flag_count",
            "question_path", "result_path",
        ],
    )

    # 随机低风险抽样：保证即便 heuristics 没 flag，也检查 teacher 的普通样本质量。
    rng = random.Random(args.seed)
    low = [
        r for r in clean_rows
        if int(r["risk_score"]) <= 1
    ]
    rng.shuffle(low)
    low_sample = low[: max(0, args.random_low_sample)]
    write_csv(
        report_root / "random_low_risk_sample.csv",
        low_sample,
        [
            "split", "filename", "question_task_id", "answer_task_id",
            "effective_answer_task_id", "safe_normalization_count",
            "status", "risk_score", "risk_flag_count",
            "question_path", "result_path",
        ],
    )

    severity_counts = Counter(x.severity for x in issues)
    code_counts = Counter(x.code for x in issues)
    status_counts = Counter(r["status"] for r in per_answer_rows)
    split_status = defaultdict(Counter)
    for r in per_answer_rows:
        split_status[r["split"]][r["status"]] += 1

    summary = {
        "audit_version": "1.0.2",
        "safe_recovery_policy": {
            "writes_source_files": False,
            "task_id_alias_normalization": True,
            "malformed_additional_findings_drop": True,
            "empty_reason_is_risk": True,
            "helpful_with_witness_is_risk": True,
            "uncertain_pool_with_witness_is_risk": True,
            "strict_json_salvage": True,
            "allow_missing_results": bool(args.allow_missing_results),
        },
        "input_root": str(args.input_root),
        "result_root": str(args.result_root),
        "report_root": str(report_root),
        "splits": args.splits,
        "question_file_count": len(questions),
        "result_file_count": len(results),
        "per_answer_count": len(per_answer_rows),
        "issue_severity_counts": dict(severity_counts),
        "issue_code_counts": dict(code_counts.most_common()),
        "answer_status_counts": dict(status_counts),
        "split_status_counts": {
            split: dict(counts)
            for split, counts in split_status.items()
        },
        "review_queue_count": len(review_rows),
        "random_low_risk_sample_count": len(low_sample),
        "interpretation": {
            "HARD_ERROR": "机械可证明异常，应全部处理到 0。",
            "HIGH_RISK_REVIEW": "不是自动判错；优先人工审查 Witness / 执行路径 / AND-OR。",
            "MEDIUM_RISK_REVIEW": "抽查。",
            "LOW_RISK": "形式一致；仍建议随机抽样检查语义。",
            "EXCLUDED": "按显式策略排除，不进入已有答案审计/训练候选，不作为 HARD_ERROR。",
        },
    }

    (report_root / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 有 HARD_ERROR 时返回 2，方便 CI / PowerShell 自动判断。
    return 2 if severity_counts.get("HARD_ERROR", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
