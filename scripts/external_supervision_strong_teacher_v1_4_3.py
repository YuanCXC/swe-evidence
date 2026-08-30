#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Strong-Teacher External Supervision Protocol v1
（统一强教师外部监督协议 v1）

建议位置：
    scripts/external_supervision_strong_teacher.py

===============================================================================
一、为什么有这个脚本
===============================================================================

旧 External Supervision Bridge 为小模型 / 不稳定 API 设计：

    Stage 1 Requirement Decision（候选盲化）
        ↓
    Stage 2 Targeted Witness Selection
        ↓
    Program-built Supervision

现在 Teacher 改为能力更强的网页版大模型后，不再需要通过两轮物理隔离、
2-of-3、自报 confidence 等方式约束模型。

新版改成：

    refinement_requests.jsonl 中的完整 base_user_prompt
    （Issue + Original Supervision + Candidate Pool + Gold Change Hints）
        ↓
    网页版 Strong Teacher 一次性语义判断
        ↓
    Unified Result（丰富、宽松的结构化语义）
        ↓
    Identity / Integrity Gate（严格）
        ↓
    Semantic Canonicalization（保留原输出，尽量不丢信息）
        ↓
    Program-built Supervision（复用 v1.9.2.1）
        ↓
    Core v1.7 Deterministic Verification
        ↓
    VERIFIED / NEEDS_MORE / BLOCKED

===============================================================================
二、核心设计原则
===============================================================================

1. Teacher 表达层：宽松

   不再强制：
       - question_satisfied 时绝不能指出 Candidate；
       - not_required 时绝不能指出 Candidate；
       - uncertain 时必须完全不输出相关 Candidate；
       - 只有 repository_required 才允许“看”Candidate；
       - confidence threshold；
       - 2-of-3 self-consistency。

   Strong Teacher 可以同时表达：
       - 某个语义维度是否必要；
       - Question 已覆盖多少；
       - Repository Evidence 是 required / helpful / not_needed；
       - 哪些 Candidate 构成“充分 Witness”；
       - 哪些 Candidate 只是 supporting / partial evidence；
       - 7 个标准槽位之外的重要发现。

2. Identity / Integrity 层：严格

   无论模型多强，下列事实不能放宽：
       - task_id 必须绑定真实任务；
       - Candidate Number 必须落在 1..N；
       - Candidate Number 的顺序由 refinement_requests / merge context 锁定；
       - 最终 Witness 只能映射到真实 pre-fix Candidate Evidence ID；
       - V2.10 不修改，只写 sidecar。

3. Training Promotion 层：保守

   Teacher 输出不会直接变成训练标签。
   最终仍然：
       Program-built Supervision
           +
       Core v1.7 Verification

   并且本脚本始终：
       training_eligible = false

   VERIFIED 只表示：
       当前统一 Teacher 判断可以无歧义映射到程序监督，且 Core accepted。

   它不等价于数学意义上的 Semantic Truth（语义真值），
   更不表示“保证修复成功”。

===============================================================================
三、7 个 Canonical Dimensions（标准语义维度）
===============================================================================

固定接口仍保留：

    fault_location
    fault_logic
    dependency_context
    state_flow
    behavior_constraint
    repair_scope
    validation_constraint

但它们只是 canonical assessment dimensions（标准评估维度），
不是声称所有软件修复语义只能属于这 7 类。

Teacher 可以把无法自然归入 7 槽的重要信息写入：

    additional_findings

这些信息保留在 sidecar，不会被程序擅自转换成新 Obligation。

===============================================================================
四、单槽位新版语义
===============================================================================

每个槽位输出：

    applicability:
        required
        not_required
        uncertain

    question_coverage:
        sufficient
        partial
        none
        uncertain
        not_applicable

    repository_need:
        required
        helpful
        not_needed
        uncertain
        not_applicable

    candidate_pool_status:
        sufficient
        insufficient
        uncertain
        not_needed

    sufficient_witness_groups:
        Candidate Number OR-of-AND

        [[2, 5]]
            = 2 AND 5

        [[2], [5, 9]]
            = 2 OR (5 AND 9)

    supporting_candidates:
        与该槽位相关、有帮助，但不声明它们单独/联合达到充分性的 Candidate Number。

    reason:
        简洁语义理由。

重要：
    required / question coverage / repository need 是不同维度，不能再压缩成
    旧的一个 decision enum。

例如：

    behavior_constraint:
        applicability = required
        question_coverage = sufficient
        repository_need = helpful
        supporting_candidates = [7]

表示：
    该行为约束对修复重要；Issue 已经说清楚；仓库 Candidate 7 仍可作为辅助上下文，
    但 Candidate 7 不应因此被强制塞进 mandatory certificate。

===============================================================================
五、命令
===============================================================================

A. 导出一次性 Strong Teacher Markdown：

    python scripts/external_supervision_strong_teacher.py export-unified `
      --requests data/.supervision_refinement/.../refinement_requests.jsonl `
      --output-dir data/upstream/external_supervision/strong_teacher `
      --batch-size 1

B. 网页模型返回 JSON 后做宽松语义 + 严格身份校验：

    python scripts/external_supervision_strong_teacher.py validate-unified `
      --requests data/.supervision_refinement/.../refinement_requests.jsonl `
      --results data/upstream/external_supervision/strong_teacher/results.json `
      --output data/upstream/external_supervision/strong_teacher/results.normalized.jsonl `
      --errors data/upstream/external_supervision/strong_teacher/results.errors.jsonl

C. 先用 external_supervision_merge.py prepare-context 建立机器上下文 / Candidate Identity Lock：

    python scripts/external_supervision_merge.py prepare-context `
      --requests data/.supervision_refinement/.../refinement_requests.jsonl `
      --split validation `
      --output data/upstream/external_supervision/merge_context.jsonl

D. 一次性结果接回 v1.9.2.1 + Core v1.7：

    python scripts/external_supervision_strong_teacher.py build-refinement `
      --context data/upstream/external_supervision/merge_context.jsonl `
      --results data/upstream/external_supervision/strong_teacher/results.normalized.jsonl `
      --output-dir data/upstream/external_supervision/strong_teacher/refinement

E. 从冻结 V2.10 直接导出完整 20,864 个 Strong-Teacher Markdown：

    python scripts/external_supervision_strong_teacher.py export-all `
      --runner scripts/refine_supervision_with_llm.py `
      --dataset-dir data/upstream/unified_swe_dataset_v2_10 `
      --output-dir data/upstream/external_supervision/strong_teacher_v1_3_all `
      --tasks-per-md 1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from tqdm import tqdm
except ImportError as exc:
    raise RuntimeError(
        "缺少 tqdm（进度条依赖）。请执行：python -m pip install -U tqdm"
    ) from exc


SCRIPT_VERSION = "1.4.3"
PROTOCOL_VERSION = "unified-strong-teacher-v1.3"

SLOT_TYPES = (
    "fault_location",
    "fault_logic",
    "dependency_context",
    "state_flow",
    "behavior_constraint",
    "repair_scope",
    "validation_constraint",
)

APPLICABILITY_VALUES = {
    "required",
    "not_required",
    "uncertain",
}

QUESTION_COVERAGE_VALUES = {
    "sufficient",
    "partial",
    "none",
    "uncertain",
    "not_applicable",
}

REPOSITORY_NEED_VALUES = {
    "required",
    "helpful",
    "not_needed",
    "uncertain",
    "not_applicable",
}

CANDIDATE_POOL_STATUS_VALUES = {
    "sufficient",
    "insufficient",
    "uncertain",
    "not_needed",
}

FINAL_STATUSES = {
    "VERIFIED",
    "NEEDS_MORE",
    "BLOCKED",
}


# ============================================================================
# 1. 通用 I/O
# ============================================================================


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


def atomic_write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def read_external_results(path: Path) -> list[dict[str, Any]]:
    """
    网页模型常见三种返回：
        1. 一个 pretty-printed JSON object；
        2. 一个 JSON array；
        3. JSONL。

    三种都接受。

    这里不做 Markdown code-fence 自动剥离：
        Prompt 已明确要求只返回 JSON；
        如果模型仍输出 Markdown，应由用户复制 JSON 内容，或在结果文件中保留纯 JSON。

    原因：自动从自然语言中“猜 JSON”容易把多个候选答案拼错。
    """
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return read_jsonl(path)

    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        rows: list[dict[str, Any]] = []
        for index, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"{path} JSON array 第 {index} 项必须是 object"
                )
            rows.append(item)
        return rows

    raise ValueError(f"{path} 必须是 JSON object / array / JSONL")


def unique_requests(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            raise ValueError("requests 存在缺少 task_id 的记录")
        if task_id in result:
            raise ValueError(f"requests task_id 重复：{task_id}")
        result[task_id] = dict(row)
    return result


def request_candidate_ids(request: Mapping[str, Any]) -> list[str]:
    """
    与 external_supervision_merge.py 的 Candidate Identity Lock 保持一致。

    Candidate Number 是“展示顺序中的位置”，不是独立 ID。
    所以后续所有网页结果都必须绑定到这里的顺序。
    """
    diagnostics = request.get("candidate_diagnostics") or {}
    ids = diagnostics.get("selected_evidence_ids")
    if ids is None:
        ids = diagnostics.get("teacher_display_evidence_ids")

    if not isinstance(ids, list) or not ids:
        raise ValueError(
            f"task={request.get('task_id')} 缺少 "
            "candidate_diagnostics.selected_evidence_ids；"
            "无法建立 Candidate Number 身份合同"
        )

    normalized = [str(item) for item in ids]
    if any(not item for item in normalized):
        raise ValueError(
            f"task={request.get('task_id')} Candidate Evidence ID 存在空值"
        )
    if len(normalized) != len(set(normalized)):
        raise ValueError(
            f"task={request.get('task_id')} Candidate Evidence ID 重复"
        )
    return normalized


# ============================================================================
# 2. Unified Strong-Teacher Prompt
# ============================================================================


STRONG_TEACHER_INSTRUCTIONS = r"""
[UNIFIED STRONG-TEACHER PROTOCOL]

你是 Offline Strong Teacher（离线强教师）。
你的目标不是生成补丁，而是判断：
对于当前 SWE 软件修复任务，什么 pre-fix repository context（修复前仓库上下文）
足以支持下游模型进行故障定位、原因分析和补丁规划。

请综合阅读：
- Issue / Question
- Original Supervision（它可能有错，只是参考）
- Candidate Evidence Pool
- Gold Change Hints（仅用于离线理解真实修改方向）

重要边界：
1. Gold/Test Patch 或 future/post-fix code 只能帮助你理解任务，不能作为最终 Witness。
2. 最终 Witness 只能使用 Candidate Number；Candidate Pool 中的 Candidate 是可绑定的 pre-fix Evidence。
3. Original Supervision 不是语义真值。发现它有错时，应按你的语义判断回答，不要为了保持一致而迁就旧标签。
4. 不需要遵守旧的小模型 Stage-1/Stage-2 强耦合规则。请先完整理解任务，再一次性给出判断。
5. 不要因为 Candidate Pool “刚好有某段代码”，就把本来已经由 Issue 说明清楚的语义强行判成 repository required。
6. 反过来，如果 Issue 只给了 symbol 名字或现象，但真正定位/机制/状态传播必须看仓库，请明确 repository_need=required。
7. 对 Feature Addition（新增功能）：future symbol 在 pre-fix 中不存在是正常现象，不能仅因此判 Candidate Pool insufficient。应检查已有 integration point、analogous implementation、export/registration convention、dependency/state context 等。
8. 不确定就明确写 uncertain，不要为了“每项都有答案”而猜。

强模型 Grounding / Self-Check（证据落地与自检）：

9. 严格区分三类位置，不要混为一谈：
   - Crash Site（崩溃点）：traceback / assertion 实际失败的位置；
   - Root Cause Site（根因点）：导致错误状态形成的实现位置；
   - Repair Site（修复点）：Gold Change Hints 明确显示实际修改的区域。
   Issue traceback 中出现某个函数，只能证明它在失败调用链上；
   除非 Gold hunk header 或其他给定信息明确支持，否则不要说“Gold Hint 指明该函数被修改”。

10. Gold Change Hints 必须精确归因。
    只能把 changed_files / hunk_headers 明确支持的文件、symbol 或相邻代码区域称为 Gold 修改方向。
    同一文件中出现的其他 Candidate，不得仅因为“也在这个文件”就说成 Gold 修改点。
    Gold Hints 可帮助你判断 Candidate 覆盖和修复方向，但仍不能作为 Witness。

11. 在判断 candidate_pool_status=insufficient 前，必须重新扫描全部 Candidate，尤其检查：
    - Gold hunk 指向的 pre-fix 文件 / symbol / 相邻区域是否已在 Candidate Pool；
    - traceback 中直接相关的方法是否已在 Candidate Pool；
    - copy / construct / transform / registration / caller 等相关上下文是否已有候选。
    不得因为漏读 Candidate 而判 insufficient。

12. sufficient_witness_groups 表示严格的 OR-of-AND 充分性：
    - 每一个外层 OR group 都必须“独立地”足以满足该 slot；
    - 如果 Candidate A 与 B 分别只覆盖互补信息，必须写 [[A, B]]，不能写 [[A], [B]]；
    - 如果 reason 中称某 Candidate 为“必要 / 必须 / indispensable”，它应出现在所有依赖该信息的充分组合中；
    - 只是有帮助、补充背景、加强解释的 Candidate 应放 supporting_candidates，不要塞进充分组。

13. reason 与 sufficient_witness_groups 必须自洽。
    如果 reason 说“还必须结合 Candidate X 才能理解该机制”，但 X 不在任何相应充分组合中，
    则应修改 witness group、降低 candidate_pool_status，或把 reason 改为“辅助而非必要”。

14. 不要把“相关”写成“因果已证实”。
    Candidate 显示某方法保留/复制/重定位某状态，只能证明该方法的实际行为；
    若没有证据证明状态正是在该步骤丢失，不要直接断言“故障本质就是该步骤没有传播状态”。
    可以写成“该链路需要检查 / 与状态传播相关”，并把尚未确定的环节写入 uncertainties。

15. 对 behavior_constraint / validation_constraint，区分“目标行为”与“实现策略”。
    如果 Issue 已明确“什么输入下不应崩溃 / 应返回什么 / 如何复现与验证”，
    即使具体应该修改哪个函数仍未知，Question 对行为或核心验证约束仍可能是 sufficient。
    “不知道怎么实现”不能自动把 behavior_constraint 判成 partial。

16. Enum Compliance（枚举字段遵守）必须逐字段检查。
    输出前必须确认每个枚举值严格属于该字段允许集合，不得把其他字段的概念串过来。
    特别注意：
    - applicability 只能是 required / not_required / uncertain；不存在 helpful；
    - helpful 只属于 repository_need；
    - question_coverage / candidate_pool_status 也只能使用各自定义的枚举。
    如果你想表达“这个维度不是必要条件，但仓库上下文有帮助”，应分别表达为
    applicability=not_required 与 repository_need=helpful，而不是创造 applicability=helpful。

17. Claim–Uncertainty Consistency（断言与不确定性一致性）。
    已确认事实、合理推断、尚未确定的机制必须分开表述。
    如果 uncertainties 中明确写了“exact mechanism / precise propagation gap / 具体根因无法从当前候选确定”，
    则 overall_assessment、slot reason、additional_findings 中不得同时把这个尚未确定的机制写成已证实事实。
    例如：可以确定“path_to 遇到缺少 pos_marker 的 segment 后触发断言”，
    但如果不能确定“这个 marker 究竟在哪一步丢失”，就不要写成“故障本质就是 copy/apply_fixes 未传播 marker”。
    对 fault_logic 等槽位：
    - 若当前 Candidate 明确足以支撑该槽位所要求的机制，可用 candidate_pool_status=sufficient；
    - 若确定缺少必要 pre-fix context，才用 insufficient；
    - 若只是无法可靠判断现有证据是否已经充分，应使用 uncertain。
    不要一边声明“根因机制无法重建”，一边又用只覆盖 crash mechanism 的 Witness 宣称完整 fault_logic sufficient。

18. Explanation Language（解释字段语言）。
    为便于人工审计，所有“解释 / 推理结论”字段统一使用简体中文：
    - overall_assessment；
    - 每个 slot 的 reason；
    - additional_findings[].description；
    - additional_findings[].reason；
    - uncertainties[] 中的每一项。
    机器字段和值不要翻译：task_id、Candidate Number、文件路径、symbol、代码标识符以及
    required / not_required / uncertain / sufficient / partial / none / not_applicable /
    helpful / not_needed / insufficient 等枚举必须保持规定的英文形式。
    技术术语可保留英文标识符并用中文解释。只输出结论性理由，不需要输出内部逐步思考过程。

19. Behavior Isolation Check（行为约束隔离检查）。
    判断 behavior_constraint 时，只回答：Issue / Question 是否已经把“外部期望行为是什么”说明清楚。
    不要把以下实现层问题混入 question_coverage：
    - 应该修改哪个函数；
    - 内部状态或不变量应如何维护；
    - patch 应采用哪种实现策略；
    - 为避免回归还需要检查哪些内部代码。
    如果 Issue 已明确“什么输入下不应崩溃 / 应返回什么 / 应产生什么可观察结果”，
    则 behavior_constraint 的 question_coverage 通常可以是 sufficient；
    实现级约束可放 fault_logic、state_flow、repair_scope、validation_constraint 或 additional_findings。
    不要因为“不知道怎么实现”而把已明确的外部行为从 sufficient 降成 partial。

20. Root-Cause Evidence Check（根因证据检查）。
    如果你的根因结论依赖某个关键内部函数、状态转换或 helper 的具体行为，
    但该实现没有出现在 Candidate Pool 中，就不能把该行为写成已证实事实。
    例如 Candidate 只显示调用了 _position_segments()，但没有提供 _position_segments() 的实现时：
    - 可以说“证据指向 apply_fixes / repositioning 链路，需要检查该环节”；
    - 可以说“Gold Hints 表明修复与该调用区域相关”；
    - 不能仅据此断言“_position_segments 没有给 f.edit 设置 pos_marker”或
      “已确定 apply_fixes 在这里丢失 pos_marker”。
    若 crash mechanism 已确认、但 root-cause mechanism 仍依赖缺失实现，应把两者分开：
    已确认部分写入 reason，尚未确认部分写入 uncertainties；并重新检查
    fault_logic / state_flow 的 candidate_pool_status 是否应为 sufficient、uncertain 或 insufficient。

21. 最终输出前做一次静默自检（不要把自检过程输出到 JSON）：
    - 是否把 traceback symbol 错当成 Gold repair site？
    - 是否存在 Candidate 明明在池中却被说成缺失？
    - 每个 OR alternative 是否真的可以单独充分？
    - reason 中的“必要” Candidate 是否与 witness group 一致？
    - 是否把推测写成了已证实根因？
    - uncertainties 是否与 candidate_pool_status / reason 互相矛盾？
    - 所有枚举值是否严格属于对应字段，尤其 applicability 是否误用了 helpful？
    - 若 uncertainties 声明某机制仍未知，其他解释字段是否错误地把同一机制写成确定事实？
    - behavior_constraint 是否只按 Issue 的外部期望行为判断，而没有被实现细节错误降级？
    - 根因结论是否依赖 Candidate Pool 中未提供实现的关键函数；若依赖，是否错误写成确定事实？
    - overall_assessment / reason / additional_findings / uncertainties 是否全部使用简体中文？

7 个 canonical dimensions（标准维度）：
- fault_location
- fault_logic
- dependency_context
- state_flow
- behavior_constraint
- repair_scope
- validation_constraint

这 7 个维度是最终标准化接口，不是完整世界模型。
如果发现无法自然归入它们、但对修复上下文很重要的信息，写入 additional_findings，不要硬塞进错误槽位。

每个槽位请分别判断：

A. applicability
- required：这个语义维度对达到修复上下文充分性是必要的
- not_required：当前任务不需要这个维度
- uncertain：无法可靠判断

B. question_coverage
- sufficient：Issue / Question 本身已经把该维度需要知道的内容说清楚
- partial：Issue 只覆盖一部分
- none：Issue 没有提供该维度所需语义
- uncertain：无法可靠判断
- not_applicable：该维度本身不适用

C. repository_need
- required：为了达到该维度的充分性，必须使用 pre-fix repository Evidence
- helpful：仓库 Evidence 有帮助，但不是达到充分性的必要条件
- not_needed：不需要仓库 Evidence
- uncertain：无法可靠判断
- not_applicable：该维度不适用

D. candidate_pool_status
- sufficient：如果 repository Evidence 是 required，当前 Candidate Pool 中存在充分 Witness
- insufficient：repository Evidence 是 required，但当前 Candidate Pool 缺少达到充分性所需的 pre-fix context
- uncertain：无法可靠判断当前 Candidate Pool 是否足够
- not_needed：该槽位不需要 Candidate Pool 来满足必要语义

E. sufficient_witness_groups
只放“你认为足以满足该槽位 repository requirement”的 Candidate Number OR-of-AND。

例：
[[2, 5]] = Candidate 2 AND Candidate 5
[[2], [5, 9]] = Candidate 2 OR (Candidate 5 AND Candidate 9)

不要把“只是相关/有帮助”的 Candidate 塞进 sufficient_witness_groups。

F. supporting_candidates
放相关、有帮助、能解释部分机制或提供额外修复上下文，但你不声明它们构成充分 Witness 的 Candidate Number。
即使 question_coverage=sufficient 或 applicability=not_required，也允许指出 supporting candidates；
这不会自动让它们进入 mandatory certificate。

G. reason
用简洁但具体的中文语义理由说明判断。

语言要求：
- overall_assessment、所有 reason、additional_findings 的 description/reason、uncertainties 必须用简体中文；
- task_id、Candidate Number、path、symbol、代码标识符和所有枚举值保持英文/原样。

不要输出 confidence 数字。不要为了符合旧规则而改变你的语义判断。
""".strip()


OUTPUT_CONTRACT_TEMPLATE = r"""
[UNIFIED STRONG-TEACHER OUTPUT]

最终只返回 JSON array。
不要 Markdown，不要代码块，不要 JSON 之外的解释。
每个 task 一个 object，task_id 必须原样复制。

每个 object 结构：

{
  "task_id": "原样复制",
  "overall_assessment": "用简体中文对该任务修复上下文需求做简短总结",
  "slots": {
    "fault_location": {
      "applicability": "required|not_required|uncertain",
      "question_coverage": "sufficient|partial|none|uncertain|not_applicable",
      "repository_need": "required|helpful|not_needed|uncertain|not_applicable",
      "candidate_pool_status": "sufficient|insufficient|uncertain|not_needed",
      "sufficient_witness_groups": [[1]],
      "supporting_candidates": [],
      "reason": "用简体中文说明具体语义理由"
    },
    "fault_logic": { "...": "同上" },
    "dependency_context": { "...": "同上" },
    "state_flow": { "...": "同上" },
    "behavior_constraint": { "...": "同上" },
    "repair_scope": { "...": "同上" },
    "validation_constraint": { "...": "同上" }
  },
  "additional_findings": [
    {
      "description": "用简体中文描述 7 槽之外的重要修复上下文；没有则返回空数组",
      "candidate_numbers": [],
      "reason": "用简体中文说明理由"
    }
  ],
  "uncertainties": []
}

注意：
- sufficient_witness_groups 中的每个数字必须是当前任务 Candidate Number。
- supporting_candidates / additional_findings.candidate_numbers 也必须是当前任务 Candidate Number。
- 如果 repository_need=required 且当前候选确实不足，candidate_pool_status=insufficient，
  sufficient_witness_groups=[]；可以把“相关但不足”的 Candidate 放 supporting_candidates。
- 如果 repository_need 不需要作为必要条件，但仓库代码仍有帮助，可以 repository_need=helpful，
  并把 Candidate 放 supporting_candidates；不要为了记录 helpful evidence 强行改成 required。
- Gold hunk / changed file 只用于离线定位真实修改方向。不要把 Issue traceback 中出现的函数
  自动描述成 Gold 修改点；Crash Site、Root Cause Site、Repair Site 必须分别表述。
- 每一个 sufficient_witness_groups 的外层 alternative 都必须独立充分。
  互补证据必须放在同一个 AND group，不能误写成两个 OR alternatives。
- reason 中如果使用“必要/必须”等措辞，必须与 sufficient_witness_groups 对应；
  只是有帮助的 Candidate 请写 supporting_candidates。
- 在输出 insufficient 前重新扫描全部 Candidate，避免漏掉 Gold hunk 对应的候选。
- 不要把尚未被 Candidate / Issue / Gold Hint 支持的因果链写成确定事实；保留在 uncertainties。
- Enum 必须严格使用对应字段定义的值；特别是 applicability 不允许 helpful，helpful 只属于 repository_need。
- 如果 uncertainties 说某个具体根因/传播机制尚不能确定，其他解释字段不得把同一机制写成已证实事实。
- 判断 behavior_constraint 时，只判断 Issue 是否已明确外部期望行为；不要因为实现策略、内部不变量或修复位置未知而错误降级 question_coverage。
- 若根因结论依赖 Candidate Pool 中未提供实现的 helper / 状态转换函数，只能表述为“证据指向/需要检查”，不得写成已证实根因。
- overall_assessment、所有 reason、additional_findings 的 description/reason、uncertainties 必须使用简体中文；
  机器字段、枚举、Candidate Number、path、symbol 与代码标识符保持原样。
""".strip()


def remove_old_output_contract(base_user_prompt: str) -> str:
    """
    v1.9.x base_user_prompt 已经包含完整任务上下文，但尾部还有旧 [OUTPUT] 合同。
    只删除旧输出合同，不改 Issue / Candidate / Gold Hints 正文。
    """
    text = str(base_user_prompt or "").strip()
    if not text:
        raise ValueError("base_user_prompt 为空")

    marker = "\n[OUTPUT]\n"
    if marker in text:
        return text.rsplit(marker, 1)[0].rstrip()

    # 兼容 marker 位于文件尾或换行风格不同。
    marker = "[OUTPUT]"
    if marker in text:
        return text.rsplit(marker, 1)[0].rstrip()

    # 没有旧 OUTPUT 也不应阻断：可能用户已经用其他工具预处理过 Prompt。
    return text


def build_unified_prompt(request: Mapping[str, Any]) -> str:
    task_id = str(request.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("request 缺少 task_id")

    base_user_prompt = str(request.get("base_user_prompt") or "")
    context = remove_old_output_contract(base_user_prompt)

    return (
        STRONG_TEACHER_INSTRUCTIONS
        + "\n\n"
        + "[TASK CONTEXT]\n"
        + context
        + "\n\n"
        + OUTPUT_CONTRACT_TEMPLATE
        + "\n\n"
        + "当前 task_id 必须原样复制为："
        + task_id
        + "\n"
    )


def export_unified(args: argparse.Namespace) -> int:
    requests = read_jsonl(args.requests.resolve())
    request_by_task = unique_requests(requests)
    if not request_by_task:
        raise ValueError("requests 为空")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    task_rows: list[dict[str, Any]] = []
    prompts: list[tuple[str, str]] = []

    for task_id, request in tqdm(
        request_by_task.items(),
        total=len(request_by_task),
        desc="Prepare unified tasks（准备统一任务）",
        unit="task",
        dynamic_ncols=True,
    ):
        candidate_ids = request_candidate_ids(request)
        prompt = build_unified_prompt(request)
        prompts.append((task_id, prompt))
        task_rows.append({
            "task_id": task_id,
            "candidate_count": len(candidate_ids),
            "candidate_evidence_ids": candidate_ids,
            "offline_gold_reference_used": bool(
                request.get("offline_gold_reference_used")
            ),
            "prompt_chars": len(prompt),
            "prompt": prompt,
        })

    batch_size = int(args.batch_size)
    if batch_size < 1:
        raise ValueError("--batch-size 必须 >= 1")

    batch_paths: list[str] = []
    for batch_index, start in enumerate(
        range(0, len(prompts), batch_size),
        start=1,
    ):
        batch = prompts[start:start + batch_size]
        parts = [
            "# Unified Strong-Teacher Batch",
            "",
            "请完成本文件中的所有 TASK。",
            "最终只返回一个 JSON array；不要 Markdown、不要代码块、不要额外解释。",
            "每个 TASK 恰好返回一个 object。",
            "",
        ]
        for local_index, (task_id, prompt) in enumerate(batch, start=1):
            parts.extend([
                "=" * 88,
                f"TASK {local_index} — {task_id}",
                "=" * 88,
                "",
                prompt,
                "",
            ])

        path = output_dir / f"unified_batch_{batch_index:03d}.md"
        path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
        batch_paths.append(str(path))

    tasks_path = output_dir / "unified_tasks.jsonl"
    atomic_write_jsonl(tasks_path, task_rows)

    template = [
        {
            "task_id": "COPY_TASK_ID_EXACTLY",
            "overall_assessment": "",
            "slots": {
                slot_type: {
                    "applicability": "uncertain",
                    "question_coverage": "uncertain",
                    "repository_need": "uncertain",
                    "candidate_pool_status": "uncertain",
                    "sufficient_witness_groups": [],
                    "supporting_candidates": [],
                    "reason": "",
                }
                for slot_type in SLOT_TYPES
            },
            "additional_findings": [],
            "uncertainties": [],
        }
    ]
    template_path = output_dir / "unified_results_TEMPLATE.json"
    atomic_write_json(template_path, template)

    report = {
        "script_version": SCRIPT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "operation": "export-unified",
        "created_at": utc_now(),
        "task_count": len(task_rows),
        "batch_size": batch_size,
        "batch_count": len(batch_paths),
        "mean_prompt_chars": (
            sum(row["prompt_chars"] for row in task_rows) / len(task_rows)
        ),
        "max_prompt_chars": max(row["prompt_chars"] for row in task_rows),
        "outputs": {
            "tasks": str(tasks_path),
            "template": str(template_path),
            "batches": batch_paths,
        },
        "training_eligible": False,
    }
    atomic_write_json(output_dir / "unified_export_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


# ============================================================================
# 2B. 冻结 V2.10 全量导出（20,864 tasks）
# ============================================================================

EXPECTED_DATASET_VERSION = "2.10.0"
EXPECTED_SPLIT_COUNTS = {
    "train": 18347,
    "validation": 223,
    "benchmark": 2294,
}
EXPECTED_TOTAL_TASK_COUNT = sum(EXPECTED_SPLIT_COUNTS.values())


def _sanitize_filename_component(value: str) -> str:
    """只用于文件名；task_id 本身在 JSON / Prompt 中绝不改写。"""
    safe = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    result = "".join(safe).strip("._")
    return result or "task"


def _write_jsonl_record(handle: Any, row: Mapping[str, Any]) -> None:
    """流式写一条 JSONL，避免 20k 长 Prompt 全部堆进内存。"""
    handle.write(
        json.dumps(
            dict(row),
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )


def _build_request_from_prepared(item: Mapping[str, Any]) -> dict[str, Any]:
    """
    生成与 v1.9.2.1 refinement_requests.jsonl 兼容的审计记录。

    这里故意沿用 runner 的 user_prompt / candidate_metadata，
    Strong-Teacher 只替换旧 [OUTPUT] 合同，不重新拼 Issue / Candidate / Gold Hints。
    """
    candidate_ids = [str(x) for x in (item.get("candidate_ids") or [])]
    if not candidate_ids:
        raise RuntimeError(f"task={item.get('task_id')} candidate_ids 为空")

    metadata = dict(item.get("candidate_metadata") or {})
    selected_ids = metadata.get("selected_evidence_ids")
    if selected_ids is None:
        # prepare_task_payload 的 candidate_ids 是实际展示顺序；
        # 若 metadata 未保存该字段，则显式回填，建立 Candidate Number 身份合同。
        metadata["selected_evidence_ids"] = candidate_ids
    else:
        normalized = [str(x) for x in selected_ids]
        if normalized != candidate_ids:
            raise RuntimeError(
                f"task={item.get('task_id')} candidate_metadata.selected_evidence_ids "
                "与 candidate_ids 顺序不一致"
            )

    user_prompt = str(item.get("user_prompt") or "")
    if not user_prompt:
        raise RuntimeError(f"task={item.get('task_id')} user_prompt 为空")

    return {
        "task_id": str(item.get("task_id") or ""),
        "has_boundary": bool(item.get("has_boundary")),
        "base_user_prompt": user_prompt,
        "base_prompt_chars": int(item.get("prompt_chars") or len(user_prompt)),
        "candidate_count": len(candidate_ids),
        "candidate_diagnostics": metadata,
        "offline_gold_reference_used": bool(item.get("offline_gold_reference_used")),
    }


def _build_context_from_prepared(
    *,
    item: Mapping[str, Any],
    task_row: Mapping[str, Any],
    split: str,
    runner: Any,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """
    在导出网页 Prompt 的同一次 prepare 中保存后续 Core merge 所需机器上下文。

    这样网页模型返回 Candidate Number 后无需再次重建 Candidate Pool，
    从根源上避免“二次 prepare 后 Candidate #N 已变化”的身份漂移。
    """
    candidate_records = [dict(x) for x in (item.get("candidate_records") or [])]
    if not candidate_records:
        raise RuntimeError(f"task={item.get('task_id')} candidate_records 为空")

    candidate_ids = [str(record.get("evidence_id") or "") for record in candidate_records]
    if any(not evidence_id for evidence_id in candidate_ids):
        raise RuntimeError(f"task={item.get('task_id')} candidate_records 存在空 evidence_id")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError(f"task={item.get('task_id')} candidate Evidence ID 重复")

    item_candidate_ids = [str(x) for x in (item.get("candidate_ids") or candidate_ids)]
    if item_candidate_ids != candidate_ids:
        raise RuntimeError(
            f"task={item.get('task_id')} candidate_ids 与 candidate_records 顺序不一致"
        )

    supervision = item.get("supervision") or task_row.get("supervision") or {}
    if not isinstance(supervision, dict) or not supervision:
        raise RuntimeError(f"task={item.get('task_id')} supervision 为空")

    token_costs = item.get("token_costs")
    if isinstance(token_costs, dict):
        normalized_token_costs = {str(k): int(v) for k, v in token_costs.items()}
    else:
        normalized_token_costs = {
            evidence_id: int(record.get("rendered_token_count") or 2**30)
            for evidence_id, record in zip(candidate_ids, candidate_records)
        }

    return {
        "task_id": str(item.get("task_id") or ""),
        "split": split,
        "runner_version": str(getattr(runner, "RUNNER_VERSION", "")),
        "dataset_version": str(manifest.get("dataset_version") or ""),
        "manifest": str(manifest_path.resolve()),
        "supervision": supervision,
        "candidate_records": candidate_records,
        "candidate_ids": candidate_ids,
        "token_costs": normalized_token_costs,
        "existing_evidence_ids": candidate_ids,
        "has_boundary": bool(item.get("has_boundary")),
        "candidate_metadata": dict(item.get("candidate_metadata") or {}),
        "offline_gold_reference_used": bool(item.get("offline_gold_reference_used")),
        "identity_lock": {
            "status": "captured_at_export",
            "candidate_count": len(candidate_ids),
            "source": "same prepare_task_payload call used to render Strong-Teacher Markdown",
        },
    }


def _runner_candidate_config(runner: Any, args: argparse.Namespace) -> tuple[Any, Any]:
    """继承 v1.9.2.1 runner 默认参数，只有显式 CLI override 才覆盖。"""
    runner_defaults = runner.build_parser().parse_args([])

    def choose(name: str) -> Any:
        explicit = getattr(args, name)
        return getattr(runner_defaults, name) if explicit is None else explicit

    config = runner.CandidateBuilderConfig(
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
    config.validate()
    return runner_defaults, config


def _write_batch_markdown(
    *,
    path: Path,
    batch: Sequence[tuple[str, str]],
) -> None:
    """原子写一个网页 Teacher Markdown batch。"""
    parts = [
        "# Unified Strong-Teacher Batch",
        "",
        "请完成本文件中的所有 TASK。",
        "最终只返回一个 JSON array；不要 Markdown、不要代码块、不要额外解释。",
        "每个 TASK 恰好返回一个 object。",
        "所有解释/审计字段必须使用简体中文；机器字段和枚举保持规定的英文值。",
        "",
    ]
    for local_index, (task_id, prompt) in enumerate(batch, start=1):
        parts.extend([
            "=" * 88,
            f"TASK {local_index} — {task_id}",
            "=" * 88,
            "",
            prompt,
            "",
        ])

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)



def _concat_text_parts(parts: Sequence[Path], destination: Path) -> None:
    """按 shard 顺序拼接 JSONL part；不重新解析 JSON，降低主线程 CPU/内存开销。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as out_handle:
        for part in parts:
            if not part.exists():
                continue
            with part.open("r", encoding="utf-8") as in_handle:
                shutil.copyfileobj(in_handle, out_handle, length=1024 * 1024)


def _prepare_export_shard_persistent(
    *,
    runner: Any,
    shard_index: int,
    rows: Sequence[tuple[int, Mapping[str, Any]]],
    cache_path: Path,
    build_db_path: Path,
    candidate_config: Any,
    reference_mode: str,
    max_prompt_chars: int,
    max_teacher_question_chars: int,
    split: str,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    md_dir: Path,
    part_dir: Path,
    tasks_per_md: int,
    progress: Any,
) -> dict[str, Any]:
    """
    Persistent Worker（持久准备线程）。

    v1.4.3 的核心修复：
      - 一个 worker 对整个 shard 只创建一次 EvidenceCache / BuildEvidenceStore；
      - 该 worker 连续处理数千 task，split 完成后才关闭连接；
      - 不再像 v1.4.2 那样每 32 task 重新打开/关闭 SQLite；
      - SQLite connection 始终只在创建它的线程中使用并关闭；
      - worker 直接写独立 JSONL part 与唯一命名 MD，避免把巨大 Prompt/Context
        结果跨线程回传到主线程造成额外内存复制与有序等待。

    每个 shard 是原 split 的一个连续序列区间；最终按 shard_index 拼接 part，
    因而 requests / merge_context / tasks 仍保持原始数据集顺序。
    """
    part_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    req_part = part_dir / f"requests_{shard_index:03d}.jsonl"
    ctx_part = part_dir / f"merge_context_{shard_index:03d}.jsonl"
    task_part = part_dir / f"tasks_{shard_index:03d}.jsonl"
    err_part = part_dir / f"errors_{shard_index:03d}.jsonl"

    prepared_count = 0
    error_count = 0
    md_count = 0
    candidate_total = 0
    prompt_char_total = 0
    max_prompt_chars_seen = 0
    batch: list[tuple[int, str, str]] = []

    def flush_batch() -> None:
        nonlocal batch, md_count
        if not batch:
            return
        md_count += 1
        first_seq = batch[0][0]
        last_seq = batch[-1][0]
        if tasks_per_md == 1:
            _, task_id, prompt = batch[0]
            suffix = _sanitize_filename_component(task_id)
            md_name = f"task_{first_seq:06d}_{suffix}.md"
            rendered_batch = [(task_id, prompt)]
        else:
            md_name = f"batch_{first_seq:06d}_{last_seq:06d}.md"
            rendered_batch = [(task_id, prompt) for _, task_id, prompt in batch]
        _write_batch_markdown(path=md_dir / md_name, batch=rendered_batch)
        batch = []

    with (
        req_part.open("w", encoding="utf-8", newline="\n") as req_handle,
        ctx_part.open("w", encoding="utf-8", newline="\n") as ctx_handle,
        task_part.open("w", encoding="utf-8", newline="\n") as task_handle,
        err_part.open("w", encoding="utf-8", newline="\n") as err_handle,
    ):
        cache = None
        build_store = None
        try:
            # 连接在 worker thread 内创建，并在 finally 中于同一 thread 关闭。
            cache = runner.EvidenceCache(cache_path)
            build_store = runner.BuildEvidenceStore(build_db_path)
        except Exception as exc:
            # DB 初始化失败时，不能让整个 shard 静默消失。
            for sequence, row in rows:
                task_id = str(row.get("task_id") or "")
                _write_jsonl_record(err_handle, {
                    "task_id": task_id,
                    "split": split,
                    "sequence": int(sequence),
                    "stage": "worker_init",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                error_count += 1
                progress.update(1)
            return {
                "shard_index": shard_index,
                "row_count": len(rows),
                "prepared_count": 0,
                "error_count": error_count,
                "md_count": 0,
                "candidate_total": 0,
                "prompt_char_total": 0,
                "max_prompt_chars_seen": 0,
                "parts": {
                    "requests": req_part,
                    "merge_context": ctx_part,
                    "tasks": task_part,
                    "errors": err_part,
                },
            }

        try:
            for sequence, row in rows:
                task_id = str(row.get("task_id") or "").strip()
                try:
                    if not task_id:
                        raise ValueError(f"split row #{sequence} 缺少 task_id")

                    item = runner.prepare_task_payload(
                        task_row=row,
                        cache=cache,
                        build_store=build_store,
                        candidate_config=candidate_config,
                        reference_mode=reference_mode,
                        max_prompt_chars=max_prompt_chars,
                        max_teacher_question_chars=max_teacher_question_chars,
                    )
                    actual_task_id = str(item.get("task_id") or "")
                    if actual_task_id != task_id:
                        raise RuntimeError(
                            "prepare_task_payload task_id 漂移："
                            f"row={task_id}, item={actual_task_id}"
                        )

                    request = _build_request_from_prepared(item)
                    context = _build_context_from_prepared(
                        item=item,
                        task_row=row,
                        split=split,
                        runner=runner,
                        manifest_path=manifest_path,
                        manifest=manifest,
                    )
                    prompt = build_unified_prompt(request)
                    candidate_ids = request_candidate_ids(request)

                    _write_jsonl_record(req_handle, request)
                    _write_jsonl_record(ctx_handle, context)
                    _write_jsonl_record(task_handle, {
                        "task_id": task_id,
                        "split": split,
                        "sequence": int(sequence),
                        "candidate_count": len(candidate_ids),
                        "candidate_evidence_ids": candidate_ids,
                        "prompt_chars": len(prompt),
                        "offline_gold_reference_used": bool(
                            request.get("offline_gold_reference_used")
                        ),
                    })

                    prepared_count += 1
                    candidate_total += len(candidate_ids)
                    prompt_chars = len(prompt)
                    prompt_char_total += prompt_chars
                    max_prompt_chars_seen = max(max_prompt_chars_seen, prompt_chars)

                    batch.append((int(sequence), task_id, prompt))
                    if len(batch) >= tasks_per_md:
                        flush_batch()

                except Exception as exc:
                    error_count += 1
                    _write_jsonl_record(err_handle, {
                        "task_id": task_id,
                        "split": split,
                        "sequence": int(sequence),
                        "stage": "prepare_or_serialize",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                finally:
                    progress.update(1)

            flush_batch()
        finally:
            # 必须在创建 connection 的同一个 worker thread 中关闭。
            if cache is not None:
                cache.close()
            if build_store is not None:
                build_store.close()

    return {
        "shard_index": shard_index,
        "row_count": len(rows),
        "prepared_count": prepared_count,
        "error_count": error_count,
        "md_count": md_count,
        "candidate_total": candidate_total,
        "prompt_char_total": prompt_char_total,
        "max_prompt_chars_seen": max_prompt_chars_seen,
        "parts": {
            "requests": req_part,
            "merge_context": ctx_part,
            "tasks": task_part,
            "errors": err_part,
        },
    }


def export_all(args: argparse.Namespace) -> int:
    """
    从冻结 Unified SWE Dataset V2.10 直接准备并导出全量 Strong-Teacher Markdown。

    v1.4.3 性能修复：Persistent Shard Workers（持久 shard worker）

    v1.4.2 的问题：
      - 一个 future 只处理 prepare_chunk_size（默认 32）个 task；
      - 每个 future 都重新创建 EvidenceCache + BuildEvidenceStore；
      - train 18,347 条会产生约 574 个 chunk，导致 SQLite connection / page cache
        被反复创建和丢弃；
      - 多 worker 对 60GB SQLite 随机读取时，还会放大 page-cache 抖动；
      - 主线程为了按 sequence 输出，存在 ready 队列和 burst progress，tqdm 的瞬时
        task/s 会出现“最初 40+，后来快速下降”的假象。

    v1.4.3：
      - 每个 worker 对整个连续 shard 只开一次两类 SQLite store；
      - 每个 worker 直接写自己的 JSONL part 与唯一 MD；
      - split 完成后主线程按 shard 顺序拼接 sidecar；
      - 不共享 SQLite connection，不改变 Candidate Builder / Candidate Number 语义；
      - tqdm 直接按真实完成 task 更新，不再受有序 ready burst 影响。

    --prepare-chunk-size 为兼容旧命令继续保留，但在 workers>1 的 v1.4.3
    persistent 模式中不再参与调度。
    """
    merge = import_merge_module()
    runner = merge.load_runner(args.runner)

    dataset_dir = args.dataset_dir.resolve()
    manifest_path, manifest = runner.load_and_validate_manifest(dataset_dir)
    dataset_version = str(manifest.get("dataset_version") or "")
    if dataset_version != EXPECTED_DATASET_VERSION:
        raise ValueError(
            f"export-all 只允许冻结 V2.10：actual={dataset_version!r}, "
            f"expected={EXPECTED_DATASET_VERSION!r}"
        )

    requested_splits = list(args.splits)
    if len(set(requested_splits)) != len(requested_splits):
        raise ValueError("--splits 不允许重复")

    tasks_per_md = int(args.tasks_per_md)
    if tasks_per_md < 1:
        raise ValueError("--tasks-per-md 必须 >= 1")

    prepare_workers = int(args.prepare_workers)
    prepare_chunk_size = int(args.prepare_chunk_size)
    if prepare_workers < 1:
        raise ValueError("--prepare-workers 必须 >= 1")
    if prepare_workers > 32:
        raise ValueError(
            "--prepare-workers 不建议超过 32；"
            "这是 60GB SQLite 只读随机访问，不是纯 CPU 任务"
        )
    if prepare_chunk_size < 1:
        raise ValueError("--prepare-chunk-size 必须 >= 1")

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    runner_defaults, candidate_config = _runner_candidate_config(runner, args)
    reference_mode = args.reference_mode or runner_defaults.reference_mode
    max_prompt_chars = args.max_prompt_chars or runner_defaults.max_prompt_chars
    max_teacher_question_chars = (
        args.max_teacher_question_chars
        or runner_defaults.max_teacher_question_chars
    )
    cache_path = (args.evidence_cache or runner_defaults.evidence_cache).resolve()
    build_db_path = (args.build_db or runner_defaults.build_db).resolve()

    global_prepared = 0
    global_errors = 0
    split_reports: dict[str, Any] = {}
    global_started = time.perf_counter()

    for split in requested_splits:
        split_started = time.perf_counter()
        split_dir = output_root / split
        md_dir = split_dir / "md"
        md_dir.mkdir(parents=True, exist_ok=True)

        requests_path = split_dir / "requests.jsonl"
        context_path = split_dir / "merge_context.jsonl"
        tasks_path = split_dir / "tasks.jsonl"
        errors_path = split_dir / "preparation_errors.jsonl"

        temp_requests = requests_path.with_name(requests_path.name + ".tmp")
        temp_context = context_path.with_name(context_path.name + ".tmp")
        temp_tasks = tasks_path.with_name(tasks_path.name + ".tmp")
        temp_errors = errors_path.with_name(errors_path.name + ".tmp")

        split_path = runner.resolve_split_path(dataset_dir, split)
        expected_count = EXPECTED_SPLIT_COUNTS[split]

        prepared_count = 0
        error_count = 0
        seen_count = 0
        md_count = 0
        candidate_total = 0
        prompt_char_total = 0
        max_prompt_chars_seen = 0

        if prepare_workers == 1:
            # -------------------------------------------------------------
            # 单 worker：整个 split 只开一次 DB，保持最小 I/O 干扰。
            # -------------------------------------------------------------
            batch: list[tuple[int, str, str]] = []

            def flush_single_batch() -> None:
                nonlocal batch, md_count
                if not batch:
                    return
                md_count += 1
                first_seq = batch[0][0]
                last_seq = batch[-1][0]
                if tasks_per_md == 1:
                    _, task_id, prompt = batch[0]
                    suffix = _sanitize_filename_component(task_id)
                    md_name = f"task_{first_seq:06d}_{suffix}.md"
                    rendered_batch = [(task_id, prompt)]
                else:
                    md_name = f"batch_{first_seq:06d}_{last_seq:06d}.md"
                    rendered_batch = [(task_id, prompt) for _, task_id, prompt in batch]
                _write_batch_markdown(path=md_dir / md_name, batch=rendered_batch)
                batch = []

            with (
                temp_requests.open("w", encoding="utf-8", newline="\n") as req_handle,
                temp_context.open("w", encoding="utf-8", newline="\n") as ctx_handle,
                temp_tasks.open("w", encoding="utf-8", newline="\n") as task_handle,
                temp_errors.open("w", encoding="utf-8", newline="\n") as err_handle,
                tqdm(
                    total=expected_count,
                    desc=f"Prepare {split}（准备 {split}）",
                    unit="task",
                    dynamic_ncols=True,
                    smoothing=0,  # 显示累计平均吞吐，避免短时 burst 误导 ETA。
                ) as progress,
            ):
                cache = runner.EvidenceCache(cache_path)
                build_store = runner.BuildEvidenceStore(build_db_path)
                try:
                    for row in runner.iter_task_rows(split_path):
                        seen_count += 1
                        task_id = str(row.get("task_id") or "").strip()
                        try:
                            if not task_id:
                                raise ValueError(f"split row #{seen_count} 缺少 task_id")

                            item = runner.prepare_task_payload(
                                task_row=row,
                                cache=cache,
                                build_store=build_store,
                                candidate_config=candidate_config,
                                reference_mode=reference_mode,
                                max_prompt_chars=max_prompt_chars,
                                max_teacher_question_chars=max_teacher_question_chars,
                            )
                            actual_task_id = str(item.get("task_id") or "")
                            if actual_task_id != task_id:
                                raise RuntimeError(
                                    "prepare_task_payload task_id 漂移："
                                    f"row={task_id}, item={actual_task_id}"
                                )

                            request = _build_request_from_prepared(item)
                            context = _build_context_from_prepared(
                                item=item,
                                task_row=row,
                                split=split,
                                runner=runner,
                                manifest_path=manifest_path,
                                manifest=manifest,
                            )
                            prompt = build_unified_prompt(request)
                            candidate_ids = request_candidate_ids(request)

                            _write_jsonl_record(req_handle, request)
                            _write_jsonl_record(ctx_handle, context)
                            _write_jsonl_record(task_handle, {
                                "task_id": task_id,
                                "split": split,
                                "sequence": seen_count,
                                "candidate_count": len(candidate_ids),
                                "candidate_evidence_ids": candidate_ids,
                                "prompt_chars": len(prompt),
                                "offline_gold_reference_used": bool(
                                    request.get("offline_gold_reference_used")
                                ),
                            })

                            prepared_count += 1
                            candidate_total += len(candidate_ids)
                            prompt_chars = len(prompt)
                            prompt_char_total += prompt_chars
                            max_prompt_chars_seen = max(
                                max_prompt_chars_seen, prompt_chars
                            )

                            batch.append((seen_count, task_id, prompt))
                            if len(batch) >= tasks_per_md:
                                flush_single_batch()

                        except Exception as exc:
                            error_count += 1
                            _write_jsonl_record(err_handle, {
                                "task_id": task_id,
                                "split": split,
                                "sequence": seen_count,
                                "stage": "prepare_or_serialize",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            })
                        finally:
                            progress.update(1)
                    flush_single_batch()
                finally:
                    cache.close()
                    build_store.close()

        else:
            # -------------------------------------------------------------
            # Persistent Shard Parallel Prepare（持久 shard 并行准备）
            # -------------------------------------------------------------
            # 先读取本 split 的 task row。20,864 规模远小于 corpus，且这里只保存
            # dataset row，不加载 Candidate corpus；换来每个 worker 可以拥有一个连续
            # shard 并在整个 shard 生命周期复用 SQLite page cache。
            all_rows = list(runner.iter_task_rows(split_path))
            seen_count = len(all_rows)
            effective_workers = min(prepare_workers, max(1, seen_count))
            shard_size = max(1, (seen_count + effective_workers - 1) // effective_workers)

            part_dir = split_dir / ".prepare_parts_v1_4_3"
            if part_dir.exists():
                shutil.rmtree(part_dir)
            part_dir.mkdir(parents=True, exist_ok=True)

            shards: list[tuple[int, list[tuple[int, Mapping[str, Any]]]]] = []
            for worker_index in range(effective_workers):
                start_index = worker_index * shard_size
                if start_index >= seen_count:
                    break
                stop_index = min(seen_count, start_index + shard_size)
                shard_rows = [
                    (index + 1, all_rows[index])
                    for index in range(start_index, stop_index)
                ]
                shards.append((worker_index, shard_rows))

            shard_reports: list[dict[str, Any]] = []
            with tqdm(
                total=expected_count,
                desc=f"Prepare {split}（准备 {split}）",
                unit="task",
                dynamic_ncols=True,
                smoothing=0,  # 显示累计平均吞吐，避免短时 burst 误导 ETA。
            ) as progress:
                with ThreadPoolExecutor(
                    max_workers=effective_workers,
                    thread_name_prefix="strong-teacher-persistent",
                ) as executor:
                    futures = [
                        executor.submit(
                            _prepare_export_shard_persistent,
                            runner=runner,
                            shard_index=shard_index,
                            rows=shard_rows,
                            cache_path=cache_path,
                            build_db_path=build_db_path,
                            candidate_config=candidate_config,
                            reference_mode=reference_mode,
                            max_prompt_chars=max_prompt_chars,
                            max_teacher_question_chars=max_teacher_question_chars,
                            split=split,
                            manifest_path=manifest_path,
                            manifest=manifest,
                            md_dir=md_dir,
                            part_dir=part_dir,
                            tasks_per_md=tasks_per_md,
                            progress=progress,
                        )
                        for shard_index, shard_rows in shards
                    ]
                    for future in as_completed(futures):
                        shard_reports.append(future.result())

            shard_reports.sort(key=lambda row: int(row["shard_index"]))

            prepared_count = sum(int(r["prepared_count"]) for r in shard_reports)
            error_count = sum(int(r["error_count"]) for r in shard_reports)
            md_count = sum(int(r["md_count"]) for r in shard_reports)
            candidate_total = sum(int(r["candidate_total"]) for r in shard_reports)
            prompt_char_total = sum(int(r["prompt_char_total"]) for r in shard_reports)
            max_prompt_chars_seen = max(
                [int(r["max_prompt_chars_seen"]) for r in shard_reports] or [0]
            )

            _concat_text_parts(
                [Path(r["parts"]["requests"]) for r in shard_reports],
                temp_requests,
            )
            _concat_text_parts(
                [Path(r["parts"]["merge_context"]) for r in shard_reports],
                temp_context,
            )
            _concat_text_parts(
                [Path(r["parts"]["tasks"]) for r in shard_reports],
                temp_tasks,
            )
            _concat_text_parts(
                [Path(r["parts"]["errors"]) for r in shard_reports],
                temp_errors,
            )
            shutil.rmtree(part_dir, ignore_errors=True)

        # 空 split / 极端失败情况下也保证 temp 文件存在。
        for temp_path in (temp_requests, temp_context, temp_tasks, temp_errors):
            if not temp_path.exists():
                temp_path.write_text("", encoding="utf-8")

        temp_requests.replace(requests_path)
        temp_context.replace(context_path)
        temp_tasks.replace(tasks_path)
        temp_errors.replace(errors_path)

        count_ok = seen_count == expected_count
        prepared_complete = prepared_count == seen_count and error_count == 0
        split_complete = prepared_complete and (
            count_ok or bool(args.allow_count_mismatch)
        )

        split_elapsed = max(1e-9, time.perf_counter() - split_started)
        parallel_strategy = (
            "single_persistent_connection"
            if prepare_workers == 1
            else "persistent_contiguous_shard_workers"
        )
        split_report = {
            "split": split,
            "expected_task_count": expected_count,
            "scanned_task_count": seen_count,
            "prepared_task_count": prepared_count,
            "preparation_error_count": error_count,
            "md_file_count": md_count,
            "tasks_per_md": tasks_per_md,
            "prepare_workers": prepare_workers,
            "effective_prepare_workers": min(prepare_workers, max(1, seen_count)),
            "parallel_strategy": parallel_strategy,
            "prepare_chunk_size_legacy_ignored_when_parallel": prepare_chunk_size,
            "elapsed_seconds": round(split_elapsed, 3),
            "effective_tasks_per_second": round(
                seen_count / split_elapsed if seen_count else 0.0,
                4,
            ),
            "candidate_count_total": candidate_total,
            "mean_candidate_count": (
                candidate_total / prepared_count if prepared_count else 0.0
            ),
            "mean_prompt_chars": (
                prompt_char_total / prepared_count if prepared_count else 0.0
            ),
            "max_prompt_chars": max_prompt_chars_seen,
            "count_check_passed": count_ok,
            "complete": split_complete,
            "outputs": {
                "md_dir": str(md_dir),
                "requests": str(requests_path),
                "merge_context": str(context_path),
                "tasks": str(tasks_path),
                "errors": str(errors_path),
            },
        }
        atomic_write_json(split_dir / "export_report.json", split_report)
        split_reports[split] = split_report
        global_prepared += prepared_count
        global_errors += error_count

    expected_selected_total = sum(EXPECTED_SPLIT_COUNTS[s] for s in requested_splits)
    actual_seen_total = sum(
        int(report["scanned_task_count"])
        for report in split_reports.values()
    )
    complete = (
        global_errors == 0
        and global_prepared == actual_seen_total
        and all(bool(report["complete"]) for report in split_reports.values())
    )
    global_elapsed = max(1e-9, time.perf_counter() - global_started)

    report = {
        "script_version": SCRIPT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "operation": "export-all",
        "created_at": utc_now(),
        "dataset_version": dataset_version,
        "manifest": str(manifest_path.resolve()),
        "splits": requested_splits,
        "expected_selected_task_count": expected_selected_total,
        "expected_full_dataset_task_count": EXPECTED_TOTAL_TASK_COUNT,
        "scanned_task_count": actual_seen_total,
        "prepared_task_count": global_prepared,
        "preparation_error_count": global_errors,
        "tasks_per_md": tasks_per_md,
        "prepare_workers": prepare_workers,
        "parallel_strategy": (
            "single_persistent_connection"
            if prepare_workers == 1
            else "persistent_contiguous_shard_workers"
        ),
        "elapsed_seconds": round(global_elapsed, 3),
        "effective_tasks_per_second": round(
            actual_seen_total / global_elapsed if actual_seen_total else 0.0,
            4,
        ),
        "complete": complete,
        "split_reports": split_reports,
        "training_eligible": False,
        "benchmark_policy": (
            "Benchmark may be exported because the user explicitly requested the full dataset, "
            "but benchmark results must not be used to tune the protocol before protocol freeze."
        ),
        "scientific_contract": {
            "v2_10_modified": False,
            "llm_api_called": False,
            "candidate_numbers_bound_at_export": True,
            "prompt_and_merge_context_share_same_prepare_call": True,
            "teacher_output_is_not_direct_training_label": True,
            "parallel_prepare_changes_semantics": False,
            "sqlite_connection_shared_across_threads": False,
            "sqlite_connection_reopened_per_small_chunk": False,
        },
    }
    atomic_write_json(output_root / "export_all_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    return 0 if complete else 2

def add_full_export_candidate_args(parser: argparse.ArgumentParser) -> None:
    """与 v1.9.2.1 / external_supervision_merge.py 的 candidate override 保持同名。"""
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


# ============================================================================
# 3. 宽松语义 / 严格身份 Validator
# ============================================================================


def normalize_enum(
    raw: Any,
    *,
    allowed: set[str],
    field: str,
    task_id: str,
    slot_type: str,
    flags: list[dict[str, Any]],
) -> str:
    """
    只做机械字符串规范化：大小写、空格、连字符。

    不把 optional / maybe / yes / no 等自由文本自动猜成某个语义枚举。
    未知值保留在 raw_result 中，并把 canonical value 设为 uncertain。
    """
    text = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in allowed:
        return text

    flags.append({
        "severity": "blocking",
        "type": "unknown_enum",
        "task_id": task_id,
        "slot_type": slot_type,
        "field": field,
        "raw_value": raw,
        "normalized_to": "uncertain",
    })
    return "uncertain"


def canonicalize_candidate_number_list(
    raw: Any,
    *,
    candidate_count: int,
    task_id: str,
    location: str,
    identity_errors: list[dict[str, Any]],
) -> list[int]:
    """
    supporting_candidates / additional_findings.candidate_numbers 使用。

    这里可以去重、排序，因为它们只是无序 Candidate 集合。
    Candidate 越界则记录 identity error；合法项仍保留用于审计，
    但只要 task 存在 identity error，最终 build-refinement 一律 BLOCKED。
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        identity_errors.append({
            "type": "candidate_number_list_not_array",
            "task_id": task_id,
            "location": location,
            "raw_value": raw,
        })
        return []

    result: set[int] = set()
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            identity_errors.append({
                "type": "candidate_number_not_integer",
                "task_id": task_id,
                "location": location,
                "raw_value": value,
            })
            continue
        if value < 1 or value > candidate_count:
            identity_errors.append({
                "type": "candidate_number_out_of_range",
                "task_id": task_id,
                "location": location,
                "candidate_number": value,
                "valid_range": [1, candidate_count],
            })
            continue
        result.add(value)

    return sorted(result)


def canonicalize_witness_groups(
    raw: Any,
    *,
    candidate_count: int,
    task_id: str,
    slot_type: str,
    identity_errors: list[dict[str, Any]],
) -> list[list[int]]:
    """
    OR-of-AND 的语义不能猜。

    允许的机械操作：
        - AND group 内 Candidate Number 去重、升序；
        - 完全相同的 OR alternative 去重；
        - OR alternatives 做稳定排序。

    如果任意 Candidate Number 非法：
        整个 sufficient_witness_groups 不进入 canonical supervision。

    不能只删除坏号码后留下部分 AND group，因为：
        [2, 999] -> [2]
    会把“2 AND 999”错误改成“2”。
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        identity_errors.append({
            "type": "witness_groups_not_array",
            "task_id": task_id,
            "slot_type": slot_type,
            "raw_value": raw,
        })
        return []

    groups: set[tuple[int, ...]] = set()
    invalid = False

    for group_index, group in enumerate(raw, start=1):
        if not isinstance(group, list) or not group:
            identity_errors.append({
                "type": "invalid_and_group",
                "task_id": task_id,
                "slot_type": slot_type,
                "group_index": group_index,
                "raw_value": group,
            })
            invalid = True
            continue

        numbers: list[int] = []
        for value in group:
            if isinstance(value, bool) or not isinstance(value, int):
                identity_errors.append({
                    "type": "candidate_number_not_integer",
                    "task_id": task_id,
                    "slot_type": slot_type,
                    "group_index": group_index,
                    "raw_value": value,
                })
                invalid = True
                continue
            if value < 1 or value > candidate_count:
                identity_errors.append({
                    "type": "candidate_number_out_of_range",
                    "task_id": task_id,
                    "slot_type": slot_type,
                    "group_index": group_index,
                    "candidate_number": value,
                    "valid_range": [1, candidate_count],
                })
                invalid = True
                continue
            numbers.append(value)

        if numbers:
            groups.add(tuple(sorted(set(numbers))))

    if invalid:
        return []

    return [list(group) for group in sorted(groups)]


def empty_uncertain_slot(reason: str) -> dict[str, Any]:
    return {
        "applicability": "uncertain",
        "question_coverage": "uncertain",
        "repository_need": "uncertain",
        "candidate_pool_status": "uncertain",
        "sufficient_witness_groups": [],
        "supporting_candidates": [],
        "reason": reason,
    }


def normalize_slot(
    *,
    task_id: str,
    slot_type: str,
    raw_slot: Any,
    candidate_count: int,
    flags: list[dict[str, Any]],
    identity_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(raw_slot, dict):
        flags.append({
            "severity": "blocking",
            "type": "missing_or_invalid_slot",
            "task_id": task_id,
            "slot_type": slot_type,
        })
        return empty_uncertain_slot("missing_or_invalid_slot")

    applicability = normalize_enum(
        raw_slot.get("applicability"),
        allowed=APPLICABILITY_VALUES,
        field="applicability",
        task_id=task_id,
        slot_type=slot_type,
        flags=flags,
    )
    question_coverage = normalize_enum(
        raw_slot.get("question_coverage"),
        allowed=QUESTION_COVERAGE_VALUES,
        field="question_coverage",
        task_id=task_id,
        slot_type=slot_type,
        flags=flags,
    )
    repository_need = normalize_enum(
        raw_slot.get("repository_need"),
        allowed=REPOSITORY_NEED_VALUES,
        field="repository_need",
        task_id=task_id,
        slot_type=slot_type,
        flags=flags,
    )
    candidate_pool_status = normalize_enum(
        raw_slot.get("candidate_pool_status"),
        allowed=CANDIDATE_POOL_STATUS_VALUES,
        field="candidate_pool_status",
        task_id=task_id,
        slot_type=slot_type,
        flags=flags,
    )

    witness_groups = canonicalize_witness_groups(
        raw_slot.get("sufficient_witness_groups"),
        candidate_count=candidate_count,
        task_id=task_id,
        slot_type=slot_type,
        identity_errors=identity_errors,
    )

    supporting_candidates = canonicalize_candidate_number_list(
        raw_slot.get("supporting_candidates"),
        candidate_count=candidate_count,
        task_id=task_id,
        location=f"slots.{slot_type}.supporting_candidates",
        identity_errors=identity_errors,
    )

    # ------------------------------------------------------------------
    # 以下只打 semantic flag，不在 validator 阶段“替 Teacher 改答案”。
    # 是否能映射到旧 Programmatic Supervision，由 build-refinement 再判断。
    # ------------------------------------------------------------------

    if applicability == "not_required" and repository_need == "required":
        flags.append({
            "severity": "blocking",
            "type": "semantic_self_conflict",
            "task_id": task_id,
            "slot_type": slot_type,
            "detail": "applicability=not_required but repository_need=required",
        })

    if repository_need == "required":
        if candidate_pool_status == "sufficient" and not witness_groups:
            flags.append({
                "severity": "blocking",
                "type": "sufficient_pool_without_sufficient_witness",
                "task_id": task_id,
                "slot_type": slot_type,
            })
        if candidate_pool_status == "insufficient" and witness_groups:
            flags.append({
                "severity": "blocking",
                "type": "insufficient_pool_with_sufficient_witness",
                "task_id": task_id,
                "slot_type": slot_type,
            })

    # 这些不是错误：Strong Teacher 可以记录 optional support。
    if repository_need in {"helpful", "not_needed", "not_applicable"} and (
        witness_groups or supporting_candidates
    ):
        flags.append({
            "severity": "info",
            "type": "optional_repository_support_preserved",
            "task_id": task_id,
            "slot_type": slot_type,
        })

    if applicability == "not_required" and supporting_candidates:
        flags.append({
            "severity": "info",
            "type": "support_for_nonmandatory_slot_preserved",
            "task_id": task_id,
            "slot_type": slot_type,
        })

    return {
        "applicability": applicability,
        "question_coverage": question_coverage,
        "repository_need": repository_need,
        "candidate_pool_status": candidate_pool_status,
        "sufficient_witness_groups": witness_groups,
        "supporting_candidates": supporting_candidates,
        "reason": str(raw_slot.get("reason") or "").strip(),
    }


def normalize_additional_findings(
    *,
    task_id: str,
    raw: Any,
    candidate_count: int,
    identity_errors: list[dict[str, Any]],
    flags: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        flags.append({
            "severity": "warning",
            "type": "additional_findings_not_array",
            "task_id": task_id,
        })
        return []

    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            flags.append({
                "severity": "warning",
                "type": "invalid_additional_finding",
                "task_id": task_id,
                "index": index,
            })
            continue
        result.append({
            "description": str(item.get("description") or "").strip(),
            "candidate_numbers": canonicalize_candidate_number_list(
                item.get("candidate_numbers"),
                candidate_count=candidate_count,
                task_id=task_id,
                location=f"additional_findings[{index}].candidate_numbers",
                identity_errors=identity_errors,
            ),
            "reason": str(item.get("reason") or "").strip(),
        })
    return result


def normalize_result(
    *,
    request: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> dict[str, Any]:
    task_id = str(request["task_id"])
    candidate_ids = request_candidate_ids(request)
    candidate_count = len(candidate_ids)

    flags: list[dict[str, Any]] = []
    identity_errors: list[dict[str, Any]] = []

    slots_raw = raw_result.get("slots")
    if not isinstance(slots_raw, dict):
        slots_raw = {}
        flags.append({
            "severity": "blocking",
            "type": "slots_not_object",
            "task_id": task_id,
        })

    extra_slot_names = sorted(set(map(str, slots_raw)) - set(SLOT_TYPES))
    if extra_slot_names:
        # 不删除 raw_result；这里只提示 extra slot 不会直接映射成新 Obligation。
        flags.append({
            "severity": "info",
            "type": "extra_slot_names_preserved_in_raw",
            "task_id": task_id,
            "slot_names": extra_slot_names,
        })

    slots = {
        slot_type: normalize_slot(
            task_id=task_id,
            slot_type=slot_type,
            raw_slot=slots_raw.get(slot_type),
            candidate_count=candidate_count,
            flags=flags,
            identity_errors=identity_errors,
        )
        for slot_type in SLOT_TYPES
    }

    additional_findings = normalize_additional_findings(
        task_id=task_id,
        raw=raw_result.get("additional_findings"),
        candidate_count=candidate_count,
        identity_errors=identity_errors,
        flags=flags,
    )

    raw_uncertainties = raw_result.get("uncertainties")
    if raw_uncertainties is None:
        uncertainties: list[str] = []
    elif isinstance(raw_uncertainties, list):
        uncertainties = [
            str(item).strip()
            for item in raw_uncertainties
            if str(item).strip()
        ]
    else:
        uncertainties = [str(raw_uncertainties).strip()]
        flags.append({
            "severity": "warning",
            "type": "uncertainties_not_array",
            "task_id": task_id,
        })

    return {
        "task_id": task_id,
        "protocol_version": PROTOCOL_VERSION,
        "candidate_count": candidate_count,
        "candidate_evidence_ids": candidate_ids,
        "overall_assessment": str(
            raw_result.get("overall_assessment") or ""
        ).strip(),
        "slots": slots,
        "additional_findings": additional_findings,
        "uncertainties": uncertainties,
        "semantic_flags": flags,
        "identity_errors": identity_errors,
        # 原始答案完整保留，避免 canonicalization 丢掉模型表达。
        "raw_result": dict(raw_result),
    }


def synthesize_missing_result(request: Mapping[str, Any]) -> dict[str, Any]:
    task_id = str(request["task_id"])
    candidate_ids = request_candidate_ids(request)
    return {
        "task_id": task_id,
        "protocol_version": PROTOCOL_VERSION,
        "candidate_count": len(candidate_ids),
        "candidate_evidence_ids": candidate_ids,
        "overall_assessment": "",
        "slots": {
            slot_type: empty_uncertain_slot("missing_external_result")
            for slot_type in SLOT_TYPES
        },
        "additional_findings": [],
        "uncertainties": ["missing_external_result"],
        "semantic_flags": [{
            "severity": "blocking",
            "type": "missing_external_result",
            "task_id": task_id,
        }],
        "identity_errors": [],
        "raw_result": None,
    }


def validate_unified(args: argparse.Namespace) -> int:
    requests = read_jsonl(args.requests.resolve())
    request_by_task = unique_requests(requests)
    raw_results = read_external_results(args.results.resolve())

    raw_by_task: dict[str, dict[str, Any]] = {}
    duplicate_tasks: set[str] = set()
    error_rows: list[dict[str, Any]] = []

    for index, row in enumerate(raw_results, start=1):
        task_id = str(row.get("task_id") or "").strip()
        if not task_id:
            error_rows.append({
                "stage": "validate-unified",
                "error_type": "MissingTaskId",
                "result_index": index,
                "error": "结果缺少 task_id",
            })
            continue
        if task_id not in request_by_task:
            error_rows.append({
                "task_id": task_id,
                "stage": "validate-unified",
                "error_type": "UnknownTaskId",
                "result_index": index,
                "error": "结果 task_id 不在 requests 中",
            })
            continue
        if task_id in raw_by_task:
            duplicate_tasks.add(task_id)
            error_rows.append({
                "task_id": task_id,
                "stage": "validate-unified",
                "error_type": "DuplicateTaskId",
                "result_index": index,
                "error": "同一个 task_id 出现多个结果，身份含义不唯一",
            })
            continue
        raw_by_task[task_id] = dict(row)

    normalized_rows: list[dict[str, Any]] = []

    for task_id, request in tqdm(
        request_by_task.items(),
        total=len(request_by_task),
        desc="Validate unified（校验统一结果）",
        unit="task",
        dynamic_ncols=True,
    ):
        if task_id in duplicate_tasks:
            normalized = synthesize_missing_result(request)
            normalized["semantic_flags"] = [{
                "severity": "blocking",
                "type": "duplicate_external_result",
                "task_id": task_id,
            }]
            normalized["uncertainties"] = ["duplicate_external_result"]
        elif task_id not in raw_by_task:
            normalized = synthesize_missing_result(request)
            error_rows.append({
                "task_id": task_id,
                "stage": "validate-unified",
                "error_type": "MissingExternalResult",
                "error": "requests 中有任务，但外部结果缺失",
            })
        else:
            normalized = normalize_result(
                request=request,
                raw_result=raw_by_task[task_id],
            )

        normalized_rows.append(normalized)

        for error in normalized.get("identity_errors") or []:
            error_rows.append({
                "task_id": task_id,
                "stage": "candidate_identity",
                "error_type": "CandidateIdentityError",
                "error": stable_json(error),
            })

    output_path = args.output.resolve()
    errors_path = args.errors.resolve()
    atomic_write_jsonl(output_path, normalized_rows)
    atomic_write_jsonl(errors_path, error_rows)

    flag_counts: Counter[str] = Counter()
    slot_applicability: dict[str, Counter[str]] = {
        slot: Counter() for slot in SLOT_TYPES
    }
    slot_repository_need: dict[str, Counter[str]] = {
        slot: Counter() for slot in SLOT_TYPES
    }
    slot_question_coverage: dict[str, Counter[str]] = {
        slot: Counter() for slot in SLOT_TYPES
    }
    slot_pool_status: dict[str, Counter[str]] = {
        slot: Counter() for slot in SLOT_TYPES
    }

    identity_error_task_count = 0
    blocking_flag_task_count = 0

    for row in normalized_rows:
        if row.get("identity_errors"):
            identity_error_task_count += 1
        if any(
            flag.get("severity") == "blocking"
            for flag in (row.get("semantic_flags") or [])
        ):
            blocking_flag_task_count += 1

        for flag in row.get("semantic_flags") or []:
            flag_counts[str(flag.get("type") or "unknown")] += 1

        for slot_type in SLOT_TYPES:
            slot = row["slots"][slot_type]
            slot_applicability[slot_type][slot["applicability"]] += 1
            slot_repository_need[slot_type][slot["repository_need"]] += 1
            slot_question_coverage[slot_type][slot["question_coverage"]] += 1
            slot_pool_status[slot_type][slot["candidate_pool_status"]] += 1

    report = {
        "script_version": SCRIPT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "operation": "validate-unified",
        "created_at": utc_now(),
        "request_task_count": len(request_by_task),
        "raw_result_count": len(raw_results),
        "normalized_task_count": len(normalized_rows),
        "missing_result_count": sum(
            1 for row in normalized_rows
            if row.get("raw_result") is None
        ),
        "duplicate_task_count": len(duplicate_tasks),
        "identity_error_task_count": identity_error_task_count,
        "blocking_semantic_flag_task_count": blocking_flag_task_count,
        "semantic_flag_counts": dict(sorted(flag_counts.items())),
        "slot_applicability": {
            slot: dict(sorted(counter.items()))
            for slot, counter in slot_applicability.items()
        },
        "slot_repository_need": {
            slot: dict(sorted(counter.items()))
            for slot, counter in slot_repository_need.items()
        },
        "slot_question_coverage": {
            slot: dict(sorted(counter.items()))
            for slot, counter in slot_question_coverage.items()
        },
        "slot_candidate_pool_status": {
            slot: dict(sorted(counter.items()))
            for slot, counter in slot_pool_status.items()
        },
        "outputs": {
            "normalized": str(output_path),
            "errors": str(errors_path),
        },
        "important_contract": {
            "raw_teacher_output_preserved": True,
            "optional_support_is_not_discarded": True,
            "unknown_semantics_are_not_guessed": True,
            "candidate_identity_is_strict": True,
            "training_eligible": False,
        },
    }
    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


# ============================================================================
# 4. Unified -> 旧 Programmatic Supervision Adapter
# ============================================================================


def import_merge_module() -> Any:
    """
    复用已经存在的 external_supervision_merge.py：
        - load_runner
        - read_jsonl
        - unique_by_task_id

    新脚本只负责 Strong Teacher 协议与标准化，
    不复制 Candidate Identity Lock / runner loader 的另一套实现。
    """
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    try:
        import external_supervision_merge as merge
    except ImportError as exc:
        raise RuntimeError(
            "找不到 scripts/external_supervision_merge.py。"
            "请先把上一阶段生成的 merge 脚本放入 scripts/。"
        ) from exc

    return merge


def result_has_blocking_flags(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    blockers = [
        dict(flag)
        for flag in (result.get("semantic_flags") or [])
        if str(flag.get("severity") or "") == "blocking"
    ]
    for error in result.get("identity_errors") or []:
        blockers.append({
            "severity": "blocking",
            "type": "candidate_identity_error",
            "detail": error,
        })
    return blockers


def adapt_slot_to_legacy(
    *,
    task_id: str,
    slot_type: str,
    slot: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    将 richer Strong-Teacher 语义映射回 v1.9.2.1 的旧接口。

    返回：
        requirement_slot
        witness_slot_or_none
        blockers
        needs_more

    关键原则：
        这里不是重新“判断语义”，而只是把足够明确的 Strong-Teacher 结果
        投影到旧 Programmatic Builder 能理解的最小接口。

    optional/supporting evidence 不进入 mandatory witness groups，
    但完整保留在 unified_result sidecar 中。
    """
    applicability = str(slot.get("applicability") or "uncertain")
    question_coverage = str(slot.get("question_coverage") or "uncertain")
    repository_need = str(slot.get("repository_need") or "uncertain")
    pool_status = str(slot.get("candidate_pool_status") or "uncertain")
    witness_groups = list(slot.get("sufficient_witness_groups") or [])
    reason = str(slot.get("reason") or "")

    blockers: list[dict[str, Any]] = []
    needs_more: list[dict[str, Any]] = []

    def requirement(decision: str) -> dict[str, Any]:
        return {
            "decision": decision,
            "reason": reason,
        }

    if applicability == "uncertain":
        blockers.append({
            "slot_type": slot_type,
            "reason": "applicability_uncertain",
        })
        return requirement("uncertain"), None, blockers, needs_more

    if applicability == "not_required":
        if repository_need == "required":
            blockers.append({
                "slot_type": slot_type,
                "reason": "self_conflict_not_required_vs_repository_required",
            })
            return requirement("uncertain"), None, blockers, needs_more
        return requirement("not_required"), None, blockers, needs_more

    # applicability == required
    if repository_need == "required":
        req = requirement("repository_required")

        if pool_status == "insufficient":
            needs_more.append({
                "slot_type": slot_type,
                "reason": "required_repository_context_missing_from_candidate_pool",
            })
            witness = {
                "status": "agreed",
                "selection_status": "insufficient",
                "witness_groups": [],
                "reason": reason,
            }
            return req, witness, blockers, needs_more

        if pool_status == "sufficient":
            if not witness_groups:
                blockers.append({
                    "slot_type": slot_type,
                    "reason": "pool_sufficient_but_no_sufficient_witness_groups",
                })
                witness = {
                    "status": "agreed",
                    "selection_status": "uncertain",
                    "witness_groups": [],
                    "reason": reason,
                }
                return req, witness, blockers, needs_more

            witness = {
                "status": "agreed",
                "selection_status": "select",
                "witness_groups": witness_groups,
                "reason": reason,
            }
            return req, witness, blockers, needs_more

        blockers.append({
            "slot_type": slot_type,
            "reason": f"repository_required_but_pool_status_{pool_status}",
        })
        witness = {
            "status": "agreed",
            "selection_status": "uncertain",
            "witness_groups": [],
            "reason": reason,
        }
        return req, witness, blockers, needs_more

    if repository_need in {"helpful", "not_needed", "not_applicable"}:
        # 这个维度是 required，但仓库 Evidence 不是必要条件。
        # 要安全映射成旧 question_satisfied，必须明确 Question 已经充分覆盖。
        if question_coverage == "sufficient":
            return requirement("question_satisfied"), None, blockers, needs_more

        blockers.append({
            "slot_type": slot_type,
            "reason": (
                "required_dimension_not_fully_covered_by_question_"
                f"and_repository_not_required:{question_coverage}"
            ),
        })
        return requirement("uncertain"), None, blockers, needs_more

    blockers.append({
        "slot_type": slot_type,
        "reason": "repository_need_uncertain",
    })
    return requirement("uncertain"), None, blockers, needs_more


def adapt_unified_to_two_stage(
    result: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    task_id = str(result.get("task_id") or "")
    requirement_slots: dict[str, Any] = {}
    witness_by_slot: dict[str, Any] = {}
    blockers = result_has_blocking_flags(result)
    needs_more: list[dict[str, Any]] = []

    for slot_type in SLOT_TYPES:
        req, witness, slot_blockers, slot_needs_more = adapt_slot_to_legacy(
            task_id=task_id,
            slot_type=slot_type,
            slot=result["slots"][slot_type],
        )
        requirement_slots[slot_type] = req
        if witness is not None:
            witness_by_slot[slot_type] = witness
        blockers.extend(slot_blockers)
        needs_more.extend(slot_needs_more)

    requirement_consensus = {
        "status": "agreed",
        "stage": "external_unified_strong_teacher",
        "slot_results": requirement_slots,
        "independent_semantic_review": False,
    }

    return requirement_consensus, witness_by_slot, blockers, needs_more


def build_refinement(args: argparse.Namespace) -> int:
    merge = import_merge_module()
    runner = merge.load_runner(args.runner)

    contexts = merge.read_jsonl(args.context.resolve())
    context_by_task = merge.unique_by_task_id(contexts, source="context")
    results = merge.read_jsonl(args.results.resolve())
    result_by_task = merge.unique_by_task_id(results, source="unified results")

    if set(context_by_task) != set(result_by_task):
        raise ValueError(
            "context 与 unified results task 集不一致："
            f"missing={sorted(set(context_by_task) - set(result_by_task))[:10]}, "
            f"extra={sorted(set(result_by_task) - set(context_by_task))[:10]}"
        )

    output_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    verification_counts: Counter[str] = Counter()
    semantic_flag_counts: Counter[str] = Counter()

    for task_id in tqdm(
        context_by_task,
        total=len(context_by_task),
        desc="Strong Teacher gate（强教师质量门）",
        unit="task",
        dynamic_ncols=True,
    ):
        context = context_by_task[task_id]
        result = result_by_task[task_id]

        # 第二层 Candidate Identity Lock：
        # validator 中记录的 Candidate ID 顺序必须与 prepare-context 完全一致。
        result_candidate_ids = list(map(str, result.get("candidate_evidence_ids") or []))
        context_candidate_ids = list(map(str, context.get("candidate_ids") or []))
        identity_blockers: list[dict[str, Any]] = []
        if result_candidate_ids != context_candidate_ids:
            identity_blockers.append({
                "severity": "blocking",
                "type": "candidate_identity_lock_mismatch",
                "result_candidate_ids": result_candidate_ids,
                "context_candidate_ids": context_candidate_ids,
            })

        requirement_consensus, witness_by_slot, blockers, needs_more = (
            adapt_unified_to_two_stage(result)
        )
        blockers.extend(identity_blockers)

        for flag in result.get("semantic_flags") or []:
            semantic_flag_counts[str(flag.get("type") or "unknown")] += 1

        final_consensus = runner.build_two_stage_final_consensus(
            requirement_consensus=requirement_consensus,
            witness_consensus_by_slot=witness_by_slot,
        )

        base_record = {
            "task_id": task_id,
            "external_teacher_protocol": {
                "protocol_version": PROTOCOL_VERSION,
                "teacher_source": "human-selected external strong web model",
                "single_pass": True,
                "candidate_blind": False,
                "rich_semantics_preserved": True,
                "optional_support_preserved": True,
                "candidate_number_binding": (
                    "identity-locked to pre-fix Candidate Evidence IDs"
                ),
                "training_promotion": "disabled",
            },
            "unified_teacher_result": result,
            "legacy_requirement_adapter": requirement_consensus,
            "legacy_witness_adapter": witness_by_slot,
            "two_stage_final_consensus": final_consensus,
            "original_supervision": context.get("supervision") or {},
            "candidate_ids": context_candidate_ids,
            "offline_gold_reference_used": bool(
                context.get("offline_gold_reference_used")
            ),
            "candidate_identity_lock": context.get("identity_lock") or {},
            "training_eligible": False,
        }

        # Identity / ambiguous semantics 优先于 NEEDS_MORE：
        # 只有确认“唯一未满足原因就是 Candidate Pool 缺必要上下文”时，
        # 才能标 NEEDS_MORE。
        if blockers:
            status = "BLOCKED"
            status_counts[status] += 1
            reasons = blockers + needs_more
            error_rows.append({
                "task_id": task_id,
                "stage": "strong_teacher_quality_gate",
                "error_type": "BlockedUnifiedDecision",
                "error": stable_json(reasons),
            })
            output_rows.append({
                **base_record,
                "final_status": status,
                "status_reasons": reasons,
                "proposal": None,
                "verification": None,
                "supervision_verified": False,
            })
            continue

        if needs_more:
            status = "NEEDS_MORE"
            status_counts[status] += 1
            output_rows.append({
                **base_record,
                "final_status": status,
                "status_reasons": needs_more,
                "proposal": None,
                "verification": None,
                "supervision_verified": False,
            })
            continue

        # 到这里 Strong Teacher 的必要语义已经可以无歧义投影到旧 Programmatic Builder。
        # 真正的 Evidence ID / Obligation / Certificate / STOP 仍由程序所有。
        try:
            proposal, verification, construction_error = (
                runner.build_programmatic_refinement_v1_9_2(
                    task_id=task_id,
                    supervision=context["supervision"],
                    candidate_records=context["candidate_records"],
                    candidate_ids=context_candidate_ids,
                    existing_evidence_ids=set(
                        map(
                            str,
                            context.get("existing_evidence_ids")
                            or context_candidate_ids,
                        )
                    ),
                    token_costs={
                        str(key): int(value)
                        for key, value in (context.get("token_costs") or {}).items()
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
            error_rows.append({"task_id": task_id, **construction_error})
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

        if verification_status == "accepted":
            status = "VERIFIED"
            verified = True
            reasons: list[dict[str, Any]] = []
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
    refinement_path = output_dir / "strong_teacher_refinement.jsonl"
    errors_path = output_dir / "strong_teacher_refinement_errors.jsonl"
    report_path = output_dir / "strong_teacher_refinement_report.json"

    atomic_write_jsonl(refinement_path, output_rows)
    atomic_write_jsonl(errors_path, error_rows)

    report = {
        "script_version": SCRIPT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "operation": "build-refinement",
        "created_at": utc_now(),
        "runner_version": str(runner.RUNNER_VERSION),
        "task_count": len(context_by_task),
        "final_status_counts": dict(sorted(status_counts.items())),
        "verification_status_counts": dict(sorted(verification_counts.items())),
        "semantic_flag_counts": dict(sorted(semantic_flag_counts.items())),
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
            "teacher_is_single_pass_strong_model": True,
            "old_small_model_hard_coupling_removed": True,
            "optional_support_preserved": True,
            "additional_findings_preserved": True,
            "candidate_identity_strict": True,
            "pre_fix_evidence_only_via_candidate_binding": True,
            "core_accepted_is_not_semantic_truth": True,
            "teacher_output_is_not_direct_training_label": True,
            "v2_10_modified": False,
        },
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


# ============================================================================
# 5. Self-check
# ============================================================================


def self_check(_: argparse.Namespace) -> int:
    request = {
        "task_id": "task_demo",
        "base_user_prompt": (
            "[ISSUE / QUESTION]\nFix cache invalidation.\n\n"
            "[CANDIDATE EVIDENCE POOL]\n"
            "Candidate Number: 1\nPath: a.py\nContent: x\n\n"
            "Candidate Number: 2\nPath: b.py\nContent: y\n\n"
            "[OUTPUT]\nold contract"
        ),
        "candidate_diagnostics": {
            "selected_evidence_ids": ["ev1", "ev2"],
        },
    }

    prompt = build_unified_prompt(request)
    assert "old contract" not in prompt
    assert "Candidate Number: 1" in prompt
    assert "UNIFIED STRONG-TEACHER OUTPUT" in prompt

    raw = {
        "task_id": "task_demo",
        "overall_assessment": "demo",
        "slots": {
            slot_type: {
                "applicability": "not_required",
                "question_coverage": "not_applicable",
                "repository_need": "not_needed",
                "candidate_pool_status": "not_needed",
                "sufficient_witness_groups": [],
                "supporting_candidates": [],
                "reason": "demo",
            }
            for slot_type in SLOT_TYPES
        },
        "additional_findings": [],
        "uncertainties": [],
    }

    # fault_location 需要 repository，Candidate 1 单独充分。
    raw["slots"]["fault_location"] = {
        "applicability": "required",
        "question_coverage": "partial",
        "repository_need": "required",
        "candidate_pool_status": "sufficient",
        "sufficient_witness_groups": [[1]],
        "supporting_candidates": [2],
        "reason": "需要查看实现位置",
    }

    # behavior 已由 Question 满足，但 Candidate 2 仍可作为 optional support。
    raw["slots"]["behavior_constraint"] = {
        "applicability": "required",
        "question_coverage": "sufficient",
        "repository_need": "helpful",
        "candidate_pool_status": "not_needed",
        "sufficient_witness_groups": [],
        "supporting_candidates": [2],
        "reason": "Issue 已明确行为，代码仍有辅助价值",
    }

    normalized = normalize_result(request=request, raw_result=raw)
    assert not normalized["identity_errors"]
    assert normalized["slots"]["behavior_constraint"]["supporting_candidates"] == [2]

    req, witness, blockers, needs_more = adapt_unified_to_two_stage(normalized)
    assert not blockers, blockers
    assert not needs_more, needs_more
    assert req["slot_results"]["fault_location"]["decision"] == "repository_required"
    assert witness["fault_location"]["selection_status"] == "select"
    assert req["slot_results"]["behavior_constraint"]["decision"] == "question_satisfied"

    # Candidate 越界必须形成 identity error，且不能把 [1, 99] 错修成 [1]。
    raw_bad = json.loads(json.dumps(raw))
    raw_bad["slots"]["fault_location"]["sufficient_witness_groups"] = [[1, 99]]
    bad = normalize_result(request=request, raw_result=raw_bad)
    assert bad["identity_errors"]
    assert bad["slots"]["fault_location"]["sufficient_witness_groups"] == []

    # repository required + insufficient 应明确进入 NEEDS_MORE 语义。
    raw_more = json.loads(json.dumps(raw))
    raw_more["slots"]["fault_location"]["candidate_pool_status"] = "insufficient"
    raw_more["slots"]["fault_location"]["sufficient_witness_groups"] = []
    more = normalize_result(request=request, raw_result=raw_more)
    _req, _wit, blockers, needs_more = adapt_unified_to_two_stage(more)
    assert not blockers, blockers
    assert needs_more, needs_more

    print(
        json.dumps(
            {
                "script_version": SCRIPT_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "self_check": "passed",
                "checks": [
                    "old output contract removed",
                    "full candidate context preserved",
                    "optional support preserved",
                    "required repository witness mapped",
                    "question-covered slot mapped without mandatory witness",
                    "candidate out-of-range blocks without corrupting AND semantics",
                    "candidate insufficiency maps to NEEDS_MORE",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


# ============================================================================
# 6. CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified Strong-Teacher external supervision protocol: "
            "one-pass export, tolerant semantic validation, strict candidate identity, "
            "and programmatic Core v1.7 merge."
        )
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION} / {PROTOCOL_VERSION}",
        help="显示脚本版本与 Strong-Teacher 协议版本后退出。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser(
        "export-unified",
        help="把旧 refinement_requests 一次性导出给网页版 Strong Teacher。",
    )
    export.add_argument("--requests", type=Path, required=True)
    export.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/upstream/external_supervision/strong_teacher"),
    )
    export.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "默认每个 Markdown 1 task。一次性指的是每个 task 同时完成 "
            "Requirement + Witness，而不是必须把多个长任务塞进一个上下文。"
        ),
    )
    export.set_defaults(func=export_unified)

    export_all_parser = subparsers.add_parser(
        "export-all",
        help=(
            "从冻结 V2.10 直接导出 train+validation+benchmark 全量 Strong-Teacher Markdown；"
            "默认 20,864 tasks、每 task 一个 Markdown。"
        ),
    )
    export_all_parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/refine_supervision_with_llm.py"),
        help="必须是 RUNNER_VERSION=1.9.2.1 的当前 runner。",
    )
    export_all_parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/upstream/unified_swe_dataset_v2_10"),
    )
    export_all_parser.add_argument(
        "--evidence-cache",
        type=Path,
        default=None,
        help="默认继承 v1.9.2.1 runner。",
    )
    export_all_parser.add_argument(
        "--build-db",
        type=Path,
        default=None,
        help="默认继承 v1.9.2.1 runner。",
    )
    export_all_parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation", "benchmark"],
        default=["train", "validation", "benchmark"],
        help="默认导出完整 20,864 tasks。",
    )
    export_all_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/upstream/external_supervision/strong_teacher_v1_3_all"),
    )
    export_all_parser.add_argument(
        "--tasks-per-md",
        type=int,
        default=1,
        help="默认每个 Markdown 仅 1 task；网页版强模型建议保持 1。",
    )
    export_all_parser.add_argument(
        "--prepare-workers",
        type=int,
        default=4,
        help=(
            "并行 Candidate/Prompt 准备线程数，默认 4。"
            "每个 worker 使用独立只读 SQLite connection，不共享 connection。"
            "NVMe 可尝试 6~8；HDD 建议 1~2。"
        ),
    )
    export_all_parser.add_argument(
        "--prepare-chunk-size",
        type=int,
        default=32,
        help=(
            "兼容 v1.4.2 的旧参数。v1.4.3 在 --prepare-workers > 1 时使用"
            "持久 contiguous shard worker，每个 worker 整个 shard 只打开一次 SQLite，"
            "因此该参数不再参与并行调度；workers=1 时也无需设置。"
        ),
    )
    export_all_parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help=(
            "默认冻结 V2.10 必须匹配 train=18347, validation=223, benchmark=2294。"
            "仅在你明确知道 release 行数不同且仍希望导出时使用。"
        ),
    )
    add_full_export_candidate_args(export_all_parser)
    export_all_parser.set_defaults(func=export_all)

    validate = subparsers.add_parser(
        "validate-unified",
        help="校验 Strong Teacher JSON；语义尽量保留，Candidate 身份严格。",
    )
    validate.add_argument("--requests", type=Path, required=True)
    validate.add_argument("--results", type=Path, required=True)
    validate.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/upstream/external_supervision/strong_teacher/results.normalized.jsonl"
        ),
    )
    validate.add_argument(
        "--errors",
        type=Path,
        default=Path(
            "data/upstream/external_supervision/strong_teacher/results.errors.jsonl"
        ),
    )
    validate.set_defaults(func=validate_unified)

    build = subparsers.add_parser(
        "build-refinement",
        help="把统一 Strong Teacher 结果接回 v1.9.2.1 + Core v1.7。",
    )
    build.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/refine_supervision_with_llm.py"),
        help="必须是 RUNNER_VERSION=1.9.2.1。",
    )
    build.add_argument("--context", type=Path, required=True)
    build.add_argument("--results", type=Path, required=True)
    build.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/upstream/external_supervision/strong_teacher/refinement"
        ),
    )
    build.set_defaults(func=build_refinement)

    check = subparsers.add_parser(
        "self-check",
        help="纯 Python 协议自检，不需要数据集/API/GPU。",
    )
    check.set_defaults(func=self_check)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
