#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strong-Teacher Markdown Prompt Refresher v1.4
（Strong-Teacher Markdown 提示词原地升级工具）

目标：
    只修改已经导出的 Markdown 任务文件中的 Strong-Teacher Prompt。

严格不做：
    - 不重新读取 V2.10 dataset；
    - 不打开 60GB build SQLite；
    - 不运行 Candidate Builder；
    - 不重新生成 requests.jsonl；
    - 不重新生成 merge_context.jsonl；
    - 不重新生成 tasks.jsonl；
    - 不重新生成 export_report.json / export_all_report.json；
    - 不修改 Candidate Pool / Issue / Gold Hints / task_id / 文件名。

默认输入目录：
    data/upstream/external_supervision/strong_teacher_v1_3_all/

目录结构：
    train/md/*.md
    validation/md/*.md
    benchmark/md/*.md

本版只强化训练监督真正关心的 Witness 质量：
    1. Sufficiency（充分性）
    2. Minimality（AND 最小性）
    3. OR Completeness（替代最小充分组合齐全性）
    4. Gold Independence（Witness 不依赖 Gold 才成立）
    5. Execution Relevance（因果/状态 Witness 必须参与所声称机制）
    6. Conservative Empty Set（不确定时宁空不造错 Witness）

用法（PowerShell）：

    python scripts/refresh_strong_teacher_md_v1_4.py `
      --root data/upstream/external_supervision/strong_teacher_v1_3_all

先检查不写盘：

    python scripts/refresh_strong_teacher_md_v1_4.py `
      --root data/upstream/external_supervision/strong_teacher_v1_3_all `
      --dry-run
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError as exc:
    raise RuntimeError(
        "缺少 tqdm（进度条依赖）。请执行：python -m pip install -U tqdm"
    ) from exc


SCRIPT_VERSION = "1.4.0"
PROMPT_CONTRACT_VERSION = "minimal-sufficient-witness-v1.4"

DEFAULT_ROOT = Path("data/upstream/external_supervision/strong_teacher_v1_3_all")
DEFAULT_SPLITS = ("train", "validation", "benchmark")


# ---------------------------------------------------------------------------
# 规则 12~13：直接替换旧的 OR-of-AND 规则。
#
# 设计说明：
#   旧 Prompt 已经要求“每个 OR alternative 独立充分”，但没有强制：
#     - 先找 singleton；
#     - AND 做 deletion test；
#     - 消除非最小超集；
#     - 找完一个 group 后继续搜索其他独立充分 alternative。
#
#   这会导致 Teacher 把多个“相关 Candidate”堆成宽 AND，或者只给一个可用组合
#   而遗漏其他合法 OR action。
#
#   新合同把 sufficient_witness_groups 明确定义为：
#       Candidate Pool 中所有 materially distinct（实质不同）的
#       minimal sufficient witness groups（最小充分 Witness 组合）。
# ---------------------------------------------------------------------------
WITNESS_RULES_12_13 = r'''12. Minimal Sufficient Witness Search（最小充分 Witness 搜索）。
    sufficient_witness_groups 的目标不是列出“相关代码”，而是枚举当前 Candidate Pool 中
    所有能够独立满足该 slot repository requirement 的、实质不同的最小充分 Candidate 集合。

    对每个 repository_need=required 的 slot，必须静默执行以下过程：

    A. Required-Fact Decomposition（必要事实分解）
       先明确：要使该 slot 达到充分性，下游修复模型必须知道哪些必要语义事实。
       “相关”“可能有帮助”“位于修改文件”“出现在调用链附近”都不等于必要事实。

    B. Singleton Scan（单 Candidate 扫描）
       必须先逐一检查 Candidate Pool 中每个 Candidate：
       Issue / Question + 该 Candidate 是否已经足以满足该 slot？
       如果是，则 [Candidate] 本身就是一个独立 OR alternative。
       不得因为还有其他相关 Candidate，就把本可单独充分的 [A] 扩写成 [A,B] 或 [A,B,C]。

    C. Complementary Combination Search（互补组合搜索）
       只有单个 Candidate 不充分时，才组合提供互补必要事实的 Candidate。
       一个 AND group 必须整体覆盖该 slot 所需的全部必要语义；
       只是背景、旁证、增强解释的 Candidate 应放 supporting_candidates。

    D. Deletion / Minimality Test（删除 / 最小性测试）
       对每个准备输出的 group [A,B,...]，必须逐个删除其中 Candidate 并重新判断充分性：
       - 若删除 A 后其余 Candidate 仍然 sufficient，则 A 是冗余成员，必须删除；
       - 对 group 中每个 Candidate 都执行同样测试；
       - 只有“删除任意一个成员都会使该组不再充分”的组合，才允许作为 AND group 输出。

       因此：
       - 如果 [A] sufficient，则禁止同时输出 [A,B]、[A,B,C] 等包含 A 的冗余超集；
       - 如果 [A,B] sufficient，则除非 C 不可缺，否则禁止输出 [A,B,C]。

    E. Alternative Search / OR Completeness（替代方案搜索 / OR 完整性）
       找到一个 minimal sufficient group 后不得停止。
       必须重新扫描整个 Candidate Pool，检查是否还存在其他 Candidate 或 Candidate 组合，
       能够不依赖已有 group 而独立满足同一 slot。
       所有实质不同、独立成立的 minimal sufficient groups 都应作为 OR alternatives 输出。
       不要为了增加 OR 数量制造语义等价的冗余超集或仅仅“也相关”的组合。

    F. Superset Elimination（超集消除）
       最终输出前，对所有 group 做集合包含检查。
       如果 group X 是另一个已充分 group Y 的真超集，则 X 不是 minimal，必须删除。
       例如 [[2], [2,5], [7,9]] 必须规范化为 [[2], [7,9]]。

    G. Gold Independence Test（Gold 独立性测试）
       判断某个 Candidate/group 是否 sufficient 时，必须假设 Gold Change Hints 被隐藏。
       如果一个组合只有依赖 Gold 才能被认为充分，则它不能进入 sufficient_witness_groups。
       Gold 只能用于提醒你重新扫描可能遗漏的 pre-fix Candidate，不能证明 Candidate 是充分 Witness，
       也不能证明某个 Candidate 是 AND 中不可删除的成员。

    H. Execution Relevance（执行相关性）
       对 fault_logic / state_flow 等因果或传播型 slot：
       如果某 Candidate 描述的代码在 Issue 对应执行路径或触发条件下并未参与所声称机制，
       则不得把它作为该因果链的 sufficient Witness；必要时只放 supporting_candidates。
       “代码看起来可疑”不能替代执行路径证据。

    I. Conservative Empty Set（保守空集）
       如果完成上述检查后仍无法可靠确认任何 Candidate 集合充分，
       应返回 candidate_pool_status=uncertain 或 insufficient，并保持 sufficient_witness_groups=[]。
       宁可不给 Witness，也不要制造一个“看起来合理”的错误 AND/OR 组合。

13. Final Witness Canonicalization（最终 Witness 规范化）。
    reason 与 sufficient_witness_groups 必须自洽，但最终训练监督以 Witness 集合正确性为核心。
    输出前必须对每个 repository-required slot 静默检查：
    - 每个外层 OR group 是否真的可以独立 sufficient？
    - 每个 AND group 是否通过 deletion test，成员全部不可删除？
    - 是否存在某个 group 的 proper subset（真子集）已经 sufficient？若有，删除超集；
    - 是否遗漏 independently sufficient singleton（可单独充分的 Candidate）？
    - 是否遗漏其他实质不同的 independently sufficient minimal combination？
    - 是否把 supporting / related / Gold-adjacent Candidate 错塞进 AND？
    - 是否有 Candidate 的“充分性”实际上依赖 Gold Change Hints 才成立？
    - fault_logic / state_flow 中的 Candidate 是否真的参与所声称的执行机制？
    - 如果 reason 使用“必要 / 必须 / indispensable”，该 Candidate 是否确实出现在所有依赖该事实的最小充分组合中？

    只有全部通过后，才能输出 sufficient_witness_groups。
'''.strip()


# ---------------------------------------------------------------------------
# Rule 21：把最终静默自检升级为 Witness-first（Witness 优先）版本。
# 保留原来的枚举、语言、Claim-Uncertainty 等检查，但明确最先检查 AND/OR。
# ---------------------------------------------------------------------------
FINAL_SELF_CHECK_21 = r'''21. 最终输出前做一次静默自检（不要把自检过程输出到 JSON）：
    Witness / OR-of-AND 核心检查：
    - 每个 sufficient_witness_groups 外层 alternative 是否独立充分？
    - 是否先检查过 singleton，而不是直接把多个相关 Candidate 堆成 AND？
    - 每个多成员 AND 是否逐成员通过 deletion test？
    - 是否存在已输出 group 的真子集其实已经 sufficient？
    - 是否遗漏其他可单独充分的 Candidate？
    - 是否遗漏其他实质不同的 minimal sufficient combination？
    - 是否把 supporting / related Candidate 错塞进 sufficient_witness_groups？
    - 在隐藏 Gold Change Hints 后，各 Witness group 是否仍然成立？
    - fault_logic / state_flow 的 Witness 是否真的参与 Issue 对应执行路径或触发机制？
    - 如果无法确认充分性，是否正确使用 uncertain / insufficient + []，而不是猜一个组合？

    其他一致性检查：
    - 是否把 traceback symbol 错当成 Gold repair site？
    - 是否存在 Candidate 明明在池中却被说成缺失？
    - reason 中的“必要” Candidate 是否与最小充分 witness group 一致？
    - 是否把推测写成了已证实根因？
    - uncertainties 是否与 candidate_pool_status / reason 互相矛盾？
    - 所有枚举值是否严格属于对应字段，尤其 applicability 是否误用了 helpful？
    - candidate_pool_status 是否只使用 sufficient / insufficient / uncertain / not_needed，绝不能使用 not_applicable？
    - 若 uncertainties 声明某机制仍未知，其他解释字段是否错误地把同一机制写成确定事实？
    - behavior_constraint 是否只按 Issue 的外部期望行为判断，而没有被实现细节错误降级？
    - 根因结论是否依赖 Candidate Pool 中未提供实现的关键函数；若依赖，是否错误写成确定事实？
    - dependency_context 是否被误解为“只指外部 package dependency”；它也可以包括修复所需的内部模块、API、调用、注册、集成依赖上下文？
    - overall_assessment / reason / additional_findings / uncertainties 是否全部使用简体中文？
'''.strip()


# ---------------------------------------------------------------------------
# Output contract 追加的最终 Witness 规范。
# 这里不改变 JSON schema，只强化 sufficient_witness_groups 的语义。
# ---------------------------------------------------------------------------
OUTPUT_WITNESS_NOTES = r'''- sufficient_witness_groups 的目标是枚举“所有实质不同的 minimal sufficient witness groups”，不是罗列相关 Candidate。
- 生成 Witness 时必须先做 singleton scan；单个 Candidate 已充分时，不得无故扩成多成员 AND。
- 每个多成员 AND group 必须通过 deletion test：删除任意成员后都应不再充分，否则删除冗余成员。
- 如果一个已充分 group 是另一个 group 的真子集，删除较大的冗余超集。
- 找到一个充分 group 后必须继续扫描 Candidate Pool，寻找其他独立成立的 minimal OR alternatives；不能找到一个就停止。
- 判断 Witness 充分性时必须假设 Gold Change Hints 被隐藏；Gold 只能用于提示重扫，不能证明 Witness 充分或 AND 成员不可删除。
- 对 fault_logic / state_flow，若 Candidate 不参与 Issue 对应执行路径或触发机制，不得作为该因果/传播链的 sufficient Witness。
- 无法可靠确认任何充分组合时，宁可 candidate_pool_status=uncertain/insufficient 且 sufficient_witness_groups=[]，不要猜测 AND/OR。
'''.strip()


RULE_12_START = "12. sufficient_witness_groups 表示严格的 OR-of-AND 充分性："
RULE_14_START = "14. 不要把“相关”写成“因果已证实”。"
RULE_21_START = "21. 最终输出前做一次静默自检（不要把自检过程输出到 JSON）："
CANONICAL_START = "7 个 canonical dimensions（标准维度）："
OUTPUT_INSERT_BEFORE = "- reason 中如果使用“必要/必须”等措辞，必须与 sufficient_witness_groups 对应；"


def replace_between(text: str, start: str, end: str, replacement: str) -> tuple[str, bool]:
    """按唯一文本边界替换，避免用宽泛 regex 误伤 Candidate / Issue 正文。"""
    start_index = text.find(start)
    if start_index < 0:
        return text, False
    end_index = text.find(end, start_index)
    if end_index < 0:
        return text, False
    new_text = text[:start_index] + replacement.rstrip() + "\n\n" + text[end_index:]
    return new_text, True


def _replace_all_bounded_blocks(
    text: str,
    *,
    start: str,
    end: str,
    replacement: str,
) -> tuple[str, int]:
    """
    在一次 regex pass 中替换所有标准块。

    不能用 "while find(start)"：新 Rule 21 故意保留相同标题，
    循环替换会再次命中新文本自身。re.subn 不会递归处理本轮 replacement，
    因此既支持 multi-task Markdown，又不会自匹配死循环。
    """
    pattern = re.compile(
        re.escape(start) + r".*?(?=" + re.escape(end) + r")",
        flags=re.DOTALL,
    )
    return pattern.subn(replacement.rstrip() + "\n\n", text)


def patch_one_markdown(text: str) -> tuple[str, dict[str, int]]:
    """
    只修改 Strong-Teacher 规则文字，不修改 TASK CONTEXT。

    支持同一 Markdown 内多个 TASK；所有规则替换均在单次 pass 中完成。
    """
    stats = {
        "rule_12_13": 0,
        "rule_21": 0,
        "output_notes": 0,
    }

    # 已升级文件必须幂等：再次运行不重复插入。
    if "Minimal Sufficient Witness Search（最小充分 Witness 搜索）" in text:
        return text, stats

    text, stats["rule_12_13"] = _replace_all_bounded_blocks(
        text,
        start=RULE_12_START,
        end=RULE_14_START,
        replacement=WITNESS_RULES_12_13,
    )

    text, stats["rule_21"] = _replace_all_bounded_blocks(
        text,
        start=RULE_21_START,
        end=CANONICAL_START,
        replacement=FINAL_SELF_CHECK_21,
    )

    # OUTPUT contract 每个 TASK 各有一份。只在旧插入点前加入一次新说明。
    search_from = 0
    while True:
        idx = text.find(OUTPUT_INSERT_BEFORE, search_from)
        if idx < 0:
            break
        # 如果前方紧邻区域已经包含新 note，则跳过，保证局部幂等。
        window = text[max(0, idx - 3500):idx]
        if "所有实质不同的 minimal sufficient witness groups" not in window:
            text = text[:idx] + OUTPUT_WITNESS_NOTES + "\n" + text[idx:]
            stats["output_notes"] += 1
            search_from = idx + len(OUTPUT_WITNESS_NOTES) + 1 + len(OUTPUT_INSERT_BEFORE)
        else:
            search_from = idx + len(OUTPUT_INSERT_BEFORE)

    return text, stats


def validate_patched_markdown(path: Path, original: str, patched: str, stats: dict[str, int]) -> None:
    """
    Hard Fail（硬失败）条件。

    这里宁可拒绝修改，也不能“猜着替换”，因为 20k supervision 的 Prompt 污染
    比少改一个文件更危险。
    """
    task_count = len(re.findall(r"^TASK\s+\d+\s+—\s+", original, flags=re.MULTILINE))
    if task_count == 0:
        raise ValueError(f"{path}: 未找到 TASK header，不像标准 Strong-Teacher Markdown")

    for key, count in stats.items():
        if count != task_count:
            raise ValueError(
                f"{path}: {key} 替换次数={count}，但 TASK 数={task_count}；"
                "为避免误改，已拒绝写盘"
            )

    # 关键任务上下文 marker 数量必须保持不变。
    protected_markers = (
        "[TASK CONTEXT]",
        "[ISSUE / QUESTION]",
        "[CANDIDATE EVIDENCE POOL]",
        "[GOLD CHANGE HINTS - OFFLINE ONLY, NOT EVIDENCE]",
        "当前 task_id 必须原样复制为：",
    )
    for marker in protected_markers:
        before = original.count(marker)
        after = patched.count(marker)
        if before != after:
            raise ValueError(
                f"{path}: 受保护 marker {marker!r} 数量发生变化：{before} -> {after}"
            )

    # 新合同必须按 TASK 数出现。
    if patched.count("Minimal Sufficient Witness Search（最小充分 Witness 搜索）") != task_count:
        raise ValueError(f"{path}: 新 Witness 合同数量与 TASK 数不一致")
    if patched.count("Deletion / Minimality Test（删除 / 最小性测试）") != task_count:
        raise ValueError(f"{path}: deletion test 规则数量与 TASK 数不一致")
    if patched.count("Alternative Search / OR Completeness（替代方案搜索 / OR 完整性）") != task_count:
        raise ValueError(f"{path}: OR completeness 规则数量与 TASK 数不一致")


def atomic_write_text(path: Path, text: str) -> None:
    """
    原子覆盖同一个 .md 文件。

    临时文件只在写入期间短暂存在，成功后 replace 原文件；不会产生任何持久 sidecar。
    """
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def collect_md_files(root: Path, splits: list[str]) -> list[Path]:
    paths: list[Path] = []
    for split in splits:
        md_dir = root / split / "md"
        if not md_dir.is_dir():
            raise FileNotFoundError(f"缺少目录：{md_dir}")
        paths.extend(sorted(md_dir.glob("*.md")))
    return paths


def run(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    paths = collect_md_files(root, list(args.splits))
    if not paths:
        raise ValueError(f"没有找到 Markdown：{root}")

    changed = 0
    already_new = 0
    failed: list[tuple[Path, str]] = []

    for path in tqdm(
        paths,
        desc="Refresh Strong-Teacher MD（升级提示词）",
        unit="md",
        dynamic_ncols=True,
    ):
        try:
            original = path.read_text(encoding="utf-8-sig")
            if "Minimal Sufficient Witness Search（最小充分 Witness 搜索）" in original:
                already_new += 1
                continue

            patched, stats = patch_one_markdown(original)
            validate_patched_markdown(path, original, patched, stats)

            if patched == original:
                raise ValueError("没有产生任何修改")

            if not args.dry_run:
                atomic_write_text(path, patched)
            changed += 1
        except Exception as exc:  # 单文件失败不污染其他文件；最终返回非 0。
            failed.append((path, f"{type(exc).__name__}: {exc}"))

    print()
    print(f"script_version={SCRIPT_VERSION}")
    print(f"prompt_contract={PROMPT_CONTRACT_VERSION}")
    print(f"root={root}")
    print(f"scanned_md={len(paths)}")
    print(f"changed_md={changed}")
    print(f"already_new_md={already_new}")
    print(f"failed_md={len(failed)}")
    print(f"dry_run={bool(args.dry_run)}")
    print("persistent_sidecar_written=0")

    if failed:
        print("\n失败文件：")
        for path, message in failed[:100]:
            print(f"- {path}: {message}")
        if len(failed) > 100:
            print(f"... 其余 {len(failed) - 100} 个失败文件省略")
        return 2

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "只原地升级已经导出的 Strong-Teacher Markdown Prompt；"
            "不重新 prepare Candidate，不生成任何 JSONL/report sidecar。"
        )
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION} / {PROMPT_CONTRACT_VERSION}",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=(
            "Strong-Teacher 全量导出根目录。默认："
            "data/upstream/external_supervision/strong_teacher_v1_3_all"
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(DEFAULT_SPLITS),
        default=list(DEFAULT_SPLITS),
        help="默认同时修改 train / validation / benchmark 下的 md。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验每个 MD 是否能安全升级，不写盘。",
    )
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
