#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evidence Policy 离线排名评估命令行入口。

建议文件位置：
    scripts/evaluate_evidence_policy.py

========================================================================
一、文件职责
========================================================================

本脚本只负责“运行层”工作：

1. 读取冻结 V2.10 的 train / validation / benchmark Parquet；
2. 在 dry-run 模式下，只检查候选空间和监督覆盖；
3. 在正式评估模式下：
   - 加载训练后的 Evidence Policy checkpoint；
   - 从 Evidence Cache 重建冻结的 (q, K, A) 输入；
   - 执行 Cross-Encoder forward；
4. 调用：
       src/evaluation/policy.py
   中定义的核心实验指标；
5. 输出：
       report.json
       report.md
       per_state.csv

真正的实验定义，例如：
    - Oracle Candidate Ranking
    - Online Candidate Ranking
    - Positive Candidate Coverage
    - Hit@1 / Hit@5 / Hit@10
    - MRR
    - NDCG
    - STOP Recall
    - Premature STOP

全部属于：
    src/evaluation/policy.py

本脚本不重复实现这些指标。

========================================================================
二、这次修复的关键问题
========================================================================

旧版本存在一个明显问题：

    trainlib = load_trainlib()

在 main() 中被“无条件执行”。

这意味着即使用户运行：

    --dry-run

脚本仍会强制导入：
    scripts/train_evidence_policy.py

于是 dry-run 被训练代码、Transformers、模型环境甚至 Python import 路径绑死。

这是错误的依赖关系。

正确设计应该是：

    dry-run
        ↓
    只读取 Parquet
        ↓
    调用 src/evaluation/policy.py
        ↓
    输出 candidate coverage
        ↓
    完全不加载训练脚本 / 模型 / Evidence Cache

只有正式模型评估时才需要：

    train_evidence_policy.py
    Evidence Cache
    PyTorch
    Transformers
    checkpoint

因此，本版本把训练运行时加载严格放进：

    if not args.dry_run:

内部。

此外，正式评估时不再使用普通：

    import train_evidence_policy

而是使用 train_evidence_policy.py 的绝对文件路径进行动态加载。

这样即使 Python 的 sys.path 没有 scripts/，
也不会再因为模块搜索路径导致：
    ModuleNotFoundError: train_evidence_policy

========================================================================
三、dry-run 的含义
========================================================================

dry-run 不产生真正的 Policy 排名指标。

它用于验证：

1. V2.10 Parquet 能否正常读取；
2. ranking_loss_mask=true 的 state 是否有 active positive；
3. Oracle candidate universe 的规模；
4. Online candidate universe 的规模；
5. Online positive 是否可达；
6. Initial / Decision Boundary / Complete 的覆盖结构；
7. src/evaluation/policy.py 与实际数据 schema 是否兼容。

dry-run 中：

    Hit@K
    MRR
    NDCG
    STOP 排名

应该保持为 None / N/A。

这是有意设计，不是异常。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# ========================================================================
# 1. 项目路径初始化
# ========================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

# 确保可以稳定导入 src/evaluation/policy.py。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ========================================================================
# 2. 导入核心实验评估逻辑
# ========================================================================

try:
    from src.evaluation.policy import (
        ONLINE_MODE,
        ORACLE_MODE,
        PolicyRankingEvaluator,
        SUPPORTED_MODES,
        select_active_actions,
        state_result_row,
    )
except ImportError as exc:
    raise RuntimeError(
        "无法导入 src/evaluation/policy.py。\n"
        "请确认文件存在：\n"
        f"  {PROJECT_ROOT / 'src' / 'evaluation' / 'policy.py'}"
    ) from exc


# ========================================================================
# 3. V2.10 固定常量
# ========================================================================

DATASET_VERSION = "2.10.0"

DEFAULT_DATASET_DIR = Path(
    "data/upstream/unified_swe_dataset_v2_10"
)

DEFAULT_EVIDENCE_CACHE = Path(
    "data/.train_cache/policy_evidence_v2_10.sqlite3"
)

DEFAULT_OUTPUT_DIR = Path(
    "reports/evidence_policy_v2_10_validation"
)

SUPPORTED_SPLITS = (
    "train",
    "validation",
    "benchmark",
)


# ========================================================================
# 4. 数据集文件解析
# ========================================================================

def resolve_manifest_path(dataset_dir: Path) -> Path:
    """
    找到 V2.10 manifest。

    项目历史上出现过两种命名：
        manifest.json
        manifest_v2_10.json

    这里显式兼容二者。
    """

    candidates = (
        dataset_dir / "manifest.json",
        dataset_dir / "manifest_v2_10.json",
    )

    for path in candidates:
        if path.is_file():
            return path

    checked = "\n".join(
        f"  - {path}"
        for path in candidates
    )

    raise FileNotFoundError(
        "找不到 V2.10 manifest。\n"
        f"已检查：\n{checked}"
    )


def resolve_split_path(
    dataset_dir: Path,
    split: str,
) -> Path:
    """
    找到指定 split 对应的 Parquet。

    兼容：
        validation.parquet
        validation_v2_10.parquet
    """

    candidates = (
        dataset_dir / f"{split}.parquet",
        dataset_dir / f"{split}_v2_10.parquet",
    )

    for path in candidates:
        if path.is_file():
            return path

    checked = "\n".join(
        f"  - {path}"
        for path in candidates
    )

    raise FileNotFoundError(
        f"找不到 split={split!r} 的 Parquet。\n"
        f"已检查：\n{checked}"
    )


def load_and_validate_manifest(
    dataset_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """
    读取并校验冻结 V2.10 manifest。

    正式实验只允许：
        dataset version == 2.10.0
        audit_status == passed
    """

    manifest_path = resolve_manifest_path(
        dataset_dir
    )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    version = (
        manifest.get("dataset_version")
        or manifest.get("version")
    )

    if version != DATASET_VERSION:
        raise ValueError(
            "评估器只接受冻结 V2.10 release："
            f"expected={DATASET_VERSION!r}, "
            f"actual={version!r}"
        )

    audit_status = manifest.get(
        "audit_status"
    )

    if audit_status != "passed":
        raise ValueError(
            "数据集未通过正式 release audit："
            f"audit_status={audit_status!r}"
        )

    return manifest_path, manifest


# ========================================================================
# 5. 流式读取 Parquet
# ========================================================================

def iter_task_rows(
    parquet_path: Path,
) -> Iterable[dict[str, Any]]:
    """
    按 row group 流式读取任务，避免一次性展开整个 supervision。
    """

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "缺少 pyarrow，无法读取 V2.10 Parquet。"
        ) from exc

    parquet = pq.ParquetFile(
        parquet_path
    )

    for row_group_index in range(
        parquet.num_row_groups
    ):
        table = parquet.read_row_group(
            row_group_index,
            columns=[
                "task_id",
                "input",
                "supervision",
            ],
            use_threads=True,
        )

        yield from table.to_pylist()


# ========================================================================
# 6. 正式评估时加载训练运行时
# ========================================================================

def load_training_runtime() -> Any:
    """
    通过绝对文件路径加载 scripts/train_evidence_policy.py。

    这里故意不使用：
        import train_evidence_policy

    这样不会依赖 scripts/ 是否在 sys.path 中。

    注意：
    本函数只会在非 dry-run 模式下调用。
    """

    train_script_path = (
        PROJECT_ROOT
        / "scripts"
        / "train_evidence_policy.py"
    )

    if not train_script_path.is_file():
        raise FileNotFoundError(
            "正式 Policy 评估需要训练脚本中的冻结输入合同，"
            "但找不到：\n"
            f"  {train_script_path}"
        )

    module_name = (
        "_evidence_policy_training_runtime"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        train_script_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "无法为训练脚本创建 import spec："
            f"{train_script_path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(
            module_name,
            None,
        )
        raise

    required_names = (
        "EvidenceCache",
        "build_question",
        "truncate_question_view",
        "render_action_text",
        "resolve_precision",
        "model_scores",
    )

    missing = [
        name
        for name in required_names
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            "train_evidence_policy.py 缺少正式评估所需接口："
            f"{missing}\n"
            "这通常表示训练脚本与评估脚本版本不一致。"
        )

    return module


# ========================================================================
# 7. Evidence 输入重建辅助逻辑
# ========================================================================

def required_evidence_ids(
    state: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
) -> list[str]:
    """
    找出重建当前 state 所有 active action 所需的 Evidence IDs。
    """

    ids: set[str] = set()

    ids.update(
        str(evidence_id)
        for evidence_id
        in state.get("evidence_ids") or []
    )

    for action in actions:
        ids.update(
            str(evidence_id)
            for evidence_id
            in action.get("evidence_ids") or []
        )

        ids.update(
            str(evidence_id)
            for evidence_id
            in (
                action.get(
                    "rendered_state_body_evidence_ids"
                )
                or []
            )
        )

    return sorted(ids)


def render_active_action_texts(
    *,
    train_runtime: Any,
    evidence_cache: Any,
    tokenizer: Any,
    task_input: Mapping[str, Any],
    state: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    verify_remaining: int,
) -> tuple[list[str], int]:
    """
    使用训练脚本中的冻结函数重建 (q, K, A) 文本。

    同时对前若干 action 做 model_input_token_count 一致性检查。
    """

    evidence_ids = required_evidence_ids(
        state,
        actions,
    )

    evidence = evidence_cache.get_many(
        evidence_ids
    )

    question = train_runtime.build_question(
        dict(task_input)
    )

    question_view = (
        train_runtime.truncate_question_view(
            question,
            tokenizer,
        )
    )

    texts: list[str] = []

    for action_mapping in actions:
        action = dict(
            action_mapping
        )

        text = train_runtime.render_action_text(
            question_view=question_view,
            state=dict(state),
            action=action,
            evidence=evidence,
        )

        if verify_remaining > 0:
            actual_count = len(
                tokenizer.encode(
                    text,
                    add_special_tokens=True,
                )
            )

            expected_raw = action.get(
                "model_input_token_count"
            )

            if expected_raw is None:
                raise ValueError(
                    "active action 缺少 model_input_token_count："
                    f"state_id={state.get('state_id')}, "
                    f"action_id={action.get('action_id')}"
                )

            expected_count = int(
                expected_raw
            )

            if actual_count != expected_count:
                raise ValueError(
                    "V2.10 render/token contract 校验失败："
                    f"state_id={state.get('state_id')}, "
                    f"action_id={action.get('action_id')}, "
                    f"expected={expected_count}, "
                    f"actual={actual_count}"
                )

            verify_remaining -= 1

        texts.append(text)

    return texts, verify_remaining


# ========================================================================
# 8. 模型 checkpoint 加载
# ========================================================================

def load_policy_checkpoint(
    checkpoint: Path,
    *,
    device: Any,
) -> tuple[Any, Any]:
    """
    从本地 checkpoint 加载模型与 tokenizer。
    """

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "缺少 transformers，无法加载 Policy checkpoint。"
        ) from exc

    checkpoint = checkpoint.resolve()

    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"checkpoint 目录不存在：{checkpoint}"
        )

    weight_path = (
        checkpoint
        / "model.safetensors"
    )

    if not weight_path.is_file():
        raise FileNotFoundError(
            "checkpoint 缺少 model.safetensors："
            f"{weight_path}"
        )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            checkpoint,
            use_fast=True,
            local_files_only=True,
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            checkpoint,
            local_files_only=True,
            use_safetensors=True,
        )
    )

    if hasattr(
        model.config,
        "use_cache",
    ):
        model.config.use_cache = False

    model.to(device)
    model.eval()

    return model, tokenizer


# ========================================================================
# 9. 输出字段和格式
# ========================================================================

PER_STATE_FIELDS = [
    "mode",
    "task_id",
    "state_id",
    "state_type",
    "candidate_count",
    "positive_count",
    "reachable",
    "ranked",
    "best_positive_rank",
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "mrr",
    "ndcg_at_5",
    "ndcg_at_10",
    "top_action_id",
    "top_action_type",
    "top_action_scope",
    "top_action_label",
    "top_score",
    "stop_present",
    "stop_is_positive",
    "stop_rank",
    "stop_top1",
]


def format_metric(
    value: Any,
) -> str:
    """Markdown 中统一格式化实验指标。"""

    if value is None:
        return "N/A"

    return f"{float(value):.6f}"


def write_markdown_report(
    report: Mapping[str, Any],
    output_path: Path,
) -> None:
    """
    根据 report.json 同源数据生成 Markdown 摘要。
    """

    counts = report["counts"]

    lines: list[str] = [
        "# Evidence Policy V2.10 Offline Ranking Evaluation",
        "",
        f"- Split: `{report['split']}`",
        f"- Dataset version: `{report['dataset_version']}`",
        f"- Dry run: `{report['dry_run']}`",
        f"- Checkpoint: `{report.get('checkpoint')}`",
        f"- Evaluated rankable states: `{counts['evaluated_rankable_states']}`",
        f"- Model scored actions: `{counts['model_scored_actions']}`",
        f"- Verified render-contract actions: `{counts['verified_render_contract_actions']}`",
        "",
        "## 指标解释",
        "",
        "- `positive_candidate_coverage`：当前候选空间中至少存在一个监督 positive 的 state 比例。",
        "- `Hit@K / MRR / NDCG`：只在 positive reachable 的 state 上统计。",
        "- `STOP Recall@1`：当 STOP 为 positive 时，STOP 是否排第一。",
        "- `Premature STOP Rate`：当 STOP 为 negative 时，却错误排第一的比例。",
        "- Oracle 允许 offline_injected；Online 只允许在线 Evidence candidate + STOP。",
        "",
    ]

    if report["dry_run"]:
        lines.extend(
            [
                "> 当前是 dry-run，没有模型打分；排序指标应为 N/A。",
                "",
            ]
        )

    evaluation = report["evaluation"]
    results = evaluation["results"]

    for mode in evaluation["modes"]:
        mode_result = results[mode]

        overall = mode_result[
            "overall"
        ]

        ranking = overall[
            "ranking_metrics_conditioned_on_reachable"
        ]

        stop = overall["stop"]

        lines.extend(
            [
                f"## {mode}",
                "",
                f"- Known-positive states: `{overall['known_positive_state_count']}`",
                f"- Reachable-positive states: `{overall['reachable_positive_state_count']}`",
                f"- Unreachable-positive states: `{overall['unreachable_positive_state_count']}`",
                f"- Positive candidate coverage: `{format_metric(overall['positive_candidate_coverage'])}`",
                f"- Mean candidate count: `{format_metric(overall['mean_candidate_count'])}`",
                f"- Hit@1: `{format_metric(ranking['hit_at_1'])}`",
                f"- Hit@5: `{format_metric(ranking['hit_at_5'])}`",
                f"- Hit@10: `{format_metric(ranking['hit_at_10'])}`",
                f"- MRR: `{format_metric(ranking['mrr'])}`",
                f"- NDCG@5: `{format_metric(ranking['ndcg_at_5'])}`",
                f"- NDCG@10: `{format_metric(ranking['ndcg_at_10'])}`",
                f"- STOP Recall@1: `{format_metric(stop['stop_recall_at_1'])}`",
                f"- Premature STOP Rate: `{format_metric(stop['premature_stop_rate'])}`",
                "",
                "### By state type",
                "",
                "| State type | Coverage | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG@10 | STOP R@1 | Premature STOP |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for (
            state_type,
            bucket,
        ) in mode_result[
            "by_state_type"
        ].items():
            bucket_ranking = bucket[
                "ranking_metrics_conditioned_on_reachable"
            ]

            bucket_stop = bucket["stop"]

            lines.append(
                "| "
                + " | ".join(
                    [
                        state_type,
                        format_metric(
                            bucket[
                                "positive_candidate_coverage"
                            ]
                        ),
                        format_metric(
                            bucket_ranking[
                                "hit_at_1"
                            ]
                        ),
                        format_metric(
                            bucket_ranking[
                                "hit_at_5"
                            ]
                        ),
                        format_metric(
                            bucket_ranking[
                                "hit_at_10"
                            ]
                        ),
                        format_metric(
                            bucket_ranking[
                                "mrr"
                            ]
                        ),
                        format_metric(
                            bucket_ranking[
                                "ndcg_at_10"
                            ]
                        ),
                        format_metric(
                            bucket_stop[
                                "stop_recall_at_1"
                            ]
                        ),
                        format_metric(
                            bucket_stop[
                                "premature_stop_rate"
                            ]
                        ),
                    ]
                )
                + " |"
            )

        lines.append("")

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ========================================================================
# 10. CLI 参数
# ========================================================================

def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    注意：
    --dry-run 时：
        --checkpoint 可以不提供；
        --evidence-cache 不会打开；
        train_evidence_policy.py 不会导入。
    """

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate V2.10 Evidence Policy "
            "under Oracle and Online candidate universes."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="冻结 V2.10 release 目录。",
    )

    parser.add_argument(
        "--evidence-cache",
        type=Path,
        default=DEFAULT_EVIDENCE_CACHE,
        help=(
            "训练/正式评估使用的 Evidence Cache；"
            "dry-run 不读取。"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "训练后的 Policy checkpoint。"
            "非 dry-run 时必填。"
        ),
    )

    parser.add_argument(
        "--split",
        choices=SUPPORTED_SPLITS,
        default="validation",
        help=(
            "模型选择使用 validation；"
            "模型冻结后再使用 benchmark。"
        ),
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        choices=SUPPORTED_MODES,
        default=[
            ORACLE_MODE,
            ONLINE_MODE,
        ],
        help="默认同时运行 Oracle 和 Online。",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="评估报告输出目录。",
    )

    parser.add_argument(
        "--candidate-microbatch",
        type=int,
        default=2,
        help=(
            "正式模型评估时一次送入 GPU 的 action 数；"
            "只影响显存，不改变候选集合。"
        ),
    )

    parser.add_argument(
        "--precision",
        choices=[
            "auto",
            "fp32",
            "fp16",
            "bf16",
        ],
        default="auto",
        help="正式模型评估的推理精度。",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=None,
        help=(
            "仅用于 smoke/debug；"
            "正式 validation/benchmark 不设置。"
        ),
    )

    parser.add_argument(
        "--verify-render-count-actions",
        type=int,
        default=128,
        help=(
            "正式评估时重新验证前 N 个 action 的 "
            "model_input_token_count。"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "只检查数据和候选 coverage；"
            "完全不加载训练脚本、模型和 Evidence Cache。"
        ),
    )

    return parser.parse_args()


# ========================================================================
# 11. 主流程
# ========================================================================

def main() -> int:
    """
    主评估流程。

    依赖顺序：
        先验证 dataset
            ↓
        建立 PolicyRankingEvaluator
            ↓
        如果 dry-run：
            直接读取 Parquet
        否则：
            才加载训练 runtime / PyTorch / checkpoint / Evidence Cache
    """

    args = parse_args()

    if args.candidate_microbatch < 1:
        raise ValueError(
            "--candidate-microbatch 必须 >= 1"
        )

    if (
        args.max_states is not None
        and args.max_states < 1
    ):
        raise ValueError(
            "--max-states 必须 >= 1"
        )

    if args.verify_render_count_actions < 0:
        raise ValueError(
            "--verify-render-count-actions 不能为负数"
        )

    if (
        not args.dry_run
        and args.checkpoint is None
    ):
        raise ValueError(
            "正式模型评估必须提供 --checkpoint；"
            "如果当前没有训练模型，请使用 --dry-run。"
        )

    dataset_dir = (
        args.dataset_dir.resolve()
    )

    (
        manifest_path,
        manifest,
    ) = load_and_validate_manifest(
        dataset_dir
    )

    split_path = resolve_split_path(
        dataset_dir,
        args.split,
    )

    evaluator = PolicyRankingEvaluator(
        modes=args.modes
    )

    # 模型相关对象默认保持 None。
    # dry-run 不触碰这些对象。
    train_runtime = None
    evidence_cache = None
    model = None
    tokenizer = None
    device = None
    precision = None

    # 关键修复：
    # 只有正式评估才加载训练运行时。
    if not args.dry_run:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "缺少 PyTorch，无法执行正式 Policy 评估。"
            ) from exc

        train_runtime = (
            load_training_runtime()
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        precision = (
            train_runtime.resolve_precision(
                args.precision
            )
        )

        (
            model,
            tokenizer,
        ) = load_policy_checkpoint(
            args.checkpoint,
            device=device,
        )

        evidence_cache = (
            train_runtime.EvidenceCache(
                args.evidence_cache
            )
        )

        print(
            json.dumps(
                {
                    "event": "evaluation_runtime",
                    "device": str(device),
                    "precision": precision,
                    "split": args.split,
                    "checkpoint": str(
                        args.checkpoint.resolve()
                    ),
                    "candidate_microbatch": (
                        args.candidate_microbatch
                    ),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )

    output_dir = (
        args.output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_json_path = (
        output_dir / "report.json"
    )

    report_md_path = (
        output_dir / "report.md"
    )

    per_state_path = (
        output_dir / "per_state.csv"
    )

    started = time.perf_counter()

    evaluated_state_count = 0
    skipped_non_rankable_states = 0
    scored_action_count = 0

    verify_remaining = (
        args.verify_render_count_actions
    )

    try:
        with per_state_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=PER_STATE_FIELDS,
                extrasaction="ignore",
            )
            writer.writeheader()

            stop_requested = False

            for task_row in iter_task_rows(
                split_path
            ):
                task_id = str(
                    task_row.get("task_id")
                    or ""
                )

                task_input = (
                    task_row.get("input")
                    or {}
                )

                supervision = (
                    task_row.get("supervision")
                    or {}
                )

                states = (
                    supervision.get("policy_states")
                    or []
                )

                for state in states:
                    # ranking_loss_mask=false 的 state
                    # 不属于当前监督排名实验。
                    if not bool(
                        state.get("ranking_loss_mask")
                    ):
                        skipped_non_rankable_states += 1
                        continue

                    active_actions = (
                        select_active_actions(
                            state
                        )
                    )

                    if not active_actions:
                        raise ValueError(
                            "rankable state 没有 active actions："
                            f"task_id={task_id}, "
                            f"state_id={state.get('state_id')}"
                        )

                    if not any(
                        action.get("action_label")
                        == "positive"
                        for action
                        in active_actions
                    ):
                        raise ValueError(
                            "rankable state 没有 active positive："
                            f"task_id={task_id}, "
                            f"state_id={state.get('state_id')}"
                        )

                    if args.dry_run:
                        # dry-run 不打分，只统计 coverage。
                        scores = None
                    else:
                        assert train_runtime is not None
                        assert evidence_cache is not None
                        assert tokenizer is not None
                        assert model is not None
                        assert device is not None
                        assert precision is not None

                        (
                            texts,
                            verify_remaining,
                        ) = render_active_action_texts(
                            train_runtime=train_runtime,
                            evidence_cache=evidence_cache,
                            tokenizer=tokenizer,
                            task_input=task_input,
                            state=state,
                            actions=active_actions,
                            verify_remaining=verify_remaining,
                        )

                        import torch

                        with torch.inference_mode():
                            score_tensor = (
                                train_runtime.model_scores(
                                    model,
                                    tokenizer,
                                    texts,
                                    device=device,
                                    precision=precision,
                                    candidate_microbatch=(
                                        args.candidate_microbatch
                                    ),
                                )
                            )

                        scores = [
                            float(value)
                            for value
                            in (
                                score_tensor
                                .detach()
                                .float()
                                .cpu()
                                .tolist()
                            )
                        ]

                        scored_action_count += (
                            len(active_actions)
                        )

                    state_results = (
                        evaluator.add_state(
                            state=state,
                            scores=scores,
                        )
                    )

                    state_id = str(
                        state.get("state_id")
                        or ""
                    )

                    state_type = str(
                        state.get("state_type")
                        or "unknown"
                    )

                    for mode in evaluator.modes:
                        row = state_result_row(
                            task_id=task_id,
                            state_id=state_id,
                            state_type=state_type,
                            result=state_results[mode],
                        )

                        writer.writerow(row)

                    evaluated_state_count += 1

                    if (
                        args.max_states is not None
                        and evaluated_state_count
                        >= args.max_states
                    ):
                        stop_requested = True
                        break

                    if (
                        evaluated_state_count
                        % 250
                        == 0
                    ):
                        elapsed = max(
                            time.perf_counter()
                            - started,
                            1e-9,
                        )

                        print(
                            "evaluation_progress: "
                            f"states={evaluated_state_count:,}, "
                            f"scored_actions={scored_action_count:,}, "
                            "state_per_sec="
                            f"{evaluated_state_count / elapsed:.3f}",
                            file=sys.stderr,
                            flush=True,
                        )

                if stop_requested:
                    break

        elapsed_seconds = (
            time.perf_counter()
            - started
        )

        core_report = (
            evaluator.report()
        )

        report: dict[str, Any] = {
            "evaluation_name": (
                "evidence_policy_v2_10_offline_ranking"
            ),
            "dataset_version": DATASET_VERSION,
            "dataset_dir": str(dataset_dir),
            "manifest": str(manifest_path),
            "manifest_audit_status": (
                manifest.get("audit_status")
            ),
            "split": args.split,
            "split_path": str(split_path),
            "checkpoint": (
                str(args.checkpoint.resolve())
                if args.checkpoint is not None
                else None
            ),
            "dry_run": bool(args.dry_run),
            "runtime": {
                "device": (
                    str(device)
                    if device is not None
                    else None
                ),
                "precision": precision,
                "candidate_microbatch": (
                    args.candidate_microbatch
                ),
                "max_states": args.max_states,
            },
            "counts": {
                "evaluated_rankable_states": (
                    evaluated_state_count
                ),
                "skipped_non_rankable_states": (
                    skipped_non_rankable_states
                ),
                "model_scored_actions": (
                    scored_action_count
                ),
                "verified_render_contract_actions": (
                    (
                        args.verify_render_count_actions
                        - verify_remaining
                    )
                    if not args.dry_run
                    else 0
                ),
            },
            "timing_seconds": round(
                elapsed_seconds,
                3,
            ),
            "evaluation": core_report,
            "outputs": {
                "report_json": str(
                    report_json_path
                ),
                "report_markdown": str(
                    report_md_path
                ),
                "per_state_csv": str(
                    per_state_path
                ),
            },
        }

        report_json_path.write_text(
            json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        write_markdown_report(
            report,
            report_md_path,
        )

        summary = {
            "status": "passed",
            "split": args.split,
            "dry_run": args.dry_run,
            "evaluated_rankable_states": (
                evaluated_state_count
            ),
            "model_scored_actions": (
                scored_action_count
            ),
            "verified_render_contract_actions": (
                report["counts"][
                    "verified_render_contract_actions"
                ]
            ),
            "report_json": str(
                report_json_path
            ),
            "report_md": str(
                report_md_path
            ),
            "per_state_csv": str(
                per_state_path
            ),
        }

        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
        )

        return 0

    finally:
        # Windows 下显式关闭 SQLite connection。
        if evidence_cache is not None:
            evidence_cache.close()


if __name__ == "__main__":
    raise SystemExit(main())
