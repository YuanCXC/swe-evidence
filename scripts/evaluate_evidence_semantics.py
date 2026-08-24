#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evidence Package 语义充分性评估运行入口。

建议文件位置：
    scripts/evaluate_evidence_semantics.py

========================================================================
一、文件定位
========================================================================

本脚本是：

    src/evaluation/semantic.py

对应的“运行入口”。

核心语义评估协议，包括：
    - SemanticEvidencePackage
    - SemanticJudgeRequest
    - SemanticJudgeResult
    - sufficient / expand 冻结判定规则
    - 多次 Judge 聚合
    - semantic STOP 对齐
    - dataset-level 聚合

全部定义在：
    src/evaluation/semantic.py

本脚本只负责：

1. 从冻结 V2.10 Parquet 中读取任务和 Policy State；
2. 根据 state.evidence_ids 从 Evidence Cache 加载真实代码；
3. 构造 q + K：
       SemanticEvidencePackage
4. 调用 semantic.py 构造 Judge Prompt；
5. 提供本机可运行的：
       audit
       export
       mock
       deepseek
   四种模式；

   默认 Semantic Rubric：

       semantic-sufficiency-v2

   历史复现可显式使用：

       --semantic-rubric semantic-sufficiency-v1
6. 输出可审计的 JSON / JSONL / Markdown 结果。

========================================================================
二、为什么暂时不在本脚本里直接调用真实 LLM API
========================================================================

当前阶段先冻结“语义评估协议”和“评估输入”。

真实 Judge 可能来自：
    - OpenAI API；
    - 本地 vLLM；
    - Transformers；
    - 其他模型服务；
    - 人工导入结果。

如果现在直接把某个 API 写死在核心评估脚本里，会导致：
    评估协议
与：
    Judge 运行环境

耦合。

因此当前先支持：

1. audit
   完全不调用 Judge。
   检查：
       - q 是否可构造；
       - K 是否可从 Evidence Cache 完整加载；
       - evidence_count；
       - prompt 长度；
       - Initial / Boundary / Complete 分布；
       - 空 Evidence Package 比例。

2. export
   生成：
       semantic_requests.jsonl

   每行包含：
       task_id
       state_id
       state_type
       system_prompt
       user_prompt
       evidence_count
       prompt_char_count

   后续任何 Judge Adapter 都可以消费这个文件。

3. mock
   使用 src/evaluation/semantic.py 中的 StaticSemanticJudge，
   只验证：
       Prompt
       Parser
       Decision Rule
       Aggregation
       Report

   mock 结果绝对不能作为论文实验结果。

4. deepseek
   使用：
       src/evaluation/judge.py

   调用真实 DeepSeek Chat Completions API + JSON Output。

   默认不会直接跑完整 validation，
   而是稳定哈希抽取 calibration set：

       initial             10
       decision_boundary   20
       complete            20

   只有显式传：
       --allow-full-deepseek-run

   才允许对所选 state_type 全量调用 API。

========================================================================
三、Paired Calibration 为什么存在
========================================================================

独立抽样：

    一个 Initial task
    一个 Boundary task
    一个 Complete task

只能回答：
    不同类型状态的总体分数分布是否不同。

它不能回答：
    同一个任务随着 Evidence 增加，
    语义充分性是否真的提高。

因此 paired calibration 固定以 task 为抽样单位：

    Task A:
        Initial
        Decision Boundary
        Complete

    Task B:
        Initial
        Decision Boundary
        Complete

然后研究：

    Δ(Initial -> Boundary)
    Δ(Boundary -> Complete)
    Δ(Initial -> Complete)

只要保持：
    split
    deepseek-sample-seed
    deepseek-paired-tasks

不变，v1 / v2 会选择完全相同的 task/state，
因此可以做干净的 Rubric A/B 对照。

这一步的目的不是“让 LLM Judge 同意 deterministic label”。

真正目的是检验：

    V2.10 的 supervision/certificate progression

是否对应我们论文最终希望声称的：

    semantic repair-context sufficiency progression

如果不对应，我们需要知道缺口发生在哪个维度：
    localization
    mechanism
    expected behavior
    dependency/state context
    repair scope

而不是通过降低阈值把结果调成“看起来正确”。

========================================================================
四、当前语义评估对象
========================================================================

这里评价的是 V2.10 supervision state 中的：

    q + K

其中：
    q = problem_statement + hints
    K = state.evidence_ids 对应的 Evidence Units

因此它可以用于审计：

    initial
        K 通常为空

    decision_boundary
        K 为“尚不足，但距离充分很近”的上下文

    complete
        K 为监督定义下已经充分的上下文

这可以回答一个非常重要的问题：

    deterministic supervision 认为：
        complete / non-complete

和语义 Judge 认为：
        sufficient / expand

是否一致。

注意：
    这里不是实际 Agent rollout。

因此本脚本不会把：
    state_type == complete

伪装成：
    agent_declared_stop = True

SemanticEvidencePackage.agent_declared_stop 默认保持 None。

真实 Agent STOP 语义评估，应由后续 rollout evaluator 提供真实 Agent trace。

========================================================================
四、Evidence Cache
========================================================================

本脚本直接读取：

    data/.train_cache/policy_evidence_v2_10.sqlite3

而不是重新扫描 7.3GB repository corpus。

训练 cache 中已经保存：
    evidence_id
    path
    unit_type
    symbol
    start/end line
    content

这足以重建 SemanticEvidencePackage。

本脚本只读 SQLite，不修改任何数据。

========================================================================
五、严格禁止静默截断 Evidence
========================================================================

语义评估的研究对象就是：
    “这个 K 是否充分”。

如果因为 Judge context 太长而偷偷把 K 截断，
评价对象就变了。

因此本脚本：
    - 计算 prompt_char_count；
    - 统计 oversized prompt；
    - 不对 Evidence content 做静默截断。

后续真实 Judge Adapter 如果存在 context limit，
必须显式报告：
    skipped / oversized

或者使用一个正式冻结的压缩协议，
不能偷偷 trim。

========================================================================
六、推荐运行顺序
========================================================================

第一步，本机做 audit：

    python scripts/evaluate_evidence_semantics.py `
      --dataset-dir data/unified_swe_dataset_v2_10 `
      --evidence-cache data/.train_cache/policy_evidence_v2_10.sqlite3 `
      --split validation `
      --mode audit `
      --output-dir reports/evidence_semantic_v2_10_audit

第二步，导出 Judge 请求：

    python scripts/evaluate_evidence_semantics.py `
      --dataset-dir data/unified_swe_dataset_v2_10 `
      --evidence-cache data/.train_cache/policy_evidence_v2_10.sqlite3 `
      --split validation `
      --mode export `
      --output-dir reports/evidence_semantic_v2_10_export

第三步，开发期可做 mock smoke：

    python scripts/evaluate_evidence_semantics.py `
      --dataset-dir data/unified_swe_dataset_v2_10 `
      --evidence-cache data/.train_cache/policy_evidence_v2_10.sqlite3 `
      --split validation `
      --mode mock `
      --max-packages 10 `
      --output-dir reports/evidence_semantic_v2_10_mock

mock 只验证工程链路，不能用于正式结论。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


# ===========================================================================
# 1. 项目路径初始化
# ===========================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ===========================================================================
# 2. 导入语义评估核心协议
# ===========================================================================

try:
    from src.evaluation.semantic import (
        DECISION_EXPAND,
        DECISION_SUFFICIENT,
        DEFAULT_SEMANTIC_RUBRIC_VERSION,
        SEMANTIC_RUBRIC_V1,
        SEMANTIC_RUBRIC_V2,
        SUPPORTED_RUBRIC_VERSIONS,
        SemanticEvaluationAccumulator,
        SemanticEvaluationConfig,
        SemanticEvidenceItem,
        SemanticEvidencePackage,
        SemanticJudgeRequest,
        StaticSemanticJudge,
        aggregate_semantic_runs,
        build_semantic_judge_prompt,
        evaluate_semantic_once,
        semantic_judge_system_prompt,
        semantic_protocol_metadata,
    )
except ImportError as exc:
    raise RuntimeError(
        "无法导入 src/evaluation/semantic.py。\n"
        "请确认文件存在：\n"
        f"  {PROJECT_ROOT / 'src' / 'evaluation' / 'semantic.py'}"
    ) from exc

try:
    from src.evaluation.judge import (
        DEFAULT_DEEPSEEK_MODEL,
        DeepSeekJudgeConfig,
        DeepSeekSemanticJudge,
        deepseek_judge_protocol_metadata,
        validate_deepseek_environment,
    )
except ImportError as exc:
    raise RuntimeError(
        "无法导入 src/evaluation/judge.py。\n"
        "请确认文件存在：\n"
        f"  {PROJECT_ROOT / 'src' / 'evaluation' / 'judge.py'}"
    ) from exc


# ===========================================================================
# 3. V2.10 固定常量
# ===========================================================================

DATASET_VERSION = "2.10.0"

DEFAULT_DATASET_DIR = Path(
    "data/unified_swe_dataset_v2_10"
)

DEFAULT_EVIDENCE_CACHE = Path(
    "data/.train_cache/policy_evidence_v2_10.sqlite3"
)

DEFAULT_OUTPUT_DIR = Path(
    "reports/evidence_semantic_v2_10"
)

SUPPORTED_SPLITS = (
    "train",
    "validation",
    "benchmark",
)

SUPPORTED_MODES = (
    "audit",
    "export",
    "mock",
    "deepseek",
)

# DeepSeek calibration set 默认不是“顺序取前 N 个”，
# 而是使用 task_id + state_id + seed 做稳定 SHA-256 排序后抽样。
# 这样同一份数据、同一 seed 下样本集合完全可复现，
# 同时避免 Parquet 行顺序造成明显选择偏差。
DEFAULT_DEEPSEEK_SAMPLE_LIMITS = {
    "initial": 10,
    "decision_boundary": 20,
    "complete": 20,
}

DEFAULT_DEEPSEEK_SAMPLE_SEED = 20260805

# Paired calibration 是“同一 task 内三联状态”校准：
#
#     Initial -> Decision Boundary -> Complete
#
# 它不是 frozen benchmark 默认模式。
# 默认 0 表示关闭，只有显式传 --deepseek-paired-tasks 才启用。
DEFAULT_DEEPSEEK_PAIRED_TASKS = 0

PAIRED_STATE_TYPES = (
    "initial",
    "decision_boundary",
    "complete",
)

PAIRED_SCORE_FIELDS = (
    "fault_localization",
    "fault_mechanism",
    "expected_behavior",
    "dependency_state_context",
    "impact_repair_scope",
    "overall_sufficiency",
)

DEFAULT_STATE_TYPES = (
    "initial",
    "decision_boundary",
    "complete",
)


# ===========================================================================
# 4. 数据集文件解析与 Manifest 校验
# ===========================================================================

def resolve_manifest_path(
    dataset_dir: Path,
) -> Path:
    """
    找到 V2.10 manifest。

    兼容：
        manifest.json
        manifest_v2_10.json
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
    找到指定 split Parquet。

    兼容正式 release 文件名和早期开发文件名。
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
    只允许 audit_status=passed 的冻结 V2.10 release。
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
            "语义评估器只接受冻结 V2.10 release："
            f"expected={DATASET_VERSION!r}, "
            f"actual={version!r}"
        )

    if manifest.get("audit_status") != "passed":
        raise ValueError(
            "拒绝对未通过 release audit 的数据集运行正式语义评估。"
        )

    return manifest_path, manifest


# ===========================================================================
# 5. 流式读取 Parquet
# ===========================================================================

def iter_task_rows(
    parquet_path: Path,
) -> Iterable[dict[str, Any]]:
    """
    按 row group 流式读取任务。

    这里只读取：
        task_id
        input
        supervision

    不一次性把整个 split 展开进 RAM。
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


# ===========================================================================
# 6. 构造 Semantic Question
# ===========================================================================

def build_semantic_question(
    task_input: Mapping[str, Any],
) -> str:
    """
    从 V2.10 task input 构造语义评估问题 q。

    当前数据构建合同使用：
        problem_statement
        +
        hints

    语义评估不需要复用训练模型的 tokenizer truncation，
    因为这里评价的是“问题语义 + Evidence Package”，
    不是 Cross-Encoder 的 4096-token 输入合同。

    但是 q 的文字来源仍保持和数据集一致。

    这里不猜测其他字段。
    如果 problem_statement 缺失，直接失败。
    """

    problem_statement = task_input.get(
        "problem_statement"
    )

    if not isinstance(
        problem_statement,
        str,
    ) or not problem_statement.strip():
        raise ValueError(
            "task input 缺少非空 problem_statement"
        )

    parts = [
        problem_statement.strip()
    ]

    hints = task_input.get("hints")

    if hints is None:
        pass

    elif isinstance(hints, str):
        if hints.strip():
            parts.append(
                hints.strip()
            )

    elif isinstance(
        hints,
        (list, tuple),
    ):
        normalized_hints = []

        for item in hints:
            if not isinstance(item, str):
                raise ValueError(
                    "input.hints 为数组时，所有元素必须是字符串"
                )

            item = item.strip()

            if item:
                normalized_hints.append(
                    item
                )

        if normalized_hints:
            parts.extend(
                normalized_hints
            )

    else:
        raise ValueError(
            "input.hints 必须是 string / list[string] / null，"
            f"实际={type(hints)!r}"
        )

    return "\n".join(parts)


# ===========================================================================
# 7. Evidence Cache 只读访问
# ===========================================================================

class SemanticEvidenceCache:
    """
    面向语义评估的最小 SQLite Evidence Cache Reader。

    不依赖训练脚本。

    这样：
        scripts/evaluate_evidence_semantics.py

    不会因为训练代码重构而失效。

    当前只读取 SemanticEvidenceItem 所需字段。
    """

    def __init__(
        self,
        path: Path,
    ) -> None:
        self.path = path.resolve()

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Evidence Cache 不存在：{self.path}"
            )

        # 使用 SQLite URI 的 mode=ro 强制只读。
        uri = (
            self.path.as_uri()
            + "?mode=ro"
        )

        self.connection = sqlite3.connect(
            uri,
            uri=True,
        )

        self.connection.row_factory = (
            sqlite3.Row
        )

        self.table_name = (
            self._resolve_evidence_table()
        )

        self.columns = (
            self._load_columns()
        )

        self._validate_schema()

    def close(self) -> None:
        self.connection.close()

    def _resolve_evidence_table(self) -> str:
        """
        自动找到 Evidence 表。

        当前训练 cache 通常使用 evidence，
        但这里通过 schema 审计，不硬猜。
        """

        rows = self.connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()

        table_names = [
            str(row["name"])
            for row in rows
        ]

        preferred = (
            "evidence",
            "evidence_units",
            "policy_evidence",
        )

        for name in preferred:
            if name in table_names:
                return name

        raise RuntimeError(
            "Evidence Cache 中找不到可识别的 Evidence 表。\n"
            f"tables={table_names}"
        )

    def _load_columns(self) -> set[str]:
        rows = self.connection.execute(
            f"PRAGMA table_info({self.table_name})"
        ).fetchall()

        return {
            str(row["name"])
            for row in rows
        }

    def _validate_schema(self) -> None:
        """
        校验最小必需字段。

        start/end 行号允许：
            start / end
        或：
            start_line / end_line
        """

        required = {
            "evidence_id",
            "path",
            "unit_type",
            "content",
        }

        missing = sorted(
            required
            - self.columns
        )

        if missing:
            raise RuntimeError(
                "Evidence Cache schema 缺少语义评估必需字段："
                f"{missing}\n"
                f"table={self.table_name}\n"
                f"columns={sorted(self.columns)}"
            )

    def get_many(
        self,
        evidence_ids: Sequence[str],
    ) -> dict[str, SemanticEvidenceItem]:
        """
        批量读取 Evidence。

        返回：
            evidence_id -> SemanticEvidenceItem

        如果任何 ID 缺失，直接失败。

        语义评估不能静默跳过 Evidence，
        否则被评价的 K 会发生变化。
        """

        unique_ids = list(
            dict.fromkeys(
                str(item)
                for item in evidence_ids
            )
        )

        if not unique_ids:
            return {}

        result: dict[
            str,
            SemanticEvidenceItem
        ] = {}

        # Windows SQLite 默认变量数量有限，
        # 因此分批查询。
        chunk_size = 500

        for offset in range(
            0,
            len(unique_ids),
            chunk_size,
        ):
            chunk = unique_ids[
                offset:
                offset + chunk_size
            ]

            placeholders = ",".join(
                "?"
                for _ in chunk
            )

            rows = self.connection.execute(
                f"""
                SELECT *
                FROM {self.table_name}
                WHERE evidence_id IN ({placeholders})
                """,
                chunk,
            ).fetchall()

            for row in rows:
                item = self._row_to_item(
                    row
                )
                result[
                    item.evidence_id
                ] = item

        missing_ids = [
            evidence_id
            for evidence_id in unique_ids
            if evidence_id not in result
        ]

        if missing_ids:
            preview = missing_ids[:20]

            raise KeyError(
                "Evidence Cache 缺少 state 所引用的 Evidence。\n"
                f"missing_count={len(missing_ids)}\n"
                f"preview={preview}"
            )

        return result

    def _row_to_item(
        self,
        row: sqlite3.Row,
    ) -> SemanticEvidenceItem:
        """
        SQLite row -> SemanticEvidenceItem。
        """

        symbol = (
            row["symbol"]
            if "symbol" in self.columns
            else None
        )

        if "start" in self.columns:
            start_line = row["start"]
        elif "start_line" in self.columns:
            start_line = row["start_line"]
        else:
            start_line = None

        if "end" in self.columns:
            end_line = row["end"]
        elif "end_line" in self.columns:
            end_line = row["end_line"]
        else:
            end_line = None

        return SemanticEvidenceItem(
            evidence_id=str(
                row["evidence_id"]
            ),
            path=str(
                row["path"]
            ),
            unit_type=str(
                row["unit_type"]
            ),
            symbol=(
                str(symbol)
                if symbol is not None
                else None
            ),
            start_line=(
                int(start_line)
                if start_line is not None
                else None
            ),
            end_line=(
                int(end_line)
                if end_line is not None
                else None
            ),
            content=str(
                row["content"]
            ),
        )


# ===========================================================================
# 8. V2.10 State -> SemanticEvidencePackage
# ===========================================================================

def build_semantic_package(
    *,
    task_id: str,
    task_input: Mapping[str, Any],
    state: Mapping[str, Any],
    evidence_cache: SemanticEvidenceCache,
) -> SemanticEvidencePackage:
    """
    把一个 V2.10 Policy State 转换为语义评价对象。

    K 的唯一来源：
        state.evidence_ids

    不把：
        candidate_actions
        offline_injected candidate
        gold next action

    混进当前 Evidence Package。

    原因：
    Semantic Evaluation 评价的是：
        当前已经收集到的 K

    而不是：
        当前候选池里有哪些未来可能加入的 Evidence。
    """

    state_id = str(
        state.get("state_id")
        or ""
    )

    if not state_id:
        raise ValueError(
            f"task={task_id} 的 policy state 缺少 state_id"
        )

    state_type = str(
        state.get("state_type")
        or "unknown"
    )

    question = build_semantic_question(
        task_input
    )

    evidence_ids = [
        str(item)
        for item in (
            state.get("evidence_ids")
            or []
        )
    ]

    evidence_by_id = (
        evidence_cache.get_many(
            evidence_ids
        )
    )

    # 必须严格保留 state.evidence_ids 的原始顺序。
    evidence_items = tuple(
        evidence_by_id[
            evidence_id
        ]
        for evidence_id in evidence_ids
    )

    metadata = {
        "source": (
            "v2_10_supervision_policy_state"
        ),
        "supervision_expected_stop": (
            state_type == "complete"
        ),
        "ranking_loss_mask": bool(
            state.get(
                "ranking_loss_mask"
            )
        ),
    }

    return SemanticEvidencePackage(
        task_id=task_id,
        state_id=state_id,
        state_type=state_type,
        question=question,
        evidence=evidence_items,

        # 这里非常重要：
        #
        # state_type=complete 是 supervision 定义，
        # 不是实际 Agent 的 STOP 行为。
        #
        # 因此不能伪造 agent_declared_stop。
        agent_declared_stop=None,

        step_index=None,
        metadata=metadata,
    )


# ===========================================================================
# 9. Package 审计统计
# ===========================================================================

class PackageAuditAccumulator:
    """
    对待评估的 q + K 做数据质量审计。

    这些指标不需要任何 Judge。
    """

    def __init__(self) -> None:
        self.package_count = 0
        self.empty_package_count = 0

        self.state_type_counts = Counter()

        self.evidence_counts: list[int] = []
        self.question_char_counts: list[int] = []
        self.prompt_char_counts: list[int] = []

        self.oversized_prompt_count = 0

        self.by_state_type: defaultdict[
            str,
            dict[str, Any]
        ] = defaultdict(
            lambda: {
                "count": 0,
                "empty_count": 0,
                "evidence_counts": [],
                "prompt_chars": [],
                "oversized_count": 0,
            }
        )

    def add(
        self,
        package: SemanticEvidencePackage,
        *,
        prompt_char_count: int,
        oversized: bool,
    ) -> None:
        self.package_count += 1

        self.state_type_counts[
            package.state_type
        ] += 1

        evidence_count = (
            package.evidence_count
        )

        self.evidence_counts.append(
            evidence_count
        )

        self.question_char_counts.append(
            len(package.question)
        )

        self.prompt_char_counts.append(
            prompt_char_count
        )

        if evidence_count == 0:
            self.empty_package_count += 1

        if oversized:
            self.oversized_prompt_count += 1

        bucket = self.by_state_type[
            package.state_type
        ]

        bucket["count"] += 1

        bucket[
            "evidence_counts"
        ].append(
            evidence_count
        )

        bucket[
            "prompt_chars"
        ].append(
            prompt_char_count
        )

        if evidence_count == 0:
            bucket["empty_count"] += 1

        if oversized:
            bucket[
                "oversized_count"
            ] += 1

    @staticmethod
    def _safe_mean(
        values: Sequence[float | int],
    ) -> float | None:
        if not values:
            return None

        return float(mean(values))

    @staticmethod
    def _safe_max(
        values: Sequence[float | int],
    ) -> float | int | None:
        if not values:
            return None

        return max(values)

    def report(self) -> dict[str, Any]:
        return {
            "package_count": (
                self.package_count
            ),
            "empty_package_count": (
                self.empty_package_count
            ),
            "empty_package_rate": (
                self.empty_package_count
                / self.package_count
                if self.package_count
                else None
            ),
            "state_type_counts": dict(
                sorted(
                    self.state_type_counts.items()
                )
            ),
            "mean_evidence_count": (
                self._safe_mean(
                    self.evidence_counts
                )
            ),
            "max_evidence_count": (
                self._safe_max(
                    self.evidence_counts
                )
            ),
            "mean_question_chars": (
                self._safe_mean(
                    self.question_char_counts
                )
            ),
            "mean_prompt_chars": (
                self._safe_mean(
                    self.prompt_char_counts
                )
            ),
            "max_prompt_chars": (
                self._safe_max(
                    self.prompt_char_counts
                )
            ),
            "oversized_prompt_count": (
                self.oversized_prompt_count
            ),
            "oversized_prompt_rate": (
                self.oversized_prompt_count
                / self.package_count
                if self.package_count
                else None
            ),
            "by_state_type": {
                state_type: {
                    "count": bucket["count"],
                    "empty_count": (
                        bucket["empty_count"]
                    ),
                    "empty_rate": (
                        bucket["empty_count"]
                        / bucket["count"]
                        if bucket["count"]
                        else None
                    ),
                    "mean_evidence_count": (
                        self._safe_mean(
                            bucket[
                                "evidence_counts"
                            ]
                        )
                    ),
                    "max_evidence_count": (
                        self._safe_max(
                            bucket[
                                "evidence_counts"
                            ]
                        )
                    ),
                    "mean_prompt_chars": (
                        self._safe_mean(
                            bucket[
                                "prompt_chars"
                            ]
                        )
                    ),
                    "max_prompt_chars": (
                        self._safe_max(
                            bucket[
                                "prompt_chars"
                            ]
                        )
                    ),
                    "oversized_count": (
                        bucket[
                            "oversized_count"
                        ]
                    ),
                }
                for (
                    state_type,
                    bucket,
                ) in sorted(
                    self.by_state_type.items()
                )
            },
        }


# ===========================================================================
# 10. Mock Judge 输出
# ===========================================================================

def default_mock_judge_output(
    package: SemanticEvidencePackage,
) -> dict[str, Any]:
    """
    开发期 Mock Judge。

    非常重要：
        这不是语义模型。
        这不是实验结果。
        不能用于论文。

    这里只根据 Evidence 是否为空，
    返回两套固定结构化结果，
    目的是验证 semantic.py 的：
        parser
        decision
        aggregation
        report

    是否能完整跑通。
    """

    if package.evidence_count == 0:
        return {
            "fault_localization": 0.10,
            "fault_mechanism": 0.05,
            "expected_behavior": 0.30,
            "dependency_state_context": 0.05,
            "impact_repair_scope": 0.05,
            "overall_sufficiency": 0.10,
            "redundancy": 0.00,
            "contradiction": False,
            "raw_decision": DECISION_EXPAND,
            "rationale": (
                "MOCK ONLY: 空 Evidence Package，"
                "固定返回 expand。"
            ),
            "missing_information": [
                "MOCK_ONLY"
            ],
            "supporting_evidence_ids": [],
            "irrelevant_evidence_ids": [],
        }

    # 非空 K 也只是固定模板。
    supporting_id = (
        package.evidence[0].evidence_id
    )

    return {
        "fault_localization": 0.80,
        "fault_mechanism": 0.80,
        "expected_behavior": 0.80,
        "dependency_state_context": 0.70,
        "impact_repair_scope": 0.70,
        "overall_sufficiency": 0.80,
        "redundancy": 0.10,
        "contradiction": False,
        "raw_decision": DECISION_SUFFICIENT,
        "rationale": (
            "MOCK ONLY: 非空 Evidence Package，"
            "固定返回 sufficient。"
        ),
        "missing_information": [],
        "supporting_evidence_ids": [
            supporting_id
        ],
        "irrelevant_evidence_ids": [],
    }



# ===========================================================================
# 11. DeepSeek Calibration 抽样与正式语义指标辅助
# ===========================================================================

def _stable_deepseek_sample_key(
    *,
    task_id: str,
    state_id: str,
    seed: int,
) -> str:
    """
    为一个 state 生成稳定的 SHA-256 抽样键。

    这里不能使用 Python 内置 hash()：
        hash() 默认受 PYTHONHASHSEED 影响，
        不同进程之间结果可能变化。

    因此使用：
        SHA256(seed + task_id + state_id)

    只要：
        数据集
        seed
    不变，DeepSeek calibration set 就保持一致。
    """

    payload = (
        f"{seed}\0{task_id}\0{state_id}"
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def select_deepseek_calibration_states(
    *,
    split_path: Path,
    selected_state_types: set[str],
    seed: int,
    initial_limit: int,
    boundary_limit: int,
    complete_limit: int,
) -> set[tuple[str, str]]:
    """
    从一个 split 中稳定抽取 DeepSeek calibration states。

    默认目标：
        initial             10
        decision_boundary   20
        complete            20

    返回：
        {(task_id, state_id), ...}

    注意：
    这里只收集 task/state 身份信息，不加载 Evidence Cache，
    所以内存开销很小。
    """

    limits = {
        "initial": initial_limit,
        "decision_boundary": boundary_limit,
        "complete": complete_limit,
    }

    unsupported = sorted(
        selected_state_types
        - set(limits)
    )

    if unsupported:
        raise ValueError(
            "DeepSeek calibration 模式当前只定义了 "
            "initial / decision_boundary / complete 的分层抽样。"
            f"未定义 state_type={unsupported}。"
            "如需评估其他 state_type，请使用 --allow-full-deepseek-run。"
        )

    for state_type, limit in limits.items():
        if limit < 0:
            raise ValueError(
                f"{state_type} 的 DeepSeek sample limit 不能为负数"
            )

    candidates: defaultdict[
        str,
        list[tuple[str, str, str]],
    ] = defaultdict(list)

    for task_row in iter_task_rows(
        split_path
    ):
        task_id = str(
            task_row.get("task_id")
            or ""
        )

        if not task_id:
            raise ValueError(
                "Parquet row 缺少 task_id"
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
            state_type = str(
                state.get("state_type")
                or "unknown"
            )

            if (
                state_type
                not in selected_state_types
            ):
                continue

            state_id = str(
                state.get("state_id")
                or ""
            )

            if not state_id:
                raise ValueError(
                    f"task={task_id} 存在缺少 state_id 的 state"
                )

            stable_key = (
                _stable_deepseek_sample_key(
                    task_id=task_id,
                    state_id=state_id,
                    seed=seed,
                )
            )

            candidates[state_type].append(
                (
                    stable_key,
                    task_id,
                    state_id,
                )
            )

    selected: set[
        tuple[str, str]
    ] = set()

    for state_type in sorted(
        selected_state_types
    ):
        rows = sorted(
            candidates[state_type],
            key=lambda item: item[0],
        )

        requested_limit = limits[
            state_type
        ]

        if requested_limit == 0:
            continue

        if len(rows) < requested_limit:
            raise ValueError(
                "DeepSeek calibration sample 数量不足："
                f"state_type={state_type}, "
                f"available={len(rows)}, "
                f"requested={requested_limit}"
            )

        for (
            _stable_key,
            task_id,
            state_id,
        ) in rows[
            :requested_limit
        ]:
            selected.add(
                (
                    task_id,
                    state_id,
                )
            )

    return selected



def _stable_deepseek_task_key(
    *,
    task_id: str,
    seed: int,
) -> str:
    """
    对 task 而不是 state 做稳定哈希。

    Paired calibration 的抽样单位必须是 task。

    原因：
        如果分别对 initial / boundary / complete 抽样，
        三个 state 很可能来自不同 task，
        这时无法研究“同一任务证据增加后的语义变化”。

    使用 SHA-256 保证：
        同一 split + 同一 seed
        => 同一批 paired tasks。
    """

    payload = (
        f"{seed}\0{task_id}"
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def _choose_paired_state(
    states: Sequence[Mapping[str, Any]],
    *,
    state_type: str,
) -> Mapping[str, Any]:
    """
    当一个 task 某种 state_type 恰好有多个 state 时，
    选择一个确定性的代表状态。

    选择规则：

    initial:
        Evidence 数最少优先。
        理论上应为 0。

    decision_boundary:
        Evidence 数最多优先。
        这样选择离 complete 最近的 boundary。

    complete:
        Evidence 数最多优先。
        若存在多个 complete，选择信息更完整者。

    最后使用 state_id 字符串作为稳定 tie-breaker。

    当前 V2.10 通常每种类型只有一个正式状态，
    但这里不把“恰好唯一”写成隐式假设。
    """

    if not states:
        raise ValueError(
            f"没有可选择的 state_type={state_type}"
        )

    def evidence_count(
        state: Mapping[str, Any],
    ) -> int:
        return len(
            state.get("evidence_ids")
            or []
        )

    if state_type == "initial":
        return min(
            states,
            key=lambda state: (
                evidence_count(state),
                str(
                    state.get("state_id")
                    or ""
                ),
            ),
        )

    return max(
        states,
        key=lambda state: (
            evidence_count(state),
            str(
                state.get("state_id")
                or ""
            ),
        ),
    )


def select_deepseek_paired_calibration_states(
    *,
    split_path: Path,
    seed: int,
    task_limit: int,
) -> tuple[
    set[tuple[str, str]],
    dict[str, Any],
]:
    """
    稳定抽取同一 task 的：

        Initial
        Decision Boundary
        Complete

    三联状态。

    返回：
        selected_state_keys
        selection_metadata

    一个 task 只有同时拥有三种 state_type 才进入候选池。

    这意味着 paired calibration 主要落在：
        decision-boundary supervision tasks

    而不是整个 validation split。

    这正是我们需要的：
        研究同一条监督轨迹从“空证据”
        到“近充分”
        再到“监督 complete”的语义变化。
    """

    if task_limit < 1:
        raise ValueError(
            "--deepseek-paired-tasks 必须 >= 1"
        )

    candidates: list[
        tuple[
            str,
            str,
            dict[str, Mapping[str, Any]],
            dict[str, int],
        ]
    ] = []

    for task_row in iter_task_rows(
        split_path
    ):
        task_id = str(
            task_row.get("task_id")
            or ""
        )

        if not task_id:
            raise ValueError(
                "Parquet row 缺少 task_id"
            )

        supervision = (
            task_row.get("supervision")
            or {}
        )

        states = (
            supervision.get("policy_states")
            or []
        )

        by_type: defaultdict[
            str,
            list[Mapping[str, Any]],
        ] = defaultdict(list)

        for state in states:
            state_type = str(
                state.get("state_type")
                or "unknown"
            )

            if state_type in PAIRED_STATE_TYPES:
                by_type[state_type].append(
                    state
                )

        if not all(
            by_type.get(state_type)
            for state_type
            in PAIRED_STATE_TYPES
        ):
            continue

        chosen: dict[
            str,
            Mapping[str, Any],
        ] = {}

        multiplicity: dict[
            str,
            int,
        ] = {}

        for state_type in PAIRED_STATE_TYPES:
            multiplicity[state_type] = len(
                by_type[state_type]
            )

            chosen[state_type] = (
                _choose_paired_state(
                    by_type[state_type],
                    state_type=state_type,
                )
            )

        stable_key = (
            _stable_deepseek_task_key(
                task_id=task_id,
                seed=seed,
            )
        )

        candidates.append(
            (
                stable_key,
                task_id,
                chosen,
                multiplicity,
            )
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    if len(candidates) < task_limit:
        raise ValueError(
            "可用于 paired calibration 的 task 数量不足："
            f"available={len(candidates)}, "
            f"requested={task_limit}"
        )

    selected_candidates = (
        candidates[:task_limit]
    )

    selected_state_keys: set[
        tuple[str, str]
    ] = set()

    selected_tasks: list[
        dict[str, Any]
    ] = []

    for (
        stable_key,
        task_id,
        chosen,
        multiplicity,
    ) in selected_candidates:
        state_ids: dict[
            str,
            str,
        ] = {}

        evidence_counts: dict[
            str,
            int,
        ] = {}

        for state_type in PAIRED_STATE_TYPES:
            state = chosen[
                state_type
            ]

            state_id = str(
                state.get("state_id")
                or ""
            )

            if not state_id:
                raise ValueError(
                    "paired calibration 选中的 state "
                    f"缺少 state_id: task={task_id}, "
                    f"state_type={state_type}"
                )

            selected_state_keys.add(
                (
                    task_id,
                    state_id,
                )
            )

            state_ids[state_type] = (
                state_id
            )

            evidence_counts[state_type] = len(
                state.get("evidence_ids")
                or []
            )

        selected_tasks.append(
            {
                "task_id": task_id,
                "stable_hash": stable_key,
                "state_ids": state_ids,
                "evidence_counts": (
                    evidence_counts
                ),
                "state_multiplicity": (
                    multiplicity
                ),
            }
        )

    metadata = {
        "enabled": True,
        "task_limit": task_limit,
        "available_paired_task_count": (
            len(candidates)
        ),
        "selected_task_count": (
            len(selected_tasks)
        ),
        "selected_state_count": (
            len(selected_state_keys)
        ),
        "seed": seed,
        "required_state_types": list(
            PAIRED_STATE_TYPES
        ),
        "selected_tasks": (
            selected_tasks
        ),
    }

    return (
        selected_state_keys,
        metadata,
    )


def derive_paired_semantic_metrics(
    paired_results: Mapping[
        str,
        Mapping[str, Any],
    ],
) -> dict[str, Any]:
    """
    计算同任务三联状态的探索性语义变化指标。

    输入：
        paired_results[task_id][state_type]
            = SemanticAggregateResult

    这里只做 calibration diagnostics，
    暂时不修改 semantic-sufficiency-v1 的正式决策规则。

    核心关注：

    1. Initial -> Boundary
       加入第一批证据后，各语义维度是否上升？

    2. Boundary -> Complete
       加入完成 certificate 所需证据后，各维度是否继续上升？

    3. Complete >= Boundary 的比例
       用于发现明显的 Judge 非单调现象。

    注意：
        这里的 monotonicity 是“Judge consistency diagnostic”，
        不是说任何 Evidence 增加都必须严格提高分数。

        如果新增 Evidence 矛盾、错误、噪声很大，
        分数下降完全可能合理。

        因此当前只报告，不作为硬性 PASS/FAIL gate。
    """

    complete_triplets: list[
        tuple[
            Any,
            Any,
            Any,
        ]
    ] = []

    missing_tasks: list[str] = []

    for (
        task_id,
        states,
    ) in sorted(
        paired_results.items()
    ):
        if not all(
            state_type in states
            for state_type
            in PAIRED_STATE_TYPES
        ):
            missing_tasks.append(
                task_id
            )
            continue

        complete_triplets.append(
            (
                states["initial"],
                states[
                    "decision_boundary"
                ],
                states["complete"],
            )
        )

    def mean_or_none(
        values: Sequence[float],
    ) -> float | None:
        if not values:
            return None

        return float(
            mean(values)
        )

    transitions: dict[
        str,
        dict[str, Any],
    ] = {}

    for (
        transition_name,
        left_index,
        right_index,
    ) in (
        (
            "initial_to_boundary",
            0,
            1,
        ),
        (
            "boundary_to_complete",
            1,
            2,
        ),
        (
            "initial_to_complete",
            0,
            2,
        ),
    ):
        mean_delta: dict[
            str,
            float | None,
        ] = {}

        nondecrease_rate: dict[
            str,
            float | None,
        ] = {}

        positive_delta_rate: dict[
            str,
            float | None,
        ] = {}

        for field_name in (
            PAIRED_SCORE_FIELDS
        ):
            deltas: list[float] = []

            for triplet in (
                complete_triplets
            ):
                left = triplet[
                    left_index
                ]
                right = triplet[
                    right_index
                ]

                left_score = float(
                    left.mean_scores[
                        field_name
                    ]
                )

                right_score = float(
                    right.mean_scores[
                        field_name
                    ]
                )

                deltas.append(
                    right_score
                    - left_score
                )

            mean_delta[
                field_name
            ] = mean_or_none(
                deltas
            )

            nondecrease_rate[
                field_name
            ] = (
                sum(
                    delta >= 0.0
                    for delta in deltas
                )
                / len(deltas)
                if deltas
                else None
            )

            positive_delta_rate[
                field_name
            ] = (
                sum(
                    delta > 0.0
                    for delta in deltas
                )
                / len(deltas)
                if deltas
                else None
            )

        transitions[
            transition_name
        ] = {
            "mean_delta": (
                mean_delta
            ),
            "nondecrease_rate": (
                nondecrease_rate
            ),
            "positive_delta_rate": (
                positive_delta_rate
            ),
        }

    per_task: list[
        dict[str, Any]
    ] = []

    for (
        initial,
        boundary,
        complete,
    ) in complete_triplets:
        task_id = initial.task_id

        per_task.append(
            {
                "task_id": task_id,
                "evidence_count": {
                    "initial": (
                        initial.evidence_count
                    ),
                    "decision_boundary": (
                        boundary.evidence_count
                    ),
                    "complete": (
                        complete.evidence_count
                    ),
                },
                "final_decision": {
                    "initial": (
                        initial.final_decision
                    ),
                    "decision_boundary": (
                        boundary.final_decision
                    ),
                    "complete": (
                        complete.final_decision
                    ),
                },
                "overall_sufficiency": {
                    "initial": float(
                        initial.mean_scores[
                            "overall_sufficiency"
                        ]
                    ),
                    "decision_boundary": float(
                        boundary.mean_scores[
                            "overall_sufficiency"
                        ]
                    ),
                    "complete": float(
                        complete.mean_scores[
                            "overall_sufficiency"
                        ]
                    ),
                },
                "delta_overall": {
                    "initial_to_boundary": (
                        float(
                            boundary.mean_scores[
                                "overall_sufficiency"
                            ]
                        )
                        - float(
                            initial.mean_scores[
                                "overall_sufficiency"
                            ]
                        )
                    ),
                    "boundary_to_complete": (
                        float(
                            complete.mean_scores[
                                "overall_sufficiency"
                            ]
                        )
                        - float(
                            boundary.mean_scores[
                                "overall_sufficiency"
                            ]
                        )
                    ),
                    "initial_to_complete": (
                        float(
                            complete.mean_scores[
                                "overall_sufficiency"
                            ]
                        )
                        - float(
                            initial.mean_scores[
                                "overall_sufficiency"
                            ]
                        )
                    ),
                },
            }
        )

    return {
        "paired_task_count": (
            len(complete_triplets)
        ),
        "missing_triplet_task_count": (
            len(missing_tasks)
        ),
        "missing_triplet_task_ids": (
            missing_tasks
        ),
        "transitions": transitions,
        "per_task": per_task,
        "interpretation_note": (
            "paired metrics are calibration diagnostics. "
            "They do not change semantic-sufficiency-v1 thresholds "
            "and are not a frozen benchmark gate."
        ),
    }


def derive_semantic_state_expectation_metrics(
    semantic_report: Mapping[str, Any],
) -> dict[str, Any]:
    """
    从 SemanticEvaluationAccumulator.report() 中派生三个更容易解释的指标：

    1. initial_rejection_rate
       initial 被语义 Judge 判 expand 的比例。

       Initial K 通常为空，
       因此这是一个 sanity check。

    2. decision_boundary_rejection_rate
       decision_boundary 被判 expand 的比例。

       它检验“监督定义中的近充分负例”
       是否在独立语义 Judge 下也确实仍需扩展。

    3. complete_sufficiency_rate
       complete 被判 sufficient 的比例。

       它检验 deterministic certificate/state supervision
       与语义充分性判断的一致程度。

    不建议只报告一个把三类 state 混合起来的总体 sufficient rate，
    因为 state composition 会强烈影响总体数字。
    """

    by_state_type = (
        semantic_report.get(
            "by_state_type"
        )
        or {}
    )

    def sufficiency_rate(
        state_type: str,
    ) -> float | None:
        bucket = by_state_type.get(
            state_type
        )

        if not bucket:
            return None

        value = bucket.get(
            "semantic_sufficiency_rate"
        )

        if value is None:
            return None

        return float(value)

    initial_sufficient = (
        sufficiency_rate("initial")
    )

    boundary_sufficient = (
        sufficiency_rate(
            "decision_boundary"
        )
    )

    complete_sufficient = (
        sufficiency_rate("complete")
    )

    return {
        "initial_rejection_rate": (
            1.0 - initial_sufficient
            if initial_sufficient
            is not None
            else None
        ),
        "decision_boundary_rejection_rate": (
            1.0 - boundary_sufficient
            if boundary_sufficient
            is not None
            else None
        ),
        "complete_sufficiency_rate": (
            complete_sufficient
        ),
    }


# ===========================================================================
# 12. Markdown 报告
# ===========================================================================

def write_markdown_report(
    report: Mapping[str, Any],
    path: Path,
) -> None:
    """
    写人类可读报告。

    JSON 才是正式机器可读结果；
    Markdown 只做摘要。
    """

    audit = report["package_audit"]

    lines = [
        "# Evidence Semantic Evaluation",
        "",
        f"- Dataset version: `{report['dataset_version']}`",
        f"- Split: `{report['split']}`",
        f"- Mode: `{report['mode']}`",
        f"- Semantic rubric: `{report.get('semantic_rubric')}`",
        f"- Packages: `{audit['package_count']}`",
        f"- Empty packages: `{audit['empty_package_count']}`",
        f"- Mean evidence count: `{audit['mean_evidence_count']}`",
        f"- Mean prompt chars: `{audit['mean_prompt_chars']}`",
        f"- Max prompt chars: `{audit['max_prompt_chars']}`",
        f"- Oversized prompts: `{audit['oversized_prompt_count']}`",
        "",
    ]

    if report["mode"] == "mock":
        lines.extend(
            [
                "> ⚠️ 当前为 MOCK 模式。所有 semantic score 都是固定测试值，"
                "只能验证工程管线，禁止作为实验结果或论文指标。",
                "",
            ]
        )

    if report["mode"] == "deepseek":
        validity = (
            report.get(
                "scientific_validity"
            )
            or {}
        )

        if validity.get(
            "is_calibration_result"
        ):
            lines.extend(
                [
                    "> 当前为真实 DeepSeek Judge calibration run。"
                    "它用于校准 Semantic Rubric，不是 frozen benchmark 最终结果。",
                    "",
                ]
            )

        if (
            report.get(
                "semantic_rubric"
            )
            == SEMANTIC_RUBRIC_V2
        ):
            lines.extend(
                [
                    "> semantic-sufficiency-v2 使用 blind Judge："
                    "不向模型暴露 initial / decision_boundary / complete 等监督状态标签。",
                    "",
                ]
            )

        usage = (
            report.get(
                "deepseek_usage"
            )
            or {}
        )

        lines.extend(
            [
                "## DeepSeek Judge runtime",
                "",
                f"- Successful API calls: `{usage.get('successful_api_calls')}`",
                f"- Failed API calls: `{usage.get('failed_api_calls')}`",
                f"- Prompt tokens: `{usage.get('prompt_tokens')}`",
                f"- Completion tokens: `{usage.get('completion_tokens')}`",
                f"- Total tokens: `{usage.get('total_tokens')}`",
                f"- API latency seconds: `{usage.get('latency_seconds')}`",
                "",
            ]
        )

        expectation = (
            report.get(
                "semantic_state_expectation"
            )
            or {}
        )

        lines.extend(
            [
                "## Semantic state expectation",
                "",
                f"- Initial rejection rate: `{expectation.get('initial_rejection_rate')}`",
                f"- Decision-boundary rejection rate: `{expectation.get('decision_boundary_rejection_rate')}`",
                f"- Complete sufficiency rate: `{expectation.get('complete_sufficiency_rate')}`",
                "",
            ]
        )

        paired = (
            report.get(
                "paired_semantic_calibration"
            )
        )

        if paired:
            transitions = (
                paired.get("transitions")
                or {}
            )

            ib = (
                transitions.get(
                    "initial_to_boundary"
                )
                or {}
            )

            bc = (
                transitions.get(
                    "boundary_to_complete"
                )
                or {}
            )

            ib_delta = (
                ib.get("mean_delta")
                or {}
            )

            bc_delta = (
                bc.get("mean_delta")
                or {}
            )

            bc_nondecrease = (
                bc.get(
                    "nondecrease_rate"
                )
                or {}
            )

            lines.extend(
                [
                    "## Paired semantic calibration",
                    "",
                    f"- Paired tasks: `{paired.get('paired_task_count')}`",
                    "- Sampling unit: `task`",
                    "- Required trajectory: `Initial -> Decision Boundary -> Complete`",
                    f"- Mean Δ overall, Initial→Boundary: `{ib_delta.get('overall_sufficiency')}`",
                    f"- Mean Δ overall, Boundary→Complete: `{bc_delta.get('overall_sufficiency')}`",
                    f"- Overall non-decrease rate, Boundary→Complete: `{bc_nondecrease.get('overall_sufficiency')}`",
                    "",
                    "> Paired metrics are calibration diagnostics only; "
                    "they do not modify semantic-sufficiency-v1 thresholds.",
                    "",
                ]
            )

    lines.extend(
        [
            "## Package audit by state type",
            "",
            "| State type | Count | Empty | Mean Evidence | Max Evidence | Mean Prompt Chars | Max Prompt Chars | Oversized |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for (
        state_type,
        bucket,
    ) in audit[
        "by_state_type"
    ].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    state_type,
                    str(bucket["count"]),
                    str(bucket["empty_count"]),
                    str(
                        bucket[
                            "mean_evidence_count"
                        ]
                    ),
                    str(
                        bucket[
                            "max_evidence_count"
                        ]
                    ),
                    str(
                        bucket[
                            "mean_prompt_chars"
                        ]
                    ),
                    str(
                        bucket[
                            "max_prompt_chars"
                        ]
                    ),
                    str(
                        bucket[
                            "oversized_count"
                        ]
                    ),
                ]
            )
            + " |"
        )

    if (
        report["mode"] == "mock"
        and report.get(
            "semantic_evaluation"
        )
    ):
        semantic = report[
            "semantic_evaluation"
        ]

        lines.extend(
            [
                "",
                "## MOCK semantic aggregation",
                "",
                "再次强调：以下数值不是正式实验结果。",
                "",
                "```json",
                json.dumps(
                    semantic,
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
            ]
        )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ===========================================================================
# 12. CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and audit semantic evidence sufficiency "
            "evaluation requests for V2.10."
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
        help="Policy Evidence Cache SQLite。",
    )

    parser.add_argument(
        "--split",
        choices=SUPPORTED_SPLITS,
        default="validation",
        help=(
            "建议先用 validation 建立评估协议；"
            "协议冻结后再跑 benchmark。"
        ),
    )

    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default="audit",
        help=(
            "audit=只审计 q+K；"
            "export=导出 Judge 请求；"
            "mock=固定假 Judge 验证整条管线；"
            "deepseek=调用真实 DeepSeek Chat Completions Semantic Judge。"
        ),
    )

    parser.add_argument(
        "--state-types",
        nargs="+",
        default=list(
            DEFAULT_STATE_TYPES
        ),
        help=(
            "要评估的 state_type。"
            "默认 initial decision_boundary complete。"
        ),
    )

    parser.add_argument(
        "--max-packages",
        type=int,
        default=None,
        help=(
            "仅 smoke/debug 使用。"
            "正式 validation/benchmark 不设置。"
        ),
    )

    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=200_000,
        help=(
            "只用于 oversized 审计，不执行截断。"
            "超过该值的 prompt 仍保留，但会计入 oversized。"
        ),
    )

    parser.add_argument(
        "--judge-repeats",
        type=int,
        default=1,
        help="每个 package 的 Judge 重复次数。",
    )

    parser.add_argument(
        "--semantic-rubric",
        choices=sorted(
            SUPPORTED_RUBRIC_VERSIONS
        ),
        default=(
            DEFAULT_SEMANTIC_RUBRIC_VERSION
        ),
        help=(
            "Semantic Judge 协议版本。"
            "默认 semantic-sufficiency-v2；"
            "semantic-sufficiency-v1 只用于复现历史 calibration。"
        ),
    )

    parser.add_argument(
        "--sufficiency-threshold",
        type=float,
        default=0.75,
        help="semantic overall sufficiency 阈值。",
    )

    parser.add_argument(
        "--critical-dimension-floor",
        type=float,
        default=0.50,
        help="关键语义维度最低分。",
    )

    # ------------------------------------------------------------------
    # DeepSeek Semantic Judge 参数。
    #
    # 这些参数只在 --mode deepseek 时生效。
    # ------------------------------------------------------------------

    parser.add_argument(
        "--deepseek-model",
        default=DEFAULT_DEEPSEEK_MODEL,
        help=(
            "DeepSeek Semantic Judge 模型。"
            "默认读取 DEEPSEEK_SEMANTIC_JUDGE_MODEL；"
            "若环境变量未设置，则使用 judge.py 的默认模型。"
        ),
    )

    parser.add_argument(
        "--deepseek-reasoning-effort",
        default="high",
        help=(
            "DeepSeek reasoning effort。"
            "传入 none 表示不显式设置 reasoning 参数。"
            "正式实验前必须冻结该值。"
        ),
    )

    parser.add_argument(
        "--deepseek-thinking",
        choices=[
            "enabled",
            "disabled",
        ],
        default="enabled",
        help=(
            "DeepSeek Thinking Mode。正式 Semantic Judge 默认 enabled。"
        ),
    )

    parser.add_argument(
        "--deepseek-max-output-tokens",
        type=int,
        default=1800,
        help=(
            "DeepSeek JSON Judge 最大输出 token。"
            "如果 response 因该限制 incomplete，样本直接失败。"
        ),
    )

    parser.add_argument(
        "--deepseek-timeout-seconds",
        type=float,
        default=180.0,
        help="DeepSeek 单次 HTTP 请求超时秒数。",
    )

    parser.add_argument(
        "--deepseek-max-retries",
        type=int,
        default=3,
        help=(
            "交给 OpenAI Python SDK 的自动 retry 次数。"
            "本脚本不叠加第二套 HTTP 重试。"
        ),
    )

    parser.add_argument(
        "--deepseek-store",
        action="store_true",
        help=(
            "兼容保留参数；当前 DeepSeek Chat Completions 请求不会发送 store。"
            "默认 false。"
        ),
    )

    parser.add_argument(
        "--retain-deepseek-raw-output",
        action="store_true",
        help=(
            "在 deepseek_runs.jsonl 中额外保留最终 JSON content。"
            "默认只保存结构化结果与调用元数据。"
        ),
    )

    # ------------------------------------------------------------------
    # Calibration set。
    #
    # 默认先跑 10 / 20 / 20，而不是直接调用全部 523 个 validation state。
    # 这样可以先检查 Judge 是否真正区分：
    #     Empty
    #     Near-sufficient
    #     Sufficient
    # 再决定是否扩大实验。
    # ------------------------------------------------------------------

    parser.add_argument(
        "--deepseek-initial-limit",
        type=int,
        default=DEFAULT_DEEPSEEK_SAMPLE_LIMITS[
            "initial"
        ],
        help="DeepSeek calibration 中 initial 样本数，默认 10。",
    )

    parser.add_argument(
        "--deepseek-boundary-limit",
        type=int,
        default=DEFAULT_DEEPSEEK_SAMPLE_LIMITS[
            "decision_boundary"
        ],
        help=(
            "DeepSeek calibration 中 decision_boundary 样本数，默认 20。"
        ),
    )

    parser.add_argument(
        "--deepseek-complete-limit",
        type=int,
        default=DEFAULT_DEEPSEEK_SAMPLE_LIMITS[
            "complete"
        ],
        help="DeepSeek calibration 中 complete 样本数，默认 20。",
    )

    parser.add_argument(
        "--deepseek-paired-tasks",
        type=int,
        default=DEFAULT_DEEPSEEK_PAIRED_TASKS,
        help=(
            "启用同 task 的 Initial/Boundary/Complete 三联 calibration。"
            "0=关闭。建议先用 5~10；"
            "启用后独立的 initial/boundary/complete limit 不再用于选样。"
        ),
    )

    parser.add_argument(
        "--deepseek-sample-seed",
        type=int,
        default=DEFAULT_DEEPSEEK_SAMPLE_SEED,
        help=(
            "DeepSeek calibration 稳定哈希抽样 seed。"
            "相同数据和 seed 会得到同一批 state。"
        ),
    )

    parser.add_argument(
        "--allow-full-deepseek-run",
        action="store_true",
        help=(
            "显式允许对所选 state_types 全量调用 DeepSeek。"
            "未设置时，deepseek 模式默认只运行 calibration set。"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出目录。",
    )

    return parser.parse_args()


# ===========================================================================
# 13. 主流程
# ===========================================================================

def main() -> int:
    args = parse_args()

    if (
        args.max_packages is not None
        and args.max_packages < 1
    ):
        raise ValueError(
            "--max-packages 必须 >= 1"
        )

    if (
        args.mode == "deepseek"
        and args.max_packages is not None
    ):
        raise ValueError(
            "DeepSeek calibration 禁止使用 --max-packages，"
            "因为它会破坏分层抽样比例。\n"
            "三样本 smoke 请使用：\n"
            "  --deepseek-initial-limit 1\n"
            "  --deepseek-boundary-limit 1\n"
            "  --deepseek-complete-limit 1"
        )

    if args.max_prompt_chars < 1:
        raise ValueError(
            "--max-prompt-chars 必须 >= 1"
        )

    if args.judge_repeats < 1:
        raise ValueError(
            "--judge-repeats 必须 >= 1"
        )

    if args.deepseek_max_output_tokens < 1:
        raise ValueError(
            "--deepseek-max-output-tokens 必须 >= 1"
        )

    if args.deepseek_timeout_seconds <= 0:
        raise ValueError(
            "--deepseek-timeout-seconds 必须 > 0"
        )

    if args.deepseek_max_retries < 0:
        raise ValueError(
            "--deepseek-max-retries 不能为负数"
        )

    if (
        args.mode == "deepseek"
        and args.deepseek_thinking != "enabled"
    ):
        print(
            "WARNING: DeepSeek Thinking Mode 当前为 disabled。"
            "这可以用于消融/debug，但不应与 enabled 的正式结果混合。",
            file=sys.stderr,
            flush=True,
        )

    if args.deepseek_paired_tasks < 0:
        raise ValueError(
            "--deepseek-paired-tasks 不能为负数"
        )

    if (
        args.deepseek_paired_tasks > 0
        and args.mode != "deepseek"
    ):
        raise ValueError(
            "--deepseek-paired-tasks 只允许与 --mode deepseek 一起使用"
        )

    if (
        args.deepseek_paired_tasks > 0
        and args.allow_full_deepseek_run
    ):
        raise ValueError(
            "--deepseek-paired-tasks 与 --allow-full-deepseek-run 互斥"
        )

    for (
        option_name,
        option_value,
    ) in (
        (
            "--deepseek-initial-limit",
            args.deepseek_initial_limit,
        ),
        (
            "--deepseek-boundary-limit",
            args.deepseek_boundary_limit,
        ),
        (
            "--deepseek-complete-limit",
            args.deepseek_complete_limit,
        ),
    ):
        if option_value < 0:
            raise ValueError(
                f"{option_name} 不能为负数"
            )

    selected_state_types = set(
        args.state_types
    )

    if not selected_state_types:
        raise ValueError(
            "--state-types 不能为空"
        )

    if (
        args.deepseek_paired_tasks > 0
        and not set(
            PAIRED_STATE_TYPES
        ).issubset(
            selected_state_types
        )
    ):
        raise ValueError(
            "paired calibration 必须包含 state_types："
            "initial decision_boundary complete"
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

    # ------------------------------------------------------------------
    # DeepSeek 模式默认只运行 calibration set。
    #
    # 只有用户显式传：
    #     --allow-full-deepseek-run
    #
    # 才会对 selected_state_types 的全部 state 调用 API。
    # ------------------------------------------------------------------

    deepseek_selected_state_keys: (
        set[tuple[str, str]]
        | None
    ) = None

    paired_selection_metadata: (
        dict[str, Any]
        | None
    ) = None

    if (
        args.mode == "deepseek"
        and not args.allow_full_deepseek_run
    ):
        if args.deepseek_paired_tasks > 0:
            (
                deepseek_selected_state_keys,
                paired_selection_metadata,
            ) = (
                select_deepseek_paired_calibration_states(
                    split_path=split_path,
                    seed=(
                        args.deepseek_sample_seed
                    ),
                    task_limit=(
                        args.deepseek_paired_tasks
                    ),
                )
            )

        else:
            deepseek_selected_state_keys = (
                select_deepseek_calibration_states(
                    split_path=split_path,
                    selected_state_types=(
                        selected_state_types
                    ),
                    seed=(
                        args.deepseek_sample_seed
                    ),
                    initial_limit=(
                        args.deepseek_initial_limit
                    ),
                    boundary_limit=(
                        args.deepseek_boundary_limit
                    ),
                    complete_limit=(
                        args.deepseek_complete_limit
                    ),
                )
            )

        if not deepseek_selected_state_keys:
            raise ValueError(
                "DeepSeek calibration set 为空。"
                "请检查 calibration 参数。"
            )

    config = SemanticEvaluationConfig(
        sufficiency_threshold=(
            args.sufficiency_threshold
        ),
        critical_dimension_floor=(
            args.critical_dimension_floor
        ),
        repeats=args.judge_repeats,
    )

    config.validate()

    output_dir = (
        args.output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_json_path = (
        output_dir
        / "report.json"
    )

    report_md_path = (
        output_dir
        / "report.md"
    )

    requests_path = (
        output_dir
        / "semantic_requests.jsonl"
    )

    packages_path = (
        output_dir
        / "semantic_packages.jsonl"
    )

    mock_runs_path = (
        output_dir
        / "mock_runs.jsonl"
    )

    deepseek_runs_path = (
        output_dir
        / "deepseek_runs.jsonl"
    )

    deepseek_errors_path = (
        output_dir
        / "deepseek_errors.jsonl"
    )

    evidence_cache = (
        SemanticEvidenceCache(
            args.evidence_cache
        )
    )

    audit = PackageAuditAccumulator()

    semantic_accumulator = (
        SemanticEvaluationAccumulator()
    )

    # paired calibration 只保留每个 task 三个 state 的聚合结果，
    # 数量很小（例如 10 tasks -> 30 states），不会形成内存压力。
    paired_aggregate_by_task: defaultdict[
        str,
        dict[str, Any],
    ] = defaultdict(dict)

    package_count = 0
    skipped_state_type_count = 0

    # ------------------------------------------------------------------
    # DeepSeek Judge 只在 --mode deepseek 时初始化。
    #
    # API Key 不会进入 report。
    # validate_deepseek_environment() 只检查：
    #     key 是否存在
    #     OpenAI-compatible Python SDK 是否安装
    # ------------------------------------------------------------------

    deepseek_judge = None
    deepseek_judge_config = None
    deepseek_environment = None

    deepseek_usage = {
        "successful_api_calls": 0,
        "failed_api_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": 0.0,
    }

    if args.mode == "deepseek":
        reasoning_effort = (
            None
            if str(
                args.deepseek_reasoning_effort
            ).strip().lower()
            == "none"
            else str(
                args.deepseek_reasoning_effort
            ).strip()
        )

        deepseek_judge_config = (
            DeepSeekJudgeConfig(
                model=args.deepseek_model,
                reasoning_effort=(
                    reasoning_effort
                ),
                thinking_type=(
                    args.deepseek_thinking
                ),
                max_output_tokens=(
                    args.deepseek_max_output_tokens
                ),
                timeout_seconds=(
                    args.deepseek_timeout_seconds
                ),
                max_retries=(
                    args.deepseek_max_retries
                ),
                store=(
                    args.deepseek_store
                ),
                include_raw_response_text=(
                    args
                    .retain_deepseek_raw_output
                ),
            )
        )

        deepseek_environment = (
            validate_deepseek_environment(
                deepseek_judge_config
            )
        )

        if not deepseek_environment[
            "openai_sdk_present"
        ]:
            raise RuntimeError(
                "当前环境未安装 openai Python SDK。\\n"
                "请执行：\\n"
                "  python -m pip install -U openai"
            )

        if not deepseek_environment[
            "api_key_present"
        ]:
            raise RuntimeError(
                "当前环境未设置 DEEPSEEK_API_KEY（或兼容的 OPENAI_API_KEY）。\\n"
                "PowerShell 示例：\\n"
                '  $env:DEEPSEEK_API_KEY = "你的 DeepSeek API Key"'
            )

        deepseek_judge = (
            DeepSeekSemanticJudge(
                deepseek_judge_config
            )
        )

    started = time.perf_counter()

    # export 模式才需要真正写 Judge 请求；
    # audit / mock 不创建空的 requests 文件。
    requests_file = (
        requests_path.open(
            "w",
            encoding="utf-8",
        )
        if args.mode == "export"
        else None
    )

    # 所有模式都保留 package manifest，
    # 方便后续复查“到底评价了哪些 q+K”。
    packages_file = (
        packages_path.open(
            "w",
            encoding="utf-8",
        )
    )

    mock_runs_file = (
        mock_runs_path.open(
            "w",
            encoding="utf-8",
        )
        if args.mode == "mock"
        else None
    )

    deepseek_runs_file = (
        deepseek_runs_path.open(
            "w",
            encoding="utf-8",
        )
        if args.mode == "deepseek"
        else None
    )

    deepseek_errors_file = (
        deepseek_errors_path.open(
            "w",
            encoding="utf-8",
        )
        if args.mode == "deepseek"
        else None
    )

    try:
        stop_requested = False

        for task_row in iter_task_rows(
            split_path
        ):
            task_id = str(
                task_row.get("task_id")
                or ""
            )

            if not task_id:
                raise ValueError(
                    "Parquet row 缺少 task_id"
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
                supervision.get(
                    "policy_states"
                )
                or []
            )

            for state in states:
                state_type = str(
                    state.get(
                        "state_type"
                    )
                    or "unknown"
                )

                if (
                    state_type
                    not in selected_state_types
                ):
                    skipped_state_type_count += 1
                    continue

                # -------------------------------------------------------
                # DeepSeek calibration：
                # 只对稳定哈希抽样得到的 state 调用 API。
                #
                # --allow-full-deepseek-run 时：
                #     deepseek_selected_state_keys = None
                # 因此不会执行该过滤。
                # -------------------------------------------------------

                if (
                    args.mode == "deepseek"
                    and deepseek_selected_state_keys
                    is not None
                ):
                    candidate_state_id = str(
                        state.get("state_id")
                        or ""
                    )

                    if (
                        task_id,
                        candidate_state_id,
                    ) not in deepseek_selected_state_keys:
                        continue

                package = (
                    build_semantic_package(
                        task_id=task_id,
                        task_input=task_input,
                        state=state,
                        evidence_cache=(
                            evidence_cache
                        ),
                    )
                )

                request = (
                    SemanticJudgeRequest(
                        package=package,
                        reference=None,
                        rubric_version=(
                            args.semantic_rubric
                        ),
                        judge_run_index=0,
                    )
                )

                user_prompt = (
                    build_semantic_judge_prompt(
                        request
                    )
                )

                # system prompt 也必须与 request.rubric_version 同源。
                system_prompt = (
                    semantic_judge_system_prompt(
                        request.rubric_version
                    )
                )

                prompt_char_count = (
                    len(system_prompt)
                    + len(user_prompt)
                )

                oversized = (
                    prompt_char_count
                    > args.max_prompt_chars
                )

                audit.add(
                    package,
                    prompt_char_count=(
                        prompt_char_count
                    ),
                    oversized=oversized,
                )

                # -------------------------------------------------------
                # 保存 package manifest。
                #
                # 不直接把全部代码 content 重复写入 package JSONL，
                # 避免输出文件异常膨胀。
                #
                # semantic_requests.jsonl 的 user_prompt 在 export 模式
                # 已经包含完整 Evidence content。
                # -------------------------------------------------------

                package_record = {
                    "task_id": (
                        package.task_id
                    ),
                    "state_id": (
                        package.state_id
                    ),
                    "state_type": (
                        package.state_type
                    ),
                    "evidence_count": (
                        package.evidence_count
                    ),
                    "evidence_ids": [
                        item.evidence_id
                        for item
                        in package.evidence
                    ],
                    "question_char_count": (
                        len(
                            package.question
                        )
                    ),
                    "prompt_char_count": (
                        prompt_char_count
                    ),
                    "oversized_prompt": (
                        oversized
                    ),
                    "supervision_expected_stop": (
                        package.metadata.get(
                            "supervision_expected_stop"
                        )
                    ),
                }

                packages_file.write(
                    json.dumps(
                        package_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                # -------------------------------------------------------
                # export：
                # 导出真实 Judge 请求。
                # -------------------------------------------------------

                if args.mode == "export":
                    assert (
                        requests_file
                        is not None
                    )

                    request_record = {
                        "task_id": (
                            package.task_id
                        ),
                        "state_id": (
                            package.state_id
                        ),
                        "state_type": (
                            package.state_type
                        ),
                        "judge_run_index": 0,
                        "rubric_version": (
                            request.rubric_version
                        ),
                        "evidence_count": (
                            package.evidence_count
                        ),
                        "prompt_char_count": (
                            prompt_char_count
                        ),
                        "oversized_prompt": (
                            oversized
                        ),
                        "system_prompt": (
                            system_prompt
                        ),
                        "user_prompt": (
                            user_prompt
                        ),
                    }

                    requests_file.write(
                        json.dumps(
                            request_record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                # -------------------------------------------------------
                # mock：
                # 用固定输出验证 semantic.py 全链路。
                #
                # 绝对不能解释这些分数。
                # -------------------------------------------------------

                elif args.mode == "mock":
                    assert (
                        mock_runs_file
                        is not None
                    )

                    mock_output = (
                        default_mock_judge_output(
                            package
                        )
                    )

                    judge = (
                        StaticSemanticJudge(
                            mock_output
                        )
                    )

                    runs = []

                    for repeat_index in range(
                        args.judge_repeats
                    ):
                        repeated_request = (
                            SemanticJudgeRequest(
                                package=package,
                                reference=None,
                                rubric_version=(
                                    args.semantic_rubric
                                ),
                                judge_run_index=(
                                    repeat_index
                                ),
                            )
                        )

                        run = (
                            evaluate_semantic_once(
                                request=(
                                    repeated_request
                                ),
                                judge=judge,
                                config=config,
                            )
                        )

                        runs.append(run)

                        mock_runs_file.write(
                            json.dumps(
                                run.to_dict(),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                    aggregate = (
                        aggregate_semantic_runs(
                            runs
                        )
                    )

                    semantic_accumulator.add(
                        aggregate
                    )

                # -------------------------------------------------------
                # deepseek：
                # 真实调用 DeepSeek Chat Completions Semantic Judge。
                #
                # 每个 repeat：
                #   1. 调 DeepSeek 一次；
                #   2. 保存 API metadata；
                #   3. 把 Structured Output 再交给 semantic.py 的
                #      parser / evidence-id groundedness / computed decision；
                #   4. 全部 repeats 成功后再 aggregate。
                #
                # 任一 API/协议错误默认 fail-fast。
                # 这是为了防止悄悄丢样本后仍报告一个看似完整的指标。
                # -------------------------------------------------------

                elif args.mode == "deepseek":
                    assert (
                        deepseek_judge
                        is not None
                    )
                    assert (
                        deepseek_judge_config
                        is not None
                    )
                    assert (
                        deepseek_runs_file
                        is not None
                    )
                    assert (
                        deepseek_errors_file
                        is not None
                    )

                    runs = []

                    for repeat_index in range(
                        args.judge_repeats
                    ):
                        repeated_request = (
                            SemanticJudgeRequest(
                                package=package,
                                reference=None,
                                rubric_version=(
                                    args.semantic_rubric
                                ),
                                judge_run_index=(
                                    repeat_index
                                ),
                            )
                        )

                        try:
                            api_call = (
                                deepseek_judge
                                .judge_with_metadata(
                                    repeated_request
                                )
                            )

                            # -------------------------------------------
                            # 注意：
                            # 这里不能再把 deepseek_judge 直接交给
                            # evaluate_semantic_once()，
                            # 否则同一个样本会调用两次 API。
                            #
                            # 因此使用 StaticSemanticJudge 包装本次
                            # 已经取得的 Structured Output，
                            # 只复用 semantic.py 的正式验证逻辑。
                            # -------------------------------------------

                            protocol_judge = (
                                StaticSemanticJudge(
                                    api_call.output
                                )
                            )

                            run = (
                                evaluate_semantic_once(
                                    request=(
                                        repeated_request
                                    ),
                                    judge=(
                                        protocol_judge
                                    ),
                                    config=config,
                                )
                            )

                        except Exception as exc:
                            deepseek_usage[
                                "failed_api_calls"
                            ] += 1

                            error_record = {
                                "task_id": (
                                    package.task_id
                                ),
                                "state_id": (
                                    package.state_id
                                ),
                                "state_type": (
                                    package.state_type
                                ),
                                "judge_run_index": (
                                    repeat_index
                                ),
                                "error_type": (
                                    type(exc).__name__
                                ),
                                "error": str(exc),
                            }

                            deepseek_errors_file.write(
                                json.dumps(
                                    error_record,
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            deepseek_errors_file.flush()

                            # 正式语义评估默认 fail-fast。
                            raise

                        runs.append(run)

                        call_metadata = (
                            api_call.metadata.to_dict()
                        )

                        prompt_tokens = (
                            call_metadata.get(
                                "prompt_tokens"
                            )
                            or 0
                        )

                        completion_tokens = (
                            call_metadata.get(
                                "completion_tokens"
                            )
                            or 0
                        )

                        total_tokens = (
                            call_metadata.get(
                                "total_tokens"
                            )
                            or (
                                prompt_tokens
                                + completion_tokens
                            )
                        )

                        deepseek_usage[
                            "successful_api_calls"
                        ] += 1

                        deepseek_usage[
                            "prompt_tokens"
                        ] += int(
                            prompt_tokens
                        )

                        deepseek_usage[
                            "completion_tokens"
                        ] += int(
                            completion_tokens
                        )

                        deepseek_usage[
                            "total_tokens"
                        ] += int(
                            total_tokens
                        )

                        deepseek_usage[
                            "latency_seconds"
                        ] += float(
                            call_metadata.get(
                                "latency_seconds"
                            )
                            or 0.0
                        )

                        deepseek_run_record = {
                            "semantic_run": (
                                run.to_dict()
                            ),
                            "deepseek_call": (
                                api_call.to_dict()
                            ),
                        }

                        deepseek_runs_file.write(
                            json.dumps(
                                deepseek_run_record,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        deepseek_runs_file.flush()

                    aggregate = (
                        aggregate_semantic_runs(
                            runs
                        )
                    )

                    semantic_accumulator.add(
                        aggregate
                    )

                    if (
                        args.deepseek_paired_tasks
                        > 0
                    ):
                        task_bucket = (
                            paired_aggregate_by_task[
                                aggregate.task_id
                            ]
                        )

                        if (
                            aggregate.state_type
                            in task_bucket
                        ):
                            raise ValueError(
                                "paired calibration 同一 task/state_type "
                                "出现重复 aggregate："
                                f"task={aggregate.task_id}, "
                                f"state_type={aggregate.state_type}"
                            )

                        task_bucket[
                            aggregate.state_type
                        ] = aggregate

                package_count += 1

                if (
                    args.max_packages
                    is not None
                    and package_count
                    >= args.max_packages
                ):
                    stop_requested = True
                    break

                progress_interval = (
                    5
                    if args.mode == "deepseek"
                    else 250
                )

                if (
                    package_count
                    % progress_interval
                    == 0
                ):
                    elapsed = max(
                        time.perf_counter()
                        - started,
                        1e-9,
                    )

                    print(
                        "semantic_eval_progress: "
                        f"packages={package_count:,}, "
                        "package_per_sec="
                        f"{package_count / elapsed:.3f}",
                        file=sys.stderr,
                        flush=True,
                    )

            if stop_requested:
                break

        elapsed_seconds = (
            time.perf_counter()
            - started
        )

        semantic_report = (
            semantic_accumulator.report()
            if args.mode
            in {
                "mock",
                "deepseek",
            }
            else None
        )

        semantic_state_expectation = (
            derive_semantic_state_expectation_metrics(
                semantic_report
            )
            if semantic_report is not None
            else None
        )

        paired_semantic_metrics = None

        if (
            args.mode == "deepseek"
            and args.deepseek_paired_tasks > 0
        ):
            paired_semantic_metrics = (
                derive_paired_semantic_metrics(
                    paired_aggregate_by_task
                )
            )

            if (
                paired_semantic_metrics[
                    "paired_task_count"
                ]
                != args.deepseek_paired_tasks
            ):
                raise ValueError(
                    "paired calibration 实际完成 task 数与请求不一致："
                    f"expected={args.deepseek_paired_tasks}, "
                    "actual="
                    f"{paired_semantic_metrics['paired_task_count']}"
                )

        deepseek_runtime_contract = None

        if args.mode == "deepseek":
            deepseek_runtime_contract = {
                "semantic_rubric": (
                    args.semantic_rubric
                ),
                "thinking_requested": (
                    args.deepseek_thinking
                ),
                "reasoning_effort_requested": (
                    args.deepseek_reasoning_effort
                ),
                "model_requested": (
                    args.deepseek_model
                ),
                "expected_api_calls": (
                    package_count
                    * args.judge_repeats
                ),
                "successful_api_calls": (
                    deepseek_usage[
                        "successful_api_calls"
                    ]
                ),
                "failed_api_calls": (
                    deepseek_usage[
                        "failed_api_calls"
                    ]
                ),
                "all_calls_accounted_for": (
                    (
                        deepseek_usage[
                            "successful_api_calls"
                        ]
                        + deepseek_usage[
                            "failed_api_calls"
                        ]
                    )
                    == (
                        package_count
                        * args.judge_repeats
                    )
                ),
            }

        report: dict[str, Any] = {
            "evaluation_name": (
                "evidence_semantic_sufficiency_v2_10"
            ),
            "semantic_rubric": (
                args.semantic_rubric
            ),
            "dataset_version": (
                DATASET_VERSION
            ),
            "dataset_dir": str(
                dataset_dir
            ),
            "manifest": str(
                manifest_path
            ),
            "manifest_audit_status": (
                manifest.get(
                    "audit_status"
                )
            ),
            "split": args.split,
            "split_path": str(
                split_path
            ),
            "mode": args.mode,
            "selected_state_types": sorted(
                selected_state_types
            ),
            "max_packages": (
                args.max_packages
            ),
            "max_prompt_chars": (
                args.max_prompt_chars
            ),
            "package_count": (
                package_count
            ),
            "skipped_state_type_count": (
                skipped_state_type_count
            ),
            "timing_seconds": round(
                elapsed_seconds,
                3,
            ),
            "semantic_protocol": (
                semantic_protocol_metadata(
                    config,
                    rubric_version=(
                        args.semantic_rubric
                    ),
                )
            ),
            "package_audit": (
                audit.report()
            ),
            "semantic_evaluation": (
                semantic_report
            ),
            "semantic_state_expectation": (
                semantic_state_expectation
            ),
            "deepseek_judge": (
                deepseek_judge_protocol_metadata(
                    deepseek_judge_config
                )
                if deepseek_judge_config
                is not None
                else None
            ),
            "deepseek_environment": (
                deepseek_environment
                if args.mode == "deepseek"
                else None
            ),
            "deepseek_sampling": (
                {
                    "full_run": bool(
                        args.allow_full_deepseek_run
                    ),
                    "stable_hash_seed": (
                        args.deepseek_sample_seed
                    ),
                    "initial_limit": (
                        None
                        if args.allow_full_deepseek_run
                        else args.deepseek_initial_limit
                    ),
                    "decision_boundary_limit": (
                        None
                        if args.allow_full_deepseek_run
                        else args.deepseek_boundary_limit
                    ),
                    "complete_limit": (
                        None
                        if args.allow_full_deepseek_run
                        else args.deepseek_complete_limit
                    ),
                    "selected_state_count": (
                        None
                        if deepseek_selected_state_keys
                        is None
                        else len(
                            deepseek_selected_state_keys
                        )
                    ),
                    "paired_tasks_requested": (
                        args.deepseek_paired_tasks
                    ),
                    "paired_selection": (
                        paired_selection_metadata
                    ),
                }
                if args.mode == "deepseek"
                else None
            ),
            "paired_semantic_calibration": (
                paired_semantic_metrics
            ),
            "deepseek_runtime_contract": (
                deepseek_runtime_contract
            ),
            "deepseek_usage": (
                {
                    **deepseek_usage,
                    "latency_seconds": round(
                        float(
                            deepseek_usage[
                                "latency_seconds"
                            ]
                        ),
                        3,
                    ),
                }
                if args.mode == "deepseek"
                else None
            ),
            "scientific_validity": {
                "semantic_rubric": (
                    args.semantic_rubric
                ),
                "judge_blind_to_supervision_state": (
                    args.semantic_rubric
                    == SEMANTIC_RUBRIC_V2
                ),
                "has_real_judge_results": (
                    args.mode == "deepseek"
                ),
                "is_calibration_result": (
                    args.mode == "deepseek"
                    and not args.allow_full_deepseek_run
                ),
                "is_frozen_benchmark_result": (
                    args.mode == "deepseek"
                    and args.split == "benchmark"
                    and args.allow_full_deepseek_run
                    and args.max_packages is None
                ),
                "note": (
                    "audit/export 不产生 Judge 结果；"
                    "mock 使用固定假 Judge，严禁作为论文实验结果；"
                    "deepseek calibration 使用真实 Judge，"
                    "但用于协议校准，不应当作 frozen benchmark 最终结果；"
                    "semantic-sufficiency-v2 对 Judge 隐藏 supervision state_type。"
                ),
            },
            "outputs": {
                "report_json": str(
                    report_json_path
                ),
                "report_markdown": str(
                    report_md_path
                ),
                "semantic_packages_jsonl": str(
                    packages_path
                ),
                "semantic_requests_jsonl": (
                    str(requests_path)
                    if args.mode == "export"
                    else None
                ),
                "mock_runs_jsonl": (
                    str(mock_runs_path)
                    if args.mode == "mock"
                    else None
                ),
                "deepseek_runs_jsonl": (
                    str(deepseek_runs_path)
                    if args.mode == "deepseek"
                    else None
                ),
                "deepseek_errors_jsonl": (
                    str(deepseek_errors_path)
                    if args.mode == "deepseek"
                    else None
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
            "mode": args.mode,
            "split": args.split,
            "packages": package_count,
            "empty_packages": (
                report[
                    "package_audit"
                ][
                    "empty_package_count"
                ]
            ),
            "oversized_prompts": (
                report[
                    "package_audit"
                ][
                    "oversized_prompt_count"
                ]
            ),
            "report_json": str(
                report_json_path
            ),
            "report_md": str(
                report_md_path
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
        evidence_cache.close()

        packages_file.close()

        if requests_file is not None:
            requests_file.close()

        if mock_runs_file is not None:
            mock_runs_file.close()

        if deepseek_runs_file is not None:
            deepseek_runs_file.close()

        if deepseek_errors_file is not None:
            deepseek_errors_file.close()

        if deepseek_judge is not None:
            deepseek_judge.close()


if __name__ == "__main__":
    raise SystemExit(main())
