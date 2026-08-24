# -*- coding: utf-8 -*-
"""
Full-Scope Supervision Refinement Runner v1.4
（全任务监督修正执行器 v1.4）

建议位置：
    scripts/refine_supervision_with_llm.py

========================================================================
一、目标
========================================================================

本脚本服务于“训练前监督质量门”。

核心目标不是：
    “尽可能给每个任务生成一个答案”

而是：
    “只让经过严格验证的监督进入后续训练”。

因此：

    - 原始 V2.10 永远只读；
    - 所有修正写 Sidecar（旁路文件）；
    - 不确定、冲突、候选不足、结构不合法的任务一律 blocked；
    - 默认小样本模式：5 条；
    - 默认 review-policy=changed：
          Qwen3-8B 主 Teacher 处理全部样本；
          DeepSeek 只复核“发生修改 / 低置信 / 结构异常 / 不确定”样本；
          高置信 accepted KEEP 默认不调用 DeepSeek。

========================================================================
二、任务范围
========================================================================

task-scope=all：
    只要求存在 Complete（完成状态）。
    因此可以覆盖绝大多数 / 全部 V2.10 任务。

task-scope=boundary：
    要求同时存在：
        Decision Boundary（边界状态）
        Complete（完成状态）
    兼容此前 556-task 校准实验。

没有 Boundary 不代表监督正确。
对于没有 Boundary 的任务，仍然检查：

    - Obligation（证据要求）；
    - Witness（支撑证据）；
    - Complete Certificate（完成证书）；
    - 是否缺 Evidence；
    - 是否有冗余 Evidence；
    - 是否存在 Question-satisfied Requirement
      （问题描述已经满足的必要信息）。

========================================================================
三、Teacher Routing（教师路由）
========================================================================

Primary Teacher（主教师）：
    SiliconFlow + Qwen/Qwen3-8B

环境变量：
    OPENAI_BASE_URL
    OPENAI_API_KEY
    OPENAI_API_KEY_2
    OPENAI_API_KEY_3
    OPENAI_API_KEY_4
    LLM_MODEL

并发：
    每个非空 Key 最多 4 个请求。
    4 个 Key 全部存在时，总并发最多 16。

调用：
    thinking = disabled

Strong Review（强模型复核）：
    DeepSeek

环境变量：
    DEEPSEEK_API_KEY

调用：
    thinking = disabled

默认：
    review-policy=changed

DeepSeek 只复核高风险样本：
    - Qwen3 verification_status != accepted
    - Qwen3 assessment != keep
    - Qwen3 confidence 低于阈值

高置信 accepted KEEP 默认不调用 DeepSeek。

========================================================================
四、为什么不让 8B 单独直接生成训练标签
========================================================================

8B 适合作为大规模 First-pass Teacher（第一阶段教师），
但训练监督质量要求高。

因此默认只有下面条件全部满足，才定义：

    supervision_verified = true

条件：

    1. Qwen3 输出可解析；
    2. Qwen3 Deterministic Verification 通过；
    3. DeepSeek 输出可解析；
    4. DeepSeek Deterministic Verification 通过；
    5. 两个 Teacher 的关键结构签名一致；
    6. 两边 confidence 均达到阈值；
    7. 如果 assessment=keep，则 stop_assessment 必须为
       original_stop_correct；
    8. 没有 candidate_pool_insufficient / uncertain /
       needs_reconciliation。

即使 supervision_verified=true：

    - train split：
        training_eligible=true

    - validation / benchmark：
        training_eligible=false

避免验证集或基准集意外进入训练。

========================================================================
五、重要原则
========================================================================

本脚本不会保证“数学意义上的绝对零错误”。

它采取的是更保守的工程策略：

    有疑问
        => blocked
        => 不训练

宁可减少训练样本，也不让明显可疑标签污染训练。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from collections import Counter
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from tqdm import tqdm
except ImportError as exc:
    raise RuntimeError(
        "缺少 tqdm（进度条依赖）。请执行：python -m pip install -U tqdm"
    ) from exc


SCRIPT_PATH = Path(
    __file__
).resolve()

PROJECT_ROOT = (
    SCRIPT_PATH.parents[1]
)

SCRIPTS_DIR = (
    PROJECT_ROOT
    / "scripts"
)

for path in (
    PROJECT_ROOT,
    SCRIPTS_DIR,
):
    if str(path) not in sys.path:
        sys.path.insert(
            0,
            str(path),
        )


try:
    from refinement_core import (
        build_question,
        collect_candidate_evidence_stats,
        original_boundary,
        original_certificate,
        parse_refinement_proposal,
        refinement_summary,
        select_candidate_evidence_ids,
        verify_and_finalize_refinement,
    )

    from refinement_teacher import (
        REFINEMENT_SYSTEM_PROMPT,
        REQUIREMENT_DECISION_SYSTEM_PROMPT,
        WITNESS_SELECTION_SYSTEM_PROMPT,
        STRONG_REVIEW_SYSTEM_PROMPT,
        BigModelRefinementTeacherPool,
        BigModelTeacherConfig,
        DeepSeekRefinementTeacherPool,
        RefinementTeacherConfig,
        SiliconFlowRefinementTeacherPool,
        SiliconFlowTeacherConfig,
        bigmodel_teacher_protocol_metadata,
        build_refinement_user_prompt,
        parse_strong_review_decision,
        refinement_teacher_protocol_metadata,
        siliconflow_teacher_protocol_metadata,
        validate_bigmodel_environment,
        validate_refinement_environment,
        validate_siliconflow_environment,
    )

    from refinement_candidate_builder import (
        BuildEvidenceStore,
        CandidateBuilderConfig,
        build_refinement_candidates,
    )
except ImportError as exc:
    raise RuntimeError(
        "无法导入 refinement 数据处理模块。"
        "请确认以下文件存在："
        "scripts/refinement_core.py、"
        "scripts/refinement_teacher.py、"
        "scripts/refinement_candidate_builder.py。"
    ) from exc


RUNNER_VERSION = "1.9.2.1"

EXPECTED_DATASET_VERSION = (
    "2.10.0"
)

DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "unified_swe_dataset_v2_10"
)

DEFAULT_EVIDENCE_CACHE = (
    PROJECT_ROOT
    / "data"
    / ".train_cache"
    / "policy_evidence_v2_10.sqlite3"
)

DEFAULT_BUILD_DB = (
    PROJECT_ROOT
    / "data"
    / ".build"
    / "unified_swe_v1.sqlite3"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / ".supervision_refinement"
    / "v1_3_sample"
)

DEFAULT_SAMPLE_SEED = (
    20260810
)

# 小样本优先。
DEFAULT_TASK_LIMIT = 5

# v1.2 已经把 48 压到 24。
DEFAULT_CANDIDATE_LIMIT = 24

# Fail-fast gate（失败即停的上限检查）。
# 不静默截断 Prompt。
# Prompt Hard Budget（提示词硬预算）。
#
# 这是工程保护门，不是模型真实 context window（上下文窗口）。
# 在当前监督修正阶段，正确性优先于 API 成本，因此默认放宽到 110k chars。
DEFAULT_MAX_PROMPT_CHARS = 140_000

# Question Compaction Target（问题文本压缩目标）。
#
# 重要：
#   这个值不再是“Question 自身超过 30k 就压缩”的阈值。
#
#   v1.5.3 的流程是：
#
#       先用完整 Question 构造完整 Prompt
#           ↓
#       Prompt <= 110k
#           → 完整 Question 原样保留
#
#       Prompt > 110k
#           → 才把 Question 确定性压缩到最多约 30k
#
#       压缩后仍 > 110k
#           → 才删除最低优先级普通 Candidate
#
# Candidate Builder 始终使用完整 Question。
DEFAULT_MAX_TEACHER_QUESTION_CHARS = 30_000

DEFAULT_MIN_PRIMARY_CONFIDENCE = 0.95
DEFAULT_MIN_STRONG_CONFIDENCE = 0.85

DEFAULT_DEEPSEEK_CONCURRENCY = 8


# ============================================================================
# Dataset / Manifest
# ============================================================================


def resolve_manifest_path(
    dataset_dir: Path,
) -> Path:
    for path in (
        dataset_dir
        / "manifest_v2_10.json",
        dataset_dir
        / "manifest.json",
    ):
        if path.is_file():
            return path

    raise FileNotFoundError(
        "找不到 V2.10 manifest"
    )


def resolve_split_path(
    dataset_dir: Path,
    split: str,
) -> Path:
    for path in (
        dataset_dir
        / f"{split}_v2_10.parquet",
        dataset_dir
        / f"{split}.parquet",
    ):
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"找不到 split={split}"
    )


def load_and_validate_manifest(
    dataset_dir: Path,
) -> tuple[
    Path,
    dict[str, Any],
]:
    """
    监督修正只能以 audit passed 的冻结 V2.10 为输入。
    """

    path = resolve_manifest_path(
        dataset_dir
    )

    manifest = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    version = str(
        manifest.get(
            "dataset_version"
        )
        or ""
    )

    if version != EXPECTED_DATASET_VERSION:
        raise ValueError(
            "Refinement v1.3 只允许基于冻结 V2.10："
            f"actual={version!r}"
        )

    audit_status = str(
        manifest.get(
            "audit_status"
        )
        or ""
    )

    if audit_status != "passed":
        raise ValueError(
            "V2.10 manifest audit_status 必须为 passed"
        )

    return (
        path,
        manifest,
    )


def iter_task_rows(
    parquet_path: Path,
) -> Iterable[
    dict[str, Any]
]:
    """
    按 Parquet Row Group（行组）流式读取。

    不一次性加载整个 split。
    """

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "缺少 pyarrow。"
            "请执行：python -m pip install -U pyarrow"
        ) from exc

    parquet = pq.ParquetFile(
        parquet_path
    )

    for row_group_index in range(
        parquet.num_row_groups
    ):
        table = (
            parquet.read_row_group(
                row_group_index,
                columns=[
                    "task_id",
                    "input",
                    "supervision",
                ],
                use_threads=True,
            )
        )

        yield from (
            table.to_pylist()
        )


def _stable_task_key(
    task_id: str,
    seed: int,
) -> str:
    payload = (
        f"{seed}\0{task_id}"
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


def _has_state(
    supervision: Mapping[str, Any],
    state_type: str,
) -> bool:
    return any(
        str(
            state.get(
                "state_type"
            )
            or ""
        )
        == state_type
        for state in (
            supervision.get(
                "policy_states"
            )
            or []
        )
    )


def task_matches_scope(
    supervision: Mapping[str, Any],
    task_scope: str,
) -> bool:
    """
    判断任务是否属于当前 refinement scope（修正范围）。
    """

    has_complete = _has_state(
        supervision,
        "complete",
    )

    if task_scope == "all":
        # 全量监督修正只要求 Complete 存在。
        return has_complete

    if task_scope == "boundary":
        return (
            has_complete
            and _has_state(
                supervision,
                "decision_boundary",
            )
        )

    raise ValueError(
        f"未知 task_scope={task_scope!r}"
    )


def parquet_row_count(
    parquet_path: Path,
) -> int:
    """读取 Parquet metadata，用于 tqdm 显示准确总进度。"""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "缺少 pyarrow。"
            "请执行：python -m pip install -U pyarrow"
        ) from exc

    return int(
        pq.ParquetFile(
            parquet_path
        ).metadata.num_rows
    )


def select_tasks(
    *,
    split_path: Path,
    task_scope: str,
    seed: int,
    task_limit: int,
    allow_full_run: bool,
) -> tuple[
    list[dict[str, Any]],
    int,
]:
    """
    稳定选择任务。

    返回：
        selected_rows
        scope_task_count

    scope_task_count 很重要：
        它告诉我们当前 split 中一共有多少任务符合这个 scope，
        即使本次只抽 5 条。
    """

    candidates: list[
        tuple[
            str,
            dict[str, Any],
        ]
    ] = []

    scan_total = parquet_row_count(
        split_path
    )

    for row in tqdm(
        iter_task_rows(
            split_path
        ),
        total=scan_total,
        desc="Scan tasks（扫描任务）",
        unit="task",
        dynamic_ncols=True,
    ):
        task_id = str(
            row.get(
                "task_id"
            )
            or ""
        )

        if not task_id:
            raise ValueError(
                "Parquet row 缺少 task_id"
            )

        supervision = (
            row.get(
                "supervision"
            )
            or {}
        )

        if not task_matches_scope(
            supervision,
            task_scope,
        ):
            continue

        candidates.append(
            (
                _stable_task_key(
                    task_id,
                    seed,
                ),
                row,
            )
        )

    candidates.sort(
        key=lambda item: (
            item[0]
        )
    )

    scope_task_count = len(
        candidates
    )

    if allow_full_run:
        return (
            [
                row
                for (
                    _key,
                    row,
                )
                in candidates
            ],
            scope_task_count,
        )

    if task_limit < 1:
        raise ValueError(
            "--task-limit 必须 >= 1"
        )

    if scope_task_count < task_limit:
        raise ValueError(
            "可 refinement task 不足："
            f"scope={task_scope}, "
            f"available={scope_task_count}, "
            f"requested={task_limit}"
        )

    return (
        [
            row
            for (
                _key,
                row,
            )
            in candidates[
                :task_limit
            ]
        ],
        scope_task_count,
    )


# ============================================================================
# Evidence Cache
# ============================================================================


class EvidenceCache:
    """
    Refinement 专用只读 Evidence Cache（证据缓存）。

    最低字段：
        evidence_id
        path
        content

    推荐：
        unit_type
        symbol
        start_line
        end_line
        rendered_token_count
    """

    TABLE_CANDIDATES = (
        "evidence",
        "evidence_units",
        "policy_evidence",
    )

    def __init__(
        self,
        path: Path,
    ) -> None:
        self.path = path.resolve()

        if not self.path.is_file():
            raise FileNotFoundError(
                self.path
            )

        uri = (
            self.path.as_uri()
            + "?mode=ro"
        )

        self.connection = (
            sqlite3.connect(
                uri,
                uri=True,
            )
        )

        self.connection.row_factory = (
            sqlite3.Row
        )

        self.table_name = (
            self._resolve_table()
        )

        self.columns = (
            self._load_columns()
        )

        required = {
            "evidence_id",
            "path",
            "content",
        }

        missing = (
            required
            - self.columns
        )

        if missing:
            raise ValueError(
                "Evidence Cache 缺少必要字段："
                f"{sorted(missing)}"
            )

    def close(
        self,
    ) -> None:
        self.connection.close()

    def _resolve_table(
        self,
    ) -> str:
        tables = {
            str(row[0])
            for row in (
                self.connection
                .execute(
                    "SELECT name "
                    "FROM sqlite_master "
                    "WHERE type='table'"
                )
            )
        }

        for name in (
            self.TABLE_CANDIDATES
        ):
            if name in tables:
                return name

        raise ValueError(
            "找不到支持的 Evidence table："
            f"{sorted(tables)}"
        )

    def _load_columns(
        self,
    ) -> set[str]:
        return {
            str(row[1])
            for row in (
                self.connection
                .execute(
                    f"PRAGMA table_info({self.table_name})"
                )
            )
        }

    def existing_ids(
        self,
        evidence_ids: Sequence[str],
    ) -> set[str]:
        """
        只做 ID 存在性扫描，不加载代码正文。
        """

        ids = list(
            dict.fromkeys(
                map(
                    str,
                    evidence_ids,
                )
            )
        )

        sql = (
            f"SELECT evidence_id "
            f"FROM {self.table_name} "
            "WHERE evidence_id=?"
        )

        found: set[str] = set()

        for evidence_id in ids:
            row = (
                self.connection
                .execute(
                    sql,
                    (
                        evidence_id,
                    ),
                )
                .fetchone()
            )

            if row is not None:
                found.add(
                    evidence_id
                )

        return found

    def get_many(
        self,
        evidence_ids: Sequence[str],
    ) -> tuple[
        dict[str, dict[str, Any]],
        list[str],
    ]:
        ids = list(
            dict.fromkeys(
                map(
                    str,
                    evidence_ids,
                )
            )
        )

        optional = [
            name
            for name in (
                "file_version_id",
                "unit_type",
                "symbol",
                "start_line",
                "end_line",
                "rendered_token_count",
            )
            if name in self.columns
        ]

        columns = [
            "evidence_id",
            "path",
            "content",
            *optional,
        ]

        sql = (
            "SELECT "
            + ", ".join(
                columns
            )
            + f" FROM {self.table_name} "
            + "WHERE evidence_id=?"
        )

        found: dict[
            str,
            dict[str, Any],
        ] = {}

        missing: list[str] = []

        for evidence_id in ids:
            row = (
                self.connection
                .execute(
                    sql,
                    (
                        evidence_id,
                    ),
                )
                .fetchone()
            )

            if row is None:
                missing.append(
                    evidence_id
                )

                continue

            found[
                evidence_id
            ] = {
                key: row[key]
                for key in row.keys()
            }

        return (
            found,
            missing,
        )


# ============================================================================
# Prompt Preparation
# ============================================================================


# ============================================================================
# Teacher Question Compaction（教师问题文本压缩）
# ============================================================================


QUESTION_SIGNAL_PATTERNS = (
    # Markdown / code
    re.compile(r"`[^`]+`"),
    re.compile(
        r"\b(?:def|class|function|method|API|parameter|argument)\b",
        re.I,
    ),

    # Python traceback / exception
    re.compile(r"\bTraceback\b", re.I),
    re.compile(
        r"\bFile\s+[\"'][^\"']+[\"']\s*,\s*line\s+\d+",
        re.I,
    ),
    re.compile(
        r"\b(?:TypeError|ValueError|KeyError|IndexError|"
        r"AttributeError|AssertionError|RuntimeError|Exception)\b"
    ),

    # Requirement / expected behavior
    re.compile(
        r"\b(?:expected|actual|should|must|regression|fails?|failure|"
        r"bug|incorrect|correct|support|behavior|behaviour)\b",
        re.I,
    ),

    # Path / symbol-like text
    re.compile(
        r"(?:^|[\s`'\"(])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"
    ),
    re.compile(
        r"\b[A-Za-z_][A-Za-z0-9_]*\."
        r"[A-Za-z_][A-Za-z0-9_.]*\b"
    ),
    re.compile(
        r"\b[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+\b"
    ),
)


def _question_signal_score(
    line: str,
) -> int:
    """
    计算 Issue 中一行的“保留信号分”。

    这里只用于确定性文本裁剪，不判断监督标签。
    """
    text = str(
        line
        or ""
    )

    if not text.strip():
        return 0

    score = 0

    for pattern in QUESTION_SIGNAL_PATTERNS:
        if pattern.search(
            text
        ):
            score += 1

    stripped = (
        text.lstrip()
    )

    if stripped.startswith(
        (
            "def ",
            "class ",
            "async def ",
            "assert ",
            "raise ",
            "return ",
            "import ",
            "from ",
            ">>>",
            "File ",
        )
    ):
        score += 2

    return score


def compact_question_for_teacher(
    question: str,
    *,
    max_chars: int,
) -> tuple[
    str,
    dict[str, Any],
]:
    """
    Deterministic Question Compaction（确定性问题文本压缩）。

    核心原则：

        Candidate Builder:
            永远使用完整 Question。

        Teacher Prompt:
            超长时才压缩 Question 区域。

    这样既不降低候选检索召回，又能控制 API 输入成本。

    超预算时保留：
        - 开头约 45%
        - 结尾约 20%
        - 中间约 35% 的高信号行

    中间高信号包括：
        代码、路径、symbol、Traceback、异常、
        expected/actual/should/must 等约束。

    本函数不会生成摘要，也不会新增任何事实。
    """
    text = str(
        question
        or ""
    )

    original_chars = len(
        text
    )

    if max_chars < 4_000:
        raise ValueError(
            "max_teacher_question_chars 至少为 4000"
        )

    if original_chars <= max_chars:
        return (
            text,
            {
                "question_compacted": False,
                "question_original_chars": original_chars,
                "question_teacher_chars": original_chars,
                "question_removed_chars": 0,
                "question_selected_signal_line_count": 0,
            },
        )

    lines = text.splitlines(
        keepends=True
    )

    if not lines:
        compacted = text[
            :max_chars
        ]

        return (
            compacted,
            {
                "question_compacted": True,
                "question_original_chars": original_chars,
                "question_teacher_chars": len(
                    compacted
                ),
                "question_removed_chars": (
                    original_chars
                    - len(
                        compacted
                    )
                ),
                "question_selected_signal_line_count": 0,
            },
        )

    # 为省略标记留出安全空间。
    content_budget = max(
        1,
        max_chars - 512,
    )

    head_budget = int(
        content_budget
        * 0.45
    )

    tail_budget = int(
        content_budget
        * 0.20
    )

    signal_budget = (
        content_budget
        - head_budget
        - tail_budget
    )

    # ---------------------------------------------------------------
    # Head
    # ---------------------------------------------------------------

    head_indices: set[
        int
    ] = set()

    used = 0

    for index, line in enumerate(
        lines
    ):
        if (
            used + len(line)
            > head_budget
            and head_indices
        ):
            break

        head_indices.add(
            index
        )

        used += len(
            line
        )

        if used >= head_budget:
            break

    # ---------------------------------------------------------------
    # Tail
    # ---------------------------------------------------------------

    tail_indices: set[
        int
    ] = set()

    used = 0

    for index in range(
        len(lines) - 1,
        -1,
        -1,
    ):
        if index in head_indices:
            break

        line = lines[
            index
        ]

        if (
            used + len(line)
            > tail_budget
            and tail_indices
        ):
            break

        tail_indices.add(
            index
        )

        used += len(
            line
        )

        if used >= tail_budget:
            break

    # ---------------------------------------------------------------
    # Middle high-signal lines
    # ---------------------------------------------------------------

    middle_candidates: list[
        tuple[
            int,
            int,
            int,
        ]
    ] = []

    for index, line in enumerate(
        lines
    ):
        if (
            index in head_indices
            or index in tail_indices
        ):
            continue

        score = (
            _question_signal_score(
                line
            )
        )

        if score <= 0:
            continue

        middle_candidates.append(
            (
                -score,
                index,
                len(
                    line
                ),
            )
        )

    middle_candidates.sort()

    selected_middle: set[
        int
    ] = set()

    used = 0

    for (
        _negative_score,
        index,
        length,
    ) in middle_candidates:
        if (
            used + length
            > signal_budget
        ):
            continue

        selected_middle.add(
            index
        )

        used += length

        if used >= signal_budget:
            break

    selected_indices = sorted(
        head_indices
        | selected_middle
        | tail_indices
    )

    omission_marker = (
        "\n"
        "[... Issue 中间的低信号或重复内容因 Teacher 输入预算"
        "被确定性省略；Candidate 检索仍使用完整 Issue ...]"
        "\n"
    )

    output_parts: list[
        str
    ] = []

    previous_index: int | None = (
        None
    )

    for index in selected_indices:
        if (
            previous_index
            is not None
            and index
            > previous_index + 1
        ):
            output_parts.append(
                omission_marker
            )

        output_parts.append(
            lines[
                index
            ]
        )

        previous_index = (
            index
        )

    compacted = "".join(
        output_parts
    )

    # 多处分散 signal line 可能产生多个 marker。
    # 最终再做字符硬门。
    if len(compacted) > max_chars:
        compacted = (
            compacted[
                : max_chars - 160
            ]
            + "\n"
            "[... Teacher Question 已达到字符预算上限 ...]"
            "\n"
        )

    return (
        compacted,
        {
            "question_compacted": True,
            "question_original_chars": original_chars,
            "question_teacher_chars": len(
                compacted
            ),
            "question_removed_chars": (
                original_chars
                - len(
                    compacted
                )
            ),
            "question_selected_signal_line_count": len(
                selected_middle
            ),
        },
    )


def teacher_display_priority_key(
    record: Mapping[str, Any],
) -> tuple[Any, ...]:
    """
    Teacher Display Priority（教师候选展示优先级）。

    只改变展示顺序，不改变 Candidate Set。

    Candidate Builder 为了审计旧监督，会优先保留 Original Evidence；
    但 Teacher 展示不应该继承这种“旧监督优先”的视觉顺序。

    展示顺序：
        0. Issue explicit symbol
        1. Gold-guided pre-fix Evidence
        2. Issue explicit path
        3. policy:symbol
        4. policy:bm25_content
        5. policy:structure / structure_pair
        6. 普通 source
        7. test
        8. doc
        9. resource
       10. low_value / other

    Original Witness / Certificate / Boundary 身份完全不参与排序。
    """

    stats = record.get("candidate_stats") or {}
    sources = set(
        map(
            str,
            stats.get("sources") or [],
        )
    )
    category = str(
        stats.get("category") or "unknown"
    )

    if "issue_explicit_symbol" in sources:
        tier = 0
    elif (
        bool(stats.get("gold_guided"))
        or "gold_patch_hunk" in sources
        or "gold_patch_path" in sources
    ):
        tier = 1
    elif "issue_explicit_path" in sources:
        tier = 2
    elif "policy:symbol" in sources:
        tier = 3
    elif "policy:bm25_content" in sources:
        tier = 4
    elif (
        "policy:structure" in sources
        or "policy:structure_pair" in sources
    ):
        tier = 5
    elif category == "source":
        tier = 6
    elif category == "test":
        tier = 7
    elif category == "doc":
        tier = 8
    elif category == "resource":
        tier = 9
    else:
        tier = 10

    rank_value = stats.get("min_online_rank")
    rank = (
        int(rank_value)
        if isinstance(rank_value, int)
        else 2**31 - 1
    )

    start_line = int(
        record.get("start_line") or 0
    )
    end_line = int(
        record.get("end_line")
        or start_line
    )
    span = max(
        1,
        end_line - start_line + 1,
    )

    return (
        tier,
        rank,
        span,
        str(record.get("path") or ""),
        start_line,
        str(record.get("evidence_id") or ""),
    )


def prepare_task_payload(
    *,
    task_row: Mapping[str, Any],
    cache: EvidenceCache,
    build_store: BuildEvidenceStore,
    candidate_config: CandidateBuilderConfig,
    reference_mode: str,
    max_prompt_chars: int,
    max_teacher_question_chars: int,
) -> dict[str, Any]:
    """
    构造单任务 Teacher 输入（v1.4）。

    v1.3 直接从 policy candidate_actions 中取 Top-K；
    v1.4 改为专用 Candidate Builder：

        1. 旧 Certificate / Boundary / Witness 优先；
        2. cache 缺失旧 Witness 时，尝试从完整 build DB 恢复；
        3. Gold Patch 只用于在当前任务 pre-fix snapshot 中定位真实 Evidence；
        4. Issue 中明确给出的仓库路径也可补候选；
        5. Policy Candidate 只作为补充来源；
        6. docs/test/low-value 受配额限制；
        7. 同文件高度重叠 Evidence 去重。

    本函数不改变 V2.10 在线 Retriever，只改变离线监督修正阶段的 Teacher 输入。
    """

    task_id = str(task_row.get("task_id") or "")
    task_input = task_row.get("input") or {}
    supervision = task_row.get("supervision") or {}

    retrieval_scope = task_input.get("retrieval_scope") or {}
    snapshot_id = str(retrieval_scope.get("snapshot_id") or "")

    if not snapshot_id:
        raise ValueError(
            f"task={task_id} 缺少 input.retrieval_scope.snapshot_id"
        )

    stats_by_id = collect_candidate_evidence_stats(
        supervision
    )

    all_policy_candidate_ids = tuple(
        item.evidence_id
        for item in sorted(
            stats_by_id.values(),
            key=lambda item: item.ranking_key(),
        )
    )

    # training cache 不是完整 corpus，只是已经物化过的 Evidence。
    # 因此这里只把它当成快速来源，缺失项允许由完整 build DB 补充。
    cache_records, cache_missing_ids = cache.get_many(
        all_policy_candidate_ids
    )

    # 完整 Question：
    #   - Candidate Builder 永远使用它；
    #   - Teacher Prompt 也先使用它；
    #   - 只有完整 Prompt 超过 hard budget 时，才启动 Question Compaction。
    full_question = build_question(
        task_input
    )

    teacher_question = (
        full_question
    )

    question_compaction_diagnostics = {
        "question_compacted": False,
        "question_compaction_triggered_by_prompt_budget": False,
        "question_original_chars": len(
            full_question
        ),
        "question_teacher_chars": len(
            full_question
        ),
        "question_removed_chars": 0,
        "question_selected_signal_line_count": 0,
    }

    gold_patch = None
    test_patch = None

    if reference_mode == "gold":
        gold_patch = supervision.get("gold_patch") or None
        test_patch = supervision.get("test_patch") or None

    candidate_records, candidate_metadata = (
        build_refinement_candidates(
            supervision=supervision,
            question=full_question,
            snapshot_id=snapshot_id,
            policy_stats=stats_by_id,
            evidence_cache_records=cache_records,
            build_store=build_store,
            gold_patch=gold_patch,
            config=candidate_config,
        )
    )

    # ------------------------------------------------------------------
    # Teacher Presentation De-bias（教师展示去偏）
    #
    # Candidate 集合完全不变。
    # 只改变 Prompt 中的展示顺序。
    # ------------------------------------------------------------------

    builder_selected_order = [
        str(record["evidence_id"])
        for record in candidate_records
    ]

    candidate_records = sorted(
        candidate_records,
        key=teacher_display_priority_key,
    )

    teacher_display_order = [
        str(record["evidence_id"])
        for record in candidate_records
    ]

    teacher_display_reordered = (
        builder_selected_order
        != teacher_display_order
    )

    def is_prompt_protected(
        record: Mapping[str, Any],
    ) -> bool:
        """
        Prompt Budget 裁剪时必须保护：

            Original Witness / Certificate / Boundary
            Gold-guided Evidence
            Issue explicit symbol/path Evidence

        只允许删除低优先级普通 Policy Candidate。
        """
        stats = (
            record.get("candidate_stats")
            or {}
        )

        sources = set(
            map(
                str,
                stats.get("sources") or [],
            )
        )

        return bool(
            stats.get("in_original_witness")
            or stats.get("in_original_certificate")
            or stats.get("in_original_boundary")
            or stats.get("gold_guided")
            or "issue_explicit_symbol" in sources
            or "issue_explicit_path" in sources
        )

    def build_prompt(
        records: Sequence[Mapping[str, Any]],
    ) -> str:
        return build_refinement_user_prompt(
            task_id=task_id,
            question=teacher_question,
            original_obligations=(
                supervision.get("obligations")
                or []
            ),
            original_boundary_evidence_ids=(
                original_boundary(supervision)
            ),
            original_certificate_evidence_ids=(
                original_certificate(supervision)
            ),
            candidate_evidence=records,
            gold_patch=gold_patch,
            test_patch=test_patch,
        )

    prompt_budget_dropped_ids: list[str] = []

    # ---------------------------------------------------------------
    # 第一阶段：
    #     用完整 Question + 完整候选构造 Prompt。
    #
    #     只要没有超过 hard budget，就完全不压缩 Question。
    # ---------------------------------------------------------------

    user_prompt = build_prompt(
        candidate_records
    )

    prompt_chars_before_compaction = (
        len(REFINEMENT_SYSTEM_PROMPT)
        + len(user_prompt)
    )

    prompt_chars = (
        prompt_chars_before_compaction
    )

    # ---------------------------------------------------------------
    # 第二阶段：
    #     只有完整 Prompt 超过 hard budget，才压缩 Question。
    #
    #     Candidate Builder 已经完成，且使用的是 full_question，
    #     所以这一步不会影响候选检索召回。
    # ---------------------------------------------------------------

    if prompt_chars > max_prompt_chars:
        (
            teacher_question,
            question_compaction_diagnostics,
        ) = compact_question_for_teacher(
            full_question,
            max_chars=(
                max_teacher_question_chars
            ),
        )

        question_compaction_diagnostics = {
            **question_compaction_diagnostics,
            "question_compaction_triggered_by_prompt_budget": True,
        }

        user_prompt = build_prompt(
            candidate_records
        )

        prompt_chars = (
            len(REFINEMENT_SYSTEM_PROMPT)
            + len(user_prompt)
        )

    prompt_chars_after_question_compaction = (
        prompt_chars
    )

    # ---------------------------------------------------------------
    # 第三阶段：
    #     Question 压缩后仍然超过 hard budget，
    #     才允许删除最低优先级普通 Candidate。
    #
    #     Original / Gold / Issue explicit Evidence 永远受保护。
    # ---------------------------------------------------------------

    while prompt_chars > max_prompt_chars:
        drop_index = None

        # Candidate Builder 已按质量顺序排序；
        # 从尾部删最低优先级的普通 Candidate。
        for index in range(
            len(candidate_records) - 1,
            -1,
            -1,
        ):
            if not is_prompt_protected(
                candidate_records[index]
            ):
                drop_index = index
                break

        if drop_index is None:
            raise ValueError(
                "Prompt 超过 max_prompt_chars，"
                "且剩余 Candidate 全部属于保护 Evidence："
                f"task={task_id}, "
                f"chars={prompt_chars}, "
                f"limit={max_prompt_chars}"
            )

        dropped = candidate_records.pop(
            drop_index
        )

        prompt_budget_dropped_ids.append(
            str(dropped["evidence_id"])
        )

        user_prompt = build_prompt(
            candidate_records
        )

        prompt_chars = (
            len(REFINEMENT_SYSTEM_PROMPT)
            + len(user_prompt)
        )

    candidate_ids = [
        str(record["evidence_id"])
        for record in candidate_records
    ]

    if not candidate_ids:
        raise ValueError(
            f"task={task_id} 的 Teacher Candidate Pool 为空"
        )

    token_costs = {
        str(record["evidence_id"]): int(
            record.get("rendered_token_count")
            or 2**30
        )
        for record in candidate_records
    }

    # Prompt budget 裁剪后重新计算真实发送集合的统计。
    selected_source_counts: dict[str, int] = {}
    selected_category_counts: dict[str, int] = {}

    for record in candidate_records:
        stats = (
            record.get("candidate_stats")
            or {}
        )

        category = str(
            stats.get("category")
            or "unknown"
        )

        selected_category_counts[category] = (
            selected_category_counts.get(
                category,
                0,
            )
            + 1
        )

        for source in (
            stats.get("sources")
            or []
        ):
            source = str(source)
            selected_source_counts[source] = (
                selected_source_counts.get(
                    source,
                    0,
                )
                + 1
            )

    candidate_metadata = {
        **candidate_metadata,
        **question_compaction_diagnostics,

        # 完整 Prompt（尚未压缩 Question）的字符数。
        "prompt_chars_before_question_compaction": (
            prompt_chars_before_compaction
        ),

        # 如果触发 Question Compaction，这里记录压缩后的 Prompt；
        # 未触发时与 before 相同。
        "prompt_chars_after_question_compaction": (
            prompt_chars_after_question_compaction
        ),

        "prompt_hard_budget_chars": (
            max_prompt_chars
        ),

        "prompt_budget_dropped_count": len(
            prompt_budget_dropped_ids
        ),
        "prompt_budget_dropped_ids": (
            prompt_budget_dropped_ids
        ),
        "selected_candidate_count": len(
            candidate_records
        ),
        "builder_selected_evidence_ids_before_teacher_reorder": (
            builder_selected_order
        ),
        "teacher_display_evidence_ids": (
            candidate_ids
        ),
        "teacher_display_reordered": (
            teacher_display_reordered
        ),
        "selected_evidence_ids": candidate_ids,
        "selected_paths": [
            str(record.get("path") or "")
            for record in candidate_records
        ],
        "selected_category_counts": dict(
            sorted(
                selected_category_counts.items()
            )
        ),
        "selected_source_counts": dict(
            sorted(
                selected_source_counts.items()
            )
        ),
        "policy_candidate_count": len(
            all_policy_candidate_ids
        ),
        "policy_candidate_cache_present_count": len(
            cache_records
        ),
        "policy_candidate_cache_missing_count": len(
            cache_missing_ids
        ),
        "policy_candidate_cache_missing_ids_sample": sorted(
            map(str, cache_missing_ids)
        )[:20],
    }

    gold_used = bool(
        (
            gold_patch
            and str(gold_patch).strip()
        )
        or (
            test_patch
            and str(test_patch).strip()
        )
    )

    return {
        "task_id": task_id,
        "snapshot_id": snapshot_id,
        "supervision": supervision,
        "has_boundary": _has_state(
            supervision,
            "decision_boundary",
        ),
        "candidate_ids": candidate_ids,
        "candidate_records": candidate_records,
        "candidate_metadata": candidate_metadata,
        "token_costs": token_costs,
        "user_prompt": user_prompt,
        "prompt_chars": prompt_chars,
        "offline_gold_reference_used": gold_used,
    }


# ============================================================================
# Teacher Execution / Verification
# ============================================================================


# ============================================================================
# Teacher Proposal Evidence-ID Normalization
# ============================================================================


ALLOWED_REFINEMENT_DEFECT_CODES = {
    "wrong_witness",
    "missing_obligation",
    "unnecessary_obligation",
    "wrong_question_satisfied",
    "wrong_and_or",
    "wrong_certificate",
    "stop_too_early",
    "stop_too_late",
    "other_semantic_defect",
}


def _normalize_optional_symbol(
    value: Any,
) -> str | None:
    """
    Candidate symbol 的机械规范化。

    None / "" / "null" / "none"
        -> None

    其它值：
        strip 后原样保留。

    不做任何 symbol 猜测。
    """

    if value is None:
        return None

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            "candidate_ref.symbol 必须是 string 或 null"
        )

    text = value.strip()

    if text.lower() in {
        "",
        "null",
        "none",
    }:
        return None

    return text


def validate_refinement_defects(
    proposal: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Anti-over-refinement Gate（防过度修正门）。

    REFINE 必须证明“为什么原监督必须修改”。

    只接受有限 defect code，
    避免模型用“可以更详细”这种模糊理由强制重写全部数据。
    """

    raw = proposal.get(
        "refinement_defects"
    )

    if not isinstance(
        raw,
        list,
    ):
        raise ValueError(
            "refinement_defects 必须是 list"
        )

    normalized: list[
        dict[str, Any]
    ] = []

    allowed_types = {
        "fault_location",
        "fault_logic",
        "dependency_context",
        "state_flow",
        "behavior_constraint",
        "repair_scope",
        "validation_constraint",
    }

    for index, item in enumerate(
        raw
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                f"refinement_defects[{index}] 必须是 object"
            )

        code = str(
            item.get(
                "code"
            )
            or ""
        ).strip()

        if (
            code
            not in ALLOWED_REFINEMENT_DEFECT_CODES
        ):
            raise ValueError(
                f"非法 refinement defect code：{code!r}"
            )

        obligation_type = (
            item.get(
                "obligation_type"
            )
        )

        if obligation_type is not None:
            obligation_type = str(
                obligation_type
            ).strip()

            if obligation_type.lower() in {
                "",
                "null",
                "none",
            }:
                obligation_type = None

        if (
            obligation_type is not None
            and obligation_type
            not in allowed_types
        ):
            raise ValueError(
                "refinement_defects"
                f"[{index}].obligation_type 非法："
                f"{obligation_type!r}"
            )

        reason = str(
            item.get(
                "reason"
            )
            or ""
        ).strip()

        if not reason:
            raise ValueError(
                "refinement_defects"
                f"[{index}].reason 不能为空"
            )

        normalized.append(
            {
                "code": code,
                "obligation_type": (
                    obligation_type
                ),
                "reason": reason,
            }
        )

    assessment = str(
        proposal.get(
            "assessment"
        )
        or ""
    ).strip()

    if (
        assessment == "refine"
        and not normalized
    ):
        raise ValueError(
            "assessment=refine 必须提供至少一个 "
            "concrete refinement_defect"
        )

    if (
        assessment == "keep"
        and normalized
    ):
        raise ValueError(
            "assessment=keep 时 refinement_defects 必须为空"
        )

    return normalized


def bind_teacher_candidate_references(
    *,
    proposal: Mapping[str, Any],
    candidate_records: Sequence[
        Mapping[str, Any]
    ],
    original_certificate_ids: Sequence[str],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """
    Evidence Binding Gate（证据绑定门）。

    v1.8.3 后 Teacher 不再输出 Evidence ID。

    Teacher 只输出：
        candidate_number
        path
        symbol

    程序根据“当前 Prompt 中真实 Candidate 顺序”机械映射：

        candidate_number
            -> exact Candidate record
            -> exact Evidence ID

    并验证：
        path 完全一致
        symbol 完全一致

    这能从设计上消除：
        ev_14
        假 Evidence ID
        把 A 的 ID 绑定到 B 的 path/symbol

    注意：
        这仍然不是 Semantic Judge（语义评审）。
        如果模型正确绑定了 Candidate 19=localize_to_utc，
        但错误认为 localize_to_utc 足以支持 bishop88，
        这种“语义选择错误”仍需要后续抽样/校准发现。

    本函数只保证：
        Teacher 引用的 Candidate 身份是精确、不可混淆的。
    """

    normalized = json.loads(
        json.dumps(
            proposal,
            ensure_ascii=False,
        )
    )

    defects = (
        validate_refinement_defects(
            normalized
        )
    )

    normalized[
        "refinement_defects"
    ] = defects

    if not candidate_records:
        raise ValueError(
            "Candidate Pool 为空，无法做 Candidate Binding"
        )

    ordinal_map: dict[
        int,
        Mapping[str, Any],
    ] = {
        index: record
        for index, record in enumerate(
            candidate_records,
            start=1,
        )
    }

    events: list[
        dict[str, Any]
    ] = []

    def bind_ref(
        raw_ref: Any,
        *,
        field_path: str,
    ) -> str:
        if not isinstance(
            raw_ref,
            dict,
        ):
            raise ValueError(
                f"{field_path} 必须是 object"
            )

        number = raw_ref.get(
            "candidate_number"
        )

        if (
            isinstance(
                number,
                bool,
            )
            or not isinstance(
                number,
                int,
            )
        ):
            raise ValueError(
                f"{field_path}.candidate_number 必须是整数"
            )

        record = ordinal_map.get(
            number
        )

        if record is None:
            raise ValueError(
                f"{field_path}.candidate_number 越界："
                f"{number}，当前 Candidate 数={len(candidate_records)}"
            )

        actual_path = str(
            record.get(
                "path"
            )
            or ""
        )

        proposed_path = raw_ref.get(
            "path"
        )

        if not isinstance(
            proposed_path,
            str,
        ):
            raise ValueError(
                f"{field_path}.path 必须是 string"
            )

        proposed_path = (
            proposed_path.strip()
        )

        if proposed_path != actual_path:
            raise ValueError(
                "Candidate Binding path 不一致："
                f"{field_path}, "
                f"candidate_number={number}, "
                f"teacher={proposed_path!r}, "
                f"actual={actual_path!r}"
            )

        actual_symbol = (
            str(
                record.get(
                    "symbol"
                )
                or ""
            ).strip()
            or None
        )

        proposed_symbol = (
            _normalize_optional_symbol(
                raw_ref.get(
                    "symbol"
                )
            )
        )

        if actual_symbol is None:
            # Candidate metadata 本身没有 symbol。
            #
            # 此时不能因为 Teacher 从正文推断出 symbol 就误杀样本。
            # candidate_number + path 已经唯一绑定当前 Candidate。
            if proposed_symbol is not None:
                events.append(
                    {
                        "field": field_path + ".symbol",
                        "normalization": (
                            "candidate_metadata_symbol_missing"
                        ),
                        "teacher_symbol": proposed_symbol,
                        "actual_symbol": None,
                    }
                )

            raw_ref["symbol"] = None
            proposed_symbol = None

        elif proposed_symbol is None:
            # Candidate Number + path 已经唯一绑定 Candidate。
            # Teacher 没抄 symbol 只是字段省略，不等于声明了错误 symbol。
            events.append(
                {
                    "field": (
                        field_path
                        + ".symbol"
                    ),
                    "normalization": (
                        "teacher_symbol_omitted"
                    ),
                    "teacher_symbol": None,
                    "actual_symbol": (
                        actual_symbol
                    ),
                    "action": (
                        "fill_actual_candidate_symbol"
                    ),
                }
            )

            raw_ref[
                "symbol"
            ] = actual_symbol
            proposed_symbol = actual_symbol

        elif proposed_symbol == actual_symbol:
            pass

        elif (
            proposed_symbol is not None
            and (
                proposed_symbol.startswith(
                    actual_symbol + "."
                )
                or actual_symbol.startswith(
                    proposed_symbol + "."
                )
            )
        ):
            # 允许 class <-> method 粒度兼容，例如：
            # actual=Rule_L026
            # teacher=Rule_L026._lint_references_and_aliases
            events.append(
                {
                    "field": field_path + ".symbol",
                    "normalization": (
                        "symbol_hierarchy_compatible"
                    ),
                    "teacher_symbol": proposed_symbol,
                    "actual_symbol": actual_symbol,
                }
            )

            raw_ref["symbol"] = actual_symbol
            proposed_symbol = actual_symbol

        else:
            # 真正的 symbol 冲突继续硬拦。
            raise ValueError(
                "Candidate Binding symbol 真实冲突："
                f"{field_path}, "
                f"candidate_number={number}, "
                f"teacher={proposed_symbol!r}, "
                f"actual={actual_symbol!r}"
            )

        evidence_id = str(
            record.get(
                "evidence_id"
            )
            or ""
        )

        if not evidence_id:
            raise ValueError(
                f"Candidate {number} 缺少 evidence_id"
            )

        events.append(
            {
                "field": field_path,
                "candidate_number": (
                    number
                ),
                "path": actual_path,
                "symbol": (
                    actual_symbol
                ),
                "resolved_evidence_id": (
                    evidence_id
                ),
            }
        )

        return evidence_id

    assessment = str(
        normalized.get(
            "assessment"
        )
        or ""
    ).strip()

    raw_obligations = (
        normalized.get(
            "refined_obligations"
        )
    )

    if not isinstance(
        raw_obligations,
        list,
    ):
        raise ValueError(
            "refined_obligations 必须是 list"
        )

    # KEEP / candidate_pool_insufficient / uncertain 都不会采用
    # Teacher 输出的新 obligation graph。
    # 若模型重复输出 refined_obligations，只属于结构噪声。
    if (
        assessment
        in {
            "keep",
            "candidate_pool_insufficient",
            "uncertain",
        }
        and raw_obligations
    ):
        events.append(
            {
                "field": (
                    "refined_obligations"
                ),
                "normalization": (
                    "non_refine_obligations_cleared"
                ),
                "assessment": assessment,
                "original_obligation_count": (
                    len(raw_obligations)
                ),
            }
        )

        normalized[
            "refined_obligations"
        ] = []
        raw_obligations = []

    for obligation_index, obligation in enumerate(
        raw_obligations
    ):
        if not isinstance(
            obligation,
            dict,
        ):
            raise ValueError(
                "refined_obligations 元素必须是 object"
            )

        groups = obligation.get(
            "witness_groups"
        )

        if not isinstance(
            groups,
            list,
        ):
            raise ValueError(
                "witness_groups 必须是 list"
            )

        # satisfied_by_question=true 时 retrieval_required=false。
        # Teacher 偶尔会输出一个 candidate_refs=[] 的空 group；
        # 这是纯结构噪声，可以安全规范化成 witness_groups=[]。
        if bool(
            obligation.get(
                "satisfied_by_question"
            )
        ):
            if groups:
                events.append(
                    {
                        "field": (
                            "refined_obligations"
                            f"[{obligation_index}].witness_groups"
                        ),
                        "normalization": (
                            "question_satisfied_witness_groups_cleared"
                        ),
                        "original_group_count": len(groups),
                    }
                )

            obligation["witness_groups"] = []
            groups = []

        for group_index, group in enumerate(
            groups
        ):
            if not isinstance(
                group,
                dict,
            ):
                raise ValueError(
                    "witness_group 必须是 object"
                )

            refs = group.get(
                "candidate_refs"
            )

            if not isinstance(
                refs,
                list,
            ):
                raise ValueError(
                    "v1.8.3 witness_group 必须提供 candidate_refs list"
                )

            evidence_ids = [
                bind_ref(
                    ref,
                    field_path=(
                        "refined_obligations"
                        f"[{obligation_index}]"
                        ".witness_groups"
                        f"[{group_index}]"
                        ".candidate_refs"
                        f"[{ref_index}]"
                    ),
                )
                for ref_index, ref in enumerate(
                    refs
                )
            ]

            if not evidence_ids:
                # satisfied_by_question=true 时 Core 会清空 group，
                # 但 Teacher Schema 已要求此时 witness_groups=[]。
                raise ValueError(
                    "candidate_refs 不能为空"
                )

            # Core v1.7 继续消费 evidence_ids。
            # Evidence ID 完全由程序生成，不再由 Teacher 输出。
            group[
                "evidence_ids"
            ] = list(
                dict.fromkeys(
                    evidence_ids
                )
            )

    raw_certificate_numbers = (
        normalized.get(
            "proposed_certificate_candidate_numbers"
        )
    )

    if not isinstance(
        raw_certificate_numbers,
        list,
    ):
        raise ValueError(
            "proposed_certificate_candidate_numbers 必须是 list"
        )

    if assessment == "keep":
        if raw_certificate_numbers:
            # KEEP 本身已经明确“保留原监督”。
            # Teacher 再抄一遍 Candidate Number 不增加任何语义；
            # 安全忽略并继续使用 Original Complete Certificate。
            events.append(
                {
                    "field": (
                        "proposed_certificate_candidate_numbers"
                    ),
                    "normalization": (
                        "keep_certificate_numbers_ignored"
                    ),
                    "teacher_values": list(
                        raw_certificate_numbers
                    ),
                }
            )
            normalized[
                "proposed_certificate_candidate_numbers"
            ] = []

        # KEEP 的 Certificate 由程序保留旧 Complete，
        # 避免让模型再次手工映射旧 Evidence ID。
        normalized[
            "proposed_certificate_evidence_ids"
        ] = list(
            map(
                str,
                original_certificate_ids,
            )
        )

    elif assessment in {
        "candidate_pool_insufficient",
        "uncertain",
    }:
        if raw_certificate_numbers:
            events.append(
                {
                    "field": (
                        "proposed_certificate_candidate_numbers"
                    ),
                    "normalization": (
                        "non_certificate_assessment_numbers_cleared"
                    ),
                    "assessment": assessment,
                    "teacher_values": list(
                        raw_certificate_numbers
                    ),
                }
            )
            normalized[
                "proposed_certificate_candidate_numbers"
            ] = []

        normalized[
            "proposed_certificate_evidence_ids"
        ] = []

    else:
        certificate_ids: list[
            str
        ] = []

        seen_numbers: set[
            int
        ] = set()

        for index, value in enumerate(
            raw_certificate_numbers
        ):
            if (
                isinstance(
                    value,
                    bool,
                )
                or not isinstance(
                    value,
                    int,
                )
            ):
                raise ValueError(
                    "proposed_certificate_candidate_numbers"
                    f"[{index}] 必须是整数"
                )

            if value in seen_numbers:
                continue

            seen_numbers.add(
                value
            )

            record = ordinal_map.get(
                value
            )

            if record is None:
                raise ValueError(
                    "proposed_certificate_candidate_numbers"
                    f"[{index}] 越界：{value}"
                )

            evidence_id = str(
                record.get(
                    "evidence_id"
                )
                or ""
            )

            if not evidence_id:
                raise ValueError(
                    f"Candidate {value} 缺少 evidence_id"
                )

            certificate_ids.append(
                evidence_id
            )

            events.append(
                {
                    "field": (
                        "proposed_certificate_candidate_numbers"
                        f"[{index}]"
                    ),
                    "candidate_number": (
                        value
                    ),
                    "path": str(
                        record.get(
                            "path"
                        )
                        or ""
                    ),
                    "symbol": (
                        str(
                            record.get(
                                "symbol"
                            )
                            or ""
                        ).strip()
                        or None
                    ),
                    "resolved_evidence_id": (
                        evidence_id
                    ),
                }
            )

        normalized[
            "proposed_certificate_evidence_ids"
        ] = certificate_ids

    return (
        normalized,
        events,
    )


def normalize_candidate_evidence_id_aliases(
    *,
    proposal: Mapping[str, Any],
    candidate_ids: Sequence[str],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    """
    只修复一种“完全可机械判定”的 Evidence ID 格式错误：

        Prompt 展示：
            [CANDIDATE 14] id=ev_e56fb3ef...

        Teacher 错写：
            ev_14

    若：
        - "ev_14" 本身不是真实 Candidate ID；
        - 14 在当前 Candidate ordinal（候选展示序号）范围内；

    则确定性映射为第 14 条 Candidate 的真实完整 ID。

    安全边界：
        - 已经是真实 Candidate ID：绝不改；
        - 不是严格 ev_<正整数>：绝不猜；
        - 序号越界：绝不改，交给 Verifier 报错；
        - 不修改自然语言 rationale；
        - 不修改 obligation / AND/OR / certificate 语义。

    返回：
        normalized_proposal
        normalization_events
    """

    normalized = json.loads(
        json.dumps(
            proposal,
            ensure_ascii=False,
        )
    )

    candidate_ids = [
        str(
            evidence_id
        )
        for evidence_id in candidate_ids
    ]

    candidate_set = set(
        candidate_ids
    )

    ordinal_map = {
        f"ev_{index}": evidence_id
        for index, evidence_id in enumerate(
            candidate_ids,
            start=1,
        )
    }

    events: list[
        dict[str, Any]
    ] = []

    def normalize_id(
        value: Any,
        *,
        field_path: str,
    ) -> Any:
        if not isinstance(
            value,
            str,
        ):
            return value

        text = value.strip()

        # 真实 ID 优先，防止极端情况下真实 Evidence 真叫 ev_14。
        if text in candidate_set:
            return text

        replacement = (
            ordinal_map.get(
                text
            )
        )

        if replacement is None:
            return text

        events.append(
            {
                "field": field_path,
                "old_value": text,
                "new_value": (
                    replacement
                ),
                "reason": (
                    "normalized display ordinal alias "
                    "to exact Candidate Evidence ID"
                ),
            }
        )

        return replacement

    certificate = normalized.get(
        "proposed_certificate_evidence_ids"
    )

    if isinstance(
        certificate,
        list,
    ):
        normalized[
            "proposed_certificate_evidence_ids"
        ] = [
            normalize_id(
                value,
                field_path=(
                    "proposed_certificate_evidence_ids"
                    f"[{index}]"
                ),
            )
            for index, value in enumerate(
                certificate
            )
        ]

    obligations = normalized.get(
        "refined_obligations"
    )

    if isinstance(
        obligations,
        list,
    ):
        for obligation_index, obligation in enumerate(
            obligations
        ):
            if not isinstance(
                obligation,
                dict,
            ):
                continue

            groups = obligation.get(
                "witness_groups"
            )

            if not isinstance(
                groups,
                list,
            ):
                continue

            for group_index, group in enumerate(
                groups
            ):
                if not isinstance(
                    group,
                    dict,
                ):
                    continue

                evidence_ids = group.get(
                    "evidence_ids"
                )

                if not isinstance(
                    evidence_ids,
                    list,
                ):
                    continue

                group[
                    "evidence_ids"
                ] = [
                    normalize_id(
                        value,
                        field_path=(
                            "refined_obligations"
                            f"[{obligation_index}]"
                            ".witness_groups"
                            f"[{group_index}]"
                            ".evidence_ids"
                            f"[{evidence_index}]"
                        ),
                    )
                    for evidence_index, value in enumerate(
                        evidence_ids
                    )
                ]

    return (
        normalized,
        events,
    )


def clean_verified_obligation_for_teacher(
    obligation: Mapping[str, Any],
) -> dict[str, Any]:
    """
    把 Verifier 的内部 obligation 结构还原成 Teacher 输出 Schema。

    不把：
        group_id / annotation_ids / construction_method 等
    内部审计字段再塞回 LLM。
    """

    return {
        "source_obligation_id": (
            obligation.get(
                "source_obligation_id"
            )
        ),
        "type": str(
            obligation.get(
                "type"
            )
            or ""
        ),
        "description": str(
            obligation.get(
                "description"
            )
            or ""
        ),
        "applicable": bool(
            obligation.get(
                "applicable"
            )
        ),
        "required_for_sufficiency": bool(
            obligation.get(
                "required_for_sufficiency"
            )
        ),
        "satisfied_by_question": bool(
            obligation.get(
                "satisfied_by_question"
            )
        ),
        "witness_groups": [
            {
                "evidence_ids": list(
                    map(
                        str,
                        (
                            group.get(
                                "evidence_ids"
                            )
                            or []
                        ),
                    )
                ),
                "reason": str(
                    group.get(
                        "reason"
                    )
                    or ""
                ),
            }
            for group in (
                obligation.get(
                    "witness_groups"
                )
                or []
            )
        ],
    }


def canonical_primary_proposal_for_review(
    primary_result: Mapping[str, Any],
) -> dict[str, Any] | None:
    """
    构造 DeepSeek 真正要审核的 Primary Canonical Proposal。

    如果 Primary 已被 Verifier accepted：
        - Certificate 使用 verified minimal certificate；
        - obligation 使用 Verifier 规范化后的结构；
        - 保留 assessment / stop / confidence / rationale。

    如果 Primary 尚未 accepted：
        返回原 proposal，DeepSeek 仍可针对失败提案做纠正。
    """

    proposal = (
        primary_result.get(
            "proposal"
        )
        or {}
    )

    if not proposal:
        return None

    verification = (
        primary_result.get(
            "verification"
        )
        or {}
    )

    if (
        str(
            verification.get(
                "verification_status"
            )
            or ""
        )
        != "accepted"
    ):
        return json.loads(
            json.dumps(
                proposal,
                ensure_ascii=False,
            )
        )

    assessment = str(
        verification.get(
            "assessment"
        )
        or proposal.get(
            "assessment"
        )
        or ""
    )

    canonical = {
        "assessment": assessment,
        "stop_assessment": str(
            verification.get(
                "stop_assessment"
            )
            or proposal.get(
                "stop_assessment"
            )
            or ""
        ),
        "confidence": (
            proposal.get(
                "confidence"
            )
        ),
        "rationale": str(
            proposal.get(
                "rationale"
            )
            or ""
        ),
        "missing_candidate_requests": (
            proposal.get(
                "missing_candidate_requests"
            )
            or []
        ),
        "refined_obligations": [],
        "proposed_certificate_evidence_ids": list(
            map(
                str,
                (
                    verification.get(
                        "verified_minimal_certificate_evidence_ids"
                    )
                    or proposal.get(
                        "proposed_certificate_evidence_ids"
                    )
                    or []
                ),
            )
        ),
    }

    if assessment == "refine":
        canonical[
            "refined_obligations"
        ] = [
            clean_verified_obligation_for_teacher(
                obligation
            )
            for obligation in (
                verification.get(
                    "refined_obligations"
                )
                or []
            )
        ]

    return canonical


def build_strong_review_user_prompt(
    *,
    base_user_prompt: str,
    primary_result: Mapping[str, Any],
) -> str:
    """
    Strong Reviewer 输入。

    DeepSeek 看得到：
        原任务上下文
        + Primary Canonical Proposal
        + Verifier 摘要

    但只输出 review verdict，
    不再输出第二份 supervision proposal。
    """

    canonical = (
        canonical_primary_proposal_for_review(
            primary_result
        )
    )

    verification = (
        primary_result.get(
            "verification"
        )
        or {}
    )

    verification_summary = {
        "verification_status": (
            verification.get(
                "verification_status"
            )
        ),
        "assessment": (
            verification.get(
                "assessment"
            )
        ),
        "stop_assessment": (
            verification.get(
                "stop_assessment"
            )
        ),
        "verified_minimal_certificate_evidence_ids": (
            verification.get(
                "verified_minimal_certificate_evidence_ids"
            )
        ),
        "certificate_normalized_from_teacher_graph": (
            verification.get(
                "certificate_normalized_from_teacher_graph"
            )
        ),
        "certificate_normalization_reason": (
            verification.get(
                "certificate_normalization_reason"
            )
        ),
    }

    review_context = {
        "primary_canonical_proposal": (
            canonical
        ),
        "primary_verification_summary": (
            verification_summary
        ),
        "primary_verification_exception": bool(
            primary_result.get(
                "verification_exception"
            )
        ),
        "primary_error": (
            primary_result.get(
                "error"
            )
        ),
    }

    return (
        str(
            base_user_prompt
        )
        + "\n\n"
        + "[STRONG REVIEW CONTEXT]\n"
        + "你现在只做 Strong Semantic Review（强语义审核）。\n"
        + "不要重新输出 refinement proposal。\n"
        + "请按六项 checks 输出 approve / reject / uncertain。\n"
        + "特别检查 satisfied_by_question："
          "Issue 已明确的 expected behavior 不应被错误要求再次检索。\n"
        + json.dumps(
            review_context,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


# ============================================================================
# v1.9 Fixed Decision Consensus（固定决策共识）
# ============================================================================

FIXED_DECISION_SLOT_TYPES = (
    "fault_location",
    "fault_logic",
    "dependency_context",
    "state_flow",
    "behavior_constraint",
    "repair_scope",
    "validation_constraint",
)

FIXED_DECISION_VALUES = {
    "keep_original",
    "use_candidates",
    "question_satisfied",
    "not_required",
    "missing_pre_fix_context",
    "uncertain",
}

DEFAULT_SLOT_DESCRIPTIONS = {
    "fault_location": "定位修复前故障代码或新增功能的现有集成位置",
    "fault_logic": "理解修复前已经存在的故障机制或错误逻辑",
    "dependency_context": "理解必要的依赖、调用、配置或外部组件上下文",
    "state_flow": "理解必要的状态生产、传播、缓存、失效或更新关系",
    "behavior_constraint": "理解问题描述明确要求的预期行为或输入输出约束",
    "repair_scope": "理解修复影响的现有模块、接口、导出点或注册点",
    "validation_constraint": "理解问题描述明确给出的验证、兼容或边界约束",
}


def _json_object_from_teacher_text(text: str) -> dict[str, Any]:
    """解析 Teacher JSON；仅容忍外层 Markdown code fence。"""
    if not isinstance(text, str):
        raise ValueError("Teacher output 必须是 string")

    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()

    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Teacher JSON 顶层必须是 object")
    return parsed


def _original_obligations_by_type(
    supervision: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """固定槽位要求一个 type 最多对应一个原 obligation；重复时直接阻断。"""
    result: dict[str, Mapping[str, Any]] = {}
    for obligation in (supervision.get("obligations") or []):
        if not isinstance(obligation, dict):
            continue
        otype = str(obligation.get("type") or "").strip()
        if not otype:
            continue
        if otype not in FIXED_DECISION_SLOT_TYPES:
            raise ValueError(f"原监督存在 v1.9 未知 obligation type：{otype!r}")
        if otype in result:
            raise ValueError(f"原监督存在重复 obligation type，固定槽位无法安全映射：{otype}")
        result[otype] = obligation
    return result


def _canonical_candidate_groups(
    raw_groups: Any,
    *,
    candidate_count: int,
    field_name: str,
) -> list[list[int]]:
    """规范化 OR-of-AND Candidate Number：组内去重排序，组间去重排序。"""
    if not isinstance(raw_groups, list):
        raise ValueError(f"{field_name} 必须是 list")

    normalized: list[tuple[int, ...]] = []
    for group_index, group in enumerate(raw_groups):
        if not isinstance(group, list):
            raise ValueError(f"{field_name}[{group_index}] 必须是 list")
        values: list[int] = []
        for value_index, value in enumerate(group):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"{field_name}[{group_index}][{value_index}] 必须是整数 Candidate Number"
                )
            if not (1 <= value <= candidate_count):
                raise ValueError(
                    f"{field_name}[{group_index}][{value_index}] Candidate Number 越界："
                    f"{value}; 当前 Candidate 数={candidate_count}"
                )
            values.append(value)
        if not values:
            raise ValueError(f"{field_name}[{group_index}] 不能为空")
        normalized.append(tuple(sorted(set(values))))

    return [list(group) for group in sorted(set(normalized))]


def parse_fixed_decision_table(
    *,
    text: str,
    candidate_count: int,
    supervision: Mapping[str, Any],
) -> dict[str, Any]:
    """严格解析 v1.9 Fixed Decision Table。"""
    raw = _json_object_from_teacher_text(text)
    extra_top = set(raw) - {"confidence", "slots"}
    if extra_top:
        raise ValueError(f"Fixed Decision JSON 存在禁止的顶层字段：{sorted(extra_top)}")

    confidence = raw.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence 必须是 number")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence 必须在 [0,1]")

    slots = raw.get("slots")
    if not isinstance(slots, dict):
        raise ValueError("slots 必须是 object")
    if set(slots) != set(FIXED_DECISION_SLOT_TYPES):
        missing = sorted(set(FIXED_DECISION_SLOT_TYPES) - set(slots))
        extra = sorted(set(slots) - set(FIXED_DECISION_SLOT_TYPES))
        raise ValueError(f"slots 必须恰好包含 7 个固定类型；missing={missing}, extra={extra}")

    original_by_type = _original_obligations_by_type(supervision)
    normalized_slots: dict[str, dict[str, Any]] = {}

    for slot_type in FIXED_DECISION_SLOT_TYPES:
        raw_slot = slots[slot_type]
        if not isinstance(raw_slot, dict):
            raise ValueError(f"slots.{slot_type} 必须是 object")
        extra_fields = set(raw_slot) - {
            "decision", "witness_groups", "missing_context", "reason"
        }
        if extra_fields:
            raise ValueError(f"slots.{slot_type} 存在禁止字段：{sorted(extra_fields)}")

        decision = str(raw_slot.get("decision") or "").strip()
        if decision not in FIXED_DECISION_VALUES:
            raise ValueError(f"slots.{slot_type}.decision 非法：{decision!r}")
        reason = str(raw_slot.get("reason") or "").strip()
        if not reason:
            raise ValueError(f"slots.{slot_type}.reason 不能为空")

        raw_groups = raw_slot.get("witness_groups")
        if raw_groups is None:
            raw_groups = []

        if decision == "use_candidates":
            groups = _canonical_candidate_groups(
                raw_groups,
                candidate_count=candidate_count,
                field_name=f"slots.{slot_type}.witness_groups",
            )
            if not groups:
                raise ValueError(f"slots.{slot_type}=use_candidates 要求非空 witness_groups")
        else:
            if not isinstance(raw_groups, list) or raw_groups:
                raise ValueError(
                    f"slots.{slot_type}={decision} 时 witness_groups 必须为空 list"
                )
            groups = []

        if decision == "keep_original" and slot_type not in original_by_type:
            raise ValueError(
                f"slots.{slot_type}=keep_original，但 Original Supervision 没有该 type"
            )

        missing_context = raw_slot.get("missing_context")
        if decision == "missing_pre_fix_context":
            if not isinstance(missing_context, dict):
                raise ValueError(
                    f"slots.{slot_type}=missing_pre_fix_context 要求 missing_context object"
                )
            extra_missing = set(missing_context) - {
                "path_hint", "symbol_hint", "keywords", "reason"
            }
            if extra_missing:
                raise ValueError(
                    f"slots.{slot_type}.missing_context 存在禁止字段：{sorted(extra_missing)}"
                )
            keywords = missing_context.get("keywords")
            if not isinstance(keywords, list):
                raise ValueError(
                    f"slots.{slot_type}.missing_context.keywords 必须是 list"
                )
            normalized_missing = {
                "path_hint": str(missing_context.get("path_hint") or "").strip(),
                "symbol_hint": str(missing_context.get("symbol_hint") or "").strip(),
                "keywords": [
                    str(value).strip() for value in keywords if str(value).strip()
                ],
                "reason": str(missing_context.get("reason") or "").strip(),
            }
            if not normalized_missing["reason"]:
                raise ValueError(
                    f"slots.{slot_type}.missing_context.reason 不能为空"
                )
        else:
            if missing_context not in {None, ""}:
                raise ValueError(
                    f"slots.{slot_type}={decision} 时 missing_context 必须是 null"
                )
            normalized_missing = None

        normalized_slots[slot_type] = {
            "decision": decision,
            "witness_groups": groups,
            "missing_context": normalized_missing,
            "reason": reason,
        }

    return {"confidence": confidence, "slots": normalized_slots}


def fixed_decision_signature(decision_table: Mapping[str, Any]) -> str:
    """共识只比较结构：slot decision + Candidate OR-of-AND；忽略解释文字与置信度。"""
    slots = decision_table.get("slots") or {}
    signature = {
        slot_type: {
            "decision": slots[slot_type]["decision"],
            "witness_groups": (
                slots[slot_type]["witness_groups"]
                if slots[slot_type]["decision"] == "use_candidates"
                else []
            ),
        }
        for slot_type in FIXED_DECISION_SLOT_TYPES
    }
    return json.dumps(
        signature,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )



def _parse_fixed_slot_vote(
    *,
    slot_type: str,
    raw_slot: Any,
    candidate_count: int,
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """
    解析一个固定语义槽位。

    v1.9 的问题：
        一个 slot 违反 Schema，会让整次 A/B/C 调用的 7 个 slot 全部失效。

    v1.9.1：
        一个 slot 出错，只废掉这个 slot 的一票；
        其它合法 slot 继续参加共识。
    """

    if not isinstance(
        raw_slot,
        dict,
    ):
        return (
            None,
            {
                "slot_type": slot_type,
                "error_type": "SlotNotObject",
                "error": (
                    f"slots.{slot_type} 必须是 object"
                ),
            },
        )

    allowed_fields = {
        "decision",
        "witness_groups",
        "missing_context",
        "reason",
    }

    extra_fields = (
        set(
            raw_slot
        )
        - allowed_fields
    )

    if extra_fields:
        return (
            None,
            {
                "slot_type": slot_type,
                "error_type": "ExtraSlotFields",
                "error": (
                    f"slots.{slot_type} 存在禁止字段："
                    f"{sorted(extra_fields)}"
                ),
            },
        )

    decision = str(
        raw_slot.get(
            "decision"
        )
        or ""
    ).strip()

    if (
        decision
        not in FIXED_DECISION_VALUES
    ):
        return (
            None,
            {
                "slot_type": slot_type,
                "error_type": "InvalidDecision",
                "error": (
                    f"slots.{slot_type}.decision 非法："
                    f"{decision!r}"
                ),
            },
        )

    reason = str(
        raw_slot.get(
            "reason"
        )
        or ""
    ).strip()

    if not reason:
        return (
            None,
            {
                "slot_type": slot_type,
                "error_type": "MissingReason",
                "error": (
                    f"slots.{slot_type}.reason 不能为空"
                ),
            },
        )

    raw_groups = (
        raw_slot.get(
            "witness_groups"
        )
    )

    if raw_groups is None:
        raw_groups = []

    if (
        decision
        == "use_candidates"
    ):
        try:
            groups = (
                _canonical_candidate_groups(
                    raw_groups,
                    candidate_count=(
                        candidate_count
                    ),
                    field_name=(
                        f"slots.{slot_type}.witness_groups"
                    ),
                )
            )
        except Exception as exc:
            return (
                None,
                {
                    "slot_type": slot_type,
                    "error_type": (
                        type(
                            exc
                        ).__name__
                    ),
                    "error": str(
                        exc
                    ),
                },
            )

        if not groups:
            return (
                None,
                {
                    "slot_type": slot_type,
                    "error_type": "EmptyWitnessGroups",
                    "error": (
                        f"slots.{slot_type}=use_candidates "
                        "要求非空 witness_groups"
                    ),
                },
            )

    else:
        if (
            not isinstance(
                raw_groups,
                list,
            )
            or raw_groups
        ):
            return (
                None,
                {
                    "slot_type": slot_type,
                    "error_type": "UnexpectedWitnessGroups",
                    "error": (
                        f"slots.{slot_type}={decision} 时 "
                        "witness_groups 必须为空"
                    ),
                },
            )

        groups = []

    missing_context = (
        raw_slot.get(
            "missing_context"
        )
    )

    if (
        decision
        == "missing_pre_fix_context"
    ):
        if not isinstance(
            missing_context,
            dict,
        ):
            return (
                None,
                {
                    "slot_type": slot_type,
                    "error_type": "MissingContextObject",
                    "error": (
                        f"slots.{slot_type}=missing_pre_fix_context "
                        "要求 missing_context object"
                    ),
                },
            )

        keywords = (
            missing_context.get(
                "keywords"
            )
        )

        if not isinstance(
            keywords,
            list,
        ):
            return (
                None,
                {
                    "slot_type": slot_type,
                    "error_type": "InvalidMissingKeywords",
                    "error": (
                        f"slots.{slot_type}.missing_context.keywords "
                        "必须是 list"
                    ),
                },
            )

        normalized_missing = {
            "path_hint": str(
                missing_context.get(
                    "path_hint"
                )
                or ""
            ).strip(),
            "symbol_hint": str(
                missing_context.get(
                    "symbol_hint"
                )
                or ""
            ).strip(),
            "keywords": [
                str(
                    value
                ).strip()
                for value in (
                    keywords
                )
                if str(
                    value
                ).strip()
            ],
            "reason": str(
                missing_context.get(
                    "reason"
                )
                or ""
            ).strip(),
        }

        if not normalized_missing[
            "reason"
        ]:
            return (
                None,
                {
                    "slot_type": slot_type,
                    "error_type": "MissingContextReason",
                    "error": (
                        f"slots.{slot_type}.missing_context.reason "
                        "不能为空"
                    ),
                },
            )

    else:
        if (
            missing_context
            not in {
                None,
                "",
            }
        ):
            return (
                None,
                {
                    "slot_type": slot_type,
                    "error_type": "UnexpectedMissingContext",
                    "error": (
                        f"slots.{slot_type}={decision} 时 "
                        "missing_context 必须为 null"
                    ),
                },
            )

        normalized_missing = None

    return (
        {
            "decision": decision,
            "witness_groups": (
                groups
            ),
            "missing_context": (
                normalized_missing
            ),
            "reason": reason,
        },
        None,
    )


def parse_fixed_decision_table_slot_local(
    *,
    text: str,
    candidate_count: int,
) -> dict[str, Any]:
    """
    JSON 顶层必须合法；
    7 个 slot 分别校验。

    parse_success 表示 JSON + 顶层结构成功，
    不再要求每一个 slot 都成功。
    """

    raw = (
        _json_object_from_teacher_text(
            text
        )
    )

    extra_top = (
        set(
            raw
        )
        - {
            "confidence",
            "slots",
        }
    )

    if extra_top:
        raise ValueError(
            "Fixed Decision JSON 存在禁止顶层字段："
            f"{sorted(extra_top)}"
        )

    confidence = (
        raw.get(
            "confidence"
        )
    )

    if (
        isinstance(
            confidence,
            bool,
        )
        or not isinstance(
            confidence,
            (
                int,
                float,
            ),
        )
    ):
        raise ValueError(
            "confidence 必须是 number"
        )

    confidence = float(
        confidence
    )

    if not (
        0.0
        <= confidence
        <= 1.0
    ):
        raise ValueError(
            "confidence 必须在 [0,1]"
        )

    raw_slots = (
        raw.get(
            "slots"
        )
    )

    if not isinstance(
        raw_slots,
        dict,
    ):
        raise ValueError(
            "slots 必须是 object"
        )

    extra_slots = (
        set(
            raw_slots
        )
        - set(
            FIXED_DECISION_SLOT_TYPES
        )
    )

    if extra_slots:
        raise ValueError(
            "slots 存在未知固定类型："
            f"{sorted(extra_slots)}"
        )

    slots: dict[
        str,
        dict[str, Any] | None,
    ] = {}

    slot_errors: list[
        dict[str, Any]
    ] = []

    for slot_type in (
        FIXED_DECISION_SLOT_TYPES
    ):
        if (
            slot_type
            not in raw_slots
        ):
            slots[
                slot_type
            ] = None

            slot_errors.append(
                {
                    "slot_type": slot_type,
                    "error_type": "MissingSlot",
                    "error": (
                        f"缺少 slots.{slot_type}"
                    ),
                }
            )

            continue

        (
            value,
            error,
        ) = (
            _parse_fixed_slot_vote(
                slot_type=(
                    slot_type
                ),
                raw_slot=(
                    raw_slots[
                        slot_type
                    ]
                ),
                candidate_count=(
                    candidate_count
                ),
            )
        )

        slots[
            slot_type
        ] = value

        if (
            error
            is not None
        ):
            slot_errors.append(
                error
            )

    return {
        "confidence": confidence,
        "slots": slots,
        "slot_errors": (
            slot_errors
        ),
        "valid_slot_count": sum(
            value is not None
            for value in (
                slots.values()
            )
        ),
    }


def _original_effective_slot_vote(
    *,
    slot_type: str,
    supervision: Mapping[str, Any],
    candidate_records: Sequence[
        Mapping[str, Any]
    ],
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """
    将 keep_original 展开成 Original Slot 的实际语义。

    关键点：
    keep_original 不是独立的语义类别。

    如果 Original Slot 是 required：
        -> use_candidates + Original Witness

    如果 Original Slot 是 optional：
        -> not_required

    因此：
        keep_original
    可以与：
        use_candidates + 完全相同的 Original Witness
    形成共识。
    """

    original_by_type = (
        _original_obligations_by_type(
            supervision
        )
    )

    original = (
        original_by_type.get(
            slot_type
        )
    )

    if (
        original
        is None
    ):
        return (
            None,
            {
                "slot_type": slot_type,
                "error_type": (
                    "KeepOriginalWithoutOriginalSlot"
                ),
                "error": (
                    f"{slot_type}=keep_original，"
                    "但 Original Supervision 不存在该 slot"
                ),
            },
        )

    required = bool(
        original.get(
            "required_for_sufficiency",
            original.get(
                "mandatory",
                False,
            ),
        )
    )

    if not required:
        return (
            {
                "decision": (
                    "not_required"
                ),
                "witness_groups": [],
            },
            None,
        )

    id_to_number = {
        str(
            record.get(
                "evidence_id"
            )
            or ""
        ): index
        for index, record in enumerate(
            candidate_records,
            start=1,
        )
        if str(
            record.get(
                "evidence_id"
            )
            or ""
        )
    }

    groups: list[
        tuple[int, ...]
    ] = []

    for group in (
        original.get(
            "witness_groups"
        )
        or []
    ):
        evidence_ids = (
            group.get(
                "evidence_ids"
            )
            if isinstance(
                group,
                dict,
            )
            else group
        )

        if not isinstance(
            evidence_ids,
            list,
        ):
            continue

        numbers: list[
            int
        ] = []

        for evidence_id in (
            evidence_ids
        ):
            number = (
                id_to_number.get(
                    str(
                        evidence_id
                    )
                )
            )

            if (
                number
                is None
            ):
                return (
                    None,
                    {
                        "slot_type": slot_type,
                        "error_type": (
                            "OriginalWitnessMissingFromCandidatePool"
                        ),
                        "error": (
                            f"Original {slot_type} Witness "
                            f"{evidence_id} 不在 Candidate Pool"
                        ),
                    },
                )

            numbers.append(
                number
            )

        if numbers:
            groups.append(
                tuple(
                    sorted(
                        set(
                            numbers
                        )
                    )
                )
            )

    groups = list(
        sorted(
            set(
                groups
            )
        )
    )

    if not groups:
        return (
            None,
            {
                "slot_type": slot_type,
                "error_type": (
                    "RequiredOriginalSlotWithoutWitness"
                ),
                "error": (
                    f"Original {slot_type} required "
                    "但没有可映射 Witness"
                ),
            },
        )

    return (
        {
            "decision": (
                "use_candidates"
            ),
            "witness_groups": [
                list(
                    group
                )
                for group in (
                    groups
                )
            ],
        },
        None,
    )


def normalize_run_effective_votes(
    *,
    parsed_table: Mapping[str, Any],
    supervision: Mapping[str, Any],
    candidate_records: Sequence[
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    """
    一次 A/B/C 调用 -> 7 个 Effective Slot Votes（有效槽位投票）。
    """

    output_slots: dict[
        str,
        dict[str, Any] | None,
    ] = {}

    errors = list(
        parsed_table.get(
            "slot_errors"
        )
        or []
    )

    parsed_slots = (
        parsed_table.get(
            "slots"
        )
        or {}
    )

    for slot_type in (
        FIXED_DECISION_SLOT_TYPES
    ):
        slot = (
            parsed_slots.get(
                slot_type
            )
        )

        if not isinstance(
            slot,
            dict,
        ):
            output_slots[
                slot_type
            ] = None
            continue

        if (
            slot.get(
                "decision"
            )
            == "keep_original"
        ):
            try:
                (
                    effective,
                    error,
                ) = (
                    _original_effective_slot_vote(
                        slot_type=(
                            slot_type
                        ),
                        supervision=(
                            supervision
                        ),
                        candidate_records=(
                            candidate_records
                        ),
                    )
                )
            except Exception as exc:
                effective = None
                error = {
                    "slot_type": (
                        slot_type
                    ),
                    "error_type": (
                        type(
                            exc
                        ).__name__
                    ),
                    "error": str(
                        exc
                    ),
                }

            output_slots[
                slot_type
            ] = (
                effective
            )

            if (
                error
                is not None
            ):
                errors.append(
                    error
                )

            continue

        output_slots[
            slot_type
        ] = {
            "decision": (
                slot.get(
                    "decision"
                )
            ),
            "witness_groups": (
                slot.get(
                    "witness_groups"
                )
                or []
            ),
            "missing_context": (
                slot.get(
                    "missing_context"
                )
            ),
        }

    return {
        "confidence": float(
            parsed_table.get(
                "confidence"
            )
            or 0.0
        ),
        "slots": (
            output_slots
        ),
        "slot_errors": (
            errors
        ),
        "valid_slot_count": sum(
            value is not None
            for value in (
                output_slots.values()
            )
        ),
    }


def effective_slot_signature(
    slot: Mapping[str, Any],
) -> str:
    """
    单槽位有效结构签名。
    """
    return json.dumps(
        {
            "decision": (
                slot.get(
                    "decision"
                )
            ),
            "witness_groups": (
                slot.get(
                    "witness_groups"
                )
                or []
                if (
                    slot.get(
                        "decision"
                    )
                    == "use_candidates"
                )
                else []
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )


def _single_slot_consensus(
    *,
    slot_type: str,
    runs: Mapping[
        str,
        Mapping[str, Any],
    ],
) -> dict[str, Any]:
    """
    槽位级共识：

    1. decision 至少 2 票；
    2. 如果 decision=use_candidates：
       每个 AND-group 也至少需要 2 个 agreeing runs 支持。

    多个通过 2 票的 AND-group 共同形成 OR alternatives。
    """

    votes = []

    for (
        label,
        run,
    ) in (
        runs.items()
    ):
        table = (
            run.get(
                "effective_decision_table"
            )
            or {}
        )

        slot = (
            (
                table.get(
                    "slots"
                )
                or {}
            ).get(
                slot_type
            )
        )

        if not isinstance(
            slot,
            dict,
        ):
            continue

        votes.append(
            (
                label,
                slot,
                float(
                    table.get(
                        "confidence"
                    )
                    or 0.0
                ),
            )
        )

    decision_counts = Counter(
        str(
            slot.get(
                "decision"
            )
        )
        for (
            _label,
            slot,
            _confidence,
        ) in (
            votes
        )
    )

    if (
        not decision_counts
    ):
        return {
            "status": (
                "no_consensus"
            ),
            "reason": (
                "no_valid_votes"
            ),
            "decision": None,
            "witness_groups": [],
            "valid_vote_count": 0,
            "decision_vote_counts": {},
            "slot_confidence": None,
        }

    ranked = sorted(
        decision_counts.items(),
        key=lambda item: (
            -item[
                1
            ],
            item[
                0
            ],
        ),
    )

    decision = (
        ranked[
            0
        ][
            0
        ]
    )

    count = (
        ranked[
            0
        ][
            1
        ]
    )

    if (
        count
        < 2
    ):
        return {
            "status": (
                "no_consensus"
            ),
            "reason": (
                "no_2of3_decision_majority"
            ),
            "decision": None,
            "witness_groups": [],
            "valid_vote_count": (
                len(
                    votes
                )
            ),
            "decision_vote_counts": dict(
                sorted(
                    decision_counts.items()
                )
            ),
            "slot_confidence": None,
        }

    agreeing = [
        (
            label,
            slot,
            confidence,
        )
        for (
            label,
            slot,
            confidence,
        ) in (
            votes
        )
        if str(
            slot.get(
                "decision"
            )
        )
        == decision
    ]

    slot_confidence = min(
        confidence
        for (
            _label,
            _slot,
            confidence,
        ) in (
            agreeing
        )
    )

    supporting_runs = sorted(
        label
        for (
            label,
            _slot,
            _confidence,
        ) in (
            agreeing
        )
    )

    if (
        decision
        != "use_candidates"
    ):
        missing_context = None

        if (
            decision
            == "missing_pre_fix_context"
        ):
            # 解释文本不参与一致性，只用于诊断。
            best = max(
                agreeing,
                key=lambda item: (
                    item[
                        2
                    ],
                    item[
                        0
                    ],
                ),
            )

            missing_context = (
                best[
                    1
                ].get(
                    "missing_context"
                )
            )

        return {
            "status": (
                "agreed"
            ),
            "reason": (
                "2of3_decision_majority"
            ),
            "decision": (
                decision
            ),
            "witness_groups": [],
            "missing_context": (
                missing_context
            ),
            "valid_vote_count": (
                len(
                    votes
                )
            ),
            "decision_vote_counts": dict(
                sorted(
                    decision_counts.items()
                )
            ),
            "supporting_runs": (
                supporting_runs
            ),
            "slot_confidence": (
                slot_confidence
            ),
        }

    group_counts = Counter()
    group_runs: dict[
        tuple[int, ...],
        list[str],
    ] = {}

    for (
        label,
        slot,
        _confidence,
    ) in (
        agreeing
    ):
        # 同一 run 对同一 AND-group 最多计 1 票。
        groups = {
            tuple(
                group
            )
            for group in (
                slot.get(
                    "witness_groups"
                )
                or []
            )
        }

        for group in (
            groups
        ):
            group_counts[
                group
            ] += 1

            group_runs.setdefault(
                group,
                [],
            ).append(
                label
            )

    accepted_groups = [
        group
        for (
            group,
            group_count,
        ) in sorted(
            group_counts.items()
        )
        if (
            group_count
            >= 2
        )
    ]

    if (
        not accepted_groups
    ):
        return {
            "status": (
                "no_consensus"
            ),
            "reason": (
                "decision_majority_but_no_2vote_witness_group"
            ),
            "decision": (
                decision
            ),
            "witness_groups": [],
            "valid_vote_count": (
                len(
                    votes
                )
            ),
            "decision_vote_counts": dict(
                sorted(
                    decision_counts.items()
                )
            ),
            "group_vote_counts": {
                ",".join(
                    map(
                        str,
                        group,
                    )
                ): group_count
                for (
                    group,
                    group_count,
                ) in sorted(
                    group_counts.items()
                )
            },
            "supporting_runs": (
                supporting_runs
            ),
            "slot_confidence": (
                slot_confidence
            ),
        }

    return {
        "status": (
            "agreed"
        ),
        "reason": (
            "2of3_decision_plus_group_vote"
        ),
        "decision": (
            decision
        ),
        "witness_groups": [
            list(
                group
            )
            for group in (
                accepted_groups
            )
        ],
        "missing_context": None,
        "valid_vote_count": (
            len(
                votes
            )
        ),
        "decision_vote_counts": dict(
            sorted(
                decision_counts.items()
            )
        ),
        "group_vote_counts": {
            ",".join(
                map(
                    str,
                    group,
                )
            ): group_count
            for (
                group,
                group_count,
            ) in sorted(
                group_counts.items()
            )
        },
        "group_supporting_runs": {
            ",".join(
                map(
                    str,
                    group,
                )
            ): sorted(
                group_runs[
                    group
                ]
            )
            for group in (
                accepted_groups
            )
        },
        "supporting_runs": (
            supporting_runs
        ),
        "slot_confidence": (
            slot_confidence
        ),
    }


def build_slot_level_consensus(
    *,
    run_a: Mapping[str, Any],
    run_b: Mapping[str, Any],
    run_c: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    7 个槽位各自形成共识。

    只有 7/7 agreed，整个 task 才 agreed。
    """

    runs: dict[
        str,
        Mapping[str, Any],
    ] = {
        "A": run_a,
        "B": run_b,
    }

    if (
        run_c
        is not None
    ):
        runs[
            "C"
        ] = (
            run_c
        )

    slot_results = {
        slot_type: (
            _single_slot_consensus(
                slot_type=(
                    slot_type
                ),
                runs=(
                    runs
                ),
            )
        )
        for slot_type in (
            FIXED_DECISION_SLOT_TYPES
        )
    }

    agreed_slots = [
        slot_type
        for (
            slot_type,
            result,
        ) in (
            slot_results.items()
        )
        if (
            result.get(
                "status"
            )
            == "agreed"
        )
    ]

    if (
        len(
            agreed_slots
        )
        != len(
            FIXED_DECISION_SLOT_TYPES
        )
    ):
        return {
            "status": (
                "no_consensus"
            ),
            "consensus_mode": (
                "slot_level_2of3_decision_plus_group_vote"
            ),
            "tie_break_used": (
                run_c
                is not None
            ),
            "agreed_slot_count": (
                len(
                    agreed_slots
                )
            ),
            "agreed_slots": (
                agreed_slots
            ),
            "slot_results": (
                slot_results
            ),
            "decision_table": None,
            "consensus_confidence": None,
            "independent_semantic_review": False,
        }

    task_confidence = min(
        float(
            slot_results[
                slot_type
            ].get(
                "slot_confidence"
            )
            or 0.0
        )
        for slot_type in (
            FIXED_DECISION_SLOT_TYPES
        )
    )

    decision_slots = {}

    for slot_type in (
        FIXED_DECISION_SLOT_TYPES
    ):
        result = (
            slot_results[
                slot_type
            ]
        )

        decision_slots[
            slot_type
        ] = {
            "decision": (
                result[
                    "decision"
                ]
            ),
            "witness_groups": (
                result.get(
                    "witness_groups"
                )
                or []
            ),
            "missing_context": (
                result.get(
                    "missing_context"
                )
            ),
            "reason": (
                "program-built from v1.9.1 slot consensus"
            ),
        }

    return {
        "status": (
            "agreed"
        ),
        "consensus_mode": (
            "slot_level_2of3_decision_plus_group_vote"
        ),
        "tie_break_used": (
            run_c
            is not None
        ),
        "agreed_slot_count": (
            len(
                FIXED_DECISION_SLOT_TYPES
            )
        ),
        "agreed_slots": list(
            FIXED_DECISION_SLOT_TYPES
        ),
        "slot_results": (
            slot_results
        ),
        "decision_table": {
            "confidence": (
                task_confidence
            ),
            "slots": (
                decision_slots
            ),
        },
        "consensus_confidence": (
            task_confidence
        ),
        "independent_semantic_review": False,
    }


def _a_b_need_slot_tie_break(
    *,
    run_a: Mapping[str, Any],
    run_b: Mapping[str, Any],
) -> bool:
    """
    如果 A/B 任一槽位没有完整有效结构一致，则调用 C。
    """

    slots_a = (
        (
            run_a.get(
                "effective_decision_table"
            )
            or {}
        ).get(
            "slots"
        )
        or {}
    )

    slots_b = (
        (
            run_b.get(
                "effective_decision_table"
            )
            or {}
        ).get(
            "slots"
        )
        or {}
    )

    for slot_type in (
        FIXED_DECISION_SLOT_TYPES
    ):
        a = (
            slots_a.get(
                slot_type
            )
        )

        b = (
            slots_b.get(
                slot_type
            )
        )

        if (
            not isinstance(
                a,
                dict,
            )
            or not isinstance(
                b,
                dict,
            )
        ):
            return True

        if (
            effective_slot_signature(
                a
            )
            != effective_slot_signature(
                b
            )
        ):
            return True

    return False



# ============================================================================
# v1.9.2 Two-Stage Consensus（两阶段共识）
# ============================================================================

REQUIREMENT_DECISIONS = {
    "repository_required",
    "question_satisfied",
    "not_required",
    "uncertain",
}

WITNESS_STATUSES = {
    "select",
    "insufficient",
    "uncertain",
}


def _base_prompt_without_output(
    user_prompt: str,
) -> str:
    """删除旧 [OUTPUT] 合同，保留完整任务/候选/Gold 提示。"""
    marker = "\n[OUTPUT]\n"
    if marker in user_prompt:
        return user_prompt.split(marker, 1)[0].rstrip()
    return user_prompt.rstrip()



def build_requirement_stage_user_prompt(
    item: Mapping[str, Any],
) -> str:
    """
    v1.9.2.1 Stage 1 Candidate-blind（候选盲化）。

    Requirement Decision（证据需求决策）只应该回答：
        “这个语义是否需要 repository context？”

    它不应该因为“当前 Candidate Pool 里刚好有什么”
    而改变 required / not_required / question_satisfied 判断。

    所以这里机械移除：
        [CANDIDATE EVIDENCE POOL]

    但保留：
        Issue / Question
        Original Supervision
        Gold Change Hints（离线教师参考）

    Candidate 可用性全部由 Stage 2 的：
        select / insufficient / uncertain
    负责。
    """

    base = _base_prompt_without_output(
        str(
            item[
                "user_prompt"
            ]
        )
    )

    candidate_marker = (
        "\n[CANDIDATE EVIDENCE POOL]\n"
    )

    gold_marker = (
        "\n[GOLD CHANGE HINTS - OFFLINE ONLY, NOT EVIDENCE]\n"
    )

    if candidate_marker in base:
        prefix, tail = base.split(
            candidate_marker,
            1,
        )

        if gold_marker in (
            "\n"
            + tail
        ):
            # tail 开头通常不是换行，因此统一补一个换行搜索。
            padded_tail = (
                "\n"
                + tail
            )

            gold_index = (
                padded_tail.index(
                    gold_marker
                )
            )

            gold_section = (
                padded_tail[
                    gold_index + 1:
                ]
            )

            base = (
                prefix.rstrip()
                + "\n\n"
                + gold_section.strip()
            )

        else:
            base = (
                prefix.rstrip()
            )

    return (
        base
        + "\n\n"
        + "[STAGE 1 INPUT POLICY]\n"
        + "Candidate Pool is intentionally hidden in Stage 1. "
        + "Candidate availability is evaluated only in Stage 2.\n\n"
        + "[STAGE]\n"
        + "Requirement Decision（证据需求决策）\n\n"
        + "[OUTPUT]\n"
        + "只输出 Requirement Decision JSON。"
        + "禁止 Candidate Number、Evidence ID、witness_groups、"
        + "missing_context、assessment、STOP、Certificate。"
    )

def build_witness_stage_user_prompt(
    *,
    item: Mapping[str, Any],
    slot_type: str,
    requirement_result: Mapping[str, Any],
) -> str:
    """Stage 2：只针对一个 repository_required slot 选择 Witness。"""
    return (
        _base_prompt_without_output(
            str(item["user_prompt"])
        )
        + "\n\n[STAGE]\n"
        + "Targeted Witness Selection（定向支撑证据选择）\n\n"
        + "[TARGET SLOT]\n"
        + slot_type
        + "\n\n[STAGE 1 CONSENSUS]\n"
        + json.dumps(
            requirement_result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n\n[OUTPUT]\n"
        + "只为 TARGET SLOT 输出 Witness Selection JSON；"
        + "只能引用 Candidate Number。"
    )


def parse_requirement_decision_output(
    *,
    text: str,
) -> dict[str, Any]:
    """
    Stage 1 JSON + slot-local schema。

    一个槽位错误只损失该槽位的一票，不污染同次调用的其它槽位。
    """
    raw = _json_object_from_teacher_text(text)

    extra_top = set(raw) - {
        "confidence",
        "slots",
    }
    if extra_top:
        raise ValueError(
            "Requirement JSON 存在禁止顶层字段："
            f"{sorted(extra_top)}"
        )

    confidence = raw.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
    ):
        raise ValueError("confidence 必须是 number")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence 必须在 [0,1]")

    raw_slots = raw.get("slots")
    if not isinstance(raw_slots, dict):
        raise ValueError("slots 必须是 object")

    extra_slots = (
        set(raw_slots)
        - set(FIXED_DECISION_SLOT_TYPES)
    )
    if extra_slots:
        raise ValueError(
            "slots 存在未知类型："
            f"{sorted(extra_slots)}"
        )

    slots: dict[
        str,
        dict[str, Any] | None,
    ] = {}
    slot_errors: list[
        dict[str, Any]
    ] = []

    for slot_type in FIXED_DECISION_SLOT_TYPES:
        raw_slot = raw_slots.get(slot_type)

        if not isinstance(raw_slot, dict):
            slots[slot_type] = None
            slot_errors.append({
                "slot_type": slot_type,
                "error_type": "MissingOrInvalidSlot",
                "error": (
                    f"slots.{slot_type} 缺失或不是 object"
                ),
            })
            continue

        extra_fields = (
            set(raw_slot)
            - {
                "decision",
                "reason",
            }
        )
        if extra_fields:
            slots[slot_type] = None
            slot_errors.append({
                "slot_type": slot_type,
                "error_type": "ExtraRequirementFields",
                "error": (
                    f"slots.{slot_type} 存在禁止字段："
                    f"{sorted(extra_fields)}"
                ),
            })
            continue

        decision = str(
            raw_slot.get("decision")
            or ""
        ).strip()

        reason = str(
            raw_slot.get("reason")
            or ""
        ).strip()

        if decision not in REQUIREMENT_DECISIONS:
            slots[slot_type] = None
            slot_errors.append({
                "slot_type": slot_type,
                "error_type": "InvalidRequirementDecision",
                "error": (
                    f"slots.{slot_type}.decision 非法："
                    f"{decision!r}"
                ),
            })
            continue

        if not reason:
            slots[slot_type] = None
            slot_errors.append({
                "slot_type": slot_type,
                "error_type": "MissingRequirementReason",
                "error": (
                    f"slots.{slot_type}.reason 不能为空"
                ),
            })
            continue

        slots[slot_type] = {
            "decision": decision,
            "reason": reason,
        }

    return {
        "confidence": confidence,
        "slots": slots,
        "slot_errors": slot_errors,
        "valid_slot_count": sum(
            value is not None
            for value in slots.values()
        ),
    }


def run_one_requirement_decision(
    *,
    teacher: Any,
    item: Mapping[str, Any],
    run_label: str,
) -> dict[str, Any]:
    """单次 Stage 1 调用。"""
    task_id = str(item["task_id"])
    user_prompt = build_requirement_stage_user_prompt(item)

    result: dict[str, Any] = {
        "task_id": task_id,
        "run_label": run_label,
        "stage": "requirement_decision",
        "api_success": False,
        "parse_success": False,
        "decision_table": None,
        "teacher_call": None,
        "raw_teacher_output": None,
        "prompt_chars": (
            len(REQUIREMENT_DECISION_SYSTEM_PROMPT)
            + len(user_prompt)
        ),
        "prompt_sha256": hashlib.sha256(
            (
                REQUIREMENT_DECISION_SYSTEM_PROMPT
                + "\n"
                + user_prompt
            ).encode("utf-8")
        ).hexdigest(),
        "error": None,
    }

    try:
        call = teacher.call(
            user_prompt=user_prompt,
            system_prompt=(
                REQUIREMENT_DECISION_SYSTEM_PROMPT
            ),
            offline_gold_reference_used=bool(
                item[
                    "offline_gold_reference_used"
                ]
            ),
        )
        result["api_success"] = True
        result["teacher_call"] = (
            call.metadata.to_dict()
        )
        result["raw_teacher_output"] = (
            call.output_text
        )
    except Exception as exc:
        result["error"] = {
            "stage": "requirement_api_call",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return result

    try:
        result["decision_table"] = (
            parse_requirement_decision_output(
                text=str(
                    result[
                        "raw_teacher_output"
                    ]
                )
            )
        )
        result["parse_success"] = True
    except Exception as exc:
        result["error"] = {
            "stage": "parse_requirement_decision",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    return result


def execute_requirement_batch(
    *,
    teacher: Any,
    items: Sequence[
        Mapping[str, Any]
    ],
    run_label: str,
    max_workers: int,
) -> dict[
    str,
    dict[str, Any],
]:
    """Stage 1 批量执行。"""
    descriptions = {
        "A": "Requirement A（需求决策A）",
        "B": "Requirement B（需求决策B）",
        "C": "Requirement C（需求决胜C）",
    }

    output: dict[
        str,
        dict[str, Any],
    ] = {}

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        future_to_task = {
            executor.submit(
                run_one_requirement_decision,
                teacher=teacher,
                item=item,
                run_label=run_label,
            ): str(item["task_id"])
            for item in items
        }

        for future in tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc=descriptions[run_label],
            unit="task",
            dynamic_ncols=True,
        ):
            task_id = future_to_task[future]
            try:
                output[task_id] = (
                    future.result()
                )
            except BaseException as exc:
                output[task_id] = {
                    "task_id": task_id,
                    "run_label": run_label,
                    "stage": (
                        "requirement_decision"
                    ),
                    "api_success": False,
                    "parse_success": False,
                    "decision_table": None,
                    "teacher_call": None,
                    "raw_teacher_output": None,
                    "error": {
                        "stage": (
                            "requirement_executor"
                        ),
                        "error_type": (
                            type(exc).__name__
                        ),
                        "error": str(exc),
                    },
                }

    return output


def _requirement_ab_needs_c(
    *,
    run_a: Mapping[str, Any],
    run_b: Mapping[str, Any],
) -> bool:
    """A/B 任一槽位缺票或 decision 不同就调用 C。"""
    slots_a = (
        (run_a.get("decision_table") or {})
        .get("slots")
        or {}
    )
    slots_b = (
        (run_b.get("decision_table") or {})
        .get("slots")
        or {}
    )

    for slot_type in FIXED_DECISION_SLOT_TYPES:
        a = slots_a.get(slot_type)
        b = slots_b.get(slot_type)

        if (
            not isinstance(a, dict)
            or not isinstance(b, dict)
        ):
            return True

        if (
            str(a.get("decision") or "")
            != str(b.get("decision") or "")
        ):
            return True

    return False


def build_requirement_consensus(
    *,
    run_a: Mapping[str, Any],
    run_b: Mapping[str, Any],
    run_c: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    7 个 slot 各自做 2-of-3 decision majority。

    GLM 自报 confidence 仅记录诊断，不参与硬门。
    """
    runs = {
        "A": run_a,
        "B": run_b,
    }
    if run_c is not None:
        runs["C"] = run_c

    slot_results = {}

    for slot_type in FIXED_DECISION_SLOT_TYPES:
        votes = []

        for label, run in runs.items():
            table = run.get(
                "decision_table"
            ) or {}
            slot = (
                table.get("slots")
                or {}
            ).get(slot_type)

            if not isinstance(slot, dict):
                continue

            votes.append({
                "run": label,
                "decision": str(
                    slot.get("decision")
                    or ""
                ),
                "reason": str(
                    slot.get("reason")
                    or ""
                ),
                "teacher_confidence": float(
                    table.get("confidence")
                    or 0.0
                ),
            })

        counts = Counter(
            vote["decision"]
            for vote in votes
        )

        ranked = sorted(
            counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        if (
            not ranked
            or ranked[0][1] < 2
        ):
            slot_results[slot_type] = {
                "status": "no_consensus",
                "decision": None,
                "vote_counts": dict(
                    sorted(counts.items())
                ),
                "valid_vote_count": len(votes),
                "supporting_runs": [],
                "teacher_confidence_diagnostic": None,
            }
            continue

        decision = ranked[0][0]
        supporters = [
            vote
            for vote in votes
            if vote["decision"] == decision
        ]

        best = max(
            supporters,
            key=lambda vote: (
                vote["teacher_confidence"],
                vote["run"],
            ),
        )

        slot_results[slot_type] = {
            "status": "agreed",
            "decision": decision,
            "vote_counts": dict(
                sorted(counts.items())
            ),
            "valid_vote_count": len(votes),
            "supporting_runs": sorted(
                vote["run"]
                for vote in supporters
            ),
            "support_count": len(supporters),
            "reason": best["reason"],
            "teacher_confidence_diagnostic": {
                "min": min(
                    vote["teacher_confidence"]
                    for vote in supporters
                ),
                "max": max(
                    vote["teacher_confidence"]
                    for vote in supporters
                ),
                "mean": (
                    sum(
                        vote["teacher_confidence"]
                        for vote in supporters
                    )
                    / len(supporters)
                ),
            },
        }

    agreed_slots = [
        slot_type
        for slot_type, result
        in slot_results.items()
        if result.get("status") == "agreed"
    ]

    return {
        "status": (
            "agreed"
            if len(agreed_slots)
            == len(FIXED_DECISION_SLOT_TYPES)
            else "no_consensus"
        ),
        "consensus_mode": (
            "stage1_requirement_slot_2of3"
        ),
        "agreed_slot_count": len(
            agreed_slots
        ),
        "agreed_slots": agreed_slots,
        "slot_results": slot_results,
        "tie_break_used": (
            run_c is not None
        ),
        "independent_semantic_review": False,
    }


def parse_witness_selection_output(
    *,
    text: str,
    candidate_count: int,
) -> dict[str, Any]:
    """Stage 2 单槽位 Witness JSON。"""
    raw = _json_object_from_teacher_text(text)

    extra = set(raw) - {
        "confidence",
        "status",
        "witness_groups",
        "reason",
    }
    if extra:
        raise ValueError(
            "Witness JSON 存在禁止字段："
            f"{sorted(extra)}"
        )

    confidence = raw.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(
            confidence,
            (int, float),
        )
    ):
        raise ValueError(
            "confidence 必须是 number"
        )
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "confidence 必须在 [0,1]"
        )

    status = str(
        raw.get("status")
        or ""
    ).strip()
    if status not in WITNESS_STATUSES:
        raise ValueError(
            f"非法 witness status：{status!r}"
        )

    reason = str(
        raw.get("reason")
        or ""
    ).strip()
    if not reason:
        raise ValueError(
            "witness reason 不能为空"
        )

    raw_groups = raw.get(
        "witness_groups"
    )
    if raw_groups is None:
        raw_groups = []

    if status == "select":
        groups = _canonical_candidate_groups(
            raw_groups,
            candidate_count=candidate_count,
            field_name="witness_groups",
        )
        if not groups:
            raise ValueError(
                "status=select 要求非空 witness_groups"
            )
    else:
        if (
            not isinstance(raw_groups, list)
            or raw_groups
        ):
            raise ValueError(
                f"status={status} 时 witness_groups 必须为空"
            )
        groups = []

    return {
        "confidence": confidence,
        "status": status,
        "witness_groups": groups,
        "reason": reason,
    }


def run_one_witness_selection(
    *,
    teacher: Any,
    item: Mapping[str, Any],
    slot_type: str,
    requirement_result: Mapping[str, Any],
    run_label: str,
) -> dict[str, Any]:
    """单次 Stage 2 Targeted Witness Selection。"""
    task_id = str(item["task_id"])

    user_prompt = build_witness_stage_user_prompt(
        item=item,
        slot_type=slot_type,
        requirement_result=(
            requirement_result
        ),
    )

    result: dict[str, Any] = {
        "task_id": task_id,
        "slot_type": slot_type,
        "run_label": run_label,
        "stage": (
            "targeted_witness_selection"
        ),
        "api_success": False,
        "parse_success": False,
        "selection": None,
        "teacher_call": None,
        "raw_teacher_output": None,
        "prompt_chars": (
            len(WITNESS_SELECTION_SYSTEM_PROMPT)
            + len(user_prompt)
        ),
        "prompt_sha256": hashlib.sha256(
            (
                WITNESS_SELECTION_SYSTEM_PROMPT
                + "\n"
                + user_prompt
            ).encode("utf-8")
        ).hexdigest(),
        "error": None,
    }

    try:
        call = teacher.call(
            user_prompt=user_prompt,
            system_prompt=(
                WITNESS_SELECTION_SYSTEM_PROMPT
            ),
            offline_gold_reference_used=bool(
                item[
                    "offline_gold_reference_used"
                ]
            ),
        )
        result["api_success"] = True
        result["teacher_call"] = (
            call.metadata.to_dict()
        )
        result["raw_teacher_output"] = (
            call.output_text
        )
    except Exception as exc:
        result["error"] = {
            "stage": "witness_api_call",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return result

    try:
        result["selection"] = (
            parse_witness_selection_output(
                text=str(
                    result[
                        "raw_teacher_output"
                    ]
                ),
                candidate_count=len(
                    item[
                        "candidate_records"
                    ]
                ),
            )
        )
        result["parse_success"] = True
    except Exception as exc:
        result["error"] = {
            "stage": "parse_witness_selection",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    return result


def witness_selection_signature(
    selection: Mapping[str, Any],
) -> str:
    """A/B 判断是否需要 C；自由文本/置信度不参与。"""
    return json.dumps(
        {
            "status": selection.get(
                "status"
            ),
            "witness_groups": (
                selection.get(
                    "witness_groups"
                )
                or []
                if selection.get(
                    "status"
                )
                == "select"
                else []
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_witness_consensus(
    *,
    run_a: Mapping[str, Any],
    run_b: Mapping[str, Any],
    run_c: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    Stage 2：
    1. status 需要 2/3；
    2. status=select 时，每个 AND-group 需要 >=2 票。
    """
    runs = {
        "A": run_a,
        "B": run_b,
    }
    if run_c is not None:
        runs["C"] = run_c

    votes = []

    for label, run in runs.items():
        selection = run.get(
            "selection"
        )
        if not isinstance(
            selection,
            dict,
        ):
            continue

        votes.append({
            "run": label,
            "status": selection[
                "status"
            ],
            "witness_groups": (
                selection.get(
                    "witness_groups"
                )
                or []
            ),
            "reason": selection[
                "reason"
            ],
            "teacher_confidence": float(
                selection.get(
                    "confidence"
                )
                or 0.0
            ),
        })

    status_counts = Counter(
        vote["status"]
        for vote in votes
    )

    ranked = sorted(
        status_counts.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    if (
        not ranked
        or ranked[0][1] < 2
    ):
        return {
            "status": "no_consensus",
            "selection_status": None,
            "status_vote_counts": dict(
                sorted(
                    status_counts.items()
                )
            ),
            "witness_groups": [],
            "supporting_runs": [],
            "teacher_confidence_diagnostic": None,
        }

    selected_status = ranked[0][0]
    supporters = [
        vote
        for vote in votes
        if vote["status"]
        == selected_status
    ]

    confidence_diag = {
        "min": min(
            vote["teacher_confidence"]
            for vote in supporters
        ),
        "max": max(
            vote["teacher_confidence"]
            for vote in supporters
        ),
        "mean": (
            sum(
                vote["teacher_confidence"]
                for vote in supporters
            )
            / len(supporters)
        ),
    }

    if selected_status != "select":
        best = max(
            supporters,
            key=lambda vote: (
                vote["teacher_confidence"],
                vote["run"],
            ),
        )
        return {
            "status": "agreed",
            "selection_status": (
                selected_status
            ),
            "status_vote_counts": dict(
                sorted(
                    status_counts.items()
                )
            ),
            "witness_groups": [],
            "supporting_runs": sorted(
                vote["run"]
                for vote in supporters
            ),
            "reason": best["reason"],
            "teacher_confidence_diagnostic": (
                confidence_diag
            ),
        }

    group_counts = Counter()
    group_support = {}

    for vote in supporters:
        unique_groups = {
            tuple(group)
            for group in vote[
                "witness_groups"
            ]
        }

        for group in unique_groups:
            group_counts[group] += 1
            group_support.setdefault(
                group,
                [],
            ).append(
                vote["run"]
            )

    accepted_groups = [
        group
        for group, count
        in sorted(
            group_counts.items()
        )
        if count >= 2
    ]

    if not accepted_groups:
        return {
            "status": "no_consensus",
            "selection_status": "select",
            "status_vote_counts": dict(
                sorted(
                    status_counts.items()
                )
            ),
            "witness_groups": [],
            "group_vote_counts": {
                ",".join(
                    map(str, group)
                ): count
                for group, count
                in sorted(
                    group_counts.items()
                )
            },
            "supporting_runs": sorted(
                vote["run"]
                for vote in supporters
            ),
            "teacher_confidence_diagnostic": (
                confidence_diag
            ),
        }

    return {
        "status": "agreed",
        "selection_status": "select",
        "status_vote_counts": dict(
            sorted(
                status_counts.items()
            )
        ),
        "witness_groups": [
            list(group)
            for group in accepted_groups
        ],
        "group_vote_counts": {
            ",".join(
                map(str, group)
            ): count
            for group, count
            in sorted(
                group_counts.items()
            )
        },
        "group_supporting_runs": {
            ",".join(
                map(str, group)
            ): sorted(
                group_support[group]
            )
            for group in accepted_groups
        },
        "supporting_runs": sorted(
            vote["run"]
            for vote in supporters
        ),
        "teacher_confidence_diagnostic": (
            confidence_diag
        ),
    }


def _original_semantic_slot(
    *,
    slot_type: str,
    supervision: Mapping[str, Any],
    candidate_records: Sequence[
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    """
    Original Supervision -> v1.9.2 语义比较空间。

    用于判断最终结果是真的 REFINE，还是只是重新表达了原监督。
    """
    original_by_type = (
        _original_obligations_by_type(
            supervision
        )
    )

    original = original_by_type.get(
        slot_type
    )

    if original is None:
        return {
            "decision": "not_required",
            "witness_groups": [],
        }

    applicable = bool(
        original.get(
            "applicable",
            True,
        )
    )
    required = bool(
        original.get(
            "required_for_sufficiency",
            original.get(
                "mandatory",
                False,
            ),
        )
    )

    if (
        not applicable
        or not required
    ):
        return {
            "decision": "not_required",
            "witness_groups": [],
        }

    if bool(
        original.get(
            "satisfied_by_question",
            False,
        )
    ):
        return {
            "decision": "question_satisfied",
            "witness_groups": [],
        }

    id_to_number = {
        str(
            record.get(
                "evidence_id"
            )
            or ""
        ): index
        for index, record
        in enumerate(
            candidate_records,
            start=1,
        )
        if str(
            record.get(
                "evidence_id"
            )
            or ""
        )
    }

    groups = []

    for group in (
        original.get(
            "witness_groups"
        )
        or []
    ):
        evidence_ids = (
            group.get(
                "evidence_ids"
            )
            if isinstance(
                group,
                dict,
            )
            else group
        )

        if not isinstance(
            evidence_ids,
            list,
        ):
            continue

        numbers = []

        for evidence_id in evidence_ids:
            number = id_to_number.get(
                str(evidence_id)
            )

            if number is None:
                raise ValueError(
                    f"Original {slot_type} Witness "
                    f"{evidence_id} 不在 Teacher Candidate Pool"
                )

            numbers.append(number)

        if numbers:
            groups.append(
                tuple(
                    sorted(
                        set(numbers)
                    )
                )
            )

    groups = sorted(set(groups))

    if not groups:
        raise ValueError(
            f"Original {slot_type} required "
            "但没有可比较 Witness"
        )

    return {
        "decision": "use_candidates",
        "witness_groups": [
            list(group)
            for group in groups
        ],
    }


def _semantic_slot_signature(
    value: Mapping[str, Any],
) -> str:
    """Original / Final Slot 的机械语义签名。"""
    return json.dumps(
        {
            "decision": value.get(
                "decision"
            ),
            "witness_groups": (
                value.get(
                    "witness_groups"
                )
                or []
                if value.get(
                    "decision"
                )
                == "use_candidates"
                else []
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_two_stage_final_consensus(
    *,
    requirement_consensus: Mapping[str, Any],
    witness_consensus_by_slot: Mapping[
        str,
        Mapping[str, Any],
    ],
) -> dict[str, Any]:
    """Stage 1 + Stage 2 合并。"""
    if (
        requirement_consensus.get(
            "status"
        )
        != "agreed"
    ):
        return {
            "status": "no_consensus",
            "stage": "requirement",
            "reason": (
                "requirement_slots_not_all_agreed"
            ),
            "decision_table": None,
            "independent_semantic_review": False,
        }

    req_slots = (
        requirement_consensus.get(
            "slot_results"
        )
        or {}
    )

    final_slots = {}
    blocking = []

    for slot_type in FIXED_DECISION_SLOT_TYPES:
        requirement = req_slots[
            slot_type
        ]
        decision = requirement[
            "decision"
        ]

        if decision == "repository_required":
            witness = (
                witness_consensus_by_slot.get(
                    slot_type
                )
                or {}
            )

            if (
                witness.get("status")
                != "agreed"
            ):
                blocking.append({
                    "slot_type": slot_type,
                    "reason": (
                        "witness_no_consensus"
                    ),
                })
                continue

            status = witness.get(
                "selection_status"
            )

            if status == "select":
                final_slots[slot_type] = {
                    "decision": "use_candidates",
                    "witness_groups": (
                        witness[
                            "witness_groups"
                        ]
                    ),
                    "missing_context": None,
                    "reason": (
                        "v1.9.2 targeted Witness consensus"
                    ),
                }
            elif status == "insufficient":
                blocking.append({
                    "slot_type": slot_type,
                    "reason": (
                        "witness_candidate_pool_insufficient"
                    ),
                })
            else:
                blocking.append({
                    "slot_type": slot_type,
                    "reason": "witness_uncertain",
                })
            continue

        if decision == "question_satisfied":
            final_slots[slot_type] = {
                "decision": (
                    "question_satisfied"
                ),
                "witness_groups": [],
                "missing_context": None,
                "reason": (
                    requirement.get("reason")
                    or ""
                ),
            }
        elif decision == "not_required":
            final_slots[slot_type] = {
                "decision": "not_required",
                "witness_groups": [],
                "missing_context": None,
                "reason": (
                    requirement.get("reason")
                    or ""
                ),
            }
        elif (
            decision
            == "missing_pre_fix_context"
        ):
            blocking.append({
                "slot_type": slot_type,
                "reason": (
                    "requirement_missing_pre_fix_context"
                ),
            })
        else:
            blocking.append({
                "slot_type": slot_type,
                "reason": (
                    "requirement_uncertain"
                ),
            })

    if blocking:
        return {
            "status": "blocked",
            "stage": "two_stage_merge",
            "reason": (
                "semantic_or_witness_block"
            ),
            "blocking_slots": blocking,
            "decision_table": None,
            "independent_semantic_review": False,
        }

    return {
        "status": "agreed",
        "stage": "two_stage_merge",
        "reason": (
            "all_requirement_and_witness_slots_resolved"
        ),
        "decision_table": {
            "slots": final_slots,
        },
        "independent_semantic_review": False,
    }


def build_programmatic_refinement_v1_9_2(
    *,
    task_id: str,
    supervision: Mapping[str, Any],
    candidate_records: Sequence[
        Mapping[str, Any]
    ],
    candidate_ids: Sequence[str],
    existing_evidence_ids: set[str],
    token_costs: Mapping[str, int],
    final_consensus: Mapping[str, Any],
) -> tuple[
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """
    最终两阶段共识 -> Programmatic Supervision。

    KEEP/REFINE 由 Original-vs-Final 语义签名比较决定。
    """
    if (
        final_consensus.get("status")
        != "agreed"
    ):
        return None, None, None

    slots = (
        (
            final_consensus.get(
                "decision_table"
            )
            or {}
        ).get("slots")
        or {}
    )

    try:
        original_by_type = (
            _original_obligations_by_type(
                supervision
            )
        )

        candidate_number_to_record = {
            index: record
            for index, record
            in enumerate(
                candidate_records,
                start=1,
            )
        }

        changed_slots = []
        refined_obligations = []
        program_defects = []

        for slot_type in FIXED_DECISION_SLOT_TYPES:
            slot = slots[
                slot_type
            ]

            original_semantic = (
                _original_semantic_slot(
                    slot_type=slot_type,
                    supervision=supervision,
                    candidate_records=(
                        candidate_records
                    ),
                )
            )

            final_semantic = {
                "decision": slot[
                    "decision"
                ],
                "witness_groups": (
                    slot.get(
                        "witness_groups"
                    )
                    or []
                ),
            }

            changed = (
                _semantic_slot_signature(
                    original_semantic
                )
                != _semantic_slot_signature(
                    final_semantic
                )
            )

            if changed:
                changed_slots.append(
                    slot_type
                )

            original = original_by_type.get(
                slot_type
            )
            decision = slot[
                "decision"
            ]

            if decision == "not_required":
                if changed:
                    program_defects.append({
                        "code": (
                            "unnecessary_obligation"
                        ),
                        "obligation_type": (
                            slot_type
                        ),
                        "reason": (
                            "v1.9.2 requirement consensus "
                            "marks the original slot as not required"
                        ),
                    })
                continue

            if (
                decision
                == "question_satisfied"
            ):
                refined_obligations.append({
                    "source_obligation_id": (
                        str(
                            original.get(
                                "obligation_id"
                            )
                        )
                        if original
                        else None
                    ),
                    "type": slot_type,
                    "description": (
                        str(
                            (
                                original
                                or {}
                            ).get(
                                "description"
                            )
                            or ""
                        )
                        or DEFAULT_SLOT_DESCRIPTIONS[
                            slot_type
                        ]
                    ),
                    "applicable": True,
                    "required_for_sufficiency": True,
                    "satisfied_by_question": True,
                    "witness_groups": [],
                })

                if changed:
                    program_defects.append({
                        "code": (
                            "wrong_question_satisfied"
                            if original
                            else "missing_obligation"
                        ),
                        "obligation_type": (
                            slot_type
                        ),
                        "reason": (
                            "v1.9.2 final Question-satisfied "
                            "semantics differs from Original"
                        ),
                    })
                continue

            if decision != "use_candidates":
                raise ValueError(
                    "非法最终训练语义："
                    f"{slot_type}={decision!r}"
                )

            groups = []

            for group in (
                slot.get(
                    "witness_groups"
                )
                or []
            ):
                evidence_ids = []

                for number in group:
                    record = (
                        candidate_number_to_record[
                            int(number)
                        ]
                    )
                    evidence_id = str(
                        record.get(
                            "evidence_id"
                        )
                        or ""
                    )
                    if not evidence_id:
                        raise ValueError(
                            f"Candidate {number} "
                            "缺少 evidence_id"
                        )
                    evidence_ids.append(
                        evidence_id
                    )

                groups.append({
                    "evidence_ids": (
                        evidence_ids
                    ),
                    "reason": (
                        "selected by v1.9.2 "
                        "targeted Witness consensus"
                    ),
                })

            if not groups:
                raise ValueError(
                    f"{slot_type}=use_candidates "
                    "但没有 Witness Group"
                )

            refined_obligations.append({
                "source_obligation_id": (
                    str(
                        original.get(
                            "obligation_id"
                        )
                    )
                    if original
                    else None
                ),
                "type": slot_type,
                "description": (
                    str(
                        (
                            original
                            or {}
                        ).get(
                            "description"
                        )
                        or ""
                    )
                    or DEFAULT_SLOT_DESCRIPTIONS[
                        slot_type
                    ]
                ),
                "applicable": True,
                "required_for_sufficiency": True,
                "satisfied_by_question": False,
                "witness_groups": groups,
            })

            if changed:
                original_decision = (
                    original_semantic[
                        "decision"
                    ]
                )

                if (
                    original_decision
                    == "use_candidates"
                ):
                    defect_code = "wrong_witness"
                elif (
                    original_decision
                    == "question_satisfied"
                ):
                    defect_code = (
                        "wrong_question_satisfied"
                    )
                else:
                    defect_code = (
                        "missing_obligation"
                    )

                program_defects.append({
                    "code": defect_code,
                    "obligation_type": (
                        slot_type
                    ),
                    "reason": (
                        "v1.9.2 final semantic slot "
                        "differs from Original Supervision"
                    ),
                })

        if not changed_slots:
            proposal = {
                "assessment": "keep",
                "stop_assessment": (
                    "original_stop_correct"
                ),
                # Core schema 需要 confidence；
                # 这里明确是程序构造，不使用 Teacher 自报分数。
                "confidence": 1.0,
                "rationale": (
                    "v1.9.2 two-stage consensus "
                    "is semantically equivalent to Original"
                ),
                "refinement_defects": [],
                "missing_candidate_requests": [],
                "refined_obligations": [],
                "proposed_certificate_evidence_ids": list(
                    original_certificate(
                        supervision
                    )
                ),
            }

            verification = (
                verify_and_finalize_refinement(
                    task_id=task_id,
                    supervision=supervision,
                    candidate_evidence_ids=(
                        candidate_ids
                    ),
                    existing_evidence_ids=(
                        existing_evidence_ids
                    ),
                    token_costs=token_costs,
                    proposal=proposal,
                )
            )
            return (
                proposal,
                verification,
                None,
            )

        union_evidence = []
        seen_evidence = set()

        for obligation in refined_obligations:
            for group in (
                obligation.get(
                    "witness_groups"
                )
                or []
            ):
                for evidence_id in (
                    group.get(
                        "evidence_ids"
                    )
                    or []
                ):
                    evidence_id = str(
                        evidence_id
                    )
                    if (
                        evidence_id
                        not in seen_evidence
                    ):
                        seen_evidence.add(
                            evidence_id
                        )
                        union_evidence.append(
                            evidence_id
                        )

        proposal = {
            "assessment": "refine",
            "stop_assessment": "uncertain",
            "confidence": 1.0,
            "rationale": (
                "v1.9.2 two-stage consensus; "
                "changed_slots="
                + ",".join(
                    changed_slots
                )
            ),
            "refinement_defects": (
                program_defects
            ),
            "missing_candidate_requests": [],
            "refined_obligations": (
                refined_obligations
            ),
            "proposed_certificate_evidence_ids": (
                union_evidence
            ),
        }

        first = (
            verify_and_finalize_refinement(
                task_id=task_id,
                supervision=supervision,
                candidate_evidence_ids=(
                    candidate_ids
                ),
                existing_evidence_ids=(
                    existing_evidence_ids
                ),
                token_costs=token_costs,
                proposal=proposal,
            )
        )

        if (
            str(
                first.get(
                    "verification_status"
                )
                or ""
            )
            != "accepted"
        ):
            return (
                proposal,
                first,
                None,
            )

        verified_certificate = list(
            first.get(
                "verified_minimal_certificate_evidence_ids"
            )
            or []
        )

        proposal["stop_assessment"] = (
            _derive_stop_assessment(
                old_certificate=(
                    original_certificate(
                        supervision
                    )
                ),
                new_certificate=(
                    verified_certificate
                ),
            )
        )
        proposal[
            "proposed_certificate_evidence_ids"
        ] = verified_certificate

        final = (
            verify_and_finalize_refinement(
                task_id=task_id,
                supervision=supervision,
                candidate_evidence_ids=(
                    candidate_ids
                ),
                existing_evidence_ids=(
                    existing_evidence_ids
                ),
                token_costs=token_costs,
                proposal=proposal,
            )
        )

        return (
            proposal,
            final,
            None,
        )

    except Exception as exc:
        return (
            None,
            None,
            {
                "stage": (
                    "programmatic_supervision_construction_v1_9_2"
                ),
                "error_type": (
                    type(exc).__name__
                ),
                "error": str(exc),
            },
        )


def build_v1_9_2_quality_gate(
    *,
    split: str,
    final_consensus: Mapping[str, Any],
    verification: Mapping[str, Any] | None,
    construction_error: Mapping[str, Any] | None,
    promote_consensus_training: bool,
) -> dict[str, Any]:
    """
    两阶段质量门。

    Teacher confidence 仅审计，不再参与硬过滤。
    """
    reasons = []

    if (
        final_consensus.get("status")
        != "agreed"
    ):
        reasons.append(
            "two_stage_consensus_not_complete"
        )

    verification_status = str(
        (
            verification
            or {}
        ).get(
            "verification_status"
        )
        or ""
    )

    if (
        verification_status
        != "accepted"
    ):
        reasons.append(
            "programmatic_verification_not_accepted"
        )

    if construction_error is not None:
        reasons.append(
            "programmatic_construction_error"
        )

    reasons = list(
        dict.fromkeys(reasons)
    )

    verified = (
        len(reasons)
        == 0
    )

    return {
        "status": (
            "verified"
            if verified
            else "blocked"
        ),
        "supervision_verified": verified,
        "two_stage_consensus_stable": (
            final_consensus.get("status")
            == "agreed"
        ),
        "deterministic_verified": (
            verification_status
            == "accepted"
        ),
        "teacher_confidence_used_as_hard_gate": False,
        "independent_semantic_review": False,
        "block_reasons": (
            []
            if verified
            else reasons
        ),
        "verification_tier": (
            "glm_two_stage_consensus_v1_9_2_1"
            if verified
            else "blocked"
        ),
        "training_eligible": bool(
            verified
            and split == "train"
            and promote_consensus_training
        ),
        "training_promotion_enabled": bool(
            promote_consensus_training
        ),
        "training_policy": (
            "train split only; requires Stage-1 Requirement 2-of-3 consensus "
            "+ Stage-2 targeted Witness group consensus "
            "+ Core v1.7 deterministic verification "
            "+ explicit --promote-consensus-training. "
            "Teacher self-confidence is diagnostic only."
        ),
    }

def run_one_fixed_decision(
    *,
    teacher: Any,
    item: Mapping[str, Any],
    run_label: str,
) -> dict[str, Any]:
    """
    v1.9.1：
        API
        -> JSON 顶层解析
        -> 7 个 slot 局部解析
        -> keep_original 有效语义展开

    单个 slot 错误不会丢弃整次调用。
    """

    task_id = str(
        item[
            "task_id"
        ]
    )

    result: dict[
        str,
        Any,
    ] = {
        "task_id": (
            task_id
        ),
        "run_label": (
            run_label
        ),
        "teacher_role": (
            "glm_fixed_decision"
        ),
        "api_success": False,
        "parse_success": False,
        "decision_table": None,
        "effective_decision_table": None,
        "decision_signature": None,
        "teacher_call": None,
        "raw_teacher_output": None,
        "error": None,
    }

    try:
        call = teacher.call(
            user_prompt=(
                item[
                    "user_prompt"
                ]
            ),
            offline_gold_reference_used=(
                bool(
                    item[
                        "offline_gold_reference_used"
                    ]
                )
            ),
        )

        result[
            "api_success"
        ] = True

        result[
            "teacher_call"
        ] = (
            call
            .metadata
            .to_dict()
        )

        result[
            "raw_teacher_output"
        ] = (
            call.output_text
        )

    except Exception as exc:
        result[
            "error"
        ] = {
            "stage": (
                "api_call"
            ),
            "error_type": (
                type(
                    exc
                ).__name__
            ),
            "error": str(
                exc
            ),
        }

        return result

    try:
        table = (
            parse_fixed_decision_table_slot_local(
                text=str(
                    result[
                        "raw_teacher_output"
                    ]
                ),
                candidate_count=len(
                    item[
                        "candidate_records"
                    ]
                ),
            )
        )

        result[
            "parse_success"
        ] = True

        result[
            "decision_table"
        ] = (
            table
        )

        effective = (
            normalize_run_effective_votes(
                parsed_table=(
                    table
                ),
                supervision=(
                    item[
                        "supervision"
                    ]
                ),
                candidate_records=(
                    item[
                        "candidate_records"
                    ]
                ),
            )
        )

        result[
            "effective_decision_table"
        ] = (
            effective
        )

        # 仅用于审计；
        # v1.9.1 最终共识不再使用 Whole-table Signature。
        if (
            effective.get(
                "valid_slot_count"
            )
            == len(
                FIXED_DECISION_SLOT_TYPES
            )
        ):
            result[
                "decision_signature"
            ] = json.dumps(
                {
                    slot_type: (
                        effective_slot_signature(
                            effective[
                                "slots"
                            ][
                                slot_type
                            ]
                        )
                    )
                    for slot_type in (
                        FIXED_DECISION_SLOT_TYPES
                    )
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
            )

    except Exception as exc:
        result[
            "error"
        ] = {
            "stage": (
                "parse_fixed_decision"
            ),
            "error_type": (
                type(
                    exc
                ).__name__
            ),
            "error": str(
                exc
            ),
        }

    return result

def execute_fixed_decision_batch(
    *,
    teacher: Any,
    items: Sequence[Mapping[str, Any]],
    run_label: str,
    max_workers: int,
) -> dict[str, dict[str, Any]]:
    """并发执行一轮 Fixed Decision。"""
    if max_workers < 1:
        raise ValueError("max_workers 必须 >= 1")

    descriptions = {
        "A": "GLM decision A（GLM 决策A）",
        "B": "GLM decision B（GLM 决策B）",
        "C": "GLM tie-break C（GLM 决胜C）",
    }
    output: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                run_one_fixed_decision,
                teacher=teacher,
                item=item,
                run_label=run_label,
            ): str(item["task_id"])
            for item in items
        }
        for future in tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc=descriptions.get(run_label, f"GLM decision {run_label}"),
            unit="task",
            dynamic_ncols=True,
        ):
            task_id = future_to_task[future]
            try:
                output[task_id] = future.result()
            except BaseException as exc:
                output[task_id] = {
                    "task_id": task_id,
                    "run_label": run_label,
                    "teacher_role": "glm_fixed_decision",
                    "api_success": False,
                    "parse_success": False,
                    "decision_table": None,
                    "effective_decision_table": None,
                    "decision_signature": None,
                    "teacher_call": None,
                    "raw_teacher_output": None,
                    "error": {
                        "stage": "executor",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                }
    return output


def build_2of3_consensus(
    *,
    run_a: Mapping[str, Any],
    run_b: Mapping[str, Any],
    run_c: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """A/B 相同直接接受；不同才调用 C；任意精确 2-of-3 结构一致即通过。"""
    runs = {"A": run_a, "B": run_b}
    if run_c is not None:
        runs["C"] = run_c

    signatures: dict[str, list[str]] = {}
    for label, result in runs.items():
        sig = result.get("decision_signature")
        if isinstance(sig, str):
            signatures.setdefault(sig, []).append(label)

    ranked = sorted(signatures.items(), key=lambda item: (-len(item[1]), item[0]))
    if not ranked or len(ranked[0][1]) < 2:
        return {
            "status": "no_consensus",
            "tie_break_used": run_c is not None,
            "agreeing_runs": [],
            "decision_signature": None,
            "decision_table": None,
            "consensus_confidence": None,
            "valid_run_count": sum(
                bool(result.get("decision_signature")) for result in runs.values()
            ),
        }

    winner_signature, labels = ranked[0]
    labels = sorted(labels)
    agreeing = [runs[label] for label in labels]
    chosen = agreeing[0]["decision_table"]
    confidences = [
        float((result.get("decision_table") or {}).get("confidence") or 0.0)
        for result in agreeing
    ]
    return {
        "status": "agreed",
        "tie_break_used": run_c is not None,
        "agreeing_runs": labels,
        "decision_signature": winner_signature,
        "decision_table": chosen,
        "consensus_confidence": min(confidences) if confidences else 0.0,
        "valid_run_count": sum(
            bool(result.get("decision_signature")) for result in runs.values()
        ),
    }


def _original_obligation_to_refined(obligation: Mapping[str, Any]) -> dict[str, Any]:
    """机械保留 Original Obligation，不新增语义。"""
    groups = []
    for group in (obligation.get("witness_groups") or []):
        ids = [str(eid) for eid in (group.get("evidence_ids") or [])]
        if ids:
            groups.append({
                "evidence_ids": ids,
                "reason": "preserved from original supervision",
            })

    required = bool(
        obligation.get(
            "required_for_sufficiency",
            obligation.get("mandatory", False),
        )
    )
    question_satisfied = bool(obligation.get("satisfied_by_question", False))
    otype = str(obligation.get("type") or "")
    return {
        "source_obligation_id": str(obligation.get("obligation_id") or ""),
        "type": otype,
        "description": str(obligation.get("description") or "")
        or DEFAULT_SLOT_DESCRIPTIONS[otype],
        "applicable": bool(obligation.get("applicable", True)),
        "required_for_sufficiency": required,
        "satisfied_by_question": question_satisfied,
        "witness_groups": [] if question_satisfied else groups,
    }


def _derive_stop_assessment(
    *,
    old_certificate: Sequence[str],
    new_certificate: Sequence[str],
) -> str:
    """STOP 完全由新旧 verified minimal certificate 的集合关系推导。"""
    old_set = set(map(str, old_certificate))
    new_set = set(map(str, new_certificate))
    if new_set == old_set:
        return "original_stop_correct"
    if old_set < new_set:
        return "too_early"
    if new_set < old_set:
        return "too_late"
    return "uncertain"


def build_programmatic_refinement_from_decision(
    *,
    task_id: str,
    supervision: Mapping[str, Any],
    candidate_records: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    existing_evidence_ids: set[str],
    token_costs: Mapping[str, int],
    consensus: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Fixed Decision -> 程序构造 Proposal -> Core v1.7 确定性验证。"""
    if consensus.get("status") != "agreed":
        return None, None, None

    table = consensus.get("decision_table") or {}
    slots = table.get("slots") or {}
    confidence = float(consensus.get("consensus_confidence") or 0.0)

    try:
        original_by_type = _original_obligations_by_type(supervision)
        candidate_number_to_record = {
            index: record for index, record in enumerate(candidate_records, start=1)
        }

        if any(
            slots[slot_type]["decision"] == "uncertain"
            for slot_type in FIXED_DECISION_SLOT_TYPES
        ):
            proposal = {
                "assessment": "uncertain",
                "stop_assessment": "uncertain",
                "confidence": confidence,
                "rationale": "v1.9 fixed-slot consensus contains uncertain slot",
                "refinement_defects": [],
                "missing_candidate_requests": [],
                "refined_obligations": [],
                "proposed_certificate_evidence_ids": [],
            }
            verification = verify_and_finalize_refinement(
                task_id=task_id,
                supervision=supervision,
                candidate_evidence_ids=candidate_ids,
                existing_evidence_ids=existing_evidence_ids,
                token_costs=token_costs,
                proposal=proposal,
            )
            return proposal, verification, None

        missing_slots = [
            slot_type
            for slot_type in FIXED_DECISION_SLOT_TYPES
            if slots[slot_type]["decision"] == "missing_pre_fix_context"
        ]
        if missing_slots:
            requests = []
            for slot_type in missing_slots:
                context = slots[slot_type]["missing_context"] or {}
                requests.append({
                    "obligation_type": slot_type,
                    "path_hint": str(context.get("path_hint") or ""),
                    "symbol_hint": str(context.get("symbol_hint") or ""),
                    "keywords": list(context.get("keywords") or []),
                    "reason": str(context.get("reason") or ""),
                })
            proposal = {
                "assessment": "candidate_pool_insufficient",
                "stop_assessment": "too_early",
                "confidence": confidence,
                "rationale": "v1.9 fixed-slot consensus reports missing necessary pre-fix context",
                "refinement_defects": [],
                "missing_candidate_requests": requests,
                "refined_obligations": [],
                "proposed_certificate_evidence_ids": [],
            }
            verification = verify_and_finalize_refinement(
                task_id=task_id,
                supervision=supervision,
                candidate_evidence_ids=candidate_ids,
                existing_evidence_ids=existing_evidence_ids,
                token_costs=token_costs,
                proposal=proposal,
            )
            return proposal, verification, None

        changed_slots: list[str] = []
        refined_obligations: list[dict[str, Any]] = []
        program_defects: list[dict[str, Any]] = []

        for slot_type in FIXED_DECISION_SLOT_TYPES:
            slot = slots[slot_type]
            decision = slot["decision"]
            original = original_by_type.get(slot_type)

            if decision == "keep_original":
                if original is None:
                    raise ValueError(f"{slot_type}=keep_original 但原监督无该 type")
                refined_obligations.append(_original_obligation_to_refined(original))
                continue

            if decision == "not_required":
                if original is not None:
                    changed_slots.append(slot_type)
                    program_defects.append({
                        "code": "unnecessary_obligation",
                        "obligation_type": slot_type,
                        "reason": "fixed-slot consensus marks original obligation as not required",
                    })
                continue

            if decision == "question_satisfied":
                changed_slots.append(slot_type)
                refined_obligations.append({
                    "source_obligation_id": (
                        str(original.get("obligation_id")) if original else None
                    ),
                    "type": slot_type,
                    "description": str((original or {}).get("description") or "")
                    or DEFAULT_SLOT_DESCRIPTIONS[slot_type],
                    "applicable": True,
                    "required_for_sufficiency": True,
                    "satisfied_by_question": True,
                    "witness_groups": [],
                })
                program_defects.append({
                    "code": "wrong_question_satisfied" if original else "missing_obligation",
                    "obligation_type": slot_type,
                    "reason": "fixed-slot consensus marks necessary semantic as satisfied by Question",
                })
                continue

            if decision == "use_candidates":
                changed_slots.append(slot_type)
                groups = []
                for group in slot["witness_groups"]:
                    evidence_ids = []
                    for number in group:
                        record = candidate_number_to_record[number]
                        evidence_id = str(record.get("evidence_id") or "")
                        if not evidence_id:
                            raise ValueError(f"Candidate {number} 缺少 evidence_id")
                        evidence_ids.append(evidence_id)
                    groups.append({
                        "evidence_ids": evidence_ids,
                        "reason": "selected by v1.9 fixed-slot 2-of-3 consensus",
                    })

                refined_obligations.append({
                    "source_obligation_id": (
                        str(original.get("obligation_id")) if original else None
                    ),
                    "type": slot_type,
                    "description": str((original or {}).get("description") or "")
                    or DEFAULT_SLOT_DESCRIPTIONS[slot_type],
                    "applicable": True,
                    "required_for_sufficiency": True,
                    "satisfied_by_question": False,
                    "witness_groups": groups,
                })
                program_defects.append({
                    "code": "wrong_witness" if original else "missing_obligation",
                    "obligation_type": slot_type,
                    "reason": "fixed-slot consensus changes repository Witness structure",
                })
                continue

            raise ValueError(f"不应到达的 fixed slot decision：{decision!r}")

        # 原本没有的 slot，not_required 才表示保持“不存在”。
        for slot_type in FIXED_DECISION_SLOT_TYPES:
            if (
                slot_type not in original_by_type
                and slots[slot_type]["decision"] != "not_required"
                and slot_type not in changed_slots
            ):
                changed_slots.append(slot_type)

        if not changed_slots:
            proposal = {
                "assessment": "keep",
                "stop_assessment": "original_stop_correct",
                "confidence": confidence,
                "rationale": "v1.9 2-of-3 consensus preserves all original semantic slots",
                "refinement_defects": [],
                "missing_candidate_requests": [],
                "refined_obligations": [],
                "proposed_certificate_evidence_ids": list(original_certificate(supervision)),
            }
            verification = verify_and_finalize_refinement(
                task_id=task_id,
                supervision=supervision,
                candidate_evidence_ids=candidate_ids,
                existing_evidence_ids=existing_evidence_ids,
                token_costs=token_costs,
                proposal=proposal,
            )
            return proposal, verification, None

        # REFINE：Teacher 不选 Certificate；先给 Witness union，由 Core 机械最小化。
        union_evidence: list[str] = []
        seen: set[str] = set()
        for obligation in refined_obligations:
            for group in (obligation.get("witness_groups") or []):
                for evidence_id in (group.get("evidence_ids") or []):
                    evidence_id = str(evidence_id)
                    if evidence_id not in seen:
                        seen.add(evidence_id)
                        union_evidence.append(evidence_id)

        proposal = {
            "assessment": "refine",
            "stop_assessment": "uncertain",
            "confidence": confidence,
            "rationale": "v1.9 fixed-slot 2-of-3 consensus; changed_slots="
            + ",".join(changed_slots),
            "refinement_defects": program_defects,
            "missing_candidate_requests": [],
            "refined_obligations": refined_obligations,
            "proposed_certificate_evidence_ids": union_evidence,
        }

        first = verify_and_finalize_refinement(
            task_id=task_id,
            supervision=supervision,
            candidate_evidence_ids=candidate_ids,
            existing_evidence_ids=existing_evidence_ids,
            token_costs=token_costs,
            proposal=proposal,
        )
        if str(first.get("verification_status") or "") != "accepted":
            return proposal, first, None

        verified_certificate = list(
            first.get("verified_minimal_certificate_evidence_ids") or []
        )
        proposal["stop_assessment"] = _derive_stop_assessment(
            old_certificate=original_certificate(supervision),
            new_certificate=verified_certificate,
        )
        proposal["proposed_certificate_evidence_ids"] = verified_certificate

        final = verify_and_finalize_refinement(
            task_id=task_id,
            supervision=supervision,
            candidate_evidence_ids=candidate_ids,
            existing_evidence_ids=existing_evidence_ids,
            token_costs=token_costs,
            proposal=proposal,
        )
        return proposal, final, None

    except Exception as exc:
        return None, None, {
            "stage": "programmatic_supervision_construction",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def build_consensus_primary_result(
    *,
    item: Mapping[str, Any],
    run_a: Mapping[str, Any],
    run_b: Mapping[str, Any],
    run_c: Mapping[str, Any] | None,
    consensus: Mapping[str, Any],
) -> dict[str, Any]:
    """把 A/B/C + 共识转换成兼容 refinement.jsonl 的任务级 Primary Result。"""
    task_id = str(item["task_id"])
    proposal, verification, construction_error = build_programmatic_refinement_from_decision(
        task_id=task_id,
        supervision=item["supervision"],
        candidate_records=item["candidate_records"],
        candidate_ids=item["candidate_ids"],
        existing_evidence_ids=set(map(str, item["candidate_ids"])),
        token_costs=item["token_costs"],
        consensus=consensus,
    )

    calls = [
        result.get("teacher_call") or {}
        for result in (run_a, run_b, run_c)
        if result is not None
    ]
    aggregate_call = {
        "provider": "bigmodel",
        "model": next(
            (str(call.get("model")) for call in calls if call.get("model")),
            "glm-4.7-flash",
        ),
        "prompt_tokens": sum(int(call.get("prompt_tokens") or 0) for call in calls),
        "completion_tokens": sum(
            int(call.get("completion_tokens") or 0) for call in calls
        ),
        "total_tokens": sum(int(call.get("total_tokens") or 0) for call in calls),
        "call_count": len(calls),
    }

    error = construction_error
    if error is None and consensus.get("status") != "agreed":
        error = {
            "stage": "consensus",
            "error_type": "NoConsensus",
            "error": "GLM fixed decision did not reach 2-of-3 structural consensus",
        }

    return {
        "task_id": task_id,
        "teacher_role": "glm_fixed_decision_2of3",
        "api_success": sum(
            bool(result.get("api_success"))
            for result in (run_a, run_b, run_c)
            if result is not None
        ) >= 2,
        "parse_success": sum(
            bool(result.get("parse_success"))
            for result in (run_a, run_b, run_c)
            if result is not None
        ) >= 2,
        "binding_success": True,
        "verification_exception": construction_error is not None,
        "proposal": proposal,
        "verification": verification,
        "teacher_call": aggregate_call,
        "raw_teacher_output": None,
        "evidence_id_alias_normalizations": [],
        "candidate_reference_binding_events": [],
        "decision_runs": {"A": run_a, "B": run_b, "C": run_c},
        "consensus": consensus,
        "error": error,
    }


def build_consensus_quality_gate(
    *,
    split: str,
    primary_result: Mapping[str, Any],
    min_consensus_confidence: float,
    promote_consensus_training: bool,
) -> dict[str, Any]:
    """v1.9 GLM-only Quality Gate；自一致不等于独立语义复核。"""
    reasons: list[str] = []
    consensus = primary_result.get("consensus") or {}
    if consensus.get("status") != "agreed":
        reasons.append("no_full_slot_consensus")

    confidence = consensus.get("consensus_confidence")
    if confidence is None or float(confidence) < min_consensus_confidence:
        reasons.append("consensus_confidence_below_threshold")

    verification = primary_result.get("verification") or {}
    verification_status = str(verification.get("verification_status") or "")
    if verification_status != "accepted":
        reasons.append("programmatic_verification_not_accepted")

    if primary_result.get("error"):
        reasons.append("primary_runtime_or_construction_error")

    reasons = list(dict.fromkeys(reasons))
    verified = len(reasons) == 0
    training_eligible = bool(
        verified
        and split == "train"
        and promote_consensus_training
    )
    return {
        "supervision_verified": verified,
        "consensus_stable": consensus.get("status") == "agreed",
        "deterministic_verified": verification_status == "accepted",
        "training_eligible": training_eligible,
        "status": "verified" if verified else "blocked",
        "verification_tier": (
            "glm_slot_consensus_v1_9_1" if verified else "blocked"
        ),
        "block_reasons": [] if verified else reasons,
        "consensus_confidence": confidence,
        "agreeing_runs": consensus.get("agreeing_runs") or [],
        "tie_break_used": bool(consensus.get("tie_break_used")),
        "training_promotion_enabled": bool(promote_consensus_training),
        "training_policy": (
            "train split only; requires v1.9.1 all-seven-slot consensus (2-of-3 decision majority + 2-vote Witness AND-groups) "
            "+ Core v1.7 deterministic verification + explicit "
            "--promote-consensus-training"
        ),
    }



def summarize_fixed_decision_runs(
    results: Sequence[
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    """
    v1.9.1 单轮汇总。

    parse_failure_count：
        只统计 JSON / 顶层 Schema 失败。

    invalid_slot_vote_count：
        统计 slot-local parse / effective normalization 失败。
    """

    calls = [
        result.get(
            "teacher_call"
        )
        or {}
        for result in (
            results
        )
        if result.get(
            "api_success"
        )
    ]

    invalid_slot_counts = Counter()

    for result in (
        results
    ):
        table = (
            result.get(
                "effective_decision_table"
            )
            or result.get(
                "decision_table"
            )
            or {}
        )

        for error in (
            table.get(
                "slot_errors"
            )
            or []
        ):
            slot_type = str(
                error.get(
                    "slot_type"
                )
                or "unknown"
            )

            invalid_slot_counts[
                slot_type
            ] += 1

    return {
        "record_count": len(
            results
        ),
        "api_success_count": sum(
            bool(
                result.get(
                    "api_success"
                )
            )
            for result in (
                results
            )
        ),
        "api_failure_count": sum(
            not bool(
                result.get(
                    "api_success"
                )
            )
            for result in (
                results
            )
        ),
        "parse_success_count": sum(
            bool(
                result.get(
                    "parse_success"
                )
            )
            for result in (
                results
            )
        ),
        "parse_failure_count": sum(
            bool(
                result.get(
                    "api_success"
                )
            )
            and not bool(
                result.get(
                    "parse_success"
                )
            )
            for result in (
                results
            )
        ),
        "invalid_slot_vote_count": sum(
            invalid_slot_counts.values()
        ),
        "invalid_slot_vote_counts_by_type": dict(
            sorted(
                invalid_slot_counts.items()
            )
        ),
        "prompt_tokens": sum(
            int(
                call.get(
                    "prompt_tokens"
                )
                or 0
            )
            for call in (
                calls
            )
        ),
        "completion_tokens": sum(
            int(
                call.get(
                    "completion_tokens"
                )
                or 0
            )
            for call in (
                calls
            )
        ),
        "total_tokens": sum(
            int(
                call.get(
                    "total_tokens"
                )
                or 0
            )
            for call in (
                calls
            )
        ),
    }

def run_one_teacher(
    *,
    teacher: Any,
    teacher_role: str,
    item: Mapping[str, Any],
) -> dict[str, Any]:
    """
    对一个 task 执行：

        API call
        -> JSON parse
        -> Deterministic Verification

    无论在哪个阶段失败，都返回结构化结果。
    不让异常直接丢失任务。
    """

    task_id = str(
        item[
            "task_id"
        ]
    )

    result: dict[str, Any] = {
        "task_id": task_id,
        "teacher_role": (
            teacher_role
        ),
        "api_success": False,
        "parse_success": False,
        "binding_success": False,
        "verification_exception": False,
        "proposal": None,
        "verification": None,
        "teacher_call": None,
        "raw_teacher_output": None,
        "evidence_id_alias_normalizations": [],
        "candidate_reference_binding_events": [],
        "error": None,
    }

    try:
        call = teacher.call(
            user_prompt=(
                item[
                    "user_prompt"
                ]
            ),
            offline_gold_reference_used=(
                bool(
                    item[
                        "offline_gold_reference_used"
                    ]
                )
            ),
        )

        result[
            "api_success"
        ] = True

        result[
            "teacher_call"
        ] = (
            call
            .metadata
            .to_dict()
        )

        result[
            "raw_teacher_output"
        ] = (
            call.output_text
        )

    except Exception as exc:
        result[
            "error"
        ] = {
            "stage": (
                "api_call"
            ),
            "error_type": (
                type(exc).__name__
            ),
            "error": str(
                exc
            ),
        }

        return result

    # Stage 1: JSON Proposal Parse（JSON 提案解析）
    try:
        parsed_proposal = (
            parse_refinement_proposal(
                str(
                    result[
                        "raw_teacher_output"
                    ]
                )
            )
        )
        result["parse_success"] = True

    except Exception as exc:
        result[
            "error"
        ] = {
            "stage": "parse_teacher_output",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return result

    # Stage 2: Candidate Binding（候选绑定）
    try:
        (
            proposal,
            candidate_reference_binding_events,
        ) = bind_teacher_candidate_references(
            proposal=parsed_proposal,
            candidate_records=(
                item["candidate_records"]
            ),
            original_certificate_ids=(
                original_certificate(
                    item["supervision"]
                )
            ),
        )

        result["proposal"] = proposal
        result["binding_success"] = True
        result["evidence_id_alias_normalizations"] = []
        result[
            "candidate_reference_binding_events"
        ] = candidate_reference_binding_events

    except Exception as exc:
        result[
            "error"
        ] = {
            "stage": "bind_candidate_references",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return result

    try:
        verification = (
            verify_and_finalize_refinement(
                task_id=task_id,
                supervision=(
                    item[
                        "supervision"
                    ]
                ),
                candidate_evidence_ids=(
                    item[
                        "candidate_ids"
                    ]
                ),
                existing_evidence_ids=set(
                    item[
                        "candidate_ids"
                    ]
                ),
                token_costs=(
                    item[
                        "token_costs"
                    ]
                ),
                proposal=(
                    proposal
                ),
            )
        )

        result[
            "verification"
        ] = (
            verification
        )

        return result

    except Exception as exc:
        result[
            "verification_exception"
        ] = True

        result[
            "error"
        ] = {
            "stage": (
                "verify_proposal"
            ),
            "error_type": (
                type(exc).__name__
            ),
            "error": str(
                exc
            ),
        }

        return result


def run_one_strong_reviewer(
    *,
    teacher: Any,
    item: Mapping[str, Any],
) -> dict[str, Any]:
    """
    DeepSeek Strong Reviewer 单任务执行。

    API
      -> Review JSON parse

    不进入 refinement_core，
    因为 Reviewer 不生成新的监督提案。
    """

    task_id = str(
        item[
            "task_id"
        ]
    )

    result: dict[str, Any] = {
        "task_id": task_id,
        "teacher_role": (
            "strong_semantic_reviewer_deepseek"
        ),
        "api_success": False,
        "parse_success": False,
        "review": None,
        "teacher_call": None,
        "raw_teacher_output": None,
        "error": None,
    }

    try:
        call = teacher.call(
            user_prompt=(
                item[
                    "user_prompt"
                ]
            ),
            offline_gold_reference_used=(
                bool(
                    item[
                        "offline_gold_reference_used"
                    ]
                )
            ),
            system_prompt=(
                STRONG_REVIEW_SYSTEM_PROMPT
            ),
        )

        result[
            "api_success"
        ] = True

        result[
            "teacher_call"
        ] = (
            call.metadata.to_dict()
        )

        result[
            "raw_teacher_output"
        ] = (
            call.output_text
        )

    except Exception as exc:
        result[
            "error"
        ] = {
            "stage": "api_call",
            "error_type": (
                type(exc).__name__
            ),
            "error": str(exc),
        }
        return result

    try:
        result[
            "review"
        ] = (
            parse_strong_review_decision(
                str(
                    result[
                        "raw_teacher_output"
                    ]
                )
            )
        )

        result[
            "parse_success"
        ] = True

        return result

    except Exception as exc:
        result[
            "error"
        ] = {
            "stage": (
                "parse_strong_review"
            ),
            "error_type": (
                type(exc).__name__
            ),
            "error": str(exc),
        }
        return result


def execute_strong_review_batch(
    *,
    teacher: Any,
    items: Sequence[
        Mapping[str, Any]
    ],
    max_workers: int,
) -> dict[str, dict[str, Any]]:
    """
    并发 Strong Reviewer Batch。
    """

    if max_workers < 1:
        raise ValueError(
            "max_workers 必须 >= 1"
        )

    output: dict[
        str,
        dict[str, Any],
    ] = {}

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        future_to_task = {
            executor.submit(
                run_one_strong_reviewer,
                teacher=teacher,
                item=item,
            ): str(
                item[
                    "task_id"
                ]
            )
            for item in items
        }

        for future in tqdm(
            as_completed(
                future_to_task
            ),
            total=len(
                future_to_task
            ),
            desc=(
                "DeepSeek review"
                "（DeepSeek 强审）"
            ),
            unit="task",
            dynamic_ncols=True,
        ):
            task_id = (
                future_to_task[
                    future
                ]
            )

            try:
                result = (
                    future.result()
                )
            except Exception as exc:
                result = {
                    "task_id": task_id,
                    "teacher_role": (
                        "strong_semantic_reviewer_deepseek"
                    ),
                    "api_success": False,
                    "parse_success": False,
                    "review": None,
                    "teacher_call": None,
                    "raw_teacher_output": None,
                    "error": {
                        "stage": (
                            "unexpected_future_exception"
                        ),
                        "error_type": (
                            type(exc).__name__
                        ),
                        "error": str(exc),
                    },
                }

            output[
                task_id
            ] = result

    return output


def summarize_strong_review_results(
    results: Sequence[
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    """
    Strong Reviewer 汇总。
    """

    decision_counts: Counter[str] = Counter()
    check_failure_counts: Counter[str] = Counter()

    api_success_count = 0
    api_failure_count = 0
    parse_failure_count = 0

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for result in results:
        if result.get(
            "api_success"
        ):
            api_success_count += 1
        else:
            api_failure_count += 1

        if (
            result.get(
                "api_success"
            )
            and not result.get(
                "parse_success"
            )
        ):
            parse_failure_count += 1

        review = (
            result.get(
                "review"
            )
            or {}
        )

        decision_counts[
            str(
                review.get(
                    "review_decision"
                )
                or "error"
            )
        ] += 1

        for key, value in (
            review.get(
                "checks"
            )
            or {}
        ).items():
            if value is False:
                check_failure_counts[
                    str(key)
                ] += 1

        call = (
            result.get(
                "teacher_call"
            )
            or {}
        )

        prompt_tokens += int(
            call.get(
                "prompt_tokens"
            )
            or 0
        )

        completion_tokens += int(
            call.get(
                "completion_tokens"
            )
            or 0
        )

        total_tokens += int(
            call.get(
                "total_tokens"
            )
            or 0
        )

    return {
        "record_count": len(results),
        "api_success_count": (
            api_success_count
        ),
        "api_failure_count": (
            api_failure_count
        ),
        "parse_failure_count": (
            parse_failure_count
        ),
        "review_decision_counts": dict(
            sorted(
                decision_counts.items()
            )
        ),
        "check_failure_counts": dict(
            sorted(
                check_failure_counts.items()
            )
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": (
            completion_tokens
        ),
        "total_tokens": total_tokens,
    }


def execute_teacher_batch(
    *,
    teacher: Any,
    teacher_role: str,
    items: Sequence[
        Mapping[str, Any]
    ],
    max_workers: int,
) -> dict[
    str,
    dict[str, Any],
]:
    """
    并发执行一个 Teacher Batch（教师批次）。

    max_workers 不能超过对应 Provider Pool 的槽位数。
    """

    if max_workers < 1:
        raise ValueError(
            "max_workers 必须 >= 1"
        )

    output: dict[
        str,
        dict[str, Any],
    ] = {}

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        future_to_task = {
            executor.submit(
                run_one_teacher,
                teacher=teacher,
                teacher_role=(
                    teacher_role
                ),
                item=item,
            ): str(
                item[
                    "task_id"
                ]
            )
            for item in items
        }

        for future in tqdm(
            as_completed(
                future_to_task
            ),
            total=len(
                future_to_task
            ),
            desc=(
                "Qwen primary（Qwen 主教师）"
                if teacher_role.startswith("primary_")
                else "DeepSeek review（DeepSeek 强审）"
            ),
            unit="task",
            dynamic_ncols=True,
        ):
            task_id = (
                future_to_task[
                    future
                ]
            )

            try:
                output[
                    task_id
                ] = (
                    future.result()
                )

            except BaseException as exc:
                # 理论上 run_one_teacher 已经捕获普通 Exception。
                # 这里作为最后一道运行时保险。
                output[
                    task_id
                ] = {
                    "task_id": task_id,
                    "teacher_role": (
                        teacher_role
                    ),
                    "api_success": False,
                    "parse_success": False,
                    "verification_exception": True,
                    "proposal": None,
                    "verification": None,
                    "teacher_call": None,
                    "raw_teacher_output": None,
                    "error": {
                        "stage": (
                            "executor"
                        ),
                        "error_type": (
                            type(exc).__name__
                        ),
                        "error": str(
                            exc
                        ),
                    },
                }

    return output


# ============================================================================
# Quality Gate（质量门）
# ============================================================================


def _normalize_witness_groups(
    obligation: Mapping[str, Any],
) -> tuple[
    tuple[str, ...],
    ...,
]:
    """
    将 OR-of-AND Witness Graph 规范化。

    外层 tuple：
        OR groups

    内层 tuple：
        同一 group 内 AND Evidence
    """

    groups = []

    for group in (
        obligation.get(
            "witness_groups"
        )
        or []
    ):
        evidence_ids = tuple(
            sorted(
                set(
                    map(
                        str,
                        group.get(
                            "evidence_ids"
                        )
                        or [],
                    )
                )
            )
        )

        if evidence_ids:
            groups.append(
                evidence_ids
            )

    return tuple(
        sorted(
            set(
                groups
            )
        )
    )


def accepted_supervision_signature(
    teacher_result: Mapping[str, Any],
) -> dict[str, Any] | None:
    """
    将一个 accepted Teacher 结果转换成可比较的关键监督签名。

    不比较：
        - rationale 自然语言；
        - description 文案；
        - confidence 数值本身；
        - 新增 obligation 的自动 ID。

    比较：
        - assessment；
        - stop_assessment；
        - Certificate Evidence 集；
        - 每种 obligation 的 applicable / required /
          satisfied_by_question；
        - Witness OR-of-AND 结构。

    这样避免两个 Teacher 因描述措辞不同被误判，
    同时仍然对训练真正依赖的结构保持严格。
    """

    verification = (
        teacher_result.get(
            "verification"
        )
        or {}
    )

    if (
        str(
            verification.get(
                "verification_status"
            )
            or ""
        )
        != "accepted"
    ):
        return None

    assessment = str(
        verification.get(
            "assessment"
        )
        or ""
    )

    certificate = tuple(
        sorted(
            set(
                map(
                    str,
                    verification.get(
                        "verified_minimal_certificate_evidence_ids"
                    )
                    or [],
                )
            )
        )
    )

    if assessment == "keep":
        # KEEP 的 refined_obligations 是原始 obligation graph。
        # 两个 Teacher 只需在“保持原监督”及 Certificate 上一致。
        obligation_signature: tuple[
            Any,
            ...
        ] = ()

    else:
        normalized = []

        for obligation in (
            verification.get(
                "refined_obligations"
            )
            or []
        ):
            normalized.append(
                (
                    str(
                        obligation.get(
                            "type"
                        )
                        or ""
                    ),
                    bool(
                        obligation.get(
                            "applicable"
                        )
                    ),
                    bool(
                        obligation.get(
                            "required_for_sufficiency"
                        )
                    ),
                    bool(
                        obligation.get(
                            "satisfied_by_question"
                        )
                    ),
                    _normalize_witness_groups(
                        obligation
                    ),
                )
            )

        obligation_signature = tuple(
            sorted(
                normalized
            )
        )

    return {
        "assessment": (
            assessment
        ),
        "stop_assessment": str(
            verification.get(
                "stop_assessment"
            )
            or ""
        ),
        "certificate": (
            certificate
        ),
        "obligations": (
            obligation_signature
        ),
    }


def teacher_confidence(
    teacher_result: Mapping[str, Any],
) -> float | None:
    proposal = (
        teacher_result.get(
            "proposal"
        )
        or {}
    )

    value = proposal.get(
        "confidence"
    )

    if isinstance(
        value,
        bool,
    ) or not isinstance(
        value,
        (int, float),
    ):
        return None

    return float(
        value
    )


def should_run_strong_review(
    *,
    primary_result: Mapping[str, Any],
    review_policy: str,
    min_primary_confidence: float,
) -> bool:
    """
    决定一个 task 是否需要 DeepSeek Strong Review（强审）。

    v1.8.2.1 关键修正：

        Reviewer 只能审核“已经通过 Deterministic Verification 的
        Primary Proposal”。

    如果 Primary 因：
        API failure
        parse failure
        verification exception / rejection

    没有形成 accepted Primary Canonical Proposal，
    DeepSeek 没有可审核对象。

    此时：
        直接由 Quality Gate 阻断；
        不浪费 Strong Reviewer 调用；
        不让技术失败污染语义审核统计。
    """

    verification = (
        primary_result.get(
            "verification"
        )
        or {}
    )

    status = str(
        verification.get(
            "verification_status"
        )
        or ""
    )

    # 没有 accepted Primary Proposal：
    # 无论 review-policy=all/changed，都不能做“语义审核空对象”。
    if status != "accepted":
        return False

    if review_policy == "all":
        return True

    if review_policy == "none":
        return False

    if review_policy != "changed":
        raise ValueError(
            f"未知 review_policy={review_policy!r}"
        )

    assessment = str(
        verification.get(
            "assessment"
        )
        or ""
    )

    # changed/refine 一律进入强审。
    if assessment != "keep":
        return True

    confidence = (
        teacher_confidence(
            primary_result
        )
    )

    # accepted KEEP 但 Primary 置信不足：
    # 进入 Strong Review。
    if (
        confidence is None
        or confidence
        < min_primary_confidence
    ):
        return True

    # 高置信 accepted KEEP 可以直接 primary_verified_keep。
    return False



def build_quality_gate(
    *,
    split: str,
    primary_result: Mapping[str, Any],
    strong_result: Mapping[str, Any] | None,
    min_primary_confidence: float,
    min_strong_confidence: float,
) -> dict[str, Any]:
    """
    Supervision Quality Gate v1.8（监督质量门）。

    通过等级：

    1. primary_verified_keep
       Qwen 高置信 accepted KEEP，未进入强审。

    2. strong_review_approved
       Qwen Deterministic Verification accepted；
       DeepSeek Reviewer approve；
       Reviewer confidence 达标；
       六项 semantic checks 全部 true。

    reject / uncertain：
        直接 blocked。

    Reviewer 不生成 replacement supervision。
    """

    reasons: list[str] = []

    primary_signature = (
        accepted_supervision_signature(
            primary_result
        )
    )

    primary_confidence = (
        teacher_confidence(
            primary_result
        )
    )

    if primary_signature is None:
        reasons.append(
            "primary_not_accepted"
        )

    primary_is_clean_keep = bool(
        primary_signature
        is not None
        and primary_signature[
            "assessment"
        ]
        == "keep"
        and primary_signature[
            "stop_assessment"
        ]
        == "original_stop_correct"
        and primary_confidence
        is not None
        and primary_confidence
        >= min_primary_confidence
    )

    if (
        primary_signature
        is not None
        and primary_signature[
            "assessment"
        ]
        == "keep"
        and primary_signature[
            "stop_assessment"
        ]
        != "original_stop_correct"
    ):
        reasons.append(
            "primary_keep_stop_inconsistent"
        )

    if strong_result is None:
        # -----------------------------------------------------------
        # 两种“没有 Strong Review”的情况必须区分：
        #
        # A. Primary 本身未 accepted：
        #       没有可审核 Proposal。
        #       primary_not_accepted 已经足够说明阻断原因。
        #
        # B. Primary accepted，但不是高置信 clean KEEP：
        #       按协议本应进入 Strong Review。
        #       如果 Strong Review 没有运行，才记录 routing failure。
        # -----------------------------------------------------------

        if (
            primary_signature
            is not None
            and not primary_is_clean_keep
        ):
            reasons.append(
                "strong_review_required_but_not_run"
            )

        reasons = list(
            dict.fromkeys(reasons)
        )

        verified = (
            len(reasons)
            == 0
        )

        return {
            "supervision_verified": verified,
            "training_eligible": bool(
                verified
                and split
                == "train"
            ),
            "status": (
                "verified"
                if verified
                else "blocked"
            ),
            "verification_tier": (
                "primary_verified_keep"
                if verified
                else "blocked"
            ),
            "block_reasons": (
                []
                if verified
                else reasons
            ),
            "primary_confidence": (
                primary_confidence
            ),
            "strong_confidence": None,
            "strong_review_decision": None,
            "strong_review_checks": None,
            "structure_agreement": None,
            "final_signature": (
                primary_signature
                if verified
                else None
            ),
            "training_split_policy": (
                "eligible only when split=train "
                "and supervision_verified=true"
            ),
        }

    if not bool(
        strong_result.get(
            "api_success"
        )
    ):
        reasons.append(
            "strong_review_api_failed"
        )

    if not bool(
        strong_result.get(
            "parse_success"
        )
    ):
        reasons.append(
            "strong_review_parse_failed"
        )

    review = (
        strong_result.get(
            "review"
        )
        or {}
    )

    decision = str(
        review.get(
            "review_decision"
        )
        or ""
    )

    confidence_raw = (
        review.get(
            "confidence"
        )
    )

    strong_confidence = (
        float(confidence_raw)
        if (
            isinstance(
                confidence_raw,
                (int, float),
            )
            and not isinstance(
                confidence_raw,
                bool,
            )
        )
        else None
    )

    if (
        strong_confidence
        is None
        or strong_confidence
        < min_strong_confidence
    ):
        reasons.append(
            "strong_confidence_below_threshold"
        )

    if decision == "reject":
        reasons.append(
            "strong_review_rejected"
        )
    elif decision == "uncertain":
        reasons.append(
            "strong_review_uncertain"
        )
    elif decision != "approve":
        reasons.append(
            "strong_review_missing_approval"
        )

    checks = (
        review.get(
            "checks"
        )
        or {}
    )

    failed_checks = sorted(
        str(key)
        for key, value in checks.items()
        if value is False
    )

    for key in failed_checks:
        reasons.append(
            "strong_check_failed:"
            + key
        )

    reasons = list(
        dict.fromkeys(reasons)
    )

    verified = (
        len(reasons)
        == 0
    )

    return {
        "supervision_verified": verified,
        "training_eligible": bool(
            verified
            and split
            == "train"
        ),
        "status": (
            "verified"
            if verified
            else "blocked"
        ),
        "verification_tier": (
            "strong_review_approved"
            if verified
            else "blocked"
        ),
        "block_reasons": (
            []
            if verified
            else reasons
        ),
        "primary_confidence": (
            primary_confidence
        ),
        "strong_confidence": (
            strong_confidence
        ),
        "strong_review_decision": (
            decision
            or None
        ),
        "strong_review_checks": (
            checks
            or None
        ),
        "structure_agreement": None,
        "final_signature": (
            primary_signature
            if verified
            else None
        ),
        "training_split_policy": (
            "eligible only when split=train "
            "and supervision_verified=true"
        ),
    }


# ============================================================================
# Output Helpers
# ============================================================================


def write_jsonl(
    path: Path,
    records: Iterable[
        Mapping[str, Any]
    ],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def summarize_teacher_results(
    results: Sequence[
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    """
    汇总一个 Teacher 的 API / parse / verification 状态。
    """

    status_counts: dict[
        str,
        int,
    ] = {}

    for item in results:
        verification = (
            item.get(
                "verification"
            )
            or {}
        )

        status = str(
            verification.get(
                "verification_status"
            )
            or (
                "error"
                if item.get(
                    "error"
                )
                else "unknown"
            )
        )

        status_counts[
            status
        ] = (
            status_counts.get(
                status,
                0,
            )
            + 1
        )

    calls = [
        item.get(
            "teacher_call"
        )
        or {}
        for item in results
        if item.get(
            "api_success"
        )
    ]

    return {
        "record_count": len(
            results
        ),
        "api_success_count": sum(
            bool(
                item.get(
                    "api_success"
                )
            )
            for item in results
        ),
        "api_failure_count": sum(
            not bool(
                item.get(
                    "api_success"
                )
            )
            for item in results
        ),
        "parse_failure_count": sum(
            bool(
                item.get(
                    "api_success"
                )
            )
            and not bool(
                item.get(
                    "parse_success"
                )
            )
            for item in results
        ),
        "binding_failure_count": sum(
            bool(
                item.get(
                    "parse_success"
                )
            )
            and not bool(
                item.get(
                    "binding_success"
                )
            )
            for item in results
        ),
        "verification_exception_count": sum(
            bool(
                item.get(
                    "verification_exception"
                )
            )
            for item in results
        ),
        "verification_status_counts": dict(
            sorted(
                status_counts.items()
            )
        ),
        "prompt_tokens": sum(
            int(
                call.get(
                    "prompt_tokens"
                )
                or 0
            )
            for call in calls
        ),
        "completion_tokens": sum(
            int(
                call.get(
                    "completion_tokens"
                )
                or 0
            )
            for call in calls
        ),
        "total_tokens": sum(
            int(
                call.get(
                    "total_tokens"
                )
                or 0
            )
            for call in calls
        ),
    }


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GLM fixed-slot supervision refinement with "
            "2-of-3 self-consistency and program-built supervision."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=(
            DEFAULT_DATASET_DIR
        ),
    )

    parser.add_argument(
        "--evidence-cache",
        type=Path,
        default=(
            DEFAULT_EVIDENCE_CACHE
        ),
    )

    parser.add_argument(
        "--build-db",
        type=Path,
        default=(
            DEFAULT_BUILD_DB
        ),
        help=(
            "只读完整 build SQLite，用于在 task 自己的 pre-fix snapshot "
            "中补 Gold-guided/original Evidence。"
        ),
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
            "benchmark",
        ],
        default="validation",
    )

    parser.add_argument(
        "--task-scope",
        choices=[
            "all",
            "boundary",
        ],
        default="all",
        help=(
            "all=所有存在 Complete 的任务；"
            "boundary=仅 Boundary+Complete 任务。"
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "audit",
            "export",
            "refine",
        ],
        default="audit",
    )

    parser.add_argument(
        "--task-limit",
        type=int,
        default=(
            DEFAULT_TASK_LIMIT
        ),
        help=(
            "默认 5 条小样本。"
        ),
    )

    parser.add_argument(
        "--allow-full-run",
        action="store_true",
        help=(
            "显式允许处理当前 split + scope 的全部任务。"
        ),
    )

    parser.add_argument(
        "--sample-seed",
        type=int,
        default=(
            DEFAULT_SAMPLE_SEED
        ),
    )

    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=(
            DEFAULT_CANDIDATE_LIMIT
        ),
    )

    parser.add_argument(
        "--candidate-max-per-file",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--candidate-test-quota",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--candidate-doc-quota",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--candidate-resource-quota",
        type=int,
        default=1,
        help=(
            "普通 data/resource 候选的全任务配额；"
            "Issue/Gold 明确命中的资源文件不受此普通配额限制。"
        ),
    )

    parser.add_argument(
        "--candidate-low-value-quota",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--candidate-overlap-threshold",
        type=float,
        default=0.65,
    )

    parser.add_argument(
        "--gold-units-per-hunk",
        type=int,
        choices=[
            1,
            2,
        ],
        default=1,
        help=(
            "每个 Gold hunk 选择的最佳 pre-fix Evidence 数；"
            "默认 Best-1，最多 Best-2。"
        ),
    )

    parser.add_argument(
        "--max-gold-units-per-file",
        type=int,
        default=8,
        help=(
            "一个 changed file 经 hunk Best-1/2 后最多保留多少 Gold-guided Evidence。"
        ),
    )

    parser.add_argument(
        "--issue-symbol-units-per-symbol",
        type=int,
        default=1,
        help=(
            "Issue 明确 symbol 每个默认保留一个最佳 pre-fix Evidence。"
        ),
    )

    parser.add_argument(
        "--issue-symbol-policy-path-limit",
        type=int,
        default=6,
        help=(
            "Issue symbol 除 Gold/Issue path 外，最多额外搜索多少个高排名 policy source path。"
        ),
    )

    parser.add_argument(
        "--max-teacher-question-chars",
        type=int,
        default=(
            DEFAULT_MAX_TEACHER_QUESTION_CHARS
        ),
        help=(
            "仅当完整 Prompt 超过 --max-prompt-chars 时，"
            "才把 Teacher Question 确定性压缩到该字符目标；"
            "Candidate Builder 始终使用完整 Question。"
        ),
    )

    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=(
            DEFAULT_MAX_PROMPT_CHARS
        ),
    )

    parser.add_argument(
        "--reference-mode",
        choices=[
            "gold",
            "none",
        ],
        default="gold",
        help=(
            "gold=只传确定性 Gold Change Hints；"
            "完整 Gold/Test Patch 正文不会发送给 Teacher。"
        ),
    )

    # ----------------------------------------------------------------
    # Strong Review 策略。
    #
    # 默认 changed：
    #     高置信 accepted KEEP 不调用 DeepSeek；
    #     修改、低置信、异常任务才进入强审。
    # ----------------------------------------------------------------

    parser.add_argument(
        "--review-policy",
        choices=[
            "all",
            "changed",
            "none",
        ],
        default="none",
        help=(
            "v1.9 正式路径不使用 DeepSeek；必须为 none。"
        ),
    )

    parser.add_argument(
        "--min-primary-confidence",
        type=float,
        default=(
            DEFAULT_MIN_PRIMARY_CONFIDENCE
        ),
    )

    parser.add_argument(
        "--min-strong-confidence",
        type=float,
        default=(
            DEFAULT_MIN_STRONG_CONFIDENCE
        ),
    )

    # ----------------------------------------------------------------
    # Primary Teacher Provider（主教师服务商）。
    #
    # 默认仍是 siliconflow，保证旧命令行为不变。
    # ----------------------------------------------------------------

    parser.add_argument(
        "--primary-provider",
        choices=[
            "siliconflow",
            "bigmodel",
        ],
        default="bigmodel",
        help=(
            "v1.9 固定决策协议当前冻结为 bigmodel=GLM-4.7-Flash。"
        ),
    )

    # ----------------------------------------------------------------
    # SiliconFlow/Qwen3。
    # 默认优先读取 .env。
    # ----------------------------------------------------------------

    parser.add_argument(
        "--siliconflow-base-url",
        default=None,
        help=(
            "默认读取 OPENAI_BASE_URL，"
            "否则使用 https://api.siliconflow.cn/v1"
        ),
    )

    parser.add_argument(
        "--siliconflow-model",
        default=None,
        help=(
            "默认读取 LLM_MODEL，"
            "否则使用 Qwen/Qwen3-8B"
        ),
    )

    parser.add_argument(
        "--siliconflow-per-key-concurrency",
        type=int,
        default=4,
        help=(
            "每个 OPENAI_API_KEY* 最大并发，默认 4。"
        ),
    )

    parser.add_argument(
        "--siliconflow-max-tokens",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--siliconflow-timeout",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--siliconflow-max-retries",
        type=int,
        default=4,
    )

    # ----------------------------------------------------------------
    # BigModel / GLM Primary Teacher。
    #
    # 默认读取：
    #     BIGMOD_API_URL
    #     BIGMOD_API_MODEL
    #     BIGMOD_API_KEY（由 refinement_teacher.py 读取）
    # ----------------------------------------------------------------

    parser.add_argument(
        "--bigmodel-base-url",
        default=None,
        help=(
            "默认读取 BIGMOD_API_URL，"
            "否则使用 https://open.bigmodel.cn/api/paas/v4/"
        ),
    )

    parser.add_argument(
        "--bigmodel-model",
        default=None,
        help=(
            "默认读取 BIGMOD_API_MODEL，"
            "否则使用 glm-4.7-flash。"
        ),
    )

    parser.add_argument(
        "--bigmodel-concurrency",
        type=int,
        default=1,
        help=(
            "BigModel 每个 API Key 的并发槽位，默认 1。"
            "例如发现 5 个 BIGMOD_API_KEY* 时，总并发=5。"
            "不建议提高单 Key 并发。"
        ),
    )

    parser.add_argument(
        "--bigmodel-global-concurrency",
        type=int,
        default=3,
        help=(
            "BigModel 所有 API Key 合计的全局并发上限，默认 3。"
            "BIGMOD_API_KEY* 仍全部可用，但同时最多发送 3 个请求。"
            "用于降低 1302/1305 RateLimit（速率限制）概率。"
        ),
    )

    parser.add_argument(
        "--bigmodel-max-tokens",
        type=int,
        default=3000,
        help=(
            "默认 3000，与 Qwen primary 保持公平 A/B。"
        ),
    )

    parser.add_argument(
        "--bigmodel-timeout",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--bigmodel-max-retries",
        type=int,
        default=6,
        help=(
            "BigModel 429/5xx 最大重试次数，默认 6。"
        ),
    )

    # ----------------------------------------------------------------
    # DeepSeek Strong Review。
    # ----------------------------------------------------------------

    parser.add_argument(
        "--deepseek-model",
        default=(
            "deepseek-v4-flash"
        ),
    )

    parser.add_argument(
        "--deepseek-base-url",
        default=(
            "https://api.deepseek.com"
        ),
    )

    parser.add_argument(
        "--deepseek-reasoning-effort",
        default="high",
    )

    parser.add_argument(
        "--deepseek-max-tokens",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--deepseek-timeout",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--deepseek-max-retries",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--deepseek-concurrency",
        type=int,
        default=(
            DEFAULT_DEEPSEEK_CONCURRENCY
        ),
        help=(
            "小样本默认 8；不要因为服务端上限高就立即开到 500。"
        ),
    )

    # ----------------------------------------------------------------
    # v1.9 2-of-3 Consensus（3次取2次共识）
    # ----------------------------------------------------------------

    parser.add_argument(
        "--min-consensus-confidence",
        type=float,
        default=0.85,
        help=(
            "兼容旧命令保留。v1.9.2 中 GLM 自报 confidence 仅用于审计，"
            "不再作为硬准入门。"
        ),
    )

    parser.add_argument(
        "--promote-consensus-training",
        action="store_true",
        help=(
            "显式允许 train split 中通过 v1.9 共识门的样本进入 training_eligible。"
            "validation 校准阶段不要开启。"
        ),
    )

    # Benchmark 保护。
    parser.add_argument(
        "--allow-benchmark",
        action="store_true",
        help=(
            "显式允许处理 benchmark。"
        ),
    )

    parser.add_argument(
        "--protocol-frozen",
        action="store_true",
        help=(
            "benchmark refine 时必须显式声明协议已冻结。"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            DEFAULT_OUTPUT_DIR
        ),
    )

    return parser


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    args = (
        build_parser()
        .parse_args()
    )

    print(
        json.dumps(
            {
                "runner_version": RUNNER_VERSION,
                "runner_path": str(
                    SCRIPT_PATH
                ),
                "review_policy": (
                    "none_deepseek_removed_v1_9_2"
                ),
                "primary_provider": (
                    args.primary_provider
                ),
                "consensus_policy": (
                    "two_stage_requirement_then_targeted_witness"
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.primary_provider != "bigmodel":
        raise ValueError(
            "v1.9.2 Two-Stage Consensus 当前只支持 --primary-provider bigmodel"
        )

    if args.review_policy != "none":
        raise ValueError(
            "v1.9.2 正式路径不调用 DeepSeek，--review-policy 必须为 none"
        )

    if not (0.0 <= args.min_consensus_confidence <= 1.0):
        raise ValueError(
            "--min-consensus-confidence 必须在 [0,1]"
        )

    if args.bigmodel_global_concurrency < 1:
        raise ValueError(
            "--bigmodel-global-concurrency 必须 >= 1"
        )

    if args.task_limit < 1:
        raise ValueError(
            "--task-limit 必须 >= 1"
        )

    if args.candidate_limit < 1:
        raise ValueError(
            "--candidate-limit 必须 >= 1"
        )

    if args.candidate_max_per_file < 1:
        raise ValueError(
            "--candidate-max-per-file 必须 >= 1"
        )

    for name, value in (
        ("--candidate-test-quota", args.candidate_test_quota),
        ("--candidate-doc-quota", args.candidate_doc_quota),
        ("--candidate-resource-quota", args.candidate_resource_quota),
        ("--candidate-low-value-quota", args.candidate_low_value_quota),
        ("--issue-symbol-policy-path-limit", args.issue_symbol_policy_path_limit),
    ):
        if value < 0:
            raise ValueError(
                f"{name} 不能为负数"
            )

    if not 0.0 <= args.candidate_overlap_threshold <= 1.0:
        raise ValueError(
            "--candidate-overlap-threshold 必须在 [0,1]"
        )

    if args.max_gold_units_per_file < 1:
        raise ValueError(
            "--max-gold-units-per-file 必须 >= 1"
        )

    if args.issue_symbol_units_per_symbol < 1:
        raise ValueError(
            "--issue-symbol-units-per-symbol 必须 >= 1"
        )

    if args.max_teacher_question_chars < 4_000:
        raise ValueError(
            "--max-teacher-question-chars 必须 >= 4000"
        )

    if args.max_prompt_chars < 10_000:
        raise ValueError(
            "--max-prompt-chars 过小"
        )

    if not (
        0.0
        <= args.min_primary_confidence
        <= 1.0
    ):
        raise ValueError(
            "--min-primary-confidence 必须在 [0,1]"
        )

    if not (
        0.0
        <= args.min_strong_confidence
        <= 1.0
    ):
        raise ValueError(
            "--min-strong-confidence 必须在 [0,1]"
        )

    if args.deepseek_concurrency < 1:
        raise ValueError(
            "--deepseek-concurrency 必须 >= 1"
        )

    if (
        args.split == "benchmark"
        and args.mode == "refine"
        and not (
            args.allow_benchmark
            and args.protocol_frozen
        )
    ):
        raise ValueError(
            "Benchmark refine 必须同时显式传："
            "--allow-benchmark --protocol-frozen。"
            "协议冻结前不要用 benchmark 调参。"
        )

    dataset_dir = (
        args.dataset_dir.resolve()
    )

    (
        manifest_path,
        manifest,
    ) = (
        load_and_validate_manifest(
            dataset_dir
        )
    )

    split_path = (
        resolve_split_path(
            dataset_dir,
            args.split,
        )
    )

    (
        selected_rows,
        scope_task_count,
    ) = (
        select_tasks(
            split_path=(
                split_path
            ),
            task_scope=(
                args.task_scope
            ),
            seed=(
                args.sample_seed
            ),
            task_limit=(
                args.task_limit
            ),
            allow_full_run=(
                args.allow_full_run
            ),
        )
    )

    output_dir = (
        args.output_dir.resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    request_path = (
        output_dir
        / "refinement_requests.jsonl"
    )

    refinement_path = (
        output_dir
        / "refinement.jsonl"
    )

    errors_path = (
        output_dir
        / "refinement_errors.jsonl"
    )

    report_path = (
        output_dir
        / "refinement_report.json"
    )

    # ----------------------------------------------------------------
    # Prepare 阶段单线程读取 SQLite。
    #
    # SQLite 连接对象默认不应跨线程共享。
    # 所以先准备 Prompt，再关闭 Cache，再并发调用 API。
    # ----------------------------------------------------------------

    cache = EvidenceCache(
        args.evidence_cache
    )

    build_store = BuildEvidenceStore(
        args.build_db
    )

    candidate_config = CandidateBuilderConfig(
        candidate_limit=(
            args.candidate_limit
        ),
        max_per_file=(
            args.candidate_max_per_file
        ),
        test_quota=(
            args.candidate_test_quota
        ),
        doc_quota=(
            args.candidate_doc_quota
        ),
        resource_quota=(
            args.candidate_resource_quota
        ),
        low_value_quota=(
            args.candidate_low_value_quota
        ),
        overlap_threshold=(
            args.candidate_overlap_threshold
        ),
        gold_units_per_hunk=(
            args.gold_units_per_hunk
        ),
        max_gold_units_per_file=(
            args.max_gold_units_per_file
        ),
        issue_symbol_units_per_symbol=(
            args.issue_symbol_units_per_symbol
        ),
        issue_symbol_policy_path_limit=(
            args.issue_symbol_policy_path_limit
        ),
    )
    candidate_config.validate()

    prepared: list[
        dict[str, Any]
    ] = []

    preparation_errors: list[
        dict[str, Any]
    ] = []

    try:
        for row in tqdm(
            selected_rows,
            total=len(selected_rows),
            desc="Prepare candidates（准备候选）",
            unit="task",
            dynamic_ncols=True,
        ):
            task_id = str(
                row.get(
                    "task_id"
                )
                or ""
            )

            try:
                item = prepare_task_payload(
                    task_row=row,
                    cache=cache,
                    build_store=build_store,
                    candidate_config=(
                        candidate_config
                    ),
                    reference_mode=(
                        args.reference_mode
                    ),
                    max_prompt_chars=(
                        args.max_prompt_chars
                    ),
                    max_teacher_question_chars=(
                        args.max_teacher_question_chars
                    ),
                )

                prepared.append(
                    item
                )

            except Exception as exc:
                preparation_errors.append(
                    {
                        "task_id": task_id,
                        "stage": "prepare",
                        "error_type": (
                            type(exc).__name__
                        ),
                        "error": str(exc),
                    }
                )

    finally:
        cache.close()
        build_store.close()

    # ----------------------------------------------------------------
    # Request 审计文件：
    # 让你可以精确检查实际发给模型的 Prompt。
    # ----------------------------------------------------------------

    request_records = [
        {
            "task_id": (
                item[
                    "task_id"
                ]
            ),
            "has_boundary": (
                item[
                    "has_boundary"
                ]
            ),
            "requirement_system_prompt": (
                REQUIREMENT_DECISION_SYSTEM_PROMPT
            ),
            "witness_system_prompt": (
                WITNESS_SELECTION_SYSTEM_PROMPT
            ),
            "base_user_prompt": (
                item[
                    "user_prompt"
                ]
            ),
            "base_prompt_chars": (
                item[
                    "prompt_chars"
                ]
            ),
            "candidate_count": len(
                item[
                    "candidate_ids"
                ]
            ),
            "candidate_diagnostics": (
                item[
                    "candidate_metadata"
                ]
            ),
            "offline_gold_reference_used": (
                item[
                    "offline_gold_reference_used"
                ]
            ),
        }
        for item in prepared
    ]

    write_jsonl(
        request_path,
        request_records,
    )

    # audit / export 不调用 API。
    primary_results_by_task: dict[
        str,
        dict[str, Any],
    ] = {}

    strong_results_by_task: dict[
        str,
        dict[str, Any],
    ] = {}

    primary_environment = None
    primary_protocol = None
    strong_environment = None
    strong_protocol = None

    started = (
        time.perf_counter()
    )

    # =====================================================================
    # v1.9.2 run-level stores
    # =====================================================================

    requirement_a_by_task: dict[str, dict[str, Any]] = {}
    requirement_b_by_task: dict[str, dict[str, Any]] = {}
    requirement_c_by_task: dict[str, dict[str, Any]] = {}
    requirement_consensus_by_task: dict[str, dict[str, Any]] = {}

    witness_runs_by_task: dict[
        str,
        dict[str, dict[str, dict[str, Any]]],
    ] = {}

    witness_consensus_by_task: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {}

    final_consensus_by_task: dict[str, dict[str, Any]] = {}

    if args.mode == "refine":
        from refinement_teacher import load_project_dotenv
        load_project_dotenv()

        import os

        bigmodel_base_url = (
            args.bigmodel_base_url
            or os.getenv("BIGMOD_API_URL")
            or "https://open.bigmodel.cn/api/paas/v4/"
        )

        bigmodel_model = str(
            args.bigmodel_model
            or os.getenv("BIGMOD_API_MODEL")
            or "glm-4.7-flash"
        ).strip().lower()

        primary_config = BigModelTeacherConfig(
            model=bigmodel_model,
            base_url=bigmodel_base_url,
            per_key_concurrency=args.bigmodel_concurrency,
            thinking_type="disabled",
            do_sample=False,
            max_tokens=args.bigmodel_max_tokens,
            timeout_seconds=args.bigmodel_timeout,
            max_retries=args.bigmodel_max_retries,
        )

        primary_environment = validate_bigmodel_environment(
            primary_config
        )

        primary_protocol = {
            **bigmodel_teacher_protocol_metadata(
                primary_config
            ),
            "protocol_version": (
                "supervision-refinement-v1.9.2.1"
            ),
            "decision_schema": (
                "two-stage-requirement-plus-targeted-witness-v1.9.2"
            ),
            "stage_1": (
                "7 fixed semantic slots; "
                "repository_required / question_satisfied / not_required / uncertain; "
                "2-of-3 decision majority"
            ),
            "stage_1_candidate_visibility": (
                "candidate_pool_hidden; requirement classification must be "
                "independent of current candidate availability"
            ),
            "candidate_insufficiency_owner": (
                "Stage 2 witness status=insufficient"
            ),
            "stage_2": (
                "only repository_required slots; targeted Candidate Number "
                "OR-of-AND Witness selection; status + group 2-of-3 consensus"
            ),
            "teacher_confidence_policy": (
                "diagnostic_only_not_a_hard_gate"
            ),
            "independent_semantic_review": False,
            "program_owned_fields": [
                "Evidence ID mapping",
                "obligation type",
                "source_obligation_id",
                "assessment",
                "stop_assessment",
                "minimal certificate",
            ],
        }

        primary_teacher = BigModelRefinementTeacherPool(
            primary_config
        )

        # -------------------------------------------------------------
        # Global Concurrency Cap（全局并发上限）
        #
        # Teacher Pool 仍然发现全部 BIGMOD_API_KEY*，
        # 但 executor 不再无条件使用 slot_count。
        #
        # 本轮 20 条实验中 Stage 1 出现 5 个 429，
        # 包括 1302 account rate limit 与 1305 model busy。
        # 因此把“Key 数量”和“请求并发”解耦。
        # -------------------------------------------------------------

        effective_bigmodel_workers = min(
            primary_teacher.slot_count,
            args.bigmodel_global_concurrency,
        )

        # -------------------------------------------------------------
        # Stage 1: Requirement Decision A/B
        # -------------------------------------------------------------

        requirement_a_by_task = execute_requirement_batch(
            teacher=primary_teacher,
            items=prepared,
            run_label="A",
            max_workers=effective_bigmodel_workers,
        )

        requirement_b_by_task = execute_requirement_batch(
            teacher=primary_teacher,
            items=prepared,
            run_label="B",
            max_workers=effective_bigmodel_workers,
        )

        requirement_tie_items = []

        for item in prepared:
            task_id = str(
                item["task_id"]
            )

            if _requirement_ab_needs_c(
                run_a=requirement_a_by_task[
                    task_id
                ],
                run_b=requirement_b_by_task[
                    task_id
                ],
            ):
                requirement_tie_items.append(
                    item
                )

        if requirement_tie_items:
            requirement_c_by_task = execute_requirement_batch(
                teacher=primary_teacher,
                items=requirement_tie_items,
                run_label="C",
                max_workers=effective_bigmodel_workers,
            )

        for item in tqdm(
            prepared,
            total=len(prepared),
            desc="Requirement consensus（需求共识）",
            unit="task",
            dynamic_ncols=True,
        ):
            task_id = str(
                item["task_id"]
            )

            requirement_consensus_by_task[
                task_id
            ] = build_requirement_consensus(
                run_a=requirement_a_by_task[
                    task_id
                ],
                run_b=requirement_b_by_task[
                    task_id
                ],
                run_c=requirement_c_by_task.get(
                    task_id
                ),
            )

        # -------------------------------------------------------------
        # Stage 2: build (task, slot) units only for repository_required
        # -------------------------------------------------------------

        witness_units = []

        for item in prepared:
            task_id = str(
                item["task_id"]
            )

            requirement_consensus = (
                requirement_consensus_by_task[
                    task_id
                ]
            )

            if (
                requirement_consensus.get(
                    "status"
                )
                != "agreed"
            ):
                continue

            slot_results = (
                requirement_consensus.get(
                    "slot_results"
                )
                or {}
            )

            for slot_type in FIXED_DECISION_SLOT_TYPES:
                result = (
                    slot_results[
                        slot_type
                    ]
                )

                if (
                    result.get(
                        "decision"
                    )
                    == "repository_required"
                ):
                    witness_units.append(
                        (
                            task_id,
                            slot_type,
                            item,
                            result,
                        )
                    )

        def execute_witness_units(
            *,
            units: Sequence[
                tuple[
                    str,
                    str,
                    Mapping[str, Any],
                    Mapping[str, Any],
                ]
            ],
            run_label: str,
        ) -> dict[
            tuple[str, str],
            dict[str, Any],
        ]:
            descriptions = {
                "A": "Witness A（证据选择A）",
                "B": "Witness B（证据选择B）",
                "C": "Witness C（证据决胜C）",
            }

            output: dict[
                tuple[str, str],
                dict[str, Any],
            ] = {}

            with ThreadPoolExecutor(
                max_workers=effective_bigmodel_workers
            ) as executor:
                future_to_key = {
                    executor.submit(
                        run_one_witness_selection,
                        teacher=primary_teacher,
                        item=item,
                        slot_type=slot_type,
                        requirement_result=(
                            requirement_result
                        ),
                        run_label=run_label,
                    ): (
                        task_id,
                        slot_type,
                    )
                    for (
                        task_id,
                        slot_type,
                        item,
                        requirement_result,
                    ) in units
                }

                for future in tqdm(
                    as_completed(
                        future_to_key
                    ),
                    total=len(
                        future_to_key
                    ),
                    desc=descriptions[
                        run_label
                    ],
                    unit="slot",
                    dynamic_ncols=True,
                ):
                    key = future_to_key[
                        future
                    ]

                    try:
                        output[key] = (
                            future.result()
                        )
                    except BaseException as exc:
                        output[key] = {
                            "task_id": key[0],
                            "slot_type": key[1],
                            "run_label": run_label,
                            "stage": (
                                "targeted_witness_selection"
                            ),
                            "api_success": False,
                            "parse_success": False,
                            "selection": None,
                            "teacher_call": None,
                            "raw_teacher_output": None,
                            "error": {
                                "stage": (
                                    "witness_executor"
                                ),
                                "error_type": (
                                    type(exc).__name__
                                ),
                                "error": str(exc),
                            },
                        }

            return output

        witness_a = (
            execute_witness_units(
                units=witness_units,
                run_label="A",
            )
            if witness_units
            else {}
        )

        witness_b = (
            execute_witness_units(
                units=witness_units,
                run_label="B",
            )
            if witness_units
            else {}
        )

        witness_tie_units = []

        for unit in witness_units:
            (
                task_id,
                slot_type,
                _item,
                _requirement_result,
            ) = unit

            key = (
                task_id,
                slot_type,
            )

            a = witness_a[
                key
            ].get(
                "selection"
            )

            b = witness_b[
                key
            ].get(
                "selection"
            )

            if (
                not isinstance(
                    a,
                    dict,
                )
                or not isinstance(
                    b,
                    dict,
                )
                or witness_selection_signature(
                    a
                )
                != witness_selection_signature(
                    b
                )
            ):
                witness_tie_units.append(
                    unit
                )

        witness_c = (
            execute_witness_units(
                units=witness_tie_units,
                run_label="C",
            )
            if witness_tie_units
            else {}
        )

        for unit in witness_units:
            (
                task_id,
                slot_type,
                _item,
                _requirement_result,
            ) = unit

            key = (
                task_id,
                slot_type,
            )

            run_a = witness_a[
                key
            ]
            run_b = witness_b[
                key
            ]
            run_c = witness_c.get(
                key
            )

            witness_runs_by_task.setdefault(
                task_id,
                {},
            )[
                slot_type
            ] = {
                "A": run_a,
                "B": run_b,
                "C": run_c,
            }

            witness_consensus_by_task.setdefault(
                task_id,
                {},
            )[
                slot_type
            ] = build_witness_consensus(
                run_a=run_a,
                run_b=run_b,
                run_c=run_c,
            )

        # -------------------------------------------------------------
        # Merge Stage 1 + Stage 2 and build supervision programmatically
        # -------------------------------------------------------------

        for item in tqdm(
            prepared,
            total=len(prepared),
            desc="Two-stage gate（两阶段合并）",
            unit="task",
            dynamic_ncols=True,
        ):
            task_id = str(
                item["task_id"]
            )

            requirement_consensus = (
                requirement_consensus_by_task[
                    task_id
                ]
            )

            witness_consensus = (
                witness_consensus_by_task.get(
                    task_id,
                    {},
                )
            )

            final_consensus = (
                build_two_stage_final_consensus(
                    requirement_consensus=(
                        requirement_consensus
                    ),
                    witness_consensus_by_slot=(
                        witness_consensus
                    ),
                )
            )

            final_consensus_by_task[
                task_id
            ] = final_consensus

            (
                proposal,
                verification,
                construction_error,
            ) = build_programmatic_refinement_v1_9_2(
                task_id=task_id,
                supervision=item[
                    "supervision"
                ],
                candidate_records=item[
                    "candidate_records"
                ],
                candidate_ids=item[
                    "candidate_ids"
                ],
                existing_evidence_ids=set(
                    map(
                        str,
                        item[
                            "candidate_ids"
                        ],
                    )
                ),
                token_costs=item[
                    "token_costs"
                ],
                final_consensus=(
                    final_consensus
                ),
            )

            primary_results_by_task[
                task_id
            ] = {
                "task_id": task_id,
                "teacher_role": (
                    "glm_two_stage_consensus_v1_9_2_1"
                ),
                "requirement_runs": {
                    "A": requirement_a_by_task[
                        task_id
                    ],
                    "B": requirement_b_by_task[
                        task_id
                    ],
                    "C": requirement_c_by_task.get(
                        task_id
                    ),
                },
                "requirement_consensus": (
                    requirement_consensus
                ),
                "witness_runs": (
                    witness_runs_by_task.get(
                        task_id,
                        {},
                    )
                ),
                "witness_consensus": (
                    witness_consensus
                ),
                "consensus": (
                    final_consensus
                ),
                "proposal": proposal,
                "verification": verification,
                "verification_exception": (
                    construction_error
                    is not None
                ),
                "error": (
                    construction_error
                ),
            }

    elapsed = (
        time.perf_counter()
        - started
    )

    # ----------------------------------------------------------------
    # 合并成“一任务一记录”。
    # 即使某个 Teacher 失败，该 task 也不会从 refinement.jsonl 消失。
    # ----------------------------------------------------------------

    refinement_records: list[
        dict[str, Any]
    ] = []

    runtime_errors: list[
        dict[str, Any]
    ] = []

    if args.mode == "refine":
        for item in tqdm(
            prepared,
            total=len(prepared),
            desc="Quality gate（质量门）",
            unit="task",
            dynamic_ncols=True,
        ):
            task_id = str(
                item["task_id"]
            )

            primary_result = (
                primary_results_by_task[
                    task_id
                ]
            )

            final_consensus = (
                primary_result.get(
                    "consensus"
                )
                or {}
            )

            verification = (
                primary_result.get(
                    "verification"
                )
                or {}
            )

            quality_gate = (
                build_v1_9_2_quality_gate(
                    split=args.split,
                    final_consensus=(
                        final_consensus
                    ),
                    verification=(
                        verification
                    ),
                    construction_error=(
                        primary_result.get(
                            "error"
                        )
                    ),
                    promote_consensus_training=(
                        args
                        .promote_consensus_training
                    ),
                )
            )

            consensus_supervision_candidate = None
            final_supervision = None

            if (
                str(
                    verification.get(
                        "verification_status"
                    )
                    or ""
                )
                == "accepted"
                and final_consensus.get(
                    "status"
                )
                == "agreed"
            ):
                consensus_supervision_candidate = {
                    "assessment": (
                        verification.get(
                            "assessment"
                        )
                    ),
                    "stop_assessment": (
                        verification.get(
                            "stop_assessment"
                        )
                    ),
                    "refined_obligations": (
                        verification.get(
                            "refined_obligations"
                        )
                    ),
                    "certificate_evidence_ids": (
                        verification.get(
                            "verified_minimal_certificate_evidence_ids"
                        )
                    ),
                    "quality_source": (
                        "glm_two_stage_consensus_v1_9_2_1"
                    ),
                    "independent_semantic_review": False,
                }

            if (
                quality_gate[
                    "supervision_verified"
                ]
            ):
                final_supervision = (
                    consensus_supervision_candidate
                )

            record = {
                "task_id": task_id,
                "dataset_version": (
                    EXPECTED_DATASET_VERSION
                ),
                "split": args.split,
                "task_scope": (
                    args.task_scope
                ),
                "has_boundary": (
                    item[
                        "has_boundary"
                    ]
                ),
                "reference_mode": (
                    args.reference_mode
                ),
                "candidate_pool": (
                    item[
                        "candidate_metadata"
                    ]
                ),
                "primary_teacher": (
                    primary_result
                ),
                "requirement_consensus": (
                    primary_result.get(
                        "requirement_consensus"
                    )
                ),
                "witness_consensus": (
                    primary_result.get(
                        "witness_consensus"
                    )
                ),
                "consensus": (
                    final_consensus
                ),
                "strong_review": None,
                "quality_gate": (
                    quality_gate
                ),
                "consensus_supervision_candidate": (
                    consensus_supervision_candidate
                ),
                "final_supervision": (
                    final_supervision
                ),
            }

            refinement_records.append(
                record
            )

            # Stage 1 API / parse errors.
            for (
                run_label,
                run_result,
            ) in (
                (
                    primary_result.get(
                        "requirement_runs"
                    )
                    or {}
                ).items()
            ):
                if (
                    run_result is not None
                    and run_result.get(
                        "error"
                    )
                ):
                    runtime_errors.append({
                        "task_id": task_id,
                        "teacher_role": (
                            f"glm_requirement_{run_label}"
                        ),
                        **run_result[
                            "error"
                        ],
                    })

            # Stage 2 API / parse errors.
            for (
                slot_type,
                slot_runs,
            ) in (
                (
                    primary_result.get(
                        "witness_runs"
                    )
                    or {}
                ).items()
            ):
                for (
                    run_label,
                    run_result,
                ) in (
                    (
                        slot_runs
                        or {}
                    ).items()
                ):
                    if (
                        run_result is not None
                        and run_result.get(
                            "error"
                        )
                    ):
                        runtime_errors.append({
                            "task_id": task_id,
                            "slot_type": (
                                slot_type
                            ),
                            "teacher_role": (
                                f"glm_witness_{run_label}"
                            ),
                            **run_result[
                                "error"
                            ],
                        })

            if (
                primary_result.get(
                    "error"
                )
            ):
                runtime_errors.append({
                    "task_id": task_id,
                    "teacher_role": (
                        "programmatic_supervision_v1_9_2"
                    ),
                    **primary_result[
                        "error"
                    ],
                })

    write_jsonl(
        refinement_path,
        (
            refinement_records
            if args.mode == "refine"
            else []
        ),
    )

    all_errors = [
        *preparation_errors,
        *runtime_errors,
    ]

    write_jsonl(
        errors_path,
        all_errors,
    )

    primary_results = list(
        primary_results_by_task.values()
    )

    strong_results = list(
        strong_results_by_task.values()
    )

    if args.mode == "refine":
        requirement_values = list(
            requirement_consensus_by_task.values()
        )

        witness_values = [
            (
                task_id,
                slot_type,
                value,
            )
            for task_id, by_slot
            in witness_consensus_by_task.items()
            for slot_type, value
            in by_slot.items()
        ]

        final_values = list(
            final_consensus_by_task.values()
        )

        def summarize_requirement_runs(
            values: Sequence[
                Mapping[str, Any]
            ],
        ) -> dict[str, Any]:
            return {
                "record_count": len(values),
                "api_success_count": sum(
                    bool(
                        value.get(
                            "api_success"
                        )
                    )
                    for value in values
                ),
                "api_failure_count": sum(
                    not bool(
                        value.get(
                            "api_success"
                        )
                    )
                    for value in values
                ),
                "parse_success_count": sum(
                    bool(
                        value.get(
                            "parse_success"
                        )
                    )
                    for value in values
                ),
                "parse_failure_count": sum(
                    bool(
                        value.get(
                            "api_success"
                        )
                    )
                    and not bool(
                        value.get(
                            "parse_success"
                        )
                    )
                    for value in values
                ),
                "invalid_slot_vote_count": sum(
                    len(
                        (
                            value.get(
                                "decision_table"
                            )
                            or {}
                        ).get(
                            "slot_errors"
                        )
                        or []
                    )
                    for value in values
                ),
            }

        requirement_slot_consensus_counts = {
            slot_type: sum(
                (
                    (
                        value.get(
                            "slot_results"
                        )
                        or {}
                    ).get(
                        slot_type,
                        {},
                    ).get(
                        "status"
                    )
                    == "agreed"
                )
                for value in requirement_values
            )
            for slot_type in FIXED_DECISION_SLOT_TYPES
        }

        requirement_decision_counts = {
            slot_type: dict(
                sorted(
                    Counter(
                        str(
                            (
                                (
                                    value.get(
                                        "slot_results"
                                    )
                                    or {}
                                ).get(
                                    slot_type,
                                    {},
                                ).get(
                                    "decision"
                                )
                                or "no_consensus"
                            )
                        )
                        for value in requirement_values
                    ).items()
                )
            )
            for slot_type in FIXED_DECISION_SLOT_TYPES
        }

        witness_target_counts = {
            slot_type: sum(
                current_slot == slot_type
                for _task_id, current_slot, _value
                in witness_values
            )
            for slot_type in FIXED_DECISION_SLOT_TYPES
        }

        witness_consensus_counts = {
            slot_type: sum(
                (
                    current_slot == slot_type
                    and value.get(
                        "status"
                    )
                    == "agreed"
                )
                for _task_id, current_slot, value
                in witness_values
            )
            for slot_type in FIXED_DECISION_SLOT_TYPES
        }

        primary_summary = {
            "protocol": (
                "glm_two_stage_consensus_v1.9.2.1"
            ),
            "stage_1_requirement": {
                "run_a": summarize_requirement_runs(
                    list(
                        requirement_a_by_task.values()
                    )
                ),
                "run_b": summarize_requirement_runs(
                    list(
                        requirement_b_by_task.values()
                    )
                ),
                "run_c_tie_break": summarize_requirement_runs(
                    list(
                        requirement_c_by_task.values()
                    )
                ),
                "tie_break_task_count": len(
                    requirement_c_by_task
                ),
                "all_slot_consensus_count": sum(
                    value.get(
                        "status"
                    )
                    == "agreed"
                    for value in requirement_values
                ),
                "no_consensus_count": sum(
                    value.get(
                        "status"
                    )
                    != "agreed"
                    for value in requirement_values
                ),
                "slot_consensus_counts": (
                    requirement_slot_consensus_counts
                ),
                "decision_counts_by_slot": (
                    requirement_decision_counts
                ),
            },
            "stage_2_witness": {
                "target_slot_count": len(
                    witness_values
                ),
                "target_counts_by_slot": (
                    witness_target_counts
                ),
                "consensus_counts_by_slot": (
                    witness_consensus_counts
                ),
                "agreed_target_count": sum(
                    value.get(
                        "status"
                    )
                    == "agreed"
                    for _task_id, _slot_type, value
                    in witness_values
                ),
                "no_consensus_target_count": sum(
                    value.get(
                        "status"
                    )
                    != "agreed"
                    for _task_id, _slot_type, value
                    in witness_values
                ),
            },
            "final_pipeline": {
                "consensus_agreed_count": sum(
                    value.get(
                        "status"
                    )
                    == "agreed"
                    for value in final_values
                ),
                "blocked_count": sum(
                    value.get(
                        "status"
                    )
                    != "agreed"
                    for value in final_values
                ),
                "programmatic_verification_status_counts": dict(
                    sorted(
                        Counter(
                            str(
                                (
                                    result.get(
                                        "verification"
                                    )
                                    or {}
                                ).get(
                                    "verification_status"
                                )
                                or "not_run"
                            )
                            for result in (
                                primary_results_by_task.values()
                            )
                        ).items()
                    )
                ),
            },
            "teacher_confidence_policy": (
                "diagnostic_only_not_a_hard_gate"
            ),
        }

        strong_summary = None
    else:
        primary_summary = None
        strong_summary = None

    verified_count = sum(
        bool(
            record[
                "quality_gate"
            ][
                "supervision_verified"
            ]
        )
        for record in refinement_records
    )

    training_eligible_count = sum(
        bool(
            record[
                "quality_gate"
            ][
                "training_eligible"
            ]
        )
        for record in refinement_records
    )

    blocked_count = (
        len(
            refinement_records
        )
        - verified_count
    )

    verification_tier_counts: dict[
        str,
        int,
    ] = {}

    for record in refinement_records:
        tier = str(
            (
                record.get(
                    "quality_gate"
                )
                or {}
            ).get(
                "verification_tier"
            )
            or "unknown"
        )

        verification_tier_counts[
            tier
        ] = (
            verification_tier_counts.get(
                tier,
                0,
            )
            + 1
        )

    block_reason_counts: dict[
        str,
        int,
    ] = {}

    for record in refinement_records:
        for reason in (
            record[
                "quality_gate"
            ][
                "block_reasons"
            ]
        ):
            block_reason_counts[
                reason
            ] = (
                block_reason_counts.get(
                    reason,
                    0,
                )
                + 1
            )

    # ----------------------------------------------------------------
    # Candidate Quality Diagnostics（候选质量诊断）
    # ----------------------------------------------------------------

    selected_category_counts: Counter[str] = Counter()
    selected_source_counts: Counter[str] = Counter()

    for item in prepared:
        metadata = item["candidate_metadata"]

        for key, value in (
            metadata.get(
                "selected_category_counts"
            )
            or {}
        ).items():
            selected_category_counts[
                str(key)
            ] += int(value)

        for key, value in (
            metadata.get(
                "selected_source_counts"
            )
            or {}
        ).items():
            selected_source_counts[
                str(key)
            ] += int(value)

    # ------------------------------------------------------------------
    # Candidate Builder Protocol Provenance（候选构造协议来源追踪）
    #
    # 不能由 runner 自己“猜” Candidate Builder 版本。
    # 必须读取每条 prepared request 真正记录的 metadata。
    #
    # 如果同一次运行混入多个版本，实验不可复现，直接失败。
    # ------------------------------------------------------------------

    candidate_builder_protocols = sorted(
        {
            str(
                (
                    item.get(
                        "candidate_metadata"
                    )
                    or {}
                ).get(
                    "protocol"
                )
                or ""
            )
            for item in prepared
            if str(
                (
                    item.get(
                        "candidate_metadata"
                    )
                    or {}
                ).get(
                    "protocol"
                )
                or ""
            )
        }
    )

    if len(candidate_builder_protocols) > 1:
        raise ValueError(
            "同一次 refinement 运行混入多个 Candidate Builder protocol："
            f"{candidate_builder_protocols}"
        )

    candidate_builder_protocol = (
        candidate_builder_protocols[0]
        if candidate_builder_protocols
        else None
    )

    candidate_quality_diagnostics = {
        "protocol": (
            candidate_builder_protocol
        ),
        "protocol_versions_seen": (
            candidate_builder_protocols
        ),
        "selected_candidate_count_total": sum(
            len(
                item[
                    "candidate_ids"
                ]
            )
            for item in prepared
        ),
        "selected_candidate_count_mean": (
            (
                sum(
                    len(
                        item[
                            "candidate_ids"
                        ]
                    )
                    for item in prepared
                )
                / len(prepared)
            )
            if prepared
            else 0.0
        ),
        "selected_category_counts": dict(
            sorted(
                selected_category_counts.items()
            )
        ),
        "selected_source_counts": dict(
            sorted(
                selected_source_counts.items()
            )
        ),
        "gold_guided_candidate_count": sum(
            len(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "gold_guided_candidate_ids"
                )
                or []
            )
            for item in prepared
        ),
        "gold_hunk_best_candidate_count_before_file_cap": sum(
            int(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "gold_hunk_best_candidate_count_before_file_cap"
                )
                or 0
            )
            for item in prepared
        ),
        "gold_file_cap_dropped_count": sum(
            int(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "gold_file_cap_dropped_count"
                )
                or 0
            )
            for item in prepared
        ),
        "issue_symbol_candidate_count": sum(
            len(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "issue_symbol_candidate_ids"
                )
                or []
            )
            for item in prepared
        ),
        "issue_symbol_match_task_count": sum(
            bool(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "issue_symbol_candidate_ids"
                )
            )
            for item in prepared
        ),
        "issue_symbol_unmatched_count": sum(
            len(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "issue_symbols_unmatched"
                )
                or []
            )
            for item in prepared
        ),
        "task_count_with_gold_guided_candidate": sum(
            bool(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "gold_guided_candidate_ids"
                )
            )
            for item in prepared
        ),
        "gold_path_missing_in_pre_fix_snapshot_count": sum(
            len(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "gold_paths_missing_in_pre_fix_snapshot"
                )
                or []
            )
            for item in prepared
        ),
        "optional_original_witness_still_missing_count": sum(
            len(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "optional_missing_original_witness_ids"
                )
                or []
            )
            for item in prepared
        ),
        "policy_candidate_cache_missing_count": sum(
            int(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "policy_candidate_cache_missing_count"
                )
                or 0
            )
            for item in prepared
        ),
        "overlap_dropped_count": sum(
            int(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "overlap_dropped_count"
                )
                or 0
            )
            for item in prepared
        ),
        "quota_dropped_count": sum(
            int(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "quota_dropped_count"
                )
                or 0
            )
            for item in prepared
        ),
        "per_file_limit_dropped_count": sum(
            int(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "per_file_limit_dropped_count"
                )
                or 0
            )
            for item in prepared
        ),
        "prompt_over_hard_budget_before_compaction_task_count": sum(
            (
                int(
                    (
                        item[
                            "candidate_metadata"
                        ]
                    ).get(
                        "prompt_chars_before_question_compaction"
                    )
                    or 0
                )
                > args.max_prompt_chars
            )
            for item in prepared
        ),
        "question_compacted_task_count": sum(
            bool(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "question_compacted"
                )
            )
            for item in prepared
        ),
        "question_original_chars_total": sum(
            int(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "question_original_chars"
                )
                or 0
            )
            for item in prepared
        ),
        "question_teacher_chars_total": sum(
            int(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "question_teacher_chars"
                )
                or 0
            )
            for item in prepared
        ),
        "prompt_chars_reduced_by_question_compaction_total": sum(
            max(
                0,
                int(
                    (
                        item[
                            "candidate_metadata"
                        ]
                    ).get(
                        "prompt_chars_before_question_compaction"
                    )
                    or 0
                )
                - int(
                    (
                        item[
                            "candidate_metadata"
                        ]
                    ).get(
                        "prompt_chars_after_question_compaction"
                    )
                    or 0
                ),
            )
            for item in prepared
        ),
        "question_removed_chars_total": sum(
            int(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "question_removed_chars"
                )
                or 0
            )
            for item in prepared
        ),
        "teacher_display_reordered_task_count": sum(
            bool(
                (
                    item["candidate_metadata"]
                ).get(
                    "teacher_display_reordered"
                )
            )
            for item in prepared
        ),
        "prompt_budget_dropped_candidate_count": sum(
            int(
                (
                    item["candidate_metadata"]
                ).get(
                    "prompt_budget_dropped_count"
                )
                or 0
            )
            for item in prepared
        ),
        "task_count_with_prompt_budget_drop": sum(
            bool(
                (
                    item["candidate_metadata"]
                ).get(
                    "prompt_budget_dropped_count"
                )
            )
            for item in prepared
        ),
        "underfilled_candidate_pool_task_count": sum(
            bool(
                (
                    item[
                        "candidate_metadata"
                    ]
                ).get(
                    "candidate_pool_underfilled"
                )
            )
            for item in prepared
        ),
    }


    report = {
        "processing_name": (
            "llm_supervision_refinement_v1_9_2_1_candidate_blind_requirement"
        ),
        "protocol_version": (
            "supervision-refinement-v1.9.2.1"
        ),
        "dataset_version": (
            manifest.get(
                "dataset_version"
            )
        ),
        "manifest_audit_status": (
            manifest.get(
                "audit_status"
            )
        ),
        "manifest": str(
            manifest_path
        ),
        "dataset_dir": str(
            dataset_dir
        ),
        "split": (
            args.split
        ),
        "split_path": str(
            split_path
        ),
        "task_scope": (
            args.task_scope
        ),
        "scope_task_count": (
            scope_task_count
        ),
        "mode": (
            args.mode
        ),
        "primary_provider_requested": (
            args.primary_provider
        ),
        "primary_provider_concurrency_semantics": (
            "for bigmodel, --bigmodel-concurrency means per-key slots; "
            "actual executor concurrency is additionally capped by "
            "--bigmodel-global-concurrency"
        ),
        "bigmodel_global_concurrency": (
            args.bigmodel_global_concurrency
        ),
        "primary_generation_policy": (
            "BigModel thinking=disabled and do_sample=false; "
            "two-stage requirement decisions + targeted Witness selection; A/B plus conditional C in each stage; "
            "temperature/top_p intentionally omitted because sampling is disabled"
        ),
        "reference_mode": (
            args.reference_mode
        ),
        "review_policy": (
            "none_deepseek_removed_v1_9_2"
        ),
        "frozen_dataset_modified": (
            False
        ),
        "sidecar_only": (
            True
        ),
        "sample_seed": (
            args.sample_seed
        ),
        "allow_full_run": (
            args.allow_full_run
        ),
        "selected_task_count": len(
            selected_rows
        ),
        "prepared_task_count": len(
            prepared
        ),
        "preparation_error_count": len(
            preparation_errors
        ),
        "candidate_limit": (
            args.candidate_limit
        ),
        "max_prompt_chars": (
            args.max_prompt_chars
        ),
        "max_actual_prompt_chars": max(
            (
                item[
                    "prompt_chars"
                ]
                for item in prepared
            ),
            default=0,
        ),
        "mean_actual_prompt_chars": (
            (
                sum(
                    item[
                        "prompt_chars"
                    ]
                    for item in prepared
                )
                / len(
                    prepared
                )
            )
            if prepared
            else 0.0
        ),
        "candidate_quality_diagnostics": (
            candidate_quality_diagnostics
        ),
        "two_stage_consensus": {
            "stage_1": (
                "7-slot requirement-only 2-of-3 consensus"
            ),
            "stage_2": (
                "targeted Witness selection only for repository_required slots; "
                "status majority + >=2 votes per accepted AND-group"
            ),
            "teacher_confidence_used_as_hard_gate": False,
            "min_consensus_confidence_argument_role": (
                "deprecated_diagnostic_only_v1_9_2"
            ),
            "independent_semantic_review": False,
            "training_promotion_enabled": (
                args.promote_consensus_training
            ),
        },
        "candidate_builder_config": {
            "candidate_limit": (
                args.candidate_limit
            ),
            "max_per_file": (
                args.candidate_max_per_file
            ),
            "test_quota": (
                args.candidate_test_quota
            ),
            "doc_quota": (
                args.candidate_doc_quota
            ),
            "resource_quota": (
                args.candidate_resource_quota
            ),
            "low_value_quota": (
                args.candidate_low_value_quota
            ),
            "overlap_threshold": (
                args.candidate_overlap_threshold
            ),
            "gold_units_per_hunk": (
                args.gold_units_per_hunk
            ),
            "max_gold_units_per_file": (
                args.max_gold_units_per_file
            ),
            "issue_symbol_units_per_symbol": (
                args.issue_symbol_units_per_symbol
            ),
            "issue_symbol_policy_path_limit": (
                args.issue_symbol_policy_path_limit
            ),
            "max_teacher_question_chars": (
                args.max_teacher_question_chars
            ),
            "prompt_budget_semantics": (
                "keep full Question when full Prompt <= max_prompt_chars; "
                "default hard budget is 140k chars in v1.9.2; "
                "compact Question only when full Prompt exceeds hard budget; "
                "drop lowest-priority ordinary candidates only if still over budget"
            ),
            "teacher_reference_contract": (
                "Stage 1 outputs only requirement decisions; Stage 2 outputs only "
                "targeted Candidate Number witness groups; program resolves "
                "Evidence IDs, obligation identity, assessment, STOP and minimal certificate"
            ),
            "teacher_candidate_presentation": (
                "same candidate set; display order prioritizes issue/gold/"
                "retrieval relevance signals; original certificate/witness "
                "identity is hidden from candidate headers and does not affect display order"
            ),
            "build_db": str(
                args.build_db.resolve()
            ),
        },
        "primary_teacher_environment": (
            primary_environment
        ),
        "primary_teacher_protocol": (
            primary_protocol
        ),
        "strong_teacher_environment": (
            strong_environment
        ),
        "strong_teacher_protocol": (
            strong_protocol
        ),
        "primary_teacher_summary": (
            primary_summary
        ),
        "strong_teacher_summary": (
            strong_summary
        ),
        "strong_review_skipped_primary_not_accepted_count": None,
        "quality_gate": {
            "teacher_confidence_used_as_hard_gate": False,
            "min_consensus_confidence_argument_role": (
                "deprecated_diagnostic_only_v1_9_2"
            ),
            "independent_semantic_review": False,
            "training_promotion_enabled": (
                args.promote_consensus_training
            ),
            "supervision_verified_count": (
                verified_count
            ),
            "blocked_count": (
                blocked_count
            ),
            "training_eligible_count": (
                training_eligible_count
            ),
            "verification_tier_counts": dict(
                sorted(
                    verification_tier_counts.items()
                )
            ),
            "block_reason_counts": dict(
                sorted(
                    block_reason_counts.items()
                )
            ),
            "training_policy": (
                "v1.9.2 requires Stage-1 Requirement 2-of-3 consensus "
                "+ Stage-2 targeted Witness group consensus "
                "+ Core v1.7 deterministic verification; "
                "train split additionally requires explicit "
                "--promote-consensus-training. "
                "Teacher self-confidence is diagnostic only; "
                "self-consistency is not independent semantic corroboration."
            ),
        },
        "error_count": len(
            all_errors
        ),
        "timing_seconds": (
            elapsed
        ),
        "outputs": {
            "refinement_requests_jsonl": str(
                request_path
            ),
            "refinement_jsonl": str(
                refinement_path
            ),
            "refinement_errors_jsonl": str(
                errors_path
            ),
            "refinement_report_json": str(
                report_path
            ),
        },
    }

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": (
                    "passed"
                    if not all_errors
                    else "completed_with_errors"
                ),
                "mode": (
                    args.mode
                ),
                "split": (
                    args.split
                ),
                "task_scope": (
                    args.task_scope
                ),
                "scope_task_count": (
                    scope_task_count
                ),
                "selected_task_count": len(
                    selected_rows
                ),
                "prepared_task_count": len(
                    prepared
                ),
                "primary_teacher": (
                    primary_summary
                ),
                "strong_teacher": (
                    strong_summary
                ),
                "supervision_verified_count": (
                    verified_count
                ),
                "blocked_count": (
                    blocked_count
                ),
                "training_eligible_count": (
                    training_eligible_count
                ),
                "error_count": len(
                    all_errors
                ),
                "outputs": (
                    report[
                        "outputs"
                    ]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return (
        0
        if not all_errors
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
