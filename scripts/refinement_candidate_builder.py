# -*- coding: utf-8 -*-
"""
Refinement Candidate Builder v1.5.2
（监督修正专用候选构造器 v1.4）

建议位置：
    scripts/refinement_candidate_builder.py

========================================================================
一、为什么单独构造 Teacher Candidate Pool（教师候选池）
========================================================================

V2.10 的 policy_states.candidate_actions 是为 Policy Ranker（策略排序模型）
准备的训练/在线候选，其中允许存在：

    - positive（正例）；
    - hard negative（困难负例）；
    - ordinary negative（普通负例）；
    - docs/tests 等词法检索噪声。

这对排序模型是合理的，但不能直接等价为：

    “给 LLM Teacher 的 24 条候选都应该是高质量修复上下文”。

本模块只用于 Data Refinement（数据修正）阶段，目标是：

    1. 保留当前被冻结监督真正使用的 Evidence；
    2. 利用 Gold Patch 的 changed path / hunk location（修改路径/修改块位置）
       在任务自己的 pre-fix snapshot（修复前快照）中补关键源码；
    3. 把 V2.10 Policy Candidate 只作为补充来源；
    4. 对 docs / changelog / fixtures 等低价值候选做配额；
    5. 从 Issue / Question 中提取明确 symbol（符号），
       在高精度文件集合中补对应 pre-fix Evidence；
    6. 每个 Gold hunk 默认只保留 Best-1 Evidence，最多允许 Best-2；
    7. data/resource 文件单独分类并限制配额；
    8. 对同文件高度重叠的 Evidence Unit 做去重；
    9. 所有最终 Candidate 必须是真实、scoreable 的 pre-fix Evidence ID。

========================================================================
二、绝对禁止的事情
========================================================================

- 不修改冻结 V2.10；
- 不把 post-fix code 当 Evidence；
- 不根据 Gold Patch 生成新的虚构 Evidence ID；
- 不跨 snapshot 查询其它版本的同名文件；
- 不把 test_patch 中新增的测试代码当 pre-fix Evidence；
- 不为了“填满 24 条”而强行加入明显无关文件。

========================================================================
三、Gold-guided Candidate（Gold 引导候选）为何仍然是合法 Evidence
========================================================================

Gold Patch 只告诉离线 Teacher：

    “真实修改发生在哪个 pre-fix 文件 / 哪个旧代码位置附近”。

随后程序在：

    snapshot_file_memberships
        -> file_versions
        -> evidence_units

中查找该任务自己的 pre-fix snapshot，并加载已有 Evidence Unit。

因此最终 Candidate 仍然满足：

    candidate Evidence == pre-fix repository Evidence

Gold 只用于离线候选定位，不会进入在线 Agent 输入。
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# ============================================================================
# 路径分类与默认配额
# ============================================================================


DOC_EXTENSIONS = {
    ".md",
    ".markdown",
    ".rst",
    ".txt",
}

DOC_PATH_MARKERS = (
    "/docs/",
    "/doc/",
    "/documentation/",
    "/whatsnew/",
    "/changelog/",
    "/changes/",
)

LOW_VALUE_BASENAMES = {
    "readme",
    "contributing",
    "changelog",
    "changes",
    "authors",
    "license",
}

FIXTURE_MARKERS = (
    "/fixtures/",
    "/fixture/",
    "/test_data/",
    "/testdata/",
    "/data/fixtures/",
)

TEST_PATH_MARKERS = (
    "/tests/",
    "/test/",
    "/testing/",
)


# 数据/资源文件不再默认归到 source（源码）。
#
# 注意：
#   如果 Issue / Gold 明确指向这些文件，它们仍可作为高精度候选进入；
#  这里只限制“普通补充候选”占用 Teacher Prompt 的预算。
RESOURCE_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".parquet",
    ".h5",
    ".hdf5",
    ".mat",
    ".dat",
    ".bin",
    ".geojson",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".webp",
}


@dataclass(frozen=True)
class CandidateBuilderConfig:
    """
    Teacher Candidate Pool 的固定策略参数。

    candidate_limit：
        非强制 Evidence 的目标总量。
        Original Certificate / Boundary / available original witness
        可以使最终数量略高于这个值，因为旧监督必须可审计。

    max_per_file：
        对普通候选限制同一文件最多占多少槽位。
        强制 Evidence 不受此限制。

    test_quota / doc_quota / low_value_quota：
        只约束“普通补充候选”。
        Gold changed source code 和强制 Evidence 不会因为配额被删掉。
    """

    candidate_limit: int = 24
    max_per_file: int = 4

    test_quota: int = 4
    doc_quota: int = 2
    resource_quota: int = 1
    low_value_quota: int = 1

    overlap_threshold: float = 0.65

    # Gold hunk 默认只取 Best-1。
    # 为了保留必要的成对上下文，允许显式调到 2，但禁止 >2。
    gold_units_per_hunk: int = 1

    # 一个 changed file 如果有很多 hunks，不把整份文件附近十几二十个
    # Evidence 全部塞给 Teacher。先选每个 hunk 的 Best-1/2，再做文件级优选。
    max_gold_units_per_file: int = 8

    gold_units_per_file_without_hunk: int = 2

    issue_path_units_per_file: int = 2

    # Issue 中明确出现的 symbol，在高精度路径集合里直接查 exact symbol。
    issue_symbol_units_per_symbol: int = 1

    # 为避免“全仓库 symbol scan（全仓库符号扫描）”重新引入噪声和昂贵查询，
    # Issue symbol 只搜索：
    #   changed paths + explicit issue paths + 前若干个高排名 policy source paths。
    issue_symbol_policy_path_limit: int = 6

    def validate(self) -> None:
        if self.candidate_limit < 1:
            raise ValueError("candidate_limit 必须 >= 1")
        if self.max_per_file < 1:
            raise ValueError("max_per_file 必须 >= 1")
        for name, value in (
            ("test_quota", self.test_quota),
            ("doc_quota", self.doc_quota),
            ("resource_quota", self.resource_quota),
            ("low_value_quota", self.low_value_quota),
        ):
            if value < 0:
                raise ValueError(f"{name} 不能为负数")

        if not 0.0 <= self.overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold 必须在 [0,1]")

        if not 1 <= self.gold_units_per_hunk <= 2:
            raise ValueError(
                "gold_units_per_hunk 必须在 [1,2]；"
                "默认 Best-1，最多 Best-2"
            )

        if self.max_gold_units_per_file < 1:
            raise ValueError("max_gold_units_per_file 必须 >= 1")

        if self.gold_units_per_file_without_hunk < 1:
            raise ValueError("gold_units_per_file_without_hunk 必须 >= 1")

        if self.issue_path_units_per_file < 1:
            raise ValueError("issue_path_units_per_file 必须 >= 1")

        if self.issue_symbol_units_per_symbol < 1:
            raise ValueError("issue_symbol_units_per_symbol 必须 >= 1")

        if self.issue_symbol_policy_path_limit < 0:
            raise ValueError("issue_symbol_policy_path_limit 不能为负数")


@dataclass
class CandidateRecord:
    """一个真实 pre-fix Evidence Candidate 及其离线来源信息。"""

    evidence_id: str
    file_version_id: str
    path: str
    unit_type: str
    symbol: str | None
    start_line: int
    end_line: int
    content: str
    rendered_token_count: int

    # 来源越靠前越可信/越需要优先保留。
    sources: set[str] = field(default_factory=set)

    # 是否属于冻结监督当前轨迹/旧 witness，必须允许 Teacher 看见。
    forced: bool = False

    # Gold patch 在 pre-fix snapshot 中定位到的真实源码 Evidence。
    gold_guided: bool = False

    # V2.10 policy candidate 的历史最好在线排名，仅作为补充排序信息。
    min_online_rank: int | None = None
    max_online_score: float | None = None

    # 旧监督标记，用于 Prompt audit。
    in_original_witness: bool = False
    in_original_certificate: bool = False
    in_original_boundary: bool = False

    def category(self) -> str:
        return classify_path(self.path)

    def priority_key(self) -> tuple[Any, ...]:
        """
        候选优先级。

        设计原则：
            Frozen trajectory/original witness
                > Gold changed pre-fix code
                > Issue explicit path
                > 普通 source code policy candidate
                > tests
                > docs
                > low-value files
        """
        if self.in_original_certificate or self.in_original_boundary:
            tier = 0
        elif self.in_original_witness:
            tier = 1
        elif "gold_patch_hunk" in self.sources:
            tier = 2
        elif "issue_explicit_symbol" in self.sources:
            tier = 3
        elif "gold_patch_path" in self.sources:
            tier = 4
        elif "issue_explicit_path" in self.sources:
            tier = 5
        else:
            category = self.category()
            if category == "source":
                tier = 6
            elif category == "test":
                tier = 7
            elif category == "doc":
                tier = 8
            elif category == "resource":
                tier = 9
            else:
                tier = 10

        rank = (
            self.min_online_rank
            if self.min_online_rank is not None
            else 2**31 - 1
        )

        score = (
            self.max_online_score
            if self.max_online_score is not None
            else float("-inf")
        )

        # 同优先级优先较小、较聚焦的 Evidence，减少 Prompt 噪声。
        line_span = max(1, self.end_line - self.start_line + 1)

        return (
            tier,
            rank,
            -score,
            line_span,
            self.path,
            self.start_line,
            self.end_line,
            self.evidence_id,
        )

    def to_prompt_record(self) -> dict[str, Any]:
        """转换成 refinement_teacher.py 已支持的紧凑 Candidate 结构。"""
        return {
            "evidence_id": self.evidence_id,
            "file_version_id": self.file_version_id,
            "path": self.path,
            "unit_type": self.unit_type,
            "symbol": self.symbol,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "rendered_token_count": self.rendered_token_count,
            "candidate_stats": {
                "sources": sorted(self.sources),
                "in_original_witness": self.in_original_witness,
                "in_original_certificate": self.in_original_certificate,
                "in_original_boundary": self.in_original_boundary,
                "min_online_rank": self.min_online_rank,
                "max_online_score": self.max_online_score,
                "category": self.category(),
                "gold_guided": self.gold_guided,
            },
        }


# ============================================================================
# 基础辅助函数
# ============================================================================


def normalize_repo_path(value: str) -> str:
    """统一仓库相对路径格式，避免 Windows/Unix slash 差异。"""
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def classify_path(path: str) -> str:
    """
    将文件路径分成 source/test/doc/low_value。

    注意：
        这只是 Teacher Candidate 的优先级/配额判断，
        不是在线 Retriever 的文件过滤规则。
    """
    normalized = "/" + normalize_repo_path(path).lower()
    suffix = Path(normalized).suffix.lower()
    stem = Path(normalized).stem.lower()

    if any(marker in normalized for marker in FIXTURE_MARKERS):
        return "low_value"

    if (
        any(marker in normalized for marker in TEST_PATH_MARKERS)
        or Path(normalized).name.lower().startswith("test_")
        or Path(normalized).name.lower().endswith("_test.py")
        or Path(normalized).name.lower() == "conftest.py"
    ):
        return "test"

    if (
        any(marker in normalized for marker in DOC_PATH_MARKERS)
        or suffix in DOC_EXTENSIONS
        or stem in LOW_VALUE_BASENAMES
    ):
        # changelog / contributing / README 比普通 docs 更低价值。
        if (
            stem in LOW_VALUE_BASENAMES
            or "/whatsnew/" in normalized
            or "/changelog/" in normalized
            or "/changes/" in normalized
        ):
            return "low_value"
        return "doc"

    if suffix in RESOURCE_EXTENSIONS:
        return "resource"

    return "source"


def line_overlap_ratio(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> float:
    """
    计算两个闭区间代码 span 的 overlap / min(span_length)。

    这个定义用于发现：
        633-713 与 671-751
    这类明显重复上下文。
    """
    left_start = int(left_start)
    left_end = int(left_end)
    right_start = int(right_start)
    right_end = int(right_end)

    overlap = max(
        0,
        min(left_end, right_end)
        - max(left_start, right_start)
        + 1,
    )

    if overlap <= 0:
        return 0.0

    left_len = max(1, left_end - left_start + 1)
    right_len = max(1, right_end - right_start + 1)

    return overlap / min(left_len, right_len)


def _retrieval_terms(text: str) -> set[str]:
    """轻量词项，用于 issue explicit path 内部的 Evidence 排序。"""
    return {
        token.lower()
        for token in re.findall(
            r"[A-Za-z_][A-Za-z0-9_]{2,}",
            str(text or ""),
        )
    }


def extract_issue_paths(question: str) -> list[str]:
    """
    从 Issue/Question 中只提取“看起来像明确仓库文件路径”的字符串。

    不根据普通函数名做全仓库 symbol 搜索：
        那会重新引入大范围噪声和昂贵查询。

    支持常见源码/配置/文档后缀。
    """
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])"
        r"([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+"
        r"\.(?:pyi|py|tsx|ts|jsx|js|java|go|rst|rs|cpp|cc|c|hpp|h|rb|php|cs|scala|kts|kt|toml|yaml|yml|json|ini|cfg|md))"
    )

    values = [
        normalize_repo_path(match.group(1))
        for match in pattern.finditer(str(question or ""))
    ]

    return list(dict.fromkeys(values))


# ============================================================================
# Issue / Question -> explicit symbol hints
# ============================================================================


ISSUE_SYMBOL_STOPWORDS = {
    "true", "false", "none", "null",
    "self", "cls",
    "str", "int", "float", "bool", "bytes",
    "list", "dict", "tuple", "set",
    "len", "range", "isinstance", "issubclass", "super", "print",
    "path", "file", "error", "exception", "traceback",
    "assertionerror", "typeerror", "valueerror", "keyerror",
    "indexerror", "attributeerror",
    "python", "pytest", "test", "tests", "github",
}

ISSUE_SYMBOL_NONCODE_SUFFIXES = {
    "com", "org", "net", "io", "dev",
    "html", "htm", "rst", "md", "txt",
    "sql", "py", "json", "yaml", "yml", "xml",
    "csv", "tsv",
}

ISSUE_SYMBOL_UPPERCASE_NOISE = {
    "SELECT", "FROM", "WHERE", "JOIN", "TABLE", "FIELD", "DISTINCT",
    "GROUP", "ORDER", "INSERT", "UPDATE", "DELETE",
    "NULL", "TRUE", "FALSE",
    "API", "URL", "HTTP", "HTTPS",
}


def _normalize_symbol_hint(value: str) -> str:
    text = str(value or "").strip().strip("`'\"")
    text = text.rstrip("()")
    return text.strip()


def _symbol_leaf(value: str) -> str:
    text = _normalize_symbol_hint(value)
    if not text:
        return ""
    return text.rsplit(".", 1)[-1]


def _looks_like_noncode_symbol(value: str) -> bool:
    """
    过滤明显不是仓库代码 symbol 的词。

    例如：
        SELECT
        Traceback
        github.com
        file.sql
    """
    value = _normalize_symbol_hint(value)
    if not value:
        return True

    leaf = _symbol_leaf(value)
    if not leaf:
        return True

    if leaf.lower() in ISSUE_SYMBOL_STOPWORDS:
        return True

    if value in ISSUE_SYMBOL_UPPERCASE_NOISE:
        return True

    if leaf.isupper() and len(leaf) >= 3:
        return True

    parts = value.split(".")
    if (
        len(parts) >= 2
        and parts[-1].lower() in ISSUE_SYMBOL_NONCODE_SUFFIXES
    ):
        return True

    if "/" in value or "\\" in value:
        return True

    lowered = value.lower()
    if "http" in lowered or "www." in lowered or "@" in value:
        return True

    return False


def extract_issue_symbols(
    question: str,
    *,
    limit: int = 16,
) -> list[str]:
    """
    高精度提取 Issue 中明确出现的代码 symbol。

    v1.5.1 不再扫描所有 snake_case / CamelCase 普通词，只使用：

        1. Markdown inline code：
               `get_pvgis_horizon`

        2. 明确调用：
               dcmread(...)
               self._do_load(...)

        3. Traceback frame：
               in _invoke_field_validators

        4. 明确自然语言：
               function foo
               method Class.foo
               class Dataset

    并按 leaf 去重：
        self._eval / _eval 只查一次。
    """
    text = str(question or "")
    candidates: list[str] = []

    for match in re.finditer(
        r"`+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)`+",
        text,
    ):
        candidates.append(match.group(1))

    for match in re.finditer(
        r"(?<![A-Za-z0-9_])"
        r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
        r"\s*\(",
        text,
    ):
        candidates.append(match.group(1))

    # 3. 严格匹配 Python Traceback frame。
    #
    # 不能使用通用：
    #     "in xxx"
    #
    # 否则普通自然语言里的：
    #     in user / in num / in the
    #
    # 都会被误认为代码 symbol。
    for match in re.finditer(
        r"(?m)^\s*File\s+[\"'][^\"']+[\"']"
        r",\s*line\s+\d+"
        r",\s*in\s+([A-Za-z_][A-Za-z0-9_]*)\s*$",
        text,
    ):
        candidates.append(match.group(1))

    for match in re.finditer(
        r"\b(?:function|method|class)\s+"
        r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)",
        text,
        flags=re.IGNORECASE,
    ):
        candidates.append(match.group(1))

    result: list[str] = []
    seen_leafs: set[str] = set()

    for raw in candidates:
        value = _normalize_symbol_hint(raw)
        leaf = _symbol_leaf(value)

        if (
            not value
            or len(leaf) < 3
            or _looks_like_noncode_symbol(value)
        ):
            continue

        leaf_key = leaf.lower()
        if leaf_key in seen_leafs:
            continue

        seen_leafs.add(leaf_key)
        result.append(value)

        if len(result) >= limit:
            break

    return result


def record_matches_issue_symbol(
    record: CandidateRecord,
    issue_symbol: str,
) -> bool:
    """
    判断 Evidence 是否对应 Issue symbol。

    优先使用 Evidence metadata 的 qualified_name/symbol。

    如果历史 Evidence 的 symbol metadata 为空，
    再检查 pre-fix 正文是否真正声明：

        def dcmread(...)
        class Dataset(...)
        SOME_CONSTANT = ...

    只匹配“声明”，不把普通函数调用当成定义。
    """
    expected = _normalize_symbol_hint(issue_symbol)
    expected_leaf = _symbol_leaf(expected)

    if not expected_leaf:
        return False

    actual = _normalize_symbol_hint(
        str(record.symbol or "")
    )

    if actual:
        actual_leaf = _symbol_leaf(actual)

        if (
            actual == expected
            or actual_leaf == expected_leaf
            or actual.endswith("." + expected)
            or expected.endswith("." + actual)
        ):
            return True

    content = str(record.content or "")
    escaped = re.escape(expected_leaf)

    declaration_patterns = (
        rf"(?m)^\s*(?:async\s+)?def\s+{escaped}\s*\(",
        rf"(?m)^\s*class\s+{escaped}\b",
        rf"(?m)^\s*{escaped}\s*=",
    )

    return any(
        re.search(pattern, content) is not None
        for pattern in declaration_patterns
    )


def _issue_symbol_record_key(
    record: CandidateRecord,
    issue_symbol: str,
) -> tuple[Any, ...]:
    actual = _normalize_symbol_hint(
        str(record.symbol or "")
    )
    expected = _normalize_symbol_hint(issue_symbol)

    exact_full = bool(actual and actual == expected)
    exact_leaf = bool(
        actual
        and _symbol_leaf(actual) == _symbol_leaf(expected)
    )

    span = max(
        1,
        record.end_line - record.start_line + 1,
    )

    return (
        0 if exact_full else 1,
        0 if exact_leaf else 1,
        span,
        record.start_line,
        record.evidence_id,
    )


# ============================================================================
# Gold Patch -> pre-fix hunk targets
# ============================================================================


@dataclass(frozen=True)
class PatchHunkTarget:
    path: str
    old_start: int
    old_length: int
    symbol_hint: str | None

    @property
    def old_end(self) -> int:
        if self.old_length <= 0:
            return max(1, self.old_start)
        return max(1, self.old_start + self.old_length - 1)


HUNK_RE = re.compile(
    r"^@@\s+-([0-9]+)(?:,([0-9]+))?\s+\+[0-9]+(?:,[0-9]+)?\s+@@(?:\s*(.*))?$"
)


def _symbol_from_hunk_header(header_tail: str | None) -> str | None:
    """从 @@ ... @@ 后面的 Python/通用函数签名中提取弱 symbol hint。"""
    text = str(header_tail or "").strip()
    if not text:
        return None

    # Python: def foo(...), class Bar
    match = re.search(
        r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
        text,
    )
    if match:
        return match.group(1)

    # C/Java/JS 风格：最后一个 identifier(...)
    calls = re.findall(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        text,
    )
    if calls:
        return calls[-1]

    return None


def parse_pre_fix_patch_targets(patch_text: str | None) -> tuple[
    list[str],
    list[PatchHunkTarget],
]:
    """
    从 unified diff 中解析 pre-fix 路径和 old-side hunk 范围。

    使用 old side（-start,length），因为 Teacher Candidate 必须来自修复前代码。
    新增文件 old path=/dev/null 时不会构造 pre-fix target。
    """
    if not isinstance(patch_text, str) or not patch_text.strip():
        return [], []

    current_old_path: str | None = None
    changed_paths: list[str] = []
    hunks: list[PatchHunkTarget] = []

    for raw_line in patch_text.splitlines():
        line = raw_line.rstrip("\n")

        if line.startswith("--- "):
            value = line[4:].strip().split("\t", 1)[0]
            if value == "/dev/null":
                current_old_path = None
                continue
            if value.startswith("a/"):
                value = value[2:]
            current_old_path = normalize_repo_path(value)
            if current_old_path:
                changed_paths.append(current_old_path)
            continue

        if current_old_path and line.startswith("@@"):
            match = HUNK_RE.match(line)
            if not match:
                continue

            old_start = int(match.group(1))
            old_length = int(match.group(2) or 1)
            symbol_hint = _symbol_from_hunk_header(match.group(3))

            hunks.append(
                PatchHunkTarget(
                    path=current_old_path,
                    old_start=max(1, old_start),
                    old_length=max(0, old_length),
                    symbol_hint=symbol_hint,
                )
            )

    return list(dict.fromkeys(changed_paths)), hunks


# ============================================================================
# 只读 Build DB Store
# ============================================================================


class BuildEvidenceStore:
    """
    只读访问 data/.build/unified_swe_v1.sqlite3。

    这是 refinement 阶段补充 Teacher Candidate 的来源，不修改 Builder 状态。
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(
                f"找不到 build database：{self.path}"
            )

        self.connection = sqlite3.connect(
            self.path.as_uri() + "?mode=ro",
            uri=True,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA query_only=ON")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute("PRAGMA cache_size=-131072")

        required_tables = {
            "snapshot_file_memberships",
            "file_versions",
            "evidence_units",
        }
        actual_tables = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing = required_tables - actual_tables
        if missing:
            raise ValueError(
                "build database 缺少 refinement 所需表："
                f"{sorted(missing)}"
            )

    def close(self) -> None:
        self.connection.close()

    @staticmethod
    def _record_from_rows(
        unit_payload: Mapping[str, Any],
        file_payload: Mapping[str, Any],
    ) -> CandidateRecord | None:
        if not bool(unit_payload.get("scoreable")):
            return None

        content = file_payload.get("content")
        if not isinstance(content, str):
            return None

        start = max(1, int(unit_payload.get("start_line") or 1))
        end = max(start, int(unit_payload.get("end_line") or start))
        lines = content.splitlines()
        body = "\n".join(lines[start - 1 : end])

        symbol = (
            unit_payload.get("qualified_name")
            or unit_payload.get("symbol")
        )

        return CandidateRecord(
            evidence_id=str(unit_payload["evidence_id"]),
            file_version_id=str(
                unit_payload.get("file_version_id")
                or file_payload.get("file_version_id")
                or ""
            ),
            path=str(file_payload.get("path") or ""),
            unit_type=str(unit_payload.get("unit_type") or "code_block"),
            symbol=None if symbol is None else str(symbol),
            start_line=start,
            end_line=end,
            content=body,
            rendered_token_count=int(unit_payload.get("rendered_token_count") or 0),
        )

    def get_by_ids(self, evidence_ids: Sequence[str]) -> dict[str, CandidateRecord]:
        """按真实 Evidence ID 加载 scoreable pre-fix Evidence。"""
        ids = list(dict.fromkeys(map(str, evidence_ids)))
        result: dict[str, CandidateRecord] = {}

        for offset in range(0, len(ids), 500):
            chunk = ids[offset : offset + 500]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)

            query = (
                "SELECT e.payload_json AS unit_json, f.payload_json AS file_json "
                "FROM evidence_units e JOIN file_versions f "
                "ON f.file_version_id=e.file_version_id "
                f"WHERE e.evidence_id IN ({placeholders}) AND e.scoreable=1"
            )

            for row in self.connection.execute(query, chunk):
                unit = json.loads(row["unit_json"])
                file_record = json.loads(row["file_json"])
                record = self._record_from_rows(unit, file_record)
                if record is not None:
                    result[record.evidence_id] = record

        return result

    def get_snapshot_path_units(
        self,
        *,
        snapshot_id: str,
        path: str,
    ) -> list[CandidateRecord]:
        """
        精确查询“该 task 的 pre-fix snapshot + exact repo path”。

        这条 snapshot 限制是 Gold-guided Candidate 安全性的关键。
        """
        normalized = normalize_repo_path(path)

        row = self.connection.execute(
            "SELECT f.file_version_id, f.payload_json "
            "FROM snapshot_file_memberships m "
            "JOIN file_versions f ON f.file_version_id=m.file_version_id "
            "WHERE m.snapshot_id=? AND m.path=?",
            (snapshot_id, normalized),
        ).fetchone()

        if row is None:
            return []

        file_version_id = str(row["file_version_id"])
        file_record = json.loads(row["payload_json"])

        result: list[CandidateRecord] = []

        for unit_row in self.connection.execute(
            "SELECT payload_json FROM evidence_units "
            "WHERE file_version_id=? AND scoreable=1 "
            "ORDER BY rowid",
            (file_version_id,),
        ):
            unit = json.loads(unit_row["payload_json"])
            record = self._record_from_rows(unit, file_record)
            if record is not None:
                result.append(record)

        result.sort(
            key=lambda item: (
                item.start_line,
                item.end_line,
                item.evidence_id,
            )
        )
        return result


# ============================================================================
# Candidate 组合 / 排序 / 去重
# ============================================================================


def _merge_record(
    target: dict[str, CandidateRecord],
    incoming: CandidateRecord,
) -> CandidateRecord:
    """同一个 Evidence ID 来自多个来源时合并 metadata，不复制正文。"""
    existing = target.get(incoming.evidence_id)
    if existing is None:
        target[incoming.evidence_id] = incoming
        return incoming

    existing.sources.update(incoming.sources)
    existing.forced = existing.forced or incoming.forced
    existing.gold_guided = existing.gold_guided or incoming.gold_guided
    existing.in_original_witness = (
        existing.in_original_witness or incoming.in_original_witness
    )
    existing.in_original_certificate = (
        existing.in_original_certificate or incoming.in_original_certificate
    )
    existing.in_original_boundary = (
        existing.in_original_boundary or incoming.in_original_boundary
    )

    if incoming.min_online_rank is not None:
        if existing.min_online_rank is None:
            existing.min_online_rank = incoming.min_online_rank
        else:
            existing.min_online_rank = min(
                existing.min_online_rank,
                incoming.min_online_rank,
            )

    if incoming.max_online_score is not None:
        if existing.max_online_score is None:
            existing.max_online_score = incoming.max_online_score
        else:
            existing.max_online_score = max(
                existing.max_online_score,
                incoming.max_online_score,
            )

    return existing


def _hunk_record_key(
    record: CandidateRecord,
    hunk: PatchHunkTarget,
) -> tuple[Any, ...]:
    """Gold hunk 附近 Evidence 的稳定优先级。"""
    symbol = str(record.symbol or "")
    symbol_match = bool(
        hunk.symbol_hint
        and (
            symbol == hunk.symbol_hint
            or symbol.endswith("." + hunk.symbol_hint)
            or hunk.symbol_hint in symbol
        )
    )

    overlap = max(
        0,
        min(record.end_line, hunk.old_end)
        - max(record.start_line, hunk.old_start)
        + 1,
    )

    distance = 0
    if overlap == 0:
        distance = min(
            abs(record.start_line - hunk.old_end),
            abs(hunk.old_start - record.end_line),
        )

    span = max(1, record.end_line - record.start_line + 1)

    return (
        0 if symbol_match else 1,
        0 if overlap > 0 else 1,
        -overlap,
        distance,
        span,
        record.evidence_id,
    )


def _record_issue_symbol_match_count(
    record: CandidateRecord,
    issue_symbols: Sequence[str],
) -> int:
    return sum(
        record_matches_issue_symbol(
            record,
            symbol,
        )
        for symbol in issue_symbols
    )


def _gold_file_record_key(
    record: CandidateRecord,
    *,
    issue_symbols: Sequence[str],
    question_terms: set[str],
) -> tuple[Any, ...]:
    """
    一个 changed file 有很多 hunks 时，文件级保留优先级：

        Issue 明确 symbol 命中
        > Question 词项覆盖
        > 更短、更聚焦 Evidence
        > 稳定行号/ID

    这样不会因为 patch 有十几个 hunk 就把十几二十条邻近代码
    全塞进 Teacher Prompt。
    """
    symbol_match_count = (
        _record_issue_symbol_match_count(
            record,
            issue_symbols,
        )
    )

    symbol_terms = _retrieval_terms(
        str(record.symbol or "")
    )
    content_terms = _retrieval_terms(
        record.content
    )

    question_overlap = len(
        question_terms
        & (
            symbol_terms
            | content_terms
        )
    )

    span = max(
        1,
        record.end_line
        - record.start_line
        + 1,
    )

    return (
        -symbol_match_count,
        -question_overlap,
        span,
        record.start_line,
        record.evidence_id,
    )


def _question_path_record_key(
    record: CandidateRecord,
    question_terms: set[str],
) -> tuple[Any, ...]:
    symbol_terms = _retrieval_terms(str(record.symbol or ""))
    content_terms = _retrieval_terms(record.content)
    overlap = len(question_terms & (symbol_terms | content_terms))
    span = max(1, record.end_line - record.start_line + 1)
    return (-overlap, span, record.start_line, record.evidence_id)


def _is_redundant_with_selected(
    candidate: CandidateRecord,
    selected: Sequence[CandidateRecord],
    threshold: float,
) -> bool:
    """
    只对同 file_version 的高度重叠 Evidence 做去重。

    forced Evidence 不调用这个函数，因此不会破坏旧轨迹可审计性。
    """
    for existing in selected:
        if existing.file_version_id != candidate.file_version_id:
            continue
        if line_overlap_ratio(
            existing.start_line,
            existing.end_line,
            candidate.start_line,
            candidate.end_line,
        ) >= threshold:
            return True
    return False


def build_refinement_candidates(
    *,
    supervision: Mapping[str, Any],
    question: str,
    snapshot_id: str,
    policy_stats: Mapping[str, Any],
    evidence_cache_records: Mapping[str, Mapping[str, Any]],
    build_store: BuildEvidenceStore,
    gold_patch: str | None,
    config: CandidateBuilderConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    构造最终 Teacher Candidate Pool。

    参数说明：
        policy_stats：
            refinement_core.collect_candidate_evidence_stats() 的结果。

        evidence_cache_records：
            training Evidence Cache 中能直接读到的 policy candidate 记录。
            Key 为 evidence_id。

        build_store：
            完整构建 SQLite 的只读接口，用于补：
              - cache 缺失 original witness；
              - Gold changed pre-fix code；
              - Issue 明确路径对应的 pre-fix code。
    """
    config.validate()

    all_records: dict[str, CandidateRecord] = {}
    diagnostics: dict[str, Any] = {
        "protocol": "refinement-candidate-builder-v1.5.2",
        "snapshot_id": snapshot_id,
        "candidate_limit": config.candidate_limit,

        "gold_changed_paths": [],
        "gold_hunk_count": 0,
        "gold_paths_missing_in_pre_fix_snapshot": [],
        "gold_guided_candidate_ids": [],
        "gold_hunk_best_candidate_count_before_file_cap": 0,
        "gold_file_cap_dropped_count": 0,

        "issue_explicit_paths": [],
        "issue_paths_missing_in_pre_fix_snapshot": [],

        "issue_explicit_symbols": [],
        "issue_symbol_search_paths": [],
        "issue_symbol_candidate_ids": [],
        "issue_symbol_matches": {},
        "issue_symbols_unmatched": [],

        "overlap_dropped_count": 0,
        "quota_dropped_count": 0,
        "per_file_limit_dropped_count": 0,
        "optional_missing_original_witness_ids": [],
    }

    # ------------------------------------------------------------------
    # 1. 先装入 Policy Candidate / Evidence Cache 中已有记录。
    # ------------------------------------------------------------------
    for evidence_id, raw in evidence_cache_records.items():
        stat = policy_stats.get(evidence_id)
        symbol = raw.get("symbol")
        record = CandidateRecord(
            evidence_id=str(evidence_id),
            file_version_id=str(raw.get("file_version_id") or ""),
            path=str(raw.get("path") or ""),
            unit_type=str(raw.get("unit_type") or "code_block"),
            symbol=None if symbol is None else str(symbol),
            start_line=max(1, int(raw.get("start_line") or 1)),
            end_line=max(1, int(raw.get("end_line") or raw.get("start_line") or 1)),
            content=str(raw.get("content") or ""),
            rendered_token_count=int(raw.get("rendered_token_count") or 0),
            sources={"policy_candidate"},
        )

        if stat is not None:
            record.in_original_witness = bool(stat.in_original_witness)
            record.in_original_certificate = bool(stat.in_original_certificate)
            record.in_original_boundary = bool(stat.in_original_boundary)
            record.forced = bool(
                stat.in_original_witness
                or stat.in_original_certificate
                or stat.in_original_boundary
            )
            record.min_online_rank = stat.min_online_rank
            record.max_online_score = stat.max_online_score
            record.sources.update(
                f"policy:{source}"
                for source in sorted(stat.candidate_sources or [])
            )

        _merge_record(all_records, record)

    # ------------------------------------------------------------------
    # 2. Original Witness / Boundary / Complete 即使不在 training cache，
    #    也尝试从完整 build DB 读取。
    # ------------------------------------------------------------------
    required_original_ids = [
        evidence_id
        for evidence_id, stat in policy_stats.items()
        if (
            stat.in_original_witness
            or stat.in_original_certificate
            or stat.in_original_boundary
        )
    ]

    missing_original_ids = [
        evidence_id
        for evidence_id in required_original_ids
        if evidence_id not in all_records
    ]

    if missing_original_ids:
        recovered = build_store.get_by_ids(missing_original_ids)
        for evidence_id, record in recovered.items():
            stat = policy_stats[evidence_id]
            record.sources.add("original_recovered_from_build_db")
            record.in_original_witness = bool(stat.in_original_witness)
            record.in_original_certificate = bool(stat.in_original_certificate)
            record.in_original_boundary = bool(stat.in_original_boundary)
            record.forced = True
            record.min_online_rank = stat.min_online_rank
            record.max_online_score = stat.max_online_score
            _merge_record(all_records, record)

    unresolved_original = sorted(
        set(required_original_ids) - set(all_records)
    )

    # Boundary / Complete 缺失仍然 hard fail；alternative witness 可报告后继续。
    hard_missing = [
        evidence_id
        for evidence_id in unresolved_original
        if (
            policy_stats[evidence_id].in_original_certificate
            or policy_stats[evidence_id].in_original_boundary
        )
    ]

    if hard_missing:
        raise ValueError(
            "Build DB 仍缺少 Boundary/Complete Evidence，不能安全 refinement："
            f"{hard_missing}"
        )

    diagnostics["optional_missing_original_witness_ids"] = [
        evidence_id
        for evidence_id in unresolved_original
        if policy_stats[evidence_id].in_original_witness
    ]

    # ------------------------------------------------------------------
    # 3. Gold Patch changed path/hunk -> 当前 task 的 pre-fix snapshot。
    # ------------------------------------------------------------------
    changed_paths, hunks = parse_pre_fix_patch_targets(
        gold_patch
    )

    diagnostics[
        "gold_changed_paths"
    ] = changed_paths

    diagnostics[
        "gold_hunk_count"
    ] = len(
        hunks
    )

    question_terms = (
        _retrieval_terms(
            question
        )
    )

    issue_paths = (
        extract_issue_paths(
            question
        )
    )

    issue_symbols = (
        extract_issue_symbols(
            question
        )
    )

    diagnostics[
        "issue_explicit_paths"
    ] = issue_paths

    diagnostics[
        "issue_explicit_symbols"
    ] = issue_symbols

    hunks_by_path: dict[
        str,
        list[
            PatchHunkTarget
        ],
    ] = defaultdict(
        list
    )

    for hunk in hunks:
        hunks_by_path[
            hunk.path
        ].append(
            hunk
        )

    # 同一个 task 内同一路径只查询一次 Build DB。
    path_units_cache: dict[
        str,
        list[
            CandidateRecord
        ],
    ] = {}

    def get_path_units(
        path: str,
    ) -> list[CandidateRecord]:
        normalized = (
            normalize_repo_path(
                path
            )
        )

        if normalized not in path_units_cache:
            path_units_cache[
                normalized
            ] = (
                build_store
                .get_snapshot_path_units(
                    snapshot_id=(
                        snapshot_id
                    ),
                    path=(
                        normalized
                    ),
                )
            )

        return path_units_cache[
            normalized
        ]

    gold_selected_ids: set[str] = set()

    for path in changed_paths:
        units = get_path_units(
            path
        )

        if not units:
            diagnostics[
                "gold_paths_missing_in_pre_fix_snapshot"
            ].append(
                path
            )
            continue

        path_hunks = (
            hunks_by_path.get(
                path
            )
            or []
        )

        if path_hunks:
            # ----------------------------------------------------------
            # 每个 hunk 先选 Best-1（可配置到 Best-2）。
            # 再做文件级上限，避免一个大 patch 把整个 Prompt 占满。
            # ----------------------------------------------------------
            hunk_winners: dict[
                str,
                CandidateRecord,
            ] = {}

            for hunk in path_hunks:
                ranked = sorted(
                    units,
                    key=lambda record: (
                        _hunk_record_key(
                            record,
                            hunk,
                        )
                    ),
                )

                for record in ranked[
                    : config.gold_units_per_hunk
                ]:
                    hunk_winners[
                        record.evidence_id
                    ] = record

            diagnostics[
                "gold_hunk_best_candidate_count_before_file_cap"
            ] += len(
                hunk_winners
            )

            winner_records = sorted(
                hunk_winners.values(),
                key=lambda record: (
                    _gold_file_record_key(
                        record,
                        issue_symbols=(
                            issue_symbols
                        ),
                        question_terms=(
                            question_terms
                        ),
                    )
                ),
            )

            kept_records = winner_records[
                : config.max_gold_units_per_file
            ]

            diagnostics[
                "gold_file_cap_dropped_count"
            ] += max(
                0,
                len(
                    winner_records
                )
                - len(
                    kept_records
                ),
            )

            for record in kept_records:
                record.sources.add(
                    "gold_patch_hunk"
                )
                record.gold_guided = True

                merged = (
                    _merge_record(
                        all_records,
                        record,
                    )
                )

                merged.sources.add(
                    "gold_patch_hunk"
                )
                merged.gold_guided = True

                gold_selected_ids.add(
                    record.evidence_id
                )

        else:
            # 只有 changed path、没有可解析 hunk 时，
            # 保守选择较小 scoreable unit。
            ranked = sorted(
                units,
                key=lambda record: (
                    max(
                        1,
                        record.end_line
                        - record.start_line
                        + 1,
                    ),
                    record.start_line,
                    record.evidence_id,
                ),
            )

            for record in ranked[
                : config.gold_units_per_file_without_hunk
            ]:
                record.sources.add(
                    "gold_patch_path"
                )
                record.gold_guided = True

                merged = (
                    _merge_record(
                        all_records,
                        record,
                    )
                )

                merged.sources.add(
                    "gold_patch_path"
                )
                merged.gold_guided = True

                gold_selected_ids.add(
                    record.evidence_id
                )

    diagnostics[
        "gold_guided_candidate_ids"
    ] = sorted(
        gold_selected_ids
    )

    # ------------------------------------------------------------------
    # 4. Issue 中明确写出的 repo path。
    # ------------------------------------------------------------------

    for path in issue_paths:
        units = get_path_units(
            path
        )

        if not units:
            diagnostics[
                "issue_paths_missing_in_pre_fix_snapshot"
            ].append(
                path
            )
            continue

        ranked = sorted(
            units,
            key=lambda record: (
                _question_path_record_key(
                    record,
                    question_terms,
                )
            ),
        )

        for record in ranked[
            : config.issue_path_units_per_file
        ]:
            record.sources.add(
                "issue_explicit_path"
            )

            merged = _merge_record(
                all_records,
                record,
            )

            merged.sources.add(
                "issue_explicit_path"
            )

    # ------------------------------------------------------------------
    # 5. Issue explicit symbol retrieval（问题显式符号检索）。
    #
    # 绝不做全仓库 symbol scan。
    #
    # 搜索范围：
    #   A. Gold changed path
    #   B. Issue explicit path
    #   C. 前若干个高排名 policy source path
    #
    # 这足以覆盖例如：
    #   “新增 get_mines_horizon，参考 get_pvgis_horizon”
    #
    # Gold changed file = pvgis.py
    # Issue symbol      = get_pvgis_horizon
    # => 直接从 pre-fix pvgis.py 中补真实函数 Evidence。
    # ------------------------------------------------------------------

    policy_source_paths: list[str] = []

    policy_records_for_paths = sorted(
        (
            item
            for item in all_records.values()
            if item.category()
            == "source"
        ),
        key=lambda item: (
            (
                item.min_online_rank
                if item.min_online_rank
                is not None
                else 2**31 - 1
            ),
            item.path,
            item.start_line,
        ),
    )

    for item in policy_records_for_paths:
        path = normalize_repo_path(
            item.path
        )

        if (
            not path
            or path in policy_source_paths
        ):
            continue

        policy_source_paths.append(
            path
        )

        if (
            len(
                policy_source_paths
            )
            >= config.issue_symbol_policy_path_limit
        ):
            break

    issue_symbol_search_paths = list(
        dict.fromkeys(
            [
                *changed_paths,
                *issue_paths,
                *policy_source_paths,
            ]
        )
    )

    diagnostics[
        "issue_symbol_search_paths"
    ] = issue_symbol_search_paths

    issue_symbol_candidate_ids: set[
        str
    ] = set()

    symbol_matches: dict[
        str,
        list[str],
    ] = {}

    for issue_symbol in issue_symbols:
        matches: list[
            CandidateRecord
        ] = []

        # 已经加载到 all_records 的 policy/original evidence 也先匹配。
        for record in all_records.values():
            if record_matches_issue_symbol(
                record,
                issue_symbol,
            ):
                matches.append(
                    record
                )

        # 再在高精度路径集合中查完整 pre-fix units。
        for path in issue_symbol_search_paths:
            for record in get_path_units(
                path
            ):
                if record_matches_issue_symbol(
                    record,
                    issue_symbol,
                ):
                    matches.append(
                        record
                    )

        deduped: dict[
            str,
            CandidateRecord,
        ] = {
            record.evidence_id: record
            for record in matches
        }

        ranked_matches = sorted(
            deduped.values(),
            key=lambda record: (
                _issue_symbol_record_key(
                    record,
                    issue_symbol,
                )
            ),
        )

        chosen = ranked_matches[
            : config.issue_symbol_units_per_symbol
        ]

        if not chosen:
            diagnostics[
                "issue_symbols_unmatched"
            ].append(
                issue_symbol
            )
            continue

        symbol_matches[
            issue_symbol
        ] = [
            record.evidence_id
            for record in chosen
        ]

        for record in chosen:
            record.sources.add(
                "issue_explicit_symbol"
            )

            merged = _merge_record(
                all_records,
                record,
            )

            merged.sources.add(
                "issue_explicit_symbol"
            )

            issue_symbol_candidate_ids.add(
                record.evidence_id
            )

    diagnostics[
        "issue_symbol_matches"
    ] = symbol_matches

    diagnostics[
        "issue_symbol_candidate_ids"
    ] = sorted(
        issue_symbol_candidate_ids
    )

    # ------------------------------------------------------------------
    # 6. 最终 Selection：强制证据先放，再按 priority + 去重 + 配额补齐。
    # ------------------------------------------------------------------
    ordered = sorted(all_records.values(), key=lambda item: item.priority_key())

    forced = [item for item in ordered if item.forced]
    selected: list[CandidateRecord] = []
    selected_ids: set[str] = set()

    for item in forced:
        if item.evidence_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item.evidence_id)

    # Gold-guided Evidence 是这次 v1.4 修复“关键源码没进入 Top24”的核心。
    # 因此它与 original witness 一样属于 preferred evidence：
    # 即使 preferred 数量导致最终候选略超过 candidate_limit，也不能为了固定 24 条
    # 把真实 changed-code Evidence 再次挤出去。
    for item in ordered:
        if not item.gold_guided:
            continue
        if item.evidence_id in selected_ids:
            continue
        if _is_redundant_with_selected(
            item,
            selected,
            config.overlap_threshold,
        ):
            diagnostics["overlap_dropped_count"] += 1
            continue
        selected.append(item)
        selected_ids.add(item.evidence_id)

    target = max(config.candidate_limit, len(selected))

    category_counts: Counter[str] = Counter(item.category() for item in selected)
    per_file_counts: Counter[str] = Counter(item.file_version_id for item in selected)

    for item in ordered:
        if len(selected) >= target:
            break
        if item.evidence_id in selected_ids:
            continue

        category = item.category()

        # Gold-guided / Issue explicit path 是高精度定位结果，不受普通 docs/test 配额限制，
        # 但仍允许 overlap dedup，避免同一 hunk 复制三份几乎相同代码。
        high_precision = bool(
            item.gold_guided
            or "issue_explicit_symbol" in item.sources
            or "issue_explicit_path" in item.sources
        )

        if not high_precision:
            if per_file_counts[item.file_version_id] >= config.max_per_file:
                diagnostics["per_file_limit_dropped_count"] += 1
                continue

            quota = None
            if category == "test":
                quota = config.test_quota
            elif category == "doc":
                quota = config.doc_quota
            elif category == "resource":
                quota = config.resource_quota
            elif category == "low_value":
                quota = config.low_value_quota

            if quota is not None and category_counts[category] >= quota:
                diagnostics["quota_dropped_count"] += 1
                continue

        if _is_redundant_with_selected(
            item,
            selected,
            config.overlap_threshold,
        ):
            diagnostics["overlap_dropped_count"] += 1
            continue

        selected.append(item)
        selected_ids.add(item.evidence_id)
        category_counts[category] += 1
        per_file_counts[item.file_version_id] += 1

    # 如果严格过滤后不足 target，不用垃圾文件强行填满。
    # Candidate Pool 不完整应该透明暴露给 Teacher/报告。
    diagnostics["candidate_pool_underfilled"] = len(selected) < target
    diagnostics["target_candidate_count"] = target
    diagnostics["selected_candidate_count"] = len(selected)
    diagnostics["forced_candidate_count"] = len(forced)
    diagnostics["selected_category_counts"] = dict(sorted(category_counts.items()))
    diagnostics["selected_source_counts"] = dict(
        sorted(
            Counter(
                source
                for item in selected
                for source in item.sources
            ).items()
        )
    )
    diagnostics["selected_paths"] = [item.path for item in selected]
    diagnostics["selected_evidence_ids"] = [item.evidence_id for item in selected]

    prompt_records = [item.to_prompt_record() for item in selected]

    return prompt_records, diagnostics


# ============================================================================
# Self-check（不依赖真实项目数据库）
# ============================================================================


def _self_check() -> None:
    assert classify_path("src/pkg/core.py") == "source"
    assert classify_path("tests/test_core.py") == "test"
    assert classify_path("docs/guide.rst") == "doc"
    assert classify_path("docs/whatsnew/1.0.rst") == "low_value"
    assert classify_path("data/example.csv") == "resource"
    assert classify_path("pkg/config.json") == "resource"

    symbols = extract_issue_symbols(
        "Please add function get_mines_horizon similar to `get_pvgis_horizon`. "
        "Then call ModelChain.prepare_inputs(). "
        "SELECT DISTINCT TABLE FIELD github.com file.sql Traceback"
    )

    assert "get_mines_horizon" in symbols
    assert "get_pvgis_horizon" in symbols
    assert any(
        symbol.endswith("prepare_inputs")
        for symbol in symbols
    )

    assert "SELECT" not in symbols
    assert "DISTINCT" not in symbols
    assert "TABLE" not in symbols
    assert "FIELD" not in symbols
    assert "github.com" not in symbols
    assert "file.sql" not in symbols
    assert "Traceback" not in symbols

    # 普通自然语言中的 "in xxx" 不能成为代码 symbol。
    prose_symbols = extract_issue_symbols(
        "The value is in user input and in the result. "
        "This is for validation and is added in solution."
    )

    assert "user" not in prose_symbols
    assert "the" not in prose_symbols
    assert "for" not in prose_symbols
    assert "solution" not in prose_symbols

    # 真正的 Python Traceback frame 仍然应该提取。
    traceback_symbols = extract_issue_symbols(
        'Traceback (most recent call last):\n'
        '  File "/tmp/demo.py", line 42, in _do_load\n'
        '    value = obj.load()\n'
    )

    assert "_do_load" in traceback_symbols

    demo = CandidateRecord(
        evidence_id="ev_demo",
        file_version_id="fv_demo",
        path="pvlib/iotools/pvgis.py",
        unit_type="function",
        symbol="pvlib.iotools.pvgis.get_pvgis_horizon",
        start_line=100,
        end_line=150,
        content="def get_pvgis_horizon(...):\n    pass",
        rendered_token_count=20,
    )

    assert record_matches_issue_symbol(
        demo,
        "get_pvgis_horizon",
    )

    no_symbol_demo = CandidateRecord(
        evidence_id="ev_no_symbol",
        file_version_id="fv_demo",
        path="pydicom/filereader.py",
        unit_type="code_block",
        symbol=None,
        start_line=100,
        end_line=140,
        content=(
            "def dcmread(fp, force=False):\n"
            "    return fp\n"
        ),
        rendered_token_count=20,
    )

    assert record_matches_issue_symbol(
        no_symbol_demo,
        "dcmread",
    )

    assert line_overlap_ratio(1, 10, 6, 15) == 0.5
    assert line_overlap_ratio(1, 10, 20, 30) == 0.0

    changed, hunks = parse_pre_fix_patch_targets(
        "diff --git a/pkg/a.py b/pkg/a.py\n"
        "--- a/pkg/a.py\n"
        "+++ b/pkg/a.py\n"
        "@@ -10,5 +10,7 @@ def run(x):\n"
        "-old\n"
        "+new\n"
    )
    assert changed == ["pkg/a.py"]
    assert len(hunks) == 1
    assert hunks[0].old_start == 10
    assert hunks[0].old_length == 5
    assert hunks[0].symbol_hint == "run"

    paths = extract_issue_paths(
        "Please inspect `pkg/a.py` and docs/guide.rst."
    )
    assert "pkg/a.py" in paths
    assert "docs/guide.rst" in paths


if __name__ == "__main__":
    _self_check()
    print("refinement_candidate_builder v1.5.2 self-check: PASS")
