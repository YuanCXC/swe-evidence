"""从候选文件中读取并排序已有 Evidence Units。"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Mapping, Sequence

from src.data import RuntimeRepository


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "when",
    "then",
    "into",
    "does",
    "not",
    "are",
    "was",
    "were",
    "have",
    "has",
    "should",
    "error",
    "issue",
    "bug",
    "fix",
    "using",
    "use",
    "return",
    "python",
}


def retrieval_terms(text: str) -> list[str]:
    """按冻结 V2.10 规则切分英文标识符与检索词。"""

    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).replace("_", " ")
    terms = [term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9]{1,}", expanded)]
    return [term for term in terms if term not in STOPWORDS]


def collect_candidate_units(
    repository: RuntimeRepository,
    file_version_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """读取候选文件中全部可评分 Evidence Units，并按 ID 去重。"""

    records: dict[str, dict[str, Any]] = {}
    for file_version_id in dict.fromkeys(map(str, file_version_ids)):
        for unit in repository.get_file_evidence(file_version_id, scoreable_only=True):
            records[str(unit["evidence_id"])] = unit
    return list(records.values())


def _bm25_ranking(
    evidence_units: Sequence[Mapping[str, Any]],
    terms: Sequence[str],
    *,
    limit: int,
) -> list[str]:
    query_terms = list(dict.fromkeys(terms))
    documents = {
        str(unit["evidence_id"]): retrieval_terms(str(unit.get("content") or ""))
        for unit in evidence_units
    }
    term_sets = {evidence_id: set(values) for evidence_id, values in documents.items()}
    frequencies = {
        evidence_id: Counter(values) for evidence_id, values in documents.items()
    }
    lengths = {evidence_id: len(values) for evidence_id, values in documents.items()}
    document_count = max(1, len(documents))
    average_length = sum(lengths.values()) / document_count or 1.0
    document_frequency = {
        term: sum(term in values for values in term_sets.values())
        for term in query_terms
    }

    scores = []
    for evidence_id, counts in frequencies.items():
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.2 * (
                1.0 - 0.75 + 0.75 * lengths[evidence_id] / average_length
            )
            score += inverse_frequency * frequency * 2.2 / denominator
        if score > 0.0:
            scores.append((score, evidence_id))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [evidence_id for _, evidence_id in scores[:limit]]


def rank_units_by_query(
    evidence_units: Sequence[Mapping[str, Any]],
    query_groups: Mapping[str, Sequence[str]],
    *,
    limit: int = 64,
) -> dict[str, list[str]]:
    """为检索计划中的各组查询生成 Evidence Unit 内容排名。"""

    return {
        name: _bm25_ranking(
            evidence_units,
            terms,
            limit=limit,
        )
        for name, terms in query_groups.items()
    }
