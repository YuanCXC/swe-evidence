"""根据 Issue 中的路径与符号线索排序 Evidence Units。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .unit_retriever import retrieval_terms


def rank_path_units(
    evidence_units: Sequence[Mapping[str, Any]],
    query_terms: Sequence[str],
    explicit_paths: Sequence[str],
    *,
    limit: int = 64,
) -> list[str]:
    """按路径词重合与显式路径命中排序。"""

    term_set = set(query_terms)
    path_set = {path.replace("\\", "/").lower() for path in explicit_paths}
    scores = []
    for unit in evidence_units:
        path = str(unit["path"]).replace("\\", "/")
        path_terms = set(retrieval_terms(path))
        overlap = len(term_set & path_terms)
        exact = int(path.lower() in path_set)
        suffix = int(any(path.lower().endswith(item) for item in path_set))
        score = overlap + 4 * exact + 2 * suffix
        if score:
            scores.append((score, str(unit["evidence_id"])))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [evidence_id for _, evidence_id in scores[:limit]]


def rank_symbol_units(
    evidence_units: Sequence[Mapping[str, Any]],
    query_terms: Sequence[str],
    explicit_symbols: Sequence[str],
    *,
    limit: int = 64,
) -> list[str]:
    """按 symbol 与 qualified_name 的词重合和显式命中排序。"""

    term_set = set(query_terms)
    symbol_set = {symbol.lower() for symbol in explicit_symbols}
    scores = []
    for unit in evidence_units:
        symbol = str(unit.get("qualified_name") or unit.get("symbol") or "")
        symbol_terms = set(retrieval_terms(symbol))
        overlap = len(term_set & symbol_terms)
        exact = int(symbol.lower() in symbol_set)
        suffix = int(any(symbol.lower().endswith(item) for item in symbol_set))
        score = overlap + 4 * exact + 2 * suffix
        if score:
            scores.append((score, str(unit["evidence_id"])))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [evidence_id for _, evidence_id in scores[:limit]]
