"""根据 Current K 扩展已有仓库结构关系。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from src.data import RuntimeRepository


RELATION_WEIGHTS = {
    "parent": 4.0,
    "child": 3.0,
    "sibling": 2.0,
    "same_file": 1.0,
    "import": 2.5,
}


class StructureExpander:
    """扩展 parent、children、siblings、同文件邻域和 imports。"""

    def __init__(self, repository: RuntimeRepository) -> None:
        self.repository = repository

    def _import_file_ids(
        self,
        snapshot_files: Sequence[Mapping[str, Any]],
        imports: Sequence[Mapping[str, Any]],
    ) -> list[str]:
        suffixes = []
        for item in imports:
            module = str(item.get("module") or "").strip(".")
            if not module:
                continue
            module_path = module.replace(".", "/")
            suffixes.extend((module_path + ".py", module_path + "/__init__.py"))
        return [
            str(file["file_version_id"])
            for file in snapshot_files
            if any(str(file["path"]).endswith(suffix) for suffix in suffixes)
        ]

    def expand(
        self,
        task: Mapping[str, Any],
        current_evidence: Sequence[Mapping[str, Any]],
        *,
        excluded_evidence_ids: Sequence[str] = (),
        limit: int = 64,
    ) -> list[dict[str, Any]]:
        """返回按结构关系强度排序且未被历史检索的 Evidence Units。"""

        selected_ids = {
            *map(str, excluded_evidence_ids),
            *(str(unit["evidence_id"]) for unit in current_evidence),
        }
        scores: dict[str, float] = defaultdict(float)
        sources: dict[str, set[str]] = defaultdict(set)
        records: dict[str, dict[str, Any]] = {}
        snapshot_files = self.repository.list_snapshot_files(str(task["snapshot_id"]))

        def add(
            unit: Mapping[str, Any], relation: str, weight: float | None = None
        ) -> None:
            evidence_id = str(unit["evidence_id"])
            if evidence_id in selected_ids:
                return
            records[evidence_id] = dict(unit)
            scores[evidence_id] += weight or RELATION_WEIGHTS[relation]
            sources[evidence_id].add(relation)

        for selected in current_evidence:
            context = self.repository.get_structure_context(
                str(selected["evidence_id"])
            )
            if context["parent"]:
                add(context["parent"], "parent")
            for unit in context["children"]:
                add(unit, "child")
            for unit in context["siblings"]:
                add(unit, "sibling")

            current_line = int(context["current"]["start_line"])
            neighbors = sorted(
                context["same_file"],
                key=lambda unit: (
                    abs(int(unit["start_line"]) - current_line),
                    str(unit["evidence_id"]),
                ),
            )[:8]
            for unit in neighbors:
                distance = abs(int(unit["start_line"]) - current_line)
                add(unit, "same_file", RELATION_WEIGHTS["same_file"] / (1 + distance))

            import_file_ids = self._import_file_ids(snapshot_files, context["imports"])
            for file_version_id in import_file_ids:
                for unit in self.repository.get_file_evidence(
                    file_version_id,
                    scoreable_only=True,
                )[:4]:
                    add(unit, "import")

        ordered = sorted(
            scores, key=lambda evidence_id: (-scores[evidence_id], evidence_id)
        )[:limit]
        return [
            {
                **records[evidence_id],
                "structure_score": scores[evidence_id],
                "structure_sources": sorted(sources[evidence_id]),
            }
            for evidence_id in ordered
        ]
