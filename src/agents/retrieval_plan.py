"""定义 Agent 每轮传给 RAG 的结构化检索计划。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


EVIDENCE_DIMENSIONS = (
    "fault_location",
    "fault_logic",
    "dependency_context",
    "state_flow",
    "behavior_constraint",
    "repair_scope",
    "validation_constraint",
)

RETRIEVAL_CHANNELS = ("content", "path", "symbol", "structure")


@dataclass(frozen=True)
class RetrievalPlan:
    """描述当前信息缺口和准备执行的仓库检索。"""

    information_gap: str
    target_dimensions: tuple[str, ...]
    queries: tuple[str, ...]
    paths: tuple[str, ...]
    symbols: tuple[str, ...]
    retrieval_channels: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> RetrievalPlan:
        """从规划模型输出的 JSON 对象创建检索计划。"""

        return cls(
            information_gap=str(value["information_gap"]),
            target_dimensions=tuple(map(str, value["target_dimensions"])),
            queries=tuple(map(str, value["queries"])),
            paths=tuple(map(str, value["paths"])),
            symbols=tuple(map(str, value["symbols"])),
            retrieval_channels=tuple(map(str, value["retrieval_channels"])),
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为轨迹中可直接序列化的记录。"""

        return {
            "information_gap": self.information_gap,
            "target_dimensions": list(self.target_dimensions),
            "queries": list(self.queries),
            "paths": list(self.paths),
            "symbols": list(self.symbols),
            "retrieval_channels": list(self.retrieval_channels),
        }
