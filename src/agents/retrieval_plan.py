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

        fields = (
            "target_dimensions",
            "queries",
            "paths",
            "symbols",
            "retrieval_channels",
        )
        for field in fields:
            if not isinstance(value.get(field), list):
                raise ValueError(f"Planner 字段 {field} 必须是 JSON 数组")
        dimensions = tuple(map(str, value["target_dimensions"]))
        channels = tuple(map(str, value["retrieval_channels"]))
        invalid_dimensions = set(dimensions) - set(EVIDENCE_DIMENSIONS)
        invalid_channels = set(channels) - set(RETRIEVAL_CHANNELS)
        if invalid_dimensions:
            raise ValueError(f"Planner 输出了未知证据维度：{sorted(invalid_dimensions)}")
        if invalid_channels:
            raise ValueError(f"Planner 输出了未知检索通道：{sorted(invalid_channels)}")
        if not channels:
            raise ValueError("Planner 的 retrieval_channels 不能为空")
        information_gap = value.get("information_gap")
        if not isinstance(information_gap, str) or not information_gap.strip():
            raise ValueError("Planner 的 information_gap 必须是非空字符串")
        return cls(
            information_gap=information_gap.strip(),
            target_dimensions=dimensions,
            queries=tuple(map(str, value["queries"])),
            paths=tuple(map(str, value["paths"])),
            symbols=tuple(map(str, value["symbols"])),
            retrieval_channels=channels,
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
