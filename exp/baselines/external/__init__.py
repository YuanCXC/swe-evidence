"""外部代码定位方法的冻结输出适配器。"""

from .agentless import AgentlessBaseline, AgentlessOutputStore
from .locagent import LocAgentBaseline, LocAgentOutputStore
from .swerank import SweRankBaseline, SweRankOutputStore


__all__ = [
    "AgentlessBaseline",
    "AgentlessOutputStore",
    "LocAgentBaseline",
    "LocAgentOutputStore",
    "SweRankBaseline",
    "SweRankOutputStore",
]
