"""
base_agent.py - Base Agent Class adhering to SOLID principles.

Guarantees:
- Every agent has a clear single role and identifier.
- Every agent can be exported as a Semantic Kernel Plugin.
- Enforces execution contract where agents explicitly read and write the MineBlackboard.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from blackboard import MineBlackboard
from sk_compat import KernelPlugin


class BaseAgent(ABC):
    """Abstract Base Class for all RockWatch domain agents."""

    def __init__(self, agent_id: str, role: str, plugin_name: str):
        self.agent_id = agent_id
        self.role = role
        self.plugin_name = plugin_name

    def to_plugin(self) -> KernelPlugin:
        """Exports this agent's @kernel_function tools as a Semantic Kernel Plugin."""
        return KernelPlugin.from_object(self.plugin_name, self, description=self.role)

    @abstractmethod
    def run(self, blackboard: MineBlackboard, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the agent's core responsibility:
        Reads from input partitions, invokes tools, writes to output partitions.
        """
        pass
