"""
Base classes for specialized agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class AgentResult:
    """Result from agent execution."""
    success: bool
    output: str
    agent_name: str
    confidence: float = 1.0
    suggestions: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

class BaseAgent(ABC):
    """
    Abstract base for all specialized agents.

    Each agent has a specific responsibility:
    - Planner: Task decomposition and planning
    - Coder: Code generation
    - Reviewer: Code review
    - Debugger: Error analysis and fixes
    - Researcher: Information gathering
    - Architect: System design
    - Security: Security analysis
    - Performance: Performance optimization
    - Tester: Test generation
    - Documentation: Documentation generation
    """

    name: str = "base_agent"
    description: str = "Base agent"
    system_prompt: str = "You are a helpful assistant."

    @abstractmethod
    async def execute(self, task: str, context: dict, **kwargs) -> AgentResult:
        """
        Execute the agent's task.

        Args:
            task: The task description
            context: Current context (conversation, files, etc.)
            **kwargs: Additional parameters

        Returns:
            AgentResult with output
        """
        ...

    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return self.system_prompt

    def __repr__(self) -> str:
        return f"Agent({self.name})"
