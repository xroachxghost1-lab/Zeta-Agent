"""
Base classes for the plugin-based tool system.
All tools inherit from BaseTool.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: str
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    artifacts: list = field(default_factory=list)

class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Tools are plugins that extend the system's capabilities.
    Each tool has a name, description, and parameter schema.
    """

    name: str = "base_tool"
    description: str = "Base tool description"
    category: str = "general"
    requires_sandbox: bool = False
    is_destructive: bool = False

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with execution output
        """
        ...

    def get_schema(self) -> dict:
        """
        Get the OpenAI-compatible function schema for this tool.

        Returns:
            Dict with 'type', 'function' keys
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_schema(),
            },
        }

    @abstractmethod
    def get_parameters_schema(self) -> dict:
        """
        Get the JSON Schema for tool parameters.

        Returns:
            Dict with JSON Schema properties
        """
        ...

    def validate_params(self, params: dict) -> bool:
        """Validate parameters against schema. Override for custom validation."""
        return True

    def __repr__(self) -> str:
        return f"Tool({self.name}, category={self.category})"
