"""
Tool Registry — Manages all tool plugins, discovery, and execution.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Optional

from rich.console import Console

from zeta_cli.config.manager import ConfigManager
from zeta_cli.tools.base import BaseTool, ToolResult

console = Console()

class ToolRegistry:
    """
    Central registry for all tools.

    Features:
    - Plugin discovery
    - Dynamic tool loading
    - Tool execution with validation
    - Schema generation for LLM function calling
    - Execution tracking and logging
    """

    def __init__(self, config: ConfigManager):
        self._config = config
        self._tools: dict[str, BaseTool] = {}
        self._execution_history: list[dict] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Discover and register all tools."""
        await self._discover_tools()
        console.print(f"[dim]Loaded {len(self._tools)} tools: {', '.join(sorted(self._tools.keys()))}[/dim]")
        self._initialized = True

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> list[BaseTool]:
        """List all registered tools, optionally filtered by category."""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return sorted(tools, key=lambda t: t.name)

    def list_categories(self) -> list[str]:
        """List all tool categories."""
        return sorted(set(t.category for t in self._tools.values()))

    async def execute(self, name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name with given parameters.

        Args:
            name: Tool name
            **kwargs: Tool parameters

        Returns:
            ToolResult from execution
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{name}' not found. Available: {', '.join(sorted(self._tools.keys()))}",
            )

        # Check sandbox requirement
        if tool.requires_sandbox and not self._config.get("tools.sandbox_enabled", True):
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{name}' requires sandbox mode which is currently disabled.",
            )

        # Validate parameters
        if not tool.validate_params(kwargs):
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid parameters for tool '{name}'.",
            )

        try:
            result = await tool.execute(**kwargs)
        except Exception as e:
            result = ToolResult(
                success=False,
                output="",
                error=f"Tool '{name}' execution failed: {e}",
            )

        # Log execution
        self._execution_history.append({
            "tool": name,
            "params": kwargs,
            "success": result.success,
            "error": result.error,
            "output_length": len(result.output),
        })

        # Trim history
        if len(self._execution_history) > 1000:
            self._execution_history = self._execution_history[-500:]

        return result

    def get_schemas(self, tool_names: Optional[list[str]] = None) -> list[dict]:
        """
        Get OpenAI-compatible function schemas for tools.

        Args:
            tool_names: Specific tools to get schemas for, or all

        Returns:
            List of tool schemas
        """
        if tool_names:
            tools = [self._tools[n] for n in tool_names if n in self._tools]
        else:
            tools = list(self._tools.values())

        return [t.get_schema() for t in tools]

    def get_execution_stats(self) -> dict:
        """Get tool execution statistics."""
        total = len(self._execution_history)
        successful = sum(1 for e in self._execution_history if e["success"])
        return {
            "total_executions": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": f"{(successful / total * 100):.1f}%" if total > 0 else "N/A",
            "tool_usage": self._get_tool_usage_counts(),
        }

    def _get_tool_usage_counts(self) -> dict:
        """Count executions per tool."""
        counts = {}
        for entry in self._execution_history:
            name = entry["tool"]
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    async def _discover_tools(self) -> None:
        """Discover and load all tool plugins from the tools directory."""
        # Import built-in tools
        from zeta_cli.tools import builtin_tools

        for name, obj in inspect.getmembers(builtin_tools):
            if inspect.isclass(obj) and issubclass(obj, BaseTool) and obj is not BaseTool:
                self.register(obj())

        # Scan for external tool plugins
        tools_dir = Path(__file__).parent / "plugins"
        if tools_dir.exists():
            for module_info in pkgutil.iter_modules([str(tools_dir)]):
                try:
                    module = importlib.import_module(f"zeta_cli.tools.plugins.{module_info.name}")
                    for name, obj in inspect.getmembers(module):
                        if (
                            inspect.isclass(obj)
                            and issubclass(obj, BaseTool)
                            and obj is not BaseTool
                            and obj.__module__ == module.__name__
                        ):
                            self.register(obj())
                            console.print(f"[dim]  Loaded plugin tool: {obj.name}[/dim]")
                except Exception as e:
                    console.print(f"[yellow]  Failed to load tool plugin '{module_info.name}': {e}[/yellow]")

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def shutdown(self) -> None:
        """Clean up tool resources."""
        for tool in self._tools.values():
            if hasattr(tool, "shutdown"):
                await tool.shutdown()
        self._tools.clear()
        self._initialized = False
