"""
Agent Manager — Manages all specialized agents with task routing and collaboration.
"""

from typing import Any, Optional

from rich.console import Console

from zeta_cli.config.manager import ConfigManager
from zeta_cli.api.manager import APIManager
from zeta_cli.tools.registry import ToolRegistry
from zeta_cli.memory.manager import MemoryManager
from zeta_cli.agents.base import BaseAgent, AgentResult
from zeta_cli.agents.specialized import (
    CoderAgent,
    ReviewerAgent,
    DebuggerAgent,
    ResearcherAgent,
    ArchitectAgent,
    SecurityAgent,
    PerformanceAgent,
    TesterAgent,
    DocumentationAgent,
)

console = Console()

class AgentManager:
    """
    Central agent management system.

    Features:
    - Agent registration and discovery
    - Task routing to appropriate agent
    - Multi-agent collaboration
    - Agent performance tracking
    - Extensible agent plugin system
    """

    def __init__(
        self,
        config: ConfigManager,
        api: APIManager,
        tools: ToolRegistry,
        memory: MemoryManager,
    ):
        self._config = config
        self._api = api
        self._tools = tools
        self._memory = memory
        self._agents: dict[str, BaseAgent] = {}
        self._execution_history: list[dict] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all agents."""
        # Register built-in agents
        self.register(CoderAgent(self._api, self._tools, self._memory))
        self.register(ReviewerAgent(self._api, self._tools, self._memory))
        self.register(DebuggerAgent(self._api, self._tools, self._memory))
        self.register(ResearcherAgent(self._api, self._tools, self._memory))
        self.register(ArchitectAgent(self._api, self._tools, self._memory))
        self.register(SecurityAgent(self._api, self._tools, self._memory))
        self.register(PerformanceAgent(self._api, self._tools, self._memory))
        self.register(TesterAgent(self._api, self._tools, self._memory))
        self.register(DocumentationAgent(self._api, self._tools, self._memory))

        console.print(f"[dim]Initialized {len(self._agents)} agents: {', '.join(sorted(self._agents.keys()))}[/dim]")
        self._initialized = True

    def register(self, agent: BaseAgent) -> None:
        """Register an agent."""
        self._agents[agent.name] = agent

    def get(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self._agents.get(name)

    def list_agents(self) -> list[BaseAgent]:
        """List all registered agents."""
        return sorted(self._agents.values(), key=lambda a: a.name)

    async def execute(self, agent_name: str, task: str, context: dict, **kwargs) -> AgentResult:
        """
        Execute a specific agent.

        Args:
            agent_name: Name of the agent to execute
            task: Task description
            context: Current context
            **kwargs: Additional parameters

        Returns:
            AgentResult
        """
        agent = self._agents.get(agent_name)
        if not agent:
            return AgentResult(
                success=False,
                output=f"Agent '{agent_name}' not found.",
                agent_name=agent_name,
                confidence=0.0,
            )

        result = await agent.execute(task, context, **kwargs)

        self._execution_history.append({
            "agent": agent_name,
            "task": task[:200],
            "success": result.success,
            "confidence": result.confidence,
        })

        return result

    async def route_task(self, task: str, context: dict) -> AgentResult:
        """
        Automatically route a task to the most appropriate agent.

        Uses keyword matching and task analysis to select the best agent.
        """
        task_lower = task.lower()

        # Route based on task content
        if any(w in task_lower for w in ["code", "implement", "write", "function", "class", "module"]):
            return await self.execute("coder", task, context)
        elif any(w in task_lower for w in ["review", "check", "audit", "inspect"]):
            return await self.execute("reviewer", task, context)
        elif any(w in task_lower for w in ["debug", "fix", "error", "bug", "exception", "traceback"]):
            return await self.execute("debugger", task, context)
        elif any(w in task_lower for w in ["research", "find", "search", "lookup", "document"]):
            return await self.execute("researcher", task, context)
        elif any(w in task_lower for w in ["architecture", "design", "structure", "system", "pattern"]):
            return await self.execute("architect", task, context)
        elif any(w in task_lower for w in ["security", "vulnerability", "exploit", "safe", "protect"]):
            return await self.execute("security", task, context)
        elif any(w in task_lower for w in ["performance", "optimize", "speed", "slow", "memory", "profile"]):
            return await self.execute("performance", task, context)
        elif any(w in task_lower for w in ["test", "unit test", "coverage", "assert", "pytest"]):
            return await self.execute("tester", task, context)
        elif any(w in task_lower for w in ["document", "readme", "docstring", "comment", "explain"]):
            return await self.execute("documentation", task, context)
        else:
            # Default to coder for general tasks
            return await self.execute("coder", task, context)

    async def collaborate(self, task: str, context: dict, agents: list[str]) -> dict[str, AgentResult]:
        """
        Execute multiple agents and combine results.

        Args:
            task: The overall task
            context: Shared context
            agents: List of agent names to collaborate

        Returns:
            Dict mapping agent name to result
        """
        results = {}
        for agent_name in agents:
            results[agent_name] = await self.execute(agent_name, task, context)
        return results

    def get_stats(self) -> dict:
        """Get agent execution statistics."""
        total = len(self._execution_history)
        successful = sum(1 for e in self._execution_history if e["success"])
        return {
            "total_executions": total,
            "successful": successful,
            "failed": total - successful,
            "agent_usage": self._get_agent_usage(),
        }

    def _get_agent_usage(self) -> dict:
        """Count executions per agent."""
        counts = {}
        for entry in self._execution_history:
            name = entry["agent"]
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def shutdown(self) -> None:
        """Clean up agent resources."""
        self._agents.clear()
        self._initialized = False
