"""
Zeta CLI — Production-grade AI coding assistant for Windows terminal.
Quality target: Codex CLI + Claude Code + OpenHands + Devin + Cursor Agent combined.
"""

__version__ = "1.0.0"
__author__ = "Alpha"
__license__ = "Zeta Proprietary"

from zeta_cli.core.engine import ExecutionEngine
from zeta_cli.planner.planner import TaskPlanner
from zeta_cli.memory.manager import MemoryManager
from zeta_cli.tools.registry import ToolRegistry
from zeta_cli.agents.manager import AgentManager
from zeta_cli.api.manager import APIManager
from zeta_cli.config.manager import ConfigManager
from zeta_cli.security.manager import SecurityManager
from zeta_cli.skills.manager import SkillManager

__all__ = [
    "ExecutionEngine",
    "TaskPlanner",
    "MemoryManager",
    "ToolRegistry",
    "AgentManager",
    "APIManager",
    "ConfigManager",
    "SecurityManager",
    "SkillManager",
]
