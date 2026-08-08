"""
Skill manager for capabilities, skill sets, and agent orchestration.
"""

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from zeta_cli.config.manager import ConfigManager
from zeta_cli.api.manager import APIManager
from zeta_cli.tools.registry import ToolRegistry
from zeta_cli.memory.manager import MemoryManager
from zeta_cli.agents.manager import AgentManager

console = Console()

@dataclass
class SkillDefinition:
    name: str
    description: str
    components: list[str]
    success_rate: float = 0.0
    tools: list[str] = field(default_factory=list)
    enabled: bool = True

@dataclass
class CapabilityDefinition:
    name: str
    tools: list[str]
    skills: list[str]
    success_rate: float = 0.0
    enabled: bool = True

class SkillManager:
    """
    Manages high-level skill sets, capability metadata, and agent routing support.

    Features:
    - Skill discovery and metadata
    - Capability mapping to tools and agent skills
    - System readiness reporting
    - Runtime skill activation and evaluation
    """

    def __init__(self, config: ConfigManager):
        self._config = config
        self._skills: dict[str, SkillDefinition] = {}
        self._capabilities: dict[str, CapabilityDefinition] = {}
        self._initialized = False
        self._skills_file = Path(self._config.get("system.workspace")) / ".zeta" / "skills.json"

    async def initialize(self) -> None:
        """Load skills and capabilities from config and persisted state."""
        self._load_defaults()
        await self._load_persisted()
        self._initialized = True
        console.print(f"[dim]Loaded {len(self._skills)} skills and {len(self._capabilities)} capabilities.[/dim]")

    def _load_defaults(self) -> None:
        skills = self._config.get("skills", {}) or {}
        capabilities = self._config.get("capabilities", {}) or {}

        for skill_key, skill_data in skills.items():
            self._skills[skill_key] = SkillDefinition(
                name=skill_data.get("name", skill_key),
                description=skill_data.get("description", ""),
                components=skill_data.get("components", []),
                success_rate=skill_data.get("success_rate", 0.0),
                tools=skill_data.get("tools", []),
                enabled=skill_data.get("enabled", True),
            )

        for cap_key, cap_data in capabilities.items():
            self._capabilities[cap_key] = CapabilityDefinition(
                name=cap_data.get("name", cap_key),
                tools=cap_data.get("tools", []),
                skills=cap_data.get("skills", []),
                success_rate=cap_data.get("success_rate", 0.0),
                enabled=cap_data.get("enabled", True),
            )

    async def _load_persisted(self) -> None:
        if not self._skills_file.exists():
            return

        try:
            with open(self._skills_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for skill_key, skill_data in payload.get("skills", {}).items():
                self._skills[skill_key] = SkillDefinition(**skill_data)
            for cap_key, cap_data in payload.get("capabilities", {}).items():
                self._capabilities[cap_key] = CapabilityDefinition(**cap_data)
        except Exception:
            console.print("[yellow]Warning: could not load persisted skills file.[/yellow]")

    async def save(self) -> None:
        self._skills_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "skills": {k: self._to_dict(v) for k, v in self._skills.items()},
            "capabilities": {k: self._to_dict(v) for k, v in self._capabilities.items()},
        }
        with open(self._skills_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _to_dict(self, obj: Any) -> dict:
        return {
            k: v for k, v in vars(obj).items()
            if not k.startswith("_")
        }

    def list_skills(self) -> list[SkillDefinition]:
        return sorted(self._skills.values(), key=lambda s: s.name)

    def list_capabilities(self) -> list[CapabilityDefinition]:
        return sorted(self._capabilities.values(), key=lambda c: c.name)

    def get_skill(self, name: str) -> Optional[SkillDefinition]:
        return self._skills.get(name)

    def get_capability(self, name: str) -> Optional[CapabilityDefinition]:
        return self._capabilities.get(name)

    def activate_skill(self, name: str) -> bool:
        if name in self._skills:
            self._skills[name].enabled = True
            return True
        return False

    def deactivate_skill(self, name: str) -> bool:
        if name in self._skills:
            self._skills[name].enabled = False
            return True
        return False

    def recommend_agents_for_skill(self, skill_name: str) -> list[str]:
        """Return agent names recommended for a skill."""
        skill = self._skills.get(skill_name)
        if not skill:
            return []

        agents = []
        if "Planning" in skill.components:
            agents.append("coder")
            agents.append("architect")
        if "Coding" in skill.components:
            agents.append("coder")
            agents.append("debugger")
        if "Testing" in skill.components:
            agents.append("tester")
        if "Debugging" in skill.components:
            agents.append("debugger")
        if "Retrieval" in skill.components:
            agents.append("researcher")
        if "Compression" in skill.components:
            agents.append("performance")
        return sorted(set(agents))

    def recommend_tools_for_capability(self, capability_name: str) -> list[str]:
        capability = self._capabilities.get(capability_name)
        return capability.tools if capability else []

    def get_system_summary(self) -> dict:
        return {
            "skills": [self._to_dict(s) for s in self.list_skills()],
            "capabilities": [self._to_dict(c) for c in self.list_capabilities()],
        }

    async def describe_skill(self, name: str) -> str:
        skill = self._skills.get(name)
        if not skill:
            return f"Skill '{name}' not found."

        tools = ", ".join(skill.tools) if skill.tools else "none"
        return (
            f"Skill: {skill.name}\n"
            f"Description: {skill.description}\n"
            f"Components: {', '.join(skill.components)}\n"
            f"Tools: {tools}\n"
            f"Success Rate: {skill.success_rate:.1%}\n"
            f"Enabled: {skill.enabled}"
        )

    async def shutdown(self) -> None:
        await self.save()
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized
