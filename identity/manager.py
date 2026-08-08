"""Identity management subsystem for Zeta CLI."""

import json
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from zeta_cli.config.manager import ConfigManager

console = Console()

class IdentityManager:
    """
    Persistent AI identity store.

    Stores:
    - Name
    - Mission
    - Principles
    - Preferences
    - Objectives
    - Learned strategies
    - Capability history
    """

    def __init__(self, config: ConfigManager):
        self._config = config
        self._identity_file = Path(self._config.get("system.workspace")) / ".zeta" / "identity.json"
        self._identity: dict[str, Any] = {}
        self._initialized = False

    async def initialize(self) -> None:
        self._load_defaults()
        await self._load_persisted()
        self._initialized = True
        console.print("[dim]Identity loaded.[/dim]")

    def _load_defaults(self) -> None:
        self._identity = {
            "name": self._config.get("identity.name", "Builder Agent"),
            "mission": self._config.get("identity.mission", "Autonomously reason, execute, and improve code."),
            "principles": self._config.get("identity.principles", []),
            "objectives": self._config.get("identity.objectives", []),
            "preferences": self._config.get("identity.preferences", {}),
            "strategies": self._config.get("identity.strategies", []),
            "capability_history": self._config.get("identity.capability_history", []),
        }

    async def _load_persisted(self) -> None:
        if not self._identity_file.exists():
            return

        try:
            with open(self._identity_file, "r", encoding="utf-8") as f:
                persisted = json.load(f)
            self._identity.update(persisted)
        except Exception:
            console.print("[yellow]Warning: could not load persisted identity.[/yellow]")

    async def save(self) -> None:
        self._identity_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._identity_file, "w", encoding="utf-8") as f:
            json.dump(self._identity, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split(".")
        current = self._identity
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        current = self._identity
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def get_all(self) -> dict:
        return self._identity.copy()

    async def add_strategy(self, strategy: str) -> None:
        self._identity.setdefault("strategies", []).append(strategy)
        await self.save()

    async def add_capability_history(self, entry: dict[str, Any]) -> None:
        self._identity.setdefault("capability_history", []).append(entry)
        await self.save()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def shutdown(self) -> None:
        await self.save()
        self._initialized = False
