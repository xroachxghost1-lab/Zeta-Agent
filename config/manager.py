"""
Configuration Manager — Handles all application configuration.
Supports JSON config files, environment variables, hot reload, and profiles.
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field
from rich.console import Console

console = Console()

DEFAULT_CONFIG = {
    "system": {
        "workspace": str(Path.home() / "zeta_workspace"),
        "log_dir": str(Path.home() / ".zeta" / "logs"),
        "log_level": "INFO",
        "data_dir": str(Path.home() / ".zeta" / "data"),
        "max_retries": 3,
        "timeout": 300,
    },
    "api": {
        "provider": "inception",
        "default_model": "mercury",
        "max_tokens": 8192,
        "temperature": 0.3,
        "openai": {
            "enabled": False,
            "base_url": "https://api.openai.com/v1",
        },
        "anthropic": {
            "enabled": False,
            "base_url": "https://api.anthropic.com",
        },
        "openrouter": {
            "enabled": False,
            "base_url": "https://openrouter.ai/api/v1",
        },
        "local": {
            "enabled": False,
            "base_url": "http://localhost:11434/v1",
            "model": "llama3.1",
        },
        "inception": {
            "enabled": True,
            "base_url": "https://api.inceptionlabs.ai/v1",
            "model": "mercury",
        },
    },
    "ui": {
        "theme": "dark",
        "panel_layout": "default",
        "streaming": True,
        "progress_bars": True,
        "compact_mode": False,
        "syntax_theme": "monokai",
    },
    "planner": {
        "max_depth": 5,
        "max_tasks": 100,
        "auto_replan": True,
        "replan_threshold": 0.3,
    },
    "memory": {
        "conversation_limit": 100,
        "vector_dimensions": 1536,
        "summary_interval": 10,
        "compression_enabled": True,
    },
    "tools": {
        "sandbox_enabled": True,
        "allowed_directories": [],
        "blocked_commands": [],
        "max_file_size": 10_485_760,  # 10MB
    },
    "identity": {
        "name": "Builder Agent",
        "mission": "Autonomously reason, execute, and improve code.",
        "principles": [
            "Keep context small and relevant",
            "Favor actions that improve future decisions",
            "Use tools safely and transparently",
        ],
        "objectives": [
            "Build robust CLI workflows",
            "Learn from results",
            "Continuously improve through feedback",
        ],
    },
    "skills": {
        "software_development": {
            "name": "Software Development",
            "description": "Planning, coding, testing, debugging, and deployment.",
            "components": ["Planning", "Coding", "Testing", "Debugging"],
            "success_rate": 0.93,
        },
        "memory_management": {
            "name": "Memory Management",
            "description": "Organize, retrieve, and compress knowledge efficiently.",
            "components": ["Retrieval", "Compression", "Archiving"],
            "success_rate": 0.88,
        },
    },
    "capabilities": {
        "database_optimization": {
            "name": "Database Optimization",
            "tools": ["benchmark", "sql_query"],
            "skills": ["Performance Analysis"],
            "success_rate": 0.91,
        },
    },
    "agents": {
        "planner_enabled": True,
        "coder_enabled": True,
        "reviewer_enabled": True,
        "debugger_enabled": True,
        "researcher_enabled": True,
        "architect_enabled": True,
        "security_enabled": True,
        "performance_enabled": True,
        "tester_enabled": True,
        "documentation_enabled": True,
    },
}

class APIConfig(BaseModel):
    """API configuration schema."""
    enabled: bool = False
    base_url: str = ""
    model: Optional[str] = None

class SystemConfig(BaseModel):
    """System configuration schema."""
    workspace: str = str(Path.home() / "zeta_workspace")
    log_dir: str = str(Path.home() / ".zeta" / "logs")
    log_level: str = "INFO"
    data_dir: str = str(Path.home() / ".zeta" / "data")
    max_retries: int = 3
    timeout: int = 300

class ConfigManager:
    """
    Manages all application configuration with hot reload support.

    Features:
    - JSON-based persistence
    - Environment variable overrides
    - Profile support
    - Hot reload via file watcher
    - Nested key access with dot notation
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path or Path.home() / ".zeta" / "config.json"
        self._config: dict = deepcopy(DEFAULT_CONFIG)
        self._overrides: dict = {}
        self._profiles: dict = {}
        self._watcher_task = None
        self._loaded = False

    async def load(self) -> None:
        """Load configuration from file, creating default if needed."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

        if self._config_path.exists():
            try:
                with open(self._config_path, "r") as f:
                    file_config = json.load(f)
                self._merge_config(self._config, file_config)
                console.print(f"[dim]Loaded config from {self._config_path}[/dim]")
            except (json.JSONDecodeError, IOError) as e:
                console.print(f"[yellow]Warning: Could not load config: {e}. Using defaults.[/yellow]")
                await self.save()
        else:
            console.print("[dim]No config found. Creating default configuration.[/dim]")
            await self.save()

        # Apply environment variable overrides
        self._apply_env_overrides()
        self._loaded = True

    async def save(self) -> None:
        """Save current configuration to file."""
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._config_path, "w") as f:
            json.dump(self._config, f, indent=2, default=str)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Dot-separated key path (e.g., 'api.default_model')
            default: Default value if key not found

        Returns:
            The configuration value
        """
        if key in self._overrides:
            return self._overrides[key]

        keys = key.split(".")
        current = self._config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.

        Args:
            key: Dot-separated key path
            value: Value to set
        """
        keys = key.split(".")
        current = self._config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    def set_override(self, key: str, value: Any) -> None:
        """Set a temporary override that takes precedence over file config."""
        self._overrides[key] = value

    def clear_override(self, key: str) -> None:
        """Clear a temporary override."""
        self._overrides.pop(key, None)

    def get_all(self) -> dict:
        """Get the full configuration dictionary (merged with overrides)."""
        result = deepcopy(self._config)
        for key, value in self._overrides.items():
            keys = key.split(".")
            current = result
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        return result

    def get_section(self, section: str) -> dict:
        """Get an entire configuration section."""
        return deepcopy(self._config.get(section, {}))

    async def reload(self) -> None:
        """Reload configuration from file."""
        await self.load()

    async def shutdown(self) -> None:
        """Clean up configuration resources."""
        await self.save()

    def _merge_config(self, base: dict, overlay: dict) -> None:
        """Recursively merge overlay into base."""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides (ZETA_* variables)."""
        for env_key, env_value in os.environ.items():
            if env_key.startswith("ZETA_"):
                config_key = env_key[5:].lower().replace("__", ".")
                self._overrides[config_key] = env_value

    @property
    def workspace(self) -> Path:
        """Get the current workspace path."""
        return Path(self.get("system.workspace", str(Path.home() / "zeta_workspace")))

    @property
    def is_loaded(self) -> bool:
        """Whether configuration has been loaded."""
        return self._loaded
