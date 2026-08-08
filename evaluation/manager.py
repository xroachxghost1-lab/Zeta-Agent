"""Evaluation management subsystem for Zeta CLI."""

import json
import time
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from zeta_cli.config.manager import ConfigManager
from zeta_cli.memory.manager import MemoryManager
from zeta_cli.api.manager import APIManager
from zeta_cli.skills.manager import SkillManager

console = Console()

class EvaluationManager:
    """
    Tracks predictions, actual outcomes, and improvement decisions.
    """

    def __init__(self, config: ConfigManager, memory: MemoryManager, api: APIManager, skills: SkillManager):
        self._config = config
        self._memory = memory
        self._api = api
        self._skills = skills
        self._evaluation_file = Path(self._config.get("system.workspace")) / ".zeta" / "evaluation.json"
        self._results: list[dict[str, Any]] = []
        self._initialized = False

    async def initialize(self) -> None:
        await self._load_persisted()
        self._initialized = True
        console.print("[dim]Evaluation system initialized.[/dim]")

    async def _load_persisted(self) -> None:
        if not self._evaluation_file.exists():
            return
        try:
            with open(self._evaluation_file, "r", encoding="utf-8") as f:
                self._results = json.load(f)
        except Exception:
            console.print("[yellow]Warning: could not load persisted evaluation state.[/yellow]")

    async def save(self) -> None:
        self._evaluation_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._evaluation_file, "w", encoding="utf-8") as f:
            json.dump(self._results, f, indent=2)

    async def record_evaluation(
        self,
        task_id: str,
        prediction: str,
        actual: str,
        success: bool,
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        entry = {
            "task_id": task_id,
            "prediction": prediction,
            "actual": actual,
            "success": success,
            "metrics": metrics or {},
            "timestamp": time.time(),
        }
        self._results.append(entry)
        await self.save()

    def get_summary(self) -> dict[str, Any]:
        total = len(self._results)
        success_count = sum(1 for r in self._results if r["success"])
        return {
            "total_evaluations": total,
            "success_rate": (success_count / total * 100) if total else 0,
            "recent": self._results[-10:],
        }

    async def reflect_on_results(self) -> str:
        if not self._results:
            return "No evaluation results yet."

        latest = self._results[-1]
        return (
            f"Last task {latest['task_id']} predicted {latest['prediction']} and got {latest['actual']}. "
            f"Success: {latest['success']}."
        )

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def shutdown(self) -> None:
        await self.save()
        self._initialized = False
