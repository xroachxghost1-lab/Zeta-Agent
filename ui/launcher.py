import asyncio
from pathlib import Path
from typing import Optional

from zeta_cli.main import initialize_system, run_gui, verify_system_health


async def _run(initial_goal: Optional[str] = None) -> None:
    systems = await initialize_system(None)
    try:
        await run_gui(systems, initial_goal=initial_goal)
    finally:
        await verify_system_health(systems)
        for system in systems.values():
            if hasattr(system, "shutdown"):
                await system.shutdown()


def main() -> None:
    asyncio.run(_run())
