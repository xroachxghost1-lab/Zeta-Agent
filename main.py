#!/usr/bin/env python3
"""
Zeta CLI — Main entry point.
Launches the production-grade AI coding assistant.
"""

import asyncio
import threading
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Optional

import psutil
import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from zeta_cli.config.manager import ConfigManager
from zeta_cli.security.manager import SecurityManager
from zeta_cli.identity.manager import IdentityManager
from zeta_cli.memory.manager import MemoryManager
from zeta_cli.evaluation.manager import EvaluationManager
from zeta_cli.api.manager import APIManager
from zeta_cli.tools.registry import ToolRegistry
from zeta_cli.agents.manager import AgentManager
from zeta_cli.planner.planner import TaskPlanner
from zeta_cli.core.engine import ExecutionEngine
from zeta_cli.ui.gui import ZetaGUIApp
from zeta_cli.skills.manager import SkillManager
from zeta_cli.utils.logger import setup_logger, get_logger

console = Console()
app = typer.Typer(
    name="zeta",
    help="Production-grade AI coding assistant for Windows terminal.",
    add_completion=False,
)

goal_app = typer.Typer(name="goal", help="Goal lifecycle commands.")
task_app = typer.Typer(name="task", help="Task management commands.")
memory_app = typer.Typer(name="memory", help="Memory subsystem commands.")
agents_app = typer.Typer(name="agents", help="Agent management commands.")
tools_app = typer.Typer(name="tools", help="Tool registry commands.")
skills_app = typer.Typer(name="skills", help="Skill and capability commands.")

app.add_typer(goal_app)
app.add_typer(task_app)
app.add_typer(memory_app)
app.add_typer(agents_app)
app.add_typer(tools_app)
app.add_typer(skills_app)

# Global shutdown event
shutdown_event = asyncio.Event()
logger = get_logger(__name__)

def handle_signal(signum, frame):
    """Handle interrupt signals gracefully."""
    console.print("\n[bold yellow]Shutting down Zeta CLI...[/bold yellow]")
    shutdown_event.set()

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def _session_file(config: ConfigManager) -> Path:
    return Path(config.get("system.workspace")) / ".zeta" / "session.pid"


def _write_session_pid(config: ConfigManager) -> None:
    path = _session_file(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()), encoding="utf-8")


def _read_session_pid(config: ConfigManager) -> Optional[int]:
    path = _session_file(config)
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def _is_process_running(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False


def _terminate_process(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(timeout=5)
        return True
    except Exception:
        return False


async def _run_system_command(
    config_path: Optional[Path],
    workspace: Optional[Path],
    model: Optional[str],
    callback,
):
    systems = await initialize_system(config_path)
    if workspace:
        systems["config"].set("system.workspace", str(workspace))
    if model:
        systems["config"].set("api.default_model", model)

    try:
        return await callback(systems)
    finally:
        await _cleanup_systems(systems)


async def _cleanup_systems(systems: dict) -> None:
    await verify_system_health(systems)
    for system in systems.values():
        if hasattr(system, "shutdown"):
            await system.shutdown()


async def initialize_system(config_path: Optional[Path] = None) -> dict:
    """
    Initialize all subsystems in proper order.

    Returns:
        Dictionary of initialized subsystem instances.
    """
    console.print(Panel.fit(
        Text("ZETA CLI v1.0.0", style="bold blue"),
        subtitle="Production-Grade AI Coding Assistant",
        border_style="blue",
    ))

    # 1. Configuration
    console.print("[dim]Initializing configuration...[/dim]")
    config = ConfigManager(config_path)
    await config.load()

    # 2. Security (encrypted secrets, key management)
    console.print("[dim]Initializing security...[/dim]")
    security = SecurityManager(config)
    await security.initialize()

    # 3. Identity
    console.print("[dim]Initializing identity...[/dim]")
    identity = IdentityManager(config)
    await identity.initialize()

    # 4. Logging
    console.print("[dim]Setting up logging...[/dim]")
    log_dir = config.get("system.log_dir", Path.home() / ".zeta" / "logs")
    setup_logger(log_dir, config.get("system.log_level", "INFO"))

    # 5. Memory (SQLite + vector store)
    console.print("[dim]Initializing memory system...[/dim]")
    memory = MemoryManager(config)
    await memory.initialize()

    # 6. Skill Manager
    console.print("[dim]Loading skills...[/dim]")
    skills = SkillManager(config)
    await skills.initialize()

    # 7. API Manager (LLM providers)
    console.print("[dim]Initializing API connections...[/dim]")
    api = APIManager(config, security)
    await api.initialize()

    # 8. Tool Registry
    console.print("[dim]Loading tools...[/dim]")
    tools = ToolRegistry(config)
    await tools.initialize()

    # 7. Agent Manager
    console.print("[dim]Initializing agents...[/dim]")
    agents = AgentManager(config, api, tools, memory)
    await agents.initialize()

    # 8. Task Planner
    console.print("[dim]Initializing planner...[/dim]")
    planner = TaskPlanner(config, api, memory)
    await planner.initialize()

    # 9. Evaluation System
    console.print("[dim]Initializing evaluation system...[/dim]")
    evaluation = EvaluationManager(config, memory, api, skills)
    await evaluation.initialize()

    # 10. Execution Engine
    console.print("[dim]Starting execution engine...[/dim]")
    engine = ExecutionEngine(config, planner, agents, tools, memory, api, skills=skills, evaluation=evaluation)
    await engine.initialize()

    console.print("[bold green]All systems initialized.[/bold green]\n")

    return {
        "config": config,
        "security": security,
        "identity": identity,
        "memory": memory,
        "api": api,
        "tools": tools,
        "skills": skills,
        "agents": agents,
        "planner": planner,
        "evaluation": evaluation,
        "engine": engine,
    }

async def run_gui(systems: dict, initial_goal: Optional[str] = None):
    """Run the native Tkinter GUI interface."""
    # Use a threading.Event the GUI can set; bridge back to asyncio via shutdown_event
    gui_shutdown = threading.Event()
    app = ZetaGUIApp(systems, gui_shutdown)

    if initial_goal:
        app.set_initial_goal(initial_goal, auto_launch=True)

    # Run the Tk mainloop in the main thread (required on Windows)
    # This blocks until the GUI closes
    app.run()
    # Signal asyncio shutdown after GUI closes
    shutdown_event.set()

async def run_interactive(systems: dict, initial_goal: Optional[str] = None):
    """Run the interactive GUI interface."""
    await run_gui(systems, initial_goal=initial_goal)

async def run_headless(systems: dict, goal: str):
    """Run in headless mode — execute goal and exit."""
    engine = systems["engine"]
    console.print(f"[bold cyan]Goal:[/bold cyan] {goal}")
    result = await engine.execute_goal(goal)
    console.print(f"\n[bold green]Result:[/bold green] {result.summary}")
    return result

@app.command()
def main(
    goal: Optional[str] = typer.Argument(
        None,
        help="Goal to execute (e.g., 'Build a Discord bot')",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file",
    ),
    headless: bool = typer.Option(
        False,
        "--headless",
        help="Run in headless mode (no interactive UI)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="LLM model to use",
    ),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace", "-w",
        help="Workspace directory",
    ),
):
    """
    Zeta CLI — Production-grade AI coding assistant.

    Launch interactively or execute a goal directly.
    """
    try:
        asyncio.run(_async_main(goal, config, headless, model, workspace))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted. Shutting down.[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        console.print(traceback.format_exc())
        sys.exit(1)

async def _async_main(
    goal: Optional[str],
    config_path: Optional[Path],
    headless: bool,
    model: Optional[str],
    workspace: Optional[Path],
):
    """Async main logic."""
    # Initialize all subsystems
    systems = await initialize_system(config_path)

    # Override model if specified
    if model:
        systems["config"].set("api.default_model", model)

    # Override workspace if specified
    if workspace:
        systems["config"].set("system.workspace", str(workspace))

    if headless and goal:
        await run_headless(systems, goal)
    else:
        _write_session_pid(systems["config"])
        try:
            await run_gui(systems, initial_goal=goal)
        finally:
            session_file = _session_file(systems["config"])
            if session_file.exists():
                session_file.unlink()

    # Verify system health before exit
    await verify_system_health(systems)

    # Cleanup
    for name, system in systems.items():
        if hasattr(system, "shutdown"):
            await system.shutdown()

    console.print("[dim]Zeta CLI terminated.[/dim]")

async def verify_system_health(systems: dict) -> None:
    """Validate that each subsystem is healthy before shutdown."""
    health = {
        "config": systems["config"].get("system.workspace", "unknown"),
        "security": systems["security"].is_initialized,
        "identity": systems["identity"].is_initialized,
        "memory": systems["memory"].is_initialized,
        "api": systems["api"]._initialized,
        "tools": systems["tools"].is_initialized,
        "skills": systems["skills"].is_initialized,
        "agents": systems["agents"].is_initialized,
        "planner": systems["planner"]._initialized,
        "evaluation": systems["evaluation"].is_initialized,
        "engine": systems["engine"].is_initialized,
    }

    console.print("[dim]System health summary:[/dim]")
    for name, status in health.items():
        console.print(f"  [cyan]{name}[/cyan]: {status}")

@app.command(name="start")
def start(
    goal: Optional[str] = typer.Argument(
        None,
        help="Goal to execute (e.g., 'Build a Discord bot')",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to configuration file",
    ),
    headless: bool = typer.Option(
        False,
        "--headless",
        help="Run in headless mode (no interactive UI)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="LLM model to use",
    ),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace", "-w",
        help="Workspace directory",
    ),
):
    """Start the Zeta CLI environment."""
    try:
        asyncio.run(_async_main(goal, config, headless, model, workspace))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted. Shutting down.[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        console.print(traceback.format_exc())
        sys.exit(1)

@app.command(name="version")
def version():
    """Show version information."""
    from zeta_cli import __version__
    console.print(f"Zeta CLI v{__version__}")

@app.command(name="status")
def status(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model to use",
    ),
    workspace: Optional[Path] = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Workspace directory",
    ),
):
    """Show system health and current runtime status."""
    try:
        asyncio.run(_run_system_command(config, workspace, model, lambda systems: verify_system_health(systems)))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted. Shutting down.[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        console.print(traceback.format_exc())
        sys.exit(1)

@app.command(name="stop")
def stop(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
):
    """Stop a running Zeta CLI session."""
    try:
        config_manager = ConfigManager(config)
        asyncio.run(config_manager.load())
        pid = _read_session_pid(config_manager)
        if not pid:
            console.print("[yellow]No active Zeta session found.[/yellow]")
            raise typer.Exit()

        if not _is_process_running(pid):
            console.print("[yellow]Session PID is stale. Removing stale lock.[/yellow]")
            session_file = _session_file(config_manager)
            if session_file.exists():
                session_file.unlink()
            raise typer.Exit()

        if _terminate_process(pid):
            console.print(f"[green]Stopped Zeta session (pid={pid}).[/green]")
            session_file = _session_file(config_manager)
            if session_file.exists():
                session_file.unlink()
        else:
            console.print(f"[red]Failed to stop Zeta session (pid={pid}).[/red]")
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted.[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        console.print(traceback.format_exc())
        sys.exit(1)

@app.command(name="logs")
def logs(
    file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Specific log file to show",
    ),
    lines: int = typer.Option(
        50,
        "--lines",
        "-n",
        help="Number of tail lines to show",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
):
    """Show recent log output."""
    config_manager = ConfigManager(config)
    asyncio.run(config_manager.load())
    log_dir = Path(config_manager.get("system.log_dir", Path.home() / ".zeta" / "logs"))
    if file:
        target = file if file.is_absolute() else log_dir / file
        if not target.exists():
            console.print(f"[red]Log file not found: {target}[/red]")
            raise typer.Exit(1)
        content = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        console.print("\n".join(content[-lines:]))
        raise typer.Exit()

    logs = sorted(log_dir.glob("zeta_*.log"), reverse=True)
    if not logs:
        console.print("[yellow]No log files found.[/yellow]")
        raise typer.Exit()

    console.print("[bold cyan]Recent log files:[/bold cyan]")
    for log in logs[:10]:
        console.print(f"  {log.name}")

    console.print("\nUse --file <file> to view a specific log.")

@goal_app.command(name="add")
def goal_add(
    description: str = typer.Argument(..., help="Description of the goal to execute."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration file."),
    headless: bool = typer.Option(False, "--headless", help="Execute without interactive UI."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use."),
    workspace: Optional[Path] = typer.Option(None, "--workspace", "-w", help="Workspace directory."),
):
    """Add and execute a new goal."""
    try:
        asyncio.run(_run_system_command(config, workspace, model, lambda systems: run_headless(systems, description) if headless else run_gui(systems, initial_goal=description)))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted. Shutting down.[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        console.print(traceback.format_exc())
        sys.exit(1)

@goal_app.command(name="list")
def goal_list(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration file."),
):
    """List active goals."""
    try:
        async def _list(systems):
            planner = systems["planner"]
            goals = planner.list_active_goals()
            if not goals:
                console.print("[yellow]No active goals found.[/yellow]")
                return
            for goal in goals:
                console.print(f"[cyan]{goal.goal_id}[/cyan] — {goal.description} ({len(goal.tasks)} tasks, status={goal.status.value})")
        asyncio.run(_run_system_command(config, None, None, _list))
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        console.print(traceback.format_exc())
        sys.exit(1)

@task_app.command(name="list")
def task_list(
    goal_id: Optional[str] = typer.Option(None, "--goal-id", "-g", help="Goal ID to list tasks for."),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter tasks by status."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration file."),
):
    """List tasks in the current plan or a specific goal."""
    try:
        async def _list(systems):
            planner = systems["planner"]
            goals = planner.list_active_goals()
            if not goals:
                console.print("[yellow]No active goals found.[/yellow]")
                return
            if goal_id:
                goal = planner.get_goal(goal_id)
                if not goal:
                    console.print(f"[red]Goal '{goal_id}' not found.[/red]")
                    return
                goals = [goal]
            for goal in goals:
                console.print(f"[bold cyan]Goal {goal.goal_id}:[/bold cyan] {goal.description}")
                for task in sorted(goal.tasks, key=lambda t: (-t.priority, t.status.value)):
                    if status and task.status.value != status:
                        continue
                    console.print(f"  [{task.status.value}] P{task.priority} {task.title} — {task.description}")
        asyncio.run(_run_system_command(config, None, None, _list))
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        console.print(traceback.format_exc())
        sys.exit(1)

@memory_app.command(name="search")
def memory_search(
    query: str = typer.Argument(..., help="Query string for long-term memory search."),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Memory category filter."),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum number of results."),
    config: Optional[Path] = typer.Option(None, "--config", "-f", help="Path to configuration file."),
):
    """Search long-term memory."""
    try:
        async def _search(systems):
            memory = systems["memory"]
            results = await memory.search_long_term(query, category=category, limit=limit)
            if not results:
                console.print("[yellow]No matching memories found.[/yellow]")
                return
            for entry in results:
                console.print(f"[cyan]{entry['key']}[/cyan] ({entry['category']}, importance={entry['importance']})")
                console.print(f"  {entry['value'][:300]}\n")
        asyncio.run(_run_system_command(config, None, None, _search))
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        console.print(traceback.format_exc())
        sys.exit(1)

@agents_app.command(name="list")
def agents_list(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration file."),
):
    """List registered agents."""
    try:
        async def _list(systems):
            agents = systems["agents"].list_agents()
            if not agents:
                console.print("[yellow]No agents available.[/yellow]")
                return
            for agent in agents:
                console.print(f"[cyan]{agent.name}[/cyan] — {getattr(agent, 'description', 'No description')}" )
        asyncio.run(_run_system_command(config, None, None, _list))
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        console.print(traceback.format_exc())
        sys.exit(1)

@tools_app.command(name="list")
def tools_list(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Tool category filter."),
    config: Optional[Path] = typer.Option(None, "--config", "-f", help="Path to configuration file."),
):
    """List available tools."""
    try:
        async def _list(systems):
            tools = systems["tools"].list_tools(category=category)
            if not tools:
                console.print("[yellow]No tools found.[/yellow]")
                return
            for tool in tools:
                console.print(f"[cyan]{tool.name}[/cyan] — {tool.description} ({tool.category})")
        asyncio.run(_run_system_command(config, None, None, _list))
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        console.print(traceback.format_exc())
        sys.exit(1)

@skills_app.command(name="list")
def skills_list(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to configuration file."),
):
    """List configured skills and capabilities."""
    try:
        async def _list(systems):
            skills = systems["skills"].list_skills()
            capabilities = systems["skills"].list_capabilities()
            if not skills and not capabilities:
                console.print("[yellow]No skills or capabilities configured.[/yellow]")
                return
            if skills:
                console.print("[bold cyan]Skills:[/bold cyan]")
                for skill in skills:
                    console.print(f"  [cyan]{skill.name}[/cyan] — {skill.description}")
            if capabilities:
                console.print("\n[bold cyan]Capabilities:[/bold cyan]")
                for cap in capabilities:
                    console.print(f"  [cyan]{cap.name}[/cyan] — tools: {', '.join(cap.tools) or 'none'}")
        asyncio.run(_run_system_command(config, None, None, _list))
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {e}")
        console.print(traceback.format_exc())
        sys.exit(1)

@app.command(name="setup")
def setup():
    """Run first-time setup wizard."""
    asyncio.run(_async_setup())

async def _async_setup():
    """Interactive first-time setup."""
    console.print(Panel.fit(
        Text("Zeta CLI — First Time Setup", style="bold blue"),
        border_style="blue",
    ))

    config = ConfigManager()
    security = SecurityManager(config)

    # API Key setup
    console.print("\n[bold]API Key Configuration[/bold]")
    console.print("Zeta CLI requires at least one LLM API key to function.")

    provider = typer.prompt(
        "Provider (inception/openai/anthropic/openrouter/local)",
        default="inception",
    )
    api_key = typer.prompt(
        f"Enter your {provider} API key",
        hide_input=True,
    )

    await security.store_secret(f"api_key_{provider}", api_key)
    config.set(f"api.{provider}.enabled", True)
    await config.save()

    # Workspace setup
    default_workspace = Path.home() / "zeta_workspace"
    workspace = typer.prompt(
        "Default workspace directory",
        default=str(default_workspace),
    )
    config.set("system.workspace", workspace)
    await config.save()

    console.print("\n[bold green]Setup complete![/bold green] Run [bold]zeta[/bold] to start.")
    await security.shutdown()
    await config.shutdown()

if __name__ == "__main__":
    app()
