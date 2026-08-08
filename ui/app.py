"""
Terminal UI Application — Rich-based professional interface.

Features:
- Codex-style appearance
- Multiple live panels
- Streaming output
- Progress indicators
- Status bar
- Command input
- Real-time updates
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.align import Align
from rich.prompt import Prompt
from rich.box import ROUNDED, SIMPLE

from zeta_cli.config.manager import ConfigManager
from zeta_cli.core.engine import ExecutionEngine, EngineState

console = Console()

class TerminalApp:
    """
    Professional terminal interface for Zeta CLI.

    Codex-style appearance with:
    - Header with status
    - Main output panel
    - Task list panel
    - Status bar with metrics
    - Command input
    """

    def __init__(self, systems: dict, shutdown_event: asyncio.Event):
        self._config: ConfigManager = systems["config"]
        self._engine: ExecutionEngine = systems["engine"]
        self._planner = systems["planner"]
        self._memory = systems["memory"]
        self._tools = systems["tools"]
        self._agents = systems["agents"]
        self._skills = systems.get("skills")
        self._identity = systems.get("identity")
        self._evaluation = systems.get("evaluation")
        self._api = systems["api"]
        self._security = systems["security"]

        self._shutdown_event = shutdown_event
        self._output_lines: list[str] = []
        self._max_output_lines = 100
        self._start_time = time.time()
        self._token_count = 0

    async def run(self, initial_goal: Optional[str] = None) -> None:
        """
        Run the interactive terminal UI.

        Args:
            initial_goal: Optional goal to execute immediately
        """
        console.clear()
        await self._show_banner()

        if initial_goal:
            await self._execute_command(f"goal {initial_goal}")

        # Main command loop
        while not self._shutdown_event.is_set():
            try:
                # Show prompt
                state = self._engine.get_state()
                prompt_style = {
                    EngineState.IDLE: "[bold green]AI[/bold green]",
                    EngineState.EXECUTING: "[bold yellow]AI (executing)[/bold yellow]",
                    EngineState.PAUSED: "[bold yellow]AI (paused)[/bold yellow]",
                    EngineState.PLANNING: "[bold cyan]AI (planning)[/bold cyan]",
                    EngineState.REFLECTING: "[bold magenta]AI (reflecting)[/bold magenta]",
                }.get(state, "[bold green]AI[/bold green]")

                user_input = await asyncio.to_thread(
                    Prompt.ask, f"\n{prompt_style} ›"
                )

                if user_input.strip():
                    await self._execute_command(user_input.strip())

            except KeyboardInterrupt:
                console.print("\n[bold yellow]Use 'exit' to quit, 'stop' to halt execution[/bold yellow]")
            except EOFError:
                break

        console.print("\n[dim]Goodbye, Alpha.[/dim]")

    async def _execute_command(self, command: str) -> None:
        """Execute a user command."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        commands = {
            "goal": self._cmd_goal,
            "status": self._cmd_status,
            "tasks": self._cmd_tasks,
            "retry": self._cmd_retry,
            "pause": self._cmd_pause,
            "resume": self._cmd_resume,
            "stop": self._cmd_stop,
            "logs": self._cmd_logs,
            "history": self._cmd_history,
            "memory": self._cmd_memory_stats,
            "agents": self._cmd_agents,
            "tools": self._cmd_tools_list,
            "skills": self._cmd_skills,
            "skill": self._cmd_skill,
            "capabilities": self._cmd_capabilities,
            "capability": self._cmd_capability,
            "workspace": self._cmd_workspace,
            "settings": self._cmd_settings,
            "config": self._cmd_config,
            "plan": self._cmd_plan,
            "reflect": self._cmd_reflect,
            "evaluate": self._cmd_evaluate,
            "learn": self._cmd_learn,
            "improve": self._cmd_improve,
            "identity": self._cmd_identity,
            "terminal": self._cmd_terminal,
            "shell": self._cmd_shell,
            "git": self._cmd_git,
            "search": self._cmd_search,
            "run": self._cmd_run,
            "edit": self._cmd_edit,
            "diff": self._cmd_diff,
            "test": self._cmd_test,
            "benchmark": self._cmd_benchmark,
            "help": self._cmd_help,
            "exit": self._cmd_exit,
            "quit": self._cmd_exit,
        }

        handler = commands.get(cmd)
        if handler:
            await handler(args)
        else:
            console.print(f"[red]Unknown command: {cmd}[/red]")
            console.print(f"[dim]Type 'help' for available commands.[/dim]")

    async def _cmd_goal(self, args: str) -> None:
        """Execute a goal."""
        if not args:
            console.print("[red]Usage: goal <description>[/red]")
            return
        await self._engine.execute_goal(args)

    async def _cmd_status(self, args: str) -> None:
        """Show current status."""
        progress = self._engine.get_progress()
        table = Table(title="Current Status", box=ROUNDED, border_style="blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        for key, value in progress.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        # Memory stats
        mem_stats = await self._memory.get_stats()
        table.add_row("Memory Entries", str(mem_stats.get("conversations", 0)))
        table.add_row("Session Tokens", str(mem_stats.get("current_session_tokens", 0)))

        # Tool stats
        tool_stats = self._tools.get_execution_stats()
        table.add_row("Tool Executions", str(tool_stats.get("total_executions", 0)))
        table.add_row("Tool Success Rate", tool_stats.get("success_rate", "N/A"))

        console.print(table)

    async def _cmd_tasks(self, args: str) -> None:
        """List tasks."""
        if self._engine._current_goal:
            summary = await self._planner.get_plan_summary(self._engine._current_goal.goal_id)
            console.print(Panel(summary, title="Task Plan", border_style="blue"))
        else:
            console.print("[yellow]No active goal. Use 'goal <description>' to start.[/yellow]")

    async def _cmd_retry(self, args: str) -> None:
        """Retry last failed task."""
        console.print("[yellow]Retry functionality — restarting current goal execution...[/yellow]")
        if self._engine._current_goal:
            await self._engine.resume()

    async def _cmd_pause(self, args: str) -> None:
        """Pause execution."""
        await self._engine.pause()

    async def _cmd_resume(self, args: str) -> None:
        """Resume execution."""
        await self._engine.resume()

    async def _cmd_stop(self, args: str) -> None:
        """Stop execution."""
        await self._engine.stop()

    async def _cmd_logs(self, args: str) -> None:
        """Show execution logs."""
        if not self._engine._step_history:
            console.print("[yellow]No execution history.[/yellow]")
            return

        table = Table(title="Execution Log", box=SIMPLE)
        table.add_column("Task", style="cyan")
        table.add_column("Result", style="white")
        table.add_column("Agent", style="dim")
        table.add_column("Time", style="dim")

        for step in self._engine._step_history[-20:]:
            result_icon = "✓" if step.success else "✗"
            result_style = "green" if step.success else "red"
            table.add_row(
                step.task_id[:12],
                f"[{result_style}]{result_icon}[/{result_style}]",
                step.agent_used,
                f"{step.duration:.1f}s",
            )

        console.print(table)

    async def _cmd_history(self, args: str) -> None:
        """Show conversation history."""
        history = await self._memory.get_conversation_history(limit=20)
        for msg in history:
            role_style = {
                "user": "bold cyan",
                "assistant": "bold green",
                "system": "bold yellow",
            }.get(msg["role"], "white")
            console.print(f"[{role_style}]{msg['role']}:[/{role_style}] {msg['content'][:200]}")
            console.print(f"[dim]{'─'*60}[/dim]")

    async def _cmd_memory_stats(self, args: str) -> None:
        """Show memory statistics."""
        stats = await self._memory.get_stats()
        table = Table(title="Memory Statistics", box=ROUNDED, border_style="blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        for key, value in stats.items():
            table.add_row(key.replace("_", " ").title(), str(value))
        console.print(table)

    async def _cmd_agents(self, args: str) -> None:
        """List agents and their stats."""
        agents = self._agents.list_agents()
        stats = self._agents.get_stats()

        table = Table(title="Agents", box=ROUNDED, border_style="blue")
        table.add_column("Agent", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Uses", style="dim")

        usage = stats.get("agent_usage", {})
        for agent in agents:
            table.add_row(
                agent.name,
                agent.description,
                str(usage.get(agent.name, 0)),
            )

        console.print(table)

    async def _cmd_tools_list(self, args: str) -> None:
        """List available tools."""
        if args:
            tools = self._tools.list_tools(category=args)
        else:
            tools = self._tools.list_tools()

        categories = self._tools.list_categories()

        for category in categories:
            cat_tools = [t for t in tools if t.category == category]
            if cat_tools:
                console.print(f"\n[bold cyan]{category.upper()}[/bold cyan]")
                for tool in cat_tools:
                    destructive = " [red]⚠[/red]" if tool.is_destructive else ""
                    sandbox = " [yellow]🔒[/yellow]" if tool.requires_sandbox else ""
                    console.print(f"  {tool.name}{destructive}{sandbox} — {tool.description[:80]}")

        stats = self._tools.get_execution_stats()
        console.print(f"\n[dim]Total executions: {stats['total_executions']} | Success rate: {stats['success_rate']}[/dim]")

    async def _cmd_workspace(self, args: str) -> None:
        """Show or change workspace."""
        if args:
            new_path = Path(args).expanduser().resolve()
            if new_path.exists() and new_path.is_dir():
                self._config.set("system.workspace", str(new_path))
                console.print(f"[green]Workspace changed to: {new_path}[/green]")
            else:
                console.print(f"[red]Invalid directory: {args}[/red]")
        else:
            ws = self._config.get("system.workspace")
            console.print(f"Current workspace: [cyan]{ws}[/cyan]")

            # Show contents
            from zeta_cli.tools.builtin_tools import ListDirectoryTool
            tool = ListDirectoryTool()
            result = await tool.execute(path=ws)
            if result.success:
                console.print(result.output)

    async def _cmd_settings(self, args: str) -> None:
        """Show current settings."""
        config = self._config.get_all()
        console.print_json(data=config)

    async def _cmd_config(self, args: str) -> None:
        """Get or set configuration."""
        if not args:
            await self._cmd_settings("")
            return

        parts = args.split(maxsplit=1)
        if len(parts) == 2 and "=" in parts[1]:
            # Set: config key = value
            key = parts[0]
            value = parts[1].split("=", 1)[1].strip()
            self._config.set(key, value)
            await self._config.save()
            console.print(f"[green]Set {key} = {value}[/green]")
        else:
            # Get
            value = self._config.get(args)
            console.print(f"{args} = [cyan]{value}[/cyan]")

    async def _cmd_plan(self, args: str) -> None:
        """Show the current plan."""
        await self._cmd_tasks(args)

    async def _cmd_reflect(self, args: str) -> None:
        """Show reflections on execution."""
        if self._engine._current_goal:
            results = [s for s in self._engine._step_history if not s.success]
            if results:
                for r in results:
                    console.print(Panel(r.output[:500], title=f"Task: {r.task_id[:12]}", border_style="red"))
            else:
                console.print("[green]No failures to reflect on.[/green]")
        else:
            console.print("[yellow]No active goal.[/yellow]")

    async def _cmd_evaluate(self, args: str) -> None:
        """Show evaluation summary."""
        if not self._evaluation:
            console.print("[yellow]Evaluation system not initialized.[/yellow]")
            return
        summary = self._evaluation.get_summary()
        table = Table(title="Evaluation Summary", box=ROUNDED, border_style="blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Total Evaluations", str(summary["total_evaluations"]))
        table.add_row("Success Rate", f"{summary['success_rate']:.1f}%")
        console.print(table)

    async def _cmd_learn(self, args: str) -> None:
        """Capture a learning entry into identity memory."""
        if not self._identity:
            console.print("[yellow]Identity system not initialized.[/yellow]")
            return
        if not args:
            console.print("[red]Usage: learn <strategy or lesson>[/red]")
            return
        await self._identity.add_strategy(args)
        console.print(f"[green]Learned strategy added:[/green] {args}")

    async def _cmd_improve(self, args: str) -> None:
        """Review recent evaluations and propose improvements."""
        if not self._evaluation:
            console.print("[yellow]Evaluation system not initialized.[/yellow]")
            return
        if not args:
            args = """Review the latest evaluation results and suggest one improvement."""
        reflection = await self._evaluation.reflect_on_results()
        console.print(Panel(reflection, title="Improvement Insight", border_style="green"))

    async def _cmd_identity(self, args: str) -> None:
        """Show current AI identity."""
        if not self._identity:
            console.print("[yellow]Identity system not initialized.[/yellow]")
            return
        data = self._identity.get_all()
        console.print_json(data=data)

    async def _cmd_terminal(self, args: str) -> None:
        """Execute a terminal command."""
        if not args:
            console.print("[red]Usage: terminal <command>[/red]")
            return

        from zeta_cli.tools.builtin_tools import RunCommandTool
        tool = RunCommandTool()
        result = await tool.execute(command=args)
        console.print(result.output)
        if result.error:
            console.print(f"[red]{result.error}[/red]")

    async def _cmd_shell(self, args: str) -> None:
        """Execute a PowerShell command."""
        await self._cmd_terminal(args)

    async def _cmd_git(self, args: str) -> None:
        """Execute a git command."""
        from zeta_cli.tools.builtin_tools import GitTool
        tool = GitTool()
        result = await tool.execute(command=args)
        console.print(result.output)
        if result.error:
            console.print(f"[red]{result.error}[/red]")

    async def _cmd_search(self, args: str) -> None:
        """Search files using grep."""
        if not args:
            console.print("[red]Usage: search <pattern> [directory][/red]")
            return

        parts = args.split()
        pattern = parts[0]
        directory = parts[1] if len(parts) > 1 else "."

        from zeta_cli.tools.builtin_tools import GrepTool
        tool = GrepTool()
        result = await tool.execute(pattern=pattern, path=directory)
        console.print(result.output)

    async def _cmd_run(self, args: str) -> None:
        """Run Python code."""
        if not args:
            console.print("[red]Usage: run <python_code>[/red]")
            return

        from zeta_cli.tools.builtin_tools import PythonTool
        tool = PythonTool()
        result = await tool.execute(code=args)
        console.print(result.output)
        if result.error:
            console.print(f"[red]{result.error}[/red]")

    async def _cmd_edit(self, args: str) -> None:
        """Edit a file (opens in default editor)."""
        if not args:
            console.print("[red]Usage: edit <file_path>[/red]")
            return

        file_path = Path(args).expanduser().resolve()
        if file_path.exists():
            import os
            os.startfile(str(file_path))
            console.print(f"[green]Opened {file_path} in default editor[/green]")
        else:
            create = Prompt.ask(f"File {args} doesn't exist. Create?", choices=["y", "n"], default="n")
            if create == "y":
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.touch()
                os.startfile(str(file_path))
                console.print(f"[green]Created and opened {file_path}[/green]")

    async def _cmd_diff(self, args: str) -> None:
        """Show diff between two files."""
        parts = args.split()
        if len(parts) < 2:
            console.print("[red]Usage: diff <file1> <file2>[/red]")
            return

        from zeta_cli.tools.builtin_tools import DiffTool
        tool = DiffTool()
        result = await tool.execute(file1=parts[0], file2=parts[1])
        console.print(result.output)

    async def _cmd_test(self, args: str) -> None:
        """Run tests."""
        test_path = args or "tests"
        from zeta_cli.tools.builtin_tools import RunCommandTool
        tool = RunCommandTool()
        result = await tool.execute(command=f"python -m pytest {test_path} -v")
        console.print(result.output)

    async def _cmd_benchmark(self, args: str) -> None:
        """Run a quick benchmark."""
        console.print("[yellow]Running quick system benchmark...[/yellow]")

        import time

        # CPU benchmark
        start = time.time()
        total = sum(i * i for i in range(10_000_000))
        cpu_time = time.time() - start

        # Memory info
        import psutil
        mem = psutil.virtual_memory()

        table = Table(title="Benchmark Results", box=ROUNDED, border_style="blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("CPU (10M operations)", f"{cpu_time:.3f}s")
        table.add_row("Memory Total", f"{mem.total / (1024**3):.1f} GB")
        table.add_row("Memory Available", f"{mem.available / (1024**3):.1f} GB")
        table.add_row("Python Version", __import__("sys").version.split()[0])

        console.print(table)

    async def _cmd_skills(self, args: str) -> None:
        """List configured skills."""
        if not self._skills:
            console.print("[yellow]Skill system not initialized.[/yellow]")
            return

        table = Table(title="Skills", box=ROUNDED, border_style="blue")
        table.add_column("Skill", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Enabled", style="green")

        for skill in self._skills.list_skills():
            table.add_row(skill.name, skill.description, str(skill.enabled))

        console.print(table)

    async def _cmd_skill(self, args: str) -> None:
        """Show details for a specific skill."""
        if not args:
            console.print("[red]Usage: skill <name>[/red]")
            return
        if not self._skills:
            console.print("[yellow]Skill system not initialized.[/yellow]")
            return

        description = await self._skills.describe_skill(args)
        console.print(Panel(description, title=f"Skill: {args}", border_style="blue"))

    async def _cmd_capabilities(self, args: str) -> None:
        """List configured capabilities."""
        if not self._skills:
            console.print("[yellow]Skill system not initialized.[/yellow]")
            return

        table = Table(title="Capabilities", box=ROUNDED, border_style="blue")
        table.add_column("Capability", style="cyan")
        table.add_column("Tools", style="white")
        table.add_column("Enabled", style="green")

        for cap in self._skills.list_capabilities():
            table.add_row(cap.name, ", ".join(cap.tools) or "none", str(cap.enabled))

        console.print(table)

    async def _cmd_capability(self, args: str) -> None:
        """Show details for a specific capability."""
        if not args:
            console.print("[red]Usage: capability <name>[/red]")
            return
        if not self._skills:
            console.print("[yellow]Skill system not initialized.[/yellow]")
            return

        capability = self._skills.get_capability(args)
        if not capability:
            console.print(f"[red]Capability '{args}' not found.[/red]")
            return

        description = (
            f"Capability: {capability.name}\n"
            f"Tools: {', '.join(capability.tools) or 'none'}\n"
            f"Skills: {', '.join(capability.skills) or 'none'}\n"
            f"Success Rate: {capability.success_rate:.1%}\n"
            f"Enabled: {capability.enabled}"
        )
        console.print(Panel(description, title=f"Capability: {args}", border_style="blue"))

    async def _cmd_help(self, args: str) -> None:
        """Show help."""
        help_text = """
[bold cyan]Zeta CLI Commands[/bold cyan]

[bold]Goal Execution:[/bold]
  [cyan]goal[/cyan] <description>     — Execute a new goal
  [cyan]status[/cyan]                  — Show current execution status
  [cyan]tasks[/cyan]                   — List current tasks
  [cyan]pause[/cyan] / [cyan]resume[/cyan] / [cyan]stop[/cyan]  — Control execution
  [cyan]retry[/cyan]                   — Retry failed tasks

[bold]Information:[/bold]
  [cyan]logs[/cyan]                    — Show execution logs
  [cyan]history[/cyan]                 — Conversation history
  [cyan]memory[/cyan]                  — Memory statistics
  [cyan]agents[/cyan]                  — List agents and usage
  [cyan]tools[/cyan]                   — List available tools
  [cyan]skills[/cyan]                  — List configured skills
  [cyan]skill[/cyan] <name>            — Show details for a skill
  [cyan]capabilities[/cyan]            — List configured capabilities
  [cyan]capability</cyan> <name>      — Show details for a capability
  [cyan]plan[/cyan]                    — Show current plan
  [cyan]reflect[/cyan]                 — Show failure reflections

[bold]Workspace:[/bold]
  [cyan]workspace[/cyan] [path]        — Show/set workspace directory
  [cyan]search[/cyan] <pattern>        — Search files with grep
  [cyan]edit[/cyan] <file>             — Open file in default editor
  [cyan]diff[/cyan] <f1> <f2>          — Show file differences

[bold]Execution:[/bold]
  [cyan]terminal[/cyan] <cmd>          — Execute terminal command
  [cyan]shell[/cyan] <cmd>             — Execute PowerShell command
  [cyan]git[/cyan] <command>           — Execute git command
  [cyan]run[/cyan] <code>              — Run Python code
  [cyan]test[/cyan] [path]             — Run tests
  [cyan]benchmark[/cyan]               — Run system benchmark

[bold]Configuration:[/bold]
  [cyan]settings[/cyan]                — Show all settings
  [cyan]config[/cyan] <key>            — Get config value
  [cyan]config[/cyan] <key> = <value>  — Set config value

[bold]System:[/bold]
  [cyan]help[/cyan]                    — Show this help
  [cyan]exit[/cyan] / [cyan]quit[/cyan]— Exit Zeta CLI
"""
        console.print(Panel(help_text, title="Help", border_style="blue"))

    async def _cmd_exit(self, args: str) -> None:
        """Exit the application."""
        console.print("[dim]Shutting down...[/dim]")
        self._shutdown_event.set()

    async def _show_banner(self) -> None:
        """Show the startup banner."""
        identity = self._identity.get_all() if self._identity else self._config.get("identity", {})
        name = identity.get("name", "Builder Agent")
        mission = identity.get(
            "mission",
            "Autonomously reason, execute, and improve code."
        )

        splash = """
[bold red]███████╗██╗ ██████╗ ███████╗[/bold red] [bold white]███████╗[/bold white] [bold cyan]███████╗████████╗[/bold cyan]
[bold red]██╔════╝██║██╔════╝ ██╔════╝[/bold red] [bold white]██╔════╝[/bold white] [bold cyan]██╔══════╝╚══██╔══╝[/bold cyan]
[bold red]█████╗  ██║██║  ███╗█████╗  [/bold red] [bold white]█████╗  [/bold white] [bold cyan]███████╗   ██║   [/bold cyan]
[bold red]██╔══╝  ██║██║   ██║██╔══╝  [/bold red] [bold white]██╔══╝  [/bold white] [bold cyan]╚═══██║   ██║   [/bold cyan]
[bold red]██║     ██║╚██████╔╝███████╗[/bold red] [bold white]███████╗[/bold white] [bold cyan]███████║   ██║   [/bold cyan]
[bold red]╚═╝     ╚═╝ ╚═════╝ ╚══════╝[/bold red] [bold white]╚══════╝[/bold white] [bold cyan]╚═══════╝   ╚═╝   [/bold cyan]
"""
        message = """
[bold magenta]Fuck other shit.......[/bold magenta]
[bold yellow]This is the new way.[/bold yellow]
"""
        header = Panel.fit(
            Text("AIOS Autonomous Intelligence System", style="bold white"),
            subtitle="[green]Running[/green]",
            border_style="bright_blue",
        )
        stats = await self._memory.get_stats()
        current_goal = self._engine._current_goal.description if self._engine._current_goal else "None"
        active_tasks = 0
        if self._engine._current_goal:
            active_tasks = sum(
                1 for t in self._engine._current_goal.tasks
                if t.status in ("pending", "in_progress", "retry", "blocked")
            )

        summary = Panel.fit(
            Text(
                f"[bold cyan]Identity:[/bold cyan] {name}\n"
                f"[bold cyan]Mission:[/bold cyan] {mission}\n"
                f"[bold cyan]Status:[/bold cyan] Running\n"
                f"[bold cyan]Current Goal:[/bold cyan] {current_goal}\n"
                f"[bold cyan]Tasks:[/bold cyan] {active_tasks} Active\n"
                f"[bold cyan]Agents:[/bold cyan] {len(self._agents.list_agents())} Running\n"
                f"[bold cyan]Memory:[/bold cyan] {stats.get('conversations', 0)} Records\n"
                f"[bold cyan]Context:[/bold cyan] {len(await self._memory.get_conversation_history(limit=10))} Relevant Items Loaded\n"
                f"[bold cyan]Tools:[/bold cyan] {len(self._tools.list_tools())} Available\n"
                f"[bold cyan]Skills:[/bold cyan] {len(self._skills.list_skills()) if self._skills else 0}\n"
                f"[bold cyan]Capabilities:[/bold cyan] {len(self._skills.list_capabilities()) if self._skills else 0}\n",
                style="white",
            ),
            title="[bold green]System Overview[/bold green]",
            border_style="cyan",
        )

        console.print(splash)
        console.print(message)
        console.print(header)
        console.print(summary)
        console.print("[dim]Type 'help' for commands, 'goal <description>' to start.[/dim]")
        console.print()

    async def shutdown(self) -> None:
        """Clean up UI resources."""
        pass
