"""
Execution Engine — The central execution loop that drives goal completion.

Features:
- Goal-driven autonomous execution
- Plan → Execute → Observe → Reflect → Improve cycle
- Dynamic replanning on failure
- Streaming progress updates
- Multi-agent orchestration
- Automatic retry with strategy variation
"""

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from rich.console import Console

from zeta_cli.config.manager import ConfigManager
from zeta_cli.planner.planner import TaskPlanner, Task, TaskStatus, Goal
from zeta_cli.agents.manager import AgentManager
from zeta_cli.tools.registry import ToolRegistry
from zeta_cli.memory.manager import MemoryManager
from zeta_cli.api.manager import APIManager

console = Console()

class EngineState(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    RETRYING = "retrying"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ExecutionResult:
    """Result of a full goal execution."""
    goal_id: str
    success: bool
    summary: str
    tasks_completed: int
    tasks_failed: int
    total_retries: int
    duration_seconds: float
    reflections: list[str] = field(default_factory=list)

@dataclass
class StepResult:
    """Result of a single execution step."""
    task_id: str
    success: bool
    output: str
    reflection: str
    agent_used: str
    duration: float

class ExecutionEngine:
    """
    Central execution engine implementing the core loop:

    Goal → Plan → Execute → Observe → Reflect → Improve → Retry → Continue
    """

    def __init__(
        self,
        config: ConfigManager,
        planner: TaskPlanner,
        agents: AgentManager,
        tools: ToolRegistry,
        memory: MemoryManager,
        api: APIManager,
        skills: Optional[Any] = None,
        evaluation: Optional[Any] = None,
    ):
        self._config = config
        self._planner = planner
        self._agents = agents
        self._tools = tools
        self._memory = memory
        self._api = api
        self._skills = skills
        self._evaluation = evaluation

        self._state = EngineState.IDLE
        self._current_goal: Optional[Goal] = None
        self._step_history: list[StepResult] = []
        self._event_callbacks: list[Callable] = []
        self._max_retries = config.get("system.max_retries", 3)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the execution engine."""
        self._initialized = True

    async def execute_goal(self, goal_description: str) -> ExecutionResult:
        """
        Execute a complete goal from start to finish.

        This is the main entry point for goal-driven execution.

        Args:
            goal_description: Natural language goal description

        Returns:
            ExecutionResult with complete execution summary
        """
        start_time = time.time()
        self._state = EngineState.PLANNING
        self._step_history = []

        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print(f"[bold blue]GOAL: {goal_description}[/bold blue]")
        console.print(f"[bold blue]{'='*60}[/bold blue]\n")

        # Store in conversation memory
        await self._memory.add_conversation(
            role="user",
            content=f"goal {goal_description}",
            tokens=await self._api.count_tokens(goal_description),
        )

        # 1. Plan
        self._emit_event("planning_started", {"goal": goal_description})
        goal = await self._planner.create_goal(goal_description)
        self._current_goal = goal
        self._emit_event("planning_complete", {"tasks": len(goal.tasks)})

        # Print plan
        plan_summary = await self._planner.get_plan_summary(goal.goal_id)
        console.print(f"[dim]{plan_summary}[/dim]\n")

        # 2. Execute loop
        self._state = EngineState.EXECUTING
        total_retries = 0
        tasks_completed = 0
        tasks_failed = 0
        reflections = []

        while True:
            # Check if paused
            while self._state == EngineState.PAUSED:
                await asyncio.sleep(0.5)

            # Get next task
            task = await self._planner.get_next_task(goal.goal_id)

            if task is None:
                # Check if all done
                if await self._planner.check_goal_completion(goal.goal_id):
                    self._state = EngineState.COMPLETED
                    break

                # Check for blocked tasks
                pending = [
                    t for t in goal.tasks
                    if t.status in (TaskStatus.PENDING, TaskStatus.RETRY, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED)
                ]
                blocked = [
                    t for t in pending
                    if t.status == TaskStatus.BLOCKED
                ]

                if blocked:
                    # Try to unblock by replanning
                    for bt in blocked[:3]:
                        await self._planner.replan(goal.goal_id, bt, "Task blocked by unmet dependencies")
                else:
                    # No executable tasks, but some pending — might be dependency issues
                    break

                continue

            # Execute task
            self._emit_event("task_started", {"task": task.title, "task_id": task.task_id})

            await self._planner.update_task_status(task.task_id, TaskStatus.IN_PROGRESS)

            step_result = await self._execute_task(task)
            self._step_history.append(step_result)

            if step_result.success:
                tasks_completed += 1
                await self._planner.update_task_status(
                    task.task_id, TaskStatus.COMPLETED, step_result.output
                )
                console.print(f"  [green]✓[/green] {task.title} [dim]({step_result.duration:.1f}s)[/dim]")
            else:
                tasks_failed += 1
                total_retries += 1
                await self._planner.update_task_status(
                    task.task_id, TaskStatus.FAILED, step_result.output
                )
                console.print(f"  [red]✗[/red] {task.title} [dim]({step_result.duration:.1f}s)[/dim]")

                # Reflect and replan
                self._state = EngineState.REFLECTING
                reflection = await self._reflect_on_failure(task, step_result)
                reflections.append(reflection)
                console.print(f"  [yellow]💡 Reflection:[/yellow] {reflection[:200]}...")

                # Generate alternative tasks
                if task.retry_count < self._max_retries:
                    self._state = EngineState.RETRYING
                    new_tasks = await self._planner.replan(
                        goal.goal_id, task, step_result.output
                    )
                    if new_tasks:
                        console.print(f"  [cyan]↻ Created {len(new_tasks)} alternative tasks[/cyan]")

                self._state = EngineState.EXECUTING

            self._emit_event("task_completed", {
                "task": task.title,
                "success": step_result.success,
            })

        # Calculate duration
        duration = time.time() - start_time

        # Build result
        success = tasks_failed == 0 and tasks_completed > 0
        summary = (
            f"Goal: {goal_description}\n"
            f"Completed: {tasks_completed} tasks\n"
            f"Failed: {tasks_failed} tasks\n"
            f"Retries: {total_retries}\n"
            f"Duration: {duration:.1f}s\n"
            f"Result: {'✓ SUCCESS' if success else '⚠ PARTIAL'}"
        )

        result = ExecutionResult(
            goal_id=goal.goal_id,
            success=success,
            summary=summary,
            tasks_completed=tasks_completed,
            tasks_failed=tasks_failed,
            total_retries=total_retries,
            duration_seconds=duration,
            reflections=reflections,
        )

        console.print(f"\n[bold]{'='*60}[/bold]")
        console.print(f"[bold]{'SUCCESS' if success else 'PARTIAL'}[/bold] — {tasks_completed}/{tasks_completed + tasks_failed} tasks completed in {duration:.1f}s")
        console.print(f"[bold]{'='*60}[/bold]\n")

        # Store result
        await self._memory.add_conversation(
            role="assistant",
            content=summary,
            tokens=await self._api.count_tokens(summary),
        )

        self._emit_event("goal_completed", {"result": result})

        return result

    async def _execute_task(self, task: Task) -> StepResult:
        """
        Execute a single task using the appropriate agent and tools.

        Args:
            task: The task to execute

        Returns:
            StepResult with execution details
        """
        start = time.time()

        try:
            # Build context
            context = {
                "task": task,
                "workspace": self._config.get("system.workspace"),
                "goal": self._current_goal.description if self._current_goal else "",
                "previous_steps": [
                    {"task": s.task_id, "success": s.success, "output": s.output[:500]}
                    for s in self._step_history[-5:]
                ],
            }

            # Get recent memory for context
            recent_memory = await self._memory.get_conversation_history(limit=5)
            context["recent_context"] = "\n".join(
                f"[{m['role']}]: {m['content'][:300]}" for m in recent_memory
            )

            # Route to appropriate agent
            execution_prompt = f"""Task: {task.title}
Description: {task.description}
Priority: {task.priority}/10
Complexity: {task.complexity}

Execute this task completely. Use available tools as needed.
Output the result clearly."""

            # Try coder agent first for implementation tasks
            if self._skills:
                suggested_agents = []
                for skill in self._skills.list_skills():
                    if any(
                        term.lower() in task.title.lower() or term.lower() in task.description.lower()
                        for term in [skill.name] + skill.components
                    ):
                        suggested_agents.extend(self._skills.recommend_agents_for_skill(skill.name))
                if suggested_agents:
                    context["recommended_agents"] = sorted(set(suggested_agents))

            result = await self._agents.execute(
                "coder", execution_prompt, context
            )

            if not result.success:
                # Try routing to most appropriate agent
                result = await self._agents.route_task(
                    f"{task.title}\n{task.description}", context
                )

            duration = time.time() - start

            return StepResult(
                task_id=task.task_id,
                success=result.success,
                output=result.output,
                reflection="",
                agent_used=result.agent_name,
                duration=duration,
            )

        except Exception as e:
            duration = time.time() - start
            error_output = f"Error: {e}\n{traceback.format_exc()}"
            return StepResult(
                task_id=task.task_id,
                success=False,
                output=error_output[:2000],
                reflection="",
                agent_used="none",
                duration=duration,
            )

    async def _reflect_on_failure(self, task: Task, step_result: StepResult) -> str:
        """
        Analyze a failure and generate insights for improvement.

        Args:
            task: The failed task
            step_result: The execution result

        Returns:
            Reflection text with insights
        """
        prompt = f"""Task failed: {task.title}
Description: {task.description}
Output: {step_result.output[:1000]}

Analyze:
1. What went wrong?
2. Why did it fail?
3. What should be done differently?
4. What specific strategy should be used next?

Be specific and actionable."""

        try:
            response = await self._api.chat_sync(
                messages=[
                    {"role": "system", "content": "You are an expert failure analyst. Be specific and actionable."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            reflection = response["content"]
            await self._memory.store_long_term(
                key=f"reflection_{task.task_id}_{int(time.time())}",
                value=reflection,
                category="reflection",
                importance=0.7,
            )

            return reflection

        except Exception:
            return "Failed to generate reflection. Will retry with simpler approach."

    async def pause(self) -> None:
        """Pause execution."""
        self._state = EngineState.PAUSED
        console.print("[yellow]Execution paused[/yellow]")

    async def resume(self) -> None:
        """Resume execution."""
        if self._state == EngineState.PAUSED:
            self._state = EngineState.EXECUTING
            console.print("[green]Execution resumed[/green]")

    async def stop(self) -> None:
        """Stop execution."""
        self._state = EngineState.IDLE
        self._current_goal = None
        console.print("[yellow]Execution stopped[/yellow]")

    def on_event(self, callback: Callable) -> None:
        """Register an event callback."""
        self._event_callbacks.append(callback)

    def _emit_event(self, event_type: str, data: dict) -> None:
        """Emit an event to all registered callbacks."""
        for callback in self._event_callbacks:
            try:
                callback(event_type, data)
            except Exception:
                pass

    def get_state(self) -> EngineState:
        """Get current engine state."""
        return self._state

    def get_progress(self) -> dict:
        """Get current execution progress."""
        if not self._current_goal:
            return {"state": self._state.value, "progress": 0}

        total = len(self._current_goal.tasks)
        completed = sum(
            1 for t in self._current_goal.tasks
            if t.status == TaskStatus.COMPLETED
        )
        return {
            "state": self._state.value,
            "goal": self._current_goal.description[:100],
            "total_tasks": total,
            "completed": completed,
            "failed": sum(1 for t in self._current_goal.tasks if t.status == TaskStatus.FAILED),
            "in_progress": sum(1 for t in self._current_goal.tasks if t.status == TaskStatus.IN_PROGRESS),
            "pending": sum(1 for t in self._current_goal.tasks if t.status == TaskStatus.PENDING),
            "progress": round((completed / total) * 100, 1) if total > 0 else 0,
        }

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def shutdown(self) -> None:
        """Clean up engine resources."""
        self._state = EngineState.IDLE
        self._current_goal = None
        self._step_history.clear()
        self._event_callbacks.clear()
        self._initialized = False
