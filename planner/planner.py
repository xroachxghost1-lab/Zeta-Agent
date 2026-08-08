"""
Task Planner — Hierarchical goal decomposition with dependency management.

Features:
- Goal analysis and task breakdown
- Priority assignment
- Dependency graph construction
- Dynamic replanning
- Task complexity estimation
- Automatic retry with strategy variation
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from rich.console import Console

from zeta_cli.config.manager import ConfigManager
from zeta_cli.api.manager import APIManager
from zeta_cli.memory.manager import MemoryManager

console = Console()

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """A single task in the plan."""
    task_id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5
    complexity: float = 0.5
    dependencies: list[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    goal_id: Optional[str] = None
    result: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

@dataclass
class Goal:
    """Top-level goal with its task tree."""
    goal_id: str
    description: str
    tasks: list[Task] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    metadata: dict = field(default_factory=dict)

class TaskPlanner:
    """
    Hierarchical task planner with automatic goal decomposition.

    Takes a high-level goal and:
    1. Analyzes the goal
    2. Breaks it into subtasks
    3. Creates dependency graph
    4. Prioritizes execution order
    5. Monitors progress
    6. Dynamically replans on failure
    """

    def __init__(self, config: ConfigManager, api: APIManager, memory: MemoryManager):
        self._config = config
        self._api = api
        self._memory = memory
        self._active_goals: dict[str, Goal] = {}
        self._max_depth = config.get("planner.max_depth", 5)
        self._max_tasks = config.get("planner.max_tasks", 100)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize planner and restore any active goals from memory."""
        pending_tasks = await self._memory.get_all_pending_tasks()
        if pending_tasks:
            console.print(f"[dim]Restored {len(pending_tasks)} pending tasks from memory[/dim]")
            goals: dict[str, Goal] = {}
            for row in pending_tasks:
                goal_id = row.get("goal_id") or "unknown"
                if goal_id not in goals:
                    description = row.get("metadata", {}).get("goal_description", goal_id)
                    goals[goal_id] = Goal(goal_id=goal_id, description=description, tasks=[])
                task_status = TaskStatus(row.get("status", TaskStatus.PENDING.value))
                task = Task(
                    task_id=row.get("task_id"),
                    title=row.get("title", "Untitled Task"),
                    description=row.get("description", ""),
                    status=task_status,
                    priority=row.get("priority", 5),
                    complexity=row.get("complexity", 0.5),
                    dependencies=row.get("dependencies", []),
                    parent_id=row.get("parent_id"),
                    goal_id=goal_id,
                    result=row.get("result"),
                    retry_count=row.get("retry_count", 0),
                    created_at=row.get("created_at", time.time()),
                    updated_at=row.get("updated_at", time.time()),
                    metadata=row.get("metadata", {}),
                )
                goals[goal_id].tasks.append(task)

            self._active_goals = goals
        self._initialized = True

    async def create_goal(self, description: str) -> Goal:
        """
        Create a new goal and generate initial task plan.

        Args:
            description: Natural language goal description

        Returns:
            Goal object with generated tasks
        """
        goal_id = hashlib.sha256(
            f"{description}-{time.time()}".encode()
        ).hexdigest()[:12]

        console.print(f"\n[bold cyan]Analyzing goal:[/bold cyan] {description}")

        # Generate task breakdown using LLM
        tasks = await self._decompose_goal(description, goal_id)

        goal = Goal(
            goal_id=goal_id,
            description=description,
            tasks=tasks,
        )
        self._active_goals[goal_id] = goal

        # Persist all tasks
        for task in tasks:
            await self._memory.create_task({
                "task_id": task.task_id,
                "goal_id": goal_id,
                "parent_id": task.parent_id,
                "title": task.title,
                "description": task.description,
                "status": task.status.value,
                "priority": task.priority,
                "complexity": task.complexity,
                "dependencies": task.dependencies,
                "metadata": {"goal_description": description},
            })

        console.print(f"[green]Created {len(tasks)} tasks for goal {goal_id}[/green]")
        return goal

    async def get_next_task(self, goal_id: str) -> Optional[Task]:
        """
        Get the next executable task for a goal.

        Returns the highest-priority task whose dependencies are all completed.

        Args:
            goal_id: Goal identifier

        Returns:
            Next Task to execute, or None if all tasks are complete/blocked
        """
        goal = self._active_goals.get(goal_id)
        if not goal:
            return None

        # Find tasks that are ready to execute
        executable = []
        completed_ids = {
            t.task_id.lower() for t in goal.tasks if t.status == TaskStatus.COMPLETED
        } | {
            t.title.lower() for t in goal.tasks if t.status == TaskStatus.COMPLETED
        }

        for task in goal.tasks:
            if task.status in (TaskStatus.PENDING, TaskStatus.RETRY):
                # Check if all dependencies are completed
                if all(dep.lower() in completed_ids for dep in task.dependencies):
                    executable.append(task)

        if not executable:
            return None

        # Sort by priority (higher first), then complexity (lower first)
        executable.sort(key=lambda t: (-t.priority, t.complexity))
        return executable[0]

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[str] = None,
    ) -> None:
        """
        Update a task's status and result.

        Args:
            task_id: Task identifier
            status: New status
            result: Optional result text
        """
        for goal in self._active_goals.values():
            for task in goal.tasks:
                if task.task_id == task_id:
                    task.status = status
                    task.updated_at = time.time()
                    if result:
                        task.result = result
                    if status == TaskStatus.FAILED:
                        task.retry_count += 1
                        if task.retry_count < task.max_retries:
                            task.status = TaskStatus.RETRY

                    # Persist update
                    await self._memory.update_task(task_id, {
                        "status": task.status.value,
                        "result": result,
                        "retry_count": task.retry_count,
                        "updated_at": task.updated_at,
                    })
                    return

    async def check_goal_completion(self, goal_id: str) -> bool:
        """
        Check if all tasks for a goal are completed.

        Args:
            goal_id: Goal identifier

        Returns:
            True if goal is complete
        """
        goal = self._active_goals.get(goal_id)
        if not goal:
            return False

        all_done = all(
            t.status == TaskStatus.COMPLETED
            for t in goal.tasks
        )

        if all_done and goal.status != TaskStatus.COMPLETED:
            goal.status = TaskStatus.COMPLETED
            goal.completed_at = time.time()
            console.print(f"[bold green]Goal completed: {goal.description}[/bold green]")

        return all_done

    async def replan(self, goal_id: str, failed_task: Task, error: str) -> list[Task]:
        """
        Dynamically replan when a task fails.

        Generates alternative tasks based on failure analysis.

        Args:
            goal_id: Goal identifier
            failed_task: The task that failed
            error: Error description

        Returns:
            List of new tasks to replace the failed one
        """
        goal = self._active_goals.get(goal_id)
        if not goal:
            return []

        console.print(f"[yellow]Replanning after failure: {failed_task.title}[/yellow]")

        # Generate alternative approach
        remaining_retries = max(1, failed_task.max_retries - failed_task.retry_count)
        prompt = f"""Task failed: {failed_task.title}
Description: {failed_task.description}
Error: {error}

Generate {remaining_retries} alternative approaches to accomplish this task.
Each approach should be substantially different from the failed one.
Output as JSON array with 'title', 'description', 'strategy' fields."""

        try:
            response = await self._api.chat_sync(
                messages=[
                    {"role": "system", "content": "You are a task planning expert. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
            )

            raw_content = response.get("content", "").strip()
            if raw_content.startswith("```"):
                lines = raw_content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_content = "\n".join(lines).strip()

            alternatives = json.loads(raw_content)
        except (json.JSONDecodeError, KeyError):
            # Fallback: create a simple retry task
            alternatives = [{
                "title": f"Retry: {failed_task.title}",
                "description": f"Retry {failed_task.description} with different approach",
                "strategy": "retry_with_variation",
            }]

        new_tasks = []
        for i, alt in enumerate(alternatives):
            task = Task(
                task_id=hashlib.sha256(
                    f"{failed_task.task_id}-replan-{i}-{time.time()}".encode()
                ).hexdigest()[:12],
                title=alt.get("title", f"Alternative {i+1}: {failed_task.title}"),
                description=alt.get("description", failed_task.description),
                priority=failed_task.priority,
                complexity=failed_task.complexity + 0.1,
                dependencies=failed_task.dependencies,
                parent_id=failed_task.parent_id,
                goal_id=goal_id,
                max_retries=failed_task.max_retries,
                metadata={"replan_of": failed_task.task_id, "strategy": alt.get("strategy", "unknown")},
            )
            new_tasks.append(task)
            goal.tasks.append(task)

            await self._memory.create_task({
                "task_id": task.task_id,
                "goal_id": goal_id,
                "parent_id": task.parent_id,
                "title": task.title,
                "description": task.description,
                "status": TaskStatus.PENDING.value,
                "priority": task.priority,
                "complexity": task.complexity,
                "dependencies": task.dependencies,
                "metadata": task.metadata,
            })

        # Mark original as cancelled
        await self.update_task_status(failed_task.task_id, TaskStatus.CANCELLED)

        return new_tasks

    async def get_plan_summary(self, goal_id: str) -> str:
        """Generate a human-readable plan summary."""
        goal = self._active_goals.get(goal_id)
        if not goal:
            return "Goal not found."

        lines = [
            f"Goal: {goal.description}",
            f"Status: {goal.status.value}",
            f"Tasks: {len(goal.tasks)} total",
            "-" * 50,
        ]

        status_counts = {}
        for task in sorted(goal.tasks, key=lambda t: t.priority, reverse=True):
            status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1
            dep_str = f" (depends on: {', '.join(task.dependencies[:3])})" if task.dependencies else ""
            lines.append(
                f"  [{task.status.value:12}] P{task.priority} [{task.complexity:.1f}] {task.title}{dep_str}"
            )

        lines.insert(3, f"Status distribution: {json.dumps(status_counts)}")
        return "\n".join(lines)

    async def _decompose_goal(self, goal: str, goal_id: str) -> list[Task]:
        """
        Use LLM to decompose a goal into hierarchical tasks.

        Args:
            goal: The natural language goal
            goal_id: Goal identifier for task association

        Returns:
            List of Task objects
        """
        prompt = f"""Break down this goal into a hierarchical task plan:
Goal: {goal}

Output a JSON object with a 'tasks' array. Each task should have:
- title: Short task name
- description: Detailed description of what to do
- priority: 1-10 (10 highest)
- complexity: 0.0-1.0 estimate
- dependencies: Array of task titles that must complete first
- subtasks: Array of smaller tasks (optional, max depth 3)

Focus on actionable, concrete tasks. Order them logically.
Maximum {self._max_tasks} tasks total across all levels."""

        try:
            response = await self._api.chat_sync(
                messages=[
                    {"role": "system", "content": "You are a task planning expert. Output valid JSON only. No markdown."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            raw_content = response.get("content", "").strip()
            if raw_content.startswith("```"):
                lines = raw_content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_content = "\n".join(lines).strip()

            plan = json.loads(raw_content)
            tasks = self._parse_task_list(plan.get("tasks", []), goal_id, None, depth=0)

        except (json.JSONDecodeError, KeyError) as e:
            console.print(f"[yellow]Plan parsing error: {e}. Using fallback plan.[/yellow]")
            tasks = self._create_fallback_plan(goal, goal_id)

        return tasks

    def _parse_task_list(
        self,
        task_list: list[dict],
        goal_id: str,
        parent_id: Optional[str],
        depth: int,
    ) -> list[Task]:
        """Recursively parse task definitions into Task objects."""
        if depth >= self._max_depth:
            return []

        tasks = []
        for item in task_list:
            if len(tasks) >= self._max_tasks:
                break

            task = Task(
                task_id=hashlib.sha256(
                    f"{item.get('title', 'task')}-{time.time()}-{len(tasks)}".encode()
                ).hexdigest()[:12],
                title=item.get("title", "Untitled Task"),
                description=item.get("description", ""),
                priority=min(max(item.get("priority", 5), 1), 10),
                complexity=min(max(item.get("complexity", 0.5), 0.0), 1.0),
                dependencies=item.get("dependencies", []),
                parent_id=parent_id,
                goal_id=goal_id,
            )
            tasks.append(task)

            # Process subtasks
            subtasks = item.get("subtasks", [])
            if subtasks:
                child_tasks = self._parse_task_list(
                    subtasks, goal_id, task.task_id, depth + 1
                )
                tasks.extend(child_tasks)

        return tasks

    def _create_fallback_plan(self, goal: str, goal_id: str) -> list[Task]:
        """Create a simple fallback plan when LLM parsing fails."""
        steps = [
            ("Analyze requirements", "Understand what needs to be built", 10, 0.2),
            ("Set up project structure", "Create directories and config files", 9, 0.1),
            ("Implement core logic", "Build the main functionality", 8, 0.7),
            ("Add error handling", "Handle edge cases and failures", 7, 0.3),
            ("Write tests", "Create unit and integration tests", 6, 0.4),
            ("Create documentation", "Write README and API docs", 5, 0.2),
            ("Review and refactor", "Code review and improvements", 4, 0.3),
        ]

        tasks = []
        prev_id = None
        for i, (title, desc, priority, complexity) in enumerate(steps):
            task = Task(
                task_id=hashlib.sha256(f"{title}-{goal_id}-{i}".encode()).hexdigest()[:12],
                title=f"{i+1}. {title}",
                description=desc,
                priority=priority,
                complexity=complexity,
                dependencies=[prev_id] if prev_id else [],
                goal_id=goal_id,
            )
            tasks.append(task)
            prev_id = task.task_id

        return tasks

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        return self._active_goals.get(goal_id)

    def list_active_goals(self) -> list[Goal]:
        """List all active goals."""
        return list(self._active_goals.values())

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def shutdown(self) -> None:
        """Clean up planner resources."""
        self._active_goals.clear()
        self._initialized = False
