"""
Specialized agent implementations.

Each agent has a distinct responsibility and system prompt.
All agents use the shared API, tools, and memory systems.
"""

import json
from typing import Any, Optional

from zeta_cli.agents.base import BaseAgent, AgentResult
from zeta_cli.api.manager import APIManager
from zeta_cli.tools.registry import ToolRegistry
from zeta_cli.memory.manager import MemoryManager

class CoderAgent(BaseAgent):
    """Generates high-quality, production-ready code."""

    name = "coder"
    description = "Generates production-ready code with full implementations"
    system_prompt = """You are an expert software engineer. Your code is production-grade.
- Write complete, working implementations. No placeholders, no TODOs.
- Include proper error handling, type hints, and docstrings.
- Follow best practices for the language and framework.
- Write tests alongside implementation code.
- Consider edge cases, performance, and security.
- Output clean, well-structured code ready for review.
- When modifying existing code, preserve style and patterns."""

    def __init__(self, api: APIManager, tools: ToolRegistry, memory: MemoryManager):
        self._api = api
        self._tools = tools
        self._memory = memory

    async def execute(self, task: str, context: dict, **kwargs) -> AgentResult:
        try:
            # Build context-rich prompt
            history = await self._memory.get_conversation_history(limit=10)
            history_text = "\n".join(
                f"[{h['role']}]: {h['content'][:500]}" for h in history
            )

            prompt = f"""Context from recent conversation:
{history_text}

Current workspace: {context.get('workspace', 'N/A')}
Available tools: {', '.join(t.name for t in self._tools.list_tools())}

Task: {task}

Generate complete, production-ready code. Include all necessary imports, type hints,
error handling, and documentation. Output the full implementation."""

            response = await self._api.chat_sync(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            return AgentResult(
                success=True,
                output=response["content"],
                agent_name=self.name,
                confidence=0.9,
                metadata={"usage": response.get("usage", {})},
            )

        except Exception as e:
            return AgentResult(
                success=False,
                output=f"Coder agent failed: {e}",
                agent_name=self.name,
                confidence=0.0,
            )

class ReviewerAgent(BaseAgent):
    """Reviews code for quality, bugs, and best practices."""

    name = "reviewer"
    description = "Reviews code for quality, bugs, security, and best practices"
    system_prompt = """You are an expert code reviewer. Analyze code thoroughly for:
- Correctness and logic errors
- Security vulnerabilities
- Performance issues
- Code style and readability
- Test coverage gaps
- Edge cases and error handling
- API design and architecture
Provide specific, actionable feedback with line references."""

    def __init__(self, api: APIManager, tools: ToolRegistry, memory: MemoryManager):
        self._api = api
        self._tools = tools
        self._memory = memory

    async def execute(self, task: str, context: dict, **kwargs) -> AgentResult:
        try:
            code = context.get("code", task)
            response = await self._api.chat_sync(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Review this code:\n\n```\n{code}\n```\n\nProvide detailed feedback."},
                ],
                temperature=0.2,
            )

            return AgentResult(
                success=True,
                output=response["content"],
                agent_name=self.name,
                confidence=0.85,
            )

        except Exception as e:
            return AgentResult(False, f"Reviewer agent failed: {e}", agent_name=self.name, confidence=0.0)

class DebuggerAgent(BaseAgent):
    """Analyzes errors and generates fixes."""

    name = "debugger"
    description = "Analyzes errors, stack traces, and generates fixes"
    system_prompt = """You are an expert debugger. Analyze errors and provide fixes.
- Parse stack traces to identify root causes
- Explain what went wrong clearly
- Provide specific code fixes
- Suggest preventive measures
- Consider environmental factors (OS, versions, dependencies)
- Output the exact code changes needed."""

    def __init__(self, api: APIManager, tools: ToolRegistry, memory: MemoryManager):
        self._api = api
        self._tools = tools
        self._memory = memory

    async def execute(self, task: str, context: dict, **kwargs) -> AgentResult:
        try:
            error_info = context.get("error", task)
            code = context.get("code", "")

            prompt = f"""Error to debug:
{error_info}

Relevant code:
```

{code[:5000] if code else 'No code provided'}

```

Analyze the error, identify the root cause, and provide the exact fix."""

            response = await self._api.chat_sync(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            return AgentResult(
                success=True,
                output=response["content"],
                agent_name=self.name,
                confidence=0.88,
            )

        except Exception as e:
            return AgentResult(False, f"Debugger agent failed: {e}", agent_name=self.name, confidence=0.0)

class ResearcherAgent(BaseAgent):
    """Researches topics, APIs, and best practices."""

    name = "researcher"
    description = "Researches technical topics, APIs, libraries, and best practices"
    system_prompt = """You are an expert technical researcher. Provide comprehensive research on:
- API documentation and usage
- Library comparisons and recommendations
- Best practices and patterns
- Performance characteristics
- Security considerations
- Version compatibility
Always cite specific documentation and sources when possible."""

    def __init__(self, api: APIManager, tools: ToolRegistry, memory: MemoryManager):
        self._api = api
        self._tools = tools
        self._memory = memory

    async def execute(self, task: str, context: dict, **kwargs) -> AgentResult:
        try:
            response = await self._api.chat_sync(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Research: {task}"},
                ],
                temperature=0.3,
            )

            return AgentResult(
                success=True,
                output=response["content"],
                agent_name=self.name,
                confidence=0.8,
            )

        except Exception as e:
            return AgentResult(False, f"Researcher agent failed: {e}", agent_name=self.name, confidence=0.0)

class ArchitectAgent(BaseAgent):
    """Designs system architecture and patterns."""

    name = "architect"
    description = "Designs system architecture, patterns, and component structure"
    system_prompt = """You are an expert software architect. Design robust systems with:
- Clean separation of concerns
- Scalable component architecture
- Appropriate design patterns
- Data flow and API design
- Deployment considerations
- Technology stack recommendations
Output structured architecture documents with diagrams (text-based)."""

    def __init__(self, api: APIManager, tools: ToolRegistry, memory: MemoryManager):
        self._api = api
        self._tools = tools
        self._memory = memory

    async def execute(self, task: str, context: dict, **kwargs) -> AgentResult:
        try:
            response = await self._api.chat_sync(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Design architecture for: {task}"},
                ],
                temperature=0.3,
            )

            return AgentResult(
                success=True,
                output=response["content"],
                agent_name=self.name,
                confidence=0.85,
            )

        except Exception as e:
            return AgentResult(False, f"Architect agent failed: {e}", agent_name=self.name, confidence=0.0)

class SecurityAgent(BaseAgent):
    """Analyzes security vulnerabilities and provides fixes."""

    name = "security"
    description = "Analyzes code and systems for security vulnerabilities"
    system_prompt = """You are an expert security engineer. Analyze for:
- OWASP Top 10 vulnerabilities
- Input validation issues
- Authentication/authorization flaws
- Data exposure risks
- Dependency vulnerabilities
- Configuration weaknesses
- Cryptographic issues
Provide specific remediation steps with code examples."""

    def __init__(self, api: APIManager, tools: ToolRegistry, memory: MemoryManager):
        self._api = api
        self._tools = tools
        self._memory = memory

    async def execute(self, task: str, context: dict, **kwargs) -> AgentResult:
        try:
            code = context.get("code", task)
            response = await self._api.chat_sync(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Security analysis:\n\n```\n{code}\n```"},
                ],
                temperature=0.2,
            )

            return AgentResult(
                success=True,
                output=response["content"],
                agent_name=self.name,
                confidence=0.9,
            )

        except Exception as e:
            return AgentResult(False, f"Security agent failed: {e}", agent_name=self.name, confidence=0.0)

class PerformanceAgent(BaseAgent):
    """Optimizes code for performance."""

    name = "performance"
    description = "Analyzes and optimizes code for performance"
    system_prompt = """You are an expert performance engineer. Optimize for:
- Algorithmic complexity
- Memory usage
- I/O efficiency
- Caching strategies
- Concurrency/parallelism
- Database query optimization
- Network efficiency
Provide benchmarks and metrics where possible."""

    def __init__(self, api: APIManager, tools: ToolRegistry, memory: MemoryManager):
        self._api = api
        self._tools = tools
        self._memory = memory

    async def execute(self, task: str, context: dict, **kwargs) -> AgentResult:
        try:
            code = context.get("code", task)
            response = await self._api.chat_sync(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Performance analysis:\n\n```\n{code}\n```"},
                ],
                temperature=0.2,
            )

            return AgentResult(
                success=True,
                output=response["content"],
                agent_name=self.name,
                confidence=0.85,
            )

        except Exception as e:
            return AgentResult(False, f"Performance agent failed: {e}", agent_name=self.name, confidence=0.0)

class TesterAgent(BaseAgent):
    """Generates comprehensive test suites."""

    name = "tester"
    description = "Generates unit tests, integration tests, and test strategies"
    system_prompt = """You are an expert test engineer. Generate comprehensive tests:
- Unit tests covering all code paths
- Integration tests for component interactions
- Edge case and boundary testing
- Mock and fixture setup
- Test data generation
- Coverage analysis
Use pytest conventions. Tests must be runnable immediately."""

    def __init__(self, api: APIManager, tools: ToolRegistry, memory: MemoryManager):
        self._api = api
        self._tools = tools
        self._memory = memory

    async def execute(self, task: str, context: dict, **kwargs) -> AgentResult:
        try:
            code = context.get("code", task)
            response = await self._api.chat_sync(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Generate tests for:\n\n```\n{code}\n```"},
                ],
                temperature=0.2,
            )

            return AgentResult(
                success=True,
                output=response["content"],
                agent_name=self.name,
                confidence=0.85,
            )

        except Exception as e:
            return AgentResult(False, f"Tester agent failed: {e}", agent_name=self.name, confidence=0.0)

class DocumentationAgent(BaseAgent):
    """Generates comprehensive documentation."""

    name = "documentation"
    description = "Generates READMEs, API docs, docstrings, and architecture docs"
    system_prompt = """You are an expert technical writer. Generate clear documentation:
- README files with setup, usage, and examples
- API documentation with parameters and return values
- Architecture decision records (ADRs)
- Contributing guides
- Code docstrings (Google/NumPy style)
- Changelog entries
Write for both beginners and experienced developers."""

    def __init__(self, api: APIManager, tools: ToolRegistry, memory: MemoryManager):
        self._api = api
        self._tools = tools
        self._memory = memory

    async def execute(self, task: str, context: dict, **kwargs) -> AgentResult:
        try:
            code = context.get("code", task)
            response = await self._api.chat_sync(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Generate documentation for:\n\n```\n{code}\n```"},
                ],
                temperature=0.3,
            )

            return AgentResult(
                success=True,
                output=response["content"],
                agent_name=self.name,
                confidence=0.9,
            )

        except Exception as e:
            return AgentResult(False, f"Documentation agent failed: {e}", agent_name=self.name, confidence=0.0)
