"""
Built-in tools — filesystem, git, shell, search, code, and more.
Every tool is fully implemented and production-ready.
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console

from zeta_cli.tools.base import BaseTool, ToolResult

console = Console()

# ─── Filesystem Tools ────────────────────────────────────────────

class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the contents of a file. Returns the file content as text."
    category = "filesystem"

    async def execute(self, path: str, encoding: str = "utf-8", start_line: int = 0, end_line: Optional[int] = None) -> ToolResult:
        try:
            file_path = Path(path).expanduser().resolve()
            if not file_path.exists():
                return ToolResult(False, "", error=f"File not found: {path}")
            if not file_path.is_file():
                return ToolResult(False, "", error=f"Not a file: {path}")

            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                lines = f.readlines()

            if end_line is not None and end_line > 0:
                lines = lines[start_line:end_line]
            elif start_line > 0:
                lines = lines[start_line:]

            content = "".join(lines)
            return ToolResult(
                True,
                content,
                metadata={
                    "path": str(file_path),
                    "lines": len(lines),
                    "size": file_path.stat().st_size,
                },
            )
        except Exception as e:
            return ToolResult(False, "", error=str(e))

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
                "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
                "start_line": {"type": "integer", "description": "Start line (0-indexed)", "default": 0},
                "end_line": {"type": "integer", "description": "End line (exclusive)"},
            },
            "required": ["path"],
        }

class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a file. Creates parent directories if needed."
    category = "filesystem"
    is_destructive = True

    async def execute(self, path: str, content: str, encoding: str = "utf-8", append: bool = False) -> ToolResult:
        try:
            file_path = Path(path).expanduser().resolve()
            file_path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            with open(file_path, mode, encoding=encoding) as f:
                f.write(content)

            return ToolResult(
                True,
                f"Successfully wrote {len(content)} bytes to {file_path}",
                metadata={
                    "path": str(file_path),
                    "size": len(content),
                    "mode": mode,
                },
            )
        except Exception as e:
            return ToolResult(False, "", error=str(e))

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write"},
                "encoding": {"type": "string", "description": "File encoding", "default": "utf-8"},
                "append": {"type": "boolean", "description": "Append instead of overwrite", "default": False},
            },
            "required": ["path", "content"],
        }

class ListDirectoryTool(BaseTool):
    name = "list_directory"
    description = "List contents of a directory with file details."
    category = "filesystem"

    async def execute(self, path: str = ".", recursive: bool = False, pattern: str = "*", max_depth: int = 3) -> ToolResult:
        try:
            dir_path = Path(path).expanduser().resolve()
            if not dir_path.exists():
                return ToolResult(False, "", error=f"Directory not found: {path}")

            entries = []
            if recursive:
                for p in dir_path.rglob(pattern):
                    if p.is_file() or p.is_dir():
                        depth = len(p.relative_to(dir_path).parts)
                        if depth <= max_depth:
                            entries.append(self._format_entry(p, dir_path))
            else:
                for p in sorted(dir_path.iterdir()):
                    if p.match(pattern):
                        entries.append(self._format_entry(p, dir_path))

            output = "\n".join(entries) if entries else "(empty directory)"
            return ToolResult(
                True,
                output,
                metadata={"path": str(dir_path), "count": len(entries), "recursive": recursive},
            )
        except Exception as e:
            return ToolResult(False, "", error=str(e))

    def _format_entry(self, p: Path, root: Path) -> str:
        rel = p.relative_to(root)
        prefix = "📁 " if p.is_dir() else "📄 "
        try:
            stat = p.stat()
            size = self._format_size(stat.st_size) if p.is_file() else ""
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            return f"{prefix}{str(rel):<50} {size:>10}  {mtime}"
        except OSError:
            return f"{prefix}{rel}"

    def _format_size(self, size: int) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path", "default": "."},
                "recursive": {"type": "boolean", "description": "Recursive listing", "default": False},
                "pattern": {"type": "string", "description": "Glob pattern filter", "default": "*"},
                "max_depth": {"type": "integer", "description": "Max recursion depth", "default": 3},
            },
            "required": [],
        }

class DeleteFileTool(BaseTool):
    name = "delete_file"
    description = "Delete a file or empty directory."
    category = "filesystem"
    is_destructive = True

    async def execute(self, path: str, force: bool = False) -> ToolResult:
        try:
            target = Path(path).expanduser().resolve()
            if not target.exists():
                return ToolResult(False, "", error=f"Not found: {path}")

            if target.is_dir():
                if force:
                    shutil.rmtree(target)
                    return ToolResult(True, f"Deleted directory tree: {target}")
                else:
                    target.rmdir()
                    return ToolResult(True, f"Deleted empty directory: {target}")
            else:
                target.unlink()
                return ToolResult(True, f"Deleted file: {target}")

        except OSError as e:
            return ToolResult(False, "", error=str(e))

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to delete"},
                "force": {"type": "boolean", "description": "Force recursive delete", "default": False},
            },
            "required": ["path"],
        }

class SearchFilesTool(BaseTool):
    name = "search_files"
    description = "Search for files by name pattern recursively."
    category = "filesystem"

    async def execute(self, pattern: str, directory: str = ".", max_results: int = 50) -> ToolResult:
        try:
            base = Path(directory).expanduser().resolve()
            results = []
            for p in base.rglob(pattern):
                if p.is_file():
                    results.append(str(p.relative_to(base)))
                if len(results) >= max_results:
                    break

            output = "\n".join(results) if results else "No files found."
            return ToolResult(
                True,
                output,
                metadata={"count": len(results), "pattern": pattern},
            )
        except Exception as e:
            return ToolResult(False, "", error=str(e))

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "File name pattern (supports glob)"},
                "directory": {"type": "string", "description": "Base directory", "default": "."},
                "max_results": {"type": "integer", "description": "Maximum results", "default": 50},
            },
            "required": ["pattern"],
        }

class MoveFileTool(BaseTool):
    name = "move_file"
    description = "Move or rename a file/directory."
    category = "filesystem"
    is_destructive = True

    async def execute(self, source: str, destination: str) -> ToolResult:
        try:
            src = Path(source).expanduser().resolve()
            dst = Path(destination).expanduser().resolve()

            if not src.exists():
                return ToolResult(False, "", error=f"Source not found: {source}")

            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

            return ToolResult(True, f"Moved {src} -> {dst}")
        except Exception as e:
            return ToolResult(False, "", error=str(e))

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source path"},
                "destination": {"type": "string", "description": "Destination path"},
            },
            "required": ["source", "destination"],
        }

# ─── Shell/Command Tools ─────────────────────────────────────────

class RunCommandTool(BaseTool):
    name = "run_command"
    description = "Execute a shell command and return the output."
    category = "shell"
    requires_sandbox = True

    async def execute(self, command: str, cwd: str = ".", timeout: int = 60, shell: str = "cmd") -> ToolResult:
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path(cwd).expanduser().resolve()),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(False, "", error=f"Command timed out after {timeout}s")

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")

            if process.returncode == 0:
                return ToolResult(
                    True,
                    output or "(no output)",
                    metadata={"exit_code": process.returncode, "stderr": error_output},
                )
            else:
                return ToolResult(
                    False,
                    output,
                    error=error_output,
                    metadata={"exit_code": process.returncode},
                )

        except Exception as e:
            return ToolResult(False, "", error=str(e))

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "cwd": {"type": "string", "description": "Working directory", "default": "."},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 60},
            },
            "required": ["command"],
        }

class PowerShellTool(BaseTool):
    name = "powershell"
    description = "Execute a PowerShell command and return the output."
    category = "shell"
    requires_sandbox = True

    async def execute(self, command: str, cwd: str = ".", timeout: int = 60) -> ToolResult:
        try:
            process = await asyncio.create_subprocess_exec(
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path(cwd).expanduser().resolve()),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(False, "", error=f"Command timed out after {timeout}s")

            output = stdout.decode("utf-8", errors="replace")
            return ToolResult(
                True,
                output.strip() or "(no output)",
                metadata={"exit_code": process.returncode},
            )

        except Exception as e:
            return ToolResult(False, "", error=str(e))

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "PowerShell command"},
                "cwd": {"type": "string", "description": "Working directory", "default": "."},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 60},
            },
            "required": ["command"],
        }

class PythonTool(BaseTool):
    name = "python"
    description = "Execute Python code and return the output."
    category = "shell"
    requires_sandbox = True

    async def execute(self, code: str, timeout: int = 30) -> ToolResult:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
                f.write(code)
                temp_path = f.name

            try:
                process = await asyncio.create_subprocess_exec(
                    "python",
                    temp_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    return ToolResult(False, "", error=f"Execution timed out after {timeout}s")

                output = stdout.decode("utf-8", errors="replace")
                error_output = stderr.decode("utf-8", errors="replace")

                if process.returncode == 0:
                    return ToolResult(True, output or "(no output)")
                else:
                    return ToolResult(False, output, error=error_output)

            finally:
                Path(temp_path).unlink(missing_ok=True)

        except Exception as e:
            return ToolResult(False, "", error=str(e))

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
            },
            "required": ["code"],
        }

# ─── Git Tools ────────────────────────────────────────────────────

class GitTool(BaseTool):
    name = "git"
    description = "Execute a git command in the workspace."
    category = "git"

    async def execute(self, command: str, cwd: str = ".") -> ToolResult:
        try:
            full_cmd = f"git {command}"
            process = await asyncio.create_subprocess_shell(
                full_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path(cwd).expanduser().resolve()),
            )

            stdout, stderr = await process.communicate()
            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")

            if process.returncode == 0:
                return ToolResult(True, output or "OK")
            else:
                return ToolResult(False, output, error=error_output)

        except FileNotFoundError:
            return ToolResult(False, "", error="Git is not installed or not in PATH")
        except Exception as e:
            return ToolResult(False, "", error=str(e))

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Git command (e.g., 'status', 'log --oneline -10')"},
                "cwd": {"type": "string", "description": "Working directory", "default": "."},
            },
            "required": ["command"],
        }

# ─── Search Tools ─────────────────────────────────────────────────

class GrepTool(BaseTool):
    name = "grep"
    description = "Search for a pattern in files using regex."
    category = "search"

    async def execute(
        self,
        pattern: str,
        path: str = ".",
        file_pattern: str = "*",
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> ToolResult:
        try:
            base = Path(path).expanduser().resolve()
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)

            results = []
            for file_path in base.rglob(file_pattern):
                try:
                    if file_path.is_file() and file_path.stat().st_size < 1_000_000:  # Skip files > 1MB
                        content = file_path.read_text(errors="replace")
                        for i, line in enumerate(content.splitlines(), 1):
                            if regex.search(line):
                                rel_path = file_path.relative_to(base)
                                results.append(f"{rel_path}:{i}: {line.strip()[:200]}")
                                if len(results) >= max_results:
                                    break
                except Exception:
                    continue
                if len(results) >= max_results:
                    break

            output = "\n".join(results) if results else "No matches found."
            return ToolResult(True, output, metadata={"matches": len(results)})

        except Exception as e:
            return ToolResult(False, "", error=str(e))

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Base path", "default": "."},
                "file_pattern": {"type": "string", "description": "File glob filter", "default": "*"},
                "case_sensitive": {"type": "boolean", "description": "Case sensitive search", "default": False},
                "max_results": {"type": "integer", "description": "Maximum results", "default": 100},
            },
            "required": ["pattern"],
        }

# ─── Web/HTTP Tools ────────────────────────────────────────────────

class HTTPGetTool(BaseTool):
    name = "http_get"
    description = "Perform an HTTP GET request."
    category = "web"

    async def execute(self, url: str, headers: str = "{}", timeout: int = 30) -> ToolResult:
        try:
            import httpx

            header_dict = json.loads(headers)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, headers=header_dict, follow_redirects=True)
                return ToolResult(
                    True,
                    response.text[:10000],  # Limit response size
                    metadata={
                        "status_code": response.status_code,
                        "content_type": response.headers.get("content-type", ""),
                        "content_length": len(response.text),
                    },
                )
        except Exception as e:
            return ToolResult(False, "", error=str(e))

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
                "headers": {"type": "string", "description": "JSON string of headers", "default": "{}"},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
            },
            "required": ["url"],
        }

# ─── File Manipulation Tools ──────────────────────────────────────

class ZipTool(BaseTool):
    name = "zip"
    description = "Create or extract ZIP archives."
    category = "filesystem"

    async def execute(self, action: str, source: str, destination: str) -> ToolResult:
        try:
            src = Path(source).expanduser().resolve()
            dst = Path(destination).expanduser().resolve()

            if action == "create":
                dst.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zf:
                    if src.is_dir():
                        for f in src.rglob("*"):
                            zf.write(f, f.relative_to(src))
                    else:
                        zf.write(src, src.name)
                return ToolResult(True, f"Created archive: {dst}")

            elif action == "extract":
                dst.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(src, "r") as zf:
                    zf.extractall(dst)
                return ToolResult(True, f"Extracted to: {dst}")

            else:
                return ToolResult(False, "", error=f"Unknown action: {action}. Use 'create' or 'extract'.")

        except Exception as e:
            return ToolResult(False, "", error=str(e))

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "Action: 'create' or 'extract'"},
                "source": {"type": "string", "description": "Source path"},
                "destination": {"type": "string", "description": "Destination path"},
            },
            "required": ["action", "source", "destination"],
        }

class DiffTool(BaseTool):
    name = "diff"
    description = "Generate a unified diff between two files."
    category = "filesystem"

    async def execute(self, file1: str, file2: str, context_lines: int = 3) -> ToolResult:
        try:
            import difflib

            p1 = Path(file1).expanduser().resolve()
            p2 = Path(file2).expanduser().resolve()

            if not p1.exists():
                return ToolResult(False, "", error=f"File not found: {file1}")
            if not p2.exists():
                return ToolResult(False, "", error=f"File not found: {file2}")

            lines1 = p1.read_text(errors="replace").splitlines(keepends=True)
            lines2 = p2.read_text(errors="replace").splitlines(keepends=True)

            diff = difflib.unified_diff(
                lines1, lines2,
                fromfile=str(p1), tofile=str(p2),
                n=context_lines,
            )
            output = "".join(diff)
            return ToolResult(True, output or "(no differences)")

        except Exception as e:
            return ToolResult(False, "", error=str(e))

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file1": {"type": "string", "description": "First file path"},
                "file2": {"type": "string", "description": "Second file path"},
                "context_lines": {"type": "integer", "description": "Context lines", "default": 3},
            },
            "required": ["file1", "file2"],
        }

# ─── System Info Tools ────────────────────────────────────────────

class SystemInfoTool(BaseTool):
    name = "system_info"
    description = "Get system information (OS, CPU, memory, disk)."
    category = "system"

    async def execute(self) -> ToolResult:
        try:
            import platform
            import psutil

            info = {
                "os": platform.system(),
                "os_version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=0.5),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 1),
                "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 1),
                "disk_usage": {},
            }

            for part in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    info["disk_usage"][part.device] = {
                        "mount": part.mountpoint,
                        "total_gb": round(usage.total / (1024**3), 1),
                        "used_gb": round(usage.used / (1024**3), 1),
                        "free_gb": round(usage.free / (1024**3), 1),
                        "percent": usage.percent,
                    }
                except Exception:
                    pass

            return ToolResult(True, json.dumps(info, indent=2), metadata=info)

        except Exception as e:
            return ToolResult(False, "", error=str(e))

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

# Export all tool classes
__all__ = [
    "ReadFileTool",
    "WriteFileTool",
    "ListDirectoryTool",
    "DeleteFileTool",
    "SearchFilesTool",
    "MoveFileTool",
    "RunCommandTool",
    "PowerShellTool",
    "PythonTool",
    "GitTool",
    "GrepTool",
    "HTTPGetTool",
    "ZipTool",
    "DiffTool",
    "SystemInfoTool",
]
