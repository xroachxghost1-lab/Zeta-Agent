"""
Memory Manager — Persistent memory system with SQLite + vector embeddings.

Features:
- Conversation memory with token-aware truncation
- Workspace memory (file indexing)
- Task memory (task state persistence)
- Long-term memory with semantic search
- Automatic summarization and compression
- Vector embeddings via configurable backend
"""

import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rich.console import Console

from zeta_cli.config.manager import ConfigManager

console = Console()

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tokens INTEGER DEFAULT 0,
    timestamp REAL NOT NULL,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS workspace_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    hash TEXT,
    size INTEGER,
    last_modified REAL,
    indexed_at REAL,
    summary TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT UNIQUE NOT NULL,
    goal_id TEXT,
    parent_id TEXT,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    complexity REAL DEFAULT 0.5,
    dependencies TEXT DEFAULT '[]',
    result TEXT,
    created_at REAL,
    updated_at REAL,
    completed_at REAL,
    retry_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS long_term_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    category TEXT DEFAULT 'general',
    importance REAL DEFAULT 0.5,
    embedding BLOB,
    created_at REAL,
    last_accessed REAL,
    access_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS embeddings_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT UNIQUE NOT NULL,
    embedding BLOB NOT NULL,
    model TEXT NOT NULL,
    created_at REAL
);

CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_tasks_goal ON tasks(goal_id, status);
CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_id);
CREATE INDEX IF NOT EXISTS idx_ltm_category ON long_term_memory(category, importance);
CREATE INDEX IF NOT EXISTS idx_workspace_path ON workspace_files(path);
"""

class MemoryManager:
    """
    Comprehensive memory system with persistent storage and semantic search.

    Manages:
    - Conversation history with token tracking
    - Workspace file indexing
    - Task state persistence
    - Long-term memory with vector search
    - Automatic summarization
    - Memory compression
    """

    def __init__(self, config: ConfigManager):
        self._config = config
        self._db_path = Path(config.get("system.data_dir")) / "memory.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._current_session: Optional[str] = None
        self._conversation_limit = config.get("memory.conversation_limit", 100)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database and create schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

        self._current_session = hashlib.sha256(
            f"{time.time()}-{Path.home()}".encode()
        ).hexdigest()[:16]

        self._initialized = True
        console.print(f"[dim]Memory initialized at {self._db_path}[/dim]")

    # ─── Conversation Memory ─────────────────────────────────────

    async def add_conversation(
        self,
        role: str,
        content: str,
        tokens: int = 0,
        metadata: Optional[dict] = None,
    ) -> int:
        """Add a conversation turn."""
        cursor = self._conn.execute(
            """INSERT INTO conversations (session_id, role, content, tokens, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                self._current_session,
                role,
                content,
                tokens,
                time.time(),
                json.dumps(metadata or {}),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid

    async def get_conversation_history(
        self, limit: Optional[int] = None, session_id: Optional[str] = None
    ) -> list[dict]:
        """Get recent conversation history."""
        limit = limit or self._conversation_limit
        session = session_id or self._current_session
        cursor = self._conn.execute(
            """SELECT role, content, tokens, timestamp, metadata
               FROM conversations
               WHERE session_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (session, limit),
        )
        rows = cursor.fetchall()
        return [
            {
                "role": r[0],
                "content": r[1],
                "tokens": r[2],
                "timestamp": r[3],
                "metadata": json.loads(r[4]),
            }
            for r in reversed(rows)
        ]

    async def get_total_tokens(self, session_id: Optional[str] = None) -> int:
        """Get total tokens used in session."""
        session = session_id or self._current_session
        cursor = self._conn.execute(
            "SELECT COALESCE(SUM(tokens), 0) FROM conversations WHERE session_id = ?",
            (session,),
        )
        return cursor.fetchone()[0]

    async def compress_conversation(self) -> str:
        """
        Compress conversation history by summarizing older messages.
        Returns summary of compressed content.
        """
        history = await self.get_conversation_history()
        if len(history) <= self._conversation_limit:
            return "No compression needed."

        # Keep most recent messages, summarize older ones
        to_summarize = history[:-self._conversation_limit // 2]
        to_keep = history[-self._conversation_limit // 2:]

        # Build summary
        summary_parts = []
        for msg in to_summarize:
            if msg["role"] in ("user", "assistant"):
                summary_parts.append(f"[{msg['role']}]: {msg['content'][:200]}...")

        summary = " | ".join(summary_parts)

        # Store summary
        await self.store_long_term(
            key=f"conv_summary_{self._current_session}_{int(time.time())}",
            value=summary,
            category="conversation_summary",
            importance=0.3,
        )

        return summary

    # ─── Task Memory ─────────────────────────────────────────────

    async def create_task(self, task_data: dict) -> str:
        """Create a new task in memory."""
        task_id = task_data.get("task_id", hashlib.sha256(
            f"{task_data.get('title')}-{time.time()}".encode()
        ).hexdigest()[:12])

        self._conn.execute(
            """INSERT OR REPLACE INTO tasks
               (task_id, goal_id, parent_id, title, description, status, priority,
                complexity, dependencies, created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                task_id,
                task_data.get("goal_id"),
                task_data.get("parent_id"),
                task_data["title"],
                task_data.get("description", ""),
                task_data.get("status", "pending"),
                task_data.get("priority", 5),
                task_data.get("complexity", 0.5),
                json.dumps(task_data.get("dependencies", [])),
                time.time(),
                time.time(),
                json.dumps(task_data.get("metadata", {})),
            ),
        )
        self._conn.commit()
        return task_id

    async def update_task(self, task_id: str, updates: dict) -> bool:
        """Update task fields."""
        updates["updated_at"] = time.time()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        raw_values = list(updates.values())
        processed_values = [
            json.dumps(v) if isinstance(v, (dict, list)) else v
            for v in raw_values
        ] + [task_id]

        self._conn.execute(
            f"UPDATE tasks SET {set_clause} WHERE task_id = ?",
            processed_values,
        )
        self._conn.commit()
        return True

    async def get_task(self, task_id: str) -> Optional[dict]:
        """Get a task by ID."""
        cursor = self._conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        )
        row = cursor.fetchone()
        if row:
            return self._row_to_task(row)
        return None

    async def get_tasks_by_goal(self, goal_id: str, status: Optional[str] = None) -> list[dict]:
        """Get all tasks for a goal, optionally filtered by status."""
        if status:
            cursor = self._conn.execute(
                "SELECT * FROM tasks WHERE goal_id = ? AND status = ? ORDER BY priority DESC",
                (goal_id, status),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM tasks WHERE goal_id = ? ORDER BY priority DESC",
                (goal_id,),
            )
        return [self._row_to_task(row) for row in cursor.fetchall()]

    async def get_all_pending_tasks(self) -> list[dict]:
        """Get all pending tasks across all goals."""
        cursor = self._conn.execute(
            "SELECT * FROM tasks WHERE status IN ('pending', 'in_progress', 'retry') ORDER BY priority DESC"
        )
        return [self._row_to_task(row) for row in cursor.fetchall()]

    # ─── Long-Term Memory ────────────────────────────────────────

    async def store_long_term(
        self,
        key: str,
        value: str,
        category: str = "general",
        importance: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> None:
        """Store a long-term memory entry."""
        self._conn.execute(
            """INSERT OR REPLACE INTO long_term_memory
               (key, value, category, importance, created_at, last_accessed, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                key,
                value,
                category,
                importance,
                time.time(),
                time.time(),
                json.dumps(metadata or {}),
            ),
        )
        self._conn.commit()

    async def retrieve_long_term(self, key: str) -> Optional[dict]:
        """Retrieve a long-term memory entry."""
        cursor = self._conn.execute(
            "SELECT * FROM long_term_memory WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if row:
            # Update access metadata
            self._conn.execute(
                "UPDATE long_term_memory SET last_accessed = ?, access_count = access_count + 1 WHERE key = ?",
                (time.time(), key),
            )
            self._conn.commit()
            return self._row_to_ltm(row)
        return None

    async def search_long_term(
        self, query: str, category: Optional[str] = None, limit: int = 10
    ) -> list[dict]:
        """
        Search long-term memory using keyword matching and importance scoring.

        For production, this would use vector similarity search via the
        embeddings stored in the database. The current implementation uses
        a relevance scoring based on keyword overlap and importance.
        """
        if category:
            cursor = self._conn.execute(
                "SELECT * FROM long_term_memory WHERE category = ? ORDER BY importance DESC",
                (category,),
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM long_term_memory ORDER BY importance DESC"
            )

        results = []
        query_terms = set(query.lower().split())

        for row in cursor.fetchall():
            entry = self._row_to_ltm(row)
            value_lower = entry["value"].lower()
            key_lower = entry["key"].lower()

            # Score based on keyword matches
            score = 0
            for term in query_terms:
                if term in key_lower:
                    score += 3
                if term in value_lower:
                    score += 1

            if score > 0:
                entry["relevance"] = score * entry["importance"]
                results.append(entry)

        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:limit]

    async def semantic_search(
        self, query_embedding: list[float], limit: int = 10
    ) -> list[dict]:
        """
        Search long-term memory using vector similarity.

        Args:
            query_embedding: Embedding vector of the search query
            limit: Maximum results

        Returns:
            List of matching memory entries with similarity scores
        """
        cursor = self._conn.execute(
            "SELECT * FROM long_term_memory WHERE embedding IS NOT NULL"
        )

        query_vec = np.array(query_embedding)
        results = []

        for row in cursor.fetchall():
            entry = self._row_to_ltm(row)
            if entry.get("embedding"):
                stored_vec = np.frombuffer(entry["embedding"], dtype=np.float32)
                if len(stored_vec) == len(query_vec):
                    similarity = np.dot(query_vec, stored_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(stored_vec) + 1e-10
                    )
                    entry["similarity"] = float(similarity)
                    results.append(entry)

        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return results[:limit]

    async def store_embedding(self, key: str, embedding: list[float]) -> None:
        """Store or update embedding for a memory entry."""
        blob = np.array(embedding, dtype=np.float32).tobytes()
        self._conn.execute(
            "UPDATE long_term_memory SET embedding = ? WHERE key = ?",
            (blob, key),
        )
        self._conn.commit()

    async def prune_memories(self, max_age_days: int = 90) -> int:
        """Remove old, low-importance memories."""
        cutoff = time.time() - (max_age_days * 86400)
        cursor = self._conn.execute(
            "DELETE FROM long_term_memory WHERE importance < 0.2 AND last_accessed < ?",
            (cutoff,),
        )
        self._conn.commit()
        return cursor.rowcount

    # ─── Workspace Memory ────────────────────────────────────────

    async def index_file(self, file_path: Path, summary: str = "") -> None:
        """Index a workspace file."""
        stat = file_path.stat()
        content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest() if file_path.exists() else ""

        self._conn.execute(
            """INSERT OR REPLACE INTO workspace_files
               (path, hash, size, last_modified, indexed_at, summary)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                str(file_path),
                content_hash,
                stat.st_size,
                stat.st_mtime,
                time.time(),
                summary,
            ),
        )
        self._conn.commit()

    async def get_file_info(self, file_path: Path) -> Optional[dict]:
        """Get indexed information about a workspace file."""
        cursor = self._conn.execute(
            "SELECT * FROM workspace_files WHERE path = ?", (str(file_path),)
        )
        row = cursor.fetchone()
        if row:
            return {
                "path": row[1],
                "hash": row[2],
                "size": row[3],
                "last_modified": row[4],
                "indexed_at": row[5],
                "summary": row[6],
            }
        return None

    # ─── Database Maintenance ────────────────────────────────────

    async def vacuum(self) -> None:
        """Optimize database storage."""
        self._conn.execute("VACUUM")

    async def get_stats(self) -> dict:
        """Get memory system statistics."""
        stats = {}
        for table in ["conversations", "tasks", "long_term_memory", "workspace_files"]:
            cursor = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]

        cursor = self._conn.execute(
            "SELECT COALESCE(SUM(tokens), 0) FROM conversations WHERE session_id = ?",
            (self._current_session,),
        )
        stats["current_session_tokens"] = cursor.fetchone()[0]
        stats["current_session"] = self._current_session

        return stats

    # ─── Helpers ─────────────────────────────────────────────────

    def _row_to_task(self, row: tuple) -> dict:
        """Convert database row to task dictionary."""
        return {
            "id": row[0],
            "task_id": row[1],
            "goal_id": row[2],
            "parent_id": row[3],
            "title": row[4],
            "description": row[5],
            "status": row[6],
            "priority": row[7],
            "complexity": row[8],
            "dependencies": json.loads(row[9]),
            "result": row[10],
            "created_at": row[11],
            "updated_at": row[12],
            "completed_at": row[13],
            "retry_count": row[14],
            "metadata": json.loads(row[15]),
        }

    def _row_to_ltm(self, row: tuple) -> dict:
        """Convert database row to long-term memory dictionary."""
        return {
            "id": row[0],
            "key": row[1],
            "value": row[2],
            "category": row[3],
            "importance": row[4],
            "embedding": row[5],
            "created_at": row[6],
            "last_accessed": row[7],
            "access_count": row[8],
            "metadata": json.loads(row[9]),
        }

    @property
    def current_session(self) -> str:
        return self._current_session

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def shutdown(self) -> None:
        """Close database connection."""
        if self._conn:
            await self.vacuum()
            self._conn.close()
            self._conn = None
        self._initialized = False
