"""
Zeta GUI — Chat-style interface.

A modern, conversational UI that feels like a premium AI chat app.
User messages appear on the right, AI responses on the left.
Execution events stream in as formatted AI messages in real time.
"""

import asyncio
import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import scrolledtext
from typing import Any, Dict, Optional

from zeta_cli.agents.manager import AgentManager
from zeta_cli.config.manager import ConfigManager
from zeta_cli.core.engine import ExecutionEngine
from zeta_cli.memory.manager import MemoryManager
from zeta_cli.planner.planner import TaskPlanner
from zeta_cli.skills.manager import SkillManager
from zeta_cli.tools.registry import ToolRegistry

# ── Palette ────────────────────────────────────────────────────────────────────
BG          = "#0d0d0d"
SIDEBAR_BG  = "#111111"
CHAT_BG     = "#0d0d0d"
INPUT_BG    = "#1a1a1a"
INPUT_FG    = "#e8e8e8"
BORDER      = "#2a2a2a"
ACCENT      = "#2563eb"       # blue – send button / user bubble
ACCENT_HOVER= "#1d4ed8"
USER_BG     = "#1e3a5f"
USER_FG     = "#e0f0ff"
AI_BG       = "#1a1a1a"
AI_FG       = "#e8e8e8"
SYSTEM_FG   = "#6b7280"
SUCCESS_FG  = "#22c55e"
ERROR_FG    = "#ef4444"
WARN_FG     = "#f59e0b"
TITLE_FG    = "#ffffff"
SUBTITLE_FG = "#9ca3af"
DIVIDER     = "#1f1f1f"


class RoundedButton(tk.Canvas):
    """A canvas-based button with rounded corners."""

    def __init__(self, parent, text, command, bg=ACCENT, fg="#ffffff",
                 hover_bg=ACCENT_HOVER, radius=8, padx=20, pady=8,
                 font_spec=("Segoe UI", 10, "bold"), **kwargs):
        super().__init__(parent, bg=parent["bg"], highlightthickness=0, cursor="hand2", **kwargs)
        self._command = command
        self._bg = bg
        self._hover_bg = hover_bg
        self._fg = fg
        self._radius = radius
        self._text = text
        self._font = tkfont.Font(family=font_spec[0], size=font_spec[1],
                                  weight=font_spec[2] if len(font_spec) > 2 else "normal")

        tw = self._font.measure(text) + padx * 2
        th = self._font.metrics("linespace") + pady * 2
        self.config(width=tw, height=th)

        self._draw(bg)
        self.bind("<Enter>", lambda e: self._draw(hover_bg))
        self.bind("<Leave>", lambda e: self._draw(bg))
        self.bind("<Button-1>", lambda e: command())

    def _draw(self, color):
        self.delete("all")
        w, h, r = int(self["width"]), int(self["height"]), self._radius
        self.create_arc(0, 0, r*2, r*2, start=90, extent=90, fill=color, outline=color)
        self.create_arc(w-r*2, 0, w, r*2, start=0, extent=90, fill=color, outline=color)
        self.create_arc(0, h-r*2, r*2, h, start=180, extent=90, fill=color, outline=color)
        self.create_arc(w-r*2, h-r*2, w, h, start=270, extent=90, fill=color, outline=color)
        self.create_rectangle(r, 0, w-r, h, fill=color, outline=color)
        self.create_rectangle(0, r, w, h-r, fill=color, outline=color)
        self.create_text(w//2, h//2, text=self._text, fill=self._fg, font=self._font)


class ChatWidget(tk.Frame):
    """
    Scrollable chat area that renders user and AI message bubbles.
    Uses a Text widget with custom tags for rich formatting.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=CHAT_BG, **kwargs)

        self._text = tk.Text(
            self,
            bg=CHAT_BG,
            fg=AI_FG,
            font=("Segoe UI", 11),
            wrap=tk.WORD,
            state="disabled",
            cursor="arrow",
            relief="flat",
            borderwidth=0,
            padx=20,
            pady=10,
            spacing3=2,
        )
        scroll = tk.Scrollbar(self, orient="vertical", command=self._text.yview,
                              bg=SIDEBAR_BG, troughcolor=BG, activebackground=BORDER,
                              width=8, relief="flat", bd=0)
        self._text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._text.pack(side="left", fill="both", expand=True)

        # Configure tags
        self._text.tag_configure("user_name",
            font=("Segoe UI", 9, "bold"), foreground=ACCENT, spacing1=16)
        self._text.tag_configure("user_msg",
            font=("Segoe UI", 11), foreground=USER_FG,
            lmargin1=40, lmargin2=40, rmargin=10, spacing3=4)
        self._text.tag_configure("ai_name",
            font=("Segoe UI", 9, "bold"), foreground="#60a5fa", spacing1=16)
        self._text.tag_configure("ai_msg",
            font=("Segoe UI", 11), foreground=AI_FG,
            lmargin1=40, lmargin2=40, rmargin=10, spacing3=4)
        self._text.tag_configure("system_msg",
            font=("Segoe UI", 10, "italic"), foreground=SYSTEM_FG,
            lmargin1=20, lmargin2=20, spacing1=4, spacing3=4)
        self._text.tag_configure("success",
            font=("Segoe UI", 10), foreground=SUCCESS_FG,
            lmargin1=40, lmargin2=40, spacing3=2)
        self._text.tag_configure("error",
            font=("Segoe UI", 10), foreground=ERROR_FG,
            lmargin1=40, lmargin2=40, spacing3=2)
        self._text.tag_configure("warning",
            font=("Segoe UI", 10), foreground=WARN_FG,
            lmargin1=40, lmargin2=40, spacing3=2)
        self._text.tag_configure("task_item",
            font=("Consolas", 10), foreground="#94a3b8",
            lmargin1=56, lmargin2=56, spacing3=1)
        self._text.tag_configure("divider",
            font=("Segoe UI", 8), foreground=DIVIDER, spacing1=8, spacing3=8)
        self._text.tag_configure("code",
            font=("Consolas", 10), foreground="#e2e8f0", background="#1e293b",
            lmargin1=40, lmargin2=40, spacing1=2, spacing3=2)
        self._text.tag_configure("timestamp",
            font=("Segoe UI", 8), foreground="#374151")

    def _insert(self, text, *tags):
        self._text.configure(state="normal")
        self._text.insert(tk.END, text, tags)
        self._text.configure(state="disabled")
        self._text.see(tk.END)

    def add_user_message(self, text: str):
        self._insert("\n  You\n", "user_name")
        self._insert(f"  {text}\n", "user_msg")

    def add_ai_message(self, text: str, sender: str = "Zeta"):
        self._insert(f"\n  {sender}\n", "ai_name")
        for line in text.split("\n"):
            self._insert(f"  {line}\n", "ai_msg")

    def add_system(self, text: str):
        self._insert(f"  {text}\n", "system_msg")

    def add_success(self, text: str):
        self._insert(f"  ✓  {text}\n", "success")

    def add_error(self, text: str):
        self._insert(f"  ✗  {text}\n", "error")

    def add_warning(self, text: str):
        self._insert(f"  ⚠  {text}\n", "warning")

    def add_task(self, text: str):
        self._insert(f"      {text}\n", "task_item")

    def add_code(self, text: str):
        self._insert(f"  {text}\n", "code")

    def add_divider(self):
        self._insert("  " + "─" * 60 + "\n", "divider")

    def clear(self):
        self._text.configure(state="normal")
        self._text.delete("1.0", tk.END)
        self._text.configure(state="disabled")


class ZetaGUIApp:
    """
    Chat-style Tkinter GUI for Zeta CLI.

    Drop-in replacement for the original GUI — same public interface:
        run()
        set_initial_goal(goal, auto_launch)
    """

    def __init__(self, systems: Dict[str, Any], shutdown_event: threading.Event):
        self._config: ConfigManager        = systems["config"]
        self._security                     = systems["security"]
        self._identity                     = systems.get("identity")
        self._memory: MemoryManager        = systems["memory"]
        self._api                          = systems["api"]
        self._tools: ToolRegistry          = systems["tools"]
        self._agents: AgentManager         = systems["agents"]
        self._skills: SkillManager         = systems.get("skills")
        self._planner: TaskPlanner         = systems["planner"]
        self._engine: ExecutionEngine      = systems["engine"]
        self._shutdown_event               = shutdown_event

        self._event_queue: "queue.Queue[tuple[str, dict]]" = queue.Queue()
        self._engine_thread: Optional[threading.Thread] = None
        self._initial_goal: str = ""
        self._auto_launch: bool = False
        self._busy: bool = False

        # Build window
        self.root = tk.Tk()
        self.root.title("Zeta AI")
        self.root.geometry("1080x720")
        self.root.minsize(800, 500)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        try:
            self.root.tk.call("tk", "scaling", 1.25)
        except Exception:
            pass

        self._build_ui()
        self.root.after(100, self._process_event_queue)
        self.root.after(200, self._show_welcome)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.root.mainloop()

    def set_initial_goal(self, goal: str, auto_launch: bool = True) -> None:
        self._initial_goal = goal
        self._auto_launch = auto_launch

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Top bar ──────────────────────────────────────────────────────────
        topbar = tk.Frame(self.root, bg="#111111", height=52)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        avatar = tk.Label(topbar, text="Z", font=("Segoe UI", 14, "bold"),
                          fg="#ffffff", bg=ACCENT, width=2, height=1)
        avatar.pack(side="left", padx=(16, 10), pady=10)

        name_frame = tk.Frame(topbar, bg="#111111")
        name_frame.pack(side="left", pady=10)
        tk.Label(name_frame, text="Zeta AI", font=("Segoe UI", 11, "bold"),
                 fg=TITLE_FG, bg="#111111").pack(anchor="w")
        self._status_label = tk.Label(name_frame, text="● Online",
                                      font=("Segoe UI", 9), fg=SUCCESS_FG, bg="#111111")
        self._status_label.pack(anchor="w")

        # Right side buttons
        btn_frame = tk.Frame(topbar, bg="#111111")
        btn_frame.pack(side="right", padx=16)
        self._make_topbar_btn(btn_frame, "⚙ Settings", self._show_settings).pack(side="right", padx=4)
        self._make_topbar_btn(btn_frame, "🗑 Clear", self._clear_chat).pack(side="right", padx=4)

        # ── Separator ────────────────────────────────────────────────────────
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        # ── Main content ─────────────────────────────────────────────────────
        content = tk.Frame(self.root, bg=BG)
        content.pack(fill="both", expand=True)

        # Sidebar
        sidebar = tk.Frame(content, bg=SIDEBAR_BG, width=200)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        tk.Frame(content, bg=BORDER, width=1).pack(side="left", fill="y")

        # Chat column
        chat_col = tk.Frame(content, bg=BG)
        chat_col.pack(side="left", fill="both", expand=True)

        self._chat = ChatWidget(chat_col)
        self._chat.pack(fill="both", expand=True)

        # ── Input bar ────────────────────────────────────────────────────────
        tk.Frame(chat_col, bg=BORDER, height=1).pack(fill="x")
        input_bar = tk.Frame(chat_col, bg=INPUT_BG, pady=12, padx=16)
        input_bar.pack(fill="x", side="bottom")

        self._input = tk.Text(
            input_bar,
            font=("Segoe UI", 11),
            bg="#252525",
            fg=INPUT_FG,
            insertbackground=INPUT_FG,
            relief="flat",
            borderwidth=0,
            height=3,
            wrap=tk.WORD,
            padx=14,
            pady=10,
        )
        self._input.pack(side="left", fill="both", expand=True)
        self._input.bind("<Return>", self._on_return)
        self._input.bind("<Shift-Return>", lambda e: None)  # allow newline with Shift+Enter
        self._input.insert("1.0", "")

        # Hint text
        self._hint_visible = True
        self._hint_text = "Type a goal or command and press Enter…"
        self._input.insert("1.0", self._hint_text)
        self._input.configure(fg="#4b5563")
        self._input.bind("<FocusIn>", self._clear_hint)
        self._input.bind("<FocusOut>", self._restore_hint)

        btn_col = tk.Frame(input_bar, bg=INPUT_BG)
        btn_col.pack(side="right", padx=(12, 0), fill="y")

        self._send_btn = RoundedButton(
            btn_col, text="Send  ↵", command=self._on_send,
            bg=ACCENT, hover_bg=ACCENT_HOVER, padx=18, pady=10,
            font_spec=("Segoe UI", 10, "bold"),
        )
        self._send_btn.pack(side="bottom", pady=(0, 2))

        hint = tk.Label(btn_col, text="Shift+Enter for newline",
                        font=("Segoe UI", 8), fg=SYSTEM_FG, bg=INPUT_BG)
        hint.pack(side="top")

        # Round the input frame corners visually with a border
        self._input.configure(highlightthickness=1, highlightcolor=ACCENT,
                               highlightbackground=BORDER)

    def _make_topbar_btn(self, parent, text, cmd):
        b = tk.Label(parent, text=text, font=("Segoe UI", 9),
                     fg=SUBTITLE_FG, bg="#111111", cursor="hand2", padx=8, pady=4)
        b.bind("<Button-1>", lambda e: cmd())
        b.bind("<Enter>", lambda e: b.configure(fg=TITLE_FG))
        b.bind("<Leave>", lambda e: b.configure(fg=SUBTITLE_FG))
        return b

    def _build_sidebar(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="QUICK ACTIONS", font=("Segoe UI", 8, "bold"),
                 fg="#4b5563", bg=SIDEBAR_BG, anchor="w").pack(
                     fill="x", padx=16, pady=(20, 8))

        actions = [
            ("📋  New Goal",        self._quick_new_goal),
            ("📊  Status",          self._quick_status),
            ("📝  Task List",       self._quick_task_list),
            ("🤖  Agents",          self._quick_agents),
            ("🔧  Tools",           self._quick_tools),
            ("🧠  Memory",          self._quick_memory),
            ("📚  Skills",          self._quick_skills),
            ("❓  Help",            self._quick_help),
        ]
        for label, cmd in actions:
            self._make_sidebar_btn(parent, label, cmd)

        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=12, pady=16)

        tk.Label(parent, text="HISTORY", font=("Segoe UI", 8, "bold"),
                 fg="#4b5563", bg=SIDEBAR_BG, anchor="w").pack(
                     fill="x", padx=16, pady=(0, 8))
        self._history_frame = tk.Frame(parent, bg=SIDEBAR_BG)
        self._history_frame.pack(fill="x")

    def _make_sidebar_btn(self, parent, text, cmd):
        f = tk.Frame(parent, bg=SIDEBAR_BG, cursor="hand2")
        f.pack(fill="x", padx=8, pady=1)

        lbl = tk.Label(f, text=text, font=("Segoe UI", 10), fg=SUBTITLE_FG,
                       bg=SIDEBAR_BG, anchor="w", padx=12, pady=7)
        lbl.pack(fill="x")

        def on_enter(e):
            f.configure(bg="#1f1f1f")
            lbl.configure(bg="#1f1f1f", fg=TITLE_FG)
        def on_leave(e):
            f.configure(bg=SIDEBAR_BG)
            lbl.configure(bg=SIDEBAR_BG, fg=SUBTITLE_FG)
        def on_click(e):
            cmd()

        for widget in (f, lbl):
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
            widget.bind("<Button-1>", on_click)

    # ── Input handling ─────────────────────────────────────────────────────────

    def _clear_hint(self, event=None):
        if self._hint_visible:
            self._input.delete("1.0", tk.END)
            self._input.configure(fg=INPUT_FG)
            self._hint_visible = False

    def _restore_hint(self, event=None):
        if not self._input.get("1.0", tk.END).strip():
            self._input.delete("1.0", tk.END)
            self._input.insert("1.0", self._hint_text)
            self._input.configure(fg="#4b5563")
            self._hint_visible = True

    def _on_return(self, event):
        if event.state & 0x1:   # Shift held
            return None          # allow newline
        self._on_send()
        return "break"

    def _on_send(self):
        if self._hint_visible:
            return
        text = self._input.get("1.0", tk.END).strip()
        if not text:
            return
        self._input.delete("1.0", tk.END)
        self._restore_hint()
        self._process_input(text)

    def _process_input(self, text: str):
        self._chat.add_user_message(text)
        low = text.lower().strip()

        # Special commands
        if low in ("clear", "/clear"):
            self._clear_chat(); return
        if low in ("help", "/help"):
            self._quick_help(); return
        if low in ("status", "/status"):
            self._quick_status(); return
        if low in ("agents", "/agents"):
            self._quick_agents(); return
        if low in ("tools", "/tools"):
            self._quick_tools(); return
        if low in ("skills", "/skills"):
            self._quick_skills(); return
        if low in ("tasks", "/tasks", "task list"):
            self._quick_task_list(); return
        if low in ("memory", "/memory"):
            self._quick_memory(); return
        if low.startswith("memory search "):
            query = text[len("memory search "):].strip()
            self._do_memory_search(query); return
        if low in ("exit", "quit", "/exit", "/quit"):
            self._on_close(); return

        # Everything else is treated as a goal
        if self._busy:
            self._chat.add_warning("Still working on previous goal. Please wait…")
            return
        self._begin_goal_execution(text)

    # ── Goal execution ─────────────────────────────────────────────────────────

    def _begin_goal_execution(self, goal: str):
        self._busy = True
        self._set_status("Working…", WARN_FG)
        self._chat.add_ai_message(f"On it! Starting: **{goal}**\n\nPlanning tasks…")
        self._engine_thread = threading.Thread(
            target=self._execute_goal_thread, args=(goal,), daemon=True)
        self._engine_thread.start()

    def _execute_goal_thread(self, goal: str):
        try:
            self._engine.on_event(self._engine_event)
            result = self._run_async(self._engine.execute_goal(goal))
            summary = getattr(result, "summary", "Execution complete.") if result else "Execution complete."
            failed  = getattr(result, "failed_count", 0) if result else 0
            total   = getattr(result, "task_count", 0) if result else 0
            dur     = getattr(result, "duration", 0) if result else 0
            if failed == 0:
                self._event_queue.put(("chat_success", {
                    "message": f"Goal completed successfully.\n{total} tasks done in {dur:.1f}s."
                }))
            else:
                self._event_queue.put(("chat_warn", {
                    "message": f"Goal finished with {failed} failed task(s) out of {total}.\n{summary}"
                }))
        except Exception as exc:
            self._event_queue.put(("chat_error", {"message": f"Execution error: {exc}"}))
        finally:
            self._event_queue.put(("done", {}))

    def _engine_event(self, event_type: str, data: Dict[str, Any]):
        self._event_queue.put((event_type, data))

    def _run_async(self, coro):
        try:
            return asyncio.run(coro)
        except Exception as exc:
            self._event_queue.put(("chat_error", {"message": str(exc)}))
            return None

    # ── Event queue processing ─────────────────────────────────────────────────

    def _process_event_queue(self):
        try:
            while True:
                event_type, data = self._event_queue.get_nowait()
                self._handle_event(event_type, data)
        except queue.Empty:
            pass
        self.root.after(80, self._process_event_queue)

    def _handle_event(self, event_type: str, data: Dict[str, Any]):
        if event_type == "planning_started":
            self._chat.add_system("  Analysing goal and building plan…")
        elif event_type == "planning_complete":
            n = data.get("tasks", 0)
            self._chat.add_system(f"  Plan ready — {n} task{'s' if n != 1 else ''} queued.")
        elif event_type == "task_started":
            name = data.get("task", "—")
            self._chat.add_task(f"▶  {name}")
        elif event_type == "task_completed":
            name    = data.get("task", "—")
            success = data.get("success", False)
            dur     = data.get("duration", 0)
            if success:
                self._chat.add_task(f"✓  {name}  ({dur:.1f}s)")
            else:
                self._chat.add_task(f"✗  {name}  ({dur:.1f}s) — failed")
        elif event_type == "goal_completed":
            pass  # handled by chat_success
        elif event_type == "chat_success":
            self._chat.add_ai_message(data.get("message", "Done."))
        elif event_type == "chat_warn":
            self._chat.add_ai_message(data.get("message", "Completed with warnings."))
        elif event_type == "chat_error":
            self._chat.add_error(data.get("message", "An error occurred."))
        elif event_type == "log":
            msg = data.get("message", "")
            if msg:
                self._chat.add_system(f"  {msg}")
        elif event_type == "done":
            self._busy = False
            self._set_status("● Online", SUCCESS_FG)

    # ── Status bar ─────────────────────────────────────────────────────────────

    def _set_status(self, text: str, color: str = SUCCESS_FG):
        if self._status_label:
            self._status_label.configure(text=text, fg=color)

    # ── Welcome message ────────────────────────────────────────────────────────

    def _show_welcome(self):
        identity = self._identity.get_all() if self._identity else {}
        name     = identity.get("name", "Alpha")
        agents   = len(self._agents.list_agents())
        tools    = len(self._tools.list_tools())
        skills   = len(self._skills.list_skills()) if self._skills else 0

        welcome = (
            f"Hey {name}! I'm ready to work.\n\n"
            f"I have {agents} agents, {tools} tools, and {skills} skills standing by.\n\n"
            "Just tell me what you want to build, fix, or explore — "
            "I'll plan it out and execute it step by step.\n\n"
            "Try something like:\n"
            "  • write a hello world script in Python\n"
            "  • build a REST API with FastAPI\n"
            "  • audit this codebase for security issues\n\n"
            "Or type **help** to see what I can do."
        )
        self._chat.add_ai_message(welcome)

        if self._auto_launch and self._initial_goal:
            self.root.after(800, lambda: self._begin_goal_execution(self._initial_goal))

    # ── Sidebar quick actions ──────────────────────────────────────────────────

    def _quick_new_goal(self):
        self._clear_hint(None)
        self._input.focus_set()

    def _quick_status(self):
        self._chat.add_user_message("status")
        prog = self._engine.get_progress()
        state = prog.get("state", "idle")
        pct   = prog.get("progress", 0)
        goal  = prog.get("goal", "None")
        total = prog.get("total_tasks", 0)
        done  = prog.get("completed", 0)
        fail  = prog.get("failed", 0)

        msg = (
            f"**Current status**\n\n"
            f"State: {state}\n"
            f"Progress: {pct}%\n"
            f"Goal: {goal}\n"
            f"Tasks: {done}/{total} completed, {fail} failed"
        )
        self._chat.add_ai_message(msg)

    def _quick_task_list(self):
        self._chat.add_user_message("tasks")
        goals = self._planner.list_active_goals()
        if not goals:
            self._chat.add_ai_message("No active goals right now. Give me something to work on!")
            return
        lines = []
        for g in goals:
            lines.append(f"**{g.description}** [{g.status.value}]")
            for t in g.tasks:
                icon = "✓" if t.status.value == "completed" else ("▶" if t.status.value == "in_progress" else "○")
                lines.append(f"  {icon} P{t.priority} {t.title}")
        self._chat.add_ai_message("\n".join(lines))

    def _quick_agents(self):
        self._chat.add_user_message("agents")
        agents = self._agents.list_agents()
        if not agents:
            self._chat.add_ai_message("No agents registered.")
            return
        lines = ["**Available agents**\n"]
        for a in agents:
            desc = getattr(a, "description", "—")
            lines.append(f"**{a.name}** — {desc}")
        self._chat.add_ai_message("\n".join(lines))

    def _quick_tools(self):
        self._chat.add_user_message("tools")
        tools = self._tools.list_tools()
        if not tools:
            self._chat.add_ai_message("No tools available.")
            return
        by_cat: Dict[str, list] = {}
        for t in tools:
            by_cat.setdefault(t.category, []).append(t)
        lines = ["**Available tools**\n"]
        for cat, ts in by_cat.items():
            lines.append(f"**{cat.upper()}**")
            for t in ts:
                lines.append(f"  • {t.name} — {t.description[:70]}")
        self._chat.add_ai_message("\n".join(lines))

    def _quick_memory(self):
        self._chat.add_user_message("memory")
        stats = self._run_async(self._memory.get_stats())
        if not stats:
            self._chat.add_ai_message("Couldn't retrieve memory stats.")
            return
        lines = ["**Memory stats**\n"]
        for k, v in stats.items():
            lines.append(f"{k.replace('_', ' ').title()}: **{v}**")
        self._chat.add_ai_message("\n".join(lines))

    def _quick_skills(self):
        self._chat.add_user_message("skills")
        if not self._skills:
            self._chat.add_ai_message("Skill system not initialized.")
            return
        skills = self._skills.list_skills()
        caps   = self._skills.list_capabilities()
        lines  = [f"**{len(skills)} skills, {len(caps)} capabilities**\n"]
        for s in skills:
            lines.append(f"  • **{s.name}** — {s.description}")
        self._chat.add_ai_message("\n".join(lines) if lines else "No skills loaded.")

    def _quick_help(self):
        self._chat.add_user_message("help")
        help_msg = (
            "**What I understand**\n\n"
            "Just talk to me naturally — describe what you want and I'll figure it out.\n\n"
            "**Quick commands:**\n"
            "  • `status` — see what I'm doing right now\n"
            "  • `tasks` — list current plan tasks\n"
            "  • `agents` — list available agents\n"
            "  • `tools` — list available tools\n"
            "  • `skills` — list loaded skills\n"
            "  • `memory` — show memory stats\n"
            "  • `memory search <query>` — search memory\n"
            "  • `clear` — clear this chat\n"
            "  • `exit` — close Zeta\n\n"
            "**Goal examples:**\n"
            "  • write a hello world in Python\n"
            "  • build a FastAPI REST endpoint\n"
            "  • find all TODO comments in this workspace\n"
            "  • create a unit test suite for my project"
        )
        self._chat.add_ai_message(help_msg)

    def _do_memory_search(self, query: str):
        results = self._run_async(self._memory.search_long_term(query, limit=8))
        if not results:
            self._chat.add_ai_message(f"No memories found matching \"{query}\".")
            return
        lines = [f"**Memory results for \"{query}\"**\n"]
        for r in results:
            lines.append(f"**{r['key']}** ({r['category']})")
            lines.append(f"  {r['value'][:200]}")
        self._chat.add_ai_message("\n".join(lines))

    def _show_settings(self):
        cfg = self._config.get_all()
        lines = ["**Current configuration**\n"]
        for k, v in cfg.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    lines.append(f"  {k}.{kk}: {vv}")
            else:
                lines.append(f"  {k}: {v}")
        self._chat.add_ai_message("\n".join(lines))

    def _clear_chat(self):
        self._chat.clear()
        self._chat.add_system("  Chat cleared.")

    # ── Window close ───────────────────────────────────────────────────────────

    def _on_close(self):
        self._shutdown_event.set()
        try:
            self.root.destroy()
        except Exception:
            pass
