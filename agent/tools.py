"""Tool definitions and executors for the agent harness.

Each tool is an Anthropic tool definition (JSON schema) plus a Python executor.
File/shell tools are confined to a workdir so the model can't read or write
outside the sandbox — the security boundary lives in the harness, not the model.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Tools that mutate state or run code. The harness gates these behind a
# confirmation prompt unless auto-approve is on. Read-only tools run freely.
MUTATING_TOOLS = {"write_file", "bash"}


@dataclass
class ToolError(Exception):
    """Raised by an executor to return an is_error tool_result to the model."""

    message: str


class Toolbox:
    """Holds the workdir sandbox and dispatches tool calls to executors."""

    def __init__(self, workdir: str | os.PathLike[str]):
        self.root = Path(workdir).resolve()
        if not self.root.is_dir():
            raise ValueError(f"workdir is not a directory: {self.root}")

    # --- path safety --------------------------------------------------------

    def _resolve(self, rel: str) -> Path:
        """Resolve a model-supplied path and confine it to the sandbox root.

        Rejects traversal (.., absolute escapes, symlinks out of root).
        """
        candidate = (self.root / rel).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ToolError(f"path escapes workdir: {rel}")
        return candidate

    # --- executors ----------------------------------------------------------

    def read_file(self, path: str, max_bytes: int = 100_000) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"not a file: {path}")
        data = p.read_text(errors="replace")
        if len(data) > max_bytes:
            return data[:max_bytes] + f"\n... [truncated at {max_bytes} bytes]"
        return data

    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} bytes to {path}"

    def list_dir(self, path: str = ".") -> str:
        p = self._resolve(path)
        if not p.is_dir():
            raise ToolError(f"not a directory: {path}")
        entries = []
        for child in sorted(p.iterdir()):
            kind = "dir " if child.is_dir() else "file"
            entries.append(f"{kind}  {child.relative_to(self.root)}")
        return "\n".join(entries) or "(empty)"

    def glob(self, pattern: str) -> str:
        matches = []
        for dirpath, _dirs, files in os.walk(self.root):
            for name in files:
                full = Path(dirpath) / name
                rel = full.relative_to(self.root).as_posix()
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
                    matches.append(rel)
        return "\n".join(sorted(matches)[:500]) or "(no matches)"

    def grep(self, pattern: str, glob_filter: str = "*") -> str:
        rx = re.compile(pattern)
        hits: list[str] = []
        for dirpath, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if d not in {".git", ".venv", "__pycache__"}]
            for name in files:
                if not fnmatch.fnmatch(name, glob_filter):
                    continue
                full = Path(dirpath) / name
                rel = full.relative_to(self.root).as_posix()
                try:
                    for i, line in enumerate(full.read_text(errors="replace").splitlines(), 1):
                        if rx.search(line):
                            hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                            if len(hits) >= 200:
                                return "\n".join(hits) + "\n... [truncated]"
                except OSError:
                    continue
        return "\n".join(hits) or "(no matches)"

    def bash(self, command: str, timeout: int = 60) -> str:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"command timed out after {timeout}s")
        out = (proc.stdout or "") + (proc.stderr or "")
        return f"[exit {proc.returncode}]\n{out[:50_000]}"

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        executor: Callable[..., str] | None = getattr(self, name, None)
        if executor is None or name not in {t["name"] for t in TOOL_DEFS}:
            raise ToolError(f"unknown tool: {name}")
        return executor(**args)


# Anthropic tool definitions. `finish` lets the model end the loop with a
# structured summary that the harness validates (see guardrails.py).
TOOL_DEFS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file relative to the workdir.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path relative to workdir"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a text file relative to the workdir.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_dir",
        "description": "List entries in a directory relative to the workdir.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "default": "."}},
        },
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern (e.g. '**/*.py' or '*.md').",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": "Search file contents with a regex; optional filename glob filter.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regex"},
                "glob_filter": {"type": "string", "default": "*"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "bash",
        "description": "Run a shell command in the workdir. Use for builds, tests, git, etc.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]
