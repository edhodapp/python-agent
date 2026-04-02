"""Tool guard: intercepts coding agent tool calls for safety."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from os.path import abspath, commonpath
from typing import Any

from claude_agent_sdk import (
    PermissionResultAllow,
    PermissionResultDeny,
)

BLOCKED_BASH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p) for p in [
        r"\brm\s+-rf\s+/",
        r"\bcurl\b",
        r"\bwget\b",
        r"\bssh\b",
        r"\bscp\b",
        r"\bnc\b",
        r"\bncat\b",
        r"\bsudo\b",
        r"\bchmod\s+777\b",
        r"\bchown\b",
        r"\bmkfs\b",
        r"\bdd\s",
        r"(>|>>)\s*/etc/",
        r"(>|>>)\s*~/\.ssh/",
        r"(>|>>)\s*~/\.bashrc",
        r"\bkill\s+-9\b",
        r"\bpkill\b",
    ]
]

_PATH_TOOLS = frozenset({"Read", "Edit", "Glob", "Grep"})


def is_path_within(
    path: str, project_dir: str,
) -> bool:
    """Check if path resolves within project_dir."""
    resolved = abspath(path)
    proj = abspath(project_dir)
    try:
        common = commonpath([resolved, proj])
    except ValueError:
        return False
    return common == proj


def is_safe_bash(
    command: str, project_dir: str,
) -> tuple[bool, str]:
    """Check if a Bash command is safe to run.

    Returns (safe, reason). reason is "" if safe.
    """
    for pattern in BLOCKED_BASH_PATTERNS:
        if pattern.search(command):
            return (
                False,
                f"Blocked: matches {pattern.pattern!r}",
            )
    return (True, "")


def is_safe_path(
    tool_name: str, tool_input: dict[str, Any],
    project_dir: str,
) -> tuple[bool, str]:
    """Check if a file tool targets a safe path.

    Returns (safe, reason). reason is "" if safe.
    """
    if tool_name not in _PATH_TOOLS:
        return (True, "")
    path = tool_input.get(
        "file_path", tool_input.get("path", ""),
    )
    if not path:
        return (True, "")
    if not is_path_within(path, project_dir):
        return (
            False,
            f"Path outside project: {path!r}",
        )
    return (True, "")


def _log_entry(
    tool: str, tool_input: dict[str, Any],
    allowed: bool, reason: str,
) -> str:
    """Format a JSON-lines audit log entry."""
    entry = {
        "timestamp": datetime.now(
            timezone.utc,
        ).isoformat(),
        "tool": tool,
        "input": tool_input,
        "allowed": allowed,
        "reason": reason,
    }
    return json.dumps(entry)


def _write_log(
    log_path: str, entry: str,
) -> None:
    """Append an entry to the audit log."""
    with open(log_path, "a") as f:
        f.write(entry + "\n")


def make_tool_guard(
    project_dir: str,
    log_path: str | None = None,
) -> Any:
    """Create a can_use_tool callback for the coding agent.

    Returns an async callback that checks Bash commands
    and file paths, optionally logging all tool calls.
    """

    async def guard(
        tool: str,
        tool_input: dict[str, Any],
        context: Any,
    ) -> Any:
        safe, reason = _check_tool(
            tool, tool_input, project_dir,
        )
        if log_path is not None:
            entry = _log_entry(
                tool, tool_input, safe, reason,
            )
            _write_log(log_path, entry)
        if safe:
            return PermissionResultAllow()
        return PermissionResultDeny(message=reason)

    return guard


def _check_tool(
    tool: str, tool_input: dict[str, Any],
    project_dir: str,
) -> tuple[bool, str]:
    """Check a tool call. Returns (safe, reason)."""
    if tool == "Bash":
        command = tool_input.get("command", "")
        return is_safe_bash(command, project_dir)
    return is_safe_path(
        tool, tool_input, project_dir,
    )
