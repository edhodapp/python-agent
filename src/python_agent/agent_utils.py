"""Shared helper functions used by multiple agents."""

from __future__ import annotations

import json
import re
from typing import Any

from claude_agent_sdk import TextBlock

_ONTOLOGY_BLOCK_RE = re.compile(
    r"```ontology\s*\n(.*?)\n```",
    re.DOTALL,
)


def print_text_blocks(message: Any) -> None:
    """Print TextBlock content from an AssistantMessage."""
    for block in message.content:
        if isinstance(block, TextBlock):
            print(block.text)


def collect_response_text(message: Any) -> str:
    """Extract concatenated text from an AssistantMessage."""
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
    return "\n".join(parts)


def read_user_input() -> str | None:
    """Read a line from the user. Return None to quit."""
    try:
        user_input = input("\n> ")
    except (EOFError, KeyboardInterrupt):
        print("\nDone.")
        return None
    if user_input.strip().lower() in (
        "quit", "exit", "done",
    ):
        return None
    return user_input


def extract_ontology_json(
    text: str,
) -> dict[str, Any] | None:
    """Extract the first ontology JSON block from text.

    Returns the parsed dict, or None if no valid block found.
    """
    match = _ONTOLOGY_BLOCK_RE.search(text)
    if match is None:
        return None
    try:
        result: dict[str, Any] = json.loads(match.group(1))
        return result
    except json.JSONDecodeError:
        return None
