"""Planning agent: interactive project design before coding begins."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
)

from python_agent.rules import frame_data, planning_system_prompt


def print_text_blocks(message: Any) -> None:
    """Print TextBlock content from an AssistantMessage."""
    for block in message.content:
        if isinstance(block, TextBlock):
            print(block.text)


async def print_response(client: Any) -> None:
    """Receive and print the agent's response."""
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            print_text_blocks(message)


def read_user_input() -> str | None:
    """Read a line from the user. Return None to quit."""
    try:
        user_input = input("\n> ")
    except (EOFError, KeyboardInterrupt):
        print("\nDone.")
        return None
    if user_input.strip().lower() in ("quit", "exit", "done"):
        return None
    return user_input


# taint: ignore[CWE-200] -- Interactive agent displays LLM output to user
async def run(initial_prompt: str, model: str) -> None:
    """Run the planning agent interactively."""
    prompt = planning_system_prompt()

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=prompt,
        allowed_tools=["Read", "Glob", "Grep"],
        permission_mode="default",
    )

    async with ClaudeSDKClient(options=options) as client:
        framed = frame_data("user-input", initial_prompt)
        await client.query(framed)
        await print_response(client)

        while True:
            user_input = read_user_input()
            if user_input is None:
                break
            framed = frame_data("user-input", user_input)
            await client.query(framed)
            await print_response(client)


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive project planning agent",
    )
    parser.add_argument(
        "description",
        help="Initial project description to start planning",
    )
    parser.add_argument(
        "-m", "--model",
        default="claude-opus-4-6",
        help="Model to use (default: claude-opus-4-6)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the planning-agent CLI."""
    args = parse_args(argv)
    asyncio.run(run(args.description, args.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
