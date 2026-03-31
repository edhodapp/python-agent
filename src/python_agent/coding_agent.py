"""Coding agent: writes Python code to production standards."""

import argparse
import asyncio
import sys

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

from python_agent.rules import coding_system_prompt

ESCALATION_MODEL = "claude-opus-4-6"


def print_text_blocks(message):
    """Print TextBlock content from an AssistantMessage."""
    for block in message.content:
        if isinstance(block, TextBlock):
            print(block.text)


async def run_query(task, options):
    """Run a single query and return the ResultMessage."""
    result = None
    async for message in query(prompt=task, options=options):
        if isinstance(message, AssistantMessage):
            print_text_blocks(message)
        elif isinstance(message, ResultMessage):
            print(f"\nDone. Cost: ${message.total_cost_usd:.4f}")
            result = message
    return result


def should_escalate(result, max_turns):
    """Decide whether to escalate to a stronger model."""
    if result is None:
        return False
    if result.is_error:
        return True
    if max_turns is not None and result.num_turns >= max_turns:
        return True
    return False


def remaining_budget(result, max_budget):
    """Calculate remaining budget after a query."""
    if max_budget is None:
        return None
    if result is None or result.total_cost_usd is None:
        return max_budget
    return max_budget - result.total_cost_usd


async def run(task, project_dir, model, max_turns, max_budget):
    """Run the coding agent on a task, escalating to Opus if stuck."""
    prompt = coding_system_prompt(project_dir)
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=prompt,
        allowed_tools=["Read", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        max_budget_usd=max_budget,
        cwd=project_dir,
    )
    result = await run_query(task, options)
    if model == ESCALATION_MODEL:
        return
    if not should_escalate(result, max_turns):
        return
    print("\nEscalating to Opus...")
    budget = remaining_budget(result, max_budget)
    escalation_task = (
        "Continue this task. Partial changes may already "
        f"exist in the working directory.\n\n{task}"
    )
    escalation_options = ClaudeAgentOptions(
        model=ESCALATION_MODEL,
        system_prompt=prompt,
        allowed_tools=["Read", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        max_budget_usd=budget,
        cwd=project_dir,
    )
    await run_query(escalation_task, escalation_options)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Python coding agent on a task",
    )
    parser.add_argument(
        "task",
        help="Task description for the agent",
    )
    parser.add_argument(
        "-d", "--project-dir",
        default=".",
        help="Project directory to work in (default: current dir)",
    )
    parser.add_argument(
        "-m", "--model",
        default="claude-sonnet-4-6",
        help="Model to use (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        help="Maximum agent turns (default: 30)",
    )
    parser.add_argument(
        "--max-budget",
        type=float,
        default=5.0,
        help="Maximum budget in USD (default: 5.0)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Entry point for the coding-agent CLI."""
    args = parse_args(argv)
    asyncio.run(
        run(
            args.task,
            args.project_dir,
            args.model,
            args.max_turns,
            args.max_budget,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
