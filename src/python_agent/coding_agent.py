"""Coding agent: writes Python code to production standards."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import AsyncIterator
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    query,
)

from python_agent.agent_utils import print_text_blocks
from python_agent.dag_utils import load_dag
from python_agent.rules import coding_system_prompt
from python_agent.tool_guard import make_tool_guard

ESCALATION_MODEL = "claude-opus-4-6"


async def _prompt_stream(
    text: str,
) -> AsyncIterator[dict[str, Any]]:
    """Wrap a string prompt as an async iterable.

    Required by claude_agent_sdk when can_use_tool is set.
    """
    yield {"role": "user", "content": text}


async def run_query(
    task: str, options: Any,
) -> Any:
    """Run a single query and return the ResultMessage."""
    result: Any = None
    async for message in query(
        prompt=_prompt_stream(task), options=options,
    ):
        if isinstance(message, AssistantMessage):
            print_text_blocks(message)
        elif isinstance(message, ResultMessage):
            cost = message.total_cost_usd or 0.0
            print(f"\nDone. Cost: ${cost:.4f}")
            result = message
    return result


def should_escalate(
    result: Any, max_turns: int | None,
) -> bool:
    """Decide whether to escalate to a stronger model."""
    if result is None:
        return False
    if result.is_error:
        return True
    if max_turns is not None and result.num_turns >= max_turns:
        return True
    return False


def remaining_budget(
    result: Any, max_budget: float | None,
) -> float | None:
    """Calculate remaining budget after a query."""
    if max_budget is None:
        return None
    if result is None or result.total_cost_usd is None:
        return max_budget
    return float(max_budget - result.total_cost_usd)


def _load_ontology_json(
    dag_file: str | None,
) -> str | None:
    """Load ontology JSON from a DAG file, if provided."""
    if dag_file is None:
        return None
    dag = load_dag(dag_file, project_name="")
    node = dag.get_current_node()
    if node is None:
        return None
    return node.ontology.model_dump_json(indent=2)


async def run(
    task: str, project_dir: str, model: str,
    max_turns: int | None, max_budget: float | None,
    dag_file: str | None = None,
) -> None:
    """Run the coding agent on a task, escalating to Opus if stuck."""
    ontology_json = _load_ontology_json(dag_file)
    prompt = coding_system_prompt(
        project_dir, ontology_json=ontology_json,
    )
    guard = make_tool_guard(project_dir)
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=prompt,
        allowed_tools=["Read", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        max_budget_usd=max_budget,
        cwd=project_dir,
        can_use_tool=guard,
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
        can_use_tool=guard,
    )
    await run_query(escalation_task, escalation_options)


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
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
    parser.add_argument(
        "--dag-file",
        default=None,
        help="Path to ontology DAG JSON file for design context",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the coding-agent CLI."""
    args = parse_args(argv)
    asyncio.run(
        run(
            args.task,
            args.project_dir,
            args.model,
            args.max_turns,
            args.max_budget,
            dag_file=args.dag_file,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
