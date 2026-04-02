"""Convergence agent: interactive candidate selection."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from pydantic import BaseModel

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
)

from python_agent.dag_utils import (
    load_dag,
    save_dag,
    save_snapshot,
)
from python_agent.discovery_agent import (
    backtrack,
    format_ontology_summary,
    print_response,
    process_response,
    read_user_input,
)
from python_agent.ontology import (
    DAGNode,
    Decision,
    Ontology,
    OntologyDAG,
)
from python_agent.rules import convergence_system_prompt


class AgentState(BaseModel):
    """Mutable state for the convergence agent loop."""

    ontology: Ontology
    accepted: bool = False


def get_children_summaries(
    dag: OntologyDAG, node_id: str,
) -> list[tuple[int, DAGNode, str]]:
    """Build list of (index, node, summary) for children."""
    children = dag.children_of(node_id)
    result: list[tuple[int, DAGNode, str]] = []
    for i, child in enumerate(children, 1):
        summary = format_ontology_summary(child.ontology)
        result.append((i, child, summary))
    return result


def format_children_list(
    children_summaries: list[tuple[int, DAGNode, str]],
) -> str:
    """Format children summaries for display."""
    if not children_summaries:
        return "No candidates at this node."
    lines: list[str] = []
    for idx, child, summary in children_summaries:
        label = child.label or child.id
        first_line = summary.split("\n")[0]
        lines.append(f"  {idx}. {label}: {first_line}")
    return "\n".join(lines)


def navigate_to_node(
    dag: OntologyDAG, node_id: str,
) -> Ontology | None:
    """Set current_node_id and return the node's ontology."""
    node = dag.get_node(node_id)
    if node is None:
        return None
    dag.current_node_id = node_id
    return node.ontology.model_copy(deep=True)


def is_command(text: str) -> bool:
    """Check if text is a convergence meta-command."""
    cmd = text.strip().lower()
    return (
        cmd in ("list", "show", "accept", "back")
        or cmd.startswith("select ")
        or cmd.startswith("save")
    )


_CmdResult = tuple[str, Ontology | None, bool]


def _handle_list_cmd(
    command: str, ontology: Ontology,
    dag: OntologyDAG, dag_path: str,
) -> _CmdResult:
    """Handle 'list' command."""
    children = get_children_summaries(
        dag, dag.current_node_id,
    )
    return (format_children_list(children), None, False)


def _handle_show_cmd(
    command: str, ontology: Ontology,
    dag: OntologyDAG, dag_path: str,
) -> _CmdResult:
    """Handle 'show' command."""
    return (format_ontology_summary(ontology), None, False)


def _handle_accept_cmd(
    command: str, ontology: Ontology,
    dag: OntologyDAG, dag_path: str,
) -> _CmdResult:
    """Handle 'accept' command."""
    node = dag.get_current_node()
    label = node.label if node else "unknown"
    decision = Decision(
        question="candidate-selection",
        options=[],
        chosen=label,
        rationale="user-accepted",
    )
    save_snapshot(
        dag, ontology,
        f"accepted-{label}", decision,
    )
    save_dag(dag, dag_path)
    return (f"Accepted: {label}. You can now refine.",
            None, True)


def _handle_back_cmd(
    command: str, ontology: Ontology,
    dag: OntologyDAG, dag_path: str,
) -> _CmdResult:
    """Handle 'back' command."""
    node = backtrack(dag)
    if node is None:
        return ("Already at root.", None, False)
    new_onto = node.ontology.model_copy(deep=True)
    save_dag(dag, dag_path)
    label = node.label or node.id
    msg = f"Backtracked to: {node.id} ({label})"
    return (msg, new_onto, False)


def _handle_select_cmd(
    command: str, ontology: Ontology,
    dag: OntologyDAG, dag_path: str,
) -> _CmdResult:
    """Handle 'select <n>' command."""
    parts = command.strip().split()
    if len(parts) < 2:
        return ("Usage: select <number>", None, False)
    try:
        index = int(parts[1])
    except ValueError:
        return ("Invalid number.", None, False)
    children = dag.children_of(dag.current_node_id)
    if index < 1 or index > len(children):
        msg = f"Range: 1-{len(children)}"
        return (msg, None, False)
    child = children[index - 1]
    new_onto = navigate_to_node(dag, child.id)
    label = child.label or child.id
    return (f"Selected: {label}", new_onto, False)


def _handle_save_cmd(
    command: str, ontology: Ontology,
    dag: OntologyDAG, dag_path: str,
) -> _CmdResult:
    """Handle 'save [label]' command."""
    label = command.strip()[4:].strip() or "snapshot"
    save_snapshot(dag, ontology, label)
    save_dag(dag, dag_path)
    return (
        f"Saved snapshot: {dag.current_node_id}",
        None, False,
    )


_CmdHandler = Any  # Callable signature for command handlers

_DISPATCH: dict[str, _CmdHandler] = {
    "list": _handle_list_cmd,
    "show": _handle_show_cmd,
    "accept": _handle_accept_cmd,
    "back": _handle_back_cmd,
}


def handle_command(
    command: str, ontology: Ontology,
    dag: OntologyDAG, dag_path: str,
) -> _CmdResult:
    """Dispatch a convergence meta-command.

    Returns (message, new_ontology_or_None, is_accept).
    """
    cmd = command.strip().lower()
    handler = _DISPATCH.get(cmd)
    if handler is not None:
        result: _CmdResult = handler(
            command, ontology, dag, dag_path,
        )
        return result
    if cmd.startswith("select "):
        return _handle_select_cmd(
            command, ontology, dag, dag_path,
        )
    return _handle_save_cmd(
        command, ontology, dag, dag_path,
    )


def dispatch_command(
    command: str, state: AgentState,
    dag: OntologyDAG, dag_path: str,
) -> None:
    """Dispatch command and update agent state."""
    msg, new_onto, is_accept = handle_command(
        command, state.ontology, dag, dag_path,
    )
    print(msg)
    if new_onto is not None:
        state.ontology = new_onto
    if is_accept:
        state.accepted = True


def maybe_process(
    text: str, state: AgentState,
) -> None:
    """Apply ontology updates if accepted."""
    if state.accepted:
        process_response(text, state.ontology)


def build_query(
    user_input: str, state: AgentState,
    dag: OntologyDAG,
) -> str:
    """Build LLM query with current context."""
    from python_agent.rules import frame_data
    children = get_children_summaries(
        dag, dag.current_node_id,
    )
    ctx = format_children_list(children)
    node = dag.get_current_node()
    label = node.label if node else "unknown"
    context = f"node={label}, children:\n{ctx}"
    framed = frame_data("context-data", context)
    return f"[Context: {framed}]\n\n{user_input}"


def _init_state(
    dag: OntologyDAG,
) -> AgentState | None:
    """Initialize agent state from DAG."""
    node = dag.get_current_node()
    if node is None:
        return None
    ontology = node.ontology.model_copy(deep=True)
    return AgentState(ontology=ontology)


def _print_status(
    state: AgentState, dag: OntologyDAG,
) -> None:
    """Print initial status."""
    print(format_ontology_summary(state.ontology))
    children = get_children_summaries(
        dag, dag.current_node_id,
    )
    print(format_children_list(children))


async def _main_loop(
    client: Any, state: AgentState,
    dag: OntologyDAG, dag_path: str,
) -> None:
    """Run the interactive convergence loop."""
    while True:
        user_input = read_user_input()
        if user_input is None:
            break
        if is_command(user_input):
            dispatch_command(
                user_input, state, dag, dag_path,
            )
            continue
        q = build_query(user_input, state, dag)
        await client.query(q)
        text = await print_response(client)
        maybe_process(text, state)


async def run(dag_path: str, model: str) -> None:
    """Run the convergence agent interactively."""
    dag = load_dag(dag_path, "unknown")
    state = _init_state(dag)
    if state is None:
        print("Error: DAG has no current node.")
        return
    _print_status(state, dag)

    ontology_json = json.dumps(
        state.ontology.model_dump(), indent=2,
    )
    children = get_children_summaries(
        dag, dag.current_node_id,
    )
    prompt = convergence_system_prompt(
        ontology_json, format_children_list(children),
    )
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=prompt,
        allowed_tools=[],
        permission_mode="default",
    )

    async with ClaudeSDKClient(options=options) as client:
        await _main_loop(client, state, dag, dag_path)


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive candidate convergence",
    )
    parser.add_argument(
        "--dag-file",
        required=True,
        help="Path to DAG JSON file",
    )
    parser.add_argument(
        "-m", "--model",
        default="claude-opus-4-6",
        help="Model to use (default: claude-opus-4-6)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the convergence-agent CLI."""
    args = parse_args(argv)
    asyncio.run(run(args.dag_file, args.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
