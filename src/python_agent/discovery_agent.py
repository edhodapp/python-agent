"""Discovery agent: interactive ontology building."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
)

from python_agent.dag_utils import (
    load_dag,
    save_dag,
    save_snapshot,
)
from python_agent.ontology import (
    DAGNode,
    DomainConstraint,
    Entity,
    Ontology,
    OntologyDAG,
    OpenQuestion,
    Relationship,
)
from python_agent.rules import (
    discovery_system_prompt,
    frame_data,
)

_ONTOLOGY_BLOCK_RE = re.compile(
    r"```ontology\s*\n(.*?)\n```",
    re.DOTALL,
)


def collect_response_text(message: Any) -> str:
    """Extract concatenated text from an AssistantMessage."""
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
    return "\n".join(parts)


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


def _upsert_entities(
    ontology: Ontology, new_entities: list[Any],
) -> None:
    """Replace or append entities by id."""
    existing = {
        e.id: i for i, e in enumerate(ontology.entities)
    }
    for item in new_entities:
        entity = Entity.model_validate(item)
        idx = existing.get(entity.id)
        if idx is not None:
            ontology.entities[idx] = entity
        else:
            ontology.entities.append(entity)
            existing[entity.id] = len(
                ontology.entities,
            ) - 1


def _append_relationships(
    ontology: Ontology, items: list[Any],
) -> None:
    """Append new relationships to the ontology."""
    for item in items:
        ontology.relationships.append(
            Relationship.model_validate(item),
        )


def _append_constraints(
    ontology: Ontology, items: list[Any],
) -> None:
    """Append new domain constraints to the ontology."""
    for item in items:
        ontology.domain_constraints.append(
            DomainConstraint.model_validate(item),
        )


def _upsert_open_questions(
    ontology: Ontology, items: list[Any],
) -> None:
    """Replace or append open questions by id."""
    existing = {
        q.id: i
        for i, q in enumerate(ontology.open_questions)
    }
    for item in items:
        question = OpenQuestion.model_validate(item)
        idx = existing.get(question.id)
        if idx is not None:
            ontology.open_questions[idx] = question
        else:
            ontology.open_questions.append(question)
            existing[question.id] = len(
                ontology.open_questions,
            ) - 1


_MERGE_DISPATCH = {
    "entities": _upsert_entities,
    "relationships": _append_relationships,
    "domain_constraints": _append_constraints,
    "open_questions": _upsert_open_questions,
}


def merge_ontology_update(
    ontology: Ontology, update_dict: dict[str, Any],
) -> bool:
    """Apply a partial ontology update to an Ontology.

    Returns True if any updates were applied.
    """
    applied = False
    for key, handler in _MERGE_DISPATCH.items():
        items = update_dict.get(key)
        if items:
            handler(ontology, items)
            applied = True
    return applied


def process_response(
    response_text: str, ontology: Ontology,
) -> bool:
    """Extract and apply ontology updates from response text.

    Returns True if an update was applied.
    """
    update = extract_ontology_json(response_text)
    if update is None:
        return False
    return merge_ontology_update(ontology, update)


def _format_entities(ontology: Ontology) -> list[str]:
    """Format entity lines for summary."""
    lines = [f"Entities ({len(ontology.entities)}):"]
    for e in ontology.entities:
        props = ", ".join(p.name for p in e.properties)
        lines.append(f"  {e.id}: {e.name} [{props}]")
    return lines


def _format_relationships(
    ontology: Ontology,
) -> list[str]:
    """Format relationship lines for summary."""
    lines = [
        f"Relationships ({len(ontology.relationships)}):",
    ]
    for r in ontology.relationships:
        lines.append(
            f"  {r.source_entity_id} --{r.name}--> "
            f"{r.target_entity_id} ({r.cardinality})",
        )
    return lines


def _format_questions(ontology: Ontology) -> list[str]:
    """Format open question lines for summary."""
    lines = [
        f"Open Questions ({len(ontology.open_questions)}):",
    ]
    for q in ontology.open_questions:
        status = "RESOLVED" if q.resolved else "open"
        lines.append(f"  [{status}] {q.id}: {q.text}")
    return lines


def format_ontology_summary(ontology: Ontology) -> str:
    """Format a human-readable summary of the ontology."""
    lines = _format_entities(ontology)
    lines += _format_relationships(ontology)
    n = len(ontology.domain_constraints)
    lines.append(f"Constraints ({n}):")
    for c in ontology.domain_constraints:
        lines.append(f"  {c.name}: {c.description}")
    lines += _format_questions(ontology)
    return "\n".join(lines)


def backtrack(dag: OntologyDAG) -> DAGNode | None:
    """Move to the parent of the current node.

    Returns the new current node, or None if at root.
    """
    parents = dag.parents_of(dag.current_node_id)
    if not parents:
        return None
    dag.current_node_id = parents[0].id
    return dag.get_current_node()


def _handle_show(ontology: Ontology) -> str:
    """Handle the 'show' command."""
    return format_ontology_summary(ontology)


def _handle_save(
    command: str, ontology: Ontology,
    dag: OntologyDAG, dag_path: str,
) -> str:
    """Handle the 'save' command."""
    label = command.strip()[4:].strip() or "snapshot"
    save_snapshot(dag, ontology, label)
    save_dag(dag, dag_path)
    return f"Saved snapshot: {dag.current_node_id}"


def _handle_back(
    ontology: Ontology, dag: OntologyDAG,
    dag_path: str,
) -> str:
    """Handle the 'back' command."""
    node = backtrack(dag)
    if node is None:
        return "Already at root."
    ontology.__dict__.update(
        node.ontology.model_copy(deep=True).__dict__,
    )
    save_dag(dag, dag_path)
    return f"Backtracked to: {node.id} ({node.label})"


def is_command(text: str) -> bool:
    """Check if user input is a meta-command."""
    cmd = text.strip().lower()
    return cmd in ("show", "back") or cmd.startswith(
        "save",
    )


def handle_command(
    command: str, ontology: Ontology,
    dag: OntologyDAG, dag_path: str,
) -> str:
    """Dispatch a user meta-command. Returns display string."""
    cmd = command.strip().lower()
    if cmd == "show":
        return _handle_show(ontology)
    if cmd == "back":
        return _handle_back(ontology, dag, dag_path)
    return _handle_save(command, ontology, dag, dag_path)


def print_text_blocks(message: Any) -> None:
    """Print TextBlock content from an AssistantMessage."""
    for block in message.content:
        if isinstance(block, TextBlock):
            print(block.text)


async def print_response(client: Any) -> str:
    """Receive and print response. Return full text."""
    parts: list[str] = []
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            print_text_blocks(message)
            parts.append(collect_response_text(message))
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


def _init_ontology(dag: OntologyDAG) -> Ontology:
    """Get ontology from current DAG node, or a new one."""
    node = dag.get_current_node()
    if node is None:
        return Ontology()
    return node.ontology.model_copy(deep=True)


# taint: ignore[CWE-200] -- Interactive agent displays LLM output to user
async def run(
    description: str, model: str, dag_path: str,
) -> None:
    """Run the discovery agent interactively."""
    prompt = discovery_system_prompt()
    dag = load_dag(dag_path, description)
    ontology = _init_ontology(dag)

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=prompt,
        allowed_tools=[],
        permission_mode="default",
    )

    async with ClaudeSDKClient(options=options) as client:
        framed = frame_data("user-input", description)
        await client.query(framed)
        text = await print_response(client)
        process_response(text, ontology)

        while True:
            user_input = read_user_input()
            if user_input is None:
                break
            if is_command(user_input):
                msg = handle_command(
                    user_input, ontology, dag, dag_path,
                )
                print(msg)
                continue
            framed = frame_data("user-input", user_input)
            await client.query(framed)
            text = await print_response(client)
            process_response(text, ontology)


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive ontology discovery agent",
    )
    parser.add_argument(
        "description",
        help="Project description to start discovery",
    )
    parser.add_argument(
        "--dag-file",
        default="ontology.json",
        help="Path to DAG JSON file "
        "(default: ontology.json)",
    )
    parser.add_argument(
        "-m", "--model",
        default="claude-opus-4-6",
        help="Model to use (default: claude-opus-4-6)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the discovery-agent CLI."""
    args = parse_args(argv)
    asyncio.run(
        run(args.description, args.model, args.dag_file),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
