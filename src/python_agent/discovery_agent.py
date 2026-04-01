"""Discovery agent: interactive ontology building."""

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime, timezone

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
)

from python_agent.ontology import (
    DAGEdge,
    DAGNode,
    Decision,
    DomainConstraint,
    Entity,
    Ontology,
    OntologyDAG,
    OpenQuestion,
    Relationship,
)
from python_agent.rules import discovery_system_prompt

_ONTOLOGY_BLOCK_RE = re.compile(
    r"```ontology\s*\n(.*?)\n```",
    re.DOTALL,
)


def collect_response_text(message):
    """Extract concatenated text from an AssistantMessage."""
    parts = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
    return "\n".join(parts)


def extract_ontology_json(text):
    """Extract the first ontology JSON block from text.

    Returns the parsed dict, or None if no valid block found.
    """
    match = _ONTOLOGY_BLOCK_RE.search(text)
    if match is None:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _upsert_entities(ontology, new_entities):
    """Replace or append entities by id."""
    existing = {
        e.id: i for i, e in enumerate(ontology.entities)
    }
    for item in new_entities:
        entity = Entity.from_dict(item)
        idx = existing.get(entity.id)
        if idx is not None:
            ontology.entities[idx] = entity
        else:
            ontology.entities.append(entity)
            existing[entity.id] = len(
                ontology.entities,
            ) - 1


def _append_relationships(ontology, items):
    """Append new relationships to the ontology."""
    for item in items:
        ontology.relationships.append(
            Relationship.from_dict(item),
        )


def _append_constraints(ontology, items):
    """Append new domain constraints to the ontology."""
    for item in items:
        ontology.domain_constraints.append(
            DomainConstraint.from_dict(item),
        )


def _upsert_open_questions(ontology, items):
    """Replace or append open questions by id."""
    existing = {
        q.id: i
        for i, q in enumerate(ontology.open_questions)
    }
    for item in items:
        question = OpenQuestion.from_dict(item)
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


def merge_ontology_update(ontology, update_dict):
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


def process_response(response_text, ontology):
    """Extract and apply ontology updates from response text.

    Returns True if an update was applied.
    """
    update = extract_ontology_json(response_text)
    if update is None:
        return False
    return merge_ontology_update(ontology, update)


def _format_entities(ontology):
    """Format entity lines for summary."""
    lines = [f"Entities ({len(ontology.entities)}):"]
    for e in ontology.entities:
        props = ", ".join(p.name for p in e.properties)
        lines.append(f"  {e.id}: {e.name} [{props}]")
    return lines


def _format_relationships(ontology):
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


def _format_questions(ontology):
    """Format open question lines for summary."""
    lines = [
        f"Open Questions ({len(ontology.open_questions)}):",
    ]
    for q in ontology.open_questions:
        status = "RESOLVED" if q.resolved else "open"
        lines.append(f"  [{status}] {q.id}: {q.text}")
    return lines


def format_ontology_summary(ontology):
    """Format a human-readable summary of the ontology."""
    lines = _format_entities(ontology)
    lines += _format_relationships(ontology)
    n = len(ontology.domain_constraints)
    lines.append(f"Constraints ({n}):")
    for c in ontology.domain_constraints:
        lines.append(f"  {c.name}: {c.description}")
    lines += _format_questions(ontology)
    return "\n".join(lines)


def load_dag(path, project_name):
    """Load an OntologyDAG from a JSON file.

    Returns a new empty DAG if the file does not exist.
    """
    try:
        with open(path) as f:
            return OntologyDAG.from_json(f.read())
    except FileNotFoundError:
        return OntologyDAG(project_name=project_name)


def save_dag(dag, path):
    """Save an OntologyDAG to a JSON file."""
    with open(path, "w") as f:
        f.write(dag.to_json())


def _make_node_id():
    """Generate a unique node ID from current timestamp."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%S")


def save_snapshot(dag, ontology, label):
    """Create a new DAG node from the current ontology.

    Links it as a child of the current node if one exists.
    Returns the new node id.
    """
    now = datetime.now(timezone.utc).isoformat()
    node_id = _make_node_id()
    node = DAGNode(
        id=node_id,
        ontology=Ontology.from_dict(ontology.to_dict()),
        created_at=now,
        label=label,
    )
    dag.nodes.append(node)
    if dag.current_node_id:
        edge = DAGEdge(
            parent_id=dag.current_node_id,
            child_id=node_id,
            decision=Decision(
                question="save",
                options=["continue"],
                chosen="continue",
                rationale=label,
            ),
            created_at=now,
        )
        dag.edges.append(edge)
    dag.current_node_id = node_id
    return node_id


def backtrack(dag):
    """Move to the parent of the current node.

    Returns the new current node, or None if at root.
    """
    parents = dag.parents_of(dag.current_node_id)
    if not parents:
        return None
    dag.current_node_id = parents[0].id
    return dag.get_current_node()


def _handle_show(ontology):
    """Handle the 'show' command."""
    return format_ontology_summary(ontology)


def _handle_save(command, ontology, dag, dag_path):
    """Handle the 'save' command."""
    label = command.strip()[4:].strip() or "snapshot"
    save_snapshot(dag, ontology, label)
    save_dag(dag, dag_path)
    return f"Saved snapshot: {dag.current_node_id}"


def _handle_back(ontology, dag, dag_path):
    """Handle the 'back' command."""
    node = backtrack(dag)
    if node is None:
        return "Already at root."
    ontology.__dict__.update(
        Ontology.from_dict(
            node.ontology.to_dict(),
        ).__dict__,
    )
    save_dag(dag, dag_path)
    return f"Backtracked to: {node.id} ({node.label})"


def is_command(text):
    """Check if user input is a meta-command."""
    cmd = text.strip().lower()
    return cmd in ("show", "back") or cmd.startswith(
        "save",
    )


def handle_command(command, ontology, dag, dag_path):
    """Dispatch a user meta-command. Returns display string."""
    cmd = command.strip().lower()
    if cmd == "show":
        return _handle_show(ontology)
    if cmd == "back":
        return _handle_back(ontology, dag, dag_path)
    return _handle_save(command, ontology, dag, dag_path)


def print_text_blocks(message):
    """Print TextBlock content from an AssistantMessage."""
    for block in message.content:
        if isinstance(block, TextBlock):
            print(block.text)


async def print_response(client):
    """Receive and print response. Return full text."""
    parts = []
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            print_text_blocks(message)
            parts.append(collect_response_text(message))
    return "\n".join(parts)


def read_user_input():
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


def _init_ontology(dag):
    """Get ontology from current DAG node, or a new one."""
    node = dag.get_current_node()
    if node is None:
        return Ontology()
    return Ontology.from_dict(node.ontology.to_dict())


async def run(description, model, dag_path):
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
        await client.query(description)
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
            await client.query(user_input)
            text = await print_response(client)
            process_response(text, ontology)


def parse_args(argv=None):
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


def main(argv=None):
    """Entry point for the discovery-agent CLI."""
    args = parse_args(argv)
    asyncio.run(
        run(args.description, args.model, args.dag_file),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
