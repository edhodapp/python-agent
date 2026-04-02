"""Divergence agent: generates multiple solution candidates."""

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
    ResultMessage,
    TextBlock,
    query,
)

from python_agent.dag_utils import load_dag, save_dag
from python_agent.ontology import (
    DAGEdge,
    DAGNode,
    Decision,
    Ontology,
    OntologyDAG,
)
from python_agent.rules import (
    divergence_system_prompt,
    strategy_system_prompt,
)

_ONTOLOGY_BLOCK_RE = re.compile(
    r"```ontology\s*\n(.*?)\n```",
    re.DOTALL,
)

_STRATEGIES_BLOCK_RE = re.compile(
    r"```strategies\s*\n(.*?)\n```",
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
    """Extract the first ontology JSON block from text."""
    match = _ONTOLOGY_BLOCK_RE.search(text)
    if match is None:
        return None
    try:
        result: dict[str, Any] = json.loads(match.group(1))
        return result
    except json.JSONDecodeError:
        return None


def extract_strategies(
    text: str,
) -> list[dict[str, Any]] | None:
    """Extract strategies JSON block from text.

    Returns a list of strategy dicts, or None.
    """
    match = _STRATEGIES_BLOCK_RE.search(text)
    if match is None:
        return None
    try:
        result = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(result, list):
        return None
    return result


async def run_query(
    task: str, options: Any,
) -> tuple[str, float]:
    """Run a single query. Return (response_text, cost)."""
    parts: list[str] = []
    cost = 0.0
    async for message in query(
        prompt=task, options=options,
    ):
        if isinstance(message, AssistantMessage):
            parts.append(collect_response_text(message))
        elif isinstance(message, ResultMessage):
            cost = message.total_cost_usd or 0.0
    return "\n".join(parts), cost


def build_decision(
    strategy: dict[str, Any],
) -> Decision:
    """Create a Decision from a strategy dict."""
    options = strategy.get("options", [])
    if not isinstance(options, list):
        options = []
    return Decision(
        question=str(strategy.get("question", "architecture")),
        options=options,
        chosen=str(strategy.get("chosen", "")),
        rationale=str(strategy.get("strategy", "")),
    )


def add_candidate_node(
    dag: OntologyDAG, parent_id: str,
    ontology_dict: dict[str, Any],
    strategy: dict[str, Any],
) -> str:
    """Create a DAG node for a candidate and link it."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    node_id = now.strftime("%Y%m%dT%H%M%S%f")
    label: str = strategy.get("label", "candidate")
    ontology = Ontology.model_validate(ontology_dict)
    node = DAGNode(
        id=node_id,
        ontology=ontology,
        created_at=now.isoformat(),
        label=label,
    )
    dag.nodes.append(node)
    edge = DAGEdge(
        parent_id=parent_id,
        child_id=node_id,
        decision=build_decision(strategy),
        created_at=now.isoformat(),
    )
    dag.edges.append(edge)
    return node_id


def remaining_budget(
    spent: float, max_budget: float | None,
) -> float | None:
    """Calculate remaining budget after spending."""
    if max_budget is None:
        return None
    return max_budget - spent


async def identify_strategies(
    ontology_json: str, num_candidates: int,
    model: str, max_budget: float | None,
) -> tuple[list[dict[str, Any]], float]:
    """Identify distinct architectural strategies."""
    prompt = strategy_system_prompt(
        ontology_json, num_candidates,
    )
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=prompt,
        allowed_tools=[],
        permission_mode="default",
        max_turns=1,
        max_budget_usd=max_budget,
    )
    task = (
        f"Identify {num_candidates} distinct architectural "
        "strategies for this problem domain."
    )
    text, cost = await run_query(task, options)
    strategies = extract_strategies(text)
    if strategies is None:
        return [], cost
    return strategies[:num_candidates], cost


async def generate_candidate(
    ontology_json: str, strategy: dict[str, Any],
    model: str, max_budget: float | None,
) -> tuple[dict[str, Any] | None, float]:
    """Generate one solution candidate for a strategy."""
    label: str = strategy.get("label", "candidate")
    description: str = strategy.get("strategy", "")
    prompt = divergence_system_prompt(
        ontology_json, description,
    )
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=prompt,
        allowed_tools=[],
        permission_mode="default",
        max_turns=1,
        max_budget_usd=max_budget,
    )
    task = (
        f"Generate a complete solution architecture "
        f"following the '{label}' strategy: {description}"
    )
    text, cost = await run_query(task, options)
    result = extract_ontology_json(text)
    return result, cost


async def run(
    dag_file: str, num_candidates: int,
    model: str, max_budget: float | None,
) -> int:
    """Run the divergence agent. Returns candidate count."""
    dag = load_dag(dag_file, "unknown")
    node = dag.get_current_node()
    if node is None:
        print("Error: DAG has no current node.")
        return 0
    ontology_json = json.dumps(
        node.ontology.model_dump(), indent=2,
    )
    parent_id = dag.current_node_id
    total_cost = 0.0

    print(f"Identifying {num_candidates} strategies...")
    strategies, cost = await identify_strategies(
        ontology_json, num_candidates, model, max_budget,
    )
    total_cost += cost
    if not strategies:
        print("Error: Could not identify strategies.")
        return 0

    generated = await _generate_all(
        dag, parent_id, ontology_json, strategies,
        model, total_cost, max_budget,
    )
    save_dag(dag, dag_file)
    return generated


async def _generate_all(
    dag: OntologyDAG, parent_id: str,
    ontology_json: str,
    strategies: list[dict[str, Any]],
    model: str, total_cost: float,
    max_budget: float | None,
) -> int:
    """Generate all candidates and add to DAG."""
    generated = 0
    for strategy in strategies:
        label: str = strategy.get("label", "candidate")
        print(f"Generating candidate: {label}...")
        budget = remaining_budget(total_cost, max_budget)
        result, cost = await generate_candidate(
            ontology_json, strategy, model, budget,
        )
        total_cost += cost
        if result is None:
            print(f"  Failed: {label}")
            continue
        add_candidate_node(
            dag, parent_id, result, strategy,
        )
        generated += 1
        print(f"  Created: {label}")
    cost_str = f"${total_cost:.4f}"
    print(f"\nDone. {generated} candidates. Cost: {cost_str}")
    return generated


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate divergent solution candidates",
    )
    parser.add_argument(
        "--dag-file",
        required=True,
        help="Path to DAG JSON file",
    )
    parser.add_argument(
        "-n", "--num-candidates",
        type=int,
        default=3,
        help="Number of candidates (default: 3)",
    )
    parser.add_argument(
        "-m", "--model",
        default="claude-sonnet-4-6",
        help="Model to use (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--max-budget",
        type=float,
        default=5.0,
        help="Maximum budget in USD (default: 5.0)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the divergence-agent CLI."""
    args = parse_args(argv)
    asyncio.run(
        run(
            args.dag_file,
            args.num_candidates,
            args.model,
            args.max_budget,
        ),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
