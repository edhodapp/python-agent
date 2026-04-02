"""Shared DAG persistence and snapshot utilities."""

import warnings
from datetime import datetime, timezone
from pathlib import Path

from python_agent.dag_integrity import (
    load_or_create_key,
    sign_node,
    verify_dag,
)
from python_agent.ontology import (
    DAGEdge,
    DAGNode,
    Decision,
    Ontology,
    OntologyDAG,
    validate_ontology_strict,
)


def _default_key_path(dag_path):
    """Derive key file path from DAG file path."""
    return str(Path(dag_path).parent / ".dag-key")


def _sign_unsigned_nodes(dag, key):
    """Sign all nodes missing an integrity hash."""
    for node in dag.nodes:
        if not node.integrity_hash:
            sign_node(node, key)


def _verify_loaded_dag(dag, key_path):
    """Verify all nodes and warn on failures."""
    try:
        key = load_or_create_key(key_path)
    except (OSError, ValueError):
        return
    failed = verify_dag(dag, key)
    if failed:
        warnings.warn(
            "DAG integrity check failed for nodes: "
            + ", ".join(failed),
            stacklevel=3,
        )


def _validate_loaded_dag(dag):
    """Validate all node ontologies. Warn on errors."""
    for node in dag.nodes:
        errors = validate_ontology_strict(
            node.ontology.to_dict(),
        )
        if errors:
            warnings.warn(
                f"Node {node.id} validation errors: "
                + "; ".join(errors),
                stacklevel=3,
            )


def load_dag(path, project_name, key_path=None):
    """Load an OntologyDAG from a JSON file.

    Returns a new empty DAG if the file does not exist.
    Verifies integrity and validates on load.
    """
    try:
        with open(path) as f:
            dag = OntologyDAG.from_json(f.read())
    except FileNotFoundError:
        return OntologyDAG(project_name=project_name)
    if key_path is None:
        key_path = _default_key_path(path)
    _verify_loaded_dag(dag, key_path)
    _validate_loaded_dag(dag)
    return dag


def save_dag(dag, path, key_path=None):
    """Save an OntologyDAG, signing unsigned nodes."""
    if key_path is None:
        key_path = _default_key_path(path)
    key = load_or_create_key(key_path)
    _sign_unsigned_nodes(dag, key)
    with open(path, "w") as f:
        f.write(dag.to_json())


def make_node_id():
    """Generate a unique node ID from current timestamp."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%S%f")


def save_snapshot(dag, ontology, label, decision=None):
    """Create a new DAG node from the current ontology.

    Links it as a child of the current node if one exists.
    If decision is None, a default decision is used.
    Returns the new node id.
    """
    now = datetime.now(timezone.utc).isoformat()
    node_id = make_node_id()
    node = DAGNode(
        id=node_id,
        ontology=Ontology.from_dict(ontology.to_dict()),
        created_at=now,
        label=label,
    )
    dag.nodes.append(node)
    if dag.current_node_id:
        if decision is None:
            decision = Decision(
                question="save",
                options=["continue"],
                chosen="continue",
                rationale=label,
            )
        edge = DAGEdge(
            parent_id=dag.current_node_id,
            child_id=node_id,
            decision=decision,
            created_at=now,
        )
        dag.edges.append(edge)
    dag.current_node_id = node_id
    return node_id
