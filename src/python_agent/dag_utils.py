"""Shared DAG persistence and snapshot utilities."""

from __future__ import annotations

import os
import tempfile
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

from python_agent.dag_integrity import (
    load_or_create_key,
    scan_ontology_for_injection,
    sign_node,
    verify_dag,
)
from python_agent.ontology import (
    DAGEdge,
    DAGNode,
    Decision,
    Ontology,
    OntologyDAG,
)


def _default_key_path(dag_path: str) -> str:
    """Derive key file path from DAG file path."""
    return str(Path(dag_path).parent / ".dag-key")


def _sign_unsigned_nodes(
    dag: OntologyDAG, key: str,
) -> None:
    """Sign all nodes missing an integrity hash."""
    for node in dag.nodes:
        if not node.integrity_hash:
            sign_node(node, key)


def _verify_loaded_dag(
    dag: OntologyDAG, key_path: str,
) -> None:
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


def _scan_loaded_dag(dag: OntologyDAG) -> None:
    """Scan all nodes for injection patterns."""
    for node in dag.nodes:
        hits = scan_ontology_for_injection(
            node.ontology.model_dump(),
        )
        for hit in hits:
            warnings.warn(
                f"Node {node.id}: {hit}",
                stacklevel=3,
            )


def _read_file(path: str) -> str | None:
    """Read file contents, or return None if not found."""
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return None


def _parse_dag(text: str) -> OntologyDAG | None:
    """Parse DAG JSON, or return None with warning."""
    try:
        return OntologyDAG.from_json(text)
    except Exception as exc:
        warnings.warn(
            f"DAG validation error: {exc}",
            stacklevel=3,
        )
        return None


def load_dag(
    path: str, project_name: str,
    key_path: str | None = None,
) -> OntologyDAG:
    """Load an OntologyDAG from a JSON file.

    Returns a new empty DAG if not found or invalid.
    """
    text = _read_file(path)
    if text is None:
        return OntologyDAG(project_name=project_name)
    dag = _parse_dag(text)
    if dag is None:
        return OntologyDAG(project_name=project_name)
    if key_path is None:
        key_path = _default_key_path(path)
    _verify_loaded_dag(dag, key_path)
    _scan_loaded_dag(dag)
    return dag


def save_dag(
    dag: OntologyDAG, path: str,
    key_path: str | None = None,
) -> None:
    """Save an OntologyDAG, signing unsigned nodes.

    Uses atomic write (temp file + rename) to prevent
    corruption from interrupted writes.
    """
    if key_path is None:
        key_path = _default_key_path(path)
    key = load_or_create_key(key_path)
    _sign_unsigned_nodes(dag, key)
    parent_dir = os.path.dirname(os.path.abspath(path))
    fd = tempfile.NamedTemporaryFile(
        mode="w", dir=parent_dir,
        suffix=".tmp", delete=False,
    )
    try:
        fd.write(dag.to_json())
        fd.close()
        os.rename(fd.name, path)
    except BaseException:
        fd.close()
        os.unlink(fd.name)
        raise


def make_node_id() -> str:
    """Generate a unique node ID using uuid4."""
    return str(uuid.uuid4())


def save_snapshot(
    dag: OntologyDAG, ontology: Ontology,
    label: str, decision: Decision | None = None,
) -> str:
    """Create a new DAG node from the current ontology.

    Links it as a child of the current node if one exists.
    If decision is None, a default decision is used.
    Returns the new node id.
    """
    now = datetime.now(timezone.utc).isoformat()
    node_id = make_node_id()
    node = DAGNode(
        id=node_id,
        ontology=ontology.model_copy(deep=True),
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
