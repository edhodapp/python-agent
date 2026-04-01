"""Shared DAG persistence and snapshot utilities."""

from datetime import datetime, timezone

from python_agent.ontology import (
    DAGEdge,
    DAGNode,
    Decision,
    Ontology,
    OntologyDAG,
)


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
