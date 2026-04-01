"""Tests for dag_utils module."""

from python_agent.dag_utils import (
    load_dag,
    make_node_id,
    save_dag,
    save_snapshot,
)
from python_agent.ontology import (
    Decision,
    Entity,
    Ontology,
    OntologyDAG,
)


class TestLoadDag:
    """Tests for load_dag."""

    def test_missing_file(self, tmp_path):
        dag = load_dag(str(tmp_path / "nope.json"), "proj")
        assert dag.project_name == "proj"
        assert dag.nodes == []

    def test_existing_file(self, tmp_path):
        path = str(tmp_path / "dag.json")
        dag = OntologyDAG(project_name="test")
        save_dag(dag, path)
        loaded = load_dag(path, "ignored")
        assert loaded.project_name == "test"


class TestSaveDag:
    """Tests for save_dag."""

    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "dag.json")
        dag = OntologyDAG(project_name="test")
        save_dag(dag, path)
        loaded = load_dag(path, "x")
        assert loaded.project_name == "test"


class TestMakeNodeId:
    """Tests for make_node_id."""

    def test_returns_string(self):
        nid = make_node_id()
        assert isinstance(nid, str)
        assert len(nid) > 0

    def test_contains_timestamp(self):
        nid = make_node_id()
        assert "T" in nid


class TestSaveSnapshot:
    """Tests for save_snapshot."""

    def test_first_snapshot_no_edge(self):
        dag = OntologyDAG(project_name="p")
        o = Ontology(
            entities=[Entity(id="e1", name="X")],
        )
        node_id = save_snapshot(dag, o, "initial")
        assert len(dag.nodes) == 1
        assert dag.nodes[0].label == "initial"
        assert dag.current_node_id == node_id
        assert len(dag.edges) == 0

    def test_second_snapshot_creates_edge(self):
        dag = OntologyDAG(project_name="p")
        save_snapshot(dag, Ontology(), "first")
        save_snapshot(dag, Ontology(), "second")
        assert len(dag.nodes) == 2
        assert len(dag.edges) == 1

    def test_default_decision(self):
        dag = OntologyDAG(project_name="p")
        save_snapshot(dag, Ontology(), "first")
        save_snapshot(dag, Ontology(), "second")
        edge = dag.edges[0]
        assert edge.decision.question == "save"
        assert edge.decision.chosen == "continue"

    def test_custom_decision(self):
        dag = OntologyDAG(project_name="p")
        save_snapshot(dag, Ontology(), "first")
        decision = Decision(
            question="Which DB?",
            options=["SQLite", "Postgres"],
            chosen="SQLite",
            rationale="Simpler for v1",
        )
        save_snapshot(
            dag, Ontology(), "second", decision=decision,
        )
        edge = dag.edges[0]
        assert edge.decision.question == "Which DB?"
        assert edge.decision.chosen == "SQLite"

    def test_deep_copy(self):
        dag = OntologyDAG(project_name="p")
        o = Ontology(
            entities=[Entity(id="e1", name="X")],
        )
        save_snapshot(dag, o, "snap")
        o.entities[0] = Entity(id="e1", name="Changed")
        assert dag.nodes[0].ontology.entities[0].name == "X"
