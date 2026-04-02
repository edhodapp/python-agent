"""Tests for dag_integrity module."""

from python_agent.dag_integrity import (
    compute_hash,
    generate_key,
    load_or_create_key,
    sign_node,
    verify_dag,
    verify_node,
)
from python_agent.ontology import (
    DAGNode,
    Entity,
    Ontology,
    OntologyDAG,
)


class TestGenerateKey:
    """Tests for generate_key."""

    def test_returns_hex_string(self):
        key = generate_key()
        assert len(key) == 64
        int(key, 16)  # validates hex

    def test_unique_keys(self):
        assert generate_key() != generate_key()


class TestLoadOrCreateKey:
    """Tests for load_or_create_key."""

    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "key")
        key = load_or_create_key(path)
        assert len(key) == 64
        with open(path) as f:
            assert f.read().strip() == key

    def test_loads_existing(self, tmp_path):
        path = str(tmp_path / "key")
        with open(path, "w") as f:
            f.write("ab" * 32)
        key = load_or_create_key(path)
        assert key == "ab" * 32

    def test_reuse(self, tmp_path):
        path = str(tmp_path / "key")
        k1 = load_or_create_key(path)
        k2 = load_or_create_key(path)
        assert k1 == k2


class TestComputeHash:
    """Tests for compute_hash."""

    def test_deterministic(self):
        key = generate_key()
        d = {"a": 1}
        assert compute_hash(d, key) == compute_hash(d, key)

    def test_different_keys(self):
        d = {"a": 1}
        k1 = generate_key()
        k2 = generate_key()
        assert compute_hash(d, k1) != compute_hash(d, k2)

    def test_different_content(self):
        key = generate_key()
        h1 = compute_hash({"a": 1}, key)
        h2 = compute_hash({"a": 2}, key)
        assert h1 != h2

    def test_returns_hex_string(self):
        key = generate_key()
        h = compute_hash({}, key)
        assert len(h) == 64
        int(h, 16)


class TestSignNode:
    """Tests for sign_node."""

    def test_sets_hash(self):
        node = DAGNode(
            id="n1", ontology=Ontology(),
            created_at="t",
        )
        key = generate_key()
        sign_node(node, key)
        assert node.integrity_hash != ""
        assert len(node.integrity_hash) == 64


class TestVerifyNode:
    """Tests for verify_node."""

    def test_signed_passes(self):
        node = DAGNode(
            id="n1", ontology=Ontology(),
            created_at="t",
        )
        key = generate_key()
        sign_node(node, key)
        assert verify_node(node, key) is True

    def test_tampered_fails(self):
        node = DAGNode(
            id="n1",
            ontology=Ontology(
                entities=[Entity(id="e1", name="X")],
            ),
            created_at="t",
        )
        key = generate_key()
        sign_node(node, key)
        node.ontology.entities[0] = Entity(
            id="e1", name="TAMPERED",
        )
        assert verify_node(node, key) is False

    def test_unsigned_fails(self):
        node = DAGNode(
            id="n1", ontology=Ontology(),
            created_at="t",
        )
        key = generate_key()
        assert verify_node(node, key) is False

    def test_wrong_key_fails(self):
        node = DAGNode(
            id="n1", ontology=Ontology(),
            created_at="t",
        )
        k1 = generate_key()
        k2 = generate_key()
        sign_node(node, k1)
        assert verify_node(node, k2) is False


class TestVerifyDag:
    """Tests for verify_dag."""

    def test_all_signed_passes(self):
        key = generate_key()
        n1 = DAGNode(
            id="n1", ontology=Ontology(), created_at="t",
        )
        n2 = DAGNode(
            id="n2", ontology=Ontology(), created_at="t",
        )
        sign_node(n1, key)
        sign_node(n2, key)
        dag = OntologyDAG(
            project_name="p", nodes=[n1, n2],
        )
        assert verify_dag(dag, key) == []

    def test_tampered_detected(self):
        key = generate_key()
        n1 = DAGNode(
            id="n1",
            ontology=Ontology(
                entities=[Entity(id="e1", name="X")],
            ),
            created_at="t",
        )
        sign_node(n1, key)
        n1.ontology.entities[0] = Entity(
            id="e1", name="TAMPERED",
        )
        dag = OntologyDAG(
            project_name="p", nodes=[n1],
        )
        assert verify_dag(dag, key) == ["n1"]

    def test_unsigned_skipped(self):
        key = generate_key()
        n1 = DAGNode(
            id="n1", ontology=Ontology(), created_at="t",
        )
        dag = OntologyDAG(
            project_name="p", nodes=[n1],
        )
        assert verify_dag(dag, key) == []
