"""Tests for ontology module."""

from python_agent.ontology import (
    ClassSpec,
    DAGEdge,
    DAGNode,
    DataModel,
    Decision,
    DomainConstraint,
    Entity,
    ExternalDependency,
    FunctionSpec,
    ModuleSpec,
    Ontology,
    OntologyDAG,
    OpenQuestion,
    Property,
    PropertyType,
    Relationship,
    validate_ontology_strict,
)


class TestPropertyType:
    """Tests for PropertyType."""

    def test_scalar(self):
        pt = PropertyType(kind="str")
        assert pt.kind == "str"
        assert pt.reference is None

    def test_entity_ref(self):
        pt = PropertyType(kind="entity_ref", reference="e1")
        assert pt.reference == "e1"

    def test_enum(self):
        pt = PropertyType(
            kind="enum", reference=["a", "b"],
        )
        assert pt.reference == ["a", "b"]

    def test_round_trip(self):
        pt = PropertyType(kind="list", reference="int")
        assert PropertyType.from_dict(pt.to_dict()) == pt


class TestProperty:
    """Tests for Property."""

    def test_defaults(self):
        p = Property(
            name="x",
            property_type=PropertyType(kind="str"),
        )
        assert p.required is True
        assert p.constraints == []
        assert p.description == ""

    def test_round_trip(self):
        p = Property(
            name="age",
            property_type=PropertyType(kind="int"),
            description="User age",
            required=False,
            constraints=["min: 0"],
        )
        assert Property.from_dict(p.to_dict()) == p

    def test_from_dict_missing_optionals(self):
        data = {
            "name": "x",
            "property_type": {"kind": "str"},
        }
        p = Property.from_dict(data)
        assert p.description == ""
        assert p.required is True
        assert p.constraints == []


class TestEntity:
    """Tests for Entity."""

    def test_construction(self):
        e = Entity(id="e1", name="User")
        assert e.id == "e1"
        assert e.properties == []
        assert e.description == ""

    def test_round_trip_with_properties(self):
        e = Entity(
            id="e1",
            name="User",
            description="A user",
            properties=[
                Property(
                    name="name",
                    property_type=PropertyType(kind="str"),
                ),
            ],
        )
        assert Entity.from_dict(e.to_dict()) == e

    def test_from_dict_missing_optionals(self):
        data = {"id": "e1", "name": "Thing"}
        e = Entity.from_dict(data)
        assert e.description == ""
        assert e.properties == []


class TestRelationship:
    """Tests for Relationship."""

    def test_construction(self):
        r = Relationship(
            source_entity_id="e1",
            target_entity_id="e2",
            name="owns",
            cardinality="one_to_many",
        )
        assert r.description == ""

    def test_round_trip(self):
        r = Relationship(
            source_entity_id="e1",
            target_entity_id="e2",
            name="owns",
            cardinality="one_to_many",
            description="ownership",
        )
        assert Relationship.from_dict(r.to_dict()) == r


class TestRelationshipFromDict:
    """Tests for Relationship.from_dict with missing optionals."""

    def test_missing_description(self):
        data = {
            "source_entity_id": "e1",
            "target_entity_id": "e2",
            "name": "owns",
            "cardinality": "one_to_many",
        }
        r = Relationship.from_dict(data)
        assert r.description == ""


class TestDomainConstraint:
    """Tests for DomainConstraint."""

    def test_construction(self):
        dc = DomainConstraint(name="c1", description="rule")
        assert dc.entity_ids == []
        assert dc.expression == ""

    def test_round_trip(self):
        dc = DomainConstraint(
            name="c1",
            description="rule",
            entity_ids=["e1"],
            expression="x > 0",
        )
        result = DomainConstraint.from_dict(dc.to_dict())
        assert result == dc

    def test_from_dict_missing_optionals(self):
        data = {"name": "c1", "description": "rule"}
        dc = DomainConstraint.from_dict(data)
        assert dc.entity_ids == []
        assert dc.expression == ""


class TestFunctionSpec:
    """Tests for FunctionSpec."""

    def test_construction(self):
        f = FunctionSpec(
            name="do_thing",
            parameters=[("x", "int")],
            return_type="str",
        )
        assert f.parameters == [("x", "int")]
        assert f.preconditions == []
        assert f.docstring == ""

    def test_round_trip(self):
        f = FunctionSpec(
            name="do_thing",
            parameters=[("x", "int"), ("y", "str")],
            return_type="bool",
            docstring="Does a thing",
            preconditions=["x > 0"],
            postconditions=["result is True"],
        )
        assert FunctionSpec.from_dict(f.to_dict()) == f

    def test_tuple_serialization(self):
        f = FunctionSpec(
            name="f",
            parameters=[("a", "int")],
            return_type="None",
        )
        d = f.to_dict()
        assert d["parameters"] == [["a", "int"]]
        restored = FunctionSpec.from_dict(d)
        assert restored.parameters == [("a", "int")]

    def test_from_dict_missing_optionals(self):
        data = {
            "name": "f",
            "return_type": "None",
        }
        f = FunctionSpec.from_dict(data)
        assert f.docstring == ""
        assert f.preconditions == []
        assert f.postconditions == []
        assert f.parameters == []


class TestClassSpec:
    """Tests for ClassSpec."""

    def test_construction(self):
        c = ClassSpec(name="Foo")
        assert c.bases == []
        assert c.methods == []
        assert c.description == ""

    def test_round_trip(self):
        c = ClassSpec(
            name="Foo",
            description="A class",
            bases=["Bar"],
            methods=[
                FunctionSpec(
                    name="m",
                    parameters=[],
                    return_type="None",
                ),
            ],
        )
        assert ClassSpec.from_dict(c.to_dict()) == c

    def test_from_dict_missing_optionals(self):
        data = {"name": "Foo"}
        c = ClassSpec.from_dict(data)
        assert c.description == ""
        assert c.bases == []
        assert c.methods == []


class TestDataModel:
    """Tests for DataModel."""

    def test_round_trip(self):
        dm = DataModel(
            entity_id="e1",
            storage="dataclass",
            class_name="User",
            notes="main model",
        )
        assert DataModel.from_dict(dm.to_dict()) == dm

    def test_default_notes(self):
        dm = DataModel(
            entity_id="e1",
            storage="dict",
            class_name="X",
        )
        assert dm.notes == ""

    def test_from_dict_missing_notes(self):
        data = {
            "entity_id": "e1",
            "storage": "dict",
            "class_name": "X",
        }
        dm = DataModel.from_dict(data)
        assert dm.notes == ""


class TestExternalDependency:
    """Tests for ExternalDependency."""

    def test_round_trip(self):
        ed = ExternalDependency(
            name="aiohttp",
            version_constraint=">=3.9",
            reason="HTTP server",
        )
        result = ExternalDependency.from_dict(ed.to_dict())
        assert result == ed

    def test_defaults(self):
        ed = ExternalDependency(name="click")
        assert ed.version_constraint == ""
        assert ed.reason == ""

    def test_from_dict_missing_optionals(self):
        data = {"name": "click"}
        ed = ExternalDependency.from_dict(data)
        assert ed.version_constraint == ""
        assert ed.reason == ""


class TestModuleSpec:
    """Tests for ModuleSpec."""

    def test_defaults(self):
        m = ModuleSpec(name="mod", responsibility="stuff")
        assert m.status == "not_started"
        assert m.classes == []
        assert m.functions == []
        assert m.test_strategy == ""

    def test_round_trip(self):
        m = ModuleSpec(
            name="app.api",
            responsibility="HTTP handlers",
            classes=[ClassSpec(name="Handler")],
            functions=[
                FunctionSpec(
                    name="route",
                    parameters=[],
                    return_type="None",
                ),
            ],
            dependencies=["app.models"],
            test_strategy="mock storage",
            status="in_progress",
        )
        assert ModuleSpec.from_dict(m.to_dict()) == m

    def test_from_dict_missing_optionals(self):
        data = {"name": "mod", "responsibility": "stuff"}
        m = ModuleSpec.from_dict(data)
        assert m.test_strategy == ""
        assert m.status == "not_started"
        assert m.classes == []
        assert m.functions == []
        assert m.dependencies == []


class TestOpenQuestion:
    """Tests for OpenQuestion."""

    def test_defaults(self):
        q = OpenQuestion(id="q1", text="What DB?")
        assert q.priority == "medium"
        assert q.resolved is False
        assert q.resolution == ""
        assert q.context == ""

    def test_round_trip(self):
        q = OpenQuestion(
            id="q1",
            text="What DB?",
            context="Need persistence",
            priority="high",
            resolved=True,
            resolution="SQLite",
        )
        assert OpenQuestion.from_dict(q.to_dict()) == q

    def test_from_dict_missing_optionals(self):
        data = {"id": "q1", "text": "What DB?"}
        q = OpenQuestion.from_dict(data)
        assert q.context == ""
        assert q.priority == "medium"
        assert q.resolved is False
        assert q.resolution == ""


class TestOntology:
    """Tests for Ontology."""

    def test_empty(self):
        o = Ontology()
        assert o.entities == []
        assert o.modules == []
        assert o.open_questions == []

    def test_round_trip(self):
        o = Ontology(
            entities=[Entity(id="e1", name="User")],
            relationships=[
                Relationship(
                    source_entity_id="e1",
                    target_entity_id="e2",
                    name="r",
                    cardinality="one_to_one",
                ),
            ],
            domain_constraints=[
                DomainConstraint(name="c", description="d"),
            ],
            modules=[
                ModuleSpec(name="m", responsibility="r"),
            ],
            data_models=[
                DataModel(
                    entity_id="e1",
                    storage="dataclass",
                    class_name="User",
                ),
            ],
            external_dependencies=[
                ExternalDependency(name="click"),
            ],
            open_questions=[
                OpenQuestion(id="q1", text="?"),
            ],
        )
        assert Ontology.from_dict(o.to_dict()) == o

    def test_from_dict_empty(self):
        o = Ontology.from_dict({})
        assert o == Ontology()


class TestDecision:
    """Tests for Decision."""

    def test_round_trip(self):
        d = Decision(
            question="Which DB?",
            options=["SQLite", "Postgres"],
            chosen="SQLite",
            rationale="Simpler for v1",
        )
        assert Decision.from_dict(d.to_dict()) == d


class TestDAGEdge:
    """Tests for DAGEdge."""

    def test_round_trip(self):
        e = DAGEdge(
            parent_id="n1",
            child_id="n2",
            decision=Decision(
                question="?",
                options=["a"],
                chosen="a",
                rationale="r",
            ),
            created_at="2026-03-31T10:00:00Z",
        )
        assert DAGEdge.from_dict(e.to_dict()) == e


class TestDAGNode:
    """Tests for DAGNode."""

    def test_round_trip(self):
        n = DAGNode(
            id="n1",
            ontology=Ontology(),
            created_at="2026-03-31T10:00:00Z",
            label="Initial",
        )
        assert DAGNode.from_dict(n.to_dict()) == n

    def test_default_label(self):
        n = DAGNode(
            id="n1",
            ontology=Ontology(),
            created_at="2026-03-31T10:00:00Z",
        )
        assert n.label == ""


class TestOntologyDAG:
    """Tests for OntologyDAG."""

    def _make_dag(self):
        """Build a 3-node DAG: root -> A, root -> B."""
        root = DAGNode(
            id="root",
            ontology=Ontology(),
            created_at="2026-01-01T00:00:00Z",
            label="root",
        )
        node_a = DAGNode(
            id="a",
            ontology=Ontology(
                entities=[Entity(id="e1", name="X")],
            ),
            created_at="2026-01-01T01:00:00Z",
            label="branch A",
        )
        node_b = DAGNode(
            id="b",
            ontology=Ontology(
                entities=[Entity(id="e2", name="Y")],
            ),
            created_at="2026-01-01T01:00:00Z",
            label="branch B",
        )
        decision = Decision(
            question="Which?",
            options=["A", "B"],
            chosen="A",
            rationale="reason",
        )
        edge_a = DAGEdge(
            parent_id="root",
            child_id="a",
            decision=decision,
            created_at="2026-01-01T01:00:00Z",
        )
        edge_b = DAGEdge(
            parent_id="root",
            child_id="b",
            decision=Decision(
                question="Which?",
                options=["A", "B"],
                chosen="B",
                rationale="alt",
            ),
            created_at="2026-01-01T01:00:00Z",
        )
        return OntologyDAG(
            project_name="test",
            nodes=[root, node_a, node_b],
            edges=[edge_a, edge_b],
            current_node_id="a",
        )

    def test_get_node(self):
        dag = self._make_dag()
        assert dag.get_node("root").label == "root"
        assert dag.get_node("missing") is None

    def test_get_current_node(self):
        dag = self._make_dag()
        assert dag.get_current_node().id == "a"

    def test_children_of(self):
        dag = self._make_dag()
        children = dag.children_of("root")
        ids = {n.id for n in children}
        assert ids == {"a", "b"}

    def test_children_of_leaf(self):
        dag = self._make_dag()
        assert dag.children_of("a") == []

    def test_parents_of(self):
        dag = self._make_dag()
        parents = dag.parents_of("a")
        assert len(parents) == 1
        assert parents[0].id == "root"

    def test_parents_of_root(self):
        dag = self._make_dag()
        assert dag.parents_of("root") == []

    def test_root_nodes(self):
        dag = self._make_dag()
        roots = dag.root_nodes()
        assert len(roots) == 1
        assert roots[0].id == "root"

    def test_edges_from(self):
        dag = self._make_dag()
        edges = dag.edges_from("root")
        assert len(edges) == 2

    def test_edges_from_leaf(self):
        dag = self._make_dag()
        assert dag.edges_from("a") == []

    def test_edges_to(self):
        dag = self._make_dag()
        edges = dag.edges_to("a")
        assert len(edges) == 1
        assert edges[0].parent_id == "root"

    def test_edges_to_root(self):
        dag = self._make_dag()
        assert dag.edges_to("root") == []

    def test_round_trip_dict(self):
        dag = self._make_dag()
        restored = OntologyDAG.from_dict(dag.to_dict())
        assert restored == dag

    def test_round_trip_json(self):
        dag = self._make_dag()
        restored = OntologyDAG.from_json(dag.to_json())
        assert restored == dag

    def test_empty_dag(self):
        dag = OntologyDAG(project_name="empty")
        assert dag.nodes == []
        assert dag.edges == []
        assert dag.root_nodes() == []
        assert dag.get_current_node() is None

    def test_single_node(self):
        node = DAGNode(
            id="n1",
            ontology=Ontology(),
            created_at="2026-01-01T00:00:00Z",
        )
        dag = OntologyDAG(
            project_name="solo",
            nodes=[node],
            current_node_id="n1",
        )
        assert dag.root_nodes() == [node]
        assert dag.children_of("n1") == []
        assert dag.parents_of("n1") == []


class TestValidateOntologyStrict:
    """Tests for validate_ontology_strict."""

    def test_valid_passes(self):
        data = {
            "entities": [{"id": "e1", "name": "User"}],
            "relationships": [{
                "source_entity_id": "e1",
                "target_entity_id": "e2",
                "name": "owns",
                "cardinality": "one_to_many",
            }],
            "modules": [{
                "name": "m", "responsibility": "r",
            }],
            "open_questions": [{"id": "q1", "text": "?"}],
        }
        assert validate_ontology_strict(data) == []

    def test_empty_passes(self):
        assert validate_ontology_strict({}) == []

    def test_missing_entity_id(self):
        data = {"entities": [{"name": "X"}]}
        errors = validate_ontology_strict(data)
        assert any("missing 'id'" in e for e in errors)

    def test_missing_entity_name(self):
        data = {"entities": [{"id": "e1"}]}
        errors = validate_ontology_strict(data)
        assert any("missing 'name'" in e for e in errors)

    def test_invalid_entity_id(self):
        data = {"entities": [
            {"id": "has spaces!", "name": "X"},
        ]}
        errors = validate_ontology_strict(data)
        assert any("invalid chars" in e for e in errors)

    def test_entity_id_too_long(self):
        data = {"entities": [
            {"id": "a" * 101, "name": "X"},
        ]}
        errors = validate_ontology_strict(data)
        assert any("too long" in e for e in errors)

    def test_entity_name_too_long(self):
        data = {"entities": [
            {"id": "e1", "name": "a" * 101},
        ]}
        errors = validate_ontology_strict(data)
        assert any(
            "name too long" in e.lower() for e in errors
        )

    def test_entity_description_too_long(self):
        data = {"entities": [{
            "id": "e1", "name": "X",
            "description": "a" * 2001,
        }]}
        errors = validate_ontology_strict(data)
        assert any(
            "description too long" in e.lower()
            for e in errors
        )

    def test_invalid_property_kind(self):
        data = {"entities": [{
            "id": "e1", "name": "X",
            "properties": [{
                "name": "p",
                "property_type": {"kind": "invalid"},
            }],
        }]}
        errors = validate_ontology_strict(data)
        assert any(
            "Invalid property kind" in e
            for e in errors
        )

    def test_valid_property_kinds(self):
        for kind in (
            "str", "int", "float", "bool",
            "datetime", "entity_ref", "list", "enum",
        ):
            data = {"entities": [{
                "id": "e1", "name": "X",
                "properties": [{
                    "name": "p",
                    "property_type": {"kind": kind},
                }],
            }]}
            assert validate_ontology_strict(data) == []

    def test_invalid_cardinality(self):
        data = {"relationships": [{
            "source_entity_id": "e1",
            "target_entity_id": "e2",
            "name": "r",
            "cardinality": "many_to_none",
        }]}
        errors = validate_ontology_strict(data)
        assert any("cardinality" in e for e in errors)

    def test_missing_relationship_fields(self):
        data = {"relationships": [{}]}
        errors = validate_ontology_strict(data)
        assert len(errors) == 4

    def test_invalid_module_status(self):
        data = {"modules": [{
            "name": "m", "responsibility": "r",
            "status": "deleted",
        }]}
        errors = validate_ontology_strict(data)
        assert any("status" in e for e in errors)

    def test_missing_module_fields(self):
        data = {"modules": [{}]}
        errors = validate_ontology_strict(data)
        assert any("missing" in e for e in errors)

    def test_invalid_priority(self):
        data = {"open_questions": [{
            "id": "q1", "text": "?",
            "priority": "urgent",
        }]}
        errors = validate_ontology_strict(data)
        assert any("priority" in e for e in errors)

    def test_missing_question_fields(self):
        data = {"open_questions": [{}]}
        errors = validate_ontology_strict(data)
        assert any("missing" in e for e in errors)
