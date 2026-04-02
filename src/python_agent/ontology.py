"""Ontology schema and DAG for project planning."""

import json
import re
from dataclasses import dataclass, field

# -- Validation constants --

VALID_PROPERTY_KINDS = frozenset({
    "str", "int", "float", "bool", "datetime",
    "entity_ref", "list", "enum",
})

VALID_CARDINALITIES = frozenset({
    "one_to_one", "one_to_many",
    "many_to_one", "many_to_many",
})

VALID_MODULE_STATUSES = frozenset({
    "not_started", "in_progress", "complete",
})

VALID_PRIORITIES = frozenset({
    "low", "medium", "high",
})

SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

MAX_NAME_LENGTH = 100
MAX_DESCRIPTION_LENGTH = 2000
MAX_ID_LENGTH = 100


def _list_to_dicts(items):
    """Serialize a list of dataclass instances to dicts."""
    return [item.to_dict() for item in items]


def _list_from_dicts(cls, items):
    """Deserialize a list of dicts to dataclass instances."""
    return [cls.from_dict(d) for d in items]


# -- Problem Domain --


@dataclass
class PropertyType:
    """Type descriptor for an entity property.

    kind: "str", "int", "float", "bool", "datetime",
          "entity_ref", "list", or "enum".
    reference: For entity_ref: target entity ID.
               For list: inner type kind string.
               For enum: list of allowed values.
               None for scalar kinds.
    """

    kind: str
    reference: str | list[str] | None = None

    def to_dict(self):
        """Serialize to dict."""
        return {
            "kind": self.kind,
            "reference": self.reference,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        return cls(
            kind=data["kind"],
            reference=data.get("reference"),
        )


@dataclass
class Property:
    """A named, typed property on a domain entity."""

    name: str
    property_type: PropertyType
    description: str = ""
    required: bool = True
    constraints: list[str] = field(default_factory=list)

    def to_dict(self):
        """Serialize to dict."""
        return {
            "name": self.name,
            "property_type": self.property_type.to_dict(),
            "description": self.description,
            "required": self.required,
            "constraints": list(self.constraints),
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        return cls(
            name=data["name"],
            property_type=PropertyType.from_dict(
                data["property_type"],
            ),
            description=data.get("description", ""),
            required=data.get("required", True),
            constraints=data.get("constraints", []),
        )


@dataclass
class Entity:
    """A business concept in the problem domain."""

    id: str
    name: str
    description: str = ""
    properties: list[Property] = field(default_factory=list)

    def to_dict(self):
        """Serialize to dict."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "properties": _list_to_dicts(self.properties),
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            properties=_list_from_dicts(
                Property, data.get("properties", []),
            ),
        )


@dataclass
class Relationship:
    """A directed relationship between two domain entities."""

    source_entity_id: str
    target_entity_id: str
    name: str
    cardinality: str
    description: str = ""

    def to_dict(self):
        """Serialize to dict."""
        return {
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "name": self.name,
            "cardinality": self.cardinality,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        return cls(
            source_entity_id=data["source_entity_id"],
            target_entity_id=data["target_entity_id"],
            name=data["name"],
            cardinality=data["cardinality"],
            description=data.get("description", ""),
        )


@dataclass
class DomainConstraint:
    """A domain-level invariant or business rule."""

    name: str
    description: str
    entity_ids: list[str] = field(default_factory=list)
    expression: str = ""

    def to_dict(self):
        """Serialize to dict."""
        return {
            "name": self.name,
            "description": self.description,
            "entity_ids": list(self.entity_ids),
            "expression": self.expression,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        return cls(
            name=data["name"],
            description=data["description"],
            entity_ids=data.get("entity_ids", []),
            expression=data.get("expression", ""),
        )


# -- Solution Domain --


@dataclass
class FunctionSpec:
    """Specification for a function to be implemented."""

    name: str
    parameters: list[tuple[str, str]]
    return_type: str
    docstring: str = ""
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)

    def to_dict(self):
        """Serialize to dict."""
        return {
            "name": self.name,
            "parameters": [list(p) for p in self.parameters],
            "return_type": self.return_type,
            "docstring": self.docstring,
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        return cls(
            name=data["name"],
            parameters=[
                tuple(p) for p in data.get("parameters", [])
            ],
            return_type=data["return_type"],
            docstring=data.get("docstring", ""),
            preconditions=data.get("preconditions", []),
            postconditions=data.get("postconditions", []),
        )


@dataclass
class ClassSpec:
    """Specification for a class to be implemented."""

    name: str
    description: str = ""
    bases: list[str] = field(default_factory=list)
    methods: list[FunctionSpec] = field(default_factory=list)

    def to_dict(self):
        """Serialize to dict."""
        return {
            "name": self.name,
            "description": self.description,
            "bases": list(self.bases),
            "methods": _list_to_dicts(self.methods),
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            bases=data.get("bases", []),
            methods=_list_from_dicts(
                FunctionSpec, data.get("methods", []),
            ),
        )


@dataclass
class DataModel:
    """Maps a problem-domain entity to a code construct."""

    entity_id: str
    storage: str
    class_name: str
    notes: str = ""

    def to_dict(self):
        """Serialize to dict."""
        return {
            "entity_id": self.entity_id,
            "storage": self.storage,
            "class_name": self.class_name,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        return cls(
            entity_id=data["entity_id"],
            storage=data["storage"],
            class_name=data["class_name"],
            notes=data.get("notes", ""),
        )


@dataclass
class ExternalDependency:
    """An external package dependency."""

    name: str
    version_constraint: str = ""
    reason: str = ""

    def to_dict(self):
        """Serialize to dict."""
        return {
            "name": self.name,
            "version_constraint": self.version_constraint,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        return cls(
            name=data["name"],
            version_constraint=data.get(
                "version_constraint", "",
            ),
            reason=data.get("reason", ""),
        )


@dataclass
class ModuleSpec:
    """Specification for a Python module."""

    name: str
    responsibility: str
    classes: list[ClassSpec] = field(default_factory=list)
    functions: list[FunctionSpec] = field(
        default_factory=list,
    )
    dependencies: list[str] = field(default_factory=list)
    test_strategy: str = ""
    status: str = "not_started"

    def to_dict(self):
        """Serialize to dict."""
        return {
            "name": self.name,
            "responsibility": self.responsibility,
            "classes": _list_to_dicts(self.classes),
            "functions": _list_to_dicts(self.functions),
            "dependencies": list(self.dependencies),
            "test_strategy": self.test_strategy,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        return cls(
            name=data["name"],
            responsibility=data["responsibility"],
            classes=_list_from_dicts(
                ClassSpec, data.get("classes", []),
            ),
            functions=_list_from_dicts(
                FunctionSpec, data.get("functions", []),
            ),
            dependencies=data.get("dependencies", []),
            test_strategy=data.get("test_strategy", ""),
            status=data.get("status", "not_started"),
        )


# -- Planning State --


@dataclass
class OpenQuestion:
    """An unresolved design question."""

    id: str
    text: str
    context: str = ""
    priority: str = "medium"
    resolved: bool = False
    resolution: str = ""

    def to_dict(self):
        """Serialize to dict."""
        return {
            "id": self.id,
            "text": self.text,
            "context": self.context,
            "priority": self.priority,
            "resolved": self.resolved,
            "resolution": self.resolution,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        return cls(
            id=data["id"],
            text=data["text"],
            context=data.get("context", ""),
            priority=data.get("priority", "medium"),
            resolved=data.get("resolved", False),
            resolution=data.get("resolution", ""),
        )


@dataclass
class Ontology:
    """Complete ontology snapshot: problem + solution domains."""

    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(
        default_factory=list,
    )
    domain_constraints: list[DomainConstraint] = field(
        default_factory=list,
    )
    modules: list[ModuleSpec] = field(default_factory=list)
    data_models: list[DataModel] = field(
        default_factory=list,
    )
    external_dependencies: list[ExternalDependency] = field(
        default_factory=list,
    )
    open_questions: list[OpenQuestion] = field(
        default_factory=list,
    )

    def to_dict(self):
        """Serialize to dict."""
        return {
            "entities": _list_to_dicts(self.entities),
            "relationships": _list_to_dicts(
                self.relationships,
            ),
            "domain_constraints": _list_to_dicts(
                self.domain_constraints,
            ),
            "modules": _list_to_dicts(self.modules),
            "data_models": _list_to_dicts(self.data_models),
            "external_dependencies": _list_to_dicts(
                self.external_dependencies,
            ),
            "open_questions": _list_to_dicts(
                self.open_questions,
            ),
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        return cls(
            entities=_list_from_dicts(
                Entity, data.get("entities", []),
            ),
            relationships=_list_from_dicts(
                Relationship, data.get("relationships", []),
            ),
            domain_constraints=_list_from_dicts(
                DomainConstraint,
                data.get("domain_constraints", []),
            ),
            modules=_list_from_dicts(
                ModuleSpec, data.get("modules", []),
            ),
            data_models=_list_from_dicts(
                DataModel, data.get("data_models", []),
            ),
            external_dependencies=_list_from_dicts(
                ExternalDependency,
                data.get("external_dependencies", []),
            ),
            open_questions=_list_from_dicts(
                OpenQuestion,
                data.get("open_questions", []),
            ),
        )


# -- DAG Structure --


@dataclass
class Decision:
    """Records a design decision that produced a new node."""

    question: str
    options: list[str]
    chosen: str
    rationale: str

    def to_dict(self):
        """Serialize to dict."""
        return {
            "question": self.question,
            "options": list(self.options),
            "chosen": self.chosen,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        return cls(
            question=data["question"],
            options=data["options"],
            chosen=data["chosen"],
            rationale=data["rationale"],
        )


@dataclass
class DAGEdge:
    """An edge in the version DAG."""

    parent_id: str
    child_id: str
    decision: Decision
    created_at: str

    def to_dict(self):
        """Serialize to dict."""
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "decision": self.decision.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        return cls(
            parent_id=data["parent_id"],
            child_id=data["child_id"],
            decision=Decision.from_dict(data["decision"]),
            created_at=data["created_at"],
        )


@dataclass
class DAGNode:
    """A node in the version DAG with an ontology snapshot."""

    id: str
    ontology: Ontology
    created_at: str
    label: str = ""
    integrity_hash: str = ""

    def to_dict(self):
        """Serialize to dict."""
        return {
            "id": self.id,
            "ontology": self.ontology.to_dict(),
            "created_at": self.created_at,
            "label": self.label,
            "integrity_hash": self.integrity_hash,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        return cls(
            id=data["id"],
            ontology=Ontology.from_dict(data["ontology"]),
            created_at=data["created_at"],
            label=data.get("label", ""),
            integrity_hash=data.get(
                "integrity_hash", "",
            ),
        )


@dataclass
class OntologyDAG:
    """Versioned ontology DAG. Serializes to one JSON file."""

    project_name: str
    nodes: list[DAGNode] = field(default_factory=list)
    edges: list[DAGEdge] = field(default_factory=list)
    current_node_id: str = ""

    # -- Navigation --

    def get_node(self, node_id):
        """Find a node by ID. Returns None if not found."""
        return next(
            (n for n in self.nodes if n.id == node_id),
            None,
        )

    def get_current_node(self):
        """Return the currently active node."""
        return self.get_node(self.current_node_id)

    def children_of(self, node_id):
        """Return all child nodes of the given node."""
        child_ids = {
            e.child_id
            for e in self.edges
            if e.parent_id == node_id
        }
        return [n for n in self.nodes if n.id in child_ids]

    def parents_of(self, node_id):
        """Return all parent nodes of the given node."""
        parent_ids = {
            e.parent_id
            for e in self.edges
            if e.child_id == node_id
        }
        return [n for n in self.nodes if n.id in parent_ids]

    def root_nodes(self):
        """Return all nodes with no parents."""
        child_ids = {e.child_id for e in self.edges}
        return [
            n for n in self.nodes if n.id not in child_ids
        ]

    def edges_from(self, node_id):
        """Return all edges originating from the given node."""
        return [
            e for e in self.edges if e.parent_id == node_id
        ]

    def edges_to(self, node_id):
        """Return all edges pointing to the given node."""
        return [
            e for e in self.edges if e.child_id == node_id
        ]

    # -- Serialization --

    def to_dict(self):
        """Serialize to dict."""
        return {
            "project_name": self.project_name,
            "nodes": _list_to_dicts(self.nodes),
            "edges": _list_to_dicts(self.edges),
            "current_node_id": self.current_node_id,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict."""
        return cls(
            project_name=data["project_name"],
            nodes=_list_from_dicts(
                DAGNode, data.get("nodes", []),
            ),
            edges=_list_from_dicts(
                DAGEdge, data.get("edges", []),
            ),
            current_node_id=data.get("current_node_id", ""),
        )

    def to_json(self):
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text):
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(text))


# -- Validation --


def _validate_id(entity_id):
    """Validate an ID string. Returns list of errors."""
    errors = []
    if len(entity_id) > MAX_ID_LENGTH:
        errors.append(f"ID too long: {len(entity_id)}")
    if not SAFE_ID_PATTERN.match(entity_id):
        errors.append(f"ID has invalid chars: {entity_id!r}")
    return errors


def _validate_entity_properties(data):
    """Validate properties within an Entity dict."""
    errors = []
    for p in data.get("properties", []):
        pt = p.get("property_type", {})
        kind = pt.get("kind", "")
        if kind and kind not in VALID_PROPERTY_KINDS:
            errors.append(
                f"Invalid property kind: {kind!r}",
            )
    return errors


def _validate_string_length(value, label, max_len):
    """Validate a string field length."""
    if len(value) > max_len:
        return [f"{label} too long"]
    return []


def _validate_entity(data):
    """Validate an Entity dict. Returns list of errors."""
    errors = []
    for field_name in ("id", "name"):
        if field_name not in data:
            errors.append(f"Entity missing '{field_name}'")
    if "id" in data:
        errors += _validate_id(data["id"])
    if "name" in data:
        errors += _validate_string_length(
            data["name"], "Entity name", MAX_NAME_LENGTH,
        )
    errors += _validate_string_length(
        data.get("description", ""),
        "Entity description", MAX_DESCRIPTION_LENGTH,
    )
    errors += _validate_entity_properties(data)
    return errors


def _validate_relationship(data):
    """Validate a Relationship dict."""
    errors = []
    for field_name in (
        "source_entity_id", "target_entity_id",
        "name", "cardinality",
    ):
        if field_name not in data:
            errors.append(
                f"Relationship missing '{field_name}'",
            )
    card = data.get("cardinality", "")
    if card and card not in VALID_CARDINALITIES:
        errors.append(f"Invalid cardinality: {card!r}")
    return errors


def _validate_module(data):
    """Validate a ModuleSpec dict."""
    errors = []
    for field_name in ("name", "responsibility"):
        if field_name not in data:
            errors.append(
                f"ModuleSpec missing '{field_name}'",
            )
    status = data.get("status", "not_started")
    if status not in VALID_MODULE_STATUSES:
        errors.append(f"Invalid status: {status!r}")
    return errors


def _validate_open_question(data):
    """Validate an OpenQuestion dict."""
    errors = []
    for field_name in ("id", "text"):
        if field_name not in data:
            errors.append(
                f"OpenQuestion missing '{field_name}'",
            )
    priority = data.get("priority", "medium")
    if priority not in VALID_PRIORITIES:
        errors.append(f"Invalid priority: {priority!r}")
    if "id" in data:
        errors += _validate_id(data["id"])
    return errors


def validate_ontology_strict(data):
    """Validate ontology data from external input.

    Returns list of error strings, empty if valid.
    """
    errors = []
    for entity in data.get("entities", []):
        errors += _validate_entity(entity)
    for rel in data.get("relationships", []):
        errors += _validate_relationship(rel)
    for mod in data.get("modules", []):
        errors += _validate_module(mod)
    for q in data.get("open_questions", []):
        errors += _validate_open_question(q)
    return errors
