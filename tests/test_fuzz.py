"""Fuzz tests for all functions that accept external inputs."""

import io
import sys
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, strategies as st

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
)

from python_agent.coding_agent import (
    parse_args as coding_parse_args,
    print_text_blocks as coding_print_text_blocks,
    remaining_budget,
    run_query,
    should_escalate,
)
from python_agent.planning_agent import (
    parse_args as planning_parse_args,
    print_response,
    print_text_blocks as planning_print_text_blocks,
    read_user_input,
)
from python_agent.rules import coding_system_prompt


# -- Strategies for SDK types --

def st_text_block():
    """Strategy for TextBlock instances."""
    return st.builds(TextBlock, text=st.text())


def st_content_block():
    """Strategy for arbitrary content blocks."""
    return st.one_of(st_text_block(), st.builds(object))


def st_assistant_message():
    """Strategy for AssistantMessage instances."""
    return st.builds(
        AssistantMessage,
        content=st.lists(st_content_block(), max_size=5),
        model=st.text(),
    )


def st_result_message():
    """Strategy for ResultMessage instances."""
    return st.builds(
        ResultMessage,
        subtype=st.text(),
        duration_ms=st.integers(min_value=0),
        duration_api_ms=st.integers(min_value=0),
        is_error=st.booleans(),
        num_turns=st.integers(min_value=0),
        session_id=st.text(),
        total_cost_usd=st.none() | st.floats(
            min_value=0, allow_nan=False,
            allow_infinity=False,
        ),
    )


def st_message():
    """Strategy for any SDK message type."""
    return st.one_of(
        st_assistant_message(), st_result_message(),
    )


def st_argv():
    """Strategy for CLI argument lists."""
    return st.lists(st.text(
        st.characters(blacklist_categories=("Cs",)),
    ))


def capture_stdout(func, *args, **kwargs):
    """Run func and return (result, stdout_text)."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old
    return result, buf.getvalue()


# -- Fuzz tests --


class TestCodingParseArgs:
    """Fuzz coding_agent.parse_args."""

    @given(argv=st_argv())
    def test_no_unhandled_exception(self, argv):
        try:
            result = coding_parse_args(argv)
            assert hasattr(result, "task")
            assert hasattr(result, "model")
        except SystemExit:
            pass


class TestPlanningParseArgs:
    """Fuzz planning_agent.parse_args."""

    @given(argv=st_argv())
    def test_no_unhandled_exception(self, argv):
        try:
            result = planning_parse_args(argv)
            assert hasattr(result, "description")
            assert hasattr(result, "model")
        except SystemExit:
            pass


class TestCodingPrintTextBlocks:
    """Fuzz coding_agent.print_text_blocks."""

    @given(message=st_assistant_message())
    def test_never_crashes(self, message):
        _, out = capture_stdout(
            coding_print_text_blocks, message,
        )
        for block in message.content:
            if isinstance(block, TextBlock):
                assert block.text in out


class TestPlanningPrintTextBlocks:
    """Fuzz planning_agent.print_text_blocks."""

    @given(message=st_assistant_message())
    def test_never_crashes(self, message):
        _, out = capture_stdout(
            planning_print_text_blocks, message,
        )
        for block in message.content:
            if isinstance(block, TextBlock):
                assert block.text in out


class TestShouldEscalate:
    """Fuzz should_escalate."""

    @given(
        is_error=st.booleans(),
        num_turns=st.integers(min_value=0),
        max_turns=st.none() | st.integers(min_value=0),
    )
    def test_returns_bool(self, is_error, num_turns, max_turns):
        result = MagicMock(
            is_error=is_error, num_turns=num_turns,
        )
        assert isinstance(
            should_escalate(result, max_turns), bool,
        )

    @given(max_turns=st.none() | st.integers(min_value=0))
    def test_none_result_returns_false(self, max_turns):
        assert should_escalate(None, max_turns) is False


class TestRemainingBudget:
    """Fuzz remaining_budget."""

    @given(
        cost=st.none() | st.floats(
            min_value=0, allow_nan=False,
            allow_infinity=False,
        ),
        max_budget=st.none() | st.floats(
            min_value=0, allow_nan=False,
            allow_infinity=False,
        ),
    )
    def test_invariants(self, cost, max_budget):
        result = MagicMock(total_cost_usd=cost)
        value = remaining_budget(result, max_budget)
        if max_budget is None:
            assert value is None
        else:
            assert isinstance(value, float)
            assert value <= max_budget

    @given(
        max_budget=st.none() | st.floats(
            min_value=0, allow_nan=False,
            allow_infinity=False,
        ),
    )
    def test_none_result(self, max_budget):
        value = remaining_budget(None, max_budget)
        if max_budget is None:
            assert value is None
        else:
            assert value == max_budget


class TestReadUserInput:
    """Fuzz read_user_input."""

    @given(user_input=st.text())
    def test_returns_string_or_none(self, user_input):
        with patch("builtins.input", return_value=user_input):
            result = read_user_input()
        quit_words = ("quit", "exit", "done")
        if user_input.strip().lower() in quit_words:
            assert result is None
        else:
            assert result == user_input


class TestRunQuery:
    """Fuzz run_query."""

    @given(messages=st.lists(st_message(), max_size=5))
    @pytest.mark.asyncio
    async def test_never_crashes(self, messages):
        async def fake_query(**kwargs):
            for msg in messages:
                yield msg

        with patch(
            "python_agent.coding_agent.query", fake_query,
        ):
            result = await run_query("task", MagicMock())

        if result is not None:
            assert isinstance(result, ResultMessage)


class TestPrintResponse:
    """Fuzz print_response."""

    @given(messages=st.lists(
        st_assistant_message(), max_size=5,
    ))
    @pytest.mark.asyncio
    async def test_never_crashes(self, messages):
        client = MagicMock()

        async def fake_receive():
            for msg in messages:
                yield msg

        client.receive_response = fake_receive
        await print_response(client)


class TestCodingSystemPrompt:
    """Fuzz coding_system_prompt."""

    @given(project_dir=st.text())
    def test_returns_string_containing_dir(self, project_dir):
        result = coding_system_prompt(project_dir)
        assert isinstance(result, str)
        assert project_dir in result


# -- Ontology round-trip fuzz tests --

from python_agent.ontology import (  # noqa: E402
    DAGEdge,
    DAGNode,
    Decision,
    Entity,
    FunctionSpec as OntFunctionSpec,
    ModuleSpec as OntModuleSpec,
    Ontology,
    OntologyDAG,
    Property,
    PropertyType,
)


def st_property_type():
    """Strategy for PropertyType."""
    return st.builds(
        PropertyType,
        kind=st.sampled_from([
            "str", "int", "float", "bool", "datetime",
            "entity_ref", "list", "enum",
        ]),
        reference=st.none() | st.text(max_size=20),
    )


def st_property():
    """Strategy for Property."""
    return st.builds(
        Property,
        name=st.text(min_size=1, max_size=20),
        property_type=st_property_type(),
        description=st.text(max_size=50),
        required=st.booleans(),
        constraints=st.lists(
            st.text(max_size=20), max_size=3,
        ),
    )


_SAFE_ID = st.from_regex(
    r"[a-zA-Z0-9_-]{1,20}", fullmatch=True,
)


def st_entity():
    """Strategy for Entity."""
    return st.builds(
        Entity,
        id=_SAFE_ID,
        name=st.text(min_size=1, max_size=20),
        description=st.text(max_size=50),
        properties=st.lists(st_property(), max_size=3),
    )


def st_function_spec():
    """Strategy for FunctionSpec."""
    return st.builds(
        OntFunctionSpec,
        name=st.text(min_size=1, max_size=20),
        parameters=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=10),
                st.text(min_size=1, max_size=10),
            ),
            max_size=3,
        ),
        return_type=st.text(min_size=1, max_size=10),
        docstring=st.text(max_size=50),
        preconditions=st.lists(
            st.text(max_size=20), max_size=2,
        ),
        postconditions=st.lists(
            st.text(max_size=20), max_size=2,
        ),
    )


def st_ontology():
    """Strategy for Ontology."""
    return st.builds(
        Ontology,
        entities=st.lists(st_entity(), max_size=2),
        modules=st.lists(
            st.builds(
                OntModuleSpec,
                name=st.text(min_size=1, max_size=20),
                responsibility=st.text(max_size=30),
            ),
            max_size=2,
        ),
    )


def st_decision():
    """Strategy for Decision."""
    return st.builds(
        Decision,
        question=st.text(min_size=1, max_size=50),
        options=st.lists(
            st.text(min_size=1, max_size=20),
            min_size=1, max_size=3,
        ),
        chosen=st.text(min_size=1, max_size=20),
        rationale=st.text(max_size=50),
    )


class TestOntologyRoundTrip:
    """Fuzz ontology serialization round-trips."""

    @given(pt=st_property_type())
    def test_property_type(self, pt):
        assert PropertyType.model_validate(pt.model_dump()) == pt

    @given(p=st_property())
    def test_property(self, p):
        assert Property.model_validate(p.model_dump()) == p

    @given(e=st_entity())
    def test_entity(self, e):
        assert Entity.model_validate(e.model_dump()) == e

    @given(f=st_function_spec())
    def test_function_spec(self, f):
        result = OntFunctionSpec.model_validate(f.model_dump())
        assert result == f

    @given(o=st_ontology())
    def test_ontology(self, o):
        assert Ontology.model_validate(o.model_dump()) == o

    @given(d=st_decision())
    def test_decision(self, d):
        assert Decision.model_validate(d.model_dump()) == d

    @given(
        o=st_ontology(),
        label=st.text(max_size=20),
    )
    def test_dag_node(self, o, label):
        node = DAGNode(
            id="n1",
            ontology=o,
            created_at="2026-01-01T00:00:00Z",
            label=label,
        )
        assert DAGNode.model_validate(node.model_dump()) == node

    @given(d=st_decision())
    def test_dag_edge(self, d):
        edge = DAGEdge(
            parent_id="p",
            child_id="c",
            decision=d,
            created_at="2026-01-01T00:00:00Z",
        )
        assert DAGEdge.model_validate(edge.model_dump()) == edge

    @given(o=st_ontology())
    def test_ontology_dag_json(self, o):
        node = DAGNode(
            id="n1",
            ontology=o,
            created_at="2026-01-01T00:00:00Z",
        )
        dag = OntologyDAG(
            project_name="test",
            nodes=[node],
            current_node_id="n1",
        )
        restored = OntologyDAG.from_json(dag.to_json())
        assert restored == dag
