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
