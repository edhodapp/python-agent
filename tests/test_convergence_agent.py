"""Tests for convergence_agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from python_agent.convergence_agent import (
    AgentState,
    _handle_accept_cmd,
    _handle_back_cmd,
    _handle_list_cmd,
    _handle_save_cmd,
    _handle_select_cmd,
    _handle_show_cmd,
    _init_state,
    build_query,
    dispatch_command,
    format_children_list,
    get_children_summaries,
    handle_command,
    is_command,
    main,
    maybe_process,
    navigate_to_node,
    parse_args,
    run,
)
from python_agent.dag_utils import save_dag
from python_agent.ontology import (
    DAGEdge,
    DAGNode,
    Decision,
    Entity,
    Ontology,
    OntologyDAG,
)


def _make_branching_dag():
    """Build a DAG: root -> child_a, root -> child_b."""
    root_onto = Ontology(
        entities=[Entity(id="e1", name="Root")],
    )
    child_a_onto = Ontology(
        entities=[Entity(id="e1", name="A")],
    )
    child_b_onto = Ontology(
        entities=[Entity(id="e1", name="B")],
    )
    dec = Decision(
        question="?", options=["a", "b"],
        chosen="a", rationale="r",
    )
    return OntologyDAG(
        project_name="test",
        nodes=[
            DAGNode(
                id="root", ontology=root_onto,
                created_at="t", label="root",
            ),
            DAGNode(
                id="child_a", ontology=child_a_onto,
                created_at="t", label="approach-a",
            ),
            DAGNode(
                id="child_b", ontology=child_b_onto,
                created_at="t", label="approach-b",
            ),
        ],
        edges=[
            DAGEdge(
                parent_id="root", child_id="child_a",
                decision=dec, created_at="t",
            ),
            DAGEdge(
                parent_id="root", child_id="child_b",
                decision=Decision(
                    question="?", options=["a", "b"],
                    chosen="b", rationale="r",
                ),
                created_at="t",
            ),
        ],
        current_node_id="root",
    )


class TestGetChildrenSummaries:
    """Tests for get_children_summaries."""

    def test_returns_summaries(self):
        dag = _make_branching_dag()
        result = get_children_summaries(dag, "root")
        assert len(result) == 2
        assert result[0][0] == 1
        assert result[0][1].id == "child_a"
        assert "Entities" in result[0][2]

    def test_no_children(self):
        dag = _make_branching_dag()
        result = get_children_summaries(dag, "child_a")
        assert result == []


class TestFormatChildrenList:
    """Tests for format_children_list."""

    def test_empty(self):
        result = format_children_list([])
        assert "No candidates" in result

    def test_populated(self):
        dag = _make_branching_dag()
        summaries = get_children_summaries(dag, "root")
        result = format_children_list(summaries)
        assert "1. approach-a" in result
        assert "2. approach-b" in result


class TestNavigateToNode:
    """Tests for navigate_to_node."""

    def test_valid_node(self):
        dag = _make_branching_dag()
        onto = navigate_to_node(dag, "child_a")
        assert onto is not None
        assert onto.entities[0].name == "A"
        assert dag.current_node_id == "child_a"

    def test_invalid_node(self):
        dag = _make_branching_dag()
        result = navigate_to_node(dag, "missing")
        assert result is None


class TestIsCommand:
    """Tests for is_command."""

    def test_list(self):
        assert is_command("list") is True

    def test_show(self):
        assert is_command("show") is True

    def test_accept(self):
        assert is_command("accept") is True

    def test_back(self):
        assert is_command("back") is True

    def test_select(self):
        assert is_command("select 1") is True

    def test_save(self):
        assert is_command("save") is True

    def test_save_with_label(self):
        assert is_command("save my label") is True

    def test_not_command(self):
        assert is_command("compare them") is False

    def test_case_insensitive(self):
        assert is_command("LIST") is True


class TestHandleListCmd:
    """Tests for _handle_list_cmd."""

    def test_returns_children(self):
        dag = _make_branching_dag()
        msg, onto, accept = _handle_list_cmd(
            "list", Ontology(), dag, "/tmp/x",
        )
        assert "approach-a" in msg
        assert onto is None
        assert accept is False


class TestHandleShowCmd:
    """Tests for _handle_show_cmd."""

    def test_returns_summary(self):
        o = Ontology(
            entities=[Entity(id="e1", name="X")],
        )
        msg, onto, accept = _handle_show_cmd(
            "show", o, MagicMock(), "/tmp/x",
        )
        assert "e1: X" in msg
        assert onto is None
        assert accept is False


class TestHandleAcceptCmd:
    """Tests for _handle_accept_cmd."""

    def test_creates_snapshot(self, tmp_path):
        path = str(tmp_path / "dag.json")
        dag = _make_branching_dag()
        navigate_to_node(dag, "child_a")
        o = Ontology()
        msg, onto, accept = _handle_accept_cmd(
            "accept", o, dag, path,
        )
        assert "Accepted" in msg
        assert "approach-a" in msg
        assert accept is True
        assert onto is None


class TestHandleBackCmd:
    """Tests for _handle_back_cmd."""

    def test_at_root(self):
        dag = _make_branching_dag()
        msg, onto, accept = _handle_back_cmd(
            "back", Ontology(), dag, "/dev/null",
        )
        assert "root" in msg.lower()
        assert onto is None
        assert accept is False

    def test_from_child(self):
        dag = _make_branching_dag()
        dag.current_node_id = "child_a"
        msg, onto, accept = _handle_back_cmd(
            "back", Ontology(), dag, "/dev/null",
        )
        assert "root" in msg
        assert onto is not None
        assert onto.entities[0].name == "Root"
        assert accept is False


class TestHandleSelectCmd:
    """Tests for _handle_select_cmd."""

    def test_valid_select(self):
        dag = _make_branching_dag()
        msg, onto, accept = _handle_select_cmd(
            "select 1", Ontology(), dag, "/tmp/x",
        )
        assert "approach-a" in msg
        assert onto is not None
        assert onto.entities[0].name == "A"
        assert accept is False

    def test_missing_number(self):
        dag = _make_branching_dag()
        msg, onto, _ = _handle_select_cmd(
            "select", Ontology(), dag, "/tmp/x",
        )
        assert "Usage" in msg
        assert onto is None

    def test_invalid_number(self):
        dag = _make_branching_dag()
        msg, onto, _ = _handle_select_cmd(
            "select abc", Ontology(), dag, "/tmp/x",
        )
        assert "Invalid" in msg
        assert onto is None

    def test_out_of_range(self):
        dag = _make_branching_dag()
        msg, onto, _ = _handle_select_cmd(
            "select 99", Ontology(), dag, "/tmp/x",
        )
        assert "Range" in msg
        assert onto is None

    def test_zero_index(self):
        dag = _make_branching_dag()
        msg, onto, _ = _handle_select_cmd(
            "select 0", Ontology(), dag, "/tmp/x",
        )
        assert "Range" in msg
        assert onto is None


class TestHandleSaveCmd:
    """Tests for _handle_save_cmd."""

    def test_default_label(self, tmp_path):
        path = str(tmp_path / "dag.json")
        dag = OntologyDAG(project_name="p")
        msg, onto, accept = _handle_save_cmd(
            "save", Ontology(), dag, path,
        )
        assert "Saved" in msg
        assert onto is None
        assert accept is False

    def test_custom_label(self, tmp_path):
        path = str(tmp_path / "dag.json")
        dag = OntologyDAG(project_name="p")
        _handle_save_cmd(
            "save my label", Ontology(), dag, path,
        )
        assert dag.nodes[0].label == "my label"


class TestHandleCommand:
    """Tests for handle_command."""

    def test_dispatches_list(self):
        dag = _make_branching_dag()
        msg, _, _ = handle_command(
            "list", Ontology(), dag, "/tmp/x",
        )
        assert "approach-a" in msg

    def test_dispatches_select(self):
        dag = _make_branching_dag()
        msg, onto, _ = handle_command(
            "select 2", Ontology(), dag, "/tmp/x",
        )
        assert "approach-b" in msg
        assert onto is not None

    def test_dispatches_save(self, tmp_path):
        path = str(tmp_path / "dag.json")
        dag = OntologyDAG(project_name="p")
        msg, _, _ = handle_command(
            "save", Ontology(), dag, path,
        )
        assert "Saved" in msg


class TestDispatchCommand:
    """Tests for dispatch_command."""

    def test_updates_ontology(self, capsys):
        dag = _make_branching_dag()
        state = AgentState(ontology=Ontology())
        dispatch_command("select 1", state, dag, "/tmp/x")
        assert state.ontology.entities[0].name == "A"

    def test_sets_accepted(self, tmp_path):
        path = str(tmp_path / "dag.json")
        dag = _make_branching_dag()
        navigate_to_node(dag, "child_a")
        state = AgentState(ontology=Ontology())
        dispatch_command("accept", state, dag, path)
        assert state.accepted is True


class TestMaybeProcess:
    """Tests for maybe_process."""

    def test_processes_when_accepted(self):
        state = AgentState(
            ontology=Ontology(), accepted=True,
        )
        text = (
            '```ontology\n'
            '{"entities": [{"id": "e1", "name": "New"}]}'
            '\n```'
        )
        maybe_process(text, state)
        assert len(state.ontology.entities) == 1

    def test_skips_when_not_accepted(self):
        state = AgentState(
            ontology=Ontology(), accepted=False,
        )
        text = (
            '```ontology\n'
            '{"entities": [{"id": "e1", "name": "New"}]}'
            '\n```'
        )
        maybe_process(text, state)
        assert len(state.ontology.entities) == 0


class TestBuildQuery:
    """Tests for build_query."""

    def test_includes_context(self):
        dag = _make_branching_dag()
        state = AgentState(ontology=Ontology())
        result = build_query("compare them", state, dag)
        assert "compare them" in result
        assert "root" in result
        assert "approach-a" in result


class TestInitState:
    """Tests for _init_state."""

    def test_no_current_node(self):
        dag = OntologyDAG(project_name="p")
        assert _init_state(dag) is None

    def test_with_node(self):
        dag = _make_branching_dag()
        state = _init_state(dag)
        assert state is not None
        assert state.ontology.entities[0].name == "Root"
        assert state.accepted is False


class TestRun:
    """Tests for run."""

    @pytest.mark.asyncio
    async def test_no_current_node(self, tmp_path, capsys):
        path = str(tmp_path / "dag.json")
        dag = OntologyDAG(project_name="p")
        save_dag(dag, path)
        await run(path, "model")
        out = capsys.readouterr().out
        assert "no current node" in out.lower()

    @pytest.mark.asyncio
    async def test_starts_and_quits(self, tmp_path, capsys):
        path = str(tmp_path / "dag.json")
        dag = _make_branching_dag()
        save_dag(dag, path)

        client = AsyncMock()

        async def fake_receive():
            return
            yield  # noqa: E501

        client.receive_response = fake_receive

        with (
            patch(
                "python_agent.convergence_agent."
                "ClaudeSDKClient",
            ) as mock_cls,
            patch(
                "python_agent.convergence_agent."
                "read_user_input",
                return_value=None,
            ),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(
                return_value=client,
            )
            mock_cls.return_value.__aexit__ = AsyncMock(
                return_value=False,
            )
            await run(path, "claude-opus-4-6")

        out = capsys.readouterr().out
        assert "Entities" in out

    @pytest.mark.asyncio
    async def test_handles_command(self, tmp_path, capsys):
        path = str(tmp_path / "dag.json")
        dag = _make_branching_dag()
        save_dag(dag, path)

        client = AsyncMock()

        async def fake_receive():
            return
            yield  # noqa: E501

        client.receive_response = fake_receive
        call_count = 0

        def fake_input():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "list"
            return None

        with (
            patch(
                "python_agent.convergence_agent."
                "ClaudeSDKClient",
            ) as mock_cls,
            patch(
                "python_agent.convergence_agent."
                "read_user_input",
                side_effect=fake_input,
            ),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(
                return_value=client,
            )
            mock_cls.return_value.__aexit__ = AsyncMock(
                return_value=False,
            )
            await run(path, "claude-opus-4-6")

        out = capsys.readouterr().out
        assert "approach-a" in out

    @pytest.mark.asyncio
    async def test_sends_llm_query(self, tmp_path):
        path = str(tmp_path / "dag.json")
        dag = _make_branching_dag()
        save_dag(dag, path)

        client = AsyncMock()

        async def fake_receive():
            return
            yield  # noqa: E501

        client.receive_response = fake_receive
        call_count = 0

        def fake_input():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "compare approaches"
            return None

        with (
            patch(
                "python_agent.convergence_agent."
                "ClaudeSDKClient",
            ) as mock_cls,
            patch(
                "python_agent.convergence_agent."
                "read_user_input",
                side_effect=fake_input,
            ),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(
                return_value=client,
            )
            mock_cls.return_value.__aexit__ = AsyncMock(
                return_value=False,
            )
            await run(path, "claude-opus-4-6")

        assert client.query.call_count >= 1
        query_text = client.query.call_args[0][0]
        assert "compare approaches" in query_text


class TestParseArgs:
    """Tests for parse_args."""

    def test_dag_file_required(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_dag_file(self):
        args = parse_args(["--dag-file", "x.json"])
        assert args.dag_file == "x.json"

    def test_default_model(self):
        args = parse_args(["--dag-file", "x.json"])
        assert args.model == "claude-opus-4-6"

    def test_custom_model(self):
        args = parse_args(
            ["--dag-file", "x.json", "-m", "sonnet"],
        )
        assert args.model == "sonnet"

    def test_help_text(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(["--help"])
        out = capsys.readouterr().out
        assert "XX" not in out
        assert "candidate convergence" in out
        assert "DAG JSON file" in out


class TestMain:
    """Tests for main."""

    @patch("python_agent.convergence_agent.asyncio.run")
    def test_returns_zero(self, mock_run):
        assert main(["--dag-file", "x.json"]) == 0

    @patch("python_agent.convergence_agent.asyncio.run")
    def test_calls_asyncio_run(self, mock_run):
        main(["--dag-file", "x.json"])
        mock_run.assert_called_once()
