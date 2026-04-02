"""Tests for discovery_agent module."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import TextBlock

from python_agent.dag_utils import (
    load_dag,
    save_dag,
    save_snapshot,
)
from python_agent.discovery_agent import (
    _handle_back,
    _handle_save,
    _init_ontology,
    backtrack,
    collect_response_text,
    extract_ontology_json,
    format_ontology_summary,
    handle_command,
    is_command,
    main,
    merge_ontology_update,
    parse_args,
    print_text_blocks,
    process_response,
    read_user_input,
    run,
)
from python_agent.ontology import (
    DAGEdge,
    DAGNode,
    Decision,
    DomainConstraint,
    Entity,
    Ontology,
    OntologyDAG,
    OpenQuestion,
    Property,
    PropertyType,
    Relationship,
)


class TestCollectResponseText:
    """Tests for collect_response_text."""

    def test_extracts_text_blocks(self):
        msg = MagicMock()
        msg.content = [
            TextBlock(text="hello"),
            TextBlock(text="world"),
        ]
        assert collect_response_text(msg) == "hello\nworld"

    def test_skips_non_text(self):
        msg = MagicMock()
        msg.content = [MagicMock(spec=[])]
        assert collect_response_text(msg) == ""

    def test_empty_content(self):
        msg = MagicMock()
        msg.content = []
        assert collect_response_text(msg) == ""


class TestExtractOntologyJson:
    """Tests for extract_ontology_json."""

    def test_valid_block(self):
        text = 'Some text\n```ontology\n{"entities": []}\n```'
        result = extract_ontology_json(text)
        assert result == {"entities": []}

    def test_no_block(self):
        assert extract_ontology_json("no block here") is None

    def test_malformed_json(self):
        text = "```ontology\n{bad json}\n```"
        assert extract_ontology_json(text) is None

    def test_takes_first_block(self):
        text = (
            '```ontology\n{"a": 1}\n```\n'
            '```ontology\n{"b": 2}\n```'
        )
        result = extract_ontology_json(text)
        assert result == {"a": 1}

    def test_block_with_whitespace(self):
        text = '```ontology  \n{"x": 1}\n```'
        result = extract_ontology_json(text)
        assert result == {"x": 1}


class TestMergeOntologyUpdate:
    """Tests for merge_ontology_update."""

    def test_add_entity(self):
        o = Ontology()
        update = {"entities": [
            {"id": "e1", "name": "User"},
        ]}
        assert merge_ontology_update(o, update) is True
        assert len(o.entities) == 1
        assert o.entities[0].name == "User"

    def test_upsert_entity(self):
        o = Ontology(entities=[
            Entity(id="e1", name="Old"),
        ])
        update = {"entities": [
            {"id": "e1", "name": "New"},
        ]}
        merge_ontology_update(o, update)
        assert len(o.entities) == 1
        assert o.entities[0].name == "New"

    def test_append_relationship(self):
        o = Ontology()
        update = {"relationships": [{
            "source_entity_id": "e1",
            "target_entity_id": "e2",
            "name": "owns",
            "cardinality": "one_to_many",
        }]}
        merge_ontology_update(o, update)
        assert len(o.relationships) == 1

    def test_append_constraint(self):
        o = Ontology()
        update = {"domain_constraints": [{
            "name": "c1",
            "description": "rule",
        }]}
        merge_ontology_update(o, update)
        assert len(o.domain_constraints) == 1

    def test_upsert_open_question(self):
        o = Ontology(open_questions=[
            OpenQuestion(id="q1", text="old?"),
        ])
        update = {"open_questions": [
            {"id": "q1", "text": "new?"},
        ]}
        merge_ontology_update(o, update)
        assert len(o.open_questions) == 1
        assert o.open_questions[0].text == "new?"

    def test_add_open_question(self):
        o = Ontology()
        update = {"open_questions": [
            {"id": "q1", "text": "what?"},
        ]}
        merge_ontology_update(o, update)
        assert len(o.open_questions) == 1

    def test_empty_update(self):
        o = Ontology()
        assert merge_ontology_update(o, {}) is False

    def test_unknown_keys_ignored(self):
        o = Ontology()
        result = merge_ontology_update(o, {"foo": "bar"})
        assert result is False


class TestProcessResponse:
    """Tests for process_response."""

    def test_with_ontology_block(self):
        o = Ontology()
        text = '```ontology\n{"entities": [{"id": "e1", "name": "X"}]}\n```'
        assert process_response(text, o) is True
        assert len(o.entities) == 1

    def test_without_block(self):
        o = Ontology()
        assert process_response("no block", o) is False
        assert len(o.entities) == 0


class TestFormatOntologySummary:
    """Tests for format_ontology_summary."""

    def test_empty(self):
        result = format_ontology_summary(Ontology())
        assert "Entities (0):" in result
        assert "Relationships (0):" in result
        assert "Constraints (0):" in result
        assert "Open Questions (0):" in result

    def test_populated(self):
        o = Ontology(
            entities=[Entity(
                id="e1", name="User",
                properties=[Property(
                    name="email",
                    property_type=PropertyType(kind="str"),
                )],
            )],
            relationships=[Relationship(
                source_entity_id="e1",
                target_entity_id="e2",
                name="owns",
                cardinality="one_to_many",
            )],
            domain_constraints=[DomainConstraint(
                name="c1", description="rule",
            )],
            open_questions=[
                OpenQuestion(id="q1", text="what?"),
                OpenQuestion(
                    id="q2", text="done",
                    resolved=True,
                ),
            ],
        )
        result = format_ontology_summary(o)
        assert "e1: User [email]" in result
        assert "e1 --owns--> e2" in result
        assert "c1: rule" in result
        assert "[open] q1: what?" in result
        assert "[RESOLVED] q2: done" in result


class TestLoadSaveDag:
    """Tests for load_dag and save_dag."""

    def test_missing_file_returns_empty(self, tmp_path):
        dag = load_dag(
            str(tmp_path / "nope.json"), "proj",
        )
        assert dag.project_name == "proj"
        assert dag.nodes == []

    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "dag.json")
        dag = OntologyDAG(project_name="test")
        save_dag(dag, path)
        loaded = load_dag(path, "ignored")
        assert loaded.project_name == "test"


class TestSaveSnapshot:
    """Tests for save_snapshot."""

    def test_first_snapshot(self):
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
        assert dag.edges[0].parent_id == dag.nodes[0].id

    def test_snapshot_is_deep_copy(self):
        dag = OntologyDAG(project_name="p")
        o = Ontology(
            entities=[Entity(id="e1", name="X")],
        )
        save_snapshot(dag, o, "snap")
        o.entities[0] = Entity(id="e1", name="Changed")
        assert dag.nodes[0].ontology.entities[0].name == "X"


class TestBacktrack:
    """Tests for backtrack."""

    def test_at_root(self):
        dag = OntologyDAG(
            project_name="p",
            nodes=[DAGNode(
                id="root",
                ontology=Ontology(),
                created_at="t",
            )],
            current_node_id="root",
        )
        assert backtrack(dag) is None
        assert dag.current_node_id == "root"

    def test_moves_to_parent(self):
        dag = OntologyDAG(
            project_name="p",
            nodes=[
                DAGNode(
                    id="root", ontology=Ontology(),
                    created_at="t",
                ),
                DAGNode(
                    id="child", ontology=Ontology(),
                    created_at="t",
                ),
            ],
            edges=[DAGEdge(
                parent_id="root",
                child_id="child",
                decision=Decision(
                    question="?",
                    options=["a"],
                    chosen="a",
                    rationale="r",
                ),
                created_at="t",
            )],
            current_node_id="child",
        )
        result = backtrack(dag)
        assert result is not None
        assert result.id == "root"
        assert dag.current_node_id == "root"


class TestIsCommand:
    """Tests for is_command."""

    def test_show(self):
        assert is_command("show") is True

    def test_back(self):
        assert is_command("back") is True

    def test_save(self):
        assert is_command("save") is True

    def test_save_with_label(self):
        assert is_command("save my snapshot") is True

    def test_not_command(self):
        assert is_command("tell me more") is False

    def test_case_insensitive(self):
        assert is_command("SHOW") is True


class TestHandleCommand:
    """Tests for handle_command."""

    def test_show(self):
        o = Ontology()
        dag = OntologyDAG(project_name="p")
        result = handle_command("show", o, dag, "/tmp/x")
        assert "Entities (0):" in result

    def test_back_at_root(self):
        dag = OntologyDAG(
            project_name="p",
            nodes=[DAGNode(
                id="root", ontology=Ontology(),
                created_at="t",
            )],
            current_node_id="root",
        )
        o = Ontology()
        result = handle_command("back", o, dag, "/tmp/x")
        assert "root" in result.lower()

    def test_save_creates_snapshot(self, tmp_path):
        path = str(tmp_path / "dag.json")
        dag = OntologyDAG(project_name="p")
        o = Ontology()
        result = handle_command("save first", o, dag, path)
        assert "Saved snapshot" in result
        assert os.path.exists(path)


class TestHandleSave:
    """Tests for _handle_save."""

    def test_default_label(self, tmp_path):
        path = str(tmp_path / "dag.json")
        dag = OntologyDAG(project_name="p")
        o = Ontology()
        _handle_save("save", o, dag, path)
        assert "snapshot" in dag.nodes[0].label

    def test_custom_label(self, tmp_path):
        path = str(tmp_path / "dag.json")
        dag = OntologyDAG(project_name="p")
        o = Ontology()
        _handle_save("save my label", o, dag, path)
        assert dag.nodes[0].label == "my label"


class TestHandleBack:
    """Tests for _handle_back."""

    def test_restores_ontology(self, tmp_path):
        root_ontology = Ontology(
            entities=[Entity(id="e1", name="Root")],
        )
        dag = OntologyDAG(
            project_name="p",
            nodes=[
                DAGNode(
                    id="root",
                    ontology=root_ontology,
                    created_at="t",
                    label="root",
                ),
                DAGNode(
                    id="child",
                    ontology=Ontology(),
                    created_at="t",
                ),
            ],
            edges=[DAGEdge(
                parent_id="root",
                child_id="child",
                decision=Decision(
                    question="?",
                    options=["a"],
                    chosen="a",
                    rationale="r",
                ),
                created_at="t",
            )],
            current_node_id="child",
        )
        o = Ontology()
        _handle_back(o, dag, str(tmp_path / "dag.json"))
        assert len(o.entities) == 1
        assert o.entities[0].name == "Root"


class TestPrintTextBlocks:
    """Tests for print_text_blocks."""

    def test_prints_text(self, capsys):
        msg = MagicMock()
        msg.content = [TextBlock(text="hello")]
        print_text_blocks(msg)
        assert capsys.readouterr().out == "hello\n"

    def test_skips_non_text(self, capsys):
        msg = MagicMock()
        msg.content = [MagicMock(spec=[])]
        print_text_blocks(msg)
        assert capsys.readouterr().out == ""


class TestPrintResponse:
    """Tests for print_response."""

    @pytest.mark.asyncio
    async def test_returns_text(self):
        from python_agent.discovery_agent import print_response
        from claude_agent_sdk import AssistantMessage

        msg = AssistantMessage(
            content=[TextBlock(text="hello")],
            model="claude-opus-4-6",
        )
        client = MagicMock()

        async def fake_receive():
            yield msg

        client.receive_response = fake_receive
        result = await print_response(client)
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_skips_non_assistant(self):
        from python_agent.discovery_agent import print_response

        client = MagicMock()

        async def fake_receive():
            yield MagicMock(spec=[])

        client.receive_response = fake_receive
        result = await print_response(client)
        assert result == ""

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        from python_agent.discovery_agent import print_response

        client = MagicMock()

        async def fake_receive():
            return
            yield  # noqa: E501

        client.receive_response = fake_receive
        result = await print_response(client)
        assert result == ""


class TestReadUserInput:
    """Tests for read_user_input."""

    @patch("builtins.input", return_value="hello")
    def test_returns_input(self, mock):
        assert read_user_input() == "hello"

    @patch("builtins.input", return_value="quit")
    def test_quit(self, mock):
        assert read_user_input() is None

    @patch("builtins.input", return_value="exit")
    def test_exit(self, mock):
        assert read_user_input() is None

    @patch("builtins.input", return_value="done")
    def test_done(self, mock):
        assert read_user_input() is None

    @patch("builtins.input", side_effect=EOFError)
    def test_eof(self, mock):
        assert read_user_input() is None

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_interrupt(self, mock):
        assert read_user_input() is None


class TestInitOntology:
    """Tests for _init_ontology."""

    def test_empty_dag(self):
        dag = OntologyDAG(project_name="p")
        o = _init_ontology(dag)
        assert o == Ontology()

    def test_with_current_node(self):
        ontology = Ontology(
            entities=[Entity(id="e1", name="X")],
        )
        dag = OntologyDAG(
            project_name="p",
            nodes=[DAGNode(
                id="n1",
                ontology=ontology,
                created_at="t",
            )],
            current_node_id="n1",
        )
        o = _init_ontology(dag)
        assert len(o.entities) == 1
        assert o.entities[0].name == "X"


class TestRun:
    """Tests for run."""

    @pytest.mark.asyncio
    async def test_sends_initial_query(self, tmp_path):
        client = AsyncMock()

        async def fake_receive():
            return
            yield  # noqa: E501

        client.receive_response = fake_receive
        path = str(tmp_path / "dag.json")

        with (
            patch(
                "python_agent.discovery_agent."
                "ClaudeSDKClient",
            ) as mock_cls,
            patch(
                "python_agent.discovery_agent."
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
            await run("build X", "claude-opus-4-6", path)

        client.query.assert_any_call("build X")

    @pytest.mark.asyncio
    async def test_handles_command(self, tmp_path, capsys):
        client = AsyncMock()

        async def fake_receive():
            return
            yield  # noqa: E501

        client.receive_response = fake_receive
        path = str(tmp_path / "dag.json")
        call_count = 0

        def fake_input():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "show"
            return None

        with (
            patch(
                "python_agent.discovery_agent."
                "ClaudeSDKClient",
            ) as mock_cls,
            patch(
                "python_agent.discovery_agent."
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
            await run("desc", "claude-opus-4-6", path)

        out = capsys.readouterr().out
        assert "Entities (0):" in out
        assert client.query.call_count == 1

    @pytest.mark.asyncio
    async def test_processes_normal_input(self, tmp_path):
        from claude_agent_sdk import AssistantMessage

        client = AsyncMock()
        msg = AssistantMessage(
            content=[TextBlock(text="got it")],
            model="claude-opus-4-6",
        )

        async def fake_receive():
            yield msg

        client.receive_response = fake_receive
        path = str(tmp_path / "dag.json")
        call_count = 0

        def fake_input():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "add a User entity"
            return None

        with (
            patch(
                "python_agent.discovery_agent."
                "ClaudeSDKClient",
            ) as mock_cls,
            patch(
                "python_agent.discovery_agent."
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
            await run("desc", "claude-opus-4-6", path)

        client.query.assert_any_call("add a User entity")


class TestParseArgs:
    """Tests for parse_args."""

    def test_description_required(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_description_positional(self):
        args = parse_args(["my project"])
        assert args.description == "my project"

    def test_default_dag_file(self):
        args = parse_args(["desc"])
        assert args.dag_file == "ontology.json"

    def test_custom_dag_file(self):
        args = parse_args(["--dag-file", "x.json", "desc"])
        assert args.dag_file == "x.json"

    def test_default_model(self):
        args = parse_args(["desc"])
        assert args.model == "claude-opus-4-6"

    def test_custom_model(self):
        args = parse_args(["-m", "claude-sonnet-4-6", "desc"])
        assert args.model == "claude-sonnet-4-6"

    def test_help_text(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(["--help"])
        out = capsys.readouterr().out
        assert "XX" not in out
        assert "Interactive ontology discovery agent" in out
        assert "Project description" in out
        assert "DAG JSON file" in out
        assert "Model to use" in out


class TestMain:
    """Tests for main."""

    @patch("python_agent.discovery_agent.asyncio.run")
    def test_returns_zero(self, mock_run):
        assert main(["desc"]) == 0

    @patch("python_agent.discovery_agent.asyncio.run")
    def test_calls_asyncio_run(self, mock_run):
        main(["desc"])
        mock_run.assert_called_once()
