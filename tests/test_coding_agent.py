"""Tests for coding_agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from python_agent.coding_agent import (
    ESCALATION_MODEL,
    _load_ontology_json,
    main,
    parse_args,
    print_text_blocks,
    remaining_budget,
    run,
    run_query,
    should_escalate,
)


class TestPrintTextBlocks:
    """Tests for print_text_blocks()."""

    def test_prints_text_content(self, capsys):
        from claude_agent_sdk import TextBlock

        block = TextBlock(text="hello world")
        message = MagicMock()
        message.content = [block]
        print_text_blocks(message)
        captured = capsys.readouterr()
        assert captured.out == "hello world\n"

    def test_skips_non_text_blocks(self, capsys):
        block = MagicMock(spec=[])
        message = MagicMock()
        message.content = [block]
        print_text_blocks(message)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_handles_empty_content(self, capsys):
        message = MagicMock()
        message.content = []
        print_text_blocks(message)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_prints_multiple_text_blocks(self, capsys):
        from claude_agent_sdk import TextBlock

        blocks = [TextBlock(text="one"), TextBlock(text="two")]
        message = MagicMock()
        message.content = blocks
        print_text_blocks(message)
        captured = capsys.readouterr()
        assert "one" in captured.out
        assert "two" in captured.out


class TestEscalationModel:
    """Tests for ESCALATION_MODEL constant."""

    def test_value(self):
        assert ESCALATION_MODEL == "claude-opus-4-6"


class TestRunQuery:
    """Tests for run_query()."""

    @pytest.mark.asyncio
    async def test_prints_assistant_text(self, capsys):
        from claude_agent_sdk import AssistantMessage, TextBlock

        msg = AssistantMessage(
            content=[TextBlock(text="wrote code")],
            model="claude-sonnet-4-6",
        )

        async def fake_query(**kwargs):
            yield msg

        with patch("python_agent.coding_agent.query", fake_query):
            result = await run_query("task", MagicMock())

        captured = capsys.readouterr()
        assert "wrote code" in captured.out
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_result_message(self, capsys):
        from claude_agent_sdk import ResultMessage

        msg = ResultMessage(
            subtype="success",
            total_cost_usd=0.0042,
            duration_ms=100,
            duration_api_ms=80,
            is_error=False,
            num_turns=1,
            session_id="test",
        )

        async def fake_query(**kwargs):
            yield msg

        with patch("python_agent.coding_agent.query", fake_query):
            result = await run_query("task", MagicMock())

        captured = capsys.readouterr()
        assert captured.out == "\nDone. Cost: $0.0042\n"
        assert result is msg

    @pytest.mark.asyncio
    async def test_ignores_unknown_messages(self, capsys):
        msg = MagicMock(spec=[])

        async def fake_query(**kwargs):
            yield msg

        with patch("python_agent.coding_agent.query", fake_query):
            result = await run_query("task", MagicMock())

        captured = capsys.readouterr()
        assert captured.out == ""
        assert result is None


class TestShouldEscalate:
    """Tests for should_escalate()."""

    def test_none_result(self):
        assert should_escalate(None, 10) is False

    def test_error_triggers_escalation(self):
        result = MagicMock(is_error=True, num_turns=3)
        assert should_escalate(result, 10) is True

    def test_max_turns_reached(self):
        result = MagicMock(is_error=False, num_turns=10)
        assert should_escalate(result, 10) is True

    def test_success_no_escalation(self):
        result = MagicMock(is_error=False, num_turns=5)
        assert should_escalate(result, 10) is False

    def test_max_turns_none(self):
        result = MagicMock(is_error=False, num_turns=5)
        assert should_escalate(result, None) is False


class TestRemainingBudget:
    """Tests for remaining_budget()."""

    def test_none_budget_returns_none(self):
        assert remaining_budget(MagicMock(), None) is None

    def test_none_result_returns_full_budget(self):
        assert remaining_budget(None, 5.0) == 5.0

    def test_none_cost_returns_full_budget(self):
        result = MagicMock(total_cost_usd=None)
        assert remaining_budget(result, 5.0) == 5.0

    def test_subtracts_cost(self):
        result = MagicMock(total_cost_usd=1.5)
        assert remaining_budget(result, 5.0) == 3.5


class TestRun:
    """Tests for run()."""

    @pytest.mark.asyncio
    async def test_prints_assistant_text(self, capsys):
        from claude_agent_sdk import AssistantMessage, TextBlock

        msg = AssistantMessage(
            content=[TextBlock(text="wrote code")],
            model="claude-sonnet-4-6",
        )

        async def fake_query(**kwargs):
            yield msg

        with patch("python_agent.coding_agent.query", fake_query):
            await run("task", "/tmp", "model", 10, 1.0)

        captured = capsys.readouterr()
        assert "wrote code" in captured.out

    @pytest.mark.asyncio
    async def test_prints_result_cost(self, capsys):
        from claude_agent_sdk import ResultMessage

        msg = ResultMessage(
            subtype="success",
            total_cost_usd=0.0042,
            duration_ms=100,
            duration_api_ms=80,
            is_error=False,
            num_turns=1,
            session_id="test",
        )

        async def fake_query(**kwargs):
            yield msg

        with patch("python_agent.coding_agent.query", fake_query):
            await run("task", "/tmp", "model", 10, 1.0)

        captured = capsys.readouterr()
        assert "0.0042" in captured.out

    @pytest.mark.asyncio
    async def test_ignores_unknown_message_types(self, capsys):
        msg = MagicMock(spec=[])

        async def fake_query(**kwargs):
            yield msg

        with patch("python_agent.coding_agent.query", fake_query):
            await run("task", "/tmp", "model", 10, 1.0)

        captured = capsys.readouterr()
        assert captured.out == ""

    @pytest.mark.asyncio
    async def test_no_escalation_on_success(self):
        result = MagicMock(
            is_error=False, num_turns=5, total_cost_usd=0.01,
        )
        mock_rq = AsyncMock(return_value=result)
        with patch("python_agent.coding_agent.run_query", mock_rq):
            await run("task", "/tmp", "claude-sonnet-4-6", 10, 5.0)
        assert mock_rq.call_count == 1

    @pytest.mark.asyncio
    async def test_first_call_options(self):
        result = MagicMock(
            is_error=False, num_turns=5, total_cost_usd=0.01,
        )
        mock_rq = AsyncMock(return_value=result)
        with patch("python_agent.coding_agent.run_query", mock_rq):
            await run("task", "/proj", "claude-sonnet-4-6", 10, 5.0)
        opts = mock_rq.call_args_list[0][0][1]
        assert opts.model == "claude-sonnet-4-6"
        expected_tools = ["Read", "Edit", "Bash", "Glob", "Grep"]
        assert opts.allowed_tools == expected_tools
        assert opts.permission_mode == "bypassPermissions"
        assert opts.max_turns == 10
        assert opts.max_budget_usd == 5.0
        assert opts.cwd == "/proj"
        assert opts.system_prompt is not None

    @pytest.mark.asyncio
    async def test_escalates_on_error(self, capsys):
        error_result = MagicMock(
            is_error=True, num_turns=3, total_cost_usd=0.50,
        )
        mock_rq = AsyncMock(side_effect=[error_result, None])
        with patch("python_agent.coding_agent.run_query", mock_rq):
            await run("task", "/tmp", "claude-sonnet-4-6", 10, 5.0)
        assert mock_rq.call_count == 2
        escalation_task = mock_rq.call_args_list[1][0][0]
        assert "Continue this task" in escalation_task
        assert "Partial changes may already exist" in escalation_task
        assert "in the working directory" in escalation_task
        assert escalation_task.endswith("task")
        captured = capsys.readouterr()
        assert "\nEscalating to Opus...\n" in captured.out

    @pytest.mark.asyncio
    async def test_escalates_on_max_turns(self):
        stuck_result = MagicMock(
            is_error=False, num_turns=10, total_cost_usd=1.0,
        )
        mock_rq = AsyncMock(side_effect=[stuck_result, None])
        with patch("python_agent.coding_agent.run_query", mock_rq):
            await run("task", "/tmp", "claude-sonnet-4-6", 10, 5.0)
        assert mock_rq.call_count == 2

    @pytest.mark.asyncio
    async def test_no_escalation_when_already_opus(self):
        error_result = MagicMock(
            is_error=True, num_turns=3, total_cost_usd=0.50,
        )
        mock_rq = AsyncMock(return_value=error_result)
        with patch("python_agent.coding_agent.run_query", mock_rq):
            await run(
                "task", "/tmp", ESCALATION_MODEL, 10, 5.0,
            )
        assert mock_rq.call_count == 1

    @pytest.mark.asyncio
    async def test_escalation_call_options(self):
        error_result = MagicMock(
            is_error=True, num_turns=3, total_cost_usd=1.5,
        )
        mock_rq = AsyncMock(side_effect=[error_result, None])
        with patch("python_agent.coding_agent.run_query", mock_rq):
            await run("task", "/proj", "claude-sonnet-4-6", 10, 5.0)
        opts = mock_rq.call_args_list[1][0][1]
        assert opts.model == ESCALATION_MODEL
        expected_tools = ["Read", "Edit", "Bash", "Glob", "Grep"]
        assert opts.allowed_tools == expected_tools
        assert opts.permission_mode == "bypassPermissions"
        assert opts.max_turns == 10
        assert opts.max_budget_usd == 3.5
        assert opts.cwd == "/proj"
        assert opts.system_prompt is not None


class TestLoadOntologyJson:
    """Tests for _load_ontology_json()."""

    def test_none_dag_file_returns_none(self):
        assert _load_ontology_json(None) is None

    def test_valid_dag_file_returns_json(self, tmp_path):
        from python_agent.dag_utils import save_dag, save_snapshot
        from python_agent.ontology import Entity, Ontology, OntologyDAG

        ontology = Ontology(
            entities=[Entity(id="test-e", name="TestEntity")],
        )
        dag = OntologyDAG(project_name="test")
        save_snapshot(dag, ontology, "initial")
        dag_path = str(tmp_path / "test.json")
        save_dag(dag, dag_path)

        result = _load_ontology_json(dag_path)
        assert result is not None
        assert "test-e" in result
        assert "TestEntity" in result

    def test_empty_dag_returns_none(self, tmp_path):
        from python_agent.dag_utils import save_dag
        from python_agent.ontology import OntologyDAG

        dag = OntologyDAG(project_name="empty")
        dag_path = str(tmp_path / "empty.json")
        save_dag(dag, dag_path)

        assert _load_ontology_json(dag_path) is None

    def test_missing_file_returns_none(self):
        assert _load_ontology_json("/no/such/file.json") is None


class TestRunWithDagFile:
    """Tests for run() with dag_file parameter."""

    @pytest.mark.asyncio
    async def test_dag_file_adds_ontology_to_prompt(self, tmp_path):
        from python_agent.dag_utils import save_dag, save_snapshot
        from python_agent.ontology import Entity, Ontology, OntologyDAG

        ontology = Ontology(
            entities=[Entity(id="my-ent", name="MyEntity")],
        )
        dag = OntologyDAG(project_name="test")
        save_snapshot(dag, ontology, "initial")
        dag_path = str(tmp_path / "test.json")
        save_dag(dag, dag_path)

        result = MagicMock(
            is_error=False, num_turns=5, total_cost_usd=0.01,
        )
        mock_rq = AsyncMock(return_value=result)
        with patch("python_agent.coding_agent.run_query", mock_rq):
            await run(
                "task", "/proj", "claude-sonnet-4-6",
                10, 5.0, dag_file=dag_path,
            )
        opts = mock_rq.call_args_list[0][0][1]
        assert "my-ent" in opts.system_prompt
        assert "MyEntity" in opts.system_prompt
        assert "ontology-data" in opts.system_prompt

    @pytest.mark.asyncio
    async def test_no_dag_file_no_ontology_in_prompt(self):
        result = MagicMock(
            is_error=False, num_turns=5, total_cost_usd=0.01,
        )
        mock_rq = AsyncMock(return_value=result)
        with patch("python_agent.coding_agent.run_query", mock_rq):
            await run(
                "task", "/proj", "claude-sonnet-4-6",
                10, 5.0, dag_file=None,
            )
        opts = mock_rq.call_args_list[0][0][1]
        assert "ontology-data" not in opts.system_prompt


class TestParseArgs:
    """Tests for parse_args()."""

    def test_task_required(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_task_positional(self):
        args = parse_args(["write a function"])
        assert args.task == "write a function"

    def test_default_project_dir(self):
        args = parse_args(["task"])
        assert args.project_dir == "."

    def test_custom_project_dir(self):
        args = parse_args(["-d", "/tmp/proj", "task"])
        assert args.project_dir == "/tmp/proj"

    def test_default_model(self):
        args = parse_args(["task"])
        assert args.model == "claude-sonnet-4-6"

    def test_custom_model(self):
        args = parse_args(["-m", "claude-opus-4-6", "task"])
        assert args.model == "claude-opus-4-6"

    def test_default_max_turns(self):
        args = parse_args(["task"])
        assert args.max_turns == 30

    def test_custom_max_turns(self):
        args = parse_args(["--max-turns", "10", "task"])
        assert args.max_turns == 10

    def test_default_max_budget(self):
        args = parse_args(["task"])
        assert args.max_budget == 5.0

    def test_custom_max_budget(self):
        args = parse_args(["--max-budget", "2.5", "task"])
        assert args.max_budget == 2.5

    def test_default_dag_file(self):
        args = parse_args(["task"])
        assert args.dag_file is None

    def test_custom_dag_file(self):
        args = parse_args(["--dag-file", "design.json", "task"])
        assert args.dag_file == "design.json"

    def test_help_text(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(["--help"])
        out = capsys.readouterr().out
        assert "XX" not in out
        assert "Run the Python coding agent on a task" in out
        assert "Task description for the agent" in out
        assert "Project directory to work in" in out
        assert "Model to use" in out
        assert "Maximum agent turns" in out
        assert "Maximum budget in USD" in out
        assert "ontology DAG JSON" in out


class TestMain:
    """Tests for main()."""

    @patch("python_agent.coding_agent.asyncio.run")
    def test_returns_zero(self, mock_run):
        result = main(["task"])
        assert result == 0

    @patch("python_agent.coding_agent.asyncio.run")
    def test_calls_asyncio_run(self, mock_run):
        main(["task"])
        mock_run.assert_called_once()
