"""Tests for divergence_agent module."""

from unittest.mock import MagicMock, patch

import pytest
from claude_agent_sdk import TextBlock

from python_agent.divergence_agent import (
    add_candidate_node,
    build_decision,
    collect_response_text,
    extract_ontology_json,
    extract_strategies,
    main,
    parse_args,
    remaining_budget,
    run,
    run_query,
)
from python_agent.ontology import (
    DAGNode,
    Entity,
    Ontology,
    OntologyDAG,
)


class TestCollectResponseText:
    """Tests for collect_response_text."""

    def test_extracts_text(self):
        msg = MagicMock()
        msg.content = [TextBlock(text="hello")]
        assert collect_response_text(msg) == "hello"

    def test_skips_non_text(self):
        msg = MagicMock()
        msg.content = [MagicMock(spec=[])]
        assert collect_response_text(msg) == ""


class TestExtractOntologyJson:
    """Tests for extract_ontology_json."""

    def test_valid(self):
        text = '```ontology\n{"entities": []}\n```'
        assert extract_ontology_json(text) == {"entities": []}

    def test_none(self):
        assert extract_ontology_json("no block") is None

    def test_bad_json(self):
        text = "```ontology\n{bad}\n```"
        assert extract_ontology_json(text) is None


class TestExtractStrategies:
    """Tests for extract_strategies."""

    def test_valid(self):
        text = '```strategies\n[{"label": "a"}]\n```'
        result = extract_strategies(text)
        assert result == [{"label": "a"}]

    def test_none(self):
        assert extract_strategies("no block") is None

    def test_bad_json(self):
        text = "```strategies\n{bad}\n```"
        assert extract_strategies(text) is None

    def test_not_a_list(self):
        text = '```strategies\n{"not": "list"}\n```'
        assert extract_strategies(text) is None

    def test_multiple_takes_first(self):
        text = (
            '```strategies\n[{"a": 1}]\n```\n'
            '```strategies\n[{"b": 2}]\n```'
        )
        result = extract_strategies(text)
        assert result == [{"a": 1}]


class TestRunQuery:
    """Tests for run_query."""

    @pytest.mark.asyncio
    async def test_returns_text_and_cost(self):
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
        )

        msg = AssistantMessage(
            content=[TextBlock(text="response")],
            model="m",
        )
        result_msg = ResultMessage(
            subtype="success",
            total_cost_usd=0.01,
            duration_ms=100,
            duration_api_ms=80,
            is_error=False,
            num_turns=1,
            session_id="s",
        )

        async def fake_query(**kwargs):
            yield msg
            yield result_msg

        with patch(
            "python_agent.divergence_agent.query",
            fake_query,
        ):
            text, cost = await run_query("t", MagicMock())

        assert "response" in text
        assert cost == 0.01

    @pytest.mark.asyncio
    async def test_cost_none_defaults_zero(self):
        from claude_agent_sdk import ResultMessage

        msg = ResultMessage(
            subtype="success",
            total_cost_usd=None,
            duration_ms=100,
            duration_api_ms=80,
            is_error=False,
            num_turns=1,
            session_id="s",
        )

        async def fake_query(**kwargs):
            yield msg

        with patch(
            "python_agent.divergence_agent.query",
            fake_query,
        ):
            _, cost = await run_query("t", MagicMock())

        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_ignores_unknown_messages(self):
        async def fake_query(**kwargs):
            yield MagicMock(spec=[])

        with patch(
            "python_agent.divergence_agent.query",
            fake_query,
        ):
            text, cost = await run_query("t", MagicMock())

        assert text == ""
        assert cost == 0.0


class TestBuildDecision:
    """Tests for build_decision."""

    def test_from_strategy(self):
        s = {
            "question": "DB?",
            "options": ["SQLite", "Postgres"],
            "chosen": "SQLite",
            "strategy": "Simple approach",
        }
        d = build_decision(s)
        assert d.question == "DB?"
        assert d.chosen == "SQLite"
        assert d.rationale == "Simple approach"

    def test_defaults(self):
        d = build_decision({})
        assert d.question == "architecture"
        assert d.options == []
        assert d.chosen == ""
        assert d.rationale == ""


class TestAddCandidateNode:
    """Tests for add_candidate_node."""

    def test_creates_node_and_edge(self):
        dag = OntologyDAG(project_name="p")
        ontology_dict = {"entities": [
            {"id": "e1", "name": "X"},
        ]}
        strategy = {
            "label": "approach-a",
            "question": "?",
            "options": ["a"],
            "chosen": "a",
            "strategy": "desc",
        }
        node_id = add_candidate_node(
            dag, "parent1", ontology_dict, strategy,
        )
        assert len(dag.nodes) == 1
        assert dag.nodes[0].label == "approach-a"
        assert dag.nodes[0].id == node_id
        assert len(dag.edges) == 1
        assert dag.edges[0].parent_id == "parent1"
        assert dag.edges[0].child_id == node_id


class TestRemainingBudget:
    """Tests for remaining_budget."""

    def test_normal(self):
        assert remaining_budget(1.5, 5.0) == 3.5

    def test_none_budget(self):
        assert remaining_budget(1.0, None) is None


class TestIdentifyStrategies:
    """Tests for identify_strategies."""

    @pytest.mark.asyncio
    async def test_returns_strategies(self):
        from python_agent.divergence_agent import (
            identify_strategies,
        )
        text = '```strategies\n[{"label": "a"}]\n```'

        async def fake_rq(task, options):
            return text, 0.01

        with patch(
            "python_agent.divergence_agent.run_query",
            fake_rq,
        ):
            result, cost = await identify_strategies(
                "{}", 3, "model", 5.0,
            )
        assert result == [{"label": "a"}]
        assert cost == 0.01

    @pytest.mark.asyncio
    async def test_returns_empty_on_failure(self):
        from python_agent.divergence_agent import (
            identify_strategies,
        )

        async def fake_rq(task, options):
            return "no block", 0.01

        with patch(
            "python_agent.divergence_agent.run_query",
            fake_rq,
        ):
            result, cost = await identify_strategies(
                "{}", 3, "model", 5.0,
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_truncates_to_num(self):
        from python_agent.divergence_agent import (
            identify_strategies,
        )
        text = (
            '```strategies\n'
            '[{"label":"a"},{"label":"b"},{"label":"c"}]'
            '\n```'
        )

        async def fake_rq(task, options):
            return text, 0.01

        with patch(
            "python_agent.divergence_agent.run_query",
            fake_rq,
        ):
            result, _ = await identify_strategies(
                "{}", 2, "model", 5.0,
            )
        assert len(result) == 2


class TestGenerateCandidate:
    """Tests for generate_candidate."""

    @pytest.mark.asyncio
    async def test_returns_ontology_dict(self):
        from python_agent.divergence_agent import (
            generate_candidate,
        )
        text = '```ontology\n{"entities": []}\n```'

        async def fake_rq(task, options):
            return text, 0.02

        with patch(
            "python_agent.divergence_agent.run_query",
            fake_rq,
        ):
            result, cost = await generate_candidate(
                "{}", {"label": "a", "strategy": "s"},
                "model", 5.0,
            )
        assert result == {"entities": []}
        assert cost == 0.02

    @pytest.mark.asyncio
    async def test_returns_none_on_failure(self):
        from python_agent.divergence_agent import (
            generate_candidate,
        )

        async def fake_rq(task, options):
            return "no block", 0.02

        with patch(
            "python_agent.divergence_agent.run_query",
            fake_rq,
        ):
            result, _ = await generate_candidate(
                "{}", {"label": "a"}, "model", 5.0,
            )
        assert result is None


class TestRun:
    """Tests for run."""

    @pytest.mark.asyncio
    async def test_no_current_node(self, tmp_path, capsys):
        from python_agent.dag_utils import save_dag
        path = str(tmp_path / "dag.json")
        dag = OntologyDAG(project_name="p")
        save_dag(dag, path)
        result = await run(path, 3, "model", 5.0)
        assert result == 0
        out = capsys.readouterr().out
        assert "no current node" in out.lower()

    @pytest.mark.asyncio
    async def test_no_strategies(self, tmp_path, capsys):
        from python_agent.dag_utils import save_dag
        path = str(tmp_path / "dag.json")
        dag = OntologyDAG(
            project_name="p",
            nodes=[DAGNode(
                id="n1",
                ontology=Ontology(
                    entities=[Entity(id="e1", name="X")],
                ),
                created_at="t",
            )],
            current_node_id="n1",
        )
        save_dag(dag, path)

        async def fake_identify(*args, **kwargs):
            return [], 0.01

        with patch(
            "python_agent.divergence_agent."
            "identify_strategies",
            fake_identify,
        ):
            result = await run(path, 3, "model", 5.0)

        assert result == 0
        out = capsys.readouterr().out
        assert "could not identify" in out.lower()

    @pytest.mark.asyncio
    async def test_generates_candidates(self, tmp_path):
        from python_agent.dag_utils import (
            load_dag as ld,
            save_dag,
        )
        path = str(tmp_path / "dag.json")
        dag = OntologyDAG(
            project_name="p",
            nodes=[DAGNode(
                id="n1",
                ontology=Ontology(
                    entities=[Entity(id="e1", name="X")],
                ),
                created_at="t",
            )],
            current_node_id="n1",
        )
        save_dag(dag, path)

        strategies = [
            {"label": "a", "strategy": "approach a",
             "question": "?", "options": ["a", "b"],
             "chosen": "a"},
        ]
        ontology_dict = {
            "entities": [{"id": "e1", "name": "X"}],
        }

        async def fake_identify(*args, **kwargs):
            return strategies, 0.01

        async def fake_generate(*args, **kwargs):
            return ontology_dict, 0.02

        with (
            patch(
                "python_agent.divergence_agent."
                "identify_strategies",
                fake_identify,
            ),
            patch(
                "python_agent.divergence_agent."
                "generate_candidate",
                fake_generate,
            ),
        ):
            result = await run(path, 1, "model", 5.0)

        assert result == 1
        loaded = ld(path, "x")
        assert len(loaded.nodes) == 2
        assert len(loaded.edges) == 1

    @pytest.mark.asyncio
    async def test_skips_failed_candidate(
        self, tmp_path, capsys,
    ):
        from python_agent.dag_utils import save_dag
        path = str(tmp_path / "dag.json")
        dag = OntologyDAG(
            project_name="p",
            nodes=[DAGNode(
                id="n1",
                ontology=Ontology(),
                created_at="t",
            )],
            current_node_id="n1",
        )
        save_dag(dag, path)

        strategies = [{"label": "bad", "strategy": "x"}]

        async def fake_identify(*args, **kwargs):
            return strategies, 0.01

        async def fake_generate(*args, **kwargs):
            return None, 0.02

        with (
            patch(
                "python_agent.divergence_agent."
                "identify_strategies",
                fake_identify,
            ),
            patch(
                "python_agent.divergence_agent."
                "generate_candidate",
                fake_generate,
            ),
        ):
            result = await run(path, 1, "model", 5.0)

        assert result == 0
        out = capsys.readouterr().out
        assert "Failed" in out


class TestParseArgs:
    """Tests for parse_args."""

    def test_dag_file_required(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_dag_file(self):
        args = parse_args(["--dag-file", "x.json"])
        assert args.dag_file == "x.json"

    def test_default_num_candidates(self):
        args = parse_args(["--dag-file", "x.json"])
        assert args.num_candidates == 3

    def test_custom_num_candidates(self):
        args = parse_args(
            ["--dag-file", "x.json", "-n", "5"],
        )
        assert args.num_candidates == 5

    def test_default_model(self):
        args = parse_args(["--dag-file", "x.json"])
        assert args.model == "claude-sonnet-4-6"

    def test_custom_model(self):
        args = parse_args(
            ["--dag-file", "x.json", "-m", "opus"],
        )
        assert args.model == "opus"

    def test_default_max_budget(self):
        args = parse_args(["--dag-file", "x.json"])
        assert args.max_budget == 5.0

    def test_help_text(self, capsys):
        with pytest.raises(SystemExit):
            parse_args(["--help"])
        out = capsys.readouterr().out
        assert "XX" not in out
        assert "divergent solution candidates" in out
        assert "DAG JSON file" in out
        assert "Number of candidates" in out


class TestMain:
    """Tests for main."""

    @patch("python_agent.divergence_agent.asyncio.run")
    def test_returns_zero(self, mock_run):
        assert main(["--dag-file", "x.json"]) == 0

    @patch("python_agent.divergence_agent.asyncio.run")
    def test_calls_asyncio_run(self, mock_run):
        main(["--dag-file", "x.json"])
        mock_run.assert_called_once()
