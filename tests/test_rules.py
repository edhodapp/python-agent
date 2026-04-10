"""Tests for rules module."""

from python_agent.rules import (
    coding_system_prompt,
    discovery_system_prompt,
    divergence_system_prompt,
    load_rules,
    planning_system_prompt,
    strategy_system_prompt,
)


class TestLoadRules:
    """Tests for load_rules()."""

    def test_returns_string(self):
        result = load_rules()
        assert isinstance(result, str)

    def test_contains_python_standards(self):
        result = load_rules()
        assert "## Python Standards" in result

    def test_contains_testing_philosophy(self):
        result = load_rules()
        assert "## Testing Philosophy" in result

    def test_contains_flake8_rule(self):
        result = load_rules()
        assert "flake8" in result

    def test_contains_branch_coverage_rule(self):
        result = load_rules()
        assert "100% branch coverage" in result

    def test_contains_mutmut_rule(self):
        result = load_rules()
        assert "mutmut" in result


class TestCodingSystemPrompt:
    """Tests for coding_system_prompt()."""

    def test_includes_rules(self):
        result = coding_system_prompt("/tmp/proj")
        assert "## Python Standards" in result

    def test_includes_project_dir(self):
        result = coding_system_prompt("/tmp/my_project")
        assert "/tmp/my_project" in result

    def test_includes_coding_role(self):
        result = coding_system_prompt("/tmp/proj")
        assert "Coding Agent" in result

    def test_includes_workflow_steps(self):
        result = coding_system_prompt("/tmp/proj")
        assert "flake8" in result
        assert "pytest" in result
        assert "mutmut" in result

    def test_includes_venv_path(self):
        result = coding_system_prompt("/home/user/proj")
        assert "/home/user/proj/.venv/" in result

    def test_no_ontology_by_default(self):
        result = coding_system_prompt("/tmp/proj")
        assert "ontology-data" not in result

    def test_includes_ontology_when_provided(self):
        result = coding_system_prompt(
            "/tmp/proj",
            ontology_json='{"entities": []}',
        )
        assert "ontology-data" in result
        assert '{"entities": []}' in result
        assert "Design Specification" in result

    def test_ontology_none_same_as_default(self):
        without = coding_system_prompt("/tmp/proj")
        with_none = coding_system_prompt(
            "/tmp/proj", ontology_json=None,
        )
        assert without == with_none


class TestPlanningSystemPrompt:
    """Tests for planning_system_prompt()."""

    def test_includes_rules(self):
        result = planning_system_prompt()
        assert "## Python Standards" in result

    def test_includes_planning_role(self):
        result = planning_system_prompt()
        assert "Planning Agent" in result

    def test_includes_plan_format(self):
        result = planning_system_prompt()
        assert "Plan Document Format" in result

    def test_does_not_include_project_dir(self):
        """Planning agent has no fixed project dir."""
        result = planning_system_prompt()
        assert "Working directory:" not in result

    def test_says_do_not_write_code(self):
        result = planning_system_prompt()
        assert "Do not write code" in result


class TestDiscoverySystemPrompt:
    """Tests for discovery_system_prompt()."""

    def test_includes_discovery_role(self):
        result = discovery_system_prompt()
        assert "Discovery Agent" in result

    def test_includes_ontology_format(self):
        result = discovery_system_prompt()
        assert "```ontology" in result

    def test_includes_rules(self):
        result = discovery_system_prompt()
        assert "## Python Standards" in result


class TestStrategySystemPrompt:
    """Tests for strategy_system_prompt()."""

    def test_includes_ontology(self):
        result = strategy_system_prompt('{"x": 1}', 3)
        assert '{"x": 1}' in result

    def test_includes_num_candidates(self):
        result = strategy_system_prompt("{}", 5)
        assert "5" in result

    def test_includes_strategies_format(self):
        result = strategy_system_prompt("{}", 3)
        assert "```strategies" in result


class TestDivergenceSystemPrompt:
    """Tests for divergence_system_prompt()."""

    def test_includes_ontology(self):
        result = divergence_system_prompt('{"x": 1}', "s")
        assert '{"x": 1}' in result

    def test_includes_strategy(self):
        result = divergence_system_prompt("{}", "my strat")
        assert "my strat" in result

    def test_includes_ontology_format(self):
        result = divergence_system_prompt("{}", "s")
        assert "```ontology" in result
