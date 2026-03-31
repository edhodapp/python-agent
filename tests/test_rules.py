"""Tests for rules module."""

from python_agent.rules import (
    coding_system_prompt,
    load_rules,
    planning_system_prompt,
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
