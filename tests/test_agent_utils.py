"""Tests for agent_utils shared module."""

from unittest.mock import MagicMock, patch

from claude_agent_sdk import TextBlock

from python_agent.agent_utils import (
    collect_response_text,
    extract_ontology_json,
    print_text_blocks,
    read_user_input,
)


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

    def test_empty(self, capsys):
        msg = MagicMock()
        msg.content = []
        print_text_blocks(msg)
        assert capsys.readouterr().out == ""


class TestCollectResponseText:
    """Tests for collect_response_text."""

    def test_extracts_text(self):
        msg = MagicMock()
        msg.content = [
            TextBlock(text="a"),
            TextBlock(text="b"),
        ]
        assert collect_response_text(msg) == "a\nb"

    def test_skips_non_text(self):
        msg = MagicMock()
        msg.content = [MagicMock(spec=[])]
        assert collect_response_text(msg) == ""

    def test_empty(self):
        msg = MagicMock()
        msg.content = []
        assert collect_response_text(msg) == ""


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


class TestExtractOntologyJson:
    """Tests for extract_ontology_json."""

    def test_valid_block(self):
        text = '```ontology\n{"entities": []}\n```'
        result = extract_ontology_json(text)
        assert result == {"entities": []}

    def test_no_block(self):
        assert extract_ontology_json("no block") is None

    def test_bad_json(self):
        text = "```ontology\n{bad}\n```"
        assert extract_ontology_json(text) is None

    def test_first_block(self):
        text = (
            '```ontology\n{"a": 1}\n```\n'
            '```ontology\n{"b": 2}\n```'
        )
        assert extract_ontology_json(text) == {"a": 1}
