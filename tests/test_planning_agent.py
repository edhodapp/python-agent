"""Tests for planning_agent module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from python_agent.planning_agent import (
    main,
    parse_args,
    print_response,
    print_text_blocks,
    read_user_input,
    run,
)


class TestPrintTextBlocks:
    """Tests for print_text_blocks()."""

    def test_prints_text_content(self, capsys):
        from claude_agent_sdk import TextBlock

        block = TextBlock(text="plan step 1")
        message = MagicMock()
        message.content = [block]
        print_text_blocks(message)
        captured = capsys.readouterr()
        assert captured.out == "plan step 1\n"

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


class TestPrintResponse:
    """Tests for print_response()."""

    @pytest.mark.asyncio
    async def test_prints_assistant_messages(self, capsys):
        from claude_agent_sdk import AssistantMessage, TextBlock

        msg = AssistantMessage(
            content=[TextBlock(text="here is the plan")],
            model="claude-opus-4-6",
        )

        client = MagicMock()

        async def fake_receive():
            yield msg

        client.receive_response = fake_receive
        await print_response(client)
        captured = capsys.readouterr()
        assert "here is the plan" in captured.out

    @pytest.mark.asyncio
    async def test_skips_non_assistant_messages(self, capsys):
        msg = MagicMock(spec=[])

        client = MagicMock()

        async def fake_receive():
            yield msg

        client.receive_response = fake_receive
        await print_response(client)
        captured = capsys.readouterr()
        assert captured.out == ""

    @pytest.mark.asyncio
    async def test_handles_empty_stream(self, capsys):
        client = MagicMock()

        async def fake_receive():
            return
            yield  # noqa: E501 — make it an async generator

        client.receive_response = fake_receive
        await print_response(client)
        captured = capsys.readouterr()
        assert captured.out == ""


class TestReadUserInput:
    """Tests for read_user_input()."""

    @patch("builtins.input", return_value="hello")
    def test_returns_input(self, mock_input):
        result = read_user_input()
        assert result == "hello"

    @patch("builtins.input", return_value="quit")
    def test_quit_returns_none(self, mock_input):
        result = read_user_input()
        assert result is None

    @patch("builtins.input", return_value="exit")
    def test_exit_returns_none(self, mock_input):
        result = read_user_input()
        assert result is None

    @patch("builtins.input", return_value="done")
    def test_done_returns_none(self, mock_input):
        result = read_user_input()
        assert result is None

    @patch("builtins.input", return_value="  QUIT  ")
    def test_quit_case_insensitive_with_whitespace(self, mock_input):
        result = read_user_input()
        assert result is None

    @patch("builtins.input", side_effect=EOFError)
    def test_eof_returns_none(self, mock_input):
        result = read_user_input()
        assert result is None

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_keyboard_interrupt_returns_none(self, mock_input):
        result = read_user_input()
        assert result is None

    @patch("builtins.input", return_value="not a quit command")
    def test_normal_input_returned(self, mock_input):
        result = read_user_input()
        assert result == "not a quit command"


class TestRun:
    """Tests for run()."""

    @pytest.mark.asyncio
    async def test_sends_initial_prompt(self):
        client = AsyncMock()

        async def fake_receive():
            return
            yield  # noqa: E501

        client.receive_response = fake_receive

        with (
            patch(
                "python_agent.planning_agent.ClaudeSDKClient",
            ) as mock_cls,
            patch(
                "python_agent.planning_agent.read_user_input",
                return_value=None,
            ),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(
                return_value=client,
            )
            mock_cls.return_value.__aexit__ = AsyncMock(
                return_value=False,
            )
            await run("build a thing", "claude-opus-4-6")

        client.query.assert_any_call("build a thing")

    @pytest.mark.asyncio
    async def test_loops_on_user_input(self):
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
                return "tell me more"
            return None

        with (
            patch(
                "python_agent.planning_agent.ClaudeSDKClient",
            ) as mock_cls,
            patch(
                "python_agent.planning_agent.read_user_input",
                side_effect=fake_input,
            ),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(
                return_value=client,
            )
            mock_cls.return_value.__aexit__ = AsyncMock(
                return_value=False,
            )
            await run("idea", "claude-opus-4-6")

        client.query.assert_any_call("tell me more")

    @pytest.mark.asyncio
    async def test_exits_on_none_input(self):
        client = AsyncMock()

        async def fake_receive():
            return
            yield  # noqa: E501

        client.receive_response = fake_receive

        with (
            patch(
                "python_agent.planning_agent.ClaudeSDKClient",
            ) as mock_cls,
            patch(
                "python_agent.planning_agent.read_user_input",
                return_value=None,
            ),
        ):
            mock_cls.return_value.__aenter__ = AsyncMock(
                return_value=client,
            )
            mock_cls.return_value.__aexit__ = AsyncMock(
                return_value=False,
            )
            await run("idea", "claude-opus-4-6")

        # Only the initial query, no follow-ups
        assert client.query.call_count == 1


class TestParseArgs:
    """Tests for parse_args()."""

    def test_description_required(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_description_positional(self):
        args = parse_args(["build a web scraper"])
        assert args.description == "build a web scraper"

    def test_default_model(self):
        args = parse_args(["desc"])
        assert args.model == "claude-opus-4-6"

    def test_custom_model(self):
        args = parse_args(["-m", "claude-sonnet-4-6", "desc"])
        assert args.model == "claude-sonnet-4-6"


class TestMain:
    """Tests for main()."""

    @patch("python_agent.planning_agent.asyncio.run")
    def test_returns_zero(self, mock_run):
        result = main(["desc"])
        assert result == 0

    @patch("python_agent.planning_agent.asyncio.run")
    def test_calls_asyncio_run(self, mock_run):
        main(["desc"])
        mock_run.assert_called_once()
