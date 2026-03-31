# python-agent

Claude-powered Python coding and planning agents with enforced quality
standards. BSD 3-Clause.

## Install

```bash
pip install python-agent
```

## Usage

### Planning Agent (interactive)

Design a project before writing code:

```bash
planning-agent "A CLI tool that converts CSV files to JSON with schema validation"
```

Uses Opus by default for judgment-heavy work. Override with `-m claude-sonnet-4-6`.

The agent asks clarifying questions, explores tradeoffs, and produces a
structured plan document. Type `quit`, `exit`, or `done` to end the session.

### Coding Agent (background)

Run against any project directory:

```bash
coding-agent "Add input validation to the parse_config function" -d /path/to/project
```

Uses Sonnet by default for cost efficiency. Override with `-m claude-opus-4-6`.

Options:
- `-d`, `--project-dir` — target project (default: current dir)
- `-m`, `--model` — Claude model (default: `claude-sonnet-4-6`)
- `--max-turns` — agent step limit (default: 30)
- `--max-budget` — USD spending cap (default: 5.0)

The agent writes code, runs flake8, pytest, and mutmut, and iterates
until all quality checks pass.

## Quality Standards

All code produced by these agents (and the agents themselves) must meet:

1. **flake8 clean** with `--max-complexity=5`
2. **100% branch coverage** via pytest
3. **100% mutant kill rate** via mutmut v2
4. **Meaningful assertions** on every test

See `CLAUDE.md` for the full rules.

## Local PyPI with devpi

### Setup

```bash
pip install devpi-server devpi-client

# Start the server (first time)
devpi-server --init
devpi-server --start --port 3141

# Create your index
devpi use http://localhost:3141
devpi user -c ed password=
devpi login ed
devpi index -c dev bases=root/pypi
devpi use ed/dev
```

### Publish locally

```bash
devpi upload
```

### Install from local index (in other projects)

```bash
pip install python-agent -i http://localhost:3141/ed/dev/+simple/
```

### Push to real PyPI

```bash
# Configure PyPI credentials first (API token)
devpi push python-agent==0.1.0 pypi:main
```

### Permanent pip config (optional)

Add to `~/.pip/pip.conf`:

```ini
[global]
index-url = http://localhost:3141/ed/dev/+simple/
```

This makes all `pip install` commands use your local devpi first,
falling back to PyPI for packages you don't host locally.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Quality checks
.venv/bin/flake8 --max-complexity=5 src/ tests/
.venv/bin/pytest --cov=python_agent --cov-branch --cov-report=term-missing
.venv/bin/mutmut run
```

## License

BSD 3-Clause. See LICENSE.
