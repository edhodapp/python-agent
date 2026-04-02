# python-agent

[![Lint](https://github.com/edhodapp/python-agent/actions/workflows/lint.yml/badge.svg)](https://github.com/edhodapp/python-agent/actions/workflows/lint.yml)
[![Type Check](https://github.com/edhodapp/python-agent/actions/workflows/typecheck.yml/badge.svg)](https://github.com/edhodapp/python-agent/actions/workflows/typecheck.yml)
[![Test](https://github.com/edhodapp/python-agent/actions/workflows/test.yml/badge.svg)](https://github.com/edhodapp/python-agent/actions/workflows/test.yml)
[![Taint Analysis](https://github.com/edhodapp/python-agent/actions/workflows/taint.yml/badge.svg)](https://github.com/edhodapp/python-agent/actions/workflows/taint.yml)
[![Fuzz](https://github.com/edhodapp/python-agent/actions/workflows/fuzz.yml/badge.svg)](https://github.com/edhodapp/python-agent/actions/workflows/fuzz.yml)
[![Mutation](https://github.com/edhodapp/python-agent/actions/workflows/mutation.yml/badge.svg)](https://github.com/edhodapp/python-agent/actions/workflows/mutation.yml)

Claude-powered Python agents with ontology-driven project planning,
autonomous code generation, and defense-in-depth security hardening.

The pipeline takes a project from idea to production code through
structured ontology exploration, branching solution candidates, and
an autonomous coding agent that enforces 10 quality gates before
every commit.

**BSD 3-Clause.** Python 3.11+. Requires the
[Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python).

## What This Is

A monorepo containing six CLI tools and a shared ontology framework:

| Tool | Mode | Purpose |
|------|------|---------|
| `discovery-agent` | Interactive | Build a domain ontology through conversation |
| `divergence-agent` | Autonomous | Generate N candidate solution architectures |
| `convergence-agent` | Interactive | Compare, select, and refine candidates |
| `coding-agent` | Autonomous | Write production-quality code with Sonnet/Opus escalation |
| `planning-agent` | Interactive | Freeform project design (no ontology) |
| `call-graph` | Analysis | Source-to-sink taint analysis with CWE tagging |

Plus shared infrastructure:

| Module | Purpose |
|--------|---------|
| `ontology.py` | 16 Pydantic models: entities, relationships, modules, DAG |
| `types.py` | `Annotated` + `Literal` shared type definitions |
| `dag_utils.py` | DAG persistence with HMAC integrity signing |
| `dag_integrity.py` | HMAC verification + injection pattern scanning |
| `tool_guard.py` | `can_use_tool` callback: Bash blocklist + path confinement |
| `rules.py` | System prompts with `frame_data()` content framing |

## Install

```bash
pip install python-agent
```

For development (includes test/analysis tools):

```bash
git clone https://github.com/edhodapp/python-agent.git
cd python-agent
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## The Ontology Pipeline

```
discovery-agent  -->  divergence-agent  -->  convergence-agent  -->  coding-agent
  (interactive)        (autonomous)          (interactive)           (autonomous)
  Build domain         Generate N            Compare, select,        Write code to
  ontology             solution candidates   accept, refine          production standards
```

All state is saved to an **ontology DAG** (a JSON file). Each node is a
complete ontology snapshot. Each edge records a design decision. You can
backtrack to any prior state and explore a different path.

### Step 1: Discovery

Build a domain ontology interactively. The agent asks questions about
your domain and constructs entities, relationships, and constraints.

```bash
discovery-agent "A URL shortener service" --dag-file shortener.json
```

Example session:

```
Planner: I'll help you design a URL shortener. Who are the users?

> Anyone can follow a link. Registered users create short URLs.

  [Agent proposes entities: User, ShortURL, relationship: User owns ShortURL]

> show
  Entities (2):
    user: User [username, api_key]
    short_url: ShortURL [slug, target_url, click_count]
  Relationships (1):
    user --owns--> short_url (one_to_many)
  Open Questions (2):
    [open] q1: Storage backend?
    [open] q2: Slug format?

> save initial domain model
  Saved snapshot: 20260401T120000...

> quit
```

Commands: `show`, `save [label]`, `back`, `quit`/`exit`/`done`

Options:
- `--dag-file PATH` -- DAG JSON file (default: `ontology.json`)
- `-m MODEL` -- model (default: `claude-opus-4-6`)

### Step 2: Divergence

Autonomously generate multiple solution candidates. The agent identifies
key architectural decision points, then generates one complete solution
per strategy.

```bash
divergence-agent --dag-file shortener.json -n 3
```

```
Identifying 3 strategies...
Generating candidate: monolith-sqlite...
  Created: monolith-sqlite
Generating candidate: microservices-postgres...
  Created: microservices-postgres
Generating candidate: serverless-dynamo...
  Created: serverless-dynamo

Done. 3 candidates. Cost: $0.1234
```

Each candidate fills in the solution domain: modules, classes, functions,
data models, external dependencies, and test strategies. The DAG now has
three branching children.

Options:
- `--dag-file PATH` -- DAG JSON file (required)
- `-n N` -- number of candidates (default: 3)
- `-m MODEL` -- model (default: `claude-sonnet-4-6`)
- `--max-budget USD` -- spending cap (default: 5.0)

### Step 3: Convergence

Compare candidates, select one, and refine it interactively. The LLM
has context of all candidates and assists with comparisons.

```bash
convergence-agent --dag-file shortener.json
```

```
> list
  1. monolith-sqlite: Entities (2), Modules (4)...
  2. microservices-postgres: Entities (2), Modules (6)...
  3. serverless-dynamo: Entities (2), Modules (5)...

> compare monolith-sqlite and microservices-postgres on complexity
  [LLM explains trade-offs between the two approaches]

> select 1
  Selected: monolith-sqlite

> show
  [Full ontology: entities, relationships, modules with classes/functions]

> accept
  Accepted: monolith-sqlite. You can now refine.

> Add rate limiting to the API module
  [LLM proposes ontology update with new RateLimiter class]

> save final design
> quit
```

Commands: `list`, `select <n>`, `back`, `show`, `accept`, `save [label]`.
Any other text goes to the LLM (e.g., "compare", "explain", "refine").

Options:
- `--dag-file PATH` -- DAG JSON file (required)
- `-m MODEL` -- model (default: `claude-opus-4-6`)

### Step 4: Coding

The coding agent writes code, runs all quality checks, and iterates until
everything passes. Starts with Sonnet for cost efficiency; automatically
escalates to Opus if it gets stuck.

```bash
coding-agent "Implement the URL shortener from the accepted design" -d ./shortener
```

The agent's workflow (11 steps):
1. Read existing code
2. Write/modify code
3. flake8 (complexity <= 5)
4. mypy --strict
5. pytest (100% branch coverage)
6. Iterate on failures
7. mutmut (100% kill rate)
8. Fuzz tests for external-input functions
9. call-graph taint analysis
10. Functional test gap analysis
11. Commit

If the agent can't fix an issue (e.g., needs a `# type: ignore`), it
presents grouped findings for user approval rather than silently
suppressing.

Options:
- `-d DIR` -- project directory (default: `.`)
- `-m MODEL` -- initial model (default: `claude-sonnet-4-6`)
- `--max-turns N` -- step limit (default: 30)
- `--max-budget USD` -- spending cap (default: 5.0)

### Backtracking

Re-run convergence on the same DAG to navigate back and try a different
branch. All intermediate states are preserved:

```bash
convergence-agent --dag-file shortener.json
> back
> select 2
> accept
```

## Standalone Planning Agent

For freeform project design without the ontology pipeline:

```bash
planning-agent "A CLI tool that converts CSV to JSON with schema validation"
```

Uses Opus by default. Produces a structured markdown plan. Type `quit` to end.

## Static Analysis: call-graph

Source-to-sink taint analysis using Python's `ast` module. Traces data
flow from external inputs through the call graph to dangerous sinks.
Each finding tagged with a CWE code.

```bash
call-graph src/python_agent/                    # text report
call-graph src/python_agent/ --sarif            # SARIF JSON for CI
call-graph src/python_agent/ --include-sanitized  # show all paths
```

Sources detected: `input()`, `json.loads`, `open()`, `.model_validate()`,
`.parse_args()`, `.query()` (SDK responses).

Sinks detected: `eval`/`exec`, `subprocess`/`os.system`, `.write()`,
`.query()` (prompt injection), `print()` (info exposure).

Suppress acknowledged findings with mandatory comments:
```python
# taint: ignore[CWE-200] -- Interactive agent displays LLM output to user
async def run(description: str, model: str) -> None:
```

## Security Hardening

Defense-in-depth across all agents:

| Layer | Defense | Protects Against |
|-------|---------|-----------------|
| 1 | `frame_data()` content framing | Prompt injection via embedded data |
| 2 | HMAC-SHA256 DAG integrity | File tampering between sessions |
| 3 | Pydantic `BaseModel` validation | Malformed data at construction |
| 4 | `can_use_tool` callback (tool guard) | Dangerous Bash commands + path escape |
| 5 | Injection pattern scanner | Common injection phrases in text fields |
| 6 | Framing escape detection | `</ontology-data>` breakout attempts |
| 7 | Call graph taint analysis | Unguarded source-to-sink data flows |
| 8 | Taint suppressions with mandatory comments | Acknowledged risks with audit trail |
| 9 | User approval workflow | Silent suppression by autonomous agent |

The coding agent's tool guard blocks: `curl`, `wget`, `ssh`, `sudo`,
`rm -rf /`, `dd`, `mkfs`, `chmod 777`, `chown`, `pkill`, writes to
`/etc`, `~/.ssh`, `~/.bashrc`. File operations confined to project
directory.

## The Ontology Format

16 Pydantic models capturing both problem and solution domains:

**Problem domain:** Entity, Property, PropertyType, Relationship,
DomainConstraint

**Solution domain:** ModuleSpec, ClassSpec, FunctionSpec, DataModel,
ExternalDependency

**Planning state:** OpenQuestion (unresolved decisions)

**DAG:** DAGNode (ontology snapshots), DAGEdge (design decisions),
OntologyDAG (the full versioned graph)

Type constraints enforced via `Annotated` types: `SafeId` (alphanumeric,
max 100 chars), `ShortName` (max 100 chars), `Description` (max 2000
chars). Enum fields use `Literal` types: `PropertyKind`, `Cardinality`,
`ModuleStatus`, `Priority`.

## Quality Standards

All code produced by these agents (and the agents themselves) meets:

1. **flake8 clean** with `--max-complexity=5`
2. **mypy --strict** with zero errors
3. **100% branch coverage** via pytest (570 tests)
4. **100% mutant kill rate** via mutmut v2
5. **Fuzz testing** via hypothesis on all external-input functions
6. **Call graph taint analysis** with CWE tagging
7. **Functional test gap analysis** as final verification step
8. **Prompt injection hardening** across all agents

See `CLAUDE.md` for the complete coding standards.

## Project Status

**Version:** 0.1.0

**What works:**
- Full ontology pipeline: discovery, divergence, convergence
- Autonomous coding agent with Sonnet/Opus escalation
- All security hardening layers active
- 570 tests, 14 source files, all quality gates pass
- call-graph reports clean (no unguarded taint paths)

**What's next:**
- Wire coding agent to consume ontology nodes as implementation specs
- End-to-end pipeline test (idea to running code)
- CI setup (GitHub Actions)
- Performance requirements as soft ontology constraints (advisory, not blocking)
- Tier 2 hardening: coding agent command audit log by default

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Full quality gate
.venv/bin/flake8 --max-complexity=5 src/ tests/
.venv/bin/mypy --strict src/
.venv/bin/pytest --cov --cov-branch --cov-report=term-missing
.venv/bin/mutmut run
.venv/bin/pytest tests/test_fuzz.py --hypothesis-profile=ci
.venv/bin/call-graph src/python_agent/
```

## Local PyPI with devpi

```bash
# Setup (first time)
pip install devpi-server devpi-client
devpi-server --init && devpi-server --start --port 3141
devpi use http://localhost:3141
devpi user -c myuser password=
devpi login myuser
devpi index -c dev bases=root/pypi
devpi use myuser/dev

# Publish
devpi upload

# Install from local index
pip install python-agent -i http://localhost:3141/myuser/dev/+simple/
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
