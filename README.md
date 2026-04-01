# python-agent

Claude-powered Python agents with ontology-driven planning and enforced
quality standards. BSD 3-Clause.

## Install

```bash
pip install python-agent
```

## Overview

Five agents form a pipeline from project idea to production code:

```
discovery-agent  -->  divergence-agent  -->  convergence-agent  -->  coding-agent
  (interactive)        (autonomous)          (interactive)           (autonomous)
  Build domain         Generate N            Compare, select,        Write code to
  ontology             solution candidates   accept, refine          production standards
```

All state is saved to an **ontology DAG** (a JSON file). Each node is a
complete ontology snapshot. Each edge records a design decision. You can
backtrack to any prior state and explore a different path.

There is also a standalone **planning-agent** for freeform project design
without the ontology pipeline.

## The Ontology Pipeline

### Step 1: Discovery

Build a domain ontology interactively. The agent asks questions about
your domain and constructs entities, relationships, and constraints.

```bash
discovery-agent "A URL shortener service" --dag-file shortener.json
```

Example session:

```
> The agent asks: Who are the users?
> Anyone can follow a link. Registered users create short URLs.

  [Agent proposes entities: User, ShortURL, relationship: User owns ShortURL]

> show
  Entities (2):
    user: User [username, api_key]
    short_url: ShortURL [slug, target_url, click_count]
  Relationships (1):
    user --owns--> short_url (one_to_many)

> save initial domain model
  Saved snapshot: 20260401T120000...

> quit
```

Commands during discovery:
- **show** -- display current ontology
- **save [label]** -- save snapshot to DAG
- **back** -- backtrack to previous snapshot
- **quit/exit/done** -- end session

Options:
- `--dag-file` -- path to DAG JSON file (default: `ontology.json`)
- `-m`, `--model` -- model to use (default: `claude-opus-4-6`)

### Step 2: Divergence

Autonomously generate multiple solution candidates from the domain
ontology. Each candidate fills in modules, classes, functions, data
models, and external dependencies.

```bash
divergence-agent --dag-file shortener.json -n 3
```

Output:

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

The DAG now has three branching children from your domain ontology node,
each with a different architectural approach.

Options:
- `--dag-file` -- path to DAG JSON file (required)
- `-n`, `--num-candidates` -- number of candidates (default: 3)
- `-m`, `--model` -- model to use (default: `claude-sonnet-4-6`)
- `--max-budget` -- USD spending cap (default: 5.0)

### Step 3: Convergence

Compare candidates, select one, and refine it. The LLM assists with
comparisons and answers questions about the architectures.

```bash
convergence-agent --dag-file shortener.json
```

Example session:

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
  [Full ontology with modules, classes, functions, test strategies]

> accept
  Accepted: monolith-sqlite. You can now refine.

> Add rate limiting to the API module
  [LLM proposes ontology update with a new RateLimiter class]

> save final design
  Saved snapshot: 20260401T130000...

> quit
```

Commands during convergence:
- **list** -- show candidate children of current node
- **select \<n\>** -- navigate to the nth candidate
- **back** -- go to parent node
- **show** -- display current ontology
- **accept** -- mark current candidate as chosen (enables refinement)
- **save [label]** -- save snapshot
- Any other text is sent to the LLM (e.g., "compare", "explain", "refine")
- **quit/exit/done** -- end session

Options:
- `--dag-file` -- path to DAG JSON file (required)
- `-m`, `--model` -- model to use (default: `claude-opus-4-6`)

### Step 4: Coding

Run the coding agent against a project directory. It writes code, runs
all quality checks, and iterates until everything passes.

```bash
coding-agent "Implement the URL shortener from the accepted design" -d /path/to/project
```

Uses Sonnet by default for cost efficiency. If Sonnet gets stuck (error
or hits the turn limit), automatically escalates to Opus using the
remaining budget.

Options:
- `-d`, `--project-dir` -- target project (default: current dir)
- `-m`, `--model` -- initial model (default: `claude-sonnet-4-6`)
- `--max-turns` -- agent step limit (default: 30)
- `--max-budget` -- USD spending cap (default: 5.0)

### Backtracking

At any point you can re-run convergence on the same DAG file to navigate
back and explore a different branch:

```bash
# Go back and try the microservices approach instead
convergence-agent --dag-file shortener.json
> back
> select 2
> accept
```

All intermediate states are preserved in the DAG.

## Standalone Planning Agent

For freeform project design without the ontology pipeline:

```bash
planning-agent "A CLI tool that converts CSV files to JSON with schema validation"
```

Uses Opus by default. Produces a structured markdown plan document.
Type `quit`, `exit`, or `done` to end the session.

## The Ontology Format

The ontology captures both problem and solution domains as typed Python
dataclasses serialized to JSON:

**Problem domain:** Entity, Property, Relationship, DomainConstraint
**Solution domain:** ModuleSpec, ClassSpec, FunctionSpec, DataModel, ExternalDependency
**Planning state:** OpenQuestion (unresolved decisions)
**DAG:** nodes (ontology snapshots), edges (design decisions)

See `src/python_agent/ontology.py` for the full schema.

## Quality Standards

All code produced by these agents (and the agents themselves) must meet:

1. **flake8 clean** with `--max-complexity=5`
2. **100% branch coverage** via pytest
3. **100% mutant kill rate** via mutmut v2 (except `if __name__` guard)
4. **Fuzz testing** via hypothesis on all external-input functions
5. **Meaningful assertions** on every test
6. **Functional test gap analysis** as final step

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
.venv/bin/pytest --cov --cov-branch --cov-report=term-missing
.venv/bin/mutmut run
.venv/bin/pytest tests/test_fuzz.py --hypothesis-profile=ci
```

## License

BSD 3-Clause. See LICENSE.
