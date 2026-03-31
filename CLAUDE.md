# Python Agent — Project Instructions

## Python Standards

All Python code must meet these standards before commit:

1. **flake8 clean** — zero warnings, default rules
2. **McCabe Cyclomatic Complexity <= 5** — `flake8 --max-complexity=5`
3. **100% branch coverage** — `pytest --cov --cov-branch --cov-report=term-missing`
4. **pytest** for all tests, **pytest-mock** for mocking
5. **pyflakes clean** (included in flake8)
6. **Mutation testing with mutmut** — target 100% kill rate on core modules
   - Use mutmut v2 (`pip install 'mutmut<3'`). v3's test mapping doesn't work with non-standard layouts.
   - Configure in `pyproject.toml`: `paths_to_mutate`, `runner` (must use `.venv/bin/python`).
   - Surviving mutants = test gaps. Write targeted tests to kill them.

### Venv

Always use the project venv. Execute Python through `.venv/bin/python3`,
`.venv/bin/pytest`, `.venv/bin/flake8`, etc. Never use the system Python.

### Before Committing

```bash
.venv/bin/flake8 --max-complexity=5
.venv/bin/pytest --cov --cov-branch --cov-report=term-missing
```

The only acceptable uncovered line is the `if __name__` guard (`sys.exit`).

## Correctness

These are foundational. They apply to all code in all languages.

- **No one writes correct code — not humans, not AI.** Confidence without evidence is the most dangerous state. The cost of catching a bug grows exponentially the later it's found. Verify now, not later.
- **A programmer's critical job is proving their code is correct.** Trust nothing without verification.
- **Failure handling code that is never tested is a liability.** It can generate new errors when it finally runs. When writing functions with failure paths, discuss whether those paths are reachable under test. If not, discuss the cost of making them testable (e.g., dependency injection so tests can supply fakes).
- **Prefer parameters over hardcoded values.** Enables dependency injection and testability.
- **An accidental fix is not a fix — it's a clue.** Ask WHY a change affects behavior before shipping it.
- **Trace symptoms to code paths, not external theories.** When debugging, grep for what produces the output, trace the loop, ask "what are we not exiting and why?" Step UP in abstraction, don't drill down into speculation.

## Testing Philosophy

These are non-negotiable. They come from hard experience across multiple projects.

- **Tests are part of the implementation, not a follow-up.** Code without tests is not done.
- **Both sides of every conditional.** Not "the important ones" — ALL of them.
- **Every test MUST have a meaningful assertion.** Never write a test that calls a function and unconditionally passes. Never write `assert len(x) > 0` when you can assert on the actual value.
- **Test from multiple angles.** Unit test + functional test on the same path catches different bugs.
- **Reproduce runtime bugs in tests first.** Write a test that triggers the failure (red), fix the code (green), commit both together.
- **If unsure what to assert, discuss it first.** "What should this test verify?" is always the right question.
- **The branch coverage analysis itself finds bugs.** Enumerating every conditional and verifying both sides are exercised catches things tests alone miss.
- **Mutation testing is the proof.** If a mutant survives, the test is broken.

## Production vs Prototype

Two modes, hard boundary between them:

- **Production code:** Full standards, thoroughly tested, no shortcuts.
- **Prototype code:** Only when the path forward is unclear. Define the questions it answers upfront. When answered, reimplement from scratch to production standards. Never evolve a prototype into production code.

## Working Style

- Trunk-based development: commit directly to `main`, push after each group of changes.
- Don't suggest breaks or stopping.
- When you catch a mistake or unintended change — stop. Understand what changed and why before moving on. Don't rationalize differences away.
- Never check files into the wrong repo by drifting directories. Stay in this project dir; use absolute paths for anything outside it.
- Always check if a target file exists before `mv`/`cp`/`Write`. Lost uncommitted work is gone forever.
- Never hide unexpected messages or errors — fix the source, not the reporter.
- Revert failed experiments immediately. Only keep changes you have high confidence are correct.

## Git

- Commit after every group of code changes. Don't wait to be asked.
- Committer: edhodapp <ed@hodapp.com>
