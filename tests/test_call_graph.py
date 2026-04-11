"""Comprehensive tests for call_graph taint analyzer."""

from __future__ import annotations

import ast
import json
import os
import textwrap
from typing import Any

from python_agent.call_graph import (
    CallEdge,
    CallGraph,
    FunctionInfo,
    TaintPath,
    _attr_parts,
    _bfs_to_sinks,
    _build_forward_adj,
    _check_sink_hit,
    _classify_as_sink,
    _classify_as_source,
    _collect_calls_in_body,
    _collect_imports,
    _collect_python_files,
    _enqueue_neighbors,
    _find_sinks,
    _find_sources,
    _is_sanitizer_name,
    _is_sink_call,
    _is_source_call,
    _module_name_from_path,
    _path_has_sanitizer,
    _resolve_call_name,
    _resolve_callee,
    _sarif_result,
    build_graph,
    find_taint_paths,
    format_sarif,
    format_text_report,
    main,
    parse_args,
    parse_file,
)


# -----------------------------------------------------------------------
# _resolve_call_name
# -----------------------------------------------------------------------

class TestResolveCallName:
    def test_simple_name(self) -> None:
        tree = ast.parse("foo()")
        call = tree.body[0].value  # type: ignore[attr-defined]
        assert _resolve_call_name(call) == "foo"

    def test_attribute(self) -> None:
        tree = ast.parse("os.path.join()")
        call = tree.body[0].value  # type: ignore[attr-defined]
        assert _resolve_call_name(call) == "os.path.join"

    def test_single_attribute(self) -> None:
        tree = ast.parse("obj.method()")
        call = tree.body[0].value  # type: ignore[attr-defined]
        assert _resolve_call_name(call) == "obj.method"

    def test_subscript_call(self) -> None:
        """Call on subscript like x[0]() returns empty string."""
        tree = ast.parse("x[0]()")
        call = tree.body[0].value  # type: ignore[attr-defined]
        assert _resolve_call_name(call) == ""

    def test_call_on_call(self) -> None:
        """Call like foo()() — inner func is a Call, not Name/Attr."""
        tree = ast.parse("foo()()")
        call = tree.body[0].value  # type: ignore[attr-defined]
        assert _resolve_call_name(call) == ""


# -----------------------------------------------------------------------
# _attr_parts
# -----------------------------------------------------------------------

class TestAttrParts:
    def test_non_name_base(self) -> None:
        """Attribute on a non-Name, non-Attribute base returns []."""
        tree = ast.parse("x[0].bar")
        # x[0].bar is an Attribute where value is Subscript
        attr_node = tree.body[0].value  # type: ignore[attr-defined]
        assert _attr_parts(attr_node) == []

    def test_deep_nesting(self) -> None:
        tree = ast.parse("a.b.c.d")
        attr_node = tree.body[0].value  # type: ignore[attr-defined]
        assert _attr_parts(attr_node) == ["a", "b", "c", "d"]


# -----------------------------------------------------------------------
# _is_source_call
# -----------------------------------------------------------------------

class TestIsSourceCall:
    def test_exact_input(self) -> None:
        assert _is_source_call("input") == (True, "CWE-20")

    def test_exact_json_loads(self) -> None:
        assert _is_source_call("json.loads") == (True, "CWE-502")

    def test_exact_json_load(self) -> None:
        assert _is_source_call("json.load") == (True, "CWE-502")

    def test_exact_open(self) -> None:
        assert _is_source_call("open") == (True, "CWE-73")

    def test_exact_sys_stdin_read(self) -> None:
        assert _is_source_call("sys.stdin.read") == (True, "CWE-73")

    def test_exact_sys_stdin_readline(self) -> None:
        assert _is_source_call("sys.stdin.readline") == (True, "CWE-73")

    def test_suffix_read_no_longer_flagged(self) -> None:
        """Issue #5: `.read` suffix match removed (too broad)."""
        assert _is_source_call("file.read") == (False, "")

    def test_suffix_query_no_longer_flagged(self) -> None:
        """Issue #5: `.query` suffix match removed for sources (too broad)."""
        assert _is_source_call("db.query") == (False, "")

    def test_suffix_model_validate_still_flagged(self) -> None:
        """Issue #5 trade-off: `.model_validate` kept (distinctive Pydantic name)."""
        assert _is_source_call("Foo.model_validate") == (True, "CWE-502")

    def test_suffix_model_validate_json_still_flagged(self) -> None:
        """Issue #5 trade-off: `.model_validate_json` kept (Pydantic)."""
        assert _is_source_call("Foo.model_validate_json") == (True, "CWE-502")

    def test_suffix_parse_args_still_flagged(self) -> None:
        """Issue #5 trade-off: `.parse_args` kept (distinctive argparse name)."""
        assert _is_source_call("parser.parse_args") == (True, "CWE-20")

    def test_not_source(self) -> None:
        assert _is_source_call("print") == (False, "")


# -----------------------------------------------------------------------
# _is_sink_call
# -----------------------------------------------------------------------

class TestIsSinkCall:
    def test_exact_eval(self) -> None:
        assert _is_sink_call("eval") == (True, "CWE-94")

    def test_exact_exec(self) -> None:
        assert _is_sink_call("exec") == (True, "CWE-94")

    def test_exact_os_system(self) -> None:
        assert _is_sink_call("os.system") == (True, "CWE-78")

    def test_exact_os_popen(self) -> None:
        assert _is_sink_call("os.popen") == (True, "CWE-78")

    def test_exact_subprocess_run(self) -> None:
        assert _is_sink_call("subprocess.run") == (True, "CWE-78")

    def test_exact_subprocess_call(self) -> None:
        assert _is_sink_call("subprocess.call") == (True, "CWE-78")

    def test_exact_subprocess_popen(self) -> None:
        assert _is_sink_call("subprocess.Popen") == (True, "CWE-78")

    def test_exact_subprocess_check_output(self) -> None:
        assert _is_sink_call("subprocess.check_output") == (True, "CWE-78")

    def test_exact_subprocess_check_call(self) -> None:
        assert _is_sink_call("subprocess.check_call") == (True, "CWE-78")

    def test_exact_print(self) -> None:
        assert _is_sink_call("print") == (True, "CWE-200")

    def test_suffix_write(self) -> None:
        assert _is_sink_call("file.write") == (True, "CWE-73")

    def test_suffix_query_sink(self) -> None:
        assert _is_sink_call("cursor.query") == (True, "CWE-74")

    def test_not_sink(self) -> None:
        assert _is_sink_call("len") == (False, "")


# -----------------------------------------------------------------------
# _is_sanitizer_name
# -----------------------------------------------------------------------

class TestIsSanitizerName:
    def test_bare_match(self) -> None:
        assert _is_sanitizer_name("frame_data") is True

    def test_dotted_match(self) -> None:
        assert _is_sanitizer_name("mod.is_safe_bash") is True

    def test_validate_ontology_strict(self) -> None:
        assert _is_sanitizer_name("validate_ontology_strict") is True

    def test_is_safe_path(self) -> None:
        assert _is_sanitizer_name("is_safe_path") is True

    def test_model_validate(self) -> None:
        assert _is_sanitizer_name("model_validate") is True

    def test_non_sanitizer(self) -> None:
        assert _is_sanitizer_name("do_stuff") is False


# -----------------------------------------------------------------------
# _collect_calls_in_body
# -----------------------------------------------------------------------

class TestCollectCallsInBody:
    def test_simple_body(self) -> None:
        src = textwrap.dedent("""\
            def f():
                x = input()
                print(x)
        """)
        tree = ast.parse(src)
        func = tree.body[0]
        calls = _collect_calls_in_body(func.body)  # type: ignore[attr-defined]
        assert "input" in calls
        assert "print" in calls

    def test_empty_body(self) -> None:
        src = textwrap.dedent("""\
            def f():
                pass
        """)
        tree = ast.parse(src)
        func = tree.body[0]
        calls = _collect_calls_in_body(func.body)  # type: ignore[attr-defined]
        assert calls == []


# -----------------------------------------------------------------------
# _classify_as_source / _classify_as_sink
# -----------------------------------------------------------------------

class TestClassify:
    def test_source_hit(self) -> None:
        assert _classify_as_source(["foo", "input"]) == (True, "CWE-20")

    def test_source_miss(self) -> None:
        assert _classify_as_source(["foo", "bar"]) == (False, "")

    def test_source_empty(self) -> None:
        assert _classify_as_source([]) == (False, "")

    def test_sink_hit(self) -> None:
        assert _classify_as_sink(["foo", "eval"]) == (True, "CWE-94")

    def test_sink_miss(self) -> None:
        assert _classify_as_sink(["foo", "bar"]) == (False, "")

    def test_sink_empty(self) -> None:
        assert _classify_as_sink([]) == (False, "")


# -----------------------------------------------------------------------
# AST visitors via _collect_imports
# -----------------------------------------------------------------------

class TestCollectImports:
    def test_import(self) -> None:
        tree = ast.parse("import os")
        result = _collect_imports(tree)
        assert result == {"os": "os"}

    def test_import_alias(self) -> None:
        tree = ast.parse("import numpy as np")
        result = _collect_imports(tree)
        assert result == {"np": "numpy"}

    def test_from_import(self) -> None:
        tree = ast.parse("from os.path import join")
        result = _collect_imports(tree)
        assert result == {"join": "os.path.join"}

    def test_from_import_alias(self) -> None:
        tree = ast.parse("from os.path import join as j")
        result = _collect_imports(tree)
        assert result == {"j": "os.path.join"}

    def test_from_import_no_module(self) -> None:
        """Relative import with no module (level-only)."""
        tree = ast.parse("from . import foo")
        # For ast: module=None for relative imports
        # Our code handles this: mod = node.module or ""
        result = _collect_imports(tree)
        assert result == {"foo": "foo"}


# -----------------------------------------------------------------------
# _resolve_callee
# -----------------------------------------------------------------------

class TestResolveCallee:
    def test_known_import_simple(self) -> None:
        imports = {"os": "os"}
        assert _resolve_callee("os", imports, "mymod") == "os"

    def test_known_import_dotted(self) -> None:
        imports = {"os": "os"}
        result = _resolve_callee("os.path", imports, "mymod")
        assert result == "os.path"

    def test_bare_local(self) -> None:
        assert _resolve_callee("foo", {}, "mymod") == "mymod.foo"

    def test_dotted_unknown(self) -> None:
        assert _resolve_callee("x.y", {}, "mymod") == "x.y"


# -----------------------------------------------------------------------
# _module_name_from_path
# -----------------------------------------------------------------------

class TestModuleNameFromPath:
    def test_simple(self) -> None:
        result = _module_name_from_path("/src/foo/bar.py", "/src")
        assert result == "foo.bar"

    def test_nested(self) -> None:
        result = _module_name_from_path("/src/a/b/c.py", "/src")
        assert result == "a.b.c"


# -----------------------------------------------------------------------
# _collect_python_files
# -----------------------------------------------------------------------

class TestCollectPythonFiles:
    def test_basic(self, tmp_path: Any) -> None:
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.txt").write_text("hello")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "c.py").write_text("y = 2")
        result = _collect_python_files(str(tmp_path))
        basenames = [os.path.basename(p) for p in result]
        assert "a.py" in basenames
        assert "c.py" in basenames
        assert "b.txt" not in basenames

    def test_empty_dir(self, tmp_path: Any) -> None:
        assert _collect_python_files(str(tmp_path)) == []


# -----------------------------------------------------------------------
# parse_file
# -----------------------------------------------------------------------

class TestParseFile:
    def test_basic(self, tmp_path: Any) -> None:
        src = textwrap.dedent("""\
            import json

            def load_data():
                return json.loads('{}')

            def process(data):
                eval(data)
        """)
        p = tmp_path / "mod.py"
        p.write_text(src)
        funcs, edges, _ = parse_file(str(p), "mod")
        names = {f.name for f in funcs}
        assert "mod.load_data" in names
        assert "mod.process" in names

        load_data = next(f for f in funcs if f.name == "mod.load_data")
        assert load_data.is_source is True
        assert load_data.source_cwe == "CWE-502"

        process = next(f for f in funcs if f.name == "mod.process")
        assert process.is_sink is True
        assert process.sink_cwe == "CWE-94"

    def test_async_function(self, tmp_path: Any) -> None:
        src = textwrap.dedent("""\
            async def read_input():
                return input()
        """)
        p = tmp_path / "amod.py"
        p.write_text(src)
        funcs, edges, _ = parse_file(str(p), "amod")
        assert len(funcs) == 1
        assert funcs[0].is_source is True

    def test_edges_produced(self, tmp_path: Any) -> None:
        src = textwrap.dedent("""\
            def caller():
                callee()

            def callee():
                pass
        """)
        p = tmp_path / "e.py"
        p.write_text(src)
        funcs, edges, _ = parse_file(str(p), "e")
        assert len(edges) == 1
        assert edges[0].caller == "e.caller"
        assert edges[0].callee == "e.callee"


# -----------------------------------------------------------------------
# build_graph
# -----------------------------------------------------------------------

class TestParseFileSyntaxError:
    """Tests for parse_file with SyntaxError handling."""

    def test_syntax_error_returns_empty(
        self, tmp_path: Any,
    ) -> None:
        p = tmp_path / "bad.py"
        p.write_text("def f(:\n    pass\n")
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            funcs, edges, supps = parse_file(
                str(p), "bad",
            )
        assert funcs == []
        assert edges == []
        assert supps == []
        syntax_warns = [
            x for x in w
            if "SyntaxError" in str(x.message)
        ]
        assert len(syntax_warns) == 1


class TestBuildGraph:
    def test_basic(self, tmp_path: Any) -> None:
        src = textwrap.dedent("""\
            def foo():
                bar()

            def bar():
                pass
        """)
        (tmp_path / "mod.py").write_text(src)
        graph = build_graph(str(tmp_path))
        assert "mod.foo" in graph.functions
        assert "mod.bar" in graph.functions
        assert len(graph.edges) == 1

    def test_multi_file(self, tmp_path: Any) -> None:
        (tmp_path / "a.py").write_text("def fa():\n    pass\n")
        (tmp_path / "b.py").write_text("def fb():\n    pass\n")
        graph = build_graph(str(tmp_path))
        assert "a.fa" in graph.functions
        assert "b.fb" in graph.functions


# -----------------------------------------------------------------------
# _build_forward_adj
# -----------------------------------------------------------------------

class TestAdjacencyBuilders:
    def _make_graph(self) -> CallGraph:
        return CallGraph(
            functions={},
            edges=[
                CallEdge(caller="a", callee="b", lineno=1),
                CallEdge(caller="a", callee="c", lineno=2),
                CallEdge(caller="b", callee="c", lineno=3),
            ],
        )

    def test_forward(self) -> None:
        g = self._make_graph()
        fwd = _build_forward_adj(g)
        assert fwd["a"] == ["b", "c"]
        assert fwd["b"] == ["c"]


# -----------------------------------------------------------------------
# _find_sources / _find_sinks
# -----------------------------------------------------------------------

class TestFindSourcesSinks:
    def test_find_sources(self) -> None:
        g = CallGraph(functions={
            "a": FunctionInfo(
                module="m", name="a", lineno=1,
                is_source=True, source_cwe="CWE-20",
            ),
            "b": FunctionInfo(
                module="m", name="b", lineno=2,
            ),
        })
        assert _find_sources(g) == ["a"]

    def test_find_sinks(self) -> None:
        g = CallGraph(functions={
            "a": FunctionInfo(
                module="m", name="a", lineno=1,
                is_sink=True, sink_cwe="CWE-94",
            ),
            "b": FunctionInfo(
                module="m", name="b", lineno=2,
            ),
        })
        assert _find_sinks(g) == {"a": "CWE-94"}


# -----------------------------------------------------------------------
# _check_sink_hit
# -----------------------------------------------------------------------

class TestCheckSinkHit:
    def test_hit(self) -> None:
        sinks = {"sink_fn": "CWE-94"}
        result = _check_sink_hit("sink_fn", "start", ["start", "sink_fn"], sinks)
        assert result == ("sink_fn", ["start", "sink_fn"], "CWE-94")

    def test_miss(self) -> None:
        result = _check_sink_hit("other", "start", ["start", "other"], {"sink_fn": "CWE-94"})
        assert result is None

    def test_start_is_sink(self) -> None:
        """Start node should not be reported as a sink hit."""
        sinks = {"start": "CWE-94"}
        result = _check_sink_hit("start", "start", ["start"], sinks)
        assert result is None


# -----------------------------------------------------------------------
# _enqueue_neighbors
# -----------------------------------------------------------------------

class TestEnqueueNeighbors:
    def test_adds_unvisited(self) -> None:
        fwd: dict[str, list[str]] = {"a": ["b", "c"]}
        visited: set[str] = {"b"}
        queue: list[list[str]] = []
        _enqueue_neighbors("a", ["a"], fwd, visited, queue)
        assert queue == [["a", "c"]]

    def test_no_neighbors(self) -> None:
        queue: list[list[str]] = []
        _enqueue_neighbors("x", ["x"], {}, set(), queue)
        assert queue == []


# -----------------------------------------------------------------------
# _bfs_to_sinks
# -----------------------------------------------------------------------

class TestBfsToSinks:
    def test_finds_sink(self) -> None:
        fwd: dict[str, list[str]] = {"a": ["b"], "b": ["c"]}
        sinks = {"c": "CWE-94"}
        g = CallGraph()
        result = _bfs_to_sinks("a", fwd, sinks, g)
        assert len(result) == 1
        assert result[0][0] == "c"
        assert result[0][2] == "CWE-94"

    def test_no_path(self) -> None:
        fwd: dict[str, list[str]] = {"a": ["b"]}
        sinks = {"c": "CWE-94"}
        g = CallGraph()
        result = _bfs_to_sinks("a", fwd, sinks, g)
        assert result == []

    def test_start_is_sink_not_reported(self) -> None:
        """If start is itself a sink, it should not self-report."""
        fwd: dict[str, list[str]] = {"a": ["b"]}
        sinks = {"a": "CWE-94", "b": "CWE-78"}
        g = CallGraph()
        result = _bfs_to_sinks("a", fwd, sinks, g)
        assert len(result) == 1
        assert result[0][0] == "b"


# -----------------------------------------------------------------------
# _path_has_sanitizer
# -----------------------------------------------------------------------

class TestPathHasSanitizer:
    def test_has_sanitizer(self) -> None:
        g = CallGraph(functions={
            "a": FunctionInfo(module="m", name="a", lineno=1),
            "b": FunctionInfo(
                module="m", name="b", lineno=2, is_sanitizer=True,
            ),
            "c": FunctionInfo(module="m", name="c", lineno=3),
        })
        assert _path_has_sanitizer(["a", "b", "c"], g) is True

    def test_no_sanitizer(self) -> None:
        g = CallGraph(functions={
            "a": FunctionInfo(module="m", name="a", lineno=1),
            "c": FunctionInfo(module="m", name="c", lineno=3),
        })
        assert _path_has_sanitizer(["a", "c"], g) is False

    def test_unknown_node(self) -> None:
        """Node not in functions dict is not a sanitizer."""
        g = CallGraph()
        assert _path_has_sanitizer(["unknown"], g) is False


# -----------------------------------------------------------------------
# find_taint_paths
# -----------------------------------------------------------------------

class TestFindTaintPaths:
    def test_end_to_end(self) -> None:
        g = CallGraph(
            functions={
                "src": FunctionInfo(
                    module="m", name="src", lineno=1,
                    is_source=True, source_cwe="CWE-20",
                ),
                "mid": FunctionInfo(
                    module="m", name="mid", lineno=2,
                ),
                "snk": FunctionInfo(
                    module="m", name="snk", lineno=3,
                    is_sink=True, sink_cwe="CWE-94",
                ),
            },
            edges=[
                CallEdge(caller="src", callee="mid", lineno=1),
                CallEdge(caller="mid", callee="snk", lineno=2),
            ],
        )
        paths = find_taint_paths(g)
        assert len(paths) == 1
        assert paths[0].source == "src"
        assert paths[0].sink == "snk"
        assert paths[0].cwe == "CWE-94"
        assert paths[0].sanitized is False

    def test_sanitized_path(self) -> None:
        g = CallGraph(
            functions={
                "src": FunctionInfo(
                    module="m", name="src", lineno=1,
                    is_source=True, source_cwe="CWE-20",
                ),
                "san": FunctionInfo(
                    module="m", name="san", lineno=2,
                    is_sanitizer=True,
                ),
                "snk": FunctionInfo(
                    module="m", name="snk", lineno=3,
                    is_sink=True, sink_cwe="CWE-94",
                ),
            },
            edges=[
                CallEdge(caller="src", callee="san", lineno=1),
                CallEdge(caller="san", callee="snk", lineno=2),
            ],
        )
        paths = find_taint_paths(g)
        assert len(paths) == 1
        assert paths[0].sanitized is True

    def test_no_paths(self) -> None:
        g = CallGraph(
            functions={
                "a": FunctionInfo(module="m", name="a", lineno=1),
            },
        )
        assert find_taint_paths(g) == []

    def test_no_sources(self) -> None:
        g = CallGraph(
            functions={
                "a": FunctionInfo(
                    module="m", name="a", lineno=1,
                    is_sink=True, sink_cwe="CWE-94",
                ),
            },
        )
        assert find_taint_paths(g) == []


# -----------------------------------------------------------------------
# format_text_report
# -----------------------------------------------------------------------

class TestFormatTextReport:
    def test_with_paths(self) -> None:
        paths = [
            TaintPath(
                source="src", sink="snk",
                path=["src", "mid", "snk"],
                cwe="CWE-94", sanitized=False,
            ),
        ]
        report = format_text_report(paths)
        assert "CWE-94" in report
        assert "src -> snk" in report
        assert "src -> mid -> snk" in report

    def test_no_paths(self) -> None:
        assert format_text_report([]) == "No taint paths found."

    def test_sanitized_excluded_by_default(self) -> None:
        paths = [
            TaintPath(
                source="src", sink="snk",
                path=["src", "snk"],
                cwe="CWE-94", sanitized=True,
            ),
        ]
        report = format_text_report(paths)
        assert report == "No taint paths found."

    def test_sanitized_included_when_requested(self) -> None:
        paths = [
            TaintPath(
                source="src", sink="snk",
                path=["src", "snk"],
                cwe="CWE-94", sanitized=True,
            ),
        ]
        report = format_text_report(paths, include_sanitized=True)
        assert "[SANITIZED]" in report
        assert "CWE-94" in report


# -----------------------------------------------------------------------
# _sarif_result
# -----------------------------------------------------------------------

class TestSarifResult:
    def test_structure(self) -> None:
        tp = TaintPath(
            source="src", sink="snk",
            path=["src", "snk"],
            cwe="CWE-94", sanitized=False,
        )
        result = _sarif_result(tp)
        assert result["ruleId"] == "CWE-94"
        assert "text" in result["message"]  # type: ignore[operator]
        assert result["properties"]["sanitized"] is False  # type: ignore[index]
        assert result["properties"]["path"] == ["src", "snk"]  # type: ignore[index]


# -----------------------------------------------------------------------
# format_sarif
# -----------------------------------------------------------------------

class TestFormatSarif:
    def test_structure(self) -> None:
        paths = [
            TaintPath(
                source="src", sink="snk",
                path=["src", "snk"],
                cwe="CWE-94", sanitized=False,
            ),
        ]
        sarif = format_sarif(paths)
        assert sarif["version"] == "2.1.0"
        assert "$schema" in sarif
        runs = sarif["runs"]
        assert isinstance(runs, list)
        assert len(runs) == 1
        assert len(runs[0]["results"]) == 1  # type: ignore[index]

    def test_sanitized_filtered(self) -> None:
        paths = [
            TaintPath(
                source="src", sink="snk",
                path=["src", "snk"],
                cwe="CWE-94", sanitized=True,
            ),
        ]
        sarif = format_sarif(paths)
        runs = sarif["runs"]
        assert isinstance(runs, list)
        assert len(runs[0]["results"]) == 0  # type: ignore[index]

    def test_sanitized_included(self) -> None:
        paths = [
            TaintPath(
                source="src", sink="snk",
                path=["src", "snk"],
                cwe="CWE-94", sanitized=True,
            ),
        ]
        sarif = format_sarif(paths, include_sanitized=True)
        runs = sarif["runs"]
        assert isinstance(runs, list)
        assert len(runs[0]["results"]) == 1  # type: ignore[index]


# -----------------------------------------------------------------------
# parse_args
# -----------------------------------------------------------------------

class TestParseArgs:
    def test_directory_only(self) -> None:
        ns = parse_args(["src/"])
        assert ns.directory == "src/"
        assert ns.sarif is False
        assert ns.include_sanitized is False

    def test_sarif_flag(self) -> None:
        ns = parse_args(["--sarif", "src/"])
        assert ns.sarif is True

    def test_include_sanitized_flag(self) -> None:
        ns = parse_args(["--include-sanitized", "src/"])
        assert ns.include_sanitized is True

    def test_all_flags(self) -> None:
        ns = parse_args(["--sarif", "--include-sanitized", "src/"])
        assert ns.sarif is True
        assert ns.include_sanitized is True


# -----------------------------------------------------------------------
# main (CLI)
# -----------------------------------------------------------------------

class TestMain:
    def test_text_output(self, tmp_path: Any, capsys: Any) -> None:
        src = textwrap.dedent("""\
            def reader():
                return input()

            def printer():
                print("hi")
        """)
        (tmp_path / "code.py").write_text(src)
        ret = main([str(tmp_path)])
        assert ret == 0
        out = capsys.readouterr().out
        # Should produce some output (text report)
        assert isinstance(out, str)

    def test_sarif_output(self, tmp_path: Any, capsys: Any) -> None:
        src = textwrap.dedent("""\
            def reader():
                return input()
        """)
        (tmp_path / "code.py").write_text(src)
        ret = main(["--sarif", str(tmp_path)])
        assert ret == 0
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["version"] == "2.1.0"

    def test_include_sanitized_text(
        self, tmp_path: Any, capsys: Any,
    ) -> None:
        src = textwrap.dedent("""\
            def reader():
                return input()

            def frame_data(x):
                return x

            def writer(x):
                print(x)
        """)
        (tmp_path / "code.py").write_text(src)
        ret = main(["--include-sanitized", str(tmp_path)])
        assert ret == 0

    def test_no_py_files(self, tmp_path: Any, capsys: Any) -> None:
        ret = main([str(tmp_path)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "No taint paths found" in out


# -----------------------------------------------------------------------
# End-to-end integration with temp files
# -----------------------------------------------------------------------

class TestEndToEnd:
    def test_source_to_sink_via_call(self, tmp_path: Any) -> None:
        src = textwrap.dedent("""\
            def get_input():
                return input()

            def process(data):
                eval(data)

            def main():
                data = get_input()
                process(data)
        """)
        (tmp_path / "app.py").write_text(src)
        graph = build_graph(str(tmp_path))
        sources = [f.name for f in graph.functions.values() if f.is_source]
        sinks = [f.name for f in graph.functions.values() if f.is_sink]
        assert "app.get_input" in sources
        assert "app.process" in sinks

    def test_direct_source_sink_chain(self, tmp_path: Any) -> None:
        """Source function directly calls sink function."""
        src = textwrap.dedent("""\
            def dangerous():
                data = input()
                eval(data)
        """)
        (tmp_path / "d.py").write_text(src)
        graph = build_graph(str(tmp_path))
        # dangerous is both source and sink
        info = graph.functions["d.dangerous"]
        assert info.is_source is True
        assert info.is_sink is True

    def test_sanitizer_on_path(self, tmp_path: Any) -> None:
        src = textwrap.dedent("""\
            def get_data():
                return input()

            def is_safe_bash(cmd):
                return True

            def run_cmd(cmd):
                eval(cmd)
        """)
        (tmp_path / "s.py").write_text(src)
        graph = build_graph(str(tmp_path))
        san = graph.functions["s.is_safe_bash"]
        assert san.is_sanitizer is True

    def test_cross_module_edges(self, tmp_path: Any) -> None:
        (tmp_path / "a.py").write_text(textwrap.dedent("""\
            from b import sink_fn

            def source_fn():
                data = input()
                sink_fn(data)
        """))
        (tmp_path / "b.py").write_text(textwrap.dedent("""\
            def sink_fn(data):
                eval(data)
        """))
        graph = build_graph(str(tmp_path))
        # source_fn calls b.sink_fn via import resolution
        callee_names = [e.callee for e in graph.edges]
        assert "b.sink_fn" in callee_names


# -----------------------------------------------------------------------
# _FunctionVisitor / _CallVisitor indirectly
# -----------------------------------------------------------------------

class TestFunctionVisitor:
    def test_sanitizer_detection(self, tmp_path: Any) -> None:
        src = textwrap.dedent("""\
            def is_safe_path(p):
                return True
        """)
        p = tmp_path / "v.py"
        p.write_text(src)
        funcs, _, _ = parse_file(str(p), "v")
        assert funcs[0].is_sanitizer is True

    def test_non_sanitizer(self, tmp_path: Any) -> None:
        src = textwrap.dedent("""\
            def normal():
                pass
        """)
        p = tmp_path / "v.py"
        p.write_text(src)
        funcs, _, _ = parse_file(str(p), "v")
        assert funcs[0].is_sanitizer is False


# -----------------------------------------------------------------------
# _CallVisitor — calls outside functions
# -----------------------------------------------------------------------

class TestCallVisitorModuleLevel:
    def test_module_level_call_not_captured(self, tmp_path: Any) -> None:
        """Module-level calls (outside any function) don't create edges."""
        src = textwrap.dedent("""\
            print("hello")

            def foo():
                bar()
        """)
        p = tmp_path / "ml.py"
        p.write_text(src)
        _, edges, _ = parse_file(str(p), "ml")
        # Only edge should be foo -> bar
        assert len(edges) == 1
        assert edges[0].caller == "ml.foo"


# -----------------------------------------------------------------------
# Edge cases for _attr_parts with non-resolvable base
# -----------------------------------------------------------------------

class TestAttrPartsEdgeCases:
    def test_attribute_on_call_result(self) -> None:
        """foo().bar — Attribute where value is Call, not Name/Attr."""
        tree = ast.parse("foo().bar")
        attr_node = tree.body[0].value  # type: ignore[attr-defined]
        result = _attr_parts(attr_node)
        assert result == []


# -----------------------------------------------------------------------
# Branch coverage: unresolvable call in function body
# -----------------------------------------------------------------------

class TestUnresolvableCallInFunction:
    def test_subscript_call_in_function(self, tmp_path: Any) -> None:
        """x[0]() inside a function — raw is empty, no edge created."""
        src = textwrap.dedent("""\
            def f():
                handlers[0]()
        """)
        p = tmp_path / "ur.py"
        p.write_text(src)
        _, edges, _ = parse_file(str(p), "ur")
        # handlers[0]() is a Subscript call — can't resolve name
        assert edges == []

    def test_call_on_call_in_body(self) -> None:
        """foo()() in body — resolve returns empty for inner."""
        src = textwrap.dedent("""\
            def g():
                foo()()
        """)
        tree = ast.parse(src)
        func = tree.body[0]
        calls = _collect_calls_in_body(func.body)  # type: ignore[attr-defined]
        # foo() resolves but foo()() does not (outer Call on Call)
        assert "foo" in calls


# -----------------------------------------------------------------------
# Branch coverage: BFS cycle handling
# -----------------------------------------------------------------------

class TestBfsCycleHandling:
    def test_bfs_to_sinks_cycle(self) -> None:
        """Cycle in forward adj: visited check prevents re-visit."""
        fwd: dict[str, list[str]] = {
            "a": ["b"], "b": ["a", "c"],
        }
        sinks = {"c": "CWE-94"}
        g = CallGraph()
        result = _bfs_to_sinks("a", fwd, sinks, g)
        assert len(result) == 1
        assert result[0][0] == "c"

    def test_bfs_to_sinks_revisit_skipped(self) -> None:
        """Multiple paths to same node — second visit is skipped."""
        fwd: dict[str, list[str]] = {
            "a": ["b", "c"], "b": ["d"], "c": ["d"],
        }
        sinks = {"d": "CWE-78"}
        g = CallGraph()
        result = _bfs_to_sinks("a", fwd, sinks, g)
        # d is reached first via a->b->d; second path a->c->d is skipped
        assert len(result) == 1


# -----------------------------------------------------------------------
# Regression tests for issue #2:
# Class-scoped FQNs must distinguish methods of different classes.
# https://github.com/edhodapp/python-agent/issues/2
# -----------------------------------------------------------------------


class TestClassScopedFQN:
    """Regression tests for issue #2."""

    def test_two_classes_same_method_name(
        self, tmp_path: Any,
    ) -> None:
        """Methods in different classes produce distinct FQN keys.

        Before the fix, both Handler.run and Logger.run collapsed to
        `mod.run`, so one overwrote the other in graph.functions.
        """
        src = textwrap.dedent("""\
            class Handler:
                def run(self, payload):
                    eval(payload)

            class Logger:
                def run(self, message):
                    pass
        """)
        p = tmp_path / "mod.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "mod")
        names = {f.name for f in funcs}
        assert "mod.Handler.run" in names
        assert "mod.Logger.run" in names
        handler_run = next(
            f for f in funcs if f.name == "mod.Handler.run"
        )
        assert handler_run.is_sink is True
        assert handler_run.sink_cwe == "CWE-94"
        logger_run = next(
            f for f in funcs if f.name == "mod.Logger.run"
        )
        assert logger_run.is_sink is False

    def test_nested_class_contributes_to_fqn(
        self, tmp_path: Any,
    ) -> None:
        """Nested classes prefix their methods' FQNs."""
        src = textwrap.dedent("""\
            class Outer:
                class Inner:
                    def method(self):
                        pass
        """)
        p = tmp_path / "nested.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "nested")
        names = {f.name for f in funcs}
        assert "nested.Outer.Inner.method" in names

    def test_top_level_function_fqn_unchanged(
        self, tmp_path: Any,
    ) -> None:
        """Top-level functions keep their module-only FQN."""
        src = textwrap.dedent("""\
            def standalone():
                pass
        """)
        p = tmp_path / "top.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "top")
        names = {f.name for f in funcs}
        assert "top.standalone" in names
        assert not any("top.." in n for n in names)

    def test_class_method_edge_caller_is_class_scoped(
        self, tmp_path: Any,
    ) -> None:
        """Edges from class methods use the class-scoped caller FQN.

        Also verifies the class stack is popped correctly: a top-level
        function defined AFTER a class must not inherit the class in
        its FQN. A mutmut mutation that drops the pop would cause
        `helper` to become `e.Handler.helper` here.
        """
        src = textwrap.dedent("""\
            class Handler:
                def run(self, payload):
                    helper()

            def helper():
                pass
        """)
        p = tmp_path / "e.py"
        p.write_text(src)
        funcs, edges, _supps = parse_file(str(p), "e")
        assert any(
            e.caller == "e.Handler.run" and e.callee == "e.helper"
            for e in edges
        )
        assert not any(e.caller == "e.run" for e in edges)
        # Stack-pop correctness: helper() is top-level, not a class member.
        names = {f.name for f in funcs}
        assert "e.helper" in names
        assert "e.Handler.helper" not in names

    def test_async_method_in_class(
        self, tmp_path: Any,
    ) -> None:
        """Async methods inside a class get class-scoped FQN."""
        src = textwrap.dedent("""\
            class Runner:
                async def go(self):
                    pass
        """)
        p = tmp_path / "am.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "am")
        names = {f.name for f in funcs}
        assert "am.Runner.go" in names
        assert "am.go" not in names

    def test_suppression_on_class_method_binds_to_class_fqn(
        self, tmp_path: Any,
    ) -> None:
        """# taint: ignore[...] on a class method binds to the class-scoped FQN.

        Regression for the MEDIUM finding raised during review of the
        #2 fix: before the suppression path was made class-aware, a
        # taint: ignore comment on a class method produced a Suppression
        with function="mod.method_name" which no longer matched the
        class-scoped function FQN "mod.Class.method_name", so
        suppressions on class methods silently stopped working.
        """
        src = textwrap.dedent("""\
            class Api:
                # taint: ignore[CWE-200] -- LLM output is trusted here
                def display(self, text):
                    print(text)
        """)
        p = tmp_path / "api.py"
        p.write_text(src)
        _funcs, _edges, supps = parse_file(str(p), "api")
        assert len(supps) == 1
        assert supps[0].function == "api.Api.display"
        assert supps[0].cwe == "CWE-200"
        assert "LLM output" in supps[0].reason

    def test_suppression_on_free_function_still_works(
        self, tmp_path: Any,
    ) -> None:
        """Regression guard: free-function suppressions still bind to `mod.func`."""
        src = textwrap.dedent("""\
            # taint: ignore[CWE-94] -- developer-only tool
            def eval_helper(x):
                eval(x)
        """)
        p = tmp_path / "ff.py"
        p.write_text(src)
        _funcs, _edges, supps = parse_file(str(p), "ff")
        assert len(supps) == 1
        assert supps[0].function == "ff.eval_helper"
        assert supps[0].cwe == "CWE-94"

    def test_suppression_inline_on_def_line(
        self, tmp_path: Any,
    ) -> None:
        """Suppression comment inline on the def line binds correctly.

        Exercises the `i + 1` branch of
        `line_to_fqn.get(i + 1) or line_to_fqn.get(i + 2)` — without
        this, a mutmut mutation swapping `or` for `and` (or dropping
        the first lookup) survives because the other suppression
        tests use only the `i + 2` (comment-one-line-before-def)
        branch.
        """
        src = textwrap.dedent("""\
            def run_eval():  # taint: ignore[CWE-94] -- inline
                eval("")
        """)
        p = tmp_path / "inline.py"
        p.write_text(src)
        _funcs, _edges, supps = parse_file(str(p), "inline")
        assert len(supps) == 1
        assert supps[0].function == "inline.run_eval"
        assert supps[0].cwe == "CWE-94"
        assert "inline" in supps[0].reason

    def test_orphan_suppression_comment_is_dropped(
        self, tmp_path: Any,
    ) -> None:
        """A `# taint: ignore[...]` with no following def contributes no Suppression.

        Covers the branch in _collect_suppressions where the line_to_fqn
        lookup returns no FQN for the comment's line (or line + 1).
        """
        src = textwrap.dedent("""\
            # taint: ignore[CWE-94] -- attached to nothing
            x = 42
        """)
        p = tmp_path / "orphan.py"
        p.write_text(src)
        _funcs, _edges, supps = parse_file(str(p), "orphan")
        assert supps == []

    def test_suppressed_taint_paths_skipped_in_default_report(
        self, tmp_path: Any,
    ) -> None:
        """End-to-end: two suppressed source-to-sink paths, different CWEs.

        Two distinct paths with distinct CWE suppressions — when
        checking the second path's suppression list, the first
        suppression in the list does NOT match (different CWE),
        forcing the loop to `continue` past it. That exercises the
        `_check_suppressed` continue-branch (547->546) in addition
        to the positive-match branch and the `_should_skip`
        `tp.suppressed=True` branch.
        """
        src = textwrap.dedent("""\
            # taint: ignore[CWE-78] -- shell access is intentional here
            def run_shell():
                cmd = input()
                shell_runner(cmd)

            def shell_runner(cmd):
                os.system(cmd)

            # taint: ignore[CWE-94] -- developer-only eval tool
            def read_user_code():
                code = input()
                evaluator(code)

            def evaluator(code):
                eval(code)
        """)
        (tmp_path / "ev.py").write_text(src)
        graph = build_graph(str(tmp_path))
        paths = find_taint_paths(graph)
        assert len(paths) == 2
        for p in paths:
            assert p.suppressed is True
        cwes = {p.cwe for p in paths}
        assert cwes == {"CWE-78", "CWE-94"}
        # Default report hides suppressed paths entirely.
        report = format_text_report(paths)
        assert report == "No taint paths found."


# -----------------------------------------------------------------------
# Regression tests for issue #3:
# Nested function calls must NOT be attributed to the outer function.
# Nested defs appear as their own nodes in graph.functions with
# scope-qualified FQNs like `mod.outer.inner`.
# https://github.com/edhodapp/python-agent/issues/3
# -----------------------------------------------------------------------


class TestNestedFunctionScope:
    """Regression tests for issue #3."""

    def test_nested_sink_call_not_attributed_to_outer(
        self, tmp_path: Any,
    ) -> None:
        """eval() inside an inner function does NOT make the outer a sink.

        Before the fix, `_collect_calls_in_body` used `ast.walk` on
        the outer function's body, which descends recursively into
        nested FunctionDef bodies and picks up their Call nodes. The
        outer was therefore incorrectly classified as a sink (CWE-94)
        because its "calls" included the inner's eval().
        """
        src = textwrap.dedent("""\
            def outer(x):
                def inner():
                    eval(x)
                return 42
        """)
        p = tmp_path / "nest.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "nest")
        names = {f.name for f in funcs}
        assert "nest.outer" in names
        outer = next(f for f in funcs if f.name == "nest.outer")
        assert outer.is_sink is False
        # Inner appears as its own node with the sink tag.
        assert "nest.outer.inner" in names
        inner = next(f for f in funcs if f.name == "nest.outer.inner")
        assert inner.is_sink is True
        assert inner.sink_cwe == "CWE-94"

    def test_nested_source_call_not_attributed_to_outer(
        self, tmp_path: Any,
    ) -> None:
        """input() inside an inner function does NOT make the outer a source."""
        src = textwrap.dedent("""\
            def outer():
                def inner():
                    return input()
                return None
        """)
        p = tmp_path / "nsrc.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "nsrc")
        outer = next(f for f in funcs if f.name == "nsrc.outer")
        assert outer.is_source is False
        inner = next(
            f for f in funcs if f.name == "nsrc.outer.inner"
        )
        assert inner.is_source is True
        assert inner.source_cwe == "CWE-20"

    def test_deeply_nested_functions_get_full_scope_fqn(
        self, tmp_path: Any,
    ) -> None:
        """Three levels of nested functions each carry full scope in FQN."""
        src = textwrap.dedent("""\
            def a():
                def b():
                    def c():
                        pass
                    return c
                return b
        """)
        p = tmp_path / "deep.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "deep")
        names = {f.name for f in funcs}
        assert "deep.a" in names
        assert "deep.a.b" in names
        assert "deep.a.b.c" in names

    def test_async_nested_def_is_fenced(
        self, tmp_path: Any,
    ) -> None:
        """`async def` inside a sync outer is fenced.

        Covers the `_FencedCallCollector.visit_AsyncFunctionDef`
        branch. Without this test, a mutmut mutation that removes
        the `visit_AsyncFunctionDef` override survives because
        nothing else exercises async nested defs.
        """
        src = textwrap.dedent("""\
            def outer():
                async def inner():
                    eval("")
        """)
        p = tmp_path / "asn.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "asn")
        outer = next(f for f in funcs if f.name == "asn.outer")
        assert outer.is_sink is False
        inner = next(
            f for f in funcs if f.name == "asn.outer.inner"
        )
        assert inner.is_sink is True

    def test_nested_lambda_sink_not_attributed_to_outer(
        self, tmp_path: Any,
    ) -> None:
        """Lambdas are fenced too — eval in a lambda doesn't flag outer."""
        src = textwrap.dedent("""\
            def outer():
                handlers = [lambda x: eval(x)]
                return handlers
        """)
        p = tmp_path / "lam.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "lam")
        outer = next(f for f in funcs if f.name == "lam.outer")
        assert outer.is_sink is False

    def test_nested_function_inside_class_method(
        self, tmp_path: Any,
    ) -> None:
        """Class method with a nested function: scope stack combines class + function."""
        src = textwrap.dedent("""\
            class Api:
                def handle(self, x):
                    def validator(v):
                        eval(v)
                    validator(x)
        """)
        p = tmp_path / "cn.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "cn")
        names = {f.name for f in funcs}
        assert "cn.Api.handle" in names
        assert "cn.Api.handle.validator" in names
        handle = next(
            f for f in funcs if f.name == "cn.Api.handle"
        )
        # eval lives in the nested validator; handle must not inherit
        assert handle.is_sink is False
        validator = next(
            f for f in funcs if f.name == "cn.Api.handle.validator"
        )
        assert validator.is_sink is True

    def test_top_level_function_fqn_regression_guard(
        self, tmp_path: Any,
    ) -> None:
        """Simple top-level function — no nesting, no scope prefix."""
        src = textwrap.dedent("""\
            def standalone():
                eval("")
        """)
        p = tmp_path / "s.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "s")
        names = {f.name for f in funcs}
        assert "s.standalone" in names
        # And specifically nothing like s.standalone.standalone.
        matching = [n for n in names if n.startswith("s.standalone")]
        assert matching == ["s.standalone"]
        standalone = next(
            f for f in funcs if f.name == "s.standalone"
        )
        assert standalone.is_sink is True

    def test_nested_function_edge_caller_is_scope_qualified(
        self, tmp_path: Any,
    ) -> None:
        """Edges from a nested function use the full scope-qualified caller FQN.

        Explicit coverage for `_CallVisitor._scope_stack` push/pop.
        A mutmut mutation removing the push in `_CallVisitor._visit_func`
        would cause `inner`'s edges to be attributed to `ne.inner`
        instead of `ne.outer.inner`, which this test catches.
        """
        src = textwrap.dedent("""\
            def outer():
                def inner():
                    helper()

            def helper():
                pass
        """)
        p = tmp_path / "ne.py"
        p.write_text(src)
        _funcs, edges, _supps = parse_file(str(p), "ne")
        assert any(
            e.caller == "ne.outer.inner" and e.callee == "ne.helper"
            for e in edges
        )
        # Specifically NOT the pre-fix broken caller `ne.inner`.
        assert not any(e.caller == "ne.inner" for e in edges)


# -----------------------------------------------------------------------
# Regression tests for issue #4:
# _resolve_callee must resolve `self.method()` / `cls.method()` to the
# enclosing class's method FQN, or taint flows through method calls
# are invisible to the analyzer.
# https://github.com/edhodapp/python-agent/issues/4
# -----------------------------------------------------------------------


class TestSelfMethodResolution:
    """Regression tests for issue #4."""

    def test_self_method_edge_resolves_to_class_fqn(
        self, tmp_path: Any,
    ) -> None:
        """`self.helper()` inside a class method becomes `mod.Class.helper`.

        Before the fix, _resolve_callee returned the raw `self.helper`
        unchanged. The edge was a ghost — the callee never matched
        any node in graph.functions, so BFS couldn't follow the
        taint through it.
        """
        src = textwrap.dedent("""\
            class Api:
                def handle(self):
                    self.helper()

                def helper(self):
                    pass
        """)
        p = tmp_path / "sm.py"
        p.write_text(src)
        _funcs, edges, _supps = parse_file(str(p), "sm")
        assert any(
            e.caller == "sm.Api.handle" and e.callee == "sm.Api.helper"
            for e in edges
        )
        # Specifically NOT the raw unresolved "self.helper".
        assert not any(e.callee == "self.helper" for e in edges)

    def test_self_method_call_enables_taint_path(
        self, tmp_path: Any,
    ) -> None:
        """End-to-end: taint flows through a `self.method()` edge.

        The real user-visible bug: a call from a source method to a
        sink method via `self.X()` was invisible to BFS because the
        edge went to an unresolved raw name. Now the edge resolves
        to the class-scoped method FQN, the BFS reaches the sink,
        and the taint path is reported.
        """
        src = textwrap.dedent("""\
            class Api:
                def handle(self):
                    code = input()
                    self.run_code(code)

                def run_code(self, code):
                    eval(code)
        """)
        (tmp_path / "tp.py").write_text(src)
        graph = build_graph(str(tmp_path))
        paths = find_taint_paths(graph)
        assert len(paths) == 1
        assert paths[0].source == "tp.Api.handle"
        assert paths[0].sink == "tp.Api.run_code"
        assert paths[0].cwe == "CWE-94"

    def test_cls_method_call_resolves_to_class_fqn(
        self, tmp_path: Any,
    ) -> None:
        """`cls.method()` in a classmethod resolves like `self.method()`."""
        src = textwrap.dedent("""\
            class Api:
                @classmethod
                def factory(cls):
                    cls.make()

                @classmethod
                def make(cls):
                    pass
        """)
        p = tmp_path / "cm.py"
        p.write_text(src)
        _funcs, edges, _supps = parse_file(str(p), "cm")
        assert any(
            e.caller == "cm.Api.factory" and e.callee == "cm.Api.make"
            for e in edges
        )

    def test_self_in_nested_class_uses_full_class_path(
        self, tmp_path: Any,
    ) -> None:
        """`self.X` in a method of a nested class uses the full class path."""
        src = textwrap.dedent("""\
            class Outer:
                class Inner:
                    def method(self):
                        self.helper()

                    def helper(self):
                        pass
        """)
        p = tmp_path / "nc.py"
        p.write_text(src)
        _funcs, edges, _supps = parse_file(str(p), "nc")
        assert any(
            e.caller == "nc.Outer.Inner.method"
            and e.callee == "nc.Outer.Inner.helper"
            for e in edges
        )

    def test_unrelated_obj_method_call_stays_unresolved(
        self, tmp_path: Any,
    ) -> None:
        """`obj.method()` where `obj` is not self/cls is NOT over-resolved.

        Type inference is out of scope. The tool must not invent a
        wrong resolution (e.g., pretending `other.method()` means
        the enclosing class's `method`). Raw names stay as raw names
        so BFS misses are visible as missing edges, not wrong edges.
        """
        src = textwrap.dedent("""\
            class Api:
                def handle(self, other):
                    other.method()
        """)
        p = tmp_path / "oo.py"
        p.write_text(src)
        _funcs, edges, _supps = parse_file(str(p), "oo")
        # The edge exists with the raw "other.method" callee.
        assert any(
            e.caller == "oo.Api.handle" and e.callee == "other.method"
            for e in edges
        )
        # Specifically NOT falsely resolved to oo.Api.method.
        assert not any(e.callee == "oo.Api.method" for e in edges)

    def test_self_attribute_chain_falls_through(
        self, tmp_path: Any,
    ) -> None:
        """`self.foo.bar()` is too deep — not resolved by _resolve_with_self.

        The simple `self.X` rule cannot handle attribute chains —
        what `self.foo` is bound to requires type inference we do
        not attempt. The resolver must leave such raw names alone
        rather than inventing a wrong resolution like
        `mod.Class.bar` or `mod.Class.foo.bar`. This locks in the
        `len == 2` guard, killing `>= 2` / `>= 1` mutations.
        """
        src = textwrap.dedent("""\
            class Api:
                def handle(self):
                    self.foo.bar()
        """)
        p = tmp_path / "ac.py"
        p.write_text(src)
        _funcs, edges, _supps = parse_file(str(p), "ac")
        assert any(
            e.caller == "ac.Api.handle" and e.callee == "self.foo.bar"
            for e in edges
        )
        assert not any(e.callee == "ac.Api.bar" for e in edges)
        assert not any(e.callee == "ac.Api.foo.bar" for e in edges)

    def test_self_method_from_nested_function_uses_enclosing_class(
        self, tmp_path: Any,
    ) -> None:
        """`self.X()` inside a nested def inside a method resolves to the class.

        A closure-captured `self` in a nested `def inner()` inside
        `Api.handle` still refers to the `Api` instance, so
        `self.helper()` inside `inner` must resolve to `mod.Api.helper`,
        NOT `mod.Api.handle.helper`. Locks in the invariant that
        `_class_stack` is unaffected by entering a function scope
        — only `_scope_stack` changes in `_visit_func`.
        """
        src = textwrap.dedent("""\
            class Api:
                def handle(self):
                    def inner():
                        self.helper()
                    inner()

                def helper(self):
                    pass
        """)
        p = tmp_path / "cf.py"
        p.write_text(src)
        _funcs, edges, _supps = parse_file(str(p), "cf")
        assert any(
            e.caller == "cf.Api.handle.inner"
            and e.callee == "cf.Api.helper"
            for e in edges
        )
        # Specifically NOT cf.Api.handle.helper (which would happen
        # if _class_stack incorrectly included the function name).
        assert not any(
            e.callee == "cf.Api.handle.helper" for e in edges
        )


# -----------------------------------------------------------------------
# Regression tests for issue #5:
# Source detection is suffix-free — only exact matches in SOURCE_EXACT
# are flagged. Removes the false positives from matching any `.read`,
# `.parse_args`, `.model_validate`, `.query` method call regardless of
# receiver type. https://github.com/edhodapp/python-agent/issues/5
# -----------------------------------------------------------------------


class TestSourceDetectionSpecificity:
    """Regression tests for issue #5."""

    def test_bytesio_read_not_flagged_as_source(
        self, tmp_path: Any,
    ) -> None:
        """Internal buffer `.read()` is NOT a taint source.

        Before the fix, any call ending in `.read` matched the
        SOURCE_SUFFIX table and flagged the enclosing function
        as a source (CWE-73). That's a massive false positive:
        internal buffers like io.BytesIO() are trusted data.
        """
        src = textwrap.dedent("""\
            import io

            def load_default():
                buf = io.BytesIO(b"{}")
                return buf.read()
        """)
        p = tmp_path / "bufsrc.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "bufsrc")
        load_default = next(
            f for f in funcs if f.name == "bufsrc.load_default"
        )
        assert load_default.is_source is False

    def test_model_validate_still_flagged(
        self, tmp_path: Any,
    ) -> None:
        """`.model_validate()` remains flagged — distinctive Pydantic name.

        The #5 fix keeps `.model_validate` in SOURCE_SUFFIX because
        the name is distinctive enough to argparse/Pydantic that the
        false-positive rate is acceptable. Without this, python_agent's
        own ontology deserialization (Entity.model_validate,
        OntologyDAG.model_validate_json) would stop being detected.
        """
        src = textwrap.dedent("""\
            def parse_user_input(data):
                return MyModel.model_validate(data)
        """)
        p = tmp_path / "mv.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "mv")
        parse_user_input = next(
            f for f in funcs if f.name == "mv.parse_user_input"
        )
        assert parse_user_input.is_source is True
        assert parse_user_input.source_cwe == "CWE-502"

    def test_parse_args_still_flagged(
        self, tmp_path: Any,
    ) -> None:
        """`parser.parse_args()` remains flagged — distinctive argparse name.

        Without this, ALL of python_agent's CLI entry-point `main()`
        functions would silently stop being classified as CWE-20
        sources. The suffix rule stays because `.parse_args` is
        distinctive enough to rarely collide outside argparse/click.
        """
        src = textwrap.dedent("""\
            import argparse

            def build_and_parse():
                parser = argparse.ArgumentParser()
                return parser.parse_args([])
        """)
        p = tmp_path / "pa.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "pa")
        build_and_parse = next(
            f for f in funcs if f.name == "pa.build_and_parse"
        )
        assert build_and_parse.is_source is True
        assert build_and_parse.source_cwe == "CWE-20"

    def test_query_suffix_no_longer_flagged(
        self, tmp_path: Any,
    ) -> None:
        """`.query()` suffix no longer flagged.

        Same trade-off. An arbitrary object's `.query(...)` method
        is not a guaranteed source; the prior suffix rule flagged
        Django ORM, SDK, SQL, and any other unrelated method with
        the same name.
        """
        src = textwrap.dedent("""\
            def run_orm():
                return db.query("SELECT 1")
        """)
        p = tmp_path / "q.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "q")
        run_orm = next(f for f in funcs if f.name == "q.run_orm")
        assert run_orm.is_source is False

    def test_input_builtin_still_flagged(
        self, tmp_path: Any,
    ) -> None:
        """`input()` exact match remains flagged (regression guard)."""
        src = textwrap.dedent("""\
            def read_name():
                return input()
        """)
        p = tmp_path / "inp.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "inp")
        read_name = next(
            f for f in funcs if f.name == "inp.read_name"
        )
        assert read_name.is_source is True
        assert read_name.source_cwe == "CWE-20"

    def test_json_loads_still_flagged(
        self, tmp_path: Any,
    ) -> None:
        """`json.loads()` exact match remains flagged (regression guard)."""
        src = textwrap.dedent("""\
            import json

            def parse_body():
                return json.loads("{}")
        """)
        p = tmp_path / "jl.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "jl")
        parse_body = next(
            f for f in funcs if f.name == "jl.parse_body"
        )
        assert parse_body.is_source is True
        assert parse_body.source_cwe == "CWE-502"

    def test_open_still_flagged(
        self, tmp_path: Any,
    ) -> None:
        """`open()` exact match remains flagged (regression guard)."""
        src = textwrap.dedent("""\
            def load_file():
                f = open("/tmp/x")
                return f
        """)
        p = tmp_path / "op.py"
        p.write_text(src)
        funcs, _edges, _supps = parse_file(str(p), "op")
        load_file = next(
            f for f in funcs if f.name == "op.load_file"
        )
        assert load_file.is_source is True
        assert load_file.source_cwe == "CWE-73"
