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
    _bfs_tainted,
    _bfs_to_sinks,
    _build_forward_adj,
    _build_reverse_adj,
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

    def test_suffix_read(self) -> None:
        assert _is_source_call("file.read") == (True, "CWE-73")

    def test_suffix_model_validate(self) -> None:
        assert _is_source_call("Foo.model_validate") == (True, "CWE-502")

    def test_suffix_model_validate_json(self) -> None:
        result = _is_source_call("Foo.model_validate_json")
        assert result == (True, "CWE-502")

    def test_suffix_parse_args(self) -> None:
        assert _is_source_call("parser.parse_args") == (True, "CWE-20")

    def test_suffix_query_source(self) -> None:
        assert _is_source_call("db.query") == (True, "CWE-74")

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
        funcs, edges = parse_file(str(p), "mod")
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
        funcs, edges = parse_file(str(p), "amod")
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
        funcs, edges = parse_file(str(p), "e")
        assert len(edges) == 1
        assert edges[0].caller == "e.caller"
        assert edges[0].callee == "e.callee"


# -----------------------------------------------------------------------
# build_graph
# -----------------------------------------------------------------------

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
# _build_forward_adj / _build_reverse_adj
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

    def test_reverse(self) -> None:
        g = self._make_graph()
        rev = _build_reverse_adj(g)
        assert "a" in rev["b"]
        assert set(rev["c"]) == {"a", "b"}


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
# _bfs_tainted
# -----------------------------------------------------------------------

class TestBfsTainted:
    def test_basic(self) -> None:
        rev = {"a": ["b", "c"], "b": ["d"]}
        result = _bfs_tainted("a", rev)
        assert result == {"a", "b", "c", "d"}

    def test_isolated(self) -> None:
        result = _bfs_tainted("x", {})
        assert result == {"x"}


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
        funcs, _ = parse_file(str(p), "v")
        assert funcs[0].is_sanitizer is True

    def test_non_sanitizer(self, tmp_path: Any) -> None:
        src = textwrap.dedent("""\
            def normal():
                pass
        """)
        p = tmp_path / "v.py"
        p.write_text(src)
        funcs, _ = parse_file(str(p), "v")
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
        _, edges = parse_file(str(p), "ml")
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
        _, edges = parse_file(str(p), "ur")
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
    def test_bfs_tainted_cycle(self) -> None:
        """Cycle in reverse adj: a->b->a should not loop forever."""
        rev: dict[str, list[str]] = {"a": ["b"], "b": ["a"]}
        result = _bfs_tainted("a", rev)
        assert result == {"a", "b"}

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
