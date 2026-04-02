"""Source-to-sink taint analysis via call graph construction."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import defaultdict

from pydantic import BaseModel

_SUPPRESS_RE = re.compile(
    r"#\s*taint:\s*ignore\[([A-Z]+-\d+)\]\s*--\s*(.+)",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class FunctionInfo(BaseModel):
    """Metadata about a single function definition."""

    module: str
    name: str
    lineno: int
    is_source: bool = False
    is_sink: bool = False
    is_sanitizer: bool = False
    source_cwe: str = ""
    sink_cwe: str = ""


class CallEdge(BaseModel):
    """A directed edge from caller to callee."""

    caller: str
    callee: str
    lineno: int


class Suppression(BaseModel):
    """A user-acknowledged taint suppression."""

    function: str
    cwe: str
    reason: str
    lineno: int


class TaintPath(BaseModel):
    """A path from a taint source to a taint sink."""

    source: str
    sink: str
    path: list[str]
    cwe: str
    sanitized: bool
    suppressed: bool = False
    suppression_reason: str = ""


class CallGraph(BaseModel):
    """Aggregated call graph with function metadata."""

    functions: dict[str, FunctionInfo] = {}
    edges: list[CallEdge] = []
    suppressions: list[Suppression] = []


# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------

SOURCE_EXACT: dict[str, str] = {
    "input": "CWE-20",
    "json.loads": "CWE-502",
    "json.load": "CWE-502",
    "open": "CWE-73",
}

SOURCE_SUFFIX: dict[str, str] = {
    ".read": "CWE-73",
    ".model_validate": "CWE-502",
    ".model_validate_json": "CWE-502",
    ".parse_args": "CWE-20",
    ".query": "CWE-74",
}

SINK_EXACT: dict[str, str] = {
    "eval": "CWE-94",
    "exec": "CWE-94",
    "os.system": "CWE-78",
    "os.popen": "CWE-78",
    "subprocess.run": "CWE-78",
    "subprocess.call": "CWE-78",
    "subprocess.Popen": "CWE-78",
    "subprocess.check_output": "CWE-78",
    "subprocess.check_call": "CWE-78",
    "print": "CWE-200",
}

SINK_SUFFIX: dict[str, str] = {
    ".write": "CWE-73",
    ".query": "CWE-74",
}

SANITIZER_NAMES: set[str] = {
    "frame_data",
    "validate_ontology_strict",
    "is_safe_bash",
    "is_safe_path",
    "model_validate",
}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _resolve_call_name(node: ast.Call) -> str:
    """Extract dotted name from a Call AST node."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = _attr_parts(func)
        return ".".join(parts) if parts else ""
    return ""


def _attr_parts(node: ast.Attribute) -> list[str]:
    """Recursively collect dotted attribute parts."""
    if isinstance(node.value, ast.Name):
        return [node.value.id, node.attr]
    if isinstance(node.value, ast.Attribute):
        inner = _attr_parts(node.value)
        return inner + [node.attr] if inner else []
    return []


def _is_source_call(name: str) -> tuple[bool, str]:
    """Check whether *name* matches a source pattern."""
    if name in SOURCE_EXACT:
        return True, SOURCE_EXACT[name]
    for suffix, cwe in SOURCE_SUFFIX.items():
        if name.endswith(suffix):
            return True, cwe
    return False, ""


def _is_sink_call(name: str) -> tuple[bool, str]:
    """Check whether *name* matches a sink pattern."""
    if name in SINK_EXACT:
        return True, SINK_EXACT[name]
    for suffix, cwe in SINK_SUFFIX.items():
        if name.endswith(suffix):
            return True, cwe
    return False, ""


def _is_sanitizer_name(name: str) -> bool:
    """Check whether the bare tail of *name* is a sanitizer."""
    bare = name.rsplit(".", 1)[-1]
    return bare in SANITIZER_NAMES


def _collect_calls_in_body(
    body: list[ast.stmt],
) -> list[str]:
    """Walk a function body and return all call names."""
    names: list[str] = []
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.Call):
            n = _resolve_call_name(node)
            if n:
                names.append(n)
    return names


def _classify_as_source(
    calls: list[str],
) -> tuple[bool, str]:
    """Return (True, cwe) if any call in *calls* is a source."""
    for c in calls:
        hit, cwe = _is_source_call(c)
        if hit:
            return True, cwe
    return False, ""


def _classify_as_sink(
    calls: list[str],
) -> tuple[bool, str]:
    """Return (True, cwe) if any call in *calls* is a sink."""
    for c in calls:
        hit, cwe = _is_sink_call(c)
        if hit:
            return True, cwe
    return False, ""


# ---------------------------------------------------------------------------
# AST visitors
# ---------------------------------------------------------------------------

class _ImportCollector(ast.NodeVisitor):
    """Collect import aliases from a module AST."""

    def __init__(self) -> None:
        self.aliases: dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local = alias.asname if alias.asname else alias.name
            self.aliases[local] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            local = alias.asname if alias.asname else alias.name
            self.aliases[local] = f"{mod}.{alias.name}" if mod else alias.name


class _FunctionVisitor(ast.NodeVisitor):
    """Collect FunctionInfo entries from a module AST."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.functions: list[FunctionInfo] = []

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        fqn = f"{self.module}.{node.name}"
        calls = _collect_calls_in_body(node.body)
        is_src, src_cwe = _classify_as_source(calls)
        is_snk, snk_cwe = _classify_as_sink(calls)
        self.functions.append(FunctionInfo(
            module=self.module,
            name=fqn,
            lineno=node.lineno,
            is_source=is_src,
            is_sink=is_snk,
            is_sanitizer=_is_sanitizer_name(node.name),
            source_cwe=src_cwe,
            sink_cwe=snk_cwe,
        ))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node)


class _CallVisitor(ast.NodeVisitor):
    """Collect CallEdge entries from a module AST."""

    def __init__(
        self,
        module: str,
        imports: dict[str, str],
    ) -> None:
        self.module = module
        self.imports = imports
        self.edges: list[CallEdge] = []
        self._current_func: str | None = None

    def _visit_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        old = self._current_func
        self._current_func = f"{self.module}.{node.name}"
        self.generic_visit(node)
        self._current_func = old

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._current_func is not None:
            raw = _resolve_call_name(node)
            if raw:
                callee = _resolve_callee(
                    raw, self.imports, self.module,
                )
                self.edges.append(CallEdge(
                    caller=self._current_func,
                    callee=callee,
                    lineno=node.lineno,
                ))
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------

def _collect_imports(tree: ast.Module) -> dict[str, str]:
    """Return {local_alias: fully_qualified_name} for a module."""
    collector = _ImportCollector()
    collector.visit(tree)
    return collector.aliases


def _resolve_callee(
    raw: str, imports: dict[str, str], module: str,
) -> str:
    """Resolve a raw call name to a fully qualified name."""
    parts = raw.split(".", 1)
    head = parts[0]
    if head in imports:
        base = imports[head]
        if len(parts) > 1:
            return f"{base}.{parts[1]}"
        return base
    if "." not in raw:
        return f"{module}.{raw}"
    return raw


def _module_name_from_path(path: str, root: str) -> str:
    """Derive a dotted module name from a file path."""
    rel = os.path.relpath(path, root)
    no_ext = rel.removesuffix(".py")
    return no_ext.replace(os.sep, ".")


def _collect_python_files(root: str) -> list[str]:
    """Return sorted list of .py files under *root*."""
    result: list[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".py"):
                result.append(os.path.join(dirpath, f))
    result.sort()
    return result


def _collect_suppressions(
    source: str, module_name: str,
) -> list[Suppression]:
    """Scan source for taint suppression comments.

    Format: # taint: ignore[CWE-200] -- reason text
    Can appear on a def line or the line immediately before.
    """
    lines = source.splitlines()
    results: list[Suppression] = []
    for i, line in enumerate(lines):
        match = _SUPPRESS_RE.search(line)
        if match is None:
            continue
        cwe = match.group(1)
        reason = match.group(2).strip()
        func_name = _find_func_for_suppress(lines, i)
        if func_name:
            fqn = f"{module_name}.{func_name}"
            results.append(Suppression(
                function=fqn, cwe=cwe,
                reason=reason, lineno=i + 1,
            ))
    return results


def _find_func_for_suppress(
    lines: list[str], comment_idx: int,
) -> str:
    """Find the function name for a suppression comment.

    Checks the comment line itself and the next line.
    """
    name = _extract_func_name(lines[comment_idx])
    if name:
        return name
    if comment_idx + 1 < len(lines):
        return _extract_func_name(lines[comment_idx + 1])
    return ""


def _extract_func_name(line: str) -> str:
    """Extract function name from a def line."""
    stripped = line.strip()
    for prefix in ("async def ", "def "):
        if stripped.startswith(prefix):
            rest = stripped[len(prefix):]
            paren = rest.find("(")
            if paren > 0:
                return rest[:paren]
    return ""


def parse_file(
    path: str, module_name: str,
) -> tuple[
    list[FunctionInfo], list[CallEdge],
    list[Suppression],
]:
    """Parse a .py file. Return functions, edges, suppressions.

    Returns empty results with a warning on SyntaxError.
    """
    with open(path) as fh:
        source = fh.read()
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        import warnings
        warnings.warn(
            f"SyntaxError in {path}, skipping",
            stacklevel=2,
        )
        return [], [], []
    imports = _collect_imports(tree)
    fv = _FunctionVisitor(module_name)
    fv.visit(tree)
    cv = _CallVisitor(module_name, imports)
    cv.visit(tree)
    supps = _collect_suppressions(source, module_name)
    return fv.functions, cv.edges, supps


def build_graph(root_dir: str) -> CallGraph:
    """Build a CallGraph by scanning all .py files."""
    graph = CallGraph()
    for path in _collect_python_files(root_dir):
        mod = _module_name_from_path(path, root_dir)
        funcs, edges, supps = parse_file(path, mod)
        for f in funcs:
            graph.functions[f.name] = f
        graph.edges.extend(edges)
        graph.suppressions.extend(supps)
    return graph


# ---------------------------------------------------------------------------
# Taint tracing
# ---------------------------------------------------------------------------

def _build_forward_adj(graph: CallGraph) -> dict[str, list[str]]:
    """Build caller -> [callee] adjacency list."""
    adj: dict[str, list[str]] = defaultdict(list)
    for e in graph.edges:
        adj[e.caller].append(e.callee)
    return dict(adj)


def _find_sources(graph: CallGraph) -> list[str]:
    """Return names of all source functions in the graph."""
    return [
        name for name, info in graph.functions.items()
        if info.is_source
    ]


def _find_sinks(graph: CallGraph) -> dict[str, str]:
    """Return {name: cwe} for all sink functions in the graph."""
    return {
        name: info.sink_cwe
        for name, info in graph.functions.items()
        if info.is_sink
    }


def _check_sink_hit(
    node: str,
    start: str,
    path: list[str],
    sinks: dict[str, str],
) -> tuple[str, list[str], str] | None:
    """Return a sink hit tuple if *node* is a sink (and not start)."""
    if node in sinks and node != start:
        return (node, path, sinks[node])
    return None


def _enqueue_neighbors(
    node: str,
    path: list[str],
    forward_adj: dict[str, list[str]],
    visited: set[str],
    queue: list[list[str]],
) -> None:
    """Append unvisited neighbor paths to *queue*."""
    for nb in forward_adj.get(node, []):
        if nb not in visited:
            queue.append(path + [nb])


def _bfs_to_sinks(
    start: str,
    forward_adj: dict[str, list[str]],
    sinks: dict[str, str],
    graph: CallGraph,
) -> list[tuple[str, list[str], str]]:
    """Forward BFS from *start*, return paths reaching sinks."""
    results: list[tuple[str, list[str], str]] = []
    queue: list[list[str]] = [[start]]
    visited: set[str] = set()
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node in visited:
            continue
        visited.add(node)
        hit = _check_sink_hit(node, start, path, sinks)
        if hit:
            results.append(hit)
        _enqueue_neighbors(node, path, forward_adj, visited, queue)
    return results


def _path_has_sanitizer(
    path: list[str], graph: CallGraph,
) -> bool:
    """Return True if any node on *path* is a sanitizer."""
    for node in path:
        info = graph.functions.get(node)
        if info and info.is_sanitizer:
            return True
    return False


def _check_suppressed(
    path: list[str], cwe: str,
    suppressions: list[Suppression],
) -> tuple[bool, str]:
    """Check if any function on path has a suppression.

    Returns (suppressed, reason).
    """
    for s in suppressions:
        if s.cwe == cwe and s.function in path:
            return (True, s.reason)
    return (False, "")


def find_taint_paths(graph: CallGraph) -> list[TaintPath]:
    """Find all source-to-sink taint paths in the graph."""
    forward = _build_forward_adj(graph)
    sinks = _find_sinks(graph)
    sources = _find_sources(graph)
    results: list[TaintPath] = []
    for src in sources:
        hits = _bfs_to_sinks(src, forward, sinks, graph)
        for sink_name, path, cwe in hits:
            sanitized = _path_has_sanitizer(path, graph)
            suppressed, reason = _check_suppressed(
                path, cwe, graph.suppressions,
            )
            results.append(TaintPath(
                source=src,
                sink=sink_name,
                path=path,
                cwe=cwe,
                sanitized=sanitized,
                suppressed=suppressed,
                suppression_reason=reason,
            ))
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _should_skip(
    tp: TaintPath, include_sanitized: bool,
) -> bool:
    """Check if a path should be skipped in output."""
    if tp.suppressed:
        return True
    if tp.sanitized and not include_sanitized:
        return True
    return False


def format_text_report(
    paths: list[TaintPath],
    include_sanitized: bool = False,
) -> str:
    """Format taint paths as a human-readable text report."""
    lines: list[str] = []
    for tp in paths:
        if _should_skip(tp, include_sanitized):
            continue
        status = " [SANITIZED]" if tp.sanitized else ""
        chain = " -> ".join(tp.path)
        lines.append(
            f"{tp.cwe}: {tp.source} -> {tp.sink}{status}"
        )
        lines.append(f"  path: {chain}")
    if not lines:
        return "No taint paths found."
    return "\n".join(lines)


def _sarif_result(tp: TaintPath) -> dict[str, object]:
    """Build a single SARIF result dict for a TaintPath."""
    return {
        "ruleId": tp.cwe,
        "message": {
            "text": f"Taint flow: {tp.source} -> {tp.sink}",
        },
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": tp.source},
                },
            },
        ],
        "properties": {
            "sanitized": tp.sanitized,
            "path": tp.path,
        },
    }


def format_sarif(
    paths: list[TaintPath],
    include_sanitized: bool = False,
) -> dict[str, object]:
    """Format taint paths as a SARIF JSON structure."""
    filtered = [
        p for p in paths
        if not _should_skip(p, include_sanitized)
    ]
    results = [_sarif_result(p) for p in filtered]
    return {
        "version": "2.1.0",
        "$schema": (
            "https://raw.githubusercontent.com/oasis-tcs/"
            "sarif-spec/main/sarif-2.1/"
            "schema/sarif-schema-2.1.0.json"
        ),
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "call-graph",
                        "version": "0.1.0",
                    },
                },
                "results": results,
            },
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Source-to-sink taint analysis via call graph",
    )
    parser.add_argument(
        "directory",
        help="Root directory of Python source to analyze",
    )
    parser.add_argument(
        "--sarif",
        action="store_true",
        help="Output in SARIF format",
    )
    parser.add_argument(
        "--include-sanitized",
        action="store_true",
        help="Include sanitized paths in output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the call-graph CLI."""
    args = parse_args(argv)
    graph = build_graph(args.directory)
    paths = find_taint_paths(graph)
    if args.sarif:
        sarif = format_sarif(paths, args.include_sanitized)
        print(json.dumps(sarif, indent=2))
    else:
        report = format_text_report(paths, args.include_sanitized)
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
