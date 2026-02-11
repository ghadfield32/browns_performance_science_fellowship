"""Inspect notebook files for syntax and cell-structure issues.

This script is intentionally diagnostic-only: it reports root causes and does
not rewrite files.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable


MARKDOWN_HEADING_RE = re.compile(r"^\s*#{1,6}\s+\S")
MARKDOWN_BULLET_RE = re.compile(r"^\s*[-*]\s+\S")


@dataclass(frozen=True)
class CellIssue:
    file_path: str
    cell_index: int
    cell_type: str
    line_no: int
    message: str
    line_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Debug notebook execution issues by checking code-cell syntax and "
            "markdown leakage in .ipynb and #%% .py notebook scripts."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Notebook paths (.ipynb or .py in #%% format).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code when any issue is found.",
    )
    parser.add_argument(
        "--flow",
        action="store_true",
        help="Enable top-level execution-flow checks (use-before-define diagnostics).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_issues: list[CellIssue] = []

    for raw_path in args.paths:
        path = Path(raw_path)
        if not path.exists():
            print(f"[MISSING] {path}")
            all_issues.append(
                CellIssue(
                    file_path=str(path),
                    cell_index=-1,
                    cell_type="file",
                    line_no=0,
                    message="File does not exist.",
                    line_text="",
                )
            )
            continue

        if path.suffix == ".ipynb":
            issues = analyze_ipynb(path, check_flow=args.flow)
        elif path.suffix == ".py":
            issues = analyze_percent_script(path, check_flow=args.flow)
        else:
            print(f"[SKIP] {path} (unsupported extension)")
            continue

        all_issues.extend(issues)

    if not all_issues:
        print("No notebook integrity issues found.")
        return 0

    print("\nIssues found:")
    for issue in all_issues:
        location = f"{issue.file_path} | cell {issue.cell_index} | line {issue.line_no}"
        print(f"- {location} | {issue.message}")
        if issue.line_text:
            print(f"  -> {issue.line_text}")

    root_cause_hints = summarize_root_cause_hints(all_issues)
    if root_cause_hints:
        print("\nRoot-cause hints:")
        for hint in root_cause_hints:
            print(f"- {hint}")

    if args.strict:
        return 1
    return 0


def analyze_ipynb(path: Path, *, check_flow: bool) -> list[CellIssue]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = payload.get("cells", [])
    cell_types: dict[str, int] = {}
    for cell in cells:
        cell_type = str(cell.get("cell_type", "unknown"))
        cell_types[cell_type] = cell_types.get(cell_type, 0) + 1

    print(f"\n[IPYNB] {path}")
    print(
        "Cell counts: "
        + ", ".join(
            f"{kind}={count}" for kind, count in sorted(cell_types.items(), key=lambda x: x[0])
        )
    )

    issues: list[CellIssue] = []
    code_cells: list[tuple[int, str]] = []
    for idx, cell in enumerate(cells):
        cell_type = str(cell.get("cell_type", "unknown"))
        source = "".join(cell.get("source", []))
        first_line = source.strip().splitlines()[0] if source.strip() else ""
        print(f"  cell {idx:02d} | {cell_type:<8} | {first_line[:90]}")
        if cell_type != "code":
            continue
        code_cells.append((idx, source))
        issues.extend(check_code_cell(path=path, cell_index=idx, cell_type=cell_type, source=source))

    if check_flow:
        issues.extend(check_execution_flow(path=path, code_cells=code_cells))
    return issues


def analyze_percent_script(path: Path, *, check_flow: bool) -> list[CellIssue]:
    print(f"\n[PERCENT PY] {path}")
    cells = split_percent_cells(path.read_text(encoding="utf-8"))
    print(f"Cell counts: total={len(cells)}")

    issues: list[CellIssue] = []
    code_cells: list[tuple[int, str]] = []
    for idx, (cell_type, source) in enumerate(cells):
        first_line = source.strip().splitlines()[0] if source.strip() else ""
        print(f"  cell {idx:02d} | {cell_type:<8} | {first_line[:90]}")

        if cell_type == "markdown":
            issues.extend(
                check_markdown_percent_cell(path=path, cell_index=idx, cell_type=cell_type, source=source)
            )
            continue

        code_cells.append((idx, source))
        issues.extend(check_code_cell(path=path, cell_index=idx, cell_type=cell_type, source=source))
    if check_flow:
        issues.extend(check_execution_flow(path=path, code_cells=code_cells))
    return issues


def split_percent_cells(content: str) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    current_type = "code"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        if current_lines:
            cells.append((current_type, "\n".join(current_lines)))
            current_lines = []

    for line in content.splitlines():
        if line.startswith("# %%"):
            flush()
            current_type = "markdown" if "[markdown]" in line.lower() else "code"
            continue
        current_lines.append(line)
    flush()
    return cells


def check_markdown_percent_cell(
    *,
    path: Path,
    cell_index: int,
    cell_type: str,
    source: str,
) -> list[CellIssue]:
    issues: list[CellIssue] = []
    python_like_comment_lines: list[tuple[int, str]] = []

    for line_no, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        # In percent scripts, markdown content should remain commented.
        if not stripped.startswith("#"):
            issues.append(
                CellIssue(
                    file_path=str(path),
                    cell_index=cell_index,
                    cell_type=cell_type,
                    line_no=line_no,
                    message=(
                        "Uncommented text found in markdown-designated #%% cell; "
                        "this will be executed as Python."
                    ),
                    line_text=line,
                )
            )
            continue

        comment_body = stripped.lstrip("#").strip()
        if comment_body and looks_like_python_statement(comment_body):
            python_like_comment_lines.append((line_no, line))

    if len(python_like_comment_lines) >= 3:
        first_line_no, first_line = python_like_comment_lines[0]
        issues.append(
            CellIssue(
                file_path=str(path),
                cell_index=cell_index,
                cell_type=cell_type,
                line_no=first_line_no,
                message=(
                    "Markdown-designated #%% cell contains multiple Python-like commented "
                    "lines; likely a code cell was converted to markdown."
                ),
                line_text=first_line,
            )
        )
    return issues


def check_code_cell(
    *,
    path: Path,
    cell_index: int,
    cell_type: str,
    source: str,
) -> list[CellIssue]:
    issues: list[CellIssue] = []
    syntax_error = parse_syntax_error(source)
    if syntax_error is not None:
        line_no, msg, line_text = syntax_error
        root_hint = " (looks like markdown text in code cell)" if looks_like_markdown_line(line_text) else ""
        issues.append(
            CellIssue(
                file_path=str(path),
                cell_index=cell_index,
                cell_type=cell_type,
                line_no=line_no,
                message=f"SyntaxError: {msg}{root_hint}",
                line_text=line_text,
            )
        )

    for line_no, line in detect_markdown_like_lines(source):
        # Markdown-like lines in a valid Python code cell are unusual and are
        # often the direct source of the syntax error seen by users.
        issues.append(
            CellIssue(
                file_path=str(path),
                cell_index=cell_index,
                cell_type=cell_type,
                line_no=line_no,
                message="Markdown-like line found in code cell.",
                line_text=line,
            )
        )

    return deduplicate_issues(issues)


def check_execution_flow(
    *,
    path: Path,
    code_cells: list[tuple[int, str]],
) -> list[CellIssue]:
    """Check top-level name flow across code cells to catch use-before-define."""
    issues: list[CellIssue] = []
    defined_names: set[str] = set(dir(__builtins__))
    print("  flow-check: top-level name dependencies")

    for cell_index, source in code_cells:
        try:
            module = ast.parse(source)
        except SyntaxError:
            # Syntax issues are already reported by syntax checks.
            continue

        analyzer = TopLevelFlowAnalyzer()
        analyzer.visit(module)
        unresolved = sorted(
            name
            for name in analyzer.used
            if (
                name not in defined_names
                and name not in analyzer.defined
                and name not in analyzer.local_only
                and not name.startswith("__")
            )
        )

        sample_defined = ", ".join(sorted(analyzer.defined)[:6]) if analyzer.defined else "-"
        sample_unresolved = ", ".join(unresolved[:6]) if unresolved else "-"
        print(
            f"    cell {cell_index:02d} | defines: {sample_defined} | unresolved uses: {sample_unresolved}"
        )

        if unresolved:
            first_name = unresolved[0]
            line_no = _first_name_load_line(module, first_name)
            line_text = _line_text(source, line_no)
            issues.append(
                CellIssue(
                    file_path=str(path),
                    cell_index=cell_index,
                    cell_type="code",
                    line_no=line_no,
                    message=(
                        "Top-level use-before-define risk: "
                        f"{', '.join(unresolved[:8])}"
                    ),
                    line_text=line_text,
                )
            )
        defined_names.update(analyzer.defined)

    return issues


class TopLevelFlowAnalyzer(ast.NodeVisitor):
    """Collect top-level defined and used names without descending into nested defs."""

    def __init__(self) -> None:
        self.defined: set[str] = set()
        self.used: set[str] = set()
        # Comprehension target names are expression-local and should not be
        # treated as unresolved global dependencies.
        self.local_only: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.used.add(node.id)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.defined.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.defined.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.defined.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        if node.returns is not None:
            self.visit(node.returns)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.defined.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self.defined.update(_target_names(target))

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self.defined.update(_target_names(node.target))

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self.visit(node.target)
        self.defined.update(_target_names(node.target))

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self.defined.update(_target_names(node.target))
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.defined.update(_target_names(item.optional_vars))
        for stmt in node.body:
            self.visit(stmt)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name:
            self.defined.add(node.name)
        for stmt in node.body:
            self.visit(stmt)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension(node.generators)
        self.visit(node.elt)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension(node.generators)
        self.visit(node.elt)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension(node.generators)
        self.visit(node.key)
        self.visit(node.value)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension(node.generators)
        self.visit(node.elt)

    def _visit_comprehension(self, generators: list[ast.comprehension]) -> None:
        for comp in generators:
            self.local_only.update(_target_names(comp.target))
        for comp in generators:
            self.visit(comp.iter)
            for cond in comp.ifs:
                self.visit(cond)


def _target_names(target: ast.expr) -> set[str]:
    names: set[str] = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names.update(_target_names(elt))
    return names


def _first_name_load_line(module: ast.AST, name: str) -> int:
    class _LineFinder(ast.NodeVisitor):
        def __init__(self) -> None:
            self.line_no: int | None = None

        def visit_Name(self, node: ast.Name) -> None:
            if self.line_no is not None:
                return
            if node.id == name and isinstance(node.ctx, ast.Load):
                self.line_no = int(getattr(node, "lineno", 1))

    finder = _LineFinder()
    finder.visit(module)
    return finder.line_no or 1


def _line_text(source: str, line_no: int) -> str:
    lines = source.splitlines()
    if line_no <= 0 or line_no > len(lines):
        return ""
    return lines[line_no - 1]


def parse_syntax_error(source: str) -> tuple[int, str, str] | None:
    try:
        ast.parse(source)
    except SyntaxError as exc:
        lines = source.splitlines()
        line_text = lines[exc.lineno - 1] if exc.lineno and exc.lineno - 1 < len(lines) else ""
        line_no = int(exc.lineno or 0)
        return line_no, exc.msg, line_text
    return None


def detect_markdown_like_lines(source: str) -> Iterable[tuple[int, str]]:
    for line_no, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if MARKDOWN_HEADING_RE.match(line) or MARKDOWN_BULLET_RE.match(line):
            yield line_no, line


def looks_like_markdown_line(line: str) -> bool:
    return bool(MARKDOWN_HEADING_RE.match(line) or MARKDOWN_BULLET_RE.match(line))


def looks_like_python_statement(text: str) -> bool:
    prefixes = (
        "import ",
        "from ",
        "def ",
        "class ",
        "for ",
        "while ",
        "if ",
        "try:",
        "except ",
        "return ",
        "display(",
        "print(",
        "pd.",
    )
    if text.startswith(prefixes):
        return True
    if "=" in text and not text.startswith(("-", "*")):
        return True
    if text.endswith((")", "]", "}")) and "(" in text:
        return True
    return False


def deduplicate_issues(issues: list[CellIssue]) -> list[CellIssue]:
    seen: set[tuple[str, int, str, int, str, str]] = set()
    deduped: list[CellIssue] = []
    for issue in issues:
        key = (
            issue.file_path,
            issue.cell_index,
            issue.cell_type,
            issue.line_no,
            issue.message,
            issue.line_text,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def summarize_root_cause_hints(issues: list[CellIssue]) -> list[str]:
    hints: list[str] = []
    if any("markdown-designated" in issue.message for issue in issues):
        hints.append(
            "In #%% scripts, markdown cells must use `# %% [markdown]` and markdown lines "
            "must stay commented with `#`."
        )
    if any("likely a code cell was converted to markdown" in issue.message for issue in issues):
        hints.append(
            "A `# %% [markdown]` block appears to contain commented Python statements; "
            "convert that block back to a code cell."
        )
    if any("Markdown-like line found in code cell" in issue.message for issue in issues):
        hints.append(
            "A code cell contains markdown bullets/headings (e.g., `- item`, `## title`) "
            "that should be moved to a markdown cell."
        )
    if any("SyntaxError" in issue.message for issue in issues):
        hints.append(
            "Syntax errors are structural here; do not patch with defaults. "
            "Fix cell type/source so Python only sees valid code."
        )
    if any("Top-level use-before-define risk" in issue.message for issue in issues):
        hints.append(
            "A code cell references names not created earlier in the execution order. "
            "Run all prerequisite cells or restore accidentally commented/concealed pipeline cells."
        )
    return hints


if __name__ == "__main__":
    raise SystemExit(main())
