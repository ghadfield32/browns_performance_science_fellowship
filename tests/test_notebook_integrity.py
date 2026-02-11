from __future__ import annotations

import ast
import json
from pathlib import Path


NOTEBOOKS = (
    Path("notebooks/01_tracking_analysis_template.ipynb"),
    Path("notebooks/02_coach_slide_ready_template.ipynb"),
)


def test_generated_notebook_code_cells_are_syntax_valid() -> None:
    for notebook_path in NOTEBOOKS:
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
        for idx, cell in enumerate(payload.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            try:
                ast.parse(source)
            except SyntaxError as exc:
                raise AssertionError(
                    f"{notebook_path} has invalid syntax in cell {idx}, line {exc.lineno}: {exc.msg}"
                ) from exc
