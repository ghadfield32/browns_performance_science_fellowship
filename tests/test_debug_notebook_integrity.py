from __future__ import annotations

from pathlib import Path
import subprocess
import sys


SCRIPT = Path("scripts/debug_notebook_integrity.py")


def _run_checker(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(path), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_flow_detects_missing_prerequisites_in_percent_script(tmp_path: Path) -> None:
    notebook_script = tmp_path / "repro_missing_df.py"
    notebook_script.write_text(
        "\n".join(
            [
                "# %% [markdown]",
                "# Analysis",
                "# %%",
                "model = object()",
                "# %% [markdown]",
                "# df = load_tracking_data('data.csv')",
                "# %%",
                "segmented_df = detect_segments(df, model)",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_checker(notebook_script, "--flow", "--strict")

    assert result.returncode == 1
    assert "Top-level use-before-define risk" in result.stdout
    assert "df" in result.stdout


def test_flow_ignores_list_comprehension_locals(tmp_path: Path) -> None:
    notebook_script = tmp_path / "repro_comprehensions.py"
    notebook_script.write_text(
        "\n".join(
            [
                "# %%",
                "items = [1, 2, 3]",
                "# %%",
                "scaled = [p * 2 for p in items]",
                "bullets = [f'- {line}' for line in ['a', 'b']]",
            ]
        ),
        encoding="utf-8",
    )

    result = _run_checker(notebook_script, "--flow", "--strict")

    assert result.returncode == 0
    assert "No notebook integrity issues found." in result.stdout
