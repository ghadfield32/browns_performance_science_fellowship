# Work Log

## Notebook Execution Integrity
- 2026-02-09 Done: Root cause confirmed for `SyntaxError` (`- Speed conversion...`) as markdown text being executed in a Python code cell.
- 2026-02-09 Done: Added `scripts/debug_notebook_integrity.py` to inspect `.ipynb` and `# %%` `.py` notebook files with per-cell syntax diagnostics.
- 2026-02-09 Done: Extended integrity checker with `--flow` to flag top-level use-before-define dependencies across cells (e.g., `df` missing before segmentation).
- 2026-02-09 Done: Reproduced `NameError: df is not defined` from percent-script execution order; root cause is prerequisite analysis cells converted/commented out while downstream segmentation stayed active.
- 2026-02-09 Next: Run integrity checks before execution to catch cell-type/source errors early (`python3 scripts/debug_notebook_integrity.py ... --strict`).

## Coach/Player Report Pipeline
- 2026-02-09 Done: Notebook templates now display canonical visuals in-line and include deck map + coach/player summary sections.
- 2026-02-09 Done: Presentation export writes `outputs/tables/final_deck_outline.csv` for slide assembly consistency.
- 2026-02-09 Next: Keep Notebook 01 as single compute source and Notebook 02 as report consumer.

## Validation and Testing
- 2026-02-09 Done: Added notebook syntax regression test (`tests/test_notebook_integrity.py`) to prevent invalid code cells from entering repo.
- 2026-02-09 Done: Build scripts now fail fast if generated notebook code cells contain syntax errors.
- 2026-02-09 Done: Added flow-debug regression tests (`tests/test_debug_notebook_integrity.py`) for true missing-name detection and comprehension-local false-positive prevention.
- 2026-02-09 Next: Include notebook integrity check in pre-submission routine with lint/tests.
