# Browns Performance Science Fellowship Project

Reproducible Python pipeline for one anonymized player 10 Hz practice tracking session, aligned to `docs/browns_docs/performance_science_fellow_work_project.pptx`.

## Project intent

- Build a reproducible analysis pipeline from raw tracking data.
- Produce practical external workload metrics.
- Deliver coach-readable visuals and slide-ready narrative.

## Requirement coverage

- Session metrics: distance, speed, accel/decel, peak demands, demanding windows.
- Speed zones: explicit band definitions and threshold model.
- Segmentation: transparent algorithmic blocks plus coach-readable merged phases.
- Communication: notebook + slide-ready text/tables/figures with action-oriented takeaways.
- QA packaging: explicit `QC PASS` / `QC FAILED` status with compact checks + validation takeaways.
- Units: yards, mph, m/s^2 with explicit conversion assumptions.

## Data and unit assumptions

- Raw columns include timestamps, x/y position (yards), speed (`s`, yd/s), accel (`a`, `sa`, yd/s^2).
- Conversions used:
  - speed: `yd/s -> mph` multiply by `2.0454545`
  - accel: `yd/s^2 -> m/s^2` multiply by `0.9144`
- Default thresholds:
  - Speed bands (mph): `Walk 0-3`, `Cruise 3-9`, `Run 9-13`, `HSR 13-16`, `Sprint >=16`
  - HSR threshold: `13.0 mph`
  - Sprint threshold: `16.0 mph`
  - Accel/decel thresholds: `>= +3.0 / <= -3.0 m/s^2`
  - Event definition: contiguous threshold exposure `>= 1.0s`

## Quickstart

```bash
uv venv
uv sync --group dev
```

## Main workflows

### 1) Regenerate notebooks (recommended after code changes)

```bash
uv run --group dev python scripts/build_notebook.py
uv run --group dev python scripts/build_presentation_notebook.py
```

Generated notebooks:
- `notebooks/01_tracking_analysis_template.ipynb`
- `notebooks/02_coach_slide_ready_template.ipynb`

### 2) Export analysis artifacts directly (no Jupyter required)

```bash
uv run python scripts/render_visual_templates.py --output-dir outputs
uv run python scripts/render_presentation_assets.py --output-dir outputs
```

`render_visual_templates.py` is the single compute step (validation + metrics + figures + `results.json`).
`render_presentation_assets.py` only reads those outputs and formats slide assets.

### 3) Debug notebook execution errors (root-cause first)

```bash
python3 scripts/debug_notebook_integrity.py notebooks/01_tracking_analysis_template.ipynb notebooks/02_coach_slide_ready_template.ipynb --strict --flow
```

For VS Code `# %%` notebook scripts, run the same command on the `.py` file to detect markdown leakage into code cells.
Typical failure root cause: markdown bullet lines (e.g., `- item`) in a Python code cell.
For `NameError` execution-order failures (e.g., `df` not defined), `--flow` reports unresolved top-level dependencies by cell.

## Notebook organization

### `notebooks/01_tracking_analysis_template.ipynb`

- Single-pass compute notebook.
- Writes `outputs/results.json`, `outputs/phase_table.csv`, `outputs/peak_windows.csv`.
- Includes explicit validation gates (cadence, gaps, position jumps, speed alignment).
- Exports `outputs/figures/01_space.png`, `outputs/figures/02_time.png`, `outputs/figures/03_peaks.png`.

### `notebooks/02_coach_slide_ready_template.ipynb`

- Contract-consumer notebook (no heavy recomputation).
- Reads `outputs/results.json` + `outputs/tables/*.csv`.
- Copy/paste text sections for definitions, QA, and actionable takeaways.
- Slide tables for speed zones, peak windows, event counts, coach phases, and early-vs-late.
- Figure alias cells aligned with presentation workflow.

## Output artifacts

### Figures (`outputs/figures`)

- `01_space.png`
- `02_time.png`
- `03_peaks.png`
- `coach_slide_movement_map.png`
- `coach_slide_intensity_timeline.png`
- `coach_slide_peak_demand_summary.png`

Movement map style is coach-focused:
- density background + neutral full-session path
- highlight top 1-3 demanding windows
- direct window/phase annotations on plot (no giant segment legend)

### Tables (`outputs/tables`)

Contract-first:
- `phase_table.csv` (also at `outputs/phase_table.csv`)
- `peak_windows.csv` (also at `outputs/peak_windows.csv`)
- `validation_gates.csv`

Core:
- `absolute_speed_band_summary.csv`
- `relative_speed_band_summary.csv`
- `peak_distance_windows.csv`
- `top_1m_distance_windows.csv`
- `top_windows_by_duration.csv`
- `session_extrema.csv`
- `metrics_summary.csv`

Context/QA:
- `session_event_counts.csv`
- `early_vs_late_summary.csv`
- `data_quality_summary.csv`
- `qc_checks.csv`
- `coach_phase_summary.csv`

Raw segmentation (kept for transparency):
- `raw_segment_boundaries.csv`
- `raw_segment_summary.csv`
- `segment_boundaries.csv`
- `segment_summary.csv`

Presentation tables:
- `slide_1_data_quality_table.csv`
- `slide_2_speed_zone_table.csv`
- `slide_3_peak_distance_table.csv`
- `slide_3_extrema_table.csv`
- `slide_3_event_counts_table.csv`
- `slide_3_top_windows_table.csv`
- `slide_4_segment_table.csv`
- `slide_5_early_late_table.csv`

Layered output directories:
- `outputs/qc_summary/` (QA tables + QC status)
- `outputs/metrics_summary/` (session metrics package)
- `outputs/coach_package/` (coach bullets + package-ready tables)

### Slide text (`outputs/slide_text`)

- `slide_1_session_snapshot.txt`
- `slide_1_definitions_and_assumptions.txt`
- `slide_1_data_quality_takeaways.txt`
- `slide_2_speed_zone_takeaways.txt`
- `slide_3_peak_takeaways.txt`
- `slide_4_segment_takeaways.txt`
- `slide_5_early_late_takeaways.txt`

## QA policy (explicit)

- Sampling gaps are flagged (`dt_s > 0.15s` by default), not silently dropped.
- Step-distance outliers are flagged via high-quantile threshold (default 99.5th percentile), not silently dropped.
- Validation gates produce PASS/WARN status for cadence, max gap, XY jump plausibility, and provided-vs-derived speed agreement.
- QA tables and narrative state handling decisions directly for auditability.

## Config and path automation

- Primary runtime config: `config/browns_tracking.yaml`
- Merge precedence:
  - code defaults
  - `pyproject.toml` `[tool.browns_tracking.paths]`
  - `config/browns_tracking.yaml`
  - environment variables
- Validation/merge stack:
  - `Pydantic` for schema validation
  - `OmegaConf` for nested config merging

Environment overrides:
- `BROWNS_TRACKING_PROJECT_ROOT`
- `BROWNS_TRACKING_CONFIG_FILE`
- `BROWNS_TRACKING_DATA_FILE`
- `BROWNS_TRACKING_DATA_FALLBACKS` (comma-separated)
- `BROWNS_TRACKING_OUTPUT_DIR`
- `BROWNS_TRACKING_DOCS_PPTX`
- `BROWNS_TRACKING_CREATE_OUTPUT_DIRS` (`true/false`)

## Repository layout

- `data/`: primary data input (`tracking_data.csv`)
- `config/`: runtime config (`browns_tracking.yaml`)
- `docs/`: brief and planning docs
- `src/browns_tracking/`: analysis package
- `scripts/`: notebook builders + exporters
- `notebooks/`: generated analysis and slide notebooks
- `outputs/`: generated figures/tables/slide text
- `tests/`: unit tests
- `WORK_LOG.md`: compact engineering log (done / doing / next by topic)

## Submission checklist

- Regenerate notebooks from scripts.
- Re-export all artifacts to `outputs/`.
- Confirm `outputs/slide_text/*.txt` are current.
- Confirm movement map uses highlighted phases (not dense raw-block legend).
- Include README + code + outputs for fully reproducible review.
