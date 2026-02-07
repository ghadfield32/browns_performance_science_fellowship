# Browns Performance Science Fellowship Project

This repository contains a reproducible Python analysis pipeline for a single-player 10 Hz tracking session, aligned to the work-project brief in `docs/browns_docs/performance_science_fellow_work_project.pptx`.

## Project goals

- Build a reproducible workflow from raw tracking data.
- Derive clear external workload metrics.
- Produce coach-readable visuals and practical takeaways.

## Setup with uv

```bash
uv venv
uv sync --group dev
```

Run the starter session summary CLI:

```bash
uv run browns-tracking-summary
```

Generate the scaffold notebook:

```bash
uv run python scripts/build_notebook.py
```

Open `notebooks/01_tracking_analysis_template.ipynb` and run cells top-to-bottom.
The notebook exports slide-ready figures to `outputs/figures/`.

Generate the presentation-first notebook (copy/paste tables + text blocks for slides):

```bash
uv run python scripts/build_presentation_notebook.py
```

Open `notebooks/02_coach_slide_ready_template.ipynb` for a PowerPoint drafting workflow.

Batch-export the same first three visual templates (plus support tables) without opening Jupyter:

```bash
uv run python scripts/render_visual_templates.py
```

Batch-export presentation-first assets (slide text + formatted tables + figure set):

```bash
uv run python scripts/render_presentation_assets.py
```

## Coach-context outputs

- Raw algorithmic segmentation is collapsed into coach-readable phases (default max: 8).
- Movement maps now emphasize density + top workload phases instead of dozens of raw block labels.
- Event context is exported for submission narrative:
  - `outputs/tables/session_event_counts.csv`
  - `outputs/tables/early_vs_late_summary.csv`
  - `outputs/tables/coach_phase_summary.csv`

## Repository layout

- `data/`: Primary data inputs (`tracking_data.csv`).
- `config/`: Versioned runtime config (`browns_tracking.yaml`).
- `docs/project_setup_and_plan.md`: Requirements and analysis plan.
- `docs/browns_docs/`: Source assignment brief and legacy copy of provided assets.
- `src/browns_tracking/`: Reusable analysis code.
- `notebooks/`: Analysis notebook scaffold.
- `scripts/build_notebook.py`: Notebook generator.
- `scripts/build_presentation_notebook.py`: Presentation-first notebook generator.
- `scripts/render_presentation_assets.py`: Presentation asset exporter.
- `tests/`: Unit tests for core helpers.

## Preferred threshold model (current default)

- Speed zones (mph): `Walk 0-3`, `Cruise 3-9`, `Run 9-13`, `HSR 13-16`, `Sprint >=16`
- HSR threshold: `13.0 mph`
- Accel/decel event thresholds: `>= +3.0 m/s^2`, `<= -3.0 m/s^2`
- Rest segmentation: `<1.2 mph` sustained for `>=25 s`

## Path automation

- Central runtime config lives in `config/browns_tracking.yaml`.
- Config is merged in this order (lowest to highest precedence):
  defaults in code -> `pyproject.toml` `[tool.browns_tracking.paths]` -> `config/browns_tracking.yaml` -> env vars.
- Pydantic validates the merged schema; OmegaConf handles nested merge behavior.
- Default data file is `data/tracking_data.csv`.
- Optional fallbacks include `docs/browns_docs/tracking_data.csv` for backward compatibility.
- Override any path via env vars:
  - `BROWNS_TRACKING_CONFIG_FILE` (optional external YAML file path)
  - `BROWNS_TRACKING_DATA_FILE`
  - `BROWNS_TRACKING_DATA_FALLBACKS` (comma-separated)
  - `BROWNS_TRACKING_OUTPUT_DIR`
  - `BROWNS_TRACKING_DOCS_PPTX`
  - `BROWNS_TRACKING_CREATE_OUTPUT_DIRS` (`true/false`)
  - `BROWNS_TRACKING_PROJECT_ROOT`
