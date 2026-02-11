"""Generate the analysis notebook scaffold."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nbf


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# Browns Tracking Analysis Template\n\n"
            "This notebook is organized around three coaching questions:\n"
            "1. Where did the player spend time (role/space)?\n"
            "2. When did intensity happen (session structure + peaks)?\n"
            "3. What were peak demands and repeatable windows?\n\n"
            "It computes once, validates once, and writes a reusable results contract."
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 0. Setup\n"
            "- Unit conversions are explicit and stored in `outputs/results.json`.\n"
            "- Acceleration policy: `a` is raw magnitude; workload accel/decel logic uses signed `sa`.\n"
            "- Continuity policy: windows/events/lines reset at flagged gaps and improbable jumps.\n"
            "- Notebook 02 should only read this notebook's outputs."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "from pathlib import Path\n"
            "from IPython.display import Image, Markdown, display\n"
            "\n"
            "from browns_tracking.config import default_project_paths\n"
            "from browns_tracking.pipeline import load_tracking_data\n"
            "from browns_tracking.presets import preferred_performance_model\n"
            "from browns_tracking.results_contract import (\n"
            "    run_session_analysis,\n"
            "    write_results_contract,\n"
            "    write_results_tables,\n"
            ")\n"
            "from browns_tracking.visuals import (\n"
            "    close_figures,\n"
            "    plot_intensity_timeline,\n"
            "    plot_movement_map,\n"
            "    plot_peak_demand_summary,\n"
            "    save_figure,\n"
            ")\n"
            "\n"
            "pd.set_option('display.max_columns', 120)\n"
            "pd.set_option('display.width', 220)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "paths = default_project_paths()\n"
            "DATA_PATH = paths.data_file\n"
            "OUTPUT_DIR = paths.output_dir\n"
            "FIG_DIR = OUTPUT_DIR / 'figures'\n"
            "FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
            "model = preferred_performance_model()\n"
            "\n"
            "pd.Series({\n"
            "    'Data path': str(DATA_PATH),\n"
            "    'Output dir': str(OUTPUT_DIR),\n"
            "    'Model': model.name,\n"
            "    'Rationale': model.rationale,\n"
            "}, name='value').to_frame()"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "def show_saved_png(path, caption='', width=1200):\n"
            "    png = Path(path)\n"
            "    if caption:\n"
            "        display(Markdown(f'**{caption}**'))\n"
            "    if not png.exists():\n"
            "        display(Markdown(f':warning: Missing `{png}`. Run Notebook 01 from top.'))\n"
            "        return\n"
            "    display(Image(filename=str(png), width=width))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Assignment Alignment Checklist\n"
            "- Reproducible pipeline: one compute pass -> `outputs/results.json` + canonical tables.\n"
            "- Meaningful workload metrics: speed zones, events, peak windows by duration, phase profile, early-vs-late drift.\n"
            "- Coach visuals (<30s scan): `01_space.png`, `02_time.png`, `03_peaks.png`.\n"
            "- QC package every run: `qc_summary/*.csv` + PASS/WARN/FAIL status and gate takeaways.\n"
            "- Explicit assumptions: thresholds + unit conversions stored in contract and slide text.\n"
            "- Practical insight over volume: top windows and phase actions prioritized over raw metric dumps."
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 1. Run Analysis Once (Results Contract Source)"))
    cells.append(
        nbf.v4.new_code_cell(
            "df = load_tracking_data(DATA_PATH)\n"
            "results = run_session_analysis(df, model=model)\n"
            "\n"
            "display(pd.Series(results.session_summary, name='value').to_frame())\n"
            "display(pd.DataFrame([{'qc_status': results.qc_status}]))\n"
            "display(pd.DataFrame([{\n"
            "    'analysis_rows': len(results.analysis_df),\n"
            "    'viz_rows': len(results.viz_df),\n"
            "    'viz_share_pct': (len(results.viz_df) / max(len(results.analysis_df), 1)) * 100.0,\n"
            "}]))\n"
            "display(pd.DataFrame([results.qa_summary]))\n"
            "display(results.qc_checks)\n"
            "display(results.validation_gates)\n"
            "for line in results.validation_takeaways:\n"
            "    print(f'- {line}')"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 2. Core Story Tables"))
    cells.append(
        nbf.v4.new_code_cell(
            "display(results.coach_phase_summary)\n"
            "display(results.peak_windows)\n"
            "display(results.top_windows_by_duration)\n"
            "display(results.distance_table)\n"
            "display(results.metrics_summary)\n"
            "display(pd.DataFrame([results.event_counts]))"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 3. Coach-Ready Visuals (3 Figure Story)"))
    cells.append(
        nbf.v4.new_code_cell(
            "fig1, _ = plot_movement_map(\n"
            "    results.viz_df,\n"
            "    segment_col='coach_phase_label',\n"
            "    highlight_top_n=3,\n"
            "    highlight_windows=results.peak_windows,\n"
            "    show_full_trace=False,\n"
            "    show_role_hypothesis=False,\n"
            ")\n"
            "fig2, _ = plot_intensity_timeline(\n"
            "    results.viz_df,\n"
            "    top_windows=results.peak_windows,\n"
            "    hsr_threshold_mph=model.peak_demand_config.hsr_threshold_mph,\n"
            "    phase_col='coach_intensity_level',\n"
            "    show_cumulative_distance=False,\n"
            "    show_hsr_ticks=False,\n"
            "    show_early_late_comparison=False,\n"
            "    display_resample_s=1,\n"
            "    distance_window_s=60,\n"
            ")\n"
            "fig3, _ = plot_peak_demand_summary(\n"
            "    results.distance_table,\n"
            "    results.extrema_table,\n"
            "    peak_windows=results.top_windows_by_duration,\n"
            ")\n"
            "\n"
            "save_figure(fig1, FIG_DIR / '01_space.png')\n"
            "save_figure(fig2, FIG_DIR / '02_time.png')\n"
            "save_figure(fig3, FIG_DIR / '03_peaks.png')\n"
            "\n"
            "close_figures([fig1, fig2, fig3])\n"
            "\n"
            "show_saved_png(FIG_DIR / '01_space.png', 'Where: Spatial usage and key peak-demand windows')\n"
            "show_saved_png(FIG_DIR / '02_time.png', 'When: Speed + distance-rate timeline with phase context and highlighted top windows')\n"
            "show_saved_png(FIG_DIR / '03_peaks.png', 'What: Peak intensity (yd/min) across durations with top-window context')\n"
            "\n"
            "(\n"
            "    'Saved and displayed',\n"
            "    FIG_DIR / '01_space.png',\n"
            "    FIG_DIR / '02_time.png',\n"
            "    FIG_DIR / '03_peaks.png',\n"
            ")"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 4. Write Results Contract + Canonical Tables"))
    cells.append(
        nbf.v4.new_code_cell(
            "table_paths = write_results_tables(results, OUTPUT_DIR)\n"
            "contract_path = write_results_contract(\n"
            "    results,\n"
            "    input_path=DATA_PATH,\n"
            "    output_dir=OUTPUT_DIR,\n"
            "    table_paths=table_paths,\n"
            "    figure_paths={\n"
            "        'space': FIG_DIR / '01_space.png',\n"
            "        'time': FIG_DIR / '02_time.png',\n"
            "        'peaks': FIG_DIR / '03_peaks.png',\n"
            "    },\n"
            ")\n"
            "print('Contract:', contract_path)\n"
            "print('Canonical tables:', OUTPUT_DIR / 'phase_table.csv', OUTPUT_DIR / 'peak_windows.csv')"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 5. Hand-off to Notebook 02\n"
            "Notebook 02 should only read from:\n"
            "- `outputs/results.json`\n"
            "- `outputs/phase_table.csv`\n"
            "- `outputs/peak_windows.csv`\n"
            "- `outputs/session_structure_map.csv`\n"
            "- `outputs/qc_summary/*` and `outputs/metrics_summary/*`\n"
            "- `outputs/coach_package/coach_package_bullets.txt`\n"
            "- `outputs/tables/*.csv`\n"
            "- `outputs/tables/analysis_df.csv`, `outputs/tables/viz_df.csv`\n"
            "- `outputs/qc_summary/data_qc_summary.json`\n"
            "- `outputs/figures/01_space.png`, `02_time.png`, `03_peaks.png`"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 6. Final Deck Outline (6 Slides)"))
    cells.append(
        nbf.v4.new_code_cell(
            "deck_outline = pd.DataFrame([\n"
            "    {\n"
            "        'slide': 1,\n"
            "        'title': 'Session Snapshot + Validation Gates',\n"
            "        'figure_or_table': (\n"
            "            'outputs/tables/slide_1_data_quality_table.csv; '\n"
            "            'outputs/tables/slide_1_validation_gates_table.csv; '\n"
            "            'outputs/tables/slide_1_qc_checks_table.csv'\n"
            "        ),\n"
            "        'one_line_takeaway': 'Validation gate summary and trust policy for this session.'\n"
            "    },\n"
            "    {\n"
            "        'slide': 2,\n"
            "        'title': 'Where: Spatial Usage and Role Signature',\n"
            "        'figure_or_table': (\n"
            "            'outputs/figures/01_space.png; outputs/phase_table.csv; '\n"
            "            'outputs/session_structure_map.csv'\n"
            "        ),\n"
            "        'one_line_takeaway': 'The session lived in specific field zones and repeated role-consistent movement patterns.'\n"
            "    },\n"
            "    {\n"
            "        'slide': 3,\n"
            "        'title': 'When: Intensity Timeline and Session Structure',\n"
            "        'figure_or_table': 'outputs/figures/02_time.png; outputs/tables/slide_3_top_windows_table.csv',\n"
            "        'one_line_takeaway': 'High-intensity work clustered in identifiable blocks rather than being evenly distributed.'\n"
            "    },\n"
            "    {\n"
            "        'slide': 4,\n"
            "        'title': 'What: Peak Demands and Repeatable Windows',\n"
            "        'figure_or_table': (\n"
            "            'outputs/figures/03_peaks.png; outputs/tables/slide_3_peak_distance_table.csv; '\n"
            "            'outputs/tables/slide_3_event_counts_table.csv'\n"
            "        ),\n"
            "        'one_line_takeaway': 'Top windows define the true worst-case demands to anchor drill and conditioning targets.'\n"
            "    },\n"
            "    {\n"
            "        'slide': 5,\n"
            "        'title': 'Phase-Level Load Profile',\n"
            "        'figure_or_table': 'outputs/tables/slide_4_segment_table.csv',\n"
            "        'one_line_takeaway': 'Merged coach phases show which blocks carried most distance and high-speed stress.'\n"
            "    },\n"
            "    {\n"
            "        'slide': 6,\n"
            "        'title': 'Early vs Late Readiness Signal',\n"
            "        'figure_or_table': 'outputs/tables/slide_5_early_late_table.csv',\n"
            "        'one_line_takeaway': 'Late-session drift quantifies whether second-half output matched training intent.'\n"
            "    },\n"
            "])\n"
            "display(deck_outline)\n"
            "(OUTPUT_DIR / 'tables').mkdir(parents=True, exist_ok=True)\n"
            "deck_outline.to_csv(OUTPUT_DIR / 'tables' / 'final_deck_outline.csv', index=False)"
        )
    )

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    }
    return nb


def _validate_generated_notebook(notebook: nbf.NotebookNode) -> None:
    """Fail fast if any generated code cell is syntactically invalid."""
    for idx, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            ast.parse(source)
        except SyntaxError as exc:
            lines = source.splitlines()
            bad_line = lines[exc.lineno - 1] if exc.lineno and exc.lineno - 1 < len(lines) else ""
            raise SyntaxError(
                f"Generated notebook has invalid code at cell {idx}, line {exc.lineno}: "
                f"{exc.msg} | {bad_line}"
            ) from exc


def main() -> None:
    notebook = build_notebook()
    _validate_generated_notebook(notebook)
    output_path = Path("notebooks/01_tracking_analysis_template.ipynb")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
