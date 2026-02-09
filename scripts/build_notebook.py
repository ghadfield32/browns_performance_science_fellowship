"""Generate the analysis scaffold notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# Browns Tracking Analysis Template\n\n"
            "This scaffold is aligned to the Performance Science work project brief and focuses on:\n"
            "1) speed-band workload metrics, 2) rolling peak-demand windows,\n"
            "3) transparent session segmentation, and 4) slide-ready visuals."
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 0. Setup\n"
            "- Update thresholds/config values as needed for your final narrative.\n"
            "- Keep outputs reproducible by saving figures/tables to `outputs/`."
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## Unit Assumptions\n"
            "- Speed conversion: yards/second -> mph (x 2.0454545).\n"
            "- Acceleration conversion: yards/second^2 -> m/second^2 (x 0.9144)."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "from IPython.display import display\n"
            "\n"
            "from browns_tracking.metrics import (\n"
            "    compute_peak_demand_timeseries,\n"
            "    peak_distance_table,\n"
            "    relative_speed_bands,\n"
            "    session_extrema_table,\n"
            "    summarize_speed_bands,\n"
            "    top_non_overlapping_windows,\n"
            ")\n"
            "from browns_tracking.config import default_project_paths\n"
            "from browns_tracking.pipeline import (\n"
            "    compute_data_quality_summary,\n"
            "    compute_session_event_counts,\n"
            "    load_tracking_data,\n"
            "    split_early_late_summary,\n"
            "    summarize_session,\n"
            ")\n"
            "from browns_tracking.presets import preferred_performance_model\n"
            "from browns_tracking.segmentation import (\n"
            "    build_coach_phase_summary,\n"
            "    detect_segments,\n"
            "    summarize_segments,\n"
            ")\n"
            "from browns_tracking.visuals import (\n"
            "    plot_intensity_timeline,\n"
            "    plot_movement_map,\n"
            "    plot_peak_demand_summary,\n"
            "    save_figure,\n"
            ")\n"
            "\n"
            "pd.set_option('display.max_columns', 100)\n"
            "pd.set_option('display.width', 200)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "paths = default_project_paths()\n"
            "DATA_PATH = paths.data_file\n"
            "OUTPUT_DIR = paths.output_dir\n"
            "FIG_DIR = OUTPUT_DIR / 'figures'\n"
            "TABLE_DIR = OUTPUT_DIR / 'tables'\n"
            "FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
            "TABLE_DIR.mkdir(parents=True, exist_ok=True)\n"
            "model = preferred_performance_model()\n"
            "\n"
            "model.name, DATA_PATH"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Preferred Threshold Model"))
    cells.append(
        nbf.v4.new_code_cell(
            "model_summary = {\n"
            "    'model_name': model.name,\n"
            "    'rationale': model.rationale,\n"
            "    'hsr_threshold_mph': model.peak_demand_config.hsr_threshold_mph,\n"
            "    'accel_threshold_ms2': model.peak_demand_config.accel_threshold_ms2,\n"
            "    'decel_threshold_ms2': model.peak_demand_config.decel_threshold_ms2,\n"
            "    'rest_speed_threshold_mph': model.segmentation_config.rest_speed_threshold_mph,\n"
            "    'rest_min_duration_s': model.segmentation_config.rest_min_duration_s,\n"
            "    'speed_bands': [\n"
            "        f\"{b.name}: {b.lower_mph}-{b.upper_mph if b.upper_mph is not None else 'max'} mph\"\n"
            "        for b in model.absolute_speed_bands\n"
            "    ],\n"
            "}\n"
            "pd.Series(model_summary, name='value').to_frame()"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Definitions and Threshold Rationale"))
    cells.append(
        nbf.v4.new_code_cell(
            "definitions = pd.DataFrame(\n"
            "    [\n"
            "        {'definition': 'Speed bands (mph)', 'value': ', '.join([f\"{b.name} {b.lower_mph}-{b.upper_mph if b.upper_mph is not None else 'max'}\" for b in model.absolute_speed_bands])},\n"
            "        {'definition': 'HSR threshold (mph)', 'value': f\"{model.peak_demand_config.hsr_threshold_mph:.1f}\"},\n"
            "        {'definition': 'Sprint threshold (mph)', 'value': '16.0'},\n"
            "        {'definition': 'Accel/Decel thresholds (m/s^2)', 'value': f\">= {model.peak_demand_config.accel_threshold_ms2:.1f} / <= {model.peak_demand_config.decel_threshold_ms2:.1f}\"},\n"
            "        {'definition': 'Event definition', 'value': 'Contiguous threshold exposure >= 1.0 s'},\n"
            "        {'definition': 'Relative bands', 'value': 'Anchored to session max speed'},\n"
            "        {'definition': 'Rationale', 'value': 'Thresholds reflect common team-practice reporting and coach readability'},\n"
            "    ]\n"
            ")\n"
            "display(definitions)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 1. Load Data and Session QA Summary"))
    cells.append(
        nbf.v4.new_code_cell(
            "df = load_tracking_data(DATA_PATH)\n"
            "qa_summary = compute_data_quality_summary(df)\n"
            "session_summary = pd.Series(summarize_session(df), name='value').to_frame()\n"
            "display(pd.DataFrame([qa_summary]))\n"
            "display(session_summary)\n"
            "pd.DataFrame([qa_summary]).to_csv(TABLE_DIR / 'data_quality_summary.csv', index=False)\n"
            "\n"
            "df.head()"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 2. Speed Bands (Absolute + Relative-to-Max)"))
    cells.append(
        nbf.v4.new_code_cell(
            "absolute_bands = list(model.absolute_speed_bands)\n"
            "absolute_band_summary = summarize_speed_bands(df, absolute_bands)\n"
            "\n"
            "relative_bands = relative_speed_bands(\n"
            "    df['speed_mph'].max(),\n"
            "    percent_edges=model.relative_band_edges,\n"
            ")\n"
            "relative_band_summary = summarize_speed_bands(df, relative_bands)\n"
            "\n"
            "display(absolute_band_summary)\n"
            "display(relative_band_summary)\n"
            "\n"
            "absolute_band_summary.to_csv(TABLE_DIR / 'absolute_speed_band_summary.csv', index=False)\n"
            "relative_band_summary.to_csv(TABLE_DIR / 'relative_speed_band_summary.csv', index=False)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 3. Rolling Peak-Demand Windows + Event Counts"))
    cells.append(
        nbf.v4.new_code_cell(
            "peak_cfg = model.peak_demand_config\n"
            "\n"
            "rolling = compute_peak_demand_timeseries(df, peak_cfg)\n"
            "distance_table = peak_distance_table(rolling, peak_cfg.distance_windows_s)\n"
            "top_1m_distance_windows = top_non_overlapping_windows(\n"
            "    rolling, metric_column='distance_60s_yd', window_s=60, top_n=3\n"
            ")\n"
            "extrema_table = session_extrema_table(df)\n"
            "event_counts = compute_session_event_counts(\n"
            "    df,\n"
            "    hsr_threshold_mph=peak_cfg.hsr_threshold_mph,\n"
            "    accel_threshold_ms2=peak_cfg.accel_threshold_ms2,\n"
            "    decel_threshold_ms2=peak_cfg.decel_threshold_ms2,\n"
            ")\n"
            "early_late_summary = split_early_late_summary(\n"
            "    df,\n"
            "    hsr_threshold_mph=peak_cfg.hsr_threshold_mph,\n"
            "    accel_threshold_ms2=peak_cfg.accel_threshold_ms2,\n"
            "    decel_threshold_ms2=peak_cfg.decel_threshold_ms2,\n"
            ")\n"
            "\n"
            "display(distance_table)\n"
            "display(top_1m_distance_windows)\n"
            "display(extrema_table)\n"
            "display(pd.DataFrame([event_counts]))\n"
            "display(early_late_summary)\n"
            "\n"
            "distance_table.to_csv(TABLE_DIR / 'peak_distance_windows.csv', index=False)\n"
            "top_1m_distance_windows.to_csv(TABLE_DIR / 'top_1m_distance_windows.csv', index=False)\n"
            "extrema_table.to_csv(TABLE_DIR / 'session_extrema.csv', index=False)\n"
            "pd.DataFrame([event_counts]).to_csv(TABLE_DIR / 'session_event_counts.csv', index=False)\n"
            "early_late_summary.to_csv(TABLE_DIR / 'early_vs_late_summary.csv', index=False)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 4. Session Segmentation and Coach Phases (Merged for Context)"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "segmented_df, segment_boundaries = detect_segments(df, model.segmentation_config)\n"
            "segment_summary = summarize_segments(segmented_df, speed_bands=absolute_bands)\n"
            "coach_df, coach_phase_summary = build_coach_phase_summary(\n"
            "    segmented_df,\n"
            "    min_phase_duration_s=30.0,\n"
            "    max_phases=8,\n"
            "    hsr_threshold_mph=peak_cfg.hsr_threshold_mph,\n"
            "    accel_threshold_ms2=peak_cfg.accel_threshold_ms2,\n"
            "    decel_threshold_ms2=peak_cfg.decel_threshold_ms2,\n"
            ")\n"
            "\n"
            "raw_segment_count = int(segment_boundaries.shape[0])\n"
            "print(f'Raw algorithmic segments detected: {raw_segment_count}')\n"
            "display(coach_phase_summary)\n"
            "\n"
            "segment_boundaries.to_csv(TABLE_DIR / 'raw_segment_boundaries.csv', index=False)\n"
            "segment_summary.to_csv(TABLE_DIR / 'raw_segment_summary.csv', index=False)\n"
            "coach_phase_summary.to_csv(TABLE_DIR / 'coach_phase_summary.csv', index=False)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 5. Visual Template 1: Movement Map (X-Y)"))
    cells.append(
        nbf.v4.new_code_cell(
            "fig1, _ = plot_movement_map(coach_df, segment_col='coach_phase_label', highlight_top_n=3)\n"
            "save_figure(fig1, FIG_DIR / 'movement_map.png')\n"
            "fig1"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 6. Visual Template 2: Intensity Timeline"))
    cells.append(
        nbf.v4.new_code_cell(
            "fig2, _ = plot_intensity_timeline(\n"
            "    segmented_df,\n"
            "    top_windows=top_1m_distance_windows,\n"
            "    hsr_threshold_mph=peak_cfg.hsr_threshold_mph,\n"
            ")\n"
            "save_figure(fig2, FIG_DIR / 'intensity_timeline.png')\n"
            "fig2"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 7. Visual Template 3: Peak-Demand Summary"))
    cells.append(
        nbf.v4.new_code_cell(
            "fig3, _ = plot_peak_demand_summary(distance_table, extrema_table)\n"
            "save_figure(fig3, FIG_DIR / 'peak_demand_summary.png')\n"
            "fig3"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 8. Slide Export Checklist\n"
            "- `outputs/figures/movement_map.png`\n"
            "- `outputs/figures/intensity_timeline.png`\n"
            "- `outputs/figures/peak_demand_summary.png`\n"
            "- `outputs/tables/coach_phase_summary.csv` for coach-context phases\n"
            "- `outputs/tables/session_event_counts.csv` and `outputs/tables/early_vs_late_summary.csv`\n"
            "- `outputs/tables/*.csv` supporting tables for notebook/slide text"
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


def main() -> None:
    notebook = build_notebook()
    output_path = Path("notebooks/01_tracking_analysis_template.ipynb")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
