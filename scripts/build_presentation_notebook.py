"""Generate a presentation-first notebook scaffold for coach slides."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# Coach Slide-Ready Workbook\n\n"
            "This notebook is optimized for PowerPoint drafting.\n"
            "Each section outputs copy/paste text and preformatted tables."
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 0. Setup and Tuned Threshold Model\n"
            "- Uses the preferred model preset for speed bands, HSR, accel/decel, and rest segmentation.\n"
            "- Exports text blocks to `outputs/slide_text/` and tables to `outputs/tables/`.\n"
            "- Unit conversions used in pipeline: yards/second -> mph (x 2.0454545), yards/second^2 -> m/second^2 (x 0.9144)."
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
            "    summarize_speed_bands,\n"
            "    top_non_overlapping_windows,\n"
            "    session_extrema_table,\n"
            ")\n"
            "from browns_tracking.config import default_project_paths\n"
            "from browns_tracking.pipeline import (\n"
            "    classify_hsr_exposure,\n"
            "    compute_data_quality_summary,\n"
            "    compute_session_event_counts,\n"
            "    load_tracking_data,\n"
            "    split_early_late_summary,\n"
            "    summarize_window_context,\n"
            "    summarize_session,\n"
            ")\n"
            "from browns_tracking.presets import preferred_performance_model\n"
            "from browns_tracking.presentation import (\n"
            "    build_slide_1_snapshot_text,\n"
            "    coach_early_late_table,\n"
            "    coach_extrema_table,\n"
            "    coach_peak_distance_table,\n"
            "    coach_segment_table,\n"
            "    coach_speed_band_table,\n"
            "    write_slide_text,\n"
            ")\n"
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
            "TABLE_DIR = OUTPUT_DIR / 'tables'\n"
            "TEXT_DIR = OUTPUT_DIR / 'slide_text'\n"
            "FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
            "TABLE_DIR.mkdir(parents=True, exist_ok=True)\n"
            "TEXT_DIR.mkdir(parents=True, exist_ok=True)\n"
            "model = preferred_performance_model()\n"
            "\n"
            "pd.Series({\n"
            "    'Model': model.name,\n"
            "    'Rationale': model.rationale,\n"
            "    'HSR threshold (mph)': model.peak_demand_config.hsr_threshold_mph,\n"
            "    'Accel threshold (m/s^2)': model.peak_demand_config.accel_threshold_ms2,\n"
            "    'Decel threshold (m/s^2)': model.peak_demand_config.decel_threshold_ms2,\n"
            "    'Rest threshold (mph)': model.segmentation_config.rest_speed_threshold_mph,\n"
            "    'Rest min duration (s)': model.segmentation_config.rest_min_duration_s,\n"
            "}, name='value').to_frame()"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 1. Run Core Analysis Once"))
    cells.append(
        nbf.v4.new_code_cell(
            "df = load_tracking_data(DATA_PATH)\n"
            "session_summary = summarize_session(df)\n"
            "qa_summary = compute_data_quality_summary(df)\n"
            "\n"
            "abs_bands = list(model.absolute_speed_bands)\n"
            "speed_band_summary = summarize_speed_bands(df, abs_bands)\n"
            "\n"
            "peak_cfg = model.peak_demand_config\n"
            "rolling = compute_peak_demand_timeseries(df, peak_cfg)\n"
            "distance_table = peak_distance_table(rolling, peak_cfg.distance_windows_s)\n"
            "top_windows = top_non_overlapping_windows(rolling, 'distance_60s_yd', window_s=60, top_n=3)\n"
            "extrema = session_extrema_table(df)\n"
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
            "seg_df, segment_boundaries = detect_segments(df, model.segmentation_config)\n"
            "segment_summary = summarize_segments(seg_df, speed_bands=abs_bands)\n"
            "coach_df, coach_phase_summary = build_coach_phase_summary(\n"
            "    seg_df,\n"
            "    min_phase_duration_s=30.0,\n"
            "    max_phases=8,\n"
            "    hsr_threshold_mph=peak_cfg.hsr_threshold_mph,\n"
            "    accel_threshold_ms2=peak_cfg.accel_threshold_ms2,\n"
            "    decel_threshold_ms2=peak_cfg.decel_threshold_ms2,\n"
            ")\n"
            "\n"
            "session_summary"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 2. Slide 1: Session Snapshot (Copy/Paste Text)"))
    cells.append(
        nbf.v4.new_code_cell(
            "slide1_text = build_slide_1_snapshot_text(\n"
            "    session_summary,\n"
            "    hsr_threshold_mph=peak_cfg.hsr_threshold_mph,\n"
            "    event_summary=event_counts,\n"
            ")\n"
            "print(slide1_text)\n"
            "write_slide_text(TEXT_DIR / 'slide_1_session_snapshot.txt', slide1_text)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 3. Definitions and Data QA"))
    cells.append(
        nbf.v4.new_code_cell(
            "definitions_text = (\n"
            "    'Definitions and Assumptions\\n'\n"
            "    '- Speed bands (mph): Walk 0-3, Cruise 3-9, Run 9-13, HSR 13-16, Sprint >=16.\\n'\n"
            "    f\"- HSR/Sprint thresholds: >= {peak_cfg.hsr_threshold_mph:.1f} mph / >= 16.0 mph.\\n\"\n"
            "    f\"- Accel/Decel thresholds: >= {peak_cfg.accel_threshold_ms2:.1f} / <= {peak_cfg.decel_threshold_ms2:.1f} m/s^2.\\n\"\n"
            "    '- Event definition: contiguous threshold exposure >= 1.0 s.\\n'\n"
            "    '- Relative zones are anchored to session max speed.\\n'\n"
            "    '- Rationale: thresholds align with common field-practice reporting and coach readability.'\n"
            ")\n"
            "print(definitions_text)\n"
            "write_slide_text(TEXT_DIR / 'slide_1_definitions_and_assumptions.txt', definitions_text)\n"
            "\n"
            "qa_table = pd.DataFrame([qa_summary])[\n"
            "    [\n"
            "        'sample_count', 'expected_cadence_s', 'pct_on_expected_cadence',\n"
            "        'max_gap_s', 'gap_count', 'gap_threshold_s',\n"
            "        'step_distance_outlier_threshold_yd', 'step_distance_outlier_count',\n"
            "        'step_distance_outlier_pct', 'gap_handling', 'outlier_handling',\n"
            "    ]\n"
            "].rename(columns={\n"
            "    'sample_count': 'Sample count',\n"
            "    'expected_cadence_s': 'Expected cadence (s)',\n"
            "    'pct_on_expected_cadence': '% at expected cadence',\n"
            "    'max_gap_s': 'Max gap (s)',\n"
            "    'gap_count': 'Gap count',\n"
            "    'gap_threshold_s': 'Gap threshold (s)',\n"
            "    'step_distance_outlier_threshold_yd': 'Outlier threshold (yd)',\n"
            "    'step_distance_outlier_count': 'Outlier count',\n"
            "    'step_distance_outlier_pct': 'Outlier count (%)',\n"
            "    'gap_handling': 'Gap handling',\n"
            "    'outlier_handling': 'Outlier handling',\n"
            "})\n"
            "for col, digits in [\n"
            "    ('Expected cadence (s)', 3),\n"
            "    ('% at expected cadence', 2),\n"
            "    ('Max gap (s)', 2),\n"
            "    ('Gap threshold (s)', 2),\n"
            "    ('Outlier threshold (yd)', 2),\n"
            "    ('Outlier count (%)', 2),\n"
            "]:\n"
            "    qa_table[col] = qa_table[col].round(digits)\n"
            "display(qa_table)\n"
            "qa_table.to_csv(TABLE_DIR / 'slide_1_data_quality_table.csv', index=False)\n"
            "\n"
            "qa_text = (\n"
            "    'Data QA Summary\\n'\n"
            "    f\"- {qa_table['% at expected cadence'].iloc[0]:.1f}% samples at 0.1s cadence.\\n\"\n"
            "    f\"- Max gap: {qa_table['Max gap (s)'].iloc[0]:.2f}s; gaps flagged above {qa_table['Gap threshold (s)'].iloc[0]:.2f}s.\\n\"\n"
            "    f\"- Outlier threshold: {qa_table['Outlier threshold (yd)'].iloc[0]:.2f} yd; flagged count {int(qa_table['Outlier count'].iloc[0])} ({qa_table['Outlier count (%)'].iloc[0]:.2f}%).\\n\"\n"
            "    '- Handling: rows retained; gaps/outliers are flagged for interpretation (no hidden exclusion).'\n"
            ")\n"
            "print(qa_text)\n"
            "write_slide_text(TEXT_DIR / 'slide_1_data_quality_takeaways.txt', qa_text)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 4. Slide 2: Workload by Speed Zone"))
    cells.append(
        nbf.v4.new_code_cell(
            "slide2_table = coach_speed_band_table(speed_band_summary)\n"
            "display(slide2_table)\n"
            "slide2_table.to_csv(TABLE_DIR / 'slide_2_speed_zone_table.csv', index=False)\n"
            "\n"
            "top_zone = slide2_table.sort_values('Distance (yd)', ascending=False).iloc[0]\n"
            "hsr_label = classify_hsr_exposure(\n"
            "    total_distance_yd=float(session_summary['distance_yd_from_speed']),\n"
            "    hsr_distance_yd=float(event_counts['hsr_distance_yd']),\n"
            ")\n"
            "slide2_text = (\n"
            "    'Speed Zone Takeaways\\n'\n"
            "    f\"- Largest distance accumulation: {top_zone['Zone']} ({top_zone['Distance (%)']:.1f}% of total distance).\\n\"\n"
            "    f\"- HSR exposure classification for this session: {hsr_label}.\\n\"\n"
            "    '- Action: if the objective was high-speed exposure, increase planned high-speed volume in key phases.'\n"
            ")\n"
            "print(slide2_text)\n"
            "write_slide_text(TEXT_DIR / 'slide_2_speed_zone_takeaways.txt', slide2_text)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 5. Slide 3: Peak Demands"))
    cells.append(
        nbf.v4.new_code_cell(
            "slide3_distance = coach_peak_distance_table(distance_table)\n"
            "slide3_extrema = coach_extrema_table(extrema)\n"
            "slide3_events = pd.DataFrame([\n"
            "    {\n"
            "        'HSR events (>=1s)': int(event_counts['hsr_event_count']),\n"
            "        'Sprint events (>=1s)': int(event_counts['sprint_event_count']),\n"
            "        'Accel events': int(event_counts['accel_event_count']),\n"
            "        'Decel events': int(event_counts['decel_event_count']),\n"
            "        'HSR distance (yd)': float(event_counts['hsr_distance_yd']),\n"
            "        'Sprint distance (yd)': float(event_counts['sprint_distance_yd']),\n"
            "    }\n"
            "])\n"
            "slide3_events['HSR distance (yd)'] = slide3_events['HSR distance (yd)'].round(1)\n"
            "slide3_events['Sprint distance (yd)'] = slide3_events['Sprint distance (yd)'].round(1)\n"
            "slide3_top_windows = top_windows[['window_start_utc', 'window_end_utc', 'value']].copy()\n"
            "slide3_top_windows = slide3_top_windows.rename(\n"
            "    columns={'window_start_utc': 'Start (UTC)', 'window_end_utc': 'End (UTC)', 'value': 'Distance in 60s (yd)'}\n"
            ")\n"
            "slide3_top_windows['Start (UTC)'] = pd.to_datetime(slide3_top_windows['Start (UTC)']).dt.strftime('%H:%M:%S')\n"
            "slide3_top_windows['End (UTC)'] = pd.to_datetime(slide3_top_windows['End (UTC)']).dt.strftime('%H:%M:%S')\n"
            "slide3_top_windows['Distance in 60s (yd)'] = slide3_top_windows['Distance in 60s (yd)'].round(1)\n"
            "\n"
            "display(slide3_distance)\n"
            "display(slide3_extrema)\n"
            "display(slide3_events)\n"
            "display(slide3_top_windows)\n"
            "\n"
            "slide3_distance.to_csv(TABLE_DIR / 'slide_3_peak_distance_table.csv', index=False)\n"
            "slide3_extrema.to_csv(TABLE_DIR / 'slide_3_extrema_table.csv', index=False)\n"
            "slide3_events.to_csv(TABLE_DIR / 'slide_3_event_counts_table.csv', index=False)\n"
            "slide3_top_windows.to_csv(TABLE_DIR / 'slide_3_top_windows_table.csv', index=False)\n"
            "\n"
            "if slide3_top_windows.empty:\n"
            "    slide3_text = (\n"
            "        'Peak Demand Takeaways\\n'\n"
            "        '- Not enough samples to derive a stable 1-minute peak-demand window.'\n"
            "    )\n"
            "else:\n"
            "    best_window = slide3_top_windows.iloc[0]\n"
            "    best_window_raw = top_windows.iloc[0]\n"
            "    best_window_context = summarize_window_context(\n"
            "        df,\n"
            "        window_start_utc=pd.to_datetime(best_window_raw['window_start_utc'], utc=True),\n"
            "        window_end_utc=pd.to_datetime(best_window_raw['window_end_utc'], utc=True),\n"
            "        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,\n"
            "        accel_threshold_ms2=peak_cfg.accel_threshold_ms2,\n"
            "        decel_threshold_ms2=peak_cfg.decel_threshold_ms2,\n"
            "    )\n"
            "    slide3_text = (\n"
            "        'Peak Demand Takeaways\\n'\n"
            "        f\"- Best 1-min demand: {best_window['Distance in 60s (yd)']:.1f} yd from {best_window['Start (UTC)']} to {best_window['End (UTC)']} UTC.\\n\"\n"
            "        f\"- Window context: HSR/Sprint events {int(best_window_context['hsr_event_count'])} / {int(best_window_context['sprint_event_count'])}; accel/decel {int(best_window_context['accel_event_count'])} / {int(best_window_context['decel_event_count'])}.\\n\"\n"
            "        '- Action: replicate this window\\'s work:rest pattern for conditioning, and monitor decel load tolerance.'\n"
            "    )\n"
            "print(slide3_text)\n"
            "write_slide_text(TEXT_DIR / 'slide_3_peak_takeaways.txt', slide3_text)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 6. Slide 4: Session Phases (Coach Context)"))
    cells.append(
        nbf.v4.new_code_cell(
            "slide4_table = coach_segment_table(coach_phase_summary, top_n=8)\n"
            "display(slide4_table)\n"
            "slide4_table.to_csv(TABLE_DIR / 'slide_4_segment_table.csv', index=False)\n"
            "\n"
            "top_phase = coach_phase_summary.sort_values('distance_yd', ascending=False).iloc[0]\n"
            "high_phase_count = int((coach_phase_summary['intensity_level'] == 'High').sum())\n"
            "slide4_text = (\n"
            "    'Session Phase Takeaways\\n'\n"
            "    f\"- Algorithmic blocks were merged into {len(coach_phase_summary)} coach-readable phases.\\n\"\n"
            "    f\"- Highest volume phase: {top_phase['coach_phase_label']} ({top_phase['distance_yd']:.1f} yd across {top_phase['duration_s'] / 60.0:.1f} min).\\n\"\n"
            "    f\"- High-intensity phases identified: {high_phase_count}.\\n\"\n"
            "    '- Action: use highlighted phases for drill debrief and keep low-intensity transitions explicit in planning.'\n"
            ")\n"
            "print(slide4_text)\n"
            "write_slide_text(TEXT_DIR / 'slide_4_segment_takeaways.txt', slide4_text)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 7. Slide 5: Early vs Late Session Context"))
    cells.append(
        nbf.v4.new_code_cell(
            "slide5_table = coach_early_late_table(early_late_summary)\n"
            "display(slide5_table)\n"
            "slide5_table.to_csv(TABLE_DIR / 'slide_5_early_late_table.csv', index=False)\n"
            "\n"
            "if len(slide5_table) == 2:\n"
            "    late = slide5_table.loc[slide5_table['Period'] == 'Late Half'].iloc[0]\n"
            "    slide5_text = (\n"
            "        'Early vs Late Takeaways\\n'\n"
            "        f\"- Late-half distance vs early-half: {late['Distance vs early (%)']:+.1f}%.\\n\"\n"
            "        f\"- Late-half HSR/Sprint events: {int(late['HSR events'])} / {int(late['Sprint events'])}.\\n\"\n"
            "        f\"- Late-half accel/decel events: {int(late['Accel events'])} / {int(late['Decel events'])}.\\n\"\n"
            "        '- Action: adjust second-half load progression if high-speed or decel demand deviates from intent.'\n"
            "    )\n"
            "else:\n"
            "    slide5_text = 'Early vs Late Takeaways\\n- Insufficient data to compute a stable split-half comparison.'\n"
            "print(slide5_text)\n"
            "write_slide_text(TEXT_DIR / 'slide_5_early_late_takeaways.txt', slide5_text)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 8. Slide Figure Exports (PNG)"))
    cells.append(
        nbf.v4.new_code_cell(
            "fig1, _ = plot_movement_map(coach_df, segment_col='coach_phase_label', highlight_top_n=3)\n"
            "fig2, _ = plot_intensity_timeline(\n"
            "    seg_df,\n"
            "    top_windows=top_windows,\n"
            "    hsr_threshold_mph=peak_cfg.hsr_threshold_mph,\n"
            ")\n"
            "fig3, _ = plot_peak_demand_summary(distance_table, extrema)\n"
            "\n"
            "save_figure(fig1, FIG_DIR / 'coach_slide_movement_map.png')\n"
            "save_figure(fig2, FIG_DIR / 'coach_slide_intensity_timeline.png')\n"
            "save_figure(fig3, FIG_DIR / 'coach_slide_peak_demand_summary.png')\n"
            "\n"
            "('Saved', FIG_DIR / 'coach_slide_movement_map.png', FIG_DIR / 'coach_slide_intensity_timeline.png', FIG_DIR / 'coach_slide_peak_demand_summary.png')"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 9. PowerPoint Copy Checklist\n"
            "- Text blocks: `outputs/slide_text/*.txt`\n"
            "- Slide tables: `outputs/tables/slide_*.csv`\n"
            "- Slide figures: `outputs/figures/coach_slide_*.png`\n"
            "- Include assumptions + QA artifacts from Slide 1 support files"
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
    output_path = Path("notebooks/02_coach_slide_ready_template.ipynb")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
