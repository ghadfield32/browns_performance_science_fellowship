"""Generate a presentation-first notebook scaffold that reads the results contract."""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nbf


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# Coach Slide-Ready Workbook\n\n"
            "This notebook is a consumer of Notebook 01 outputs.\n"
            "It reads `outputs/results.json` and exported tables/figures without recomputing analysis."
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 0. Setup\n"
            "- Run Notebook 01 first (or `scripts/render_visual_templates.py`).\n"
            "- This notebook only formats text/tables and aliases figures for deck insertion."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "import shutil\n"
            "\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, Markdown, display\n"
            "\n"
            "from browns_tracking.config import default_project_paths\n"
            "from browns_tracking.pipeline import classify_hsr_exposure\n"
            "from browns_tracking.presentation import (\n"
            "    build_slide_1_snapshot_text,\n"
            "    coach_early_late_table,\n"
            "    coach_extrema_table,\n"
            "    coach_peak_distance_table,\n"
            "    coach_segment_table,\n"
            "    coach_speed_band_table,\n"
            "    write_slide_text,\n"
            ")\n"
            "from browns_tracking.results_contract import load_results_contract\n"
            "\n"
            "pd.set_option('display.max_columns', 140)\n"
            "pd.set_option('display.width', 240)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "paths = default_project_paths()\n"
            "OUTPUT_DIR = paths.output_dir\n"
            "FIG_DIR = OUTPUT_DIR / 'figures'\n"
            "TABLE_DIR = OUTPUT_DIR / 'tables'\n"
            "TEXT_DIR = OUTPUT_DIR / 'slide_text'\n"
            "TABLE_DIR.mkdir(parents=True, exist_ok=True)\n"
            "TEXT_DIR.mkdir(parents=True, exist_ok=True)\n"
            "\n"
            "contract = load_results_contract(OUTPUT_DIR)\n"
            "session_summary = contract['session_summary']\n"
            "qa_summary = contract['qa_summary']\n"
            "qc_status = contract.get('qc_status', 'QC FAILED')\n"
            "overall_status = str(qa_summary.get('overall_status', 'UNKNOWN')).upper()\n"
            "if overall_status == 'UNKNOWN':\n"
            "    if qc_status == 'QC PASS':\n"
            "        overall_status = 'PASS'\n"
            "    elif qc_status == 'QC WARN':\n"
            "        overall_status = 'WARN'\n"
            "    else:\n"
            "        overall_status = 'FAIL'\n"
            "validation_takeaways = contract.get('validation_takeaways', [])\n"
            "thresholds = contract['thresholds']\n"
            "hsr_threshold_mph = float(thresholds['hsr_threshold_mph'])\n"
            "sprint_threshold_mph = float(\n"
            "    thresholds.get(\n"
            "        'sprint_threshold_mph',\n"
            "        next(\n"
            "            (\n"
            "                float(band.get('lower_mph', 16.0))\n"
            "                for band in thresholds.get('speed_bands_mph', [])\n"
            "                if str(band.get('name', '')).strip().lower() == 'sprint'\n"
            "            ),\n"
            "            16.0,\n"
            "        ),\n"
            "    )\n"
            ")\n"
            "accel_threshold_ms2 = float(thresholds['accel_threshold_ms2'])\n"
            "decel_threshold_ms2 = float(thresholds['decel_threshold_ms2'])\n"
            "speed_bands = thresholds['speed_bands_mph']\n"
            "units = contract.get('units', {})\n"
            "speed_conversion_factor = float(units.get('speed_conversion', {}).get('factor', 2.0454545))\n"
            "accel_conversion_factor = float(units.get('acceleration_conversion', {}).get('factor', 0.9144))\n"
            "\n"
            "speed_band_summary = pd.read_csv(TABLE_DIR / 'absolute_speed_band_summary.csv')\n"
            "distance_table = pd.read_csv(TABLE_DIR / 'peak_distance_windows.csv')\n"
            "extrema = pd.read_csv(TABLE_DIR / 'session_extrema.csv')\n"
            "event_counts = pd.read_csv(TABLE_DIR / 'session_event_counts.csv').iloc[0].to_dict()\n"
            "early_late_summary = pd.read_csv(TABLE_DIR / 'early_vs_late_summary.csv')\n"
            "coach_phase_summary = pd.read_csv(TABLE_DIR / 'coach_phase_summary.csv')\n"
            "session_structure_map = pd.read_csv(TABLE_DIR / 'session_structure_map.csv')\n"
            "validation_gates = pd.read_csv(TABLE_DIR / 'validation_gates.csv')\n"
            "top_windows_by_duration = pd.read_csv(TABLE_DIR / 'top_windows_by_duration.csv')\n"
            "\n"
            "required_figures = [\n"
            "    FIG_DIR / '01_space.png',\n"
            "    FIG_DIR / '02_time.png',\n"
            "    FIG_DIR / '03_peaks.png',\n"
            "]\n"
            "missing_figures = [str(p) for p in required_figures if not p.exists()]\n"
            "if missing_figures:\n"
            "    raise FileNotFoundError(\n"
            "        'Missing canonical figures. Run Notebook 01 or scripts/render_visual_templates.py first:\\n'\n"
            "        + '\\n'.join(missing_figures)\n"
            "    )\n"
            "\n"
            "pd.Series({\n"
            "    'Contract': str(OUTPUT_DIR / 'results.json'),\n"
            "    'Model': contract['model']['name'],\n"
            "    'Questions': '; '.join(contract['story_questions']),\n"
            "}, name='value').to_frame()"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "def show_saved_png(path, caption='', width=1250):\n"
            "    png = Path(path)\n"
            "    if caption:\n"
            "        display(Markdown(f'**{caption}**'))\n"
            "    if not png.exists():\n"
            "        display(Markdown(f':warning: Missing `{png}`.'))\n"
            "        return\n"
            "    display(Image(filename=str(png), width=width))"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Deck Map (Use This 6-Slide Flow)"))
    cells.append(
        nbf.v4.new_code_cell(
            "if overall_status == 'PASS':\n"
            "    slide1_takeaway = 'Data quality gates passed at a usable level, so workload outputs are decision-ready.'\n"
            "elif overall_status == 'WARN':\n"
            "    slide1_takeaway = 'Data quality is usable with caveats; continuity masking and block-limited windows were applied.'\n"
            "else:\n"
            "    slide1_takeaway = 'Data quality failed trust gates; use this session only for descriptive review, not workload decisions.'\n"
            "\n"
            "deck_outline = pd.DataFrame([\n"
            "    {\n"
            "        'slide': 1,\n"
            "        'title': 'Session Snapshot + Validation Gates',\n"
            "        'figure_or_table': (\n"
            "            'outputs/tables/slide_1_data_quality_table.csv; '\n"
            "            'outputs/tables/slide_1_validation_gates_table.csv; '\n"
            "            'outputs/tables/slide_1_qc_checks_table.csv'\n"
            "        ),\n"
            "        'one_line_takeaway': slide1_takeaway,\n"
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
            "deck_outline.to_csv(TABLE_DIR / 'final_deck_outline.csv', index=False)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Coach/Player Visual Report (Auto-Display)"))
    cells.append(
        nbf.v4.new_code_cell(
            "show_saved_png(\n"
            "    FIG_DIR / '01_space.png',\n"
            "    'Slide 2 | Where: Spatial usage and role signature (density + key demand windows)'\n"
            ")\n"
            "show_saved_png(\n"
            "    FIG_DIR / '02_time.png',\n"
            "    'Slide 3 | When: Intensity over time with phase strip, HSR events, and peak windows'\n"
            ")\n"
            "show_saved_png(\n"
            "    FIG_DIR / '03_peaks.png',\n"
            "    'Slide 4 | What: Peak demand windows and session maxima with event context'\n"
            ")"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## Coach/Player Summary Block"))
    cells.append(
        nbf.v4.new_code_cell(
            "best_window = None\n"
            "if not top_windows_by_duration.empty:\n"
            "    best_candidates = top_windows_by_duration[top_windows_by_duration['window_s'] == 60]\n"
            "    if best_candidates.empty:\n"
            "        best_candidates = top_windows_by_duration\n"
            "    best_window = best_candidates.copy()\n"
            "    best_window['peak_intensity_yd_min'] = (\n"
            "        pd.to_numeric(best_window['distance_yd'], errors='coerce')\n"
            "        * (60.0 / pd.to_numeric(best_window['window_s'], errors='coerce').replace(0.0, pd.NA))\n"
            "    ).fillna(0.0)\n"
            "    best_window = best_window.sort_values('peak_intensity_yd_min', ascending=False).iloc[0]\n"
            "coach_lines = [\n"
            "    'Coach Lens',\n"
            "    f\"- Session phases: {len(coach_phase_summary)} merged blocks with clear intensity labels.\",\n"
            "    f\"- QA status: {overall_status} ({qc_status}).\",\n"
            "    f\"- Validation gates passed: {int((validation_gates['status'] == 'PASS').sum())}/{len(validation_gates)}.\",\n"
            "]\n"
            "player_lines = [\n"
            "    'Player Lens',\n"
            "    f\"- Peak speed: {float(session_summary['peak_speed_mph']):.2f} mph.\",\n"
            "    f\"- HSR distance: {float(event_counts.get('hsr_distance_yd', 0.0)):.1f} yd.\",\n"
            "]\n"
            "if best_window is not None:\n"
            "    start = pd.Timestamp(best_window['window_start_utc']).strftime('%H:%M:%S')\n"
            "    end = pd.Timestamp(best_window['window_end_utc']).strftime('%H:%M:%S')\n"
            "    duration_label = f\"{int(best_window['window_s'] // 60)}-min\" if int(best_window['window_s']) % 60 == 0 else f\"{int(best_window['window_s'])}s\"\n"
            "    coach_lines.append(\n"
            "        f\"- Key drill anchor: best {duration_label} intensity {float(best_window['peak_intensity_yd_min']):.1f} yd/min ({start}-{end} UTC).\"\n"
            "    )\n"
            "    player_lines.append(\n"
            "        f\"- Best window load: {float(best_window['peak_intensity_yd_min']):.1f} yd/min ({float(best_window['distance_yd']):.1f} yd total).\"\n"
            "    )\n"
            "summary_md = '### ' + '\\n'.join(coach_lines) + '\\n\\n### ' + '\\n'.join(player_lines)\n"
            "display(Markdown(summary_md))"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 1. Slide 1: Snapshot + Validation Gates"))
    cells.append(
        nbf.v4.new_code_cell(
            "slide1_text = build_slide_1_snapshot_text(\n"
            "    session_summary,\n"
            "    hsr_threshold_mph=hsr_threshold_mph,\n"
            "    event_summary=event_counts,\n"
            ")\n"
            "print(slide1_text)\n"
            "write_slide_text(TEXT_DIR / 'slide_1_session_snapshot.txt', slide1_text)\n"
            "\n"
            "speed_band_text = ', '.join(\n"
            "    [\n"
            "        (\n"
            "            f\"{band['name']} {band['lower_mph']:.0f}-{band['upper_mph']:.0f}\"\n"
            "            if band['upper_mph'] is not None\n"
            "            else f\"{band['name']} >= {band['lower_mph']:.0f}\"\n"
            "        )\n"
            "        for band in speed_bands\n"
            "    ]\n"
            ")\n"
            "definitions_text = (\n"
            "    'Definitions and Assumptions\\n'\n"
            "    f\"- Speed bands (mph): {speed_band_text}.\\n\"\n"
            "    f\"- HSR/Sprint thresholds: >= {hsr_threshold_mph:.1f} mph / >= {sprint_threshold_mph:.1f} mph.\\n\"\n"
            "    f\"- Accel/Decel thresholds: >= {accel_threshold_ms2:.1f} / <= {decel_threshold_ms2:.1f} m/s^2.\\n\"\n"
            "    '- Acceleration signal policy: use signed `sa` channel for accel/decel events and extrema.\\n'\n"
            "    '- Event definition: contiguous threshold exposure >= 1.0 s.\\n'\n"
            "    f\"- Unit conversions: speed yd/s -> mph x {speed_conversion_factor:.7f}; \"\n"
            "    f\"accel yd/s^2 -> m/s^2 x {accel_conversion_factor:.4f}.\\n\"\n"
            "    '- Distance policy: workload totals use speed-derived step distance (`s*dt`); vendor `dis` is QA-only.\\n'\n"
            "    '- Continuity policy: windows/events and trajectory lines reset at flagged gaps/improbable jumps.'\n"
            ")\n"
            "print(definitions_text)\n"
            "write_slide_text(TEXT_DIR / 'slide_1_definitions_and_assumptions.txt', definitions_text)\n"
            "\n"
            "qa_cols = [\n"
            "    'overall_status',\n"
            "    'sample_count', 'expected_cadence_s', 'pct_on_expected_cadence',\n"
            "    'max_gap_s', 'gap_count', 'gap_pct', 'gap_threshold_s',\n"
            "    'max_xy_step_yd', 'teleport_count', 'teleport_pct', 'teleport_speed_threshold_yd_s',\n"
            "    'inactive_speed_threshold_mph', 'inactive_sample_count', 'inactive_sample_pct',\n"
            "    'position_outlier_low_quantile', 'position_outlier_high_quantile',\n"
            "    'position_outlier_count', 'position_outlier_pct',\n"
            "    'speed_xy_alignment_samples', 'speed_xy_corr', 'speed_xy_error_p95_yd_s',\n"
            "    'speed_xy_outlier_pct', 'speed_xy_outlier_threshold_yd_s',\n"
            "    'step_distance_outlier_source',\n"
            "    'step_distance_outlier_threshold_yd', 'step_distance_outlier_count',\n"
            "    'step_distance_outlier_pct',\n"
            "]\n"
            "qa_cols = [col for col in qa_cols if col in qa_summary]\n"
            "qa_table = pd.DataFrame([qa_summary])[qa_cols].rename(columns={\n"
            "    'overall_status': 'Overall status',\n"
            "    'sample_count': 'Sample count',\n"
            "    'expected_cadence_s': 'Expected cadence (s)',\n"
            "    'pct_on_expected_cadence': '% at expected cadence',\n"
            "    'max_gap_s': 'Max gap (s)',\n"
            "    'gap_count': 'Gap count',\n"
            "    'gap_pct': 'Gap count (%)',\n"
            "    'gap_threshold_s': 'Gap threshold (s)',\n"
            "    'max_xy_step_yd': 'Max XY step (yd)',\n"
            "    'teleport_count': 'Teleport count',\n"
            "    'teleport_pct': 'Teleport count (%)',\n"
            "    'teleport_speed_threshold_yd_s': 'Teleport threshold (yd/s)',\n"
            "    'inactive_speed_threshold_mph': 'Inactive threshold (mph)',\n"
            "    'inactive_sample_count': 'Inactive sample count',\n"
            "    'inactive_sample_pct': 'Inactive sample (%)',\n"
            "    'position_outlier_low_quantile': 'Position outlier q_low',\n"
            "    'position_outlier_high_quantile': 'Position outlier q_high',\n"
            "    'position_outlier_count': 'Position outlier count',\n"
            "    'position_outlier_pct': 'Position outlier (%)',\n"
            "    'speed_xy_alignment_samples': 'Speed/XY alignment samples',\n"
            "    'speed_xy_corr': 'Speed vs XY corr (r)',\n"
            "    'speed_xy_error_p95_yd_s': 'Speed vs XY error p95 (yd/s)',\n"
            "    'speed_xy_outlier_pct': 'Speed vs XY outlier (%)',\n"
            "    'speed_xy_outlier_threshold_yd_s': 'Speed vs XY outlier threshold (yd/s)',\n"
            "    'step_distance_outlier_source': 'Outlier source',\n"
            "    'step_distance_outlier_threshold_yd': 'Outlier threshold (yd)',\n"
            "    'step_distance_outlier_count': 'Outlier count',\n"
            "    'step_distance_outlier_pct': 'Outlier count (%)',\n"
            "})\n"
            "for col, digits in [\n"
            "    ('Expected cadence (s)', 3),\n"
            "    ('% at expected cadence', 2),\n"
            "    ('Max gap (s)', 2),\n"
            "    ('Gap threshold (s)', 2),\n"
            "    ('Gap count (%)', 3),\n"
            "    ('Max XY step (yd)', 2),\n"
            "    ('Teleport count (%)', 3),\n"
            "    ('Teleport threshold (yd/s)', 2),\n"
            "    ('Inactive threshold (mph)', 2),\n"
            "    ('Inactive sample (%)', 2),\n"
            "    ('Position outlier q_low', 2),\n"
            "    ('Position outlier q_high', 2),\n"
            "    ('Position outlier (%)', 2),\n"
            "    ('Speed vs XY corr (r)', 3),\n"
            "    ('Speed vs XY error p95 (yd/s)', 2),\n"
            "    ('Speed vs XY outlier (%)', 2),\n"
            "    ('Speed vs XY outlier threshold (yd/s)', 2),\n"
            "    ('Outlier threshold (yd)', 2),\n"
            "    ('Outlier count (%)', 2),\n"
            "]:\n"
            "    if col in qa_table.columns:\n"
            "        qa_table[col] = qa_table[col].round(digits)\n"
            "display(qa_table)\n"
            "qa_table.to_csv(TABLE_DIR / 'slide_1_data_quality_table.csv', index=False)\n"
            "\n"
            "validation_table = validation_gates[['gate', 'status', 'value', 'threshold', 'direction', 'unit', 'notes']].copy()\n"
            "validation_table.columns = ['Gate', 'Status', 'Value', 'Threshold', 'Direction', 'Unit', 'Notes']\n"
            "validation_table['Value'] = pd.to_numeric(validation_table['Value'], errors='coerce').round(3)\n"
            "display(validation_table)\n"
            "validation_table.to_csv(TABLE_DIR / 'slide_1_validation_gates_table.csv', index=False)\n"
            "\n"
            "pass_count = int((validation_gates['status'] == 'PASS').sum()) if not validation_gates.empty else 0\n"
            "qa_text_lines = [\n"
            "    'Data QA Summary',\n"
            "    f\"- QC: {qc_status}; Overall status: {overall_status}.\",\n"
            "    f\"- {qa_table['% at expected cadence'].iloc[0]:.1f}% samples at 0.1s cadence.\",\n"
            "    f\"- Max gap: {qa_table['Max gap (s)'].iloc[0]:.2f}s; gaps flagged above {qa_table['Gap threshold (s)'].iloc[0]:.2f}s.\",\n"
            "    (\n"
            "        f\"- Inactive samples (<= {qa_table['Inactive threshold (mph)'].iloc[0]:.1f} mph): {qa_table['Inactive sample (%)'].iloc[0]:.1f}%.\"\n"
            "        if 'Inactive threshold (mph)' in qa_table.columns and 'Inactive sample (%)' in qa_table.columns\n"
            "        else '- Inactive sample % unavailable.'\n"
            "    ),\n"
            "    (\n"
            "        f\"- Position outliers ({qa_table['Position outlier q_low'].iloc[0]:.2f}-{qa_table['Position outlier q_high'].iloc[0]:.2f} quantiles): {qa_table['Position outlier (%)'].iloc[0]:.2f}%.\"\n"
            "        if 'Position outlier (%)' in qa_table.columns\n"
            "        else '- Position outlier % unavailable.'\n"
            "    ),\n"
            "    (\n"
            "        f\"- Speed-vs-XY sanity: corr {qa_table['Speed vs XY corr (r)'].iloc[0]:.3f}, outliers {qa_table['Speed vs XY outlier (%)'].iloc[0]:.2f}%.\"\n"
            "        if 'Speed vs XY corr (r)' in qa_table.columns and 'Speed vs XY outlier (%)' in qa_table.columns\n"
            "        else '- Speed-vs-XY sanity metrics unavailable.'\n"
            "    ),\n"
            "    f\"- Outlier threshold: {qa_table['Outlier threshold (yd)'].iloc[0]:.2f} yd; flagged {int(qa_table['Outlier count'].iloc[0])} samples ({qa_table['Outlier count (%)'].iloc[0]:.2f}%).\",\n"
            "    f\"- Validation gates passed: {pass_count}/{len(validation_gates)}.\",\n"
            "]\n"
            "if overall_status == 'WARN':\n"
            "    qa_text_lines.append('- Interpretation policy: usable with caveats; windows/events/maps are continuity-masked.')\n"
            "if overall_status == 'FAIL':\n"
            "    qa_text_lines.append('- Interpretation policy: do not use this session for workload decision thresholds.')\n"
            "qa_text_lines.extend([f\"- {line}\" for line in validation_takeaways[:3]])\n"
            "qa_text = '\\n'.join(qa_text_lines)\n"
            "print(qa_text)\n"
            "write_slide_text(TEXT_DIR / 'slide_1_data_quality_takeaways.txt', qa_text)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 2. Slide 2: Speed Zones"))
    cells.append(
        nbf.v4.new_code_cell(
            "slide2_table = coach_speed_band_table(speed_band_summary)\n"
            "display(slide2_table)\n"
            "slide2_table.to_csv(TABLE_DIR / 'slide_2_speed_zone_table.csv', index=False)\n"
            "\n"
            "top_zone = slide2_table.sort_values('Distance (yd)', ascending=False).iloc[0]\n"
            "hsr_label = classify_hsr_exposure(\n"
            "    total_distance_yd=float(session_summary['distance_yd_from_speed']),\n"
            "    hsr_distance_yd=float(event_counts.get('hsr_distance_yd', 0.0)),\n"
            ")\n"
            "slide2_text = (\n"
            "    'Speed Zone Takeaways\\n'\n"
            "    f\"- Largest distance accumulation: {top_zone['Zone']} ({top_zone['Distance (%)']:.1f}% of total distance).\\n\"\n"
            "    f\"- HSR exposure classification: {hsr_label}.\\n\"\n"
            "    '- Action: adjust high-speed volume by phase, not by session average only.'\n"
            ")\n"
            "print(slide2_text)\n"
            "write_slide_text(TEXT_DIR / 'slide_2_speed_zone_takeaways.txt', slide2_text)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 3. Slide 3: Peak Demands"))
    cells.append(
        nbf.v4.new_code_cell(
            "slide3_distance = coach_peak_distance_table(distance_table)\n"
            "slide3_extrema = coach_extrema_table(extrema)\n"
            "slide3_events = pd.DataFrame([\n"
            "    {\n"
            "        'HSR events (>=1s)': int(event_counts.get('hsr_event_count', 0)),\n"
            "        'Sprint events (>=1s)': int(event_counts.get('sprint_event_count', 0)),\n"
            "        'Accel events': int(event_counts.get('accel_event_count', 0)),\n"
            "        'Decel events': int(event_counts.get('decel_event_count', 0)),\n"
            "        'HSR distance (yd)': float(event_counts.get('hsr_distance_yd', 0.0)),\n"
            "        'Sprint distance (yd)': float(event_counts.get('sprint_distance_yd', 0.0)),\n"
            "    }\n"
            "])\n"
            "slide3_events['HSR distance (yd)'] = slide3_events['HSR distance (yd)'].round(1)\n"
            "slide3_events['Sprint distance (yd)'] = slide3_events['Sprint distance (yd)'].round(1)\n"
            "\n"
            "slide3_top_windows = top_windows_by_duration[\n"
            "    ['window_s', 'window_rank', 'window_start_utc', 'window_end_utc', 'distance_yd', 'dominant_phase', 'hsr_event_count', 'accel_event_count', 'decel_event_count']\n"
            "].copy()\n"
            "slide3_top_windows = slide3_top_windows.sort_values(['window_s', 'window_rank']).reset_index(drop=True)\n"
            "if not slide3_top_windows.empty:\n"
            "    slide3_top_windows['peak_intensity_yd_min'] = (\n"
            "        pd.to_numeric(slide3_top_windows['distance_yd'], errors='coerce')\n"
            "        * (60.0 / pd.to_numeric(slide3_top_windows['window_s'], errors='coerce').replace(0.0, pd.NA))\n"
            "    ).fillna(0.0)\n"
            "slide3_top_windows['Window duration'] = slide3_top_windows['window_s'].map(\n"
            "    lambda s: f\"{int(s // 60)}m\" if int(s) % 60 == 0 else f\"{int(s)}s\"\n"
            ")\n"
            "slide3_top_windows = slide3_top_windows.rename(columns={\n"
            "    'window_rank': 'Window',\n"
            "    'window_start_utc': 'Start (UTC)',\n"
            "    'window_end_utc': 'End (UTC)',\n"
            "    'peak_intensity_yd_min': 'Peak intensity (yd/min)',\n"
            "    'distance_yd': 'Distance in window (yd)',\n"
            "    'dominant_phase': 'Dominant phase',\n"
            "    'hsr_event_count': 'HSR events',\n"
            "    'accel_event_count': 'Accel events',\n"
            "    'decel_event_count': 'Decel events',\n"
            "})\n"
            "slide3_top_windows = slide3_top_windows[\n"
            "    ['Window duration', 'Window', 'Start (UTC)', 'End (UTC)', 'Peak intensity (yd/min)', 'Distance in window (yd)', 'Dominant phase', 'HSR events', 'Accel events', 'Decel events']\n"
            "]\n"
            "slide3_top_windows['Start (UTC)'] = pd.to_datetime(\n"
            "    slide3_top_windows['Start (UTC)'],\n"
            "    utc=True,\n"
            "    format='mixed',\n"
            ").dt.strftime('%H:%M:%S')\n"
            "slide3_top_windows['End (UTC)'] = pd.to_datetime(\n"
            "    slide3_top_windows['End (UTC)'],\n"
            "    utc=True,\n"
            "    format='mixed',\n"
            ").dt.strftime('%H:%M:%S')\n"
            "slide3_top_windows['Peak intensity (yd/min)'] = slide3_top_windows['Peak intensity (yd/min)'].round(1)\n"
            "slide3_top_windows['Distance in window (yd)'] = slide3_top_windows['Distance in window (yd)'].round(1)\n"
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
            "    slide3_text = 'Peak Demand Takeaways\\n- Not enough samples to derive stable top-demand windows.'\n"
            "else:\n"
            "    best_candidates = top_windows_by_duration[top_windows_by_duration['window_s'] == 60]\n"
            "    if best_candidates.empty:\n"
            "        best_candidates = top_windows_by_duration\n"
            "    best = best_candidates.copy()\n"
            "    best['peak_intensity_yd_min'] = (\n"
            "        pd.to_numeric(best['distance_yd'], errors='coerce')\n"
            "        * (60.0 / pd.to_numeric(best['window_s'], errors='coerce').replace(0.0, pd.NA))\n"
            "    ).fillna(0.0)\n"
            "    best = best.sort_values('peak_intensity_yd_min', ascending=False).iloc[0]\n"
            "    best_start = pd.Timestamp(best['window_start_utc']).strftime('%H:%M:%S')\n"
            "    best_end = pd.Timestamp(best['window_end_utc']).strftime('%H:%M:%S')\n"
            "    duration_label = f\"{int(best['window_s'] // 60)}-min\" if int(best['window_s']) % 60 == 0 else f\"{int(best['window_s'])}s\"\n"
            "    slide3_text = (\n"
            "        'Peak Demand Takeaways\\n'\n"
            "        f\"- Best {duration_label} intensity: {float(best['peak_intensity_yd_min']):.1f} yd/min ({float(best['distance_yd']):.1f} yd) from {best_start} to {best_end} UTC.\\n\"\n"
            "        f\"- Context: HSR/Sprint {int(best['hsr_event_count'])}/{int(best['sprint_event_count'])}; Acc/Dec {int(best['accel_event_count'])}/{int(best['decel_event_count'])}.\\n\"\n"
            "        f\"- Dominant phase: {best['dominant_phase'] or 'N/A'}.\\n\"\n"
            "        '- Action: use this window to set drill-level peak-demand targets.'\n"
            "    )\n"
            "print(slide3_text)\n"
            "write_slide_text(TEXT_DIR / 'slide_3_peak_takeaways.txt', slide3_text)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 4. Slide 4: Session Phases"))
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
            "    f\"- Session merged into {len(coach_phase_summary)} coach-readable phases.\\n\"\n"
            "    f\"- Highest volume phase: {top_phase['coach_phase_label']} ({top_phase['distance_yd']:.1f} yd across {top_phase['duration_s'] / 60.0:.1f} min).\\n\"\n"
            "    f\"- High-intensity phases identified: {high_phase_count}.\\n\"\n"
            "    '- Action: use phase boundaries as planning blocks for next session design.'\n"
            ")\n"
            "print(slide4_text)\n"
            "write_slide_text(TEXT_DIR / 'slide_4_segment_takeaways.txt', slide4_text)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 5. Slide 5: Early vs Late"))
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
            "        f\"- Late-half HSR/Sprint events: {int(late['HSR events'])}/{int(late['Sprint events'])}.\\n\"\n"
            "        f\"- Late-half accel/decel events: {int(late['Accel events'])}/{int(late['Decel events'])}.\\n\"\n"
            "        '- Action: adjust second-half progression if high-speed or braking load drifts from intent.'\n"
            "    )\n"
            "else:\n"
            "    slide5_text = 'Early vs Late Takeaways\\n- Insufficient data for split-half comparison.'\n"
            "print(slide5_text)\n"
            "write_slide_text(TEXT_DIR / 'slide_5_early_late_takeaways.txt', slide5_text)"
        )
    )

    cells.append(nbf.v4.new_markdown_cell("## 6. Figure Aliases for Slide Deck"))
    cells.append(
        nbf.v4.new_code_cell(
            "figure_aliases = {\n"
            "    '01_space.png': 'coach_slide_movement_map.png',\n"
            "    '02_time.png': 'coach_slide_intensity_timeline.png',\n"
            "    '03_peaks.png': 'coach_slide_peak_demand_summary.png',\n"
            "}\n"
            "for src, dst in figure_aliases.items():\n"
            "    source = FIG_DIR / src\n"
            "    if not source.exists():\n"
            "        raise FileNotFoundError(f'Missing {source}. Run Notebook 01 first.')\n"
            "    shutil.copyfile(source, FIG_DIR / dst)\n"
            "\n"
            "('Saved aliases', FIG_DIR / 'coach_slide_movement_map.png', FIG_DIR / 'coach_slide_intensity_timeline.png', FIG_DIR / 'coach_slide_peak_demand_summary.png')"
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
    output_path = Path("notebooks/02_coach_slide_ready_template.ipynb")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
