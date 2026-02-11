"""Export presentation-first assets from the existing results contract."""

from __future__ import annotations

import argparse
import shutil

import numpy as np
import pandas as pd

from browns_tracking.config import resolve_output_dir
from browns_tracking.pipeline import classify_hsr_exposure
from browns_tracking.presentation import (
    build_slide_1_snapshot_text,
    coach_early_late_table,
    coach_extrema_table,
    coach_peak_distance_table,
    coach_segment_table,
    coach_speed_band_table,
    write_slide_text,
)
from browns_tracking.results_contract import load_results_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export coach-slide text/tables/figure aliases from outputs/results.json and "
            "existing contract tables."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory with results contract artifacts (default: auto-resolve from project config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    text_dir = output_dir / "slide_text"
    table_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    contract = load_results_contract(output_dir)
    session_summary = contract["session_summary"]
    qa_summary = contract["qa_summary"]
    qc_status = contract.get("qc_status", "QC FAILED")
    overall_status = str(qa_summary.get("overall_status", "UNKNOWN")).upper()
    if overall_status == "UNKNOWN":
        if qc_status == "QC PASS":
            overall_status = "PASS"
        elif qc_status == "QC WARN":
            overall_status = "WARN"
        else:
            overall_status = "FAIL"
    validation_takeaways = contract.get("validation_takeaways", [])
    thresholds = contract["thresholds"]
    hsr_threshold_mph = float(thresholds["hsr_threshold_mph"])
    sprint_threshold_mph = float(
        thresholds.get(
            "sprint_threshold_mph",
            next(
                (
                    float(band.get("lower_mph", 16.0))
                    for band in thresholds.get("speed_bands_mph", [])
                    if str(band.get("name", "")).strip().lower() == "sprint"
                ),
                16.0,
            ),
        )
    )
    accel_threshold_ms2 = float(thresholds["accel_threshold_ms2"])
    decel_threshold_ms2 = float(thresholds["decel_threshold_ms2"])
    speed_bands = thresholds["speed_bands_mph"]
    units = contract.get("units", {})
    speed_conversion_factor = float(units.get("speed_conversion", {}).get("factor", 2.0454545))
    accel_conversion_factor = float(units.get("acceleration_conversion", {}).get("factor", 0.9144))

    speed_band_summary = pd.read_csv(table_dir / "absolute_speed_band_summary.csv")
    distance_table = pd.read_csv(table_dir / "peak_distance_windows.csv")
    extrema = pd.read_csv(table_dir / "session_extrema.csv")
    event_counts_df = pd.read_csv(table_dir / "session_event_counts.csv")
    early_late_summary = pd.read_csv(table_dir / "early_vs_late_summary.csv")
    coach_phase_summary = pd.read_csv(table_dir / "coach_phase_summary.csv")
    session_structure_map = pd.read_csv(table_dir / "session_structure_map.csv")
    validation_gates = pd.read_csv(table_dir / "validation_gates.csv")
    qc_checks = pd.read_csv(table_dir / "qc_checks.csv")
    top_windows_by_duration = pd.read_csv(table_dir / "top_windows_by_duration.csv")

    event_counts = event_counts_df.iloc[0].to_dict() if not event_counts_df.empty else {}
    if overall_status == "PASS":
        slide1_takeaway = "Data quality gates passed at a usable level, so workload outputs are decision-ready."
    elif overall_status == "WARN":
        slide1_takeaway = (
            "Data quality is usable with caveats; continuity masking and block-limited windows were applied."
        )
    else:
        slide1_takeaway = (
            "Data quality failed trust gates; use this session only for descriptive review, not workload decisions."
        )

    deck_outline = pd.DataFrame(
        [
            {
                "slide": 1,
                "title": "Session Snapshot + Validation Gates",
                "figure_or_table": (
                    "outputs/tables/slide_1_data_quality_table.csv; "
                    "outputs/tables/slide_1_validation_gates_table.csv; "
                    "outputs/tables/slide_1_qc_checks_table.csv"
                ),
                "one_line_takeaway": slide1_takeaway,
            },
            {
                "slide": 2,
                "title": "Where: Spatial Usage and Role Signature",
                "figure_or_table": (
                    "outputs/figures/01_space.png; outputs/phase_table.csv; "
                    "outputs/session_structure_map.csv"
                ),
                "one_line_takeaway": (
                    "The session lived in specific field zones and repeated role-consistent movement patterns."
                ),
            },
            {
                "slide": 3,
                "title": "When: Intensity Timeline and Session Structure",
                "figure_or_table": (
                    "outputs/figures/02_time.png; outputs/tables/slide_3_top_windows_table.csv"
                ),
                "one_line_takeaway": (
                    "High-intensity work clustered in identifiable blocks rather than being evenly distributed."
                ),
            },
            {
                "slide": 4,
                "title": "What: Peak Demands and Repeatable Windows",
                "figure_or_table": (
                    "outputs/figures/03_peaks.png; outputs/tables/slide_3_peak_distance_table.csv; "
                    "outputs/tables/slide_3_event_counts_table.csv"
                ),
                "one_line_takeaway": (
                    "Top windows define the true worst-case demands to anchor drill and conditioning targets."
                ),
            },
            {
                "slide": 5,
                "title": "Phase-Level Load Profile",
                "figure_or_table": "outputs/tables/slide_4_segment_table.csv",
                "one_line_takeaway": (
                    "Merged coach phases show which blocks carried most distance and high-speed stress."
                ),
            },
            {
                "slide": 6,
                "title": "Early vs Late Readiness Signal",
                "figure_or_table": "outputs/tables/slide_5_early_late_table.csv",
                "one_line_takeaway": (
                    "Late-session drift quantifies whether second-half output matched training intent."
                ),
            },
        ]
    )
    deck_outline.to_csv(table_dir / "final_deck_outline.csv", index=False)

    slide1_text = build_slide_1_snapshot_text(
        session_summary,
        hsr_threshold_mph=hsr_threshold_mph,
        event_summary=event_counts,
    )
    write_slide_text(text_dir / "slide_1_session_snapshot.txt", slide1_text)

    speed_band_text = ", ".join(
        [
            f"{band['name']} {band['lower_mph']:.0f}-"
            f"{band['upper_mph']:.0f}" if band["upper_mph"] is not None else f"{band['name']} >= {band['lower_mph']:.0f}"
            for band in speed_bands
        ]
    )
    definitions_text = (
        "Definitions and Assumptions\n"
        f"- Speed bands (mph): {speed_band_text}.\n"
        f"- HSR/Sprint thresholds: >= {hsr_threshold_mph:.1f} mph / >= {sprint_threshold_mph:.1f} mph.\n"
        f"- Accel/Decel thresholds: >= {accel_threshold_ms2:.1f} / <= {decel_threshold_ms2:.1f} m/s^2.\n"
        "- Acceleration signal policy: use signed `sa` channel for accel/decel events and extrema.\n"
        "- Event definition: contiguous threshold exposure >= 1.0 s.\n"
        f"- Unit conversions: speed yd/s -> mph x {speed_conversion_factor:.7f}; "
        f"accel yd/s^2 -> m/s^2 x {accel_conversion_factor:.4f}.\n"
        "- Distance policy: workload totals use speed-derived step distance (`s*dt`); vendor `dis` is QA-only.\n"
        "- Continuity policy: windows/events and trajectory lines reset at flagged gaps/improbable jumps."
    )
    write_slide_text(text_dir / "slide_1_definitions_and_assumptions.txt", definitions_text)

    qa_cols = [
        "overall_status",
        "sample_count",
        "expected_cadence_s",
        "pct_on_expected_cadence",
        "cadence_p95_s",
        "cadence_p99_s",
        "max_gap_s",
        "gap_count",
        "gap_pct",
        "gap_threshold_s",
        "max_xy_step_yd",
        "teleport_count",
        "teleport_pct",
        "teleport_speed_threshold_yd_s",
        "inactive_speed_threshold_mph",
        "inactive_sample_count",
        "inactive_sample_pct",
        "position_outlier_low_quantile",
        "position_outlier_high_quantile",
        "position_outlier_count",
        "position_outlier_pct",
        "speed_xy_alignment_samples",
        "speed_xy_corr",
        "speed_xy_error_p95_yd_s",
        "speed_xy_outlier_pct",
        "speed_xy_outlier_threshold_yd_s",
        "step_distance_outlier_source",
        "step_distance_outlier_threshold_yd",
        "step_distance_outlier_count",
        "step_distance_outlier_pct",
        "distance_yd_from_speed",
        "distance_yd_from_dis",
        "distance_dis_minus_speed_pct",
    ]
    qa_cols = [col for col in qa_cols if col in qa_summary]
    qa_table = pd.DataFrame([qa_summary])[qa_cols].rename(
        columns={
            "overall_status": "Overall status",
            "sample_count": "Sample count",
            "expected_cadence_s": "Expected cadence (s)",
            "pct_on_expected_cadence": "% at expected cadence",
            "cadence_p95_s": "Cadence p95 (s)",
            "cadence_p99_s": "Cadence p99 (s)",
            "max_gap_s": "Max gap (s)",
            "gap_count": "Gap count",
            "gap_pct": "Gap count (%)",
            "gap_threshold_s": "Gap threshold (s)",
            "max_xy_step_yd": "Max XY step (yd)",
            "teleport_count": "Teleport count",
            "teleport_pct": "Teleport count (%)",
            "teleport_speed_threshold_yd_s": "Teleport threshold (yd/s)",
            "inactive_speed_threshold_mph": "Inactive threshold (mph)",
            "inactive_sample_count": "Inactive sample count",
            "inactive_sample_pct": "Inactive sample (%)",
            "position_outlier_low_quantile": "Position outlier q_low",
            "position_outlier_high_quantile": "Position outlier q_high",
            "position_outlier_count": "Position outlier count",
            "position_outlier_pct": "Position outlier (%)",
            "speed_xy_alignment_samples": "Speed/XY alignment samples",
            "speed_xy_corr": "Speed vs XY corr (r)",
            "speed_xy_error_p95_yd_s": "Speed vs XY error p95 (yd/s)",
            "speed_xy_outlier_pct": "Speed vs XY outlier (%)",
            "speed_xy_outlier_threshold_yd_s": "Speed vs XY outlier threshold (yd/s)",
            "step_distance_outlier_source": "Outlier source",
            "step_distance_outlier_threshold_yd": "Outlier threshold (yd)",
            "step_distance_outlier_count": "Outlier count",
            "step_distance_outlier_pct": "Outlier count (%)",
            "distance_yd_from_speed": "Distance from s*dt (yd)",
            "distance_yd_from_dis": "Distance from dis (yd)",
            "distance_dis_minus_speed_pct": "dis vs s*dt delta (%)",
        }
    )
    for col, digits in [
        ("Expected cadence (s)", 3),
        ("% at expected cadence", 2),
        ("Cadence p95 (s)", 3),
        ("Cadence p99 (s)", 3),
        ("Max gap (s)", 2),
        ("Gap threshold (s)", 2),
        ("Gap count (%)", 3),
        ("Max XY step (yd)", 2),
        ("Teleport count (%)", 3),
        ("Teleport threshold (yd/s)", 2),
        ("Inactive threshold (mph)", 2),
        ("Inactive sample (%)", 2),
        ("Position outlier q_low", 2),
        ("Position outlier q_high", 2),
        ("Position outlier (%)", 2),
        ("Speed vs XY corr (r)", 3),
        ("Speed vs XY error p95 (yd/s)", 2),
        ("Speed vs XY outlier (%)", 2),
        ("Speed vs XY outlier threshold (yd/s)", 2),
        ("Outlier threshold (yd)", 2),
        ("Outlier count (%)", 2),
        ("Distance from s*dt (yd)", 1),
        ("Distance from dis (yd)", 1),
        ("dis vs s*dt delta (%)", 1),
    ]:
        if col in qa_table.columns:
            qa_table[col] = qa_table[col].round(digits)
    qa_table.to_csv(table_dir / "slide_1_data_quality_table.csv", index=False)

    gate_table = validation_gates.copy()
    gate_table = gate_table.rename(
        columns={
            "gate": "Gate",
            "status": "Status",
            "value": "Value",
            "threshold": "Threshold",
            "direction": "Direction",
            "unit": "Unit",
            "notes": "Notes",
        }
    )
    if "Value" in gate_table.columns:
        gate_table["Value"] = pd.to_numeric(gate_table["Value"], errors="coerce").round(3)
    gate_table.to_csv(table_dir / "slide_1_validation_gates_table.csv", index=False)

    qc_checks_table = qc_checks.rename(
        columns={
            "check": "Check",
            "status": "Status",
            "value": "Value",
            "threshold": "Threshold",
            "unit": "Unit",
            "notes": "Notes",
        }
    )
    if "Value" in qc_checks_table.columns:
        qc_checks_table["Value"] = pd.to_numeric(qc_checks_table["Value"], errors="coerce").round(3)
    qc_checks_table.to_csv(table_dir / "slide_1_qc_checks_table.csv", index=False)

    pass_count = int((validation_gates["status"] == "PASS").sum()) if not validation_gates.empty else 0
    qc_check_pass = int((qc_checks["status"] == "PASS").sum()) if not qc_checks.empty else 0
    qc_check_total = int(len(qc_checks))
    dis_delta_text = "N/A"
    if "dis vs s*dt delta (%)" in qa_table.columns:
        dis_delta_text = f"{qa_table['dis vs s*dt delta (%)'].iloc[0]:.1f}%"
    qa_text_lines = [
        "Data QA Summary",
        f"- QC: {qc_status}; Overall status: {overall_status}.",
        f"- {qa_table['% at expected cadence'].iloc[0]:.1f}% samples at 0.1s cadence.",
        (
            f"- Max gap: {qa_table['Max gap (s)'].iloc[0]:.2f}s; "
            f"gaps flagged above {qa_table['Gap threshold (s)'].iloc[0]:.2f}s."
        ),
        (
            f"- Inactive samples (<= {qa_table['Inactive threshold (mph)'].iloc[0]:.1f} mph): "
            f"{qa_table['Inactive sample (%)'].iloc[0]:.1f}%."
        ) if "Inactive threshold (mph)" in qa_table.columns and "Inactive sample (%)" in qa_table.columns else "- Inactive sample % unavailable.",
        (
            f"- Position outliers ({qa_table['Position outlier q_low'].iloc[0]:.2f}-{qa_table['Position outlier q_high'].iloc[0]:.2f} quantiles): "
            f"{qa_table['Position outlier (%)'].iloc[0]:.2f}%."
        ) if "Position outlier (%)" in qa_table.columns else "- Position outlier % unavailable.",
        (
            f"- Speed-vs-XY sanity: corr {qa_table['Speed vs XY corr (r)'].iloc[0]:.3f}, "
            f"outliers {qa_table['Speed vs XY outlier (%)'].iloc[0]:.2f}%."
        ) if "Speed vs XY corr (r)" in qa_table.columns and "Speed vs XY outlier (%)" in qa_table.columns else "- Speed-vs-XY sanity metrics unavailable.",
        (
            f"- Outlier threshold: {qa_table['Outlier threshold (yd)'].iloc[0]:.2f} yd; flagged "
            f"{int(qa_table['Outlier count'].iloc[0])} samples "
            f"({qa_table['Outlier count (%)'].iloc[0]:.2f}%)."
        ),
        f"- Vendor `dis` vs speed-integrated distance delta: {dis_delta_text}.",
        f"- Validation gates passed: {pass_count}/{len(validation_gates)}.",
        f"- QC checks passed: {qc_check_pass}/{qc_check_total}.",
    ]
    if overall_status == "WARN":
        qa_text_lines.append("- Interpretation policy: usable with caveats; windows/events/maps are continuity-masked.")
    if overall_status == "FAIL":
        qa_text_lines.append("- Interpretation policy: do not use this session for workload decision thresholds.")
    qa_text_lines.extend([f"- {line}" for line in validation_takeaways[:3]])
    write_slide_text(text_dir / "slide_1_data_quality_takeaways.txt", "\n".join(qa_text_lines))

    slide2_table = coach_speed_band_table(speed_band_summary)
    slide2_table.to_csv(table_dir / "slide_2_speed_zone_table.csv", index=False)
    top_zone = slide2_table.sort_values("Distance (yd)", ascending=False).iloc[0]
    hsr_label = classify_hsr_exposure(
        total_distance_yd=float(session_summary["distance_yd_from_speed"]),
        hsr_distance_yd=float(event_counts.get("hsr_distance_yd", 0.0)),
    )
    slide2_text = (
        "Speed Zone Takeaways\n"
        f"- Largest distance accumulation: {top_zone['Zone']} ({top_zone['Distance (%)']:.1f}% of total distance).\n"
        f"- HSR exposure classification for this session: {hsr_label}.\n"
        "- Action: increase high-speed drill density only in phases already showing stable quality gates."
    )
    write_slide_text(text_dir / "slide_2_speed_zone_takeaways.txt", slide2_text)

    slide3_distance = coach_peak_distance_table(distance_table)
    slide3_extrema = coach_extrema_table(extrema)
    slide3_events = pd.DataFrame(
        [
            {
                "HSR events (>=1s)": int(event_counts.get("hsr_event_count", 0)),
                "Sprint events (>=1s)": int(event_counts.get("sprint_event_count", 0)),
                "Accel events": int(event_counts.get("accel_event_count", 0)),
                "Decel events": int(event_counts.get("decel_event_count", 0)),
                "HSR distance (yd)": float(event_counts.get("hsr_distance_yd", 0.0)),
                "Sprint distance (yd)": float(event_counts.get("sprint_distance_yd", 0.0)),
            }
        ]
    )
    slide3_events["HSR distance (yd)"] = slide3_events["HSR distance (yd)"].round(1)
    slide3_events["Sprint distance (yd)"] = slide3_events["Sprint distance (yd)"].round(1)

    slide3_top_windows = top_windows_by_duration[
        [
            "window_s",
            "window_rank",
            "window_start_utc",
            "window_end_utc",
            "distance_yd",
            "dominant_phase",
            "hsr_event_count",
            "accel_event_count",
            "decel_event_count",
        ]
    ].copy()
    slide3_top_windows = slide3_top_windows.sort_values(["window_s", "window_rank"]).reset_index(drop=True)
    if not slide3_top_windows.empty:
        slide3_top_windows["peak_intensity_yd_min"] = (
            pd.to_numeric(slide3_top_windows["distance_yd"], errors="coerce")
            * (60.0 / pd.to_numeric(slide3_top_windows["window_s"], errors="coerce").replace(0.0, np.nan))
        ).fillna(0.0)
    slide3_top_windows["Window duration"] = slide3_top_windows["window_s"].map(
        lambda s: f"{int(s // 60)}m" if int(s) % 60 == 0 else f"{int(s)}s"
    )
    slide3_top_windows = slide3_top_windows.rename(
        columns={
            "window_rank": "Window",
            "window_start_utc": "Start (UTC)",
            "window_end_utc": "End (UTC)",
            "peak_intensity_yd_min": "Peak intensity (yd/min)",
            "distance_yd": "Distance in window (yd)",
            "dominant_phase": "Dominant phase",
            "hsr_event_count": "HSR events",
            "accel_event_count": "Accel events",
            "decel_event_count": "Decel events",
        }
    )
    slide3_top_windows = slide3_top_windows[
        [
            "Window duration",
            "Window",
            "Start (UTC)",
            "End (UTC)",
            "Peak intensity (yd/min)",
            "Distance in window (yd)",
            "Dominant phase",
            "HSR events",
            "Accel events",
            "Decel events",
        ]
    ]
    slide3_top_windows["Start (UTC)"] = pd.to_datetime(
        slide3_top_windows["Start (UTC)"], utc=True, format="mixed"
    ).dt.strftime("%H:%M:%S")
    slide3_top_windows["End (UTC)"] = pd.to_datetime(
        slide3_top_windows["End (UTC)"], utc=True, format="mixed"
    ).dt.strftime("%H:%M:%S")
    if "Peak intensity (yd/min)" in slide3_top_windows.columns:
        slide3_top_windows["Peak intensity (yd/min)"] = slide3_top_windows["Peak intensity (yd/min)"].round(1)
    slide3_top_windows["Distance in window (yd)"] = slide3_top_windows["Distance in window (yd)"].round(1)

    slide3_distance.to_csv(table_dir / "slide_3_peak_distance_table.csv", index=False)
    slide3_extrema.to_csv(table_dir / "slide_3_extrema_table.csv", index=False)
    slide3_events.to_csv(table_dir / "slide_3_event_counts_table.csv", index=False)
    slide3_top_windows.to_csv(table_dir / "slide_3_top_windows_table.csv", index=False)

    if slide3_top_windows.empty:
        slide3_text = (
            "Peak Demand Takeaways\n"
            "- Not enough samples to derive stable top-demand windows in this session."
        )
    else:
        best_candidates = top_windows_by_duration[top_windows_by_duration["window_s"] == 60]
        if best_candidates.empty:
            best_candidates = top_windows_by_duration
        best = best_candidates.copy()
        best["peak_intensity_yd_min"] = (
            pd.to_numeric(best["distance_yd"], errors="coerce")
            * (60.0 / pd.to_numeric(best["window_s"], errors="coerce").replace(0.0, np.nan))
        ).fillna(0.0)
        best = best.sort_values("peak_intensity_yd_min", ascending=False).iloc[0]
        best_start = pd.Timestamp(best["window_start_utc"]).strftime("%H:%M:%S")
        best_end = pd.Timestamp(best["window_end_utc"]).strftime("%H:%M:%S")
        duration_label = f"{int(best['window_s'] // 60)}-min" if int(best["window_s"]) % 60 == 0 else f"{int(best['window_s'])}s"
        slide3_text = (
            "Peak Demand Takeaways\n"
            f"- Best {duration_label} intensity: {float(best['peak_intensity_yd_min']):.1f} yd/min "
            f"({float(best['distance_yd']):.1f} yd) from {best_start} to {best_end} UTC.\n"
            f"- Context: HSR/Sprint events {int(best['hsr_event_count'])}/{int(best['sprint_event_count'])}; "
            f"Accel/Decel {int(best['accel_event_count'])}/{int(best['decel_event_count'])}.\n"
            f"- Dominant phase: {best['dominant_phase'] or 'N/A'}.\n"
            "- Action: replicate this window pattern when the objective is peak-demand conditioning."
        )
    write_slide_text(text_dir / "slide_3_peak_takeaways.txt", slide3_text)

    slide4_table = coach_segment_table(coach_phase_summary, top_n=8)
    slide4_table.to_csv(table_dir / "slide_4_segment_table.csv", index=False)
    structure_cols = [
        "coach_phase_name",
        "dominant_field_zone",
        "duration_min",
        "distance_per_min_yd",
        "hsr_distance_per_min_yd",
        "hsr_time_pct",
    ]
    structure_cols = [col for col in structure_cols if col in session_structure_map.columns]
    structure_table = session_structure_map[structure_cols].copy()
    structure_table = structure_table.rename(
        columns={
            "coach_phase_name": "Phase",
            "dominant_field_zone": "Dominant zone",
            "duration_min": "Duration (min)",
            "distance_per_min_yd": "Distance rate (yd/min)",
            "hsr_distance_per_min_yd": "HSR rate (yd/min)",
            "hsr_time_pct": "HSR time (%)",
        }
    )
    for col, digits in [
        ("Duration (min)", 2),
        ("Distance rate (yd/min)", 1),
        ("HSR rate (yd/min)", 1),
        ("HSR time (%)", 1),
    ]:
        if col in structure_table.columns:
            structure_table[col] = pd.to_numeric(structure_table[col], errors="coerce").round(digits)
    structure_table.to_csv(table_dir / "slide_4_structure_map_table.csv", index=False)

    top_phase = coach_phase_summary.sort_values("distance_yd", ascending=False).iloc[0]
    high_phase_count = int((coach_phase_summary["intensity_level"] == "High").sum())
    slide4_text = (
        "Session Phase Takeaways\n"
        f"- Session merged into {len(coach_phase_summary)} coach-readable phases.\n"
        f"- Highest-volume phase: {top_phase['coach_phase_label']} "
        f"({top_phase['distance_yd']:.1f} yd across {top_phase['duration_s'] / 60.0:.1f} min).\n"
        f"- High-intensity phases: {high_phase_count}.\n"
        "- Action: use phase timing to place high-demand blocks where technical quality remains stable."
    )
    write_slide_text(text_dir / "slide_4_segment_takeaways.txt", slide4_text)

    slide5_table = coach_early_late_table(early_late_summary)
    slide5_table.to_csv(table_dir / "slide_5_early_late_table.csv", index=False)
    if len(slide5_table) == 2:
        late = slide5_table.loc[slide5_table["Period"] == "Late Half"].iloc[0]
        slide5_text = (
            "Early vs Late Takeaways\n"
            f"- Late-half distance vs early-half: {late['Distance vs early (%)']:+.1f}%.\n"
            f"- Late-half HSR/Sprint events: {int(late['HSR events'])} / {int(late['Sprint events'])}.\n"
            f"- Late-half accel/decel events: {int(late['Accel events'])} / {int(late['Decel events'])}.\n"
            "- Action: adjust second-half progression when high-speed or braking load drifts from intent."
        )
    else:
        slide5_text = "Early vs Late Takeaways\n- Insufficient data for a stable split-half comparison."
    write_slide_text(text_dir / "slide_5_early_late_takeaways.txt", slide5_text)

    coach_pkg_dir = output_dir / "coach_package"
    coach_pkg_dir.mkdir(parents=True, exist_ok=True)
    if top_windows_by_duration.empty:
        peak_line = "Peak demand window unavailable from current data."
    else:
        top_peak = top_windows_by_duration.copy()
        top_peak["peak_intensity_yd_min"] = (
            pd.to_numeric(top_peak["distance_yd"], errors="coerce")
            * (60.0 / pd.to_numeric(top_peak["window_s"], errors="coerce").replace(0.0, np.nan))
        ).fillna(0.0)
        top_peak = top_peak.sort_values("peak_intensity_yd_min", ascending=False).iloc[0]
        top_peak_start = pd.Timestamp(top_peak["window_start_utc"]).strftime("%H:%M:%S")
        top_peak_end = pd.Timestamp(top_peak["window_end_utc"]).strftime("%H:%M:%S")
        peak_line = (
            f"Top demand window: {float(top_peak['peak_intensity_yd_min']):.0f} yd/min "
            f"({float(top_peak['distance_yd']):.0f} yd in {int(top_peak['window_s'])}s, "
            f"{top_peak_start}-{top_peak_end} UTC), phase {top_peak['dominant_phase'] or 'N/A'}."
        )
    if len(slide5_table) == 2:
        late_row = slide5_table.loc[slide5_table["Period"] == "Late Half"].iloc[0]
        drift_line = f"Late vs early distance drift: {late_row['Distance vs early (%)']:+.1f}%."
    else:
        drift_line = "Late-vs-early comparison unavailable."
    coach_package_lines = [
        "Coach Package (3 bullets)",
        f"- QC status: {qc_status}.",
        f"- {peak_line}",
        f"- {drift_line}",
    ]
    write_slide_text(coach_pkg_dir / "coach_package_bullets.txt", "\n".join(coach_package_lines))

    figure_aliases = {
        "01_space.png": "coach_slide_movement_map.png",
        "02_time.png": "coach_slide_intensity_timeline.png",
        "03_peaks.png": "coach_slide_peak_demand_summary.png",
    }
    deprecated_aliases = [
        "coach_slide_qc_overview.png",
        "coach_slide_session_structure_map.png",
        "coach_slide_phase_speed_bands.png",
    ]
    for alias_name in deprecated_aliases:
        alias_path = fig_dir / alias_name
        if alias_path.exists():
            alias_path.unlink()

    for source_name, alias_name in figure_aliases.items():
        source = fig_dir / source_name
        if not source.exists():
            raise FileNotFoundError(
                f"Missing {source}. Run scripts/render_visual_templates.py before this script."
            )
        shutil.copyfile(source, fig_dir / alias_name)

    print("Presentation asset export complete")
    print(f"Results contract: {output_dir / 'results.json'}")
    print(f"Text blocks: {text_dir}")
    print(f"Tables: {table_dir}")
    print(f"Coach package: {coach_pkg_dir}")
    print(f"Figure aliases: {', '.join(figure_aliases.values())}")


if __name__ == "__main__":
    main()
