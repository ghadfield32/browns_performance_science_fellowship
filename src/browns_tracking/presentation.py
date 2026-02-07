"""Coach-facing text/table formatting helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_slide_1_snapshot_text(
    session_summary: dict[str, float | int | str],
    *,
    hsr_threshold_mph: float,
    event_summary: dict[str, float | int] | None = None,
) -> str:
    """Create a short text block for the session-overview slide."""
    duration_min = float(session_summary["duration_s"]) / 60.0
    event_lines = ""
    if event_summary is not None:
        event_lines = (
            f"\n- HSR/Sprint events (>=1s): {int(event_summary['hsr_event_count'])} / "
            f"{int(event_summary['sprint_event_count'])}"
            f"\n- Accel/Decel events: {int(event_summary['accel_event_count'])} / "
            f"{int(event_summary['decel_event_count'])}"
        )
    text = (
        "Session Snapshot\n"
        f"- Duration: {duration_min:.1f} min\n"
        f"- Total distance: {float(session_summary['distance_yd_from_speed']):,.0f} yd\n"
        f"- Mean speed: {float(session_summary['mean_speed_mph']):.2f} mph\n"
        f"- Peak speed: {float(session_summary['peak_speed_mph']):.2f} mph\n"
        f"- Peak accel/decel: {float(session_summary['peak_accel_ms2']):.2f} / "
        f"{float(session_summary['peak_decel_ms2']):.2f} m/s^2\n"
        f"- HSR threshold for this report: {hsr_threshold_mph:.1f} mph\n"
        f"- Sampling gaps (>0.15 s): {int(session_summary['sampling_gap_count'])}"
        f"{event_lines}"
    )
    return text


def coach_speed_band_table(speed_band_summary: pd.DataFrame) -> pd.DataFrame:
    """Round and rename speed-band summary columns for slide tables."""
    table = speed_band_summary.copy()
    table = table.rename(
        columns={
            "speed_band": "Zone",
            "distance_yd": "Distance (yd)",
            "distance_pct": "Distance (%)",
            "time_s": "Time (s)",
            "time_pct": "Time (%)",
            "mean_speed_mph": "Mean speed (mph)",
        }
    )
    for col, digits in [
        ("Distance (yd)", 1),
        ("Distance (%)", 1),
        ("Time (s)", 1),
        ("Time (%)", 1),
        ("Mean speed (mph)", 2),
    ]:
        if col in table.columns:
            table[col] = table[col].round(digits)

    keep = ["Zone", "Distance (yd)", "Distance (%)", "Time (s)", "Time (%)", "Mean speed (mph)"]
    return table[keep]


def coach_peak_distance_table(distance_table: pd.DataFrame) -> pd.DataFrame:
    """Round and rename best rolling distance table for slides."""
    table = distance_table.copy()
    table = table.rename(
        columns={
            "window_label": "Window",
            "best_distance_yd": "Best distance (yd)",
            "window_start_utc": "Start (UTC)",
            "window_end_utc": "End (UTC)",
        }
    )
    table["Best distance (yd)"] = table["Best distance (yd)"].round(1)
    table["Start (UTC)"] = pd.to_datetime(table["Start (UTC)"]).dt.strftime("%H:%M:%S")
    table["End (UTC)"] = pd.to_datetime(table["End (UTC)"]).dt.strftime("%H:%M:%S")
    return table[["Window", "Best distance (yd)", "Start (UTC)", "End (UTC)"]]


def coach_extrema_table(extrema_table: pd.DataFrame) -> pd.DataFrame:
    """Format max speed/accel/decel table for slides."""
    table = extrema_table.copy()
    table = table.rename(columns={"metric": "Metric", "value": "Value", "ts_utc": "Time (UTC)"})
    table["Value"] = table["Value"].round(2)
    table["Time (UTC)"] = pd.to_datetime(table["Time (UTC)"]).dt.strftime("%H:%M:%S")
    return table[["Metric", "Value", "Time (UTC)"]]


def coach_segment_table(segment_summary: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Compact phase table intended for one slide."""
    table = segment_summary.copy()
    if "duration_s" in table.columns:
        table["duration_min"] = table["duration_s"] / 60.0

    label_col = "coach_phase_label" if "coach_phase_label" in table.columns else "segment_label"
    rename_map = {
        label_col: "Phase",
        "intensity_level": "Intensity",
        "duration_min": "Duration (min)",
        "distance_yd": "Distance (yd)",
        "mean_speed_mph": "Mean speed (mph)",
        "peak_speed_mph": "Peak speed (mph)",
        "peak_60s_distance_yd": "Peak 60s distance (yd)",
        "hsr_distance_yd": "HSR distance (yd)",
        "accel_event_count": "Accel events",
        "decel_event_count": "Decel events",
    }
    table = table.rename(columns=rename_map)

    keep = [
        "Phase",
        "Intensity",
        "Duration (min)",
        "Distance (yd)",
        "HSR distance (yd)",
        "Accel events",
        "Decel events",
        "Mean speed (mph)",
        "Peak speed (mph)",
        "Peak 60s distance (yd)",
    ]
    keep = [col for col in keep if col in table.columns]
    table = table[keep].copy()

    for col, digits in [
        ("Duration (min)", 2),
        ("Distance (yd)", 1),
        ("HSR distance (yd)", 1),
        ("Mean speed (mph)", 2),
        ("Peak speed (mph)", 2),
        ("Peak 60s distance (yd)", 1),
    ]:
        if col in table.columns:
            table[col] = table[col].round(digits)

    return table.sort_values("Distance (yd)", ascending=False).head(top_n).reset_index(drop=True)


def coach_early_late_table(early_late_summary: pd.DataFrame) -> pd.DataFrame:
    """Format early-vs-late workload split for slide delivery."""
    table = early_late_summary.copy()
    table = table.rename(
        columns={
            "period": "Period",
            "duration_min": "Duration (min)",
            "distance_yd": "Distance (yd)",
            "mean_speed_mph": "Mean speed (mph)",
            "peak_speed_mph": "Peak speed (mph)",
            "hsr_distance_yd": "HSR distance (yd)",
            "sprint_distance_yd": "Sprint distance (yd)",
            "hsr_event_count": "HSR events",
            "sprint_event_count": "Sprint events",
            "accel_event_count": "Accel events",
            "decel_event_count": "Decel events",
            "distance_vs_early_pct": "Distance vs early (%)",
        }
    )

    for col, digits in [
        ("Duration (min)", 2),
        ("Distance (yd)", 1),
        ("Mean speed (mph)", 2),
        ("Peak speed (mph)", 2),
        ("HSR distance (yd)", 1),
        ("Sprint distance (yd)", 1),
        ("Distance vs early (%)", 1),
    ]:
        if col in table.columns:
            table[col] = table[col].round(digits)

    keep = [
        "Period",
        "Duration (min)",
        "Distance (yd)",
        "Distance vs early (%)",
        "Mean speed (mph)",
        "Peak speed (mph)",
        "HSR distance (yd)",
        "Sprint distance (yd)",
        "HSR events",
        "Sprint events",
        "Accel events",
        "Decel events",
    ]
    keep = [col for col in keep if col in table.columns]
    return table[keep]


def write_slide_text(output_path: str | Path, text: str) -> None:
    """Persist a copy/paste text block for slide drafting."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
