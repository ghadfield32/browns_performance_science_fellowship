"""Run core analyses and export the first slide-ready visual templates."""

from __future__ import annotations

import argparse

import pandas as pd

from browns_tracking.metrics import (
    compute_peak_demand_timeseries,
    peak_distance_table,
    session_extrema_table,
    summarize_speed_bands,
    top_non_overlapping_windows,
)
from browns_tracking.config import resolve_data_file, resolve_output_dir
from browns_tracking.pipeline import (
    compute_session_event_counts,
    load_tracking_data,
    split_early_late_summary,
    summarize_session,
)
from browns_tracking.presets import preferred_performance_model
from browns_tracking.segmentation import (
    build_coach_phase_summary,
    detect_segments,
    summarize_segments,
)
from browns_tracking.visuals import (
    plot_intensity_timeline,
    plot_movement_map,
    plot_peak_demand_summary,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export movement map, intensity timeline, and peak-demand summary figures."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to tracking_data.csv (default: auto-resolve from project config).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for figure and table exports (default: auto-resolve from project config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_data_file(args.input)
    output_dir = resolve_output_dir(args.output_dir)
    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    df = load_tracking_data(input_path)
    session_summary = summarize_session(df)

    model = preferred_performance_model()

    abs_bands = list(model.absolute_speed_bands)
    abs_band_summary = summarize_speed_bands(df, abs_bands)

    peak_cfg = model.peak_demand_config
    rolling = compute_peak_demand_timeseries(df, peak_cfg)
    distance_table = peak_distance_table(rolling, peak_cfg.distance_windows_s)
    top_windows = top_non_overlapping_windows(rolling, "distance_60s_yd", window_s=60, top_n=3)
    extrema = session_extrema_table(df)
    event_counts = compute_session_event_counts(
        df,
        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,
        accel_threshold_ms2=peak_cfg.accel_threshold_ms2,
        decel_threshold_ms2=peak_cfg.decel_threshold_ms2,
    )
    early_late = split_early_late_summary(
        df,
        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,
        accel_threshold_ms2=peak_cfg.accel_threshold_ms2,
        decel_threshold_ms2=peak_cfg.decel_threshold_ms2,
    )

    seg_df, segment_boundaries = detect_segments(df, model.segmentation_config)
    segment_summary = summarize_segments(seg_df, speed_bands=abs_bands)
    coach_df, coach_phase_summary = build_coach_phase_summary(
        seg_df,
        min_phase_duration_s=30.0,
        max_phases=8,
        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,
        accel_threshold_ms2=peak_cfg.accel_threshold_ms2,
        decel_threshold_ms2=peak_cfg.decel_threshold_ms2,
    )

    fig1, _ = plot_movement_map(coach_df, segment_col="coach_phase_label", highlight_top_n=3)
    save_figure(fig1, fig_dir / "movement_map.png")

    fig2, _ = plot_intensity_timeline(
        seg_df,
        top_windows=top_windows,
        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,
    )
    save_figure(fig2, fig_dir / "intensity_timeline.png")

    fig3, _ = plot_peak_demand_summary(distance_table, extrema)
    save_figure(fig3, fig_dir / "peak_demand_summary.png")

    abs_band_summary.to_csv(table_dir / "absolute_speed_band_summary.csv", index=False)
    distance_table.to_csv(table_dir / "peak_distance_windows.csv", index=False)
    top_windows.to_csv(table_dir / "top_1m_distance_windows.csv", index=False)
    extrema.to_csv(table_dir / "session_extrema.csv", index=False)
    pd.DataFrame([event_counts]).to_csv(table_dir / "session_event_counts.csv", index=False)
    early_late.to_csv(table_dir / "early_vs_late_summary.csv", index=False)
    segment_boundaries.to_csv(table_dir / "raw_segment_boundaries.csv", index=False)
    segment_summary.to_csv(table_dir / "raw_segment_summary.csv", index=False)
    coach_phase_summary.to_csv(table_dir / "coach_phase_summary.csv", index=False)

    print("Export complete")
    print(f"Session rows: {session_summary['rows']}")
    print(f"Input data: {input_path}")
    print(f"Figures: {fig_dir}")
    print(f"Tables: {table_dir}")


if __name__ == "__main__":
    main()
