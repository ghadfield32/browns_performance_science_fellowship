"""Export presentation-first assets (slide text, tables, and figures)."""

from __future__ import annotations

import argparse

import pandas as pd

from browns_tracking.config import resolve_data_file, resolve_output_dir
from browns_tracking.metrics import (
    compute_peak_demand_timeseries,
    peak_distance_table,
    session_extrema_table,
    summarize_speed_bands,
    top_non_overlapping_windows,
)
from browns_tracking.pipeline import (
    compute_session_event_counts,
    load_tracking_data,
    split_early_late_summary,
    summarize_session,
)
from browns_tracking.presentation import (
    build_slide_1_snapshot_text,
    coach_early_late_table,
    coach_extrema_table,
    coach_peak_distance_table,
    coach_segment_table,
    coach_speed_band_table,
    write_slide_text,
)
from browns_tracking.presets import preferred_performance_model
from browns_tracking.segmentation import build_coach_phase_summary, detect_segments
from browns_tracking.visuals import (
    plot_intensity_timeline,
    plot_movement_map,
    plot_peak_demand_summary,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export coach-slide text, tables, and figures.")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to tracking_data.csv (default: auto-resolve from project config).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for figure/table/text exports (default: auto-resolve from project config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_data_file(args.input)
    output_dir = resolve_output_dir(args.output_dir)
    fig_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    text_dir = output_dir / "slide_text"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    model = preferred_performance_model()
    df = load_tracking_data(input_path)
    session_summary = summarize_session(df)

    abs_bands = list(model.absolute_speed_bands)
    speed_band_summary = summarize_speed_bands(df, abs_bands)

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

    seg_df, _ = detect_segments(df, model.segmentation_config)
    coach_df, coach_phase_summary = build_coach_phase_summary(
        seg_df,
        min_phase_duration_s=30.0,
        max_phases=8,
        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,
        accel_threshold_ms2=peak_cfg.accel_threshold_ms2,
        decel_threshold_ms2=peak_cfg.decel_threshold_ms2,
    )

    slide1_text = build_slide_1_snapshot_text(
        session_summary,
        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,
        event_summary=event_counts,
    )
    write_slide_text(text_dir / "slide_1_session_snapshot.txt", slide1_text)

    slide2_table = coach_speed_band_table(speed_band_summary)
    slide2_table.to_csv(table_dir / "slide_2_speed_zone_table.csv", index=False)
    top_zone = slide2_table.sort_values("Distance (yd)", ascending=False).iloc[0]
    slide2_text = (
        "Speed Zone Takeaways\n"
        f"- Largest distance accumulation: {top_zone['Zone']} ({top_zone['Distance (%)']:.1f}% of total distance).\n"
        f"- HSR/Sprint thresholds begin at {peak_cfg.hsr_threshold_mph:.1f} mph and 16.0 mph, respectively."
    )
    write_slide_text(text_dir / "slide_2_speed_zone_takeaways.txt", slide2_text)

    slide3_distance = coach_peak_distance_table(distance_table)
    slide3_extrema = coach_extrema_table(extrema)
    slide3_events = pd.DataFrame(
        [
            {
                "HSR events (>=1s)": int(event_counts["hsr_event_count"]),
                "Sprint events (>=1s)": int(event_counts["sprint_event_count"]),
                "Accel events": int(event_counts["accel_event_count"]),
                "Decel events": int(event_counts["decel_event_count"]),
                "HSR distance (yd)": float(event_counts["hsr_distance_yd"]),
                "Sprint distance (yd)": float(event_counts["sprint_distance_yd"]),
            }
        ]
    )
    slide3_events["HSR distance (yd)"] = slide3_events["HSR distance (yd)"].round(1)
    slide3_events["Sprint distance (yd)"] = slide3_events["Sprint distance (yd)"].round(1)
    slide3_distance.to_csv(table_dir / "slide_3_peak_distance_table.csv", index=False)
    slide3_extrema.to_csv(table_dir / "slide_3_extrema_table.csv", index=False)
    slide3_events.to_csv(table_dir / "slide_3_event_counts_table.csv", index=False)

    slide3_top_windows = top_windows[["window_start_utc", "window_end_utc", "value"]].copy()
    slide3_top_windows = slide3_top_windows.rename(
        columns={
            "window_start_utc": "Start (UTC)",
            "window_end_utc": "End (UTC)",
            "value": "Distance in 60s (yd)",
        }
    )
    slide3_top_windows["Start (UTC)"] = pd.to_datetime(slide3_top_windows["Start (UTC)"]).dt.strftime(
        "%H:%M:%S"
    )
    slide3_top_windows["End (UTC)"] = pd.to_datetime(slide3_top_windows["End (UTC)"]).dt.strftime(
        "%H:%M:%S"
    )
    slide3_top_windows["Distance in 60s (yd)"] = slide3_top_windows["Distance in 60s (yd)"].round(1)
    slide3_top_windows.to_csv(table_dir / "slide_3_top_windows_table.csv", index=False)

    best_window = slide3_top_windows.iloc[0]
    slide3_text = (
        "Peak Demand Takeaways\n"
        f"- Best 1-min demand: {best_window['Distance in 60s (yd)']:.1f} yd from {best_window['Start (UTC)']} to {best_window['End (UTC)']} UTC.\n"
        f"- Max speed: {slide3_extrema.loc[slide3_extrema['Metric'] == 'Max speed (mph)', 'Value'].iloc[0]:.2f} mph.\n"
        f"- HSR/Sprint events (>=1s): {int(event_counts['hsr_event_count'])} / {int(event_counts['sprint_event_count'])}.\n"
        f"- Accel/Decel events (±{peak_cfg.accel_threshold_ms2:.1f} m/s^2): "
        f"{int(event_counts['accel_event_count'])} / {int(event_counts['decel_event_count'])}."
    )
    write_slide_text(text_dir / "slide_3_peak_takeaways.txt", slide3_text)

    slide4_table = coach_segment_table(coach_phase_summary, top_n=8)
    slide4_table.to_csv(table_dir / "slide_4_segment_table.csv", index=False)
    top_phase = coach_phase_summary.sort_values("distance_yd", ascending=False).iloc[0]
    high_phase_count = int((coach_phase_summary["intensity_level"] == "High").sum())
    slide4_text = (
        "Session Phase Takeaways\n"
        f"- Algorithmic blocks were merged into {len(coach_phase_summary)} coach-readable phases.\n"
        f"- Highest volume phase: {top_phase['coach_phase_label']} "
        f"({top_phase['distance_yd']:.1f} yd across {top_phase['duration_s'] / 60.0:.1f} min).\n"
        f"- High-intensity phases identified: {high_phase_count}."
    )
    write_slide_text(text_dir / "slide_4_segment_takeaways.txt", slide4_text)

    slide5_table = coach_early_late_table(early_late)
    slide5_table.to_csv(table_dir / "slide_5_early_late_table.csv", index=False)
    if len(slide5_table) == 2:
        late = slide5_table.loc[slide5_table["Period"] == "Late Half"].iloc[0]
        slide5_text = (
            "Early vs Late Takeaways\n"
            f"- Late-half distance vs early-half: {late['Distance vs early (%)']:+.1f}%.\n"
            f"- Late-half HSR/Sprint events: {int(late['HSR events'])} / {int(late['Sprint events'])}.\n"
            f"- Late-half accel/decel events: {int(late['Accel events'])} / {int(late['Decel events'])}."
        )
    else:
        slide5_text = "Early vs Late Takeaways\n- Insufficient data to compute a stable split-half comparison."
    write_slide_text(text_dir / "slide_5_early_late_takeaways.txt", slide5_text)

    fig1, _ = plot_movement_map(coach_df, segment_col="coach_phase_label", highlight_top_n=3)
    fig2, _ = plot_intensity_timeline(
        seg_df,
        top_windows=top_windows,
        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,
    )
    fig3, _ = plot_peak_demand_summary(distance_table, extrema)
    save_figure(fig1, fig_dir / "coach_slide_movement_map.png")
    save_figure(fig2, fig_dir / "coach_slide_intensity_timeline.png")
    save_figure(fig3, fig_dir / "coach_slide_peak_demand_summary.png")

    print("Presentation asset export complete")
    print(f"Input data: {input_path}")
    print(f"Text blocks: {text_dir}")
    print(f"Tables: {table_dir}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
