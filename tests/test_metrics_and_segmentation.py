from __future__ import annotations

import pandas as pd

from browns_tracking.metrics import (
    PeakDemandConfig,
    compute_peak_demand_timeseries,
    default_absolute_speed_bands,
    peak_distance_table,
    summarize_speed_bands,
    top_non_overlapping_windows,
)
from browns_tracking.pipeline import (
    classify_hsr_exposure,
    compute_data_quality_summary,
    compute_session_event_counts,
    split_early_late_summary,
    summarize_window_context,
)
from browns_tracking.segmentation import (
    SegmentationConfig,
    build_coach_phase_summary,
    detect_segments,
    summarize_segments,
)


def test_summarize_speed_bands_totals_match() -> None:
    df = pd.DataFrame(
        {
            "speed_mph": [1.0, 4.0, 10.0, 16.0],
            "step_distance_yd_from_speed": [1.0, 2.0, 3.0, 4.0],
            "dt_s": [1.0, 1.0, 1.0, 1.0],
        }
    )
    summary = summarize_speed_bands(df, default_absolute_speed_bands())

    assert summary["distance_yd"].sum() == 10.0
    sprint_distance = summary.loc[summary["speed_band"] == "Sprint", "distance_yd"].iloc[0]
    assert sprint_distance == 4.0


def test_peak_distance_table_and_top_windows() -> None:
    ts = pd.date_range("2025-01-01", periods=10, freq="1s", tz="UTC")
    df = pd.DataFrame(
        {
            "ts": ts,
            "speed_mph": [5.0] * 10,
            "signed_accel_ms2": [0.0] * 10,
            "step_distance_yd_from_speed": [1, 1, 1, 0, 0, 0, 2, 2, 2, 0],
        }
    )
    rolling = compute_peak_demand_timeseries(df, PeakDemandConfig(distance_windows_s=(3,)))
    table = peak_distance_table(rolling, [3])

    assert table["best_distance_yd"].iloc[0] == 6.0

    ranking_input = pd.DataFrame(
        {
            "ts": pd.date_range("2025-01-01", periods=4, freq="60s", tz="UTC"),
            "distance_60s_yd": [10.0, 9.0, 8.0, 7.0],
        }
    )
    top = top_non_overlapping_windows(ranking_input, "distance_60s_yd", window_s=60, top_n=2)

    assert len(top) == 2
    assert top["value"].tolist() == [10.0, 9.0]


def test_detect_segments_with_rest_period() -> None:
    ts = pd.date_range("2025-01-01", periods=100, freq="1s", tz="UTC")
    speed = ([5.0] * 30) + ([0.5] * 25) + ([5.0] * 45)
    df = pd.DataFrame(
        {
            "ts": ts,
            "dt_s": [1.0] * 100,
            "speed_mph": speed,
            "step_distance_yd_from_speed": speed,
            "signed_accel_ms2": [0.0] * 100,
        }
    )

    segmented, boundaries = detect_segments(
        df,
        SegmentationConfig(
            rest_speed_threshold_mph=1.0,
            rest_min_duration_s=20.0,
            change_point_penalty=1_000_000.0,
            min_segment_duration_s=5.0,
        ),
    )
    summary = summarize_segments(segmented, speed_bands=default_absolute_speed_bands())

    assert not boundaries.empty
    assert summary["segment_label"].iloc[0] == "Block A"
    assert summary["duration_s"].sum() == 100.0


def test_build_coach_phase_summary_limits_phase_count() -> None:
    ts = pd.date_range("2025-01-01", periods=120, freq="1s", tz="UTC")
    speed = ([3.0] * 20) + ([8.0] * 20) + ([1.0] * 10) + ([12.0] * 20) + ([2.0] * 10) + ([9.0] * 40)
    df = pd.DataFrame(
        {
            "ts": ts,
            "dt_s": [1.0] * 120,
            "speed_mph": speed,
            "step_distance_yd_from_speed": speed,
            "signed_accel_ms2": [0.0] * 120,
        }
    )
    segmented = df.copy()
    segmented["segment_id"] = [i // 10 for i in range(120)]
    segmented["segment_label"] = segmented["segment_id"].map(lambda i: f"Block {i}")

    coach_df, coach_summary = build_coach_phase_summary(
        segmented,
        min_phase_duration_s=25.0,
        max_phases=4,
    )

    assert coach_summary["coach_phase_id"].nunique() <= 4
    assert coach_df["coach_phase_label"].nunique() <= 4
    assert coach_summary["intensity_level"].isin({"Low", "Moderate", "High"}).all()


def test_event_counts_and_early_late_summary() -> None:
    ts = pd.date_range("2025-01-01", periods=40, freq="1s", tz="UTC")
    speed = ([5.0] * 10) + ([14.0] * 8) + ([5.0] * 6) + ([16.5] * 8) + ([4.0] * 8)
    signed_accel = ([0.0] * 10) + ([3.5] * 2) + ([0.0] * 8) + ([-3.5] * 2) + ([0.0] * 18)
    df = pd.DataFrame(
        {
            "ts": ts,
            "dt_s": [1.0] * 40,
            "speed_mph": speed,
            "signed_accel_ms2": signed_accel,
            "step_distance_yd_from_speed": speed,
        }
    )

    event_counts = compute_session_event_counts(df)
    split = split_early_late_summary(df)

    assert event_counts["hsr_event_count"] >= 1
    assert event_counts["sprint_event_count"] >= 1
    assert event_counts["accel_event_count"] >= 1
    assert event_counts["decel_event_count"] >= 1
    assert split["period"].tolist() == ["Early Half", "Late Half"]


def test_data_quality_summary_and_exposure_labels() -> None:
    df = pd.DataFrame(
        {
            "dt_s": [0.1, 0.1, 0.3, 0.1],
            "step_distance_yd_from_speed": [1.0, 1.2, 10.0, 1.1],
        }
    )
    qa = compute_data_quality_summary(df, outlier_quantile=0.75, gap_threshold_s=0.15)
    assert qa["gap_count"] == 1
    assert qa["step_distance_outlier_count"] >= 1

    assert classify_hsr_exposure(1000.0, 20.0) == "Low"
    assert classify_hsr_exposure(1000.0, 70.0) == "Moderate"
    assert classify_hsr_exposure(1000.0, 150.0) == "High"


def test_summarize_window_context() -> None:
    ts = pd.date_range("2025-01-01", periods=20, freq="1s", tz="UTC")
    df = pd.DataFrame(
        {
            "ts": ts,
            "dt_s": [1.0] * 20,
            "speed_mph": ([5.0] * 8) + ([14.0] * 8) + ([5.0] * 4),
            "signed_accel_ms2": ([0.0] * 10) + ([3.5] * 2) + ([0.0] * 8),
            "step_distance_yd_from_speed": [1.0] * 20,
        }
    )
    out = summarize_window_context(
        df,
        window_start_utc=ts[6],
        window_end_utc=ts[15],
    )

    assert out["distance_yd"] > 0
    assert out["hsr_event_count"] >= 1
