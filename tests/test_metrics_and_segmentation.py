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
    annotate_tracking_continuity,
    build_validation_takeaways,
    classify_hsr_exposure,
    compute_data_quality_summary,
    compute_session_event_counts,
    compute_validation_gates,
    load_tracking_data,
    split_early_late_summary,
    summarize_window_context,
)
from browns_tracking.results_contract import (
    build_session_structure_map_table,
    load_results_contract,
    run_session_analysis,
    write_results_contract,
    write_results_tables,
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
            "dt_s": [1.0] * 10,
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


def test_annotate_tracking_continuity_breaks_on_gap_and_jump() -> None:
    ts = pd.to_datetime(
        [
            "2025-01-01T00:00:00Z",
            "2025-01-01T00:00:01Z",
            "2025-01-01T00:00:02Z",
            "2025-01-01T00:00:03Z",
        ],
        utc=True,
    )
    df = pd.DataFrame(
        {
            "ts": ts,
            "dt_s": [1.0, 1.0, 2.0, 1.0],
            "step_distance_xy_yd": [0.0, 1.0, 1.0, 20.0],
            "step_distance_yd_from_speed": [0.0, 1.0, 1.0, 1.0],
            "speed_mph": [1.0, 2.0, 3.0, 4.0],
            "signed_accel_ms2": [0.0, 0.0, 0.0, 0.0],
        }
    )

    out = annotate_tracking_continuity(df, gap_threshold_s=1.5, max_plausible_xy_speed_yd_s=10.0)

    assert out["continuity_break_before"].tolist() == [False, False, True, True]
    assert out["continuous_block_id"].tolist() == [0, 0, 1, 2]


def test_event_counts_reset_at_continuity_boundaries() -> None:
    ts = pd.date_range("2025-01-01", periods=6, freq="1s", tz="UTC")
    df = pd.DataFrame(
        {
            "ts": ts,
            "dt_s": [1.0] * 6,
            "speed_mph": [14.0] * 6,
            "signed_accel_ms2": [0.0] * 6,
            "step_distance_yd_from_speed": [1.0] * 6,
            "continuity_break_before": [False, False, False, True, False, False],
        }
    )

    out = compute_session_event_counts(
        df,
        hsr_threshold_mph=13.0,
        min_speed_event_duration_s=2.0,
    )
    assert out["hsr_event_count"] == 2


def test_peak_demand_timeseries_respects_blocks() -> None:
    ts = pd.date_range("2025-01-01", periods=8, freq="1s", tz="UTC")
    df = pd.DataFrame(
        {
            "ts": ts,
            "dt_s": [1.0] * 8,
            "speed_mph": [8.0] * 8,
            "signed_accel_ms2": [0.0] * 8,
            "step_distance_yd_from_speed": [3.0, 3.0, 3.0, 3.0, 10.0, 10.0, 10.0, 10.0],
            "continuous_block_id": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    rolling = compute_peak_demand_timeseries(
        df,
        PeakDemandConfig(distance_windows_s=(5,)),
        block_col="continuous_block_id",
    )
    table = peak_distance_table(rolling, [5])

    # With block-aware rolling, 5s windows cannot bridge the block boundary.
    assert table["best_distance_yd"].iloc[0] == 0.0


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


def test_validation_gates_and_takeaways() -> None:
    df = pd.DataFrame(
        {
            "dt_s": [0.1, 0.1, 0.3, 0.1, 0.1],
            "step_distance_xy_yd": [0.0, 0.1, 8.0, 0.2, 0.2],
            "s": [1.0, 1.0, 1.1, 1.0, 1.0],
            "speed_mph": [2.0, 2.0, 2.2, 2.0, 2.0],
            "step_distance_yd_from_speed": [0.1, 0.1, 0.33, 0.1, 0.1],
        }
    )
    gates = compute_validation_gates(df, max_plausible_xy_speed_yd_s=10.0, max_jump_pct=0.0)
    takeaways = build_validation_takeaways(gates)

    assert set(gates["gate_key"]) == {
        "cadence_on_target_pct",
        "max_gap_s",
        "improbable_jump_pct",
        "speed_alignment_corr",
        "speed_alignment_outlier_pct",
    }
    assert len(takeaways) >= 3
    assert "Validation gates" in takeaways[0]


def test_results_contract_roundtrip(tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "ts": pd.date_range("2025-01-01", periods=180, freq="1s", tz="UTC"),
            "x": [float(i) * 0.8 for i in range(180)],
            "y": [float(i % 20) for i in range(180)],
            "s": [3.0] * 60 + [6.0] * 60 + [4.0] * 60,
            "a": [0.0] * 180,
            "sa": [0.0] * 180,
        }
    )
    csv_path = tmp_path / "tracking.csv"
    raw.to_csv(csv_path, index=False)

    loaded = load_tracking_data(csv_path)
    results = run_session_analysis(loaded)
    table_paths = write_results_tables(results, tmp_path)
    contract_path = write_results_contract(
        results,
        input_path=csv_path,
        output_dir=tmp_path,
        table_paths=table_paths,
        figure_paths={"space": tmp_path / "figures/01_space.png"},
    )
    contract = load_results_contract(tmp_path)

    assert contract_path.exists()
    assert (tmp_path / "phase_table.csv").exists()
    assert (tmp_path / "peak_windows.csv").exists()
    assert (tmp_path / "session_structure_map.csv").exists()
    assert contract["story_questions"][0].startswith("Where did the player")
    assert "validation_gates" in contract


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


def test_build_session_structure_map_table() -> None:
    ts = pd.date_range("2025-01-01", periods=6, freq="1s", tz="UTC")
    coach_df = pd.DataFrame(
        {
            "ts": ts,
            "dt_s": [1.0] * 6,
            "x": [0.0, 1.0, 2.0, 20.0, 21.0, 22.0],
            "speed_mph": [4.0, 5.0, 6.0, 12.0, 13.0, 14.0],
            "step_distance_yd_from_speed": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "coach_phase_id": [1, 1, 1, 2, 2, 2],
            "coach_phase_name": ["Phase 01"] * 3 + ["Phase 02"] * 3,
            "coach_phase_label": ["Phase 01: Low Intensity"] * 3 + ["Phase 02: High Intensity"] * 3,
            "coach_intensity_level": ["Low"] * 3 + ["High"] * 3,
        }
    )
    out = build_session_structure_map_table(coach_df, hsr_threshold_mph=13.0)

    assert len(out) == 2
    assert set(out["dominant_field_zone"]) <= {"Back Third", "Middle Third", "Front Third"}
