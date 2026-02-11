from __future__ import annotations

import pandas as pd

from browns_tracking.metrics import (
    PeakDemandConfig,
    SpeedBand,
    compute_peak_demand_timeseries,
    default_absolute_speed_bands,
    peak_distance_table,
    summarize_speed_bands,
    top_non_overlapping_windows,
)
from browns_tracking.pipeline import (
    annotate_tracking_continuity,
    build_visualization_frame,
    build_qc_check_table,
    build_validation_takeaways,
    classify_hsr_exposure,
    compute_data_quality_summary,
    compute_qc_status,
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
from browns_tracking.presets import PerformanceModelPreset


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
    assert table["best_intensity_yd_per_min"].iloc[0] == 120.0

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
    assert "coach_phase_type" in coach_summary.columns
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
    assert "inactive_sample_pct" in qa
    assert "position_outlier_pct" in qa
    assert "speed_xy_corr" in qa

    assert classify_hsr_exposure(1000.0, 20.0) == "Low"
    assert classify_hsr_exposure(1000.0, 70.0) == "Moderate"
    assert classify_hsr_exposure(1000.0, 150.0) == "High"


def test_build_visualization_frame_filters_samples() -> None:
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2025-01-01", periods=6, freq="1s", tz="UTC"),
            "x": [0.0, 0.5, 1.0, 2.0, 200.0, 2.5],
            "y": [0.0, 0.5, 1.0, 2.0, 300.0, 2.5],
            "dt_s": [0.1, 0.1, 0.1, 0.5, 0.1, 0.1],
            "speed_mph": [0.2, 0.6, 1.0, 1.2, 1.4, 1.1],
            "spatial_sample_ok": [True, True, True, True, True, True],
            "continuous_block_id": [0, 0, 0, 1, 1, 1],
        }
    )
    out = build_visualization_frame(df, active_speed_threshold_mph=0.5, gap_threshold_s=0.2)
    assert not out.empty
    assert bool((out["speed_mph"] >= 0.5).all())
    assert bool((out["dt_s"] <= 0.2).all())


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
        "vendor_dis_distance_delta_pct",
    }
    assert len(takeaways) >= 3
    assert "Validation gates" in takeaways[0]


def test_qc_checks_and_status_include_dis_alignment() -> None:
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2025-01-01", periods=6, freq="1s", tz="UTC"),
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 10.2, 10.4, 10.6, 10.8, 11.0],
            "dt_s": [1.0] * 6,
            "s": [1.0, 1.1, 1.0, 1.2, 1.1, 1.0],
            "speed_mph": [2.0, 2.2, 2.0, 2.4, 2.2, 2.0],
            "step_distance_xy_yd": [0.0, 1.0, 1.1, 0.9, 1.2, 1.0],
            "step_distance_yd_from_speed": [1.0, 1.1, 1.0, 1.2, 1.1, 1.0],
            "step_distance_yd_vendor_dis": [20.0] * 6,
            "signed_accel_ms2": [0.1, -0.1, 0.2, -0.2, 0.1, -0.1],
        }
    )
    qa = compute_data_quality_summary(df, gap_threshold_s=1.5)
    gates = compute_validation_gates(df, max_plausible_xy_speed_yd_s=10.0)
    checks = build_qc_check_table(df, qa_summary=qa, validation_gates=gates)
    status = compute_qc_status(checks)

    assert "vendor_dis_distance_delta_pct" in set(checks["check_key"])
    assert status == "QC FAILED"


def test_qc_status_warn_for_limited_impact_issues() -> None:
    speed_yd_s = [1.0 + (0.01 * i) for i in range(20)]
    dt_s = [0.1] * 10 + [6.0] + [0.1] * 9
    step_dist = [s * dt for s, dt in zip(speed_yd_s, dt_s, strict=False)]
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2025-01-01", periods=20, freq="1s", tz="UTC"),
            "x": [float(i) for i in range(20)],
            "y": [20.0 + float(i) * 0.5 for i in range(20)],
            "dt_s": dt_s,
            "s": speed_yd_s,
            "speed_mph": [s * 2.0454545454545454 for s in speed_yd_s],
            "step_distance_xy_yd": step_dist,
            "step_distance_yd_from_speed": step_dist,
            "step_distance_yd_vendor_dis": [1.5] * 20,  # 50% delta vs speed distance.
            "signed_accel_ms2": [0.1, -0.1] * 10,
        }
    )
    qa = compute_data_quality_summary(df, gap_threshold_s=1.5)
    gates = compute_validation_gates(df, max_plausible_xy_speed_yd_s=10.0)
    checks = build_qc_check_table(df, qa_summary=qa, validation_gates=gates)
    status = compute_qc_status(checks)

    assert status == "QC WARN"


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
    assert contract["thresholds"]["sprint_threshold_mph"] == 16.0


def test_run_session_analysis_uses_model_sprint_threshold(tmp_path) -> None:
    raw = pd.DataFrame(
        {
            "ts": pd.date_range("2025-01-01", periods=60, freq="1s", tz="UTC"),
            "x": [float(i) * 0.5 for i in range(60)],
            "y": [float(i % 10) for i in range(60)],
            "s": [8.8] * 20 + [7.9] * 20 + [6.0] * 20,  # ~18.0 mph, ~16.2 mph, ~12.3 mph
            "a": [0.0] * 60,
            "sa": [0.0] * 60,
        }
    )
    csv_path = tmp_path / "tracking.csv"
    raw.to_csv(csv_path, index=False)
    loaded = load_tracking_data(csv_path)

    custom_model = PerformanceModelPreset(
        name="Custom Sprint Threshold Model",
        rationale="Tests sprint-threshold propagation from speed bands.",
        absolute_speed_bands=(
            SpeedBand("Walk", 0.0, 3.0),
            SpeedBand("Cruise", 3.0, 9.0),
            SpeedBand("Run", 9.0, 13.0),
            SpeedBand("HSR", 13.0, 18.0),
            SpeedBand("Sprint", 18.0, None),
        ),
        relative_band_edges=(0.0, 0.4, 0.6, 0.8, 0.9, 1.0),
        peak_demand_config=PeakDemandConfig(
            distance_windows_s=(30, 60),
            hsr_threshold_mph=13.0,
            accel_threshold_ms2=3.0,
            decel_threshold_ms2=-3.0,
        ),
        segmentation_config=SegmentationConfig(),
    )
    results = run_session_analysis(loaded, model=custom_model)

    expected_sprint_distance = float(
        loaded.loc[loaded["speed_mph"] >= 18.0, "step_distance_yd_from_speed"].sum()
    )
    assert results.event_counts["sprint_distance_yd"] == expected_sprint_distance


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
