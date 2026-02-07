from __future__ import annotations

import pandas as pd

from browns_tracking.presentation import (
    build_slide_1_snapshot_text,
    coach_early_late_table,
    coach_speed_band_table,
)
from browns_tracking.presets import preferred_performance_model


def test_preferred_model_thresholds() -> None:
    model = preferred_performance_model()
    assert model.peak_demand_config.hsr_threshold_mph == 13.0
    assert model.peak_demand_config.accel_threshold_ms2 == 3.0
    assert model.segmentation_config.rest_speed_threshold_mph == 1.2
    assert [band.name for band in model.absolute_speed_bands] == [
        "Walk",
        "Cruise",
        "Run",
        "HSR",
        "Sprint",
    ]


def test_slide_1_snapshot_text_contains_core_fields() -> None:
    summary = {
        "duration_s": 600.0,
        "distance_yd_from_speed": 1000.0,
        "mean_speed_mph": 4.2,
        "peak_speed_mph": 17.8,
        "peak_accel_ms2": 3.6,
        "peak_decel_ms2": -3.9,
        "sampling_gap_count": 2,
    }
    text = build_slide_1_snapshot_text(summary, hsr_threshold_mph=13.0)

    assert "Session Snapshot" in text
    assert "HSR threshold for this report: 13.0 mph" in text
    assert "Sampling gaps" in text


def test_coach_speed_band_table_column_names() -> None:
    df = pd.DataFrame(
        {
            "speed_band": ["Walk"],
            "distance_yd": [100.0],
            "distance_pct": [50.0],
            "time_s": [80.0],
            "time_pct": [60.0],
            "sample_count": [20],
            "mean_speed_mph": [2.0],
        }
    )
    out = coach_speed_band_table(df)
    assert list(out.columns) == [
        "Zone",
        "Distance (yd)",
        "Distance (%)",
        "Time (s)",
        "Time (%)",
        "Mean speed (mph)",
    ]


def test_coach_early_late_table_columns() -> None:
    df = pd.DataFrame(
        {
            "period": ["Early Half", "Late Half"],
            "duration_min": [10.0, 10.0],
            "distance_yd": [1000.0, 900.0],
            "distance_vs_early_pct": [0.0, -10.0],
            "mean_speed_mph": [6.0, 5.5],
            "peak_speed_mph": [16.0, 15.0],
            "hsr_distance_yd": [120.0, 90.0],
            "sprint_distance_yd": [40.0, 25.0],
            "hsr_event_count": [4, 3],
            "sprint_event_count": [2, 1],
            "accel_event_count": [6, 4],
            "decel_event_count": [5, 4],
        }
    )
    out = coach_early_late_table(df)
    assert list(out.columns) == [
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
