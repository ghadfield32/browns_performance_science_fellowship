"""Performance-model presets for repeatable threshold choices."""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import PeakDemandConfig, SpeedBand
from .segmentation import SegmentationConfig


@dataclass(frozen=True)
class PerformanceModelPreset:
    """Single source of truth for threshold choices."""

    name: str
    rationale: str
    absolute_speed_bands: tuple[SpeedBand, ...]
    relative_band_edges: tuple[float, ...]
    peak_demand_config: PeakDemandConfig
    segmentation_config: SegmentationConfig


def preferred_performance_model() -> PerformanceModelPreset:
    """Preferred football-style external-load model for this project."""
    return PerformanceModelPreset(
        name="Balanced Football Session Model v1",
        rationale=(
            "Uses coach-readable absolute zones, moderate HSR threshold, and conservative "
            "accel/decel event cutoffs to emphasize meaningful high-intensity efforts."
        ),
        absolute_speed_bands=(
            SpeedBand("Walk", 0.0, 3.0),
            SpeedBand("Cruise", 3.0, 9.0),
            SpeedBand("Run", 9.0, 13.0),
            SpeedBand("HSR", 13.0, 16.0),
            SpeedBand("Sprint", 16.0, None),
        ),
        relative_band_edges=(0.0, 0.4, 0.6, 0.8, 0.9, 1.0),
        peak_demand_config=PeakDemandConfig(
            distance_windows_s=(30, 60, 180, 300),
            hsr_threshold_mph=13.0,
            accel_threshold_ms2=3.0,
            decel_threshold_ms2=-3.0,
        ),
        segmentation_config=SegmentationConfig(
            gap_threshold_s=0.15,
            rest_speed_threshold_mph=1.2,
            rest_min_duration_s=25.0,
            intensity_rolling_window_s=15.0,
            change_point_penalty=30.0,
            max_change_points=10,
            min_segment_duration_s=20.0,
        ),
    )

