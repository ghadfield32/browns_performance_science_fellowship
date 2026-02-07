"""Browns tracking analysis package."""

from .config import (
    BrownsTrackingConfig,
    PathSettings,
    ProjectPaths,
    clear_project_path_cache,
    default_project_config,
    default_project_paths,
    find_project_root,
    resolve_data_file,
    resolve_output_dir,
)
from .constants import (
    EXPECTED_DT_SECONDS,
    EXPECTED_SAMPLE_HZ,
    YARDS_PER_SECOND_TO_MPH,
    YARDS_PER_SECOND_SQ_TO_M_PER_SECOND_SQ,
)
from .metrics import (
    PeakDemandConfig,
    SpeedBand,
    compute_peak_demand_timeseries,
    default_absolute_speed_bands,
    peak_distance_table,
    relative_speed_bands,
    session_extrema_table,
    summarize_speed_bands,
    top_non_overlapping_windows,
)
from .pipeline import (
    compute_session_event_counts,
    load_tracking_data,
    split_early_late_summary,
    summarize_session,
)
from .presets import PerformanceModelPreset, preferred_performance_model
from .segmentation import (
    SegmentationConfig,
    build_coach_phase_summary,
    detect_segments,
    summarize_segments,
)

__all__ = [
    "ProjectPaths",
    "BrownsTrackingConfig",
    "PathSettings",
    "EXPECTED_DT_SECONDS",
    "EXPECTED_SAMPLE_HZ",
    "YARDS_PER_SECOND_TO_MPH",
    "YARDS_PER_SECOND_SQ_TO_M_PER_SECOND_SQ",
    "PeakDemandConfig",
    "PerformanceModelPreset",
    "SegmentationConfig",
    "SpeedBand",
    "clear_project_path_cache",
    "compute_peak_demand_timeseries",
    "compute_session_event_counts",
    "default_project_config",
    "default_project_paths",
    "default_absolute_speed_bands",
    "detect_segments",
    "find_project_root",
    "load_tracking_data",
    "peak_distance_table",
    "preferred_performance_model",
    "relative_speed_bands",
    "resolve_data_file",
    "resolve_output_dir",
    "session_extrema_table",
    "split_early_late_summary",
    "build_coach_phase_summary",
    "summarize_segments",
    "summarize_session",
    "summarize_speed_bands",
    "top_non_overlapping_windows",
]
