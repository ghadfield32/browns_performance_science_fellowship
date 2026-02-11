"""Shared analysis pipeline and results-contract utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import YARDS_PER_SECOND_SQ_TO_M_PER_SECOND_SQ, YARDS_PER_SECOND_TO_MPH
from .metrics import (
    peak_distance_table,
    relative_speed_bands,
    session_extrema_table,
    summarize_speed_bands,
    top_non_overlapping_windows,
    compute_peak_demand_timeseries,
)
from .pipeline import (
    annotate_tracking_continuity,
    build_validation_takeaways,
    classify_hsr_exposure,
    compute_data_quality_summary,
    compute_session_event_counts,
    compute_validation_gates,
    split_early_late_summary,
    summarize_session,
    summarize_window_context,
)
from .presets import PerformanceModelPreset, preferred_performance_model
from .segmentation import build_coach_phase_summary, detect_segments, summarize_segments

CONTINUITY_MAX_PLAUSIBLE_XY_SPEED_YD_S = 14.0
ROLLING_MIN_COVERAGE_FRACTION = 0.95


@dataclass(frozen=True)
class SessionAnalysisResults:
    """Container for one-pass session analysis outputs."""

    model: PerformanceModelPreset
    session_summary: dict[str, float | int | str]
    qa_summary: dict[str, float | int | str]
    validation_gates: pd.DataFrame
    validation_takeaways: list[str]
    absolute_band_summary: pd.DataFrame
    relative_band_summary: pd.DataFrame
    rolling: pd.DataFrame
    distance_table: pd.DataFrame
    top_windows: pd.DataFrame
    peak_windows: pd.DataFrame
    extrema_table: pd.DataFrame
    event_counts: dict[str, float | int]
    early_late_summary: pd.DataFrame
    segmented_df: pd.DataFrame
    raw_segment_boundaries: pd.DataFrame
    raw_segment_summary: pd.DataFrame
    coach_df: pd.DataFrame
    coach_phase_summary: pd.DataFrame
    structure_map_summary: pd.DataFrame


def run_session_analysis(
    df: pd.DataFrame,
    *,
    model: PerformanceModelPreset | None = None,
    top_window_s: int = 60,
    top_n_windows: int = 3,
    min_phase_duration_s: float = 30.0,
    max_phases: int = 8,
) -> SessionAnalysisResults:
    """Run the full analysis stack once and return reusable outputs."""
    active_model = model or preferred_performance_model()
    peak_cfg = active_model.peak_demand_config

    session_summary = summarize_session(df)
    qa_summary = compute_data_quality_summary(df)
    validation_gates = compute_validation_gates(
        df,
        max_plausible_xy_speed_yd_s=CONTINUITY_MAX_PLAUSIBLE_XY_SPEED_YD_S,
    )
    validation_takeaways = build_validation_takeaways(validation_gates)
    continuity_df = annotate_tracking_continuity(
        df,
        gap_threshold_s=float(qa_summary["gap_threshold_s"]),
        max_plausible_xy_speed_yd_s=CONTINUITY_MAX_PLAUSIBLE_XY_SPEED_YD_S,
    )

    absolute_bands = list(active_model.absolute_speed_bands)
    absolute_band_summary = summarize_speed_bands(continuity_df, absolute_bands)
    relative_bands = relative_speed_bands(
        max_speed_mph=float(continuity_df["speed_mph"].max()),
        percent_edges=active_model.relative_band_edges,
    )
    relative_band_summary = summarize_speed_bands(continuity_df, relative_bands)

    rolling = compute_peak_demand_timeseries(
        continuity_df,
        peak_cfg,
        block_col="continuous_block_id",
        min_window_coverage_fraction=ROLLING_MIN_COVERAGE_FRACTION,
    )
    distance_table = peak_distance_table(rolling, peak_cfg.distance_windows_s)

    requested_col = f"distance_{top_window_s}s_yd"
    if requested_col in rolling.columns:
        selected_window_s = top_window_s
    else:
        selected_window_s = int(peak_cfg.distance_windows_s[0])
        requested_col = f"distance_{selected_window_s}s_yd"

    top_windows = top_non_overlapping_windows(
        rolling,
        metric_column=requested_col,
        window_s=selected_window_s,
        top_n=top_n_windows,
        block_col="continuous_block_id",
    )
    extrema_table = session_extrema_table(continuity_df)
    event_counts = compute_session_event_counts(
        continuity_df,
        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,
        accel_threshold_ms2=peak_cfg.accel_threshold_ms2,
        decel_threshold_ms2=peak_cfg.decel_threshold_ms2,
    )
    early_late_summary = split_early_late_summary(
        continuity_df,
        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,
        accel_threshold_ms2=peak_cfg.accel_threshold_ms2,
        decel_threshold_ms2=peak_cfg.decel_threshold_ms2,
    )

    segmented_df, raw_segment_boundaries = detect_segments(continuity_df, active_model.segmentation_config)
    raw_segment_summary = summarize_segments(segmented_df, speed_bands=absolute_bands)
    coach_df, coach_phase_summary = build_coach_phase_summary(
        segmented_df,
        min_phase_duration_s=min_phase_duration_s,
        max_phases=max_phases,
        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,
        accel_threshold_ms2=peak_cfg.accel_threshold_ms2,
        decel_threshold_ms2=peak_cfg.decel_threshold_ms2,
    )

    peak_windows = build_peak_window_table(
        continuity_df,
        top_windows=top_windows,
        coach_df=coach_df,
        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,
        accel_threshold_ms2=peak_cfg.accel_threshold_ms2,
        decel_threshold_ms2=peak_cfg.decel_threshold_ms2,
    )
    structure_map_summary = build_session_structure_map_table(
        coach_df,
        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,
    )

    return SessionAnalysisResults(
        model=active_model,
        session_summary=session_summary,
        qa_summary=qa_summary,
        validation_gates=validation_gates,
        validation_takeaways=validation_takeaways,
        absolute_band_summary=absolute_band_summary,
        relative_band_summary=relative_band_summary,
        rolling=rolling,
        distance_table=distance_table,
        top_windows=top_windows,
        peak_windows=peak_windows,
        extrema_table=extrema_table,
        event_counts=event_counts,
        early_late_summary=early_late_summary,
        segmented_df=segmented_df,
        raw_segment_boundaries=raw_segment_boundaries,
        raw_segment_summary=raw_segment_summary,
        coach_df=coach_df,
        coach_phase_summary=coach_phase_summary,
        structure_map_summary=structure_map_summary,
    )


def build_peak_window_table(
    df: pd.DataFrame,
    *,
    top_windows: pd.DataFrame,
    coach_df: pd.DataFrame | None = None,
    hsr_threshold_mph: float = 13.0,
    accel_threshold_ms2: float = 3.0,
    decel_threshold_ms2: float = -3.0,
) -> pd.DataFrame:
    """Build an actionable top-window table with event context and dominant phase."""
    if top_windows.empty:
        return pd.DataFrame(
            columns=[
                "window_rank",
                "window_s",
                "continuous_block_id",
                "window_start_utc",
                "window_end_utc",
                "distance_yd",
                "dominant_phase",
                "hsr_event_count",
                "sprint_event_count",
                "accel_event_count",
                "decel_event_count",
                "hsr_exposure_level",
            ]
        )

    rows: list[dict[str, Any]] = []
    for i, row in top_windows.reset_index(drop=True).iterrows():
        start_ts = pd.to_datetime(row["window_start_utc"], utc=True)
        end_ts = pd.to_datetime(row["window_end_utc"], utc=True)
        block_id = -1
        if "continuous_block_id" in row and pd.notna(row["continuous_block_id"]):
            block_id = int(row["continuous_block_id"])
        context = summarize_window_context(
            df,
            window_start_utc=start_ts,
            window_end_utc=end_ts,
            hsr_threshold_mph=hsr_threshold_mph,
            accel_threshold_ms2=accel_threshold_ms2,
            decel_threshold_ms2=decel_threshold_ms2,
        )
        dominant_phase = ""
        if coach_df is not None and "coach_phase_label" in coach_df.columns:
            phase_slice = coach_df[(coach_df["ts"] >= start_ts) & (coach_df["ts"] <= end_ts)]
            if not phase_slice.empty:
                dominant_phase = str(phase_slice["coach_phase_label"].value_counts().index[0])

        rows.append(
            {
                "window_rank": int(i + 1),
                "window_s": int(row["window_s"]),
                "continuous_block_id": block_id,
                "window_start_utc": start_ts,
                "window_end_utc": end_ts,
                "distance_yd": float(row["value"]),
                "dominant_phase": dominant_phase,
                "hsr_event_count": int(context["hsr_event_count"]),
                "sprint_event_count": int(context["sprint_event_count"]),
                "accel_event_count": int(context["accel_event_count"]),
                "decel_event_count": int(context["decel_event_count"]),
                "hsr_exposure_level": str(context["hsr_exposure_level"]),
            }
        )

    return pd.DataFrame(rows)


def build_session_structure_map_table(
    coach_df: pd.DataFrame,
    *,
    hsr_threshold_mph: float = 13.0,
) -> pd.DataFrame:
    """Build one-row-per-phase drill taxonomy features for a structure scatter."""
    columns = [
        "coach_phase_id",
        "coach_phase_name",
        "coach_phase_label",
        "intensity_level",
        "dominant_field_zone",
        "duration_min",
        "distance_yd",
        "distance_per_min_yd",
        "hsr_distance_yd",
        "hsr_distance_per_min_yd",
        "hsr_time_pct",
        "mean_speed_mph",
        "peak_speed_mph",
    ]
    if coach_df.empty or "coach_phase_id" not in coach_df.columns:
        return pd.DataFrame(columns=columns)

    if "x" in coach_df.columns and coach_df["x"].notna().any():
        x_min = float(coach_df["x"].quantile(0.02))
        x_max = float(coach_df["x"].quantile(0.98))
    else:
        x_min, x_max = 0.0, 1.0

    rows: list[dict[str, Any]] = []
    for phase_id, phase in coach_df.groupby("coach_phase_id", sort=True):
        phase_id_int = int(phase_id)
        if phase_id_int < 0:
            continue

        duration_s = float(phase["dt_s"].sum())
        duration_min = duration_s / 60.0
        if duration_min <= 0:
            continue

        hsr_mask = phase["speed_mph"] >= hsr_threshold_mph
        distance_yd = float(phase["step_distance_yd_from_speed"].sum())
        hsr_distance_yd = float(phase.loc[hsr_mask, "step_distance_yd_from_speed"].sum())
        hsr_time_pct = float((phase.loc[hsr_mask, "dt_s"].sum() / duration_s) * 100.0) if duration_s > 0 else 0.0

        rows.append(
            {
                "coach_phase_id": phase_id_int,
                "coach_phase_name": str(phase["coach_phase_name"].iloc[0]),
                "coach_phase_label": str(phase["coach_phase_label"].iloc[0]),
                "intensity_level": str(phase["coach_intensity_level"].iloc[0]),
                "dominant_field_zone": _dominant_field_zone(phase.get("x", pd.Series(dtype=float)), x_min, x_max),
                "duration_min": duration_min,
                "distance_yd": distance_yd,
                "distance_per_min_yd": distance_yd / duration_min,
                "hsr_distance_yd": hsr_distance_yd,
                "hsr_distance_per_min_yd": hsr_distance_yd / duration_min,
                "hsr_time_pct": hsr_time_pct,
                "mean_speed_mph": float(phase["speed_mph"].mean()),
                "peak_speed_mph": float(phase["speed_mph"].max()),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("coach_phase_id").reset_index(drop=True)


def write_results_tables(
    results: SessionAnalysisResults,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write canonical contract tables and detailed support tables."""
    root = Path(output_dir)
    table_dir = root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    table_map: dict[str, tuple[pd.DataFrame, Path]] = {
        "absolute_speed_band_summary": (
            results.absolute_band_summary,
            table_dir / "absolute_speed_band_summary.csv",
        ),
        "relative_speed_band_summary": (
            results.relative_band_summary,
            table_dir / "relative_speed_band_summary.csv",
        ),
        "peak_distance_windows": (results.distance_table, table_dir / "peak_distance_windows.csv"),
        "top_peak_windows": (results.top_windows, table_dir / "top_1m_distance_windows.csv"),
        "peak_windows": (results.peak_windows, table_dir / "peak_windows.csv"),
        "session_extrema": (results.extrema_table, table_dir / "session_extrema.csv"),
        "session_event_counts": (
            pd.DataFrame([results.event_counts]),
            table_dir / "session_event_counts.csv",
        ),
        "data_quality_summary": (
            pd.DataFrame([results.qa_summary]),
            table_dir / "data_quality_summary.csv",
        ),
        "validation_gates": (results.validation_gates, table_dir / "validation_gates.csv"),
        "early_vs_late_summary": (
            results.early_late_summary,
            table_dir / "early_vs_late_summary.csv",
        ),
        "raw_segment_boundaries": (
            results.raw_segment_boundaries,
            table_dir / "raw_segment_boundaries.csv",
        ),
        "raw_segment_summary": (results.raw_segment_summary, table_dir / "raw_segment_summary.csv"),
        "segment_boundaries": (
            results.raw_segment_boundaries,
            table_dir / "segment_boundaries.csv",
        ),
        "segment_summary": (results.raw_segment_summary, table_dir / "segment_summary.csv"),
        "coach_phase_summary": (results.coach_phase_summary, table_dir / "coach_phase_summary.csv"),
        "session_structure_map": (
            results.structure_map_summary,
            table_dir / "session_structure_map.csv",
        ),
    }

    paths: dict[str, Path] = {}
    for key, (frame, path) in table_map.items():
        frame.to_csv(path, index=False)
        paths[key] = path

    phase_table = root / "phase_table.csv"
    peak_windows = root / "peak_windows.csv"
    structure_map = root / "session_structure_map.csv"
    results.coach_phase_summary.to_csv(phase_table, index=False)
    results.peak_windows.to_csv(peak_windows, index=False)
    results.structure_map_summary.to_csv(structure_map, index=False)
    paths["phase_table"] = phase_table
    paths["peak_windows_root"] = peak_windows
    paths["session_structure_map_root"] = structure_map

    return paths


def write_results_contract(
    results: SessionAnalysisResults,
    *,
    input_path: str | Path,
    output_dir: str | Path,
    table_paths: dict[str, Path] | None = None,
    figure_paths: dict[str, Path] | None = None,
) -> Path:
    """Write `results.json` with assumptions, QA gates, and artifact manifest."""
    root = Path(output_dir)
    contract_path = root / "results.json"

    speed_bands = [
        {
            "name": band.name,
            "lower_mph": float(band.lower_mph),
            "upper_mph": None if band.upper_mph is None else float(band.upper_mph),
        }
        for band in results.model.absolute_speed_bands
    ]
    peak_cfg = results.model.peak_demand_config

    contract: dict[str, Any] = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "input_path": str(Path(input_path)),
        "output_dir": str(root),
        "story_questions": [
            "Where did the player spend time (role/space)?",
            "When did intensity happen (session structure and peaks)?",
            "What were the peak demands and how repeatable were they?",
        ],
        "model": {
            "name": results.model.name,
            "rationale": results.model.rationale,
        },
        "units": {
            "speed_conversion": {
                "from": "yards/second",
                "to": "mph",
                "factor": YARDS_PER_SECOND_TO_MPH,
            },
            "acceleration_conversion": {
                "from": "yards/second^2",
                "to": "m/second^2",
                "factor": YARDS_PER_SECOND_SQ_TO_M_PER_SECOND_SQ,
            },
            "acceleration_channels": {
                "a_raw": "Non-negative acceleration magnitude from feed (yd/s^2).",
                "sa_raw": "Signed acceleration from feed (yd/s^2).",
                "analysis_source": "All accel/decel events and extrema use signed `sa` converted to m/s^2.",
            },
        },
        "thresholds": {
            "speed_bands_mph": speed_bands,
            "hsr_threshold_mph": float(peak_cfg.hsr_threshold_mph),
            "sprint_threshold_mph": 16.0,
            "accel_threshold_ms2": float(peak_cfg.accel_threshold_ms2),
            "decel_threshold_ms2": float(peak_cfg.decel_threshold_ms2),
            "speed_event_min_duration_s": 1.0,
            "coach_max_phases": 8,
            "max_plausible_xy_speed_yd_s": CONTINUITY_MAX_PLAUSIBLE_XY_SPEED_YD_S,
            "rolling_window_min_coverage_fraction": ROLLING_MIN_COVERAGE_FRACTION,
        },
        "continuity_policy": {
            "gap_threshold_s": float(results.qa_summary.get("gap_threshold_s", 0.0)),
            "gap_break_column": "continuity_break_before",
            "continuous_block_column": "continuous_block_id",
            "notes": (
                "Rolling windows and event runs reset at flagged gaps/jumps; movement lines break at the same boundaries."
            ),
        },
        "session_summary": _jsonify_obj(results.session_summary),
        "qa_summary": _jsonify_obj(results.qa_summary),
        "validation_gates": _jsonify_records(results.validation_gates),
        "validation_takeaways": _jsonify_obj(results.validation_takeaways),
        "event_counts": _jsonify_obj(results.event_counts),
        "hsr_exposure_level": classify_hsr_exposure(
            total_distance_yd=float(results.session_summary["distance_yd_from_speed"]),
            hsr_distance_yd=float(results.event_counts["hsr_distance_yd"]),
        ),
        "artifacts": {
            "tables": {
                key: str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
                for key, path in (table_paths or {}).items()
            },
            "figures": {
                key: str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
                for key, path in (figure_paths or {}).items()
            },
        },
    }

    contract_path.write_text(json.dumps(_jsonify_obj(contract), indent=2) + "\n", encoding="utf-8")
    return contract_path


def load_results_contract(output_dir: str | Path) -> dict[str, Any]:
    """Load `results.json` from an output directory."""
    contract_path = Path(output_dir) / "results.json"
    if not contract_path.exists():
        raise FileNotFoundError(
            f"Missing results contract at {contract_path}. Run render_visual_templates.py first."
        )
    return json.loads(contract_path.read_text(encoding="utf-8"))


def _jsonify_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_jsonify_obj(row) for row in frame.to_dict(orient="records")]


def _jsonify_obj(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify_obj(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify_obj(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _dominant_field_zone(x: pd.Series, x_min: float, x_max: float) -> str:
    if x.empty:
        return "Unknown"
    x_num = pd.to_numeric(x, errors="coerce").dropna()
    if x_num.empty:
        return "Unknown"

    span = max(float(x_max - x_min), 1e-9)
    norm = (x_num - x_min) / span
    zones = pd.cut(
        norm,
        bins=[-np.inf, 1.0 / 3.0, 2.0 / 3.0, np.inf],
        labels=["Back Third", "Middle Third", "Front Third"],
    )
    if zones.empty:
        return "Unknown"
    return str(zones.value_counts().index[0])
