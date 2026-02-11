"""Shared analysis pipeline and results-contract utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .constants import EXPECTED_DT_SECONDS, YARDS_PER_SECOND_SQ_TO_M_PER_SECOND_SQ, YARDS_PER_SECOND_TO_MPH
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
    build_visualization_frame,
    build_qc_check_table,
    build_validation_takeaways,
    classify_hsr_exposure,
    compute_qc_status,
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
    qc_status: str
    qc_checks: pd.DataFrame
    validation_gates: pd.DataFrame
    validation_takeaways: list[str]
    absolute_band_summary: pd.DataFrame
    relative_band_summary: pd.DataFrame
    rolling: pd.DataFrame
    metrics_summary: pd.DataFrame
    distance_table: pd.DataFrame
    top_windows: pd.DataFrame
    top_windows_by_duration: pd.DataFrame
    peak_windows: pd.DataFrame
    extrema_table: pd.DataFrame
    event_counts: dict[str, float | int]
    early_late_summary: pd.DataFrame
    analysis_df: pd.DataFrame
    viz_df: pd.DataFrame
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
    sprint_threshold_mph = _infer_sprint_threshold_mph(active_model.absolute_speed_bands)

    session_summary = summarize_session(df)
    qa_summary = compute_data_quality_summary(
        df,
        max_plausible_xy_speed_yd_s=CONTINUITY_MAX_PLAUSIBLE_XY_SPEED_YD_S,
    )
    validation_gates = compute_validation_gates(
        df,
        max_plausible_xy_speed_yd_s=CONTINUITY_MAX_PLAUSIBLE_XY_SPEED_YD_S,
    )
    qc_checks = build_qc_check_table(
        df,
        qa_summary=qa_summary,
        validation_gates=validation_gates,
    )
    qc_status = compute_qc_status(qc_checks)
    qa_summary["overall_status"] = _overall_status_from_qc_status(qc_status)
    validation_takeaways = build_validation_takeaways(validation_gates)
    analysis_df = annotate_tracking_continuity(
        df,
        gap_threshold_s=float(qa_summary["gap_threshold_s"]),
        max_plausible_xy_speed_yd_s=CONTINUITY_MAX_PLAUSIBLE_XY_SPEED_YD_S,
    )

    absolute_bands = list(active_model.absolute_speed_bands)
    absolute_band_summary = summarize_speed_bands(analysis_df, absolute_bands)
    relative_bands = relative_speed_bands(
        max_speed_mph=float(analysis_df["speed_mph"].max()),
        percent_edges=active_model.relative_band_edges,
    )
    relative_band_summary = summarize_speed_bands(analysis_df, relative_bands)

    rolling = compute_peak_demand_timeseries(
        analysis_df,
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
    top_windows_by_duration_raw = build_top_windows_by_duration(
        rolling=rolling,
        windows_s=peak_cfg.distance_windows_s,
        top_n_windows=top_n_windows,
        block_col="continuous_block_id",
    )
    extrema_table = session_extrema_table(analysis_df)
    event_counts = compute_session_event_counts(
        analysis_df,
        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,
        sprint_threshold_mph=sprint_threshold_mph,
        accel_threshold_ms2=peak_cfg.accel_threshold_ms2,
        decel_threshold_ms2=peak_cfg.decel_threshold_ms2,
    )
    early_late_summary = split_early_late_summary(
        analysis_df,
        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,
        sprint_threshold_mph=sprint_threshold_mph,
        accel_threshold_ms2=peak_cfg.accel_threshold_ms2,
        decel_threshold_ms2=peak_cfg.decel_threshold_ms2,
    )
    metrics_summary = build_metrics_summary_table(
        session_summary=session_summary,
        event_counts=event_counts,
        distance_table=distance_table,
    )

    segmented_df, raw_segment_boundaries = detect_segments(analysis_df, active_model.segmentation_config)
    raw_segment_summary = summarize_segments(segmented_df, speed_bands=absolute_bands)
    coach_df, coach_phase_summary = build_coach_phase_summary(
        segmented_df,
        min_phase_duration_s=min_phase_duration_s,
        max_phases=max_phases,
        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,
        sprint_threshold_mph=sprint_threshold_mph,
        accel_threshold_ms2=peak_cfg.accel_threshold_ms2,
        decel_threshold_ms2=peak_cfg.decel_threshold_ms2,
    )
    viz_df = build_visualization_frame(
        coach_df,
        active_speed_threshold_mph=float(qa_summary.get("inactive_speed_threshold_mph", 0.5)),
        gap_threshold_s=float(qa_summary.get("gap_threshold_s", EXPECTED_DT_SECONDS * 1.5)),
        position_quantiles=(
            float(qa_summary.get("position_outlier_low_quantile", 0.01)),
            float(qa_summary.get("position_outlier_high_quantile", 0.99)),
        ),
        require_spatial_ok=True,
    )
    if viz_df.empty:
        viz_df = coach_df.copy()

    peak_windows = build_peak_window_table(
        analysis_df,
        top_windows=top_windows,
        coach_df=coach_df,
        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,
        sprint_threshold_mph=sprint_threshold_mph,
        accel_threshold_ms2=peak_cfg.accel_threshold_ms2,
        decel_threshold_ms2=peak_cfg.decel_threshold_ms2,
    )
    top_windows_by_duration = build_peak_window_table(
        analysis_df,
        top_windows=top_windows_by_duration_raw,
        coach_df=coach_df,
        hsr_threshold_mph=peak_cfg.hsr_threshold_mph,
        sprint_threshold_mph=sprint_threshold_mph,
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
        qc_status=qc_status,
        qc_checks=qc_checks,
        validation_gates=validation_gates,
        validation_takeaways=validation_takeaways,
        absolute_band_summary=absolute_band_summary,
        relative_band_summary=relative_band_summary,
        rolling=rolling,
        metrics_summary=metrics_summary,
        distance_table=distance_table,
        top_windows=top_windows,
        top_windows_by_duration=top_windows_by_duration,
        peak_windows=peak_windows,
        extrema_table=extrema_table,
        event_counts=event_counts,
        early_late_summary=early_late_summary,
        analysis_df=analysis_df,
        viz_df=viz_df,
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
    sprint_threshold_mph: float = 16.0,
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
        window_rank = int(row["window_rank"]) if "window_rank" in row and pd.notna(row["window_rank"]) else int(i + 1)
        window_s = int(row["window_s"]) if "window_s" in row and pd.notna(row["window_s"]) else int(
            (end_ts - start_ts).total_seconds()
        )
        if "distance_yd" in row and pd.notna(row["distance_yd"]):
            distance_yd = float(row["distance_yd"])
        elif "value" in row and pd.notna(row["value"]):
            distance_yd = float(row["value"])
        else:
            distance_yd = 0.0
        block_id = -1
        if "continuous_block_id" in row and pd.notna(row["continuous_block_id"]):
            block_id = int(row["continuous_block_id"])
        context = summarize_window_context(
            df,
            window_start_utc=start_ts,
            window_end_utc=end_ts,
            hsr_threshold_mph=hsr_threshold_mph,
            sprint_threshold_mph=sprint_threshold_mph,
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
                "window_rank": window_rank,
                "window_s": window_s,
                "continuous_block_id": block_id,
                "window_start_utc": start_ts,
                "window_end_utc": end_ts,
                "distance_yd": distance_yd,
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
        "coach_phase_type",
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
                "coach_phase_type": str(phase["coach_phase_type"].iloc[0]) if "coach_phase_type" in phase.columns else "",
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


def build_top_windows_by_duration(
    *,
    rolling: pd.DataFrame,
    windows_s: Sequence[int],
    top_n_windows: int = 3,
    block_col: str = "continuous_block_id",
) -> pd.DataFrame:
    """Return top-N non-overlapping windows for each configured duration."""
    frames: list[pd.DataFrame] = []
    for window_s in windows_s:
        metric_col = f"distance_{int(window_s)}s_yd"
        if metric_col not in rolling.columns:
            continue
        top_for_window = top_non_overlapping_windows(
            rolling,
            metric_column=metric_col,
            window_s=int(window_s),
            top_n=top_n_windows,
            block_col=block_col,
        )
        if top_for_window.empty:
            continue
        top_for_window = top_for_window.copy()
        top_for_window["window_rank"] = range(1, len(top_for_window) + 1)
        frames.append(top_for_window)

    if not frames:
        return pd.DataFrame(
            columns=[
                "metric",
                "window_s",
                "window_start_utc",
                "window_end_utc",
                "value",
                block_col,
                "window_rank",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def build_metrics_summary_table(
    *,
    session_summary: dict[str, float | int | str],
    event_counts: dict[str, float | int],
    distance_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact key-value metrics table for metrics-summary exports."""
    rows: list[dict[str, object]] = [
        {"metric": "total_distance_yd", "value": float(session_summary["distance_yd_from_speed"]), "unit": "yd"},
        {"metric": "mean_speed_mph", "value": float(session_summary["mean_speed_mph"]), "unit": "mph"},
        {"metric": "peak_speed_mph", "value": float(session_summary["peak_speed_mph"]), "unit": "mph"},
        {"metric": "peak_accel_ms2", "value": float(session_summary["peak_accel_ms2"]), "unit": "m/s^2"},
        {"metric": "peak_decel_ms2", "value": float(session_summary["peak_decel_ms2"]), "unit": "m/s^2"},
        {"metric": "hsr_distance_yd", "value": float(event_counts["hsr_distance_yd"]), "unit": "yd"},
        {"metric": "sprint_distance_yd", "value": float(event_counts["sprint_distance_yd"]), "unit": "yd"},
        {"metric": "hsr_event_count", "value": int(event_counts["hsr_event_count"]), "unit": "count"},
        {"metric": "sprint_event_count", "value": int(event_counts["sprint_event_count"]), "unit": "count"},
        {"metric": "accel_event_count", "value": int(event_counts["accel_event_count"]), "unit": "count"},
        {"metric": "decel_event_count", "value": int(event_counts["decel_event_count"]), "unit": "count"},
    ]

    for row in distance_table.itertuples(index=False):
        if hasattr(row, "best_intensity_yd_per_min") and pd.notna(row.best_intensity_yd_per_min):
            best_intensity = float(row.best_intensity_yd_per_min)
        else:
            best_intensity = float(row.best_distance_yd) * (60.0 / float(row.window_s))
        rows.append(
            {
                "metric": f"best_distance_{int(row.window_s)}s_yd",
                "value": float(row.best_distance_yd),
                "unit": "yd",
            }
        )
        rows.append(
            {
                "metric": f"best_intensity_{int(row.window_s)}s_yd_per_min",
                "value": best_intensity,
                "unit": "yd/min",
            }
        )

    return pd.DataFrame(rows)


def write_results_tables(
    results: SessionAnalysisResults,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write canonical contract tables and detailed support tables."""
    root = Path(output_dir)
    table_dir = root / "tables"
    qc_dir = root / "qc_summary"
    metrics_dir = root / "metrics_summary"
    coach_pkg_dir = root / "coach_package"
    table_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    coach_pkg_dir.mkdir(parents=True, exist_ok=True)

    table_map: dict[str, tuple[pd.DataFrame, Path]] = {
        "analysis_df": (results.analysis_df, table_dir / "analysis_df.csv"),
        "viz_df": (results.viz_df, table_dir / "viz_df.csv"),
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
        "top_peak_windows_by_duration": (
            results.top_windows_by_duration,
            table_dir / "top_windows_by_duration.csv",
        ),
        "peak_windows": (results.peak_windows, table_dir / "peak_windows.csv"),
        "session_extrema": (results.extrema_table, table_dir / "session_extrema.csv"),
        "session_event_counts": (
            pd.DataFrame([results.event_counts]),
            table_dir / "session_event_counts.csv",
        ),
        "metrics_summary": (results.metrics_summary, table_dir / "metrics_summary.csv"),
        "data_quality_summary": (
            pd.DataFrame([results.qa_summary]),
            table_dir / "data_quality_summary.csv",
        ),
        "qc_checks": (results.qc_checks, table_dir / "qc_checks.csv"),
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

    # Layered output packages requested by reviewers.
    qc_summary_table = qc_dir / "qc_summary.csv"
    pd.DataFrame([results.qa_summary]).to_csv(qc_summary_table, index=False)
    paths["qc_summary_table"] = qc_summary_table
    gate_lookup = results.validation_gates.set_index("gate_key") if not results.validation_gates.empty else pd.DataFrame()
    speed_corr_value = np.nan
    speed_outlier_value = np.nan
    if not gate_lookup.empty:
        if "speed_alignment_corr" in gate_lookup.index:
            corr_val = gate_lookup.loc["speed_alignment_corr", "value"]
            speed_corr_value = float(corr_val.iloc[0] if isinstance(corr_val, pd.Series) else corr_val)
        if "speed_alignment_outlier_pct" in gate_lookup.index:
            outlier_val = gate_lookup.loc["speed_alignment_outlier_pct", "value"]
            speed_outlier_value = float(outlier_val.iloc[0] if isinstance(outlier_val, pd.Series) else outlier_val)
    data_qc_summary = {
        "overall_status": results.qa_summary.get("overall_status", "UNKNOWN"),
        "qc_status": results.qc_status,
        "sample_count": int(results.qa_summary.get("sample_count", 0) or 0),
        "gap_count": int(results.qa_summary.get("gap_count", 0) or 0),
        "gap_pct": float(results.qa_summary.get("gap_pct", 0.0) or 0.0),
        "inactive_sample_pct": results.qa_summary.get("inactive_sample_pct", np.nan),
        "position_outlier_pct": results.qa_summary.get("position_outlier_pct", np.nan),
        "teleport_pct": results.qa_summary.get("teleport_pct", np.nan),
        "speed_xy_corr": results.qa_summary.get("speed_xy_corr", speed_corr_value),
        "speed_xy_outlier_pct": results.qa_summary.get("speed_xy_outlier_pct", speed_outlier_value),
    }
    data_qc_summary_json = qc_dir / "data_qc_summary.json"
    data_qc_summary_json.write_text(
        json.dumps(_jsonify_obj(data_qc_summary), indent=2) + "\n",
        encoding="utf-8",
    )
    paths["data_qc_summary_json"] = data_qc_summary_json
    qc_checks_table = qc_dir / "qc_checks.csv"
    results.qc_checks.to_csv(qc_checks_table, index=False)
    paths["qc_checks_table"] = qc_checks_table
    qc_status_file = qc_dir / "qc_status.txt"
    qc_status_file.write_text(results.qc_status + "\n", encoding="utf-8")
    paths["qc_status_file"] = qc_status_file

    metrics_summary_table = metrics_dir / "metrics_summary.csv"
    results.metrics_summary.to_csv(metrics_summary_table, index=False)
    paths["metrics_summary_table"] = metrics_summary_table
    results.distance_table.to_csv(metrics_dir / "peak_distance_windows.csv", index=False)
    results.early_late_summary.to_csv(metrics_dir / "early_vs_late_summary.csv", index=False)
    results.top_windows_by_duration.to_csv(metrics_dir / "top_windows_by_duration.csv", index=False)

    coach_phase_package = coach_pkg_dir / "coach_phases.csv"
    results.coach_phase_summary.to_csv(coach_phase_package, index=False)
    paths["coach_phase_package_table"] = coach_phase_package
    coach_peak_package = coach_pkg_dir / "coach_peak_windows.csv"
    results.peak_windows.to_csv(coach_peak_package, index=False)
    paths["coach_peak_package_table"] = coach_peak_package

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
    sprint_threshold_mph = _infer_sprint_threshold_mph(results.model.absolute_speed_bands)

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
            "sprint_threshold_mph": sprint_threshold_mph,
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
        "distance_policy": {
            "primary_source": "speed_integrated",
            "formula": "distance_yd = s (yd/s) * dt_s",
            "vendor_dis_role": "QA diagnostic only",
            "vendor_dis_alignment_pct": results.qa_summary.get("distance_dis_minus_speed_pct", np.nan),
        },
        "visualization_policy": {
            "uses_trimmed_viz_frame": True,
            "active_speed_threshold_mph": results.qa_summary.get("inactive_speed_threshold_mph", 0.5),
            "gap_threshold_s": results.qa_summary.get("gap_threshold_s", EXPECTED_DT_SECONDS * 1.5),
            "position_quantile_bounds": [
                results.qa_summary.get("position_outlier_low_quantile", 0.01),
                results.qa_summary.get("position_outlier_high_quantile", 0.99),
            ],
            "notes": "Coach visuals use viz_df trimming; workload metrics and peak logic use analysis_df.",
        },
        "session_summary": _jsonify_obj(results.session_summary),
        "qa_summary": _jsonify_obj(results.qa_summary),
        "qc_status": results.qc_status,
        "qc_failed": bool(results.qc_status == "QC FAILED"),
        "qc_checks": _jsonify_records(results.qc_checks),
        "validation_gates": _jsonify_records(results.validation_gates),
        "validation_takeaways": _jsonify_obj(results.validation_takeaways),
        "metrics_summary": _jsonify_records(results.metrics_summary),
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
            "layered_packages": {
                "qc_summary_dir": "qc_summary",
                "metrics_summary_dir": "metrics_summary",
                "coach_package_dir": "coach_package",
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


def _infer_sprint_threshold_mph(speed_bands: Sequence[Any], fallback: float = 16.0) -> float:
    """Infer sprint threshold from configured absolute speed bands."""
    for band in speed_bands:
        name = str(getattr(band, "name", "")).strip().lower()
        lower = getattr(band, "lower_mph", None)
        if name == "sprint" and lower is not None:
            return float(lower)

    for band in reversed(list(speed_bands)):
        lower = getattr(band, "lower_mph", None)
        upper = getattr(band, "upper_mph", None)
        if upper is None and lower is not None:
            return float(lower)

    return float(fallback)


def _overall_status_from_qc_status(qc_status: str) -> str:
    if qc_status == "QC PASS":
        return "PASS"
    if qc_status == "QC WARN":
        return "WARN"
    return "FAIL"
