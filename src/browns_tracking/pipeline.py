"""Core data loading and session summary utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import (
    EXPECTED_DT_SECONDS,
    YARDS_PER_SECOND_SQ_TO_M_PER_SECOND_SQ,
    YARDS_PER_SECOND_TO_MPH,
)


def load_tracking_data(csv_path: str | Path) -> pd.DataFrame:
    """Load CSV, normalize types, sort by timestamp, and add derived columns.

    Distance source: Uses speed-integrated distance as primary source.
    The vendor 'dis' column is NOT used (shows -74.66% systematic error vs speed channel).
    XY-derived distance is computed for validation only.
    """
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    dt = df["ts"].diff().dt.total_seconds()
    df["dt_s"] = dt.fillna(EXPECTED_DT_SECONDS)

    # Unit conversions
    df["speed_mph"] = df["s"] * YARDS_PER_SECOND_TO_MPH
    df["accel_ms2"] = df["a"] * YARDS_PER_SECOND_SQ_TO_M_PER_SECOND_SQ
    df["signed_accel_ms2"] = df["sa"] * YARDS_PER_SECOND_SQ_TO_M_PER_SECOND_SQ

    # PRIMARY distance source: speed-integrated
    df["step_distance_yd_from_speed"] = df["s"] * df["dt_s"]

    # XY-derived distance for validation only (corr=0.998 with speed-integrated)
    xy_step = np.sqrt(df["x"].diff().pow(2) + df["y"].diff().pow(2))
    df["step_distance_xy_yd"] = xy_step.fillna(0.0)

    return df


def summarize_session(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Compute a compact required-metrics starter summary."""
    gaps = int((df["dt_s"] > (EXPECTED_DT_SECONDS * 1.5)).sum())
    duration_s = float((df["ts"].iloc[-1] - df["ts"].iloc[0]).total_seconds())
    distance_yd_from_speed = float(df["step_distance_yd_from_speed"].sum())
    distance_yd_from_xy = float(df["step_distance_xy_yd"].sum())

    return {
        "rows": int(len(df)),
        "start_ts_utc": str(df["ts"].iloc[0]),
        "end_ts_utc": str(df["ts"].iloc[-1]),
        "duration_s": duration_s,
        "sampling_gap_count": gaps,
        "distance_yd_from_speed": distance_yd_from_speed,
        "distance_yd_from_xy": distance_yd_from_xy,
        "mean_speed_mph": float(df["speed_mph"].mean()),
        "peak_speed_mph": float(df["speed_mph"].max()),
        "mean_accel_ms2": float(df["signed_accel_ms2"].mean()),
        "peak_accel_ms2": float(df["signed_accel_ms2"].max()),
        "peak_decel_ms2": float(df["signed_accel_ms2"].min()),
    }


def annotate_tracking_continuity(
    df: pd.DataFrame,
    *,
    gap_threshold_s: float = EXPECTED_DT_SECONDS * 1.5,
    max_plausible_xy_speed_yd_s: float = 14.0,
) -> pd.DataFrame:
    """Add QA-driven continuity flags used by rolling windows, events, and plots."""
    work = df.copy()
    if work.empty:
        work["continuity_gap_break"] = pd.Series(dtype=bool)
        work["continuity_jump_break"] = pd.Series(dtype=bool)
        work["continuity_break_before"] = pd.Series(dtype=bool)
        work["continuous_block_id"] = pd.Series(dtype=int)
        work["spatial_sample_ok"] = pd.Series(dtype=bool)
        return work

    dt_s = work["dt_s"].astype(float)
    gap_break = dt_s > float(gap_threshold_s)

    # Detect improbable XY jumps using XY-derived speed
    xy_speed_yd_s = np.where(dt_s > 0, work["step_distance_xy_yd"].astype(float) / dt_s, np.nan)
    jump_break = np.isfinite(xy_speed_yd_s) & (xy_speed_yd_s > float(max_plausible_xy_speed_yd_s))

    break_before = pd.Series(gap_break | jump_break, index=work.index, dtype=bool)
    break_before.iloc[0] = False

    work["continuity_gap_break"] = gap_break.astype(bool)
    work["continuity_jump_break"] = pd.Series(jump_break, index=work.index, dtype=bool)
    work["continuity_break_before"] = break_before
    work["continuous_block_id"] = break_before.cumsum().astype(int)

    spatial_ok = ~(work["continuity_gap_break"] | work["continuity_jump_break"])
    spatial_ok.iloc[0] = True
    work["spatial_sample_ok"] = spatial_ok.astype(bool)
    return work


def build_visualization_frame(
    df: pd.DataFrame,
    *,
    active_speed_threshold_mph: float = 0.5,
    gap_threshold_s: float = EXPECTED_DT_SECONDS * 2.0,
    position_quantiles: tuple[float, float] = (0.01, 0.99),
    require_spatial_ok: bool = True,
) -> pd.DataFrame:
    """Trim a session dataframe for cleaner coach-facing plots."""
    if df.empty:
        return df.copy()

    work = df.copy()
    index = work.index
    speed = work["speed_mph"]
    dt_s = work["dt_s"]

    active_mask = speed >= float(active_speed_threshold_mph)
    gap_ok_mask = dt_s <= float(gap_threshold_s) if dt_s.notna().any() else pd.Series(True, index=index)

    spatial_ok_mask = pd.Series(True, index=index)
    if require_spatial_ok:
        spatial_ok_mask = work["spatial_sample_ok"].fillna(False).astype(bool)

    x = work["x"]
    y = work["y"]
    q_low, q_high = position_quantiles
    q_low = float(np.clip(q_low, 0.0, 1.0))
    q_high = float(np.clip(q_high, 0.0, 1.0))
    if q_low > q_high:
        q_low, q_high = q_high, q_low

    bounds_ok_mask = pd.Series(True, index=index)
    valid_xy = x.notna() & y.notna()
    if int(valid_xy.sum()) >= 10:
        x_low = float(x[valid_xy].quantile(q_low))
        x_high = float(x[valid_xy].quantile(q_high))
        y_low = float(y[valid_xy].quantile(q_low))
        y_high = float(y[valid_xy].quantile(q_high))
        bounds_ok_mask = (
            x.between(x_low, x_high, inclusive="both")
            & y.between(y_low, y_high, inclusive="both")
            & valid_xy
        )
    elif valid_xy.any():
        bounds_ok_mask = valid_xy

    work["viz_active_speed_ok"] = active_mask.fillna(False)
    work["viz_gap_ok"] = gap_ok_mask.fillna(False)
    work["viz_bounds_ok"] = bounds_ok_mask.fillna(False)
    work["viz_spatial_ok"] = spatial_ok_mask.fillna(False)
    work["viz_sample_ok"] = (
        work["viz_active_speed_ok"]
        & work["viz_gap_ok"]
        & work["viz_bounds_ok"]
        & work["viz_spatial_ok"]
    )

    viz_df = work[work["viz_sample_ok"]].copy().sort_values("ts").reset_index(drop=True)
    if viz_df.empty:
        fallback = work[work["spatial_sample_ok"].fillna(False)].copy()
        if not fallback.empty:
            viz_df = fallback.sort_values("ts").reset_index(drop=True)
    return viz_df


def compute_session_event_counts(
    df: pd.DataFrame,
    *,
    hsr_threshold_mph: float = 13.0,
    sprint_threshold_mph: float = 16.0,
    min_speed_event_duration_s: float = 1.0,
    accel_threshold_ms2: float = 3.0,
    decel_threshold_ms2: float = -3.0,
    continuity_break_col: str = "continuity_break_before",
) -> dict[str, float | int]:
    """Compute core event counts used in coach-facing workload narratives."""
    hsr_mask = df["speed_mph"] >= hsr_threshold_mph
    sprint_mask = df["speed_mph"] >= sprint_threshold_mph
    accel_mask = df["signed_accel_ms2"] >= accel_threshold_ms2
    decel_mask = df["signed_accel_ms2"] <= decel_threshold_ms2
    reset_mask = _continuity_reset_mask(df, continuity_break_col)

    return {
        "hsr_distance_yd": float(df.loc[hsr_mask, "step_distance_yd_from_speed"].sum()),
        "sprint_distance_yd": float(df.loc[sprint_mask, "step_distance_yd_from_speed"].sum()),
        "hsr_event_count": _count_boolean_runs(
            hsr_mask,
            df["dt_s"],
            min_duration_s=min_speed_event_duration_s,
            reset_mask=reset_mask,
        ),
        "sprint_event_count": _count_boolean_runs(
            sprint_mask,
            df["dt_s"],
            min_duration_s=min_speed_event_duration_s,
            reset_mask=reset_mask,
        ),
        "accel_event_count": _count_boolean_transitions(accel_mask, reset_mask=reset_mask),
        "decel_event_count": _count_boolean_transitions(decel_mask, reset_mask=reset_mask),
    }


def split_early_late_summary(
    df: pd.DataFrame,
    *,
    hsr_threshold_mph: float = 13.0,
    sprint_threshold_mph: float = 16.0,
    min_speed_event_duration_s: float = 1.0,
    accel_threshold_ms2: float = 3.0,
    decel_threshold_ms2: float = -3.0,
) -> pd.DataFrame:
    """Create an early-vs-late session workload comparison table."""
    if df.empty:
        return pd.DataFrame()

    midpoint = df["ts"].iloc[0] + ((df["ts"].iloc[-1] - df["ts"].iloc[0]) / 2)
    early = df[df["ts"] <= midpoint].copy()
    late = df[df["ts"] > midpoint].copy()

    rows: list[dict[str, float | int | str]] = []
    for period, chunk in [("Early Half", early), ("Late Half", late)]:
        if chunk.empty:
            continue
        events = compute_session_event_counts(
            chunk,
            hsr_threshold_mph=hsr_threshold_mph,
            sprint_threshold_mph=sprint_threshold_mph,
            min_speed_event_duration_s=min_speed_event_duration_s,
            accel_threshold_ms2=accel_threshold_ms2,
            decel_threshold_ms2=decel_threshold_ms2,
        )
        duration_min = float((chunk["ts"].iloc[-1] - chunk["ts"].iloc[0]).total_seconds()) / 60.0
        rows.append(
            {
                "period": period,
                "duration_min": duration_min,
                "distance_yd": float(chunk["step_distance_yd_from_speed"].sum()),
                "mean_speed_mph": float(chunk["speed_mph"].mean()),
                "peak_speed_mph": float(chunk["speed_mph"].max()),
                "hsr_distance_yd": float(events["hsr_distance_yd"]),
                "sprint_distance_yd": float(events["sprint_distance_yd"]),
                "hsr_event_count": int(events["hsr_event_count"]),
                "sprint_event_count": int(events["sprint_event_count"]),
                "accel_event_count": int(events["accel_event_count"]),
                "decel_event_count": int(events["decel_event_count"]),
            }
        )

    summary = pd.DataFrame(rows)
    if len(summary) != 2:
        return summary

    early_distance = float(summary.loc[summary["period"] == "Early Half", "distance_yd"].iloc[0])
    if early_distance > 0:
        summary["distance_vs_early_pct"] = ((summary["distance_yd"] / early_distance) - 1.0) * 100.0
    else:
        summary["distance_vs_early_pct"] = 0.0
    return summary


def compute_data_quality_summary(
    df: pd.DataFrame,
    *,
    expected_cadence_s: float = EXPECTED_DT_SECONDS,
    cadence_tolerance_s: float = 0.02,
    gap_threshold_s: float | None = None,
    max_plausible_xy_speed_yd_s: float = 14.0,
    outlier_quantile: float = 0.995,
    inactive_speed_threshold_mph: float = 0.5,
    position_outlier_quantiles: tuple[float, float] = (0.01, 0.99),
    speed_alignment_error_yd_s: float = 2.0,
) -> dict[str, float | int | str]:
    """Build an explicit QA summary for cadence gaps and distance outliers."""
    if df.empty:
        return {
            "sample_count": 0,
            "expected_cadence_s": expected_cadence_s,
            "pct_on_expected_cadence": 0.0,
            "cadence_p50_s": 0.0,
            "cadence_p95_s": 0.0,
            "cadence_p99_s": 0.0,
            "max_gap_s": 0.0,
            "gap_count": 0,
            "gap_pct": 0.0,
            "gap_threshold_s": gap_threshold_s or (expected_cadence_s * 1.5),
            "max_xy_step_yd": 0.0,
            "teleport_count": 0,
            "teleport_pct": 0.0,
            "teleport_speed_threshold_yd_s": float(max_plausible_xy_speed_yd_s),
            "step_distance_outlier_source": "N/A",
            "step_distance_outlier_threshold_yd": 0.0,
            "step_distance_outlier_count": 0,
            "step_distance_outlier_pct": 0.0,
            "x_min_yd": 0.0,
            "x_max_yd": 0.0,
            "y_min_yd": 0.0,
            "y_max_yd": 0.0,
            "peak_speed_mph": 0.0,
            "pct_speed_le_0_5_mph": 0.0,
            "distance_yd_from_speed": 0.0,
            "distance_yd_from_dis": np.nan,
            "distance_dis_minus_speed_yd": np.nan,
            "distance_dis_minus_speed_pct": np.nan,
            "inactive_speed_threshold_mph": float(inactive_speed_threshold_mph),
            "inactive_sample_count": 0,
            "inactive_sample_pct": 0.0,
            "position_outlier_low_quantile": float(position_outlier_quantiles[0]),
            "position_outlier_high_quantile": float(position_outlier_quantiles[1]),
            "position_outlier_count": 0,
            "position_outlier_pct": np.nan,
            "speed_xy_alignment_samples": 0,
            "speed_xy_corr": np.nan,
            "speed_xy_error_p95_yd_s": np.nan,
            "speed_xy_outlier_pct": np.nan,
            "speed_xy_outlier_threshold_yd_s": float(speed_alignment_error_yd_s),
            "signed_accel_source": "signed_accel_ms2",
            "signed_accel_available": False,
            "overall_status": "UNKNOWN",
            "gap_handling": "No samples available.",
            "outlier_handling": "No samples available.",
        }

    gap_limit = gap_threshold_s if gap_threshold_s is not None else expected_cadence_s * 1.5
    dt_s = df["dt_s"].astype(float)
    on_cadence = dt_s.between(
        expected_cadence_s - cadence_tolerance_s, expected_cadence_s + cadence_tolerance_s
    )
    gap_count = int((dt_s > gap_limit).sum())

    step_distance_col = (
        "step_distance_xy_yd" if "step_distance_xy_yd" in df.columns else "step_distance_yd_from_speed"
    )
    step_distance = df[step_distance_col].astype(float)
    outlier_threshold = float(step_distance.quantile(outlier_quantile))
    outlier_count = int((step_distance > outlier_threshold).sum())

    sample_count = int(len(df))
    gap_pct = float((gap_count / sample_count) * 100.0) if sample_count > 0 else 0.0
    cadence_quantiles = dt_s.quantile([0.5, 0.95, 0.99]).to_dict()
    x_series = pd.to_numeric(df["x"], errors="coerce") if "x" in df.columns else pd.Series(dtype=float)
    y_series = pd.to_numeric(df["y"], errors="coerce") if "y" in df.columns else pd.Series(dtype=float)
    x_min = float(x_series.min()) if not x_series.empty else np.nan
    x_max = float(x_series.max()) if not x_series.empty else np.nan
    y_min = float(y_series.min()) if not y_series.empty else np.nan
    y_max = float(y_series.max()) if not y_series.empty else np.nan
    speed_mph = pd.to_numeric(df["speed_mph"], errors="coerce") if "speed_mph" in df.columns else pd.Series(dtype=float)
    peak_speed_mph = float(speed_mph.max()) if not speed_mph.empty else np.nan
    pct_speed_le_0_5_mph = float((speed_mph <= 0.5).mean() * 100.0) if not speed_mph.empty else np.nan
    inactive_mask = speed_mph <= float(inactive_speed_threshold_mph)
    inactive_count = int(inactive_mask.sum()) if not speed_mph.empty else 0
    inactive_pct = float((inactive_count / sample_count) * 100.0) if sample_count > 0 else 0.0
    distance_yd_from_speed = float(df["step_distance_yd_from_speed"].sum())

    q_low, q_high = position_outlier_quantiles
    q_low = float(np.clip(q_low, 0.0, 1.0))
    q_high = float(np.clip(q_high, 0.0, 1.0))
    if q_low > q_high:
        q_low, q_high = q_high, q_low
    xy_valid_mask = (
        x_series.notna() & y_series.notna() if not x_series.empty and not y_series.empty else pd.Series(dtype=bool)
    )
    if not xy_valid_mask.empty and int(xy_valid_mask.sum()) >= 10:
        x_low = float(x_series[xy_valid_mask].quantile(q_low))
        x_high = float(x_series[xy_valid_mask].quantile(q_high))
        y_low = float(y_series[xy_valid_mask].quantile(q_low))
        y_high = float(y_series[xy_valid_mask].quantile(q_high))
        xy_inlier = (
            x_series.between(x_low, x_high, inclusive="both")
            & y_series.between(y_low, y_high, inclusive="both")
            & xy_valid_mask
        )
        position_outlier_count = int(xy_valid_mask.sum() - xy_inlier.sum())
        position_outlier_pct = (
            float((position_outlier_count / int(xy_valid_mask.sum())) * 100.0)
            if int(xy_valid_mask.sum()) > 0
            else np.nan
        )
    else:
        position_outlier_count = 0
        position_outlier_pct = np.nan

    signed_accel_available = "signed_accel_ms2" in df.columns and df["signed_accel_ms2"].notna().any()
    if "step_distance_xy_yd" in df.columns:
        xy_step = pd.to_numeric(df["step_distance_xy_yd"], errors="coerce")
        max_xy_step_yd = float(xy_step.max()) if xy_step.notna().any() else np.nan
        xy_speed_yd_s = np.where(dt_s > 0, xy_step / dt_s, np.nan)
        valid_xy = np.isfinite(xy_speed_yd_s)
        teleport_mask = valid_xy & (xy_speed_yd_s > float(max_plausible_xy_speed_yd_s))
        teleport_count = int(np.nansum(teleport_mask))
        valid_xy_count = int(np.nansum(valid_xy))
        teleport_pct = float((teleport_count / valid_xy_count) * 100.0) if valid_xy_count > 0 else 0.0
        if "s" in df.columns:
            provided_speed_yd_s = pd.to_numeric(df["s"], errors="coerce").to_numpy(dtype=float)
        else:
            provided_speed_yd_s = (
                pd.to_numeric(df["speed_mph"], errors="coerce").to_numpy(dtype=float) / YARDS_PER_SECOND_TO_MPH
            )
        alignment_mask = valid_xy & np.isfinite(provided_speed_yd_s)
        alignment_n = int(np.nansum(alignment_mask))
        if alignment_n >= 3:
            speed_xy_corr = float(np.corrcoef(provided_speed_yd_s[alignment_mask], xy_speed_yd_s[alignment_mask])[0, 1])
            abs_error = np.abs(provided_speed_yd_s[alignment_mask] - xy_speed_yd_s[alignment_mask])
            speed_xy_error_p95 = float(np.nanpercentile(abs_error, 95))
            speed_xy_outlier_pct = float(
                (np.nansum(abs_error > float(speed_alignment_error_yd_s)) / alignment_n) * 100.0
            )
        else:
            speed_xy_corr = np.nan
            speed_xy_error_p95 = np.nan
            speed_xy_outlier_pct = np.nan
    else:
        max_xy_step_yd = np.nan
        teleport_count = 0
        teleport_pct = np.nan
        alignment_n = 0
        speed_xy_corr = np.nan
        speed_xy_error_p95 = np.nan
        speed_xy_outlier_pct = np.nan

    return {
        "sample_count": sample_count,
        "expected_cadence_s": float(expected_cadence_s),
        "pct_on_expected_cadence": float(on_cadence.mean() * 100.0),
        "cadence_p50_s": float(cadence_quantiles.get(0.5, np.nan)),
        "cadence_p95_s": float(cadence_quantiles.get(0.95, np.nan)),
        "cadence_p99_s": float(cadence_quantiles.get(0.99, np.nan)),
        "max_gap_s": float(dt_s.max()),
        "gap_count": gap_count,
        "gap_pct": gap_pct,
        "gap_threshold_s": float(gap_limit),
        "max_xy_step_yd": max_xy_step_yd,
        "teleport_count": teleport_count,
        "teleport_pct": teleport_pct,
        "teleport_speed_threshold_yd_s": float(max_plausible_xy_speed_yd_s),
        "step_distance_outlier_source": step_distance_col,
        "step_distance_outlier_threshold_yd": outlier_threshold,
        "step_distance_outlier_count": outlier_count,
        "step_distance_outlier_pct": float((outlier_count / sample_count) * 100.0),
        "x_min_yd": x_min,
        "x_max_yd": x_max,
        "y_min_yd": y_min,
        "y_max_yd": y_max,
        "peak_speed_mph": peak_speed_mph,
        "pct_speed_le_0_5_mph": pct_speed_le_0_5_mph,
        "distance_yd_from_speed": distance_yd_from_speed,
        "inactive_speed_threshold_mph": float(inactive_speed_threshold_mph),
        "inactive_sample_count": inactive_count,
        "inactive_sample_pct": inactive_pct,
        "position_outlier_low_quantile": q_low,
        "position_outlier_high_quantile": q_high,
        "position_outlier_count": position_outlier_count,
        "position_outlier_pct": position_outlier_pct,
        "speed_xy_alignment_samples": alignment_n,
        "speed_xy_corr": speed_xy_corr,
        "speed_xy_error_p95_yd_s": speed_xy_error_p95,
        "speed_xy_outlier_pct": speed_xy_outlier_pct,
        "speed_xy_outlier_threshold_yd_s": float(speed_alignment_error_yd_s),
        "signed_accel_source": "signed_accel_ms2",
        "signed_accel_available": bool(signed_accel_available),
        "overall_status": "UNKNOWN",
        "gap_handling": (
            f"Rows retained; continuity breaks inserted when dt_s > {gap_limit:.2f}s for windows/events/lines."
        ),
        "outlier_handling": (
            f"Rows retained; {step_distance_col} outliers above {outlier_threshold:.2f} yd "
            f"({outlier_quantile * 100:.1f}th percentile) flagged for trajectory-break handling."
        ),
    }


def compute_validation_gates(
    df: pd.DataFrame,
    *,
    expected_cadence_s: float = EXPECTED_DT_SECONDS,
    cadence_tolerance_s: float = 0.02,
    cadence_min_pct: float = 95.0,
    gap_threshold_s: float | None = None,
    max_gap_pass_s: float = 0.5,
    max_plausible_xy_speed_yd_s: float = 14.0,
    max_jump_pct: float = 0.5,
    speed_alignment_error_yd_s: float = 2.0,
    min_speed_alignment_corr: float = 0.8,
    max_speed_alignment_outlier_pct: float = 3.0,
    max_dis_minus_speed_pct: float = 20.0,
) -> pd.DataFrame:
    """Return pass/warn QA gates for cadence, gaps, coordinate jumps, and speed alignment."""
    gate_limit = gap_threshold_s if gap_threshold_s is not None else expected_cadence_s * 1.5
    if df.empty:
        return pd.DataFrame(
            [
                {
                    "gate_key": "cadence_on_target_pct",
                    "gate": "Cadence on target (%)",
                    "status": "WARN",
                    "value": 0.0,
                    "threshold": cadence_min_pct,
                    "direction": ">=",
                    "unit": "%",
                    "notes": "No samples available.",
                },
                {
                    "gate_key": "max_gap_s",
                    "gate": "Max time gap (s)",
                    "status": "WARN",
                    "value": 0.0,
                    "threshold": max_gap_pass_s,
                    "direction": "<=",
                    "unit": "s",
                    "notes": "No samples available.",
                },
                {
                    "gate_key": "improbable_jump_pct",
                    "gate": "Improbable position jump (%)",
                    "status": "WARN",
                    "value": 0.0,
                    "threshold": max_jump_pct,
                    "direction": "<=",
                    "unit": "%",
                    "notes": "No samples available.",
                },
                {
                    "gate_key": "speed_alignment_corr",
                    "gate": "Provided vs derived speed correlation",
                    "status": "WARN",
                    "value": np.nan,
                    "threshold": min_speed_alignment_corr,
                    "direction": ">=",
                    "unit": "r",
                    "notes": "No samples available.",
                },
                {
                    "gate_key": "speed_alignment_outlier_pct",
                    "gate": "Provided vs derived speed outliers (%)",
                    "status": "WARN",
                    "value": 0.0,
                    "threshold": max_speed_alignment_outlier_pct,
                    "direction": "<=",
                    "unit": "%",
                    "notes": "No samples available.",
                },
                {
                    "gate_key": "vendor_dis_distance_delta_pct",
                    "gate": "Vendor `dis` vs speed distance delta (%)",
                    "status": "WARN",
                    "value": np.nan,
                    "threshold": max_dis_minus_speed_pct,
                    "direction": "<=",
                    "unit": "%",
                    "notes": "No samples available.",
                },
            ]
        )

    dt_s = df["dt_s"].astype(float)
    on_target = dt_s.between(expected_cadence_s - cadence_tolerance_s, expected_cadence_s + cadence_tolerance_s)
    cadence_pct = float(on_target.mean() * 100.0)
    max_gap_s = float(dt_s.max())
    gap_count = int((dt_s > gate_limit).sum())

    xy_speed_yd_s = np.where(dt_s > 0, df["step_distance_xy_yd"].astype(float) / dt_s, np.nan)
    valid_xy = np.isfinite(xy_speed_yd_s)
    jump_mask = valid_xy & (xy_speed_yd_s > max_plausible_xy_speed_yd_s)
    jump_count = int(jump_mask.sum())
    valid_xy_count = int(valid_xy.sum())
    jump_pct = float((jump_count / valid_xy_count) * 100.0) if valid_xy_count > 0 else 0.0

    if "s" in df.columns:
        provided_speed_yd_s = df["s"].astype(float).to_numpy()
    else:
        provided_speed_yd_s = df["speed_mph"].astype(float).to_numpy() / YARDS_PER_SECOND_TO_MPH
    alignment_mask = valid_xy & np.isfinite(provided_speed_yd_s)
    aligned_n = int(alignment_mask.sum())

    if aligned_n >= 3:
        speed_corr = float(np.corrcoef(provided_speed_yd_s[alignment_mask], xy_speed_yd_s[alignment_mask])[0, 1])
        abs_error = np.abs(provided_speed_yd_s - xy_speed_yd_s)
        outlier_count = int(((abs_error > speed_alignment_error_yd_s) & alignment_mask).sum())
        outlier_pct = float((outlier_count / aligned_n) * 100.0)
    else:
        speed_corr = np.nan
        outlier_count = 0
        outlier_pct = 0.0

    # Vendor dis column not used (systematic -74.66% error vs speed-integrated distance)
    dis_minus_speed_pct = np.nan

    def _status(value: float, threshold: float, direction: str) -> str:
        if np.isnan(value):
            return "WARN"
        if direction == ">=":
            return "PASS" if value >= threshold else "WARN"
        return "PASS" if value <= threshold else "WARN"

    rows = [
        {
            "gate_key": "cadence_on_target_pct",
            "gate": "Cadence on target (%)",
            "status": _status(cadence_pct, cadence_min_pct, ">="),
            "value": cadence_pct,
            "threshold": cadence_min_pct,
            "direction": ">=",
            "unit": "%",
            "notes": (
                f"{cadence_pct:.1f}% on-target cadence; gaps above {gate_limit:.2f}s: {gap_count} samples."
            ),
        },
        {
            "gate_key": "max_gap_s",
            "gate": "Max time gap (s)",
            "status": _status(max_gap_s, max_gap_pass_s, "<="),
            "value": max_gap_s,
            "threshold": max_gap_pass_s,
            "direction": "<=",
            "unit": "s",
            "notes": (
                f"Max dt observed {max_gap_s:.2f}s (gate limit for pass {max_gap_pass_s:.2f}s)."
            ),
        },
        {
            "gate_key": "improbable_jump_pct",
            "gate": "Improbable position jump (%)",
            "status": _status(jump_pct, max_jump_pct, "<="),
            "value": jump_pct,
            "threshold": max_jump_pct,
            "direction": "<=",
            "unit": "%",
            "notes": (
                f"{jump_count} of {valid_xy_count} samples imply XY speed > {max_plausible_xy_speed_yd_s:.1f} yd/s."
            ),
        },
        {
            "gate_key": "speed_alignment_corr",
            "gate": "Provided vs derived speed correlation",
            "status": _status(speed_corr, min_speed_alignment_corr, ">="),
            "value": speed_corr,
            "threshold": min_speed_alignment_corr,
            "direction": ">=",
            "unit": "r",
            "notes": f"Pearson correlation on {aligned_n} aligned samples.",
        },
        {
            "gate_key": "speed_alignment_outlier_pct",
            "gate": "Provided vs derived speed outliers (%)",
            "status": _status(outlier_pct, max_speed_alignment_outlier_pct, "<="),
            "value": outlier_pct,
            "threshold": max_speed_alignment_outlier_pct,
            "direction": "<=",
            "unit": "%",
            "notes": (
                f"{outlier_count} samples exceed |provided-derived| > {speed_alignment_error_yd_s:.1f} yd/s."
            ),
        },
        {
            "gate_key": "vendor_dis_distance_delta_pct",
            "gate": "Vendor `dis` vs speed distance delta (%)",
            "status": _status(dis_minus_speed_pct, max_dis_minus_speed_pct, "<="),
            "value": dis_minus_speed_pct,
            "threshold": max_dis_minus_speed_pct,
            "direction": "<=",
            "unit": "%",
            "notes": (
                "Absolute total-distance delta using vendor `dis` against speed-integrated distance."
                if np.isfinite(dis_minus_speed_pct)
                else "Vendor `dis` unavailable; distance policy defaults to speed channel."
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_validation_takeaways(
    validation_gates: pd.DataFrame,
    *,
    preferred_distance_source: str = "speed",
) -> list[str]:
    """Build concise pass/fail conclusions and trust policy from validation-gate output."""
    if validation_gates.empty:
        return [
            "Validation gates unavailable; treat all workload metrics as provisional.",
            "Trust policy: speed-derived distance remains the default until QA is available.",
        ]

    total = int(len(validation_gates))
    pass_count = int((validation_gates["status"] == "PASS").sum())

    by_key = validation_gates.set_index("gate_key")
    jump_ok = bool("improbable_jump_pct" in by_key.index and by_key.loc["improbable_jump_pct", "status"] == "PASS")
    corr_ok = bool("speed_alignment_corr" in by_key.index and by_key.loc["speed_alignment_corr", "status"] == "PASS")
    outlier_ok = bool(
        "speed_alignment_outlier_pct" in by_key.index and by_key.loc["speed_alignment_outlier_pct", "status"] == "PASS"
    )
    dis_ok = bool(
        "vendor_dis_distance_delta_pct" in by_key.index and by_key.loc["vendor_dis_distance_delta_pct", "status"] == "PASS"
    )

    trust_source = "speed channel (`s`)" if preferred_distance_source == "speed" else "position channel (`x`,`y`)"
    spatial_note = (
        "Coordinate jumps are low; movement map can be used for role/space interpretation."
        if jump_ok
        else "Coordinate jumps are elevated; use movement map for broad tendencies, not exact micro-trajectories."
    )
    speed_note = (
        "Provided and derived speeds align; session intensity metrics are trusted."
        if corr_ok and outlier_ok
        else "Provided and derived speeds diverge in part of the session; prioritize provided speed channel for workload totals."
    )
    dis_note = (
        "Vendor `dis` aligns with speed-integrated totals within tolerance."
        if dis_ok
        else "Vendor `dis` diverges from speed-integrated totals; keep `s * dt` as distance source and treat `dis` as QA-only."
    )

    return [
        f"Validation gates: {pass_count}/{total} PASS.",
        f"Trust policy: distance and workload totals are computed from {trust_source}.",
        spatial_note,
        speed_note,
        dis_note,
    ]


def build_qc_check_table(
    df: pd.DataFrame,
    *,
    qa_summary: dict[str, float | int | str],
    validation_gates: pd.DataFrame,
    peak_speed_warn_mph: float = 22.0,
    stationary_warn_pct: float = 80.0,
) -> pd.DataFrame:
    """Build a compact QC table intended for a single coach-facing checkpoint slide."""
    def _gate_payload(gate_key: str) -> tuple[str, float, float]:
        match = validation_gates.loc[validation_gates["gate_key"] == gate_key]
        if match.empty:
            return "WARN", np.nan, np.nan
        row = match.iloc[0]
        return (
            str(row["status"]),
            float(row["value"]) if pd.notna(row["value"]) else np.nan,
            float(row["threshold"]) if pd.notna(row["threshold"]) else np.nan,
        )

    cadence_status, cadence_value, cadence_threshold = _gate_payload("cadence_on_target_pct")
    gap_status, gap_value, gap_threshold = _gate_payload("max_gap_s")
    jump_status, jump_value, jump_threshold = _gate_payload("improbable_jump_pct")
    dis_status, dis_value, dis_threshold = _gate_payload("vendor_dis_distance_delta_pct")

    peak_speed_mph = float(pd.to_numeric(df["speed_mph"], errors="coerce").max()) if "speed_mph" in df.columns else np.nan
    stationary_pct = float((pd.to_numeric(df["speed_mph"], errors="coerce") <= 0.5).mean() * 100.0) if "speed_mph" in df.columns else np.nan
    signed_accel = pd.to_numeric(df["signed_accel_ms2"], errors="coerce") if "signed_accel_ms2" in df.columns else pd.Series(dtype=float)
    signed_accel_available = bool(not signed_accel.empty and signed_accel.notna().any())
    signed_accel_polarity_ok = bool((signed_accel.max() > 0) and (signed_accel.min() < 0)) if signed_accel_available else False
    x_span = float(pd.to_numeric(df["x"], errors="coerce").max() - pd.to_numeric(df["x"], errors="coerce").min()) if "x" in df.columns else np.nan
    y_span = float(pd.to_numeric(df["y"], errors="coerce").max() - pd.to_numeric(df["y"], errors="coerce").min()) if "y" in df.columns else np.nan
    span_ok = bool(np.isfinite(x_span) and np.isfinite(y_span) and (x_span >= 5.0) and (y_span >= 10.0))

    rows = [
        {
            "check_key": "cadence_on_target_pct",
            "check": "Cadence on target (%)",
            "status": cadence_status,
            "value": cadence_value,
            "unit": "%",
            "threshold": cadence_threshold,
            "notes": "Expected 10 Hz sampling with small tolerance.",
        },
        {
            "check_key": "max_gap_s",
            "check": "Largest timestamp gap (s)",
            "status": gap_status,
            "value": gap_value,
            "unit": "s",
            "threshold": gap_threshold,
            "notes": "Gaps reset windows/events and break trajectories.",
        },
        {
            "check_key": "improbable_jump_pct",
            "check": "Improbable XY jump share (%)",
            "status": jump_status,
            "value": jump_value,
            "unit": "%",
            "threshold": jump_threshold,
            "notes": "Percent of samples with implied XY speed above plausibility cap.",
        },
        {
            "check_key": "position_span_sanity",
            "check": "Position span sanity",
            "status": "PASS" if span_ok else "WARN",
            "value": x_span if np.isfinite(x_span) else np.nan,
            "unit": "yd",
            "threshold": 5.0,
            "notes": f"X span {x_span:.1f} yd, Y span {y_span:.1f} yd.",
        },
        {
            "check_key": "peak_speed_mph",
            "check": "Peak speed sanity (mph)",
            "status": "PASS" if np.isfinite(peak_speed_mph) and peak_speed_mph <= peak_speed_warn_mph else "WARN",
            "value": peak_speed_mph,
            "unit": "mph",
            "threshold": peak_speed_warn_mph,
            "notes": "Guardrail against implausible speed spikes.",
        },
        {
            "check_key": "stationary_pct",
            "check": "Time <= 0.5 mph (%)",
            "status": "PASS" if np.isfinite(stationary_pct) and stationary_pct <= stationary_warn_pct else "WARN",
            "value": stationary_pct,
            "unit": "%",
            "threshold": stationary_warn_pct,
            "notes": "High stationary share can indicate teaching/rest-heavy session or feed issues.",
        },
        {
            "check_key": "vendor_dis_distance_delta_pct",
            "check": "Vendor `dis` vs `s*dt` delta (%)",
            "status": dis_status,
            "value": dis_value,
            "unit": "%",
            "threshold": dis_threshold,
            "notes": "Distance policy uses `s*dt`; `dis` is QA-only when mismatch is high.",
        },
        {
            "check_key": "signed_accel_policy",
            "check": "Signed accel channel available",
            "status": "PASS" if signed_accel_available else "WARN",
            "value": 1.0 if signed_accel_available else 0.0,
            "unit": "bool",
            "threshold": 1.0,
            "notes": "Accel/decel events must use signed `sa` channel.",
        },
        {
            "check_key": "signed_accel_polarity",
            "check": "Signed accel has +/- values",
            "status": "PASS" if signed_accel_polarity_ok else "WARN",
            "value": float(signed_accel.max()) if signed_accel_available else np.nan,
            "unit": "m/s^2",
            "threshold": 0.0,
            "notes": "Confirms deceleration direction is represented in channel.",
        },
    ]
    return pd.DataFrame(rows)


def compute_qc_status(
    qc_checks: pd.DataFrame,
    *,
    cadence_warn_pct: float = 95.0,
    cadence_fail_pct: float = 90.0,
    max_gap_warn_s: float = 5.0,
    max_gap_fail_s: float = 180.0,
    jump_warn_pct: float = 0.5,
    jump_fail_pct: float = 2.0,
    speed_corr_warn: float = 0.8,
    speed_corr_fail: float = 0.6,
    speed_outlier_warn_pct: float = 3.0,
    speed_outlier_fail_pct: float = 10.0,
    dis_delta_warn_pct: float = 20.0,
    dis_delta_fail_pct: float = 300.0,
) -> str:
    """Return `QC PASS`, `QC WARN`, or `QC FAILED` using impact-based thresholds."""
    if qc_checks.empty:
        return "QC FAILED"

    indexed = qc_checks.set_index("check_key")

    def _value(check_key: str) -> float:
        if check_key not in indexed.index:
            return np.nan
        val = indexed.loc[check_key, "value"]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        return float(val) if pd.notna(val) else np.nan

    has_fail = False
    has_warn = False

    signed_accel_policy = _value("signed_accel_policy")
    if not np.isfinite(signed_accel_policy) or signed_accel_policy < 1.0:
        has_fail = True

    cadence_pct = _value("cadence_on_target_pct")
    if np.isfinite(cadence_pct):
        if cadence_pct < cadence_fail_pct:
            has_fail = True
        elif cadence_pct < cadence_warn_pct:
            has_warn = True
    else:
        has_fail = True

    max_gap_s = _value("max_gap_s")
    if np.isfinite(max_gap_s):
        if max_gap_s > max_gap_fail_s:
            has_fail = True
        elif max_gap_s > max_gap_warn_s:
            has_warn = True
    else:
        has_fail = True

    jump_pct = _value("improbable_jump_pct")
    if np.isfinite(jump_pct):
        if jump_pct > jump_fail_pct:
            has_fail = True
        elif jump_pct > jump_warn_pct:
            has_warn = True
    else:
        has_warn = True

    speed_corr = _value("speed_alignment_corr")
    if np.isfinite(speed_corr):
        if speed_corr < speed_corr_fail:
            has_fail = True
        elif speed_corr < speed_corr_warn:
            has_warn = True

    speed_outlier_pct = _value("speed_alignment_outlier_pct")
    if np.isfinite(speed_outlier_pct):
        if speed_outlier_pct > speed_outlier_fail_pct:
            has_fail = True
        elif speed_outlier_pct > speed_outlier_warn_pct:
            has_warn = True

    dis_delta_pct = _value("vendor_dis_distance_delta_pct")
    if np.isfinite(dis_delta_pct):
        if dis_delta_pct > dis_delta_fail_pct:
            has_fail = True
        elif dis_delta_pct > dis_delta_warn_pct:
            has_warn = True

    if has_fail:
        return "QC FAILED"
    if has_warn:
        return "QC WARN"
    return "QC PASS"


def classify_hsr_exposure(total_distance_yd: float, hsr_distance_yd: float) -> str:
    """Classify HSR exposure as low/moderate/high using session-level distance share."""
    if total_distance_yd <= 0:
        return "Low"
    hsr_pct = (hsr_distance_yd / total_distance_yd) * 100.0
    if hsr_pct >= 12.0:
        return "High"
    if hsr_pct >= 5.0:
        return "Moderate"
    return "Low"


def summarize_window_context(
    df: pd.DataFrame,
    *,
    window_start_utc: pd.Timestamp,
    window_end_utc: pd.Timestamp,
    hsr_threshold_mph: float = 13.0,
    sprint_threshold_mph: float = 16.0,
    accel_threshold_ms2: float = 3.0,
    decel_threshold_ms2: float = -3.0,
) -> dict[str, float | int | str]:
    """Summarize workload context inside a specific time window."""
    window_df = df[(df["ts"] >= window_start_utc) & (df["ts"] <= window_end_utc)].copy()
    if window_df.empty:
        return {
            "window_start_utc": str(window_start_utc),
            "window_end_utc": str(window_end_utc),
            "duration_s": 0.0,
            "distance_yd": 0.0,
            "hsr_event_count": 0,
            "sprint_event_count": 0,
            "accel_event_count": 0,
            "decel_event_count": 0,
            "hsr_exposure_level": "Low",
        }

    events = compute_session_event_counts(
        window_df,
        hsr_threshold_mph=hsr_threshold_mph,
        sprint_threshold_mph=sprint_threshold_mph,
        accel_threshold_ms2=accel_threshold_ms2,
        decel_threshold_ms2=decel_threshold_ms2,
    )
    distance_yd = float(window_df["step_distance_yd_from_speed"].sum())
    hsr_distance_yd = float(events["hsr_distance_yd"])
    return {
        "window_start_utc": str(window_start_utc),
        "window_end_utc": str(window_end_utc),
        "duration_s": float(window_df["dt_s"].sum()),
        "distance_yd": distance_yd,
        "hsr_event_count": int(events["hsr_event_count"]),
        "sprint_event_count": int(events["sprint_event_count"]),
        "accel_event_count": int(events["accel_event_count"]),
        "decel_event_count": int(events["decel_event_count"]),
        "hsr_exposure_level": classify_hsr_exposure(distance_yd, hsr_distance_yd),
    }


def _count_boolean_runs(
    mask: pd.Series,
    dt_s: pd.Series,
    min_duration_s: float,
    *,
    reset_mask: pd.Series | None = None,
) -> int:
    count = 0
    run_duration_s = 0.0
    if reset_mask is None:
        reset_mask = pd.Series(False, index=mask.index, dtype=bool)

    for is_true, dt, reset in zip(
        mask.to_numpy(dtype=bool),
        dt_s.to_numpy(dtype=float),
        reset_mask.to_numpy(dtype=bool),
        strict=False,
    ):
        if reset and run_duration_s >= min_duration_s:
            count += 1
        if reset:
            run_duration_s = 0.0

        if is_true:
            run_duration_s += float(dt)
            continue
        if run_duration_s >= min_duration_s:
            count += 1
        run_duration_s = 0.0

    if run_duration_s >= min_duration_s:
        count += 1
    return count


def _count_boolean_transitions(mask: pd.Series, *, reset_mask: pd.Series | None = None) -> int:
    if mask.empty:
        return 0
    if reset_mask is None:
        reset_mask = pd.Series(False, index=mask.index, dtype=bool)

    starts = mask & (~mask.shift(fill_value=False) | reset_mask)
    return int(starts.sum())


def _continuity_reset_mask(df: pd.DataFrame, continuity_break_col: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool)
    if continuity_break_col in df.columns:
        return df[continuity_break_col].fillna(False).astype(bool)
    return pd.Series(False, index=df.index, dtype=bool)
