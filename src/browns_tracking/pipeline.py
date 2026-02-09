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
    """Load CSV, normalize types, sort by timestamp, and add derived columns."""
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.sort_values("ts").reset_index(drop=True)

    dt = df["ts"].diff().dt.total_seconds()
    df["dt_s"] = dt.fillna(EXPECTED_DT_SECONDS)

    df["speed_mph"] = df["s"] * YARDS_PER_SECOND_TO_MPH
    df["accel_ms2"] = df["a"] * YARDS_PER_SECOND_SQ_TO_M_PER_SECOND_SQ
    df["signed_accel_ms2"] = df["sa"] * YARDS_PER_SECOND_SQ_TO_M_PER_SECOND_SQ
    df["step_distance_yd_from_speed"] = df["s"] * df["dt_s"]

    xy_step = np.sqrt(df["x"].diff().pow(2) + df["y"].diff().pow(2))
    df["step_distance_xy_yd"] = xy_step.fillna(0.0)

    return df


def summarize_session(df: pd.DataFrame) -> dict[str, float | int | str]:
    """Compute a compact required-metrics starter summary."""
    gaps = int((df["dt_s"] > (EXPECTED_DT_SECONDS * 1.5)).sum())
    duration_s = float((df["ts"].iloc[-1] - df["ts"].iloc[0]).total_seconds())

    return {
        "rows": int(len(df)),
        "start_ts_utc": str(df["ts"].iloc[0]),
        "end_ts_utc": str(df["ts"].iloc[-1]),
        "duration_s": duration_s,
        "sampling_gap_count": gaps,
        "distance_yd_from_speed": float(df["step_distance_yd_from_speed"].sum()),
        "distance_yd_from_xy": float(df["step_distance_xy_yd"].sum()),
        "mean_speed_mph": float(df["speed_mph"].mean()),
        "peak_speed_mph": float(df["speed_mph"].max()),
        "mean_accel_ms2": float(df["signed_accel_ms2"].mean()),
        "peak_accel_ms2": float(df["signed_accel_ms2"].max()),
        "peak_decel_ms2": float(df["signed_accel_ms2"].min()),
    }


def compute_session_event_counts(
    df: pd.DataFrame,
    *,
    hsr_threshold_mph: float = 13.0,
    sprint_threshold_mph: float = 16.0,
    min_speed_event_duration_s: float = 1.0,
    accel_threshold_ms2: float = 3.0,
    decel_threshold_ms2: float = -3.0,
) -> dict[str, float | int]:
    """Compute core event counts used in coach-facing workload narratives."""
    hsr_mask = df["speed_mph"] >= hsr_threshold_mph
    sprint_mask = df["speed_mph"] >= sprint_threshold_mph
    accel_mask = df["signed_accel_ms2"] >= accel_threshold_ms2
    decel_mask = df["signed_accel_ms2"] <= decel_threshold_ms2

    return {
        "hsr_distance_yd": float(df.loc[hsr_mask, "step_distance_yd_from_speed"].sum()),
        "sprint_distance_yd": float(df.loc[sprint_mask, "step_distance_yd_from_speed"].sum()),
        "hsr_event_count": _count_boolean_runs(
            hsr_mask, df["dt_s"], min_duration_s=min_speed_event_duration_s
        ),
        "sprint_event_count": _count_boolean_runs(
            sprint_mask, df["dt_s"], min_duration_s=min_speed_event_duration_s
        ),
        "accel_event_count": int((accel_mask & ~accel_mask.shift(fill_value=False)).sum()),
        "decel_event_count": int((decel_mask & ~decel_mask.shift(fill_value=False)).sum()),
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
    outlier_quantile: float = 0.995,
) -> dict[str, float | int | str]:
    """Build an explicit QA summary for cadence gaps and distance outliers."""
    if df.empty:
        return {
            "sample_count": 0,
            "expected_cadence_s": expected_cadence_s,
            "pct_on_expected_cadence": 0.0,
            "max_gap_s": 0.0,
            "gap_count": 0,
            "gap_threshold_s": gap_threshold_s or (expected_cadence_s * 1.5),
            "step_distance_outlier_threshold_yd": 0.0,
            "step_distance_outlier_count": 0,
            "step_distance_outlier_pct": 0.0,
            "gap_handling": "No samples available.",
            "outlier_handling": "No samples available.",
        }

    gap_limit = gap_threshold_s if gap_threshold_s is not None else expected_cadence_s * 1.5
    dt_s = df["dt_s"].astype(float)
    on_cadence = dt_s.between(
        expected_cadence_s - cadence_tolerance_s, expected_cadence_s + cadence_tolerance_s
    )
    gap_count = int((dt_s > gap_limit).sum())

    step_distance = df["step_distance_yd_from_speed"].astype(float)
    outlier_threshold = float(step_distance.quantile(outlier_quantile))
    outlier_count = int((step_distance > outlier_threshold).sum())

    sample_count = int(len(df))
    return {
        "sample_count": sample_count,
        "expected_cadence_s": float(expected_cadence_s),
        "pct_on_expected_cadence": float(on_cadence.mean() * 100.0),
        "max_gap_s": float(dt_s.max()),
        "gap_count": gap_count,
        "gap_threshold_s": float(gap_limit),
        "step_distance_outlier_threshold_yd": outlier_threshold,
        "step_distance_outlier_count": outlier_count,
        "step_distance_outlier_pct": float((outlier_count / sample_count) * 100.0),
        "gap_handling": (
            f"Rows retained; flagged for review when dt_s > {gap_limit:.2f}s (no exclusion)."
        ),
        "outlier_handling": (
            "Rows retained; step-distance outliers flagged above "
            f"{outlier_threshold:.2f} yd ({outlier_quantile * 100:.1f}th percentile)."
        ),
    }


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


def _count_boolean_runs(mask: pd.Series, dt_s: pd.Series, min_duration_s: float) -> int:
    count = 0
    run_duration_s = 0.0
    for is_true, dt in zip(mask.to_numpy(dtype=bool), dt_s.to_numpy(dtype=float), strict=False):
        if is_true:
            run_duration_s += float(dt)
            continue
        if run_duration_s >= min_duration_s:
            count += 1
        run_duration_s = 0.0

    if run_duration_s >= min_duration_s:
        count += 1
    return count
