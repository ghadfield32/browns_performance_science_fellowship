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
