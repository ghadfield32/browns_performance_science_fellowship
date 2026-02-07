"""Workload metric helpers: speed bands and peak-demand windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpeedBand:
    """Speed band definition in mph."""

    name: str
    lower_mph: float
    upper_mph: float | None


@dataclass(frozen=True)
class PeakDemandConfig:
    """Configuration for rolling peak-demand metrics."""

    distance_windows_s: tuple[int, ...] = (30, 60, 180, 300)
    hsr_threshold_mph: float = 12.0
    accel_threshold_ms2: float = 2.5
    decel_threshold_ms2: float = -2.5


def default_absolute_speed_bands() -> list[SpeedBand]:
    """Coach-readable absolute speed bands."""
    return [
        SpeedBand("Walk", 0.0, 3.0),
        SpeedBand("Jog", 3.0, 8.0),
        SpeedBand("Run", 8.0, 12.0),
        SpeedBand("HSR", 12.0, 15.0),
        SpeedBand("Sprint", 15.0, None),
    ]


def relative_speed_bands(
    max_speed_mph: float, percent_edges: Sequence[float] = (0.0, 0.4, 0.6, 0.8, 0.9, 1.0)
) -> list[SpeedBand]:
    """Build relative-to-max speed bands from 0-100% edges."""
    if max_speed_mph <= 0:
        raise ValueError("max_speed_mph must be > 0")
    if len(percent_edges) < 2:
        raise ValueError("percent_edges must have at least two values")
    if percent_edges[0] != 0.0 or percent_edges[-1] != 1.0:
        raise ValueError("percent_edges must start at 0.0 and end at 1.0")
    if any(b <= a for a, b in zip(percent_edges, percent_edges[1:])):
        raise ValueError("percent_edges must be strictly increasing")

    bands: list[SpeedBand] = []
    for lower_pct, upper_pct in zip(percent_edges, percent_edges[1:]):
        lower = lower_pct * max_speed_mph
        upper = upper_pct * max_speed_mph
        label = f"{int(lower_pct * 100)}-{int(upper_pct * 100)}% max"
        upper_val = None if np.isclose(upper_pct, 1.0) else upper
        bands.append(SpeedBand(label, lower, upper_val))
    return bands


def assign_speed_band(
    speed_mph: pd.Series, bands: Sequence[SpeedBand], out_of_range_label: str = "Out of range"
) -> pd.Series:
    """Assign each sample to a speed-band label."""
    labels = pd.Series(out_of_range_label, index=speed_mph.index, dtype="object")
    for band in bands:
        if band.upper_mph is None:
            mask = speed_mph >= band.lower_mph
        else:
            mask = (speed_mph >= band.lower_mph) & (speed_mph < band.upper_mph)
        labels.loc[mask] = band.name
    return labels


def summarize_speed_bands(
    df: pd.DataFrame,
    bands: Sequence[SpeedBand],
    *,
    speed_col: str = "speed_mph",
    distance_col: str = "step_distance_yd_from_speed",
    time_col: str = "dt_s",
) -> pd.DataFrame:
    """Summarize distance and time accumulated in each speed band."""
    work = df[[speed_col, distance_col, time_col]].copy()
    work["speed_band"] = assign_speed_band(work[speed_col], bands)

    total_distance = work[distance_col].sum()
    total_time = work[time_col].sum()

    summary = (
        work.groupby("speed_band", dropna=False)
        .agg(
            distance_yd=(distance_col, "sum"),
            time_s=(time_col, "sum"),
            sample_count=(speed_col, "size"),
            mean_speed_mph=(speed_col, "mean"),
        )
        .reset_index()
    )

    summary["distance_pct"] = np.where(
        total_distance > 0, (summary["distance_yd"] / total_distance) * 100.0, 0.0
    )
    summary["time_pct"] = np.where(total_time > 0, (summary["time_s"] / total_time) * 100.0, 0.0)

    order = [band.name for band in bands]
    summary["speed_band"] = pd.Categorical(summary["speed_band"], categories=order, ordered=True)
    return summary.sort_values("speed_band").reset_index(drop=True)


def compute_peak_demand_timeseries(
    df: pd.DataFrame, config: PeakDemandConfig = PeakDemandConfig()
) -> pd.DataFrame:
    """Compute rolling demand metrics for each configured window."""
    indexed = df[["ts", "speed_mph", "signed_accel_ms2", "step_distance_yd_from_speed"]].copy()
    indexed["hsr_distance_yd"] = np.where(
        indexed["speed_mph"] >= config.hsr_threshold_mph,
        indexed["step_distance_yd_from_speed"],
        0.0,
    )

    accel_bool = indexed["signed_accel_ms2"] >= config.accel_threshold_ms2
    decel_bool = indexed["signed_accel_ms2"] <= config.decel_threshold_ms2
    indexed["accel_event"] = (accel_bool & ~accel_bool.shift(fill_value=False)).astype(int)
    indexed["decel_event"] = (decel_bool & ~decel_bool.shift(fill_value=False)).astype(int)

    indexed = indexed.set_index("ts")
    out = pd.DataFrame(index=indexed.index)
    for window_s in config.distance_windows_s:
        window = f"{int(window_s)}s"
        out[f"distance_{window_s}s_yd"] = indexed["step_distance_yd_from_speed"].rolling(
            window, min_periods=1
        ).sum()
        out[f"hsr_distance_{window_s}s_yd"] = indexed["hsr_distance_yd"].rolling(
            window, min_periods=1
        ).sum()
        out[f"accel_events_{window_s}s"] = indexed["accel_event"].rolling(window, min_periods=1).sum()
        out[f"decel_events_{window_s}s"] = indexed["decel_event"].rolling(window, min_periods=1).sum()

    return out.reset_index()


def top_non_overlapping_windows(
    rolling_df: pd.DataFrame, metric_column: str, window_s: int, top_n: int = 3
) -> pd.DataFrame:
    """Pick top-N non-overlapping windows from a rolling metric column."""
    required = {"ts", metric_column}
    missing = required - set(rolling_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    ranked = rolling_df[["ts", metric_column]].dropna().sort_values(metric_column, ascending=False)

    selected: list[dict[str, object]] = []
    for row in ranked.itertuples(index=False):
        end_ts = row.ts
        start_ts = end_ts - pd.to_timedelta(window_s, unit="s")

        overlaps = False
        for existing in selected:
            existing_start = existing["window_start_utc"]
            existing_end = existing["window_end_utc"]
            if (start_ts < existing_end) and (end_ts > existing_start):
                overlaps = True
                break

        if overlaps:
            continue

        selected.append(
            {
                "metric": metric_column,
                "window_s": int(window_s),
                "window_start_utc": start_ts,
                "window_end_utc": end_ts,
                "value": float(row[1]),
            }
        )
        if len(selected) >= top_n:
            break

    return pd.DataFrame(selected)


def peak_distance_table(
    rolling_df: pd.DataFrame, windows_s: Sequence[int]
) -> pd.DataFrame:
    """Best rolling distance for each window with start/end timestamps."""
    rows: list[dict[str, object]] = []
    for window_s in windows_s:
        metric_col = f"distance_{window_s}s_yd"
        if metric_col not in rolling_df.columns:
            raise ValueError(f"Missing rolling metric column: {metric_col}")
        idx = rolling_df[metric_col].idxmax()
        end_ts = rolling_df.loc[idx, "ts"]
        rows.append(
            {
                "window_s": int(window_s),
                "window_label": _window_label(window_s),
                "best_distance_yd": float(rolling_df.loc[idx, metric_col]),
                "window_start_utc": end_ts - pd.to_timedelta(window_s, unit="s"),
                "window_end_utc": end_ts,
            }
        )

    return pd.DataFrame(rows).sort_values("window_s").reset_index(drop=True)


def session_extrema_table(df: pd.DataFrame) -> pd.DataFrame:
    """Single-table summary of session max speed/acceleration demands."""
    speed_idx = df["speed_mph"].idxmax()
    accel_idx = df["signed_accel_ms2"].idxmax()
    decel_idx = df["signed_accel_ms2"].idxmin()

    return pd.DataFrame(
        [
            {
                "metric": "Max speed (mph)",
                "value": float(df.loc[speed_idx, "speed_mph"]),
                "ts_utc": df.loc[speed_idx, "ts"],
            },
            {
                "metric": "Max accel (m/s^2)",
                "value": float(df.loc[accel_idx, "signed_accel_ms2"]),
                "ts_utc": df.loc[accel_idx, "ts"],
            },
            {
                "metric": "Max decel (m/s^2)",
                "value": float(df.loc[decel_idx, "signed_accel_ms2"]),
                "ts_utc": df.loc[decel_idx, "ts"],
            },
        ]
    )


def _window_label(window_s: int) -> str:
    if window_s % 60 == 0:
        mins = window_s // 60
        return f"{mins}m" if mins > 1 else "1m"
    return f"{window_s}s"

