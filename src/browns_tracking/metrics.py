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
    df: pd.DataFrame,
    config: PeakDemandConfig = PeakDemandConfig(),
    *,
    block_col: str | None = None,
    min_window_coverage_fraction: float = 0.95,
) -> pd.DataFrame:
    """Compute rolling demand metrics for each configured window."""
    required = {"ts", "dt_s", "speed_mph", "signed_accel_ms2", "step_distance_yd_from_speed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for peak demand timeseries: {missing}")

    input_block_col = block_col if block_col is not None and block_col in df.columns else None
    work_block_col = input_block_col or "_rolling_block_id"

    indexed = df[["ts", "dt_s", "speed_mph", "signed_accel_ms2", "step_distance_yd_from_speed"]].copy()
    indexed[work_block_col] = (
        df[input_block_col].astype(int).to_numpy() if input_block_col else np.zeros(len(indexed), dtype=int)
    )
    indexed["hsr_distance_yd"] = np.where(
        indexed["speed_mph"] >= config.hsr_threshold_mph,
        indexed["step_distance_yd_from_speed"],
        0.0,
    )

    accel_bool = indexed["signed_accel_ms2"] >= config.accel_threshold_ms2
    decel_bool = indexed["signed_accel_ms2"] <= config.decel_threshold_ms2
    block_break = indexed[work_block_col].ne(indexed[work_block_col].shift(fill_value=indexed[work_block_col].iloc[0]))
    indexed["accel_event"] = (accel_bool & (~accel_bool.shift(fill_value=False) | block_break)).astype(int)
    indexed["decel_event"] = (decel_bool & (~decel_bool.shift(fill_value=False) | block_break)).astype(int)

    indexed = indexed.set_index("ts")
    out = pd.DataFrame(index=indexed.index)
    if input_block_col is not None:
        out[input_block_col] = indexed[work_block_col].astype(int)

    if min_window_coverage_fraction <= 0.0:
        min_window_coverage_fraction = 0.0
    min_window_coverage_fraction = min(float(min_window_coverage_fraction), 1.0)

    for window_s in config.distance_windows_s:
        window = f"{int(window_s)}s"
        coverage = _rolling_sum_within_blocks(indexed, "dt_s", work_block_col, window)
        min_coverage_s = float(window_s) * min_window_coverage_fraction
        valid = coverage >= min_coverage_s

        out[f"distance_{window_s}s_yd"] = _rolling_sum_within_blocks(
            indexed, "step_distance_yd_from_speed", work_block_col, window
        ).where(valid)
        out[f"hsr_distance_{window_s}s_yd"] = _rolling_sum_within_blocks(
            indexed, "hsr_distance_yd", work_block_col, window
        ).where(valid)
        out[f"accel_events_{window_s}s"] = _rolling_sum_within_blocks(
            indexed, "accel_event", work_block_col, window
        ).where(valid)
        out[f"decel_events_{window_s}s"] = _rolling_sum_within_blocks(
            indexed, "decel_event", work_block_col, window
        ).where(valid)

    return out.reset_index()


def top_non_overlapping_windows(
    rolling_df: pd.DataFrame,
    metric_column: str,
    window_s: int,
    top_n: int = 3,
    *,
    block_col: str | None = None,
) -> pd.DataFrame:
    """Pick top-N non-overlapping windows from a rolling metric column."""
    required = {"ts", metric_column}
    missing = required - set(rolling_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    active_block_col = block_col
    if active_block_col is None and "continuous_block_id" in rolling_df.columns:
        active_block_col = "continuous_block_id"

    ranked_cols = ["ts", metric_column]
    if active_block_col is not None and active_block_col in rolling_df.columns:
        ranked_cols.append(active_block_col)
    ranked = rolling_df[ranked_cols].dropna(subset=[metric_column]).sort_values(metric_column, ascending=False)

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

        block_payload: dict[str, int] = {}
        if active_block_col is not None and active_block_col in ranked_cols:
            block_value = getattr(row, active_block_col)
            if pd.notna(block_value):
                block_payload[active_block_col] = int(block_value)

        selected.append(
            {
                "metric": metric_column,
                "window_s": int(window_s),
                "window_start_utc": start_ts,
                "window_end_utc": end_ts,
                "value": float(row[1]),
                **block_payload,
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
        if not rolling_df[metric_col].notna().any():
            rows.append(
                {
                    "window_s": int(window_s),
                    "window_label": _window_label(window_s),
                    "best_distance_yd": 0.0,
                    "window_start_utc": pd.NaT,
                    "window_end_utc": pd.NaT,
                }
            )
            continue

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


def _rolling_sum_within_blocks(
    indexed_df: pd.DataFrame,
    value_col: str,
    block_col: str,
    window: str,
) -> pd.Series:
    series_parts: list[pd.Series] = []
    for _, chunk in indexed_df.groupby(block_col, sort=True):
        rolled = chunk[value_col].rolling(window, min_periods=1).sum()
        series_parts.append(rolled)
    if not series_parts:
        return pd.Series(dtype=float)
    return pd.concat(series_parts)
