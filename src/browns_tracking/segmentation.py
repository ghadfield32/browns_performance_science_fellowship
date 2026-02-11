"""Transparent session segmentation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import ruptures as rpt

from .constants import EXPECTED_DT_SECONDS
from .metrics import SpeedBand, summarize_speed_bands


@dataclass(frozen=True)
class SegmentationConfig:
    """Session segmentation settings."""

    gap_threshold_s: float = EXPECTED_DT_SECONDS * 1.5
    rest_speed_threshold_mph: float = 1.0
    rest_min_duration_s: float = 20.0
    intensity_rolling_window_s: float = 15.0
    change_point_penalty: float = 30.0
    max_change_points: int = 10
    min_segment_duration_s: float = 15.0


def detect_segments(
    df: pd.DataFrame,
    config: SegmentationConfig = SegmentationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect neutral-labeled blocks (A/B/C...) from gaps, rest, and intensity changes."""
    work = df.copy()

    gap_boundaries = _gap_boundary_indices(work, config.gap_threshold_s)
    rest_boundaries = _rest_boundary_indices(
        work, config.rest_speed_threshold_mph, config.rest_min_duration_s
    )
    cp_boundaries = _intensity_change_boundaries(
        work,
        rolling_window_s=config.intensity_rolling_window_s,
        penalty=config.change_point_penalty,
        max_change_points=config.max_change_points,
    )

    boundaries = sorted({0, len(work), *gap_boundaries, *rest_boundaries, *cp_boundaries})
    boundaries = _enforce_min_segment_duration(work, boundaries, config.min_segment_duration_s)

    segment_id = np.searchsorted(boundaries[1:], np.arange(len(work)), side="right")
    work["segment_id"] = segment_id
    work["segment_label"] = work["segment_id"].map(block_label)

    boundary_rows = []
    for seg_idx in range(len(boundaries) - 1):
        start_i = boundaries[seg_idx]
        end_i = boundaries[seg_idx + 1]
        if start_i >= end_i:
            continue
        segment = work.iloc[start_i:end_i]
        boundary_rows.append(
            {
                "segment_id": seg_idx,
                "segment_label": block_label(seg_idx),
                "start_ts_utc": segment["ts"].iloc[0],
                "end_ts_utc": segment["ts"].iloc[-1],
                "duration_s": float(segment["dt_s"].sum()),
                "n_samples": int(len(segment)),
            }
        )

    boundaries_df = pd.DataFrame(boundary_rows)
    return work, boundaries_df


def summarize_segments(
    segmented_df: pd.DataFrame,
    speed_bands: Sequence[SpeedBand] | None = None,
) -> pd.DataFrame:
    """Segment-level workload summary table."""
    rows: list[dict[str, object]] = []

    for segment_id, segment in segmented_df.groupby("segment_id", sort=True):
        record: dict[str, object] = {
            "segment_id": int(segment_id),
            "segment_label": str(segment["segment_label"].iloc[0]),
            "start_ts_utc": segment["ts"].iloc[0],
            "end_ts_utc": segment["ts"].iloc[-1],
            "duration_s": float(segment["dt_s"].sum()),
            "distance_yd": float(segment["step_distance_yd_from_speed"].sum()),
            "mean_speed_mph": float(segment["speed_mph"].mean()),
            "peak_speed_mph": float(segment["speed_mph"].max()),
            "peak_accel_ms2": float(segment["signed_accel_ms2"].max()),
            "peak_decel_ms2": float(segment["signed_accel_ms2"].min()),
            "peak_60s_distance_yd": _segment_peak_distance(segment, 60),
        }

        if speed_bands is not None:
            band_summary = summarize_speed_bands(segment, speed_bands)
            for row in band_summary.itertuples(index=False):
                key = f"time_pct_{_safe_name(row.speed_band)}"
                record[key] = float(row.time_pct)

        rows.append(record)

    return pd.DataFrame(rows).sort_values("segment_id").reset_index(drop=True)


def build_coach_phase_summary(
    segmented_df: pd.DataFrame,
    *,
    min_phase_duration_s: float = 30.0,
    max_phases: int = 8,
    hsr_threshold_mph: float = 13.0,
    sprint_threshold_mph: float = 16.0,
    accel_threshold_ms2: float = 3.0,
    decel_threshold_ms2: float = -3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse raw segments into coach-readable contiguous phases."""
    required = {"ts", "dt_s", "speed_mph", "step_distance_yd_from_speed", "signed_accel_ms2"}
    missing = required - set(segmented_df.columns)
    if missing:
        raise ValueError(f"Missing columns for coach-phase summary: {missing}")

    if min_phase_duration_s <= 0:
        raise ValueError("min_phase_duration_s must be > 0")
    if max_phases < 1:
        raise ValueError("max_phases must be >= 1")

    if segmented_df.empty:
        return segmented_df.copy(), pd.DataFrame()

    work = segmented_df.reset_index(drop=True).copy()
    phase_spans = _contiguous_phase_spans(work)
    phase_spans = _merge_short_phase_spans(work, phase_spans, min_phase_duration_s)
    phase_spans = _limit_phase_spans(work, phase_spans, max_phases)

    work["coach_phase_id"] = -1
    work["coach_phase_name"] = ""
    work["coach_phase_type"] = ""
    work["coach_phase_label"] = ""
    work["coach_intensity_level"] = ""

    rows: list[dict[str, object]] = []
    for phase_idx, span in enumerate(phase_spans, start=1):
        start_i = span["start_i"]
        end_i = span["end_i"]
        phase = work.iloc[start_i:end_i]
        phase_summary = _phase_summary_row(
            phase,
            phase_id=phase_idx,
            hsr_threshold_mph=hsr_threshold_mph,
            sprint_threshold_mph=sprint_threshold_mph,
            accel_threshold_ms2=accel_threshold_ms2,
            decel_threshold_ms2=decel_threshold_ms2,
        )

        phase_type = _infer_phase_type(
            phase_summary,
            phase_idx=phase_idx,
            total_phases=len(phase_spans),
        )
        phase_name = f"Phase {phase_idx:02d}"
        phase_label = f"{phase_name}: {phase_type} ({phase_summary['intensity_level']})"
        rows.append(
            {
                "coach_phase_id": phase_idx,
                "coach_phase_name": phase_name,
                "coach_phase_type": phase_type,
                "coach_phase_label": phase_label,
                "phase_inference_note": (
                    "Inferred from speed/HSR/accel profile; verify against practice script."
                ),
                **phase_summary,
            }
        )

        phase_index = work.index[start_i:end_i]
        work.loc[phase_index, "coach_phase_id"] = phase_idx
        work.loc[phase_index, "coach_phase_name"] = phase_name
        work.loc[phase_index, "coach_phase_type"] = phase_type
        work.loc[phase_index, "coach_phase_label"] = phase_label
        work.loc[phase_index, "coach_intensity_level"] = phase_summary["intensity_level"]

    summary = pd.DataFrame(rows)
    return work, summary


def block_label(segment_id: int) -> str:
    """Neutral segment label for coach communication."""
    if 0 <= segment_id <= 25:
        return f"Block {chr(ord('A') + segment_id)}"
    return f"Block {segment_id + 1}"


def _gap_boundary_indices(df: pd.DataFrame, gap_threshold_s: float) -> list[int]:
    return df.index[df["dt_s"] > gap_threshold_s].tolist()


def _rest_boundary_indices(
    df: pd.DataFrame, rest_speed_threshold_mph: float, rest_min_duration_s: float
) -> list[int]:
    low_speed = df["speed_mph"] < rest_speed_threshold_mph
    groups = low_speed.ne(low_speed.shift(fill_value=False)).cumsum()

    boundaries: list[int] = []
    for _, run in df.groupby(groups):
        if not bool(low_speed.loc[run.index[0]]):
            continue
        duration_s = float(run["dt_s"].sum())
        if duration_s < rest_min_duration_s:
            continue
        start_i = int(run.index[0])
        end_i = int(run.index[-1]) + 1
        boundaries.extend([start_i, end_i])
    return boundaries


def _intensity_change_boundaries(
    df: pd.DataFrame,
    *,
    rolling_window_s: float,
    penalty: float,
    max_change_points: int,
) -> list[int]:
    if len(df) < 10:
        return []

    median_dt = float(df["dt_s"].median())
    if median_dt <= 0:
        return []
    window_samples = max(3, int(round(rolling_window_s / median_dt)))

    rolling_speed = df["speed_mph"].rolling(window_samples, center=True, min_periods=1).mean()

    # Downsample to 1 Hz before change-point search to keep runtime stable on long sessions.
    cp_series = (
        pd.Series(rolling_speed.to_numpy(), index=df["ts"])
        .resample("1s")
        .mean()
        .interpolate(limit_direction="both")
    )
    if len(cp_series) < 10:
        return []

    signal = cp_series.to_numpy().reshape(-1, 1)
    algo = rpt.Pelt(model="l2", min_size=5).fit(signal)
    bkps = algo.predict(pen=penalty)

    mapped: list[int] = []
    for bp in bkps:
        if bp <= 0 or bp >= len(cp_series):
            continue
        cp_ts = cp_series.index[bp - 1]
        idx = int(df["ts"].searchsorted(cp_ts, side="left"))
        if 0 < idx < len(df):
            mapped.append(idx)

    filtered = sorted(set(mapped))
    if len(filtered) > max_change_points:
        filtered = filtered[:max_change_points]
    return filtered


def _enforce_min_segment_duration(
    df: pd.DataFrame, boundaries: Sequence[int], min_segment_duration_s: float
) -> list[int]:
    kept = [boundaries[0]]
    for boundary in boundaries[1:]:
        start = kept[-1]
        end = boundary
        if end <= start:
            continue
        duration = float(df.iloc[start:end]["dt_s"].sum())
        if duration < min_segment_duration_s and end != len(df):
            continue
        kept.append(end)

    if kept[-1] != len(df):
        kept.append(len(df))
    return kept


def _segment_peak_distance(segment: pd.DataFrame, window_s: int) -> float:
    idx = segment.set_index("ts")["step_distance_yd_from_speed"]
    rolling = idx.rolling(f"{window_s}s", min_periods=1).sum()
    return float(rolling.max())


def _safe_name(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("%", "pct")
        .replace("/", "_")
    )


def _contiguous_phase_spans(df: pd.DataFrame) -> list[dict[str, float | int]]:
    if "segment_id" in df.columns:
        marker = df["segment_id"].ne(df["segment_id"].shift(fill_value=df["segment_id"].iloc[0])).cumsum()
    else:
        marker = pd.Series(0, index=df.index)

    spans: list[dict[str, float | int]] = []
    for _, run in df.groupby(marker, sort=True):
        start_i = int(run.index[0])
        end_i = int(run.index[-1]) + 1
        spans.append(_phase_span(df, start_i, end_i))
    return spans


def _phase_span(df: pd.DataFrame, start_i: int, end_i: int) -> dict[str, float | int]:
    chunk = df.iloc[start_i:end_i]
    return {
        "start_i": start_i,
        "end_i": end_i,
        "duration_s": float(chunk["dt_s"].sum()),
        "mean_speed_mph": float(chunk["speed_mph"].mean()),
    }


def _merge_short_phase_spans(
    df: pd.DataFrame,
    spans: list[dict[str, float | int]],
    min_phase_duration_s: float,
) -> list[dict[str, float | int]]:
    merged = spans.copy()
    while len(merged) > 1:
        short_idx = next(
            (i for i, span in enumerate(merged) if float(span["duration_s"]) < min_phase_duration_s),
            None,
        )
        if short_idx is None:
            break
        neighbor_idx = _best_neighbor_idx(merged, short_idx)
        merged = _merge_two_spans(df, merged, short_idx, neighbor_idx)
    return merged


def _limit_phase_spans(
    df: pd.DataFrame,
    spans: list[dict[str, float | int]],
    max_phases: int,
) -> list[dict[str, float | int]]:
    reduced = spans.copy()
    while len(reduced) > max_phases:
        merge_idx = min(
            range(len(reduced) - 1),
            key=lambda i: (
                abs(float(reduced[i]["mean_speed_mph"]) - float(reduced[i + 1]["mean_speed_mph"])),
                float(reduced[i]["duration_s"]) + float(reduced[i + 1]["duration_s"]),
            ),
        )
        reduced = _merge_two_spans(df, reduced, merge_idx, merge_idx + 1)
    return reduced


def _best_neighbor_idx(spans: list[dict[str, float | int]], idx: int) -> int:
    if idx == 0:
        return 1
    if idx == len(spans) - 1:
        return idx - 1

    left = spans[idx - 1]
    right = spans[idx + 1]
    current = spans[idx]
    left_score = (
        abs(float(current["mean_speed_mph"]) - float(left["mean_speed_mph"])),
        -float(left["duration_s"]),
    )
    right_score = (
        abs(float(current["mean_speed_mph"]) - float(right["mean_speed_mph"])),
        -float(right["duration_s"]),
    )
    return idx - 1 if left_score <= right_score else idx + 1


def _merge_two_spans(
    df: pd.DataFrame,
    spans: list[dict[str, float | int]],
    left_idx: int,
    right_idx: int,
) -> list[dict[str, float | int]]:
    low = min(left_idx, right_idx)
    high = max(left_idx, right_idx)
    merged_span = _phase_span(
        df,
        start_i=int(spans[low]["start_i"]),
        end_i=int(spans[high]["end_i"]),
    )
    return spans[:low] + [merged_span] + spans[high + 1 :]


def _phase_summary_row(
    phase: pd.DataFrame,
    *,
    phase_id: int,
    hsr_threshold_mph: float,
    sprint_threshold_mph: float,
    accel_threshold_ms2: float,
    decel_threshold_ms2: float,
) -> dict[str, object]:
    hsr_mask = phase["speed_mph"] >= hsr_threshold_mph
    sprint_mask = phase["speed_mph"] >= sprint_threshold_mph
    accel_mask = phase["signed_accel_ms2"] >= accel_threshold_ms2
    decel_mask = phase["signed_accel_ms2"] <= decel_threshold_ms2

    distance_yd = float(phase["step_distance_yd_from_speed"].sum())
    hsr_distance_yd = float(phase.loc[hsr_mask, "step_distance_yd_from_speed"].sum())
    sprint_distance_yd = float(phase.loc[sprint_mask, "step_distance_yd_from_speed"].sum())
    hsr_distance_pct = 0.0 if distance_yd <= 0 else (hsr_distance_yd / distance_yd) * 100.0
    intensity_level = _phase_intensity_level(
        mean_speed_mph=float(phase["speed_mph"].mean()),
        hsr_distance_pct=hsr_distance_pct,
    )

    return {
        "phase_id": phase_id,
        "start_ts_utc": phase["ts"].iloc[0],
        "end_ts_utc": phase["ts"].iloc[-1],
        "duration_s": float(phase["dt_s"].sum()),
        "n_samples": int(len(phase)),
        "distance_yd": distance_yd,
        "mean_speed_mph": float(phase["speed_mph"].mean()),
        "peak_speed_mph": float(phase["speed_mph"].max()),
        "peak_accel_ms2": float(phase["signed_accel_ms2"].max()),
        "peak_decel_ms2": float(phase["signed_accel_ms2"].min()),
        "peak_60s_distance_yd": _segment_peak_distance(phase, 60),
        "hsr_distance_yd": hsr_distance_yd,
        "sprint_distance_yd": sprint_distance_yd,
        "hsr_event_count": _count_boolean_runs(hsr_mask, phase["dt_s"], min_duration_s=1.0),
        "sprint_event_count": _count_boolean_runs(sprint_mask, phase["dt_s"], min_duration_s=1.0),
        "accel_event_count": int((accel_mask & ~accel_mask.shift(fill_value=False)).sum()),
        "decel_event_count": int((decel_mask & ~decel_mask.shift(fill_value=False)).sum()),
        "intensity_level": intensity_level,
    }


def _phase_intensity_level(mean_speed_mph: float, hsr_distance_pct: float) -> str:
    if mean_speed_mph >= 10.0 or hsr_distance_pct >= 12.0:
        return "High"
    if mean_speed_mph >= 6.0 or hsr_distance_pct >= 4.0:
        return "Moderate"
    return "Low"


def _infer_phase_type(
    phase_summary: dict[str, object],
    *,
    phase_idx: int,
    total_phases: int,
) -> str:
    duration_min = float(phase_summary["duration_s"]) / 60.0
    distance_yd = float(phase_summary["distance_yd"])
    mean_speed_mph = float(phase_summary["mean_speed_mph"])
    hsr_distance_yd = float(phase_summary["hsr_distance_yd"])
    sprint_distance_yd = float(phase_summary["sprint_distance_yd"])
    hsr_pct = (hsr_distance_yd / distance_yd * 100.0) if distance_yd > 0 else 0.0
    accel_events = int(phase_summary["accel_event_count"])
    decel_events = int(phase_summary["decel_event_count"])
    movement_event_rate = (accel_events + decel_events) / max(duration_min, 1e-9)

    if phase_idx == 1 and mean_speed_mph < 6.5 and hsr_pct < 5.0:
        return "Warm-up"
    if phase_idx == total_phases and mean_speed_mph < 5.5 and hsr_pct < 5.0:
        return "Cool-down / Walkthrough"
    if duration_min <= 2.0 and mean_speed_mph < 3.5:
        return "Reset / Rest"
    if mean_speed_mph >= 11.0 or hsr_pct >= 12.0 or sprint_distance_yd >= 25.0:
        return "Conditioning Push"
    if mean_speed_mph >= 8.0 or hsr_pct >= 5.0:
        return "Team Tempo"
    if movement_event_rate >= 10.0 and hsr_pct < 5.0:
        return "Individual / COD"
    return "Technical / Install"


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
