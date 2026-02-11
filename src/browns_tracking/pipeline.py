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

    if "step_distance_xy_yd" in work.columns:
        xy_speed_yd_s = np.where(dt_s > 0, work["step_distance_xy_yd"].astype(float) / dt_s, np.nan)
        jump_break = np.isfinite(xy_speed_yd_s) & (xy_speed_yd_s > float(max_plausible_xy_speed_yd_s))
    else:
        jump_break = np.zeros(len(work), dtype=bool)

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
            "step_distance_outlier_source": "N/A",
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

    step_distance_col = (
        "step_distance_xy_yd" if "step_distance_xy_yd" in df.columns else "step_distance_yd_from_speed"
    )
    step_distance = df[step_distance_col].astype(float)
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
        "step_distance_outlier_source": step_distance_col,
        "step_distance_outlier_threshold_yd": outlier_threshold,
        "step_distance_outlier_count": outlier_count,
        "step_distance_outlier_pct": float((outlier_count / sample_count) * 100.0),
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
    jump_ok = bool(by_key.loc["improbable_jump_pct", "status"] == "PASS")
    corr_ok = bool(by_key.loc["speed_alignment_corr", "status"] == "PASS")
    outlier_ok = bool(by_key.loc["speed_alignment_outlier_pct", "status"] == "PASS")

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

    return [
        f"Validation gates: {pass_count}/{total} PASS.",
        f"Trust policy: distance and workload totals are computed from {trust_source}.",
        spatial_note,
        speed_note,
    ]


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
