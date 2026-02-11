"""Visualization templates for coach/staff communication."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


def plot_movement_map(
    df: pd.DataFrame,
    *,
    segment_col: str = "coach_phase_label",
    x_col: str = "x",
    y_col: str = "y",
    highlight_top_n: int = 3,
    annotate_highlights: bool = True,
    highlight_windows: pd.DataFrame | None = None,
    continuity_col: str = "continuous_block_id",
    spatial_ok_col: str = "spatial_sample_ok",
    normalize_field: bool = True,
    field_length_yd: float = 120.0,
    field_width_yd: float = 53.3,
    show_role_hypothesis: bool = False,
    show_full_trace: bool = False,
    qc_status: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Template 1: density base + highlighted peak-demand windows."""
    del segment_col
    _require_columns(
        df,
        ["ts", x_col, y_col, continuity_col, spatial_ok_col],
        name="movement dataframe",
    )
    if highlight_windows is None or highlight_windows.empty:
        raise ValueError("highlight_windows must contain at least one top window row.")
    _require_columns(
        highlight_windows,
        ["window_start_utc", "window_end_utc", "distance_yd", "dominant_phase"],
        name="highlight_windows",
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12.0, 6.8), constrained_layout=True)

    plot_df = df.copy()
    plot_x_col = x_col
    plot_y_col = y_col
    if normalize_field:
        norm_x, norm_y = _normalize_field_coordinates(
            plot_df,
            x_col=x_col,
            y_col=y_col,
            field_length_yd=field_length_yd,
            field_width_yd=field_width_yd,
        )
        plot_df["_plot_x_norm"] = norm_x
        plot_df["_plot_y_norm"] = norm_y
        plot_x_col = "_plot_x_norm"
        plot_y_col = "_plot_y_norm"

    density_df = plot_df[plot_df[spatial_ok_col].astype(bool)]
    if density_df.empty:
        raise ValueError(f"No rows marked True in `{spatial_ok_col}`; cannot render density map.")

    hb = ax.hexbin(
        density_df[plot_x_col],
        density_df[plot_y_col],
        gridsize=65,
        bins="log",
        mincnt=1,
        cmap="Greys",
        alpha=0.35,
        zorder=1,
    )
    cbar = fig.colorbar(hb, ax=ax, pad=0.01)
    cbar.set_label("Log point density")

    if normalize_field:
        _draw_field_template(ax, field_length_yd=field_length_yd, field_width_yd=field_width_yd)

    plot_valid = plot_df[plot_df[spatial_ok_col].astype(bool)]
    if show_full_trace:
        for _, chunk in _iter_continuous_chunks(plot_valid, continuity_col):
            if len(chunk) < 2:
                continue
            ax.plot(
                chunk[plot_x_col],
                chunk[plot_y_col],
                linewidth=0.6,
                alpha=0.16,
                color="#6b7280",
                zorder=2,
            )

    window_colors = ("#00A676", "#FF7F11", "#3A86FF")
    windows = highlight_windows.head(highlight_top_n).reset_index(drop=True)
    for i, row in windows.iterrows():
        start = pd.to_datetime(row["window_start_utc"], utc=True)
        end = pd.to_datetime(row["window_end_utc"], utc=True)
        window_duration_s = int(row["window_s"]) if "window_s" in row and pd.notna(row["window_s"]) else int(
            max((end - start).total_seconds(), 0.0)
        )
        duration_label = _window_label(window_duration_s) if window_duration_s > 0 else "window"
        window_label = f"W{i + 1}({duration_label})"
        window_df = plot_df[(plot_df["ts"] >= start) & (plot_df["ts"] <= end)]
        window_df = window_df[window_df[spatial_ok_col].astype(bool)]
        if window_df.empty:
            continue

        color = window_colors[i % len(window_colors)]
        for chunk_idx, (_, chunk) in enumerate(_iter_continuous_chunks(window_df, continuity_col)):
            if len(chunk) < 2:
                continue
            ax.plot(
                chunk[plot_x_col],
                chunk[plot_y_col],
                linewidth=2.6,
                alpha=0.95,
                color=color,
                label=window_label if chunk_idx == 0 else None,
                zorder=4,
            )

        start_pt = window_df.iloc[0]
        end_pt = window_df.iloc[-1]
        ax.scatter(float(start_pt[plot_x_col]), float(start_pt[plot_y_col]), color=color, s=36, zorder=5)
        ax.scatter(
            float(end_pt[plot_x_col]),
            float(end_pt[plot_y_col]),
            color=color,
            s=50,
            marker="X",
            edgecolor="#222222",
            linewidth=0.4,
            zorder=5,
        )

        if annotate_highlights:
            mid = window_df.iloc[len(window_df) // 2]
            ax.annotate(
                window_label,
                xy=(float(mid[plot_x_col]), float(mid[plot_y_col])),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                color="#111111",
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "alpha": 0.9,
                    "edgecolor": color,
                },
                zorder=5,
            )

    if not plot_valid.empty:
        block_counts = plot_valid.groupby(continuity_col, sort=True).size()
        active_block_id = int(block_counts.idxmax()) if not block_counts.empty else None
        if active_block_id is not None:
            active_block = plot_valid[plot_valid[continuity_col] == active_block_id]
        else:
            active_block = plot_valid
        if not active_block.empty:
            start_x = float(active_block[plot_x_col].iloc[0])
            start_y = float(active_block[plot_y_col].iloc[0])
            end_x = float(active_block[plot_x_col].iloc[-1])
            end_y = float(active_block[plot_y_col].iloc[-1])
            ax.scatter(start_x, start_y, color="#2ca02c", s=44, zorder=5, label="Active block start")
            ax.scatter(end_x, end_y, color="#d62728", s=44, zorder=5, label="Active block end")

    if show_role_hypothesis:
        role_text = _build_role_hypothesis_text(
            density_df,
            x_col=plot_x_col,
            y_col=plot_y_col,
            field_length_yd=field_length_yd,
            field_width_yd=field_width_yd,
            normalized=normalize_field,
        )
        ax.text(
            0.98,
            0.02,
            role_text,
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=8.3,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.92, "edgecolor": "#d1d5db"},
        )

    if qc_status == "QC FAILED":
        ax.text(
            0.98,
            0.98,
            qc_status,
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=9.0,
            weight="bold",
            color="#b91c1c",
            bbox={"boxstyle": "round,pad=0.24", "facecolor": "#fee2e2", "edgecolor": "#ef4444", "alpha": 0.95},
        )

    ax.set_title("Spatial Usage: Density + Peak Demand Windows", fontsize=13, weight="bold")
    if normalize_field:
        ax.set_xlabel("Field width (yards, normalized)")
        ax.set_ylabel("Field length (yards, normalized)")
        ax.set_xlim(0.0, field_width_yd)
        ax.set_ylim(0.0, field_length_yd)
    else:
        ax.set_xlabel("X position (yards)")
        ax.set_ylabel("Y position (yards)")
        ax.set_xlim(-65, 25)
        ax.set_ylim(0, 130)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique: dict[str, object] = {}
        for handle, label in zip(handles, labels, strict=False):
            if label and label not in unique:
                unique[label] = handle
        if unique:
            ax.legend(unique.values(), unique.keys(), loc="upper left", frameon=True, ncols=2, fontsize=8.5)

    ax.set_aspect("equal", adjustable="box")
    return fig, ax


def _compact_phase_label(label: str) -> str:
    if ":" not in label:
        return label
    left, right = label.split(":", maxsplit=1)
    cleaned = right.strip().replace(" Intensity", "")
    return f"{left.strip()} ({cleaned})"


def _phase_intensity_label(label: str) -> str:
    text = str(label)
    if "High" in text:
        return "High"
    if "Moderate" in text:
        return "Moderate"
    if "Low" in text:
        return "Low"
    return _compact_phase_label(text)


def _draw_field_template(
    ax: plt.Axes,
    *,
    field_length_yd: float,
    field_width_yd: float,
) -> None:
    """Draw minimal field reference lines for spatial orientation."""
    x0, x1 = 0.0, field_width_yd
    y0, y1 = 0.0, field_length_yd
    line_color = "#9ca3af"
    ax.plot([x0, x1], [y0, y0], color=line_color, linewidth=1.0, alpha=0.85, zorder=0)
    ax.plot([x0, x1], [y1, y1], color=line_color, linewidth=1.0, alpha=0.85, zorder=0)
    ax.plot([x0, x0], [y0, y1], color=line_color, linewidth=1.0, alpha=0.85, zorder=0)
    ax.plot([x1, x1], [y0, y1], color=line_color, linewidth=1.0, alpha=0.85, zorder=0)

    mid_y = field_length_yd * 0.5
    third_y_1 = field_length_yd / 3.0
    third_y_2 = 2.0 * field_length_yd / 3.0
    ax.plot([x0, x1], [mid_y, mid_y], color="#6b7280", linewidth=1.1, linestyle="--", alpha=0.7, zorder=0)
    ax.plot([x0, x1], [third_y_1, third_y_1], color="#9ca3af", linewidth=0.8, linestyle=":", alpha=0.7, zorder=0)
    ax.plot([x0, x1], [third_y_2, third_y_2], color="#9ca3af", linewidth=0.8, linestyle=":", alpha=0.7, zorder=0)


def _normalize_field_coordinates(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    field_length_yd: float,
    field_width_yd: float,
) -> tuple[pd.Series, pd.Series]:
    """Rotate/translate/scale coordinates into an approximate standardized field frame."""
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return x, y

    coords = np.column_stack([x[valid].to_numpy(), y[valid].to_numpy()])
    center = np.nanmedian(coords, axis=0)
    centered = coords - center

    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    angle = np.arctan2(principal[1], principal[0])
    rotate_theta = (np.pi / 2.0) - angle
    rot = np.array(
        [
            [np.cos(rotate_theta), -np.sin(rotate_theta)],
            [np.sin(rotate_theta), np.cos(rotate_theta)],
        ]
    )
    rotated = centered @ rot.T

    if rotated[-1, 1] < rotated[0, 1]:
        rotated[:, 1] *= -1.0

    xq_low, xq_high = np.nanquantile(rotated[:, 0], [0.01, 0.99])
    yq_low, yq_high = np.nanquantile(rotated[:, 1], [0.01, 0.99])
    x_span = max(float(xq_high - xq_low), 1e-6)
    y_span = max(float(yq_high - yq_low), 1e-6)

    norm_x = ((rotated[:, 0] - xq_low) / x_span) * field_width_yd
    norm_y = ((rotated[:, 1] - yq_low) / y_span) * field_length_yd
    norm_x = np.clip(norm_x, 0.0, field_width_yd)
    norm_y = np.clip(norm_y, 0.0, field_length_yd)

    out_x = pd.Series(np.nan, index=df.index, dtype=float)
    out_y = pd.Series(np.nan, index=df.index, dtype=float)
    out_x.loc[valid] = norm_x
    out_y.loc[valid] = norm_y
    return out_x, out_y


def _build_role_hypothesis_text(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    field_length_yd: float,
    field_width_yd: float,
    normalized: bool,
) -> str:
    x = pd.to_numeric(df[x_col], errors="coerce").dropna()
    y = pd.to_numeric(df[y_col], errors="coerce").dropna()
    if x.empty or y.empty:
        return "Role hypothesis (inferred)\nInsufficient spatial samples."

    x_bins = pd.cut(
        x,
        bins=[-np.inf, field_width_yd / 3.0, 2.0 * field_width_yd / 3.0, np.inf],
        labels=["Left", "Middle", "Right"],
    )
    y_bins = pd.cut(
        y,
        bins=[-np.inf, field_length_yd / 3.0, 2.0 * field_length_yd / 3.0, np.inf],
        labels=["Back", "Middle", "Front"],
    )
    x_counts = x_bins.value_counts(dropna=True)
    y_counts = y_bins.value_counts(dropna=True)
    dom_x = str(x_counts.index[0]) if not x_counts.empty else "Middle"
    dom_y = str(y_counts.index[0]) if not y_counts.empty else "Middle"

    y_spread = float(np.nanquantile(y, 0.9) - np.nanquantile(y, 0.1))
    if dom_x in {"Left", "Right"} and y_spread >= (0.4 * field_length_yd):
        profile = "Wide-channel coverage profile"
    elif dom_x == "Middle" and y_spread >= (0.45 * field_length_yd):
        profile = "Central transit profile"
    elif dom_y == "Front":
        profile = "Advanced-area involvement profile"
    elif dom_y == "Back":
        profile = "Deeper support profile"
    else:
        profile = "Balanced central profile"

    frame_note = "normalized frame" if normalized else "raw frame"
    return (
        "Role hypothesis (inferred)\n"
        f"{profile}\n"
        f"Dominant zone: {dom_y}-{dom_x}\n"
        f"Confidence: low; single session, {frame_note}."
    )


def plot_intensity_timeline(
    df: pd.DataFrame,
    *,
    top_windows: pd.DataFrame | None = None,
    hsr_threshold_mph: float | None = None,
    phase_col: str = "coach_phase_label",
    show_phase_strip: bool = True,
    show_cumulative_distance: bool = False,
    show_hsr_ticks: bool = False,
    show_early_late_comparison: bool = False,
    display_resample_s: int = 1,
    distance_window_s: int = 60,
    qc_status: str | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes | None]]:
    """Template 2: stacked timeline with one metric family per axis (no mixed units)."""
    _require_columns(
        df,
        ["ts", "speed_mph", "step_distance_yd_from_speed", "continuous_block_id"],
        name="intensity dataframe",
    )
    if show_phase_strip:
        _require_columns(df, [phase_col], name="intensity dataframe")
        if not df[phase_col].notna().any():
            raise ValueError(f"`{phase_col}` has no non-null values; cannot render phase strip.")

    sns.set_theme(style="whitegrid")
    use_phase_strip = bool(show_phase_strip)
    display_df = _resample_within_blocks(
        df,
        resample_seconds=max(1, int(display_resample_s)),
        phase_col=phase_col if use_phase_strip else None,
    )
    if display_df.empty:
        display_df = df[
            ["ts", "speed_mph", "step_distance_yd_from_speed", "continuous_block_id"]
            + ([phase_col] if use_phase_strip else [])
        ].copy()

    window_seconds = max(10, int(distance_window_s))
    rate_df = _resample_within_blocks(df, resample_seconds=window_seconds)
    if rate_df.empty:
        rate_df = display_df[["ts", "step_distance_yd_from_speed", "continuous_block_id"]].copy()
    rate_df["distance_rate_yd_min"] = (
        pd.to_numeric(rate_df["step_distance_yd_from_speed"], errors="coerce").fillna(0.0)
        * (60.0 / float(window_seconds))
    )

    if use_phase_strip:
        fig, (speed_ax, rate_ax, phase_ax) = plt.subplots(
            3,
            1,
            figsize=(13.6, 7.2),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [3.2, 2.4, 0.9]},
        )
    else:
        fig, (speed_ax, rate_ax) = plt.subplots(
            2,
            1,
            figsize=(13.6, 6.4),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [3.2, 2.4]},
        )
        phase_ax = None

    top_window_patch: Patch | None = None
    if top_windows is not None and not top_windows.empty:
        _require_columns(top_windows, ["window_start_utc", "window_end_utc"], name="top_windows")
        window_colors = ("#00A676", "#FF7F11", "#3A86FF")
        window_duration_label = ""
        if "window_s" in top_windows.columns:
            duration_values = pd.to_numeric(top_windows["window_s"], errors="coerce").dropna().astype(int).unique()
            if len(duration_values) == 1:
                window_duration_label = f" ({_window_label(int(duration_values[0]))})"
        for i, row in top_windows.head(3).reset_index(drop=True).iterrows():
            color = window_colors[i % len(window_colors)]
            start = pd.to_datetime(row["window_start_utc"], utc=True)
            end = pd.to_datetime(row["window_end_utc"], utc=True)
            speed_ax.axvspan(start, end, color=color, alpha=0.18, zorder=0)
            rate_ax.axvspan(start, end, color=color, alpha=0.16, zorder=0)
        top_window_patch = Patch(
            facecolor=window_colors[0],
            alpha=0.18,
            label=f"Top 3 windows{window_duration_label}",
        )

    for chunk_idx, (_, chunk) in enumerate(_iter_continuous_chunks(display_df, "continuous_block_id")):
        if chunk.empty:
            continue
        speed_ax.plot(
            chunk["ts"],
            chunk["speed_mph"],
            color="#1f77b4",
            alpha=0.45,
            linewidth=1.0,
            label="Speed (display, mph)" if chunk_idx == 0 else None,
            zorder=2,
        )
        smooth = chunk.set_index("ts")["speed_mph"].rolling("60s", min_periods=1).mean()
        speed_ax.plot(
            smooth.index,
            smooth.values,
            color="#ff6b00",
            linewidth=2.1,
            label="60s mean speed (mph)" if chunk_idx == 0 else None,
            zorder=3,
        )

    if hsr_threshold_mph is not None:
        speed_ax.axhline(
            hsr_threshold_mph,
            color="#d62728",
            linestyle="--",
            linewidth=1.4,
            label=f"HSR threshold ({hsr_threshold_mph:.1f} mph)",
            zorder=2,
        )
        if show_hsr_ticks:
            hsr_mask = display_df["speed_mph"] >= hsr_threshold_mph
            block_break = display_df["continuous_block_id"].ne(
                display_df["continuous_block_id"].shift(fill_value=display_df["continuous_block_id"].iloc[0])
            )
            hsr_onset = hsr_mask & (~hsr_mask.shift(fill_value=False) | block_break)
            if hsr_onset.any():
                speed_ax.plot(
                    display_df.loc[hsr_onset, "ts"],
                    np.full(hsr_onset.sum(), -0.02),
                    "|",
                    transform=speed_ax.get_xaxis_transform(),
                    markersize=8,
                    color="#d62728",
                    alpha=0.5,
                    label="HSR onsets",
                )

    for chunk_idx, (_, chunk) in enumerate(_iter_continuous_chunks(rate_df, "continuous_block_id")):
        if chunk.empty:
            continue
        rate_ax.plot(
            chunk["ts"],
            chunk["distance_rate_yd_min"],
            color="#0f766e",
            alpha=0.85,
            linewidth=1.5,
            label=f"Distance rate ({window_seconds}s, yd/min)" if chunk_idx == 0 else None,
            zorder=2,
        )
        rate_smooth = chunk.set_index("ts")["distance_rate_yd_min"].rolling("180s", min_periods=1).mean()
        rate_ax.plot(
            rate_smooth.index,
            rate_smooth.values,
            color="#111827",
            alpha=0.85,
            linewidth=1.2,
            linestyle="-.",
            label="3-min mean distance rate" if chunk_idx == 0 else None,
            zorder=3,
        )

    cumulative_ax = None
    if show_cumulative_distance:
        cumulative_ax = rate_ax.twinx()
        cumulative = df.set_index("ts")["step_distance_yd_from_speed"].cumsum()
        cumulative_ax.plot(
            cumulative.index,
            cumulative.values,
            color="#4b5563",
            linewidth=1.1,
            alpha=0.65,
            label="Cumulative distance (yd)",
            zorder=1,
        )
        cumulative_ax.set_ylabel("Cumulative distance (yd)")
        cumulative_ax.grid(False)

    if phase_ax is not None:
        spans = _contiguous_spans(display_df, phase_col, continuity_col="continuous_block_id")
        for span_idx, (label, start, end) in enumerate(spans):
            color = _phase_color(label, span_idx)
            phase_ax.axvspan(start, end, color=color, alpha=0.85)
            duration_min = max((end - start).total_seconds() / 60.0, 0.0)
            if duration_min >= 3.0:
                midpoint = start + ((end - start) / 2)
                phase_ax.text(
                    midpoint,
                    0.5,
                    _phase_intensity_label(str(label)),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#111111",
                )
        phase_ax.set_ylim(0.0, 1.0)
        phase_ax.set_yticks([])
        phase_ax.set_ylabel("Phase")
        phase_ax.set_xlabel("Time (UTC)")
        phase_ax.grid(False)
        phase_ax.legend(
            handles=[
                Patch(facecolor=_phase_color("Low", 0), label="Low"),
                Patch(facecolor=_phase_color("Moderate", 0), label="Moderate"),
                Patch(facecolor=_phase_color("High", 0), label="High"),
            ],
            loc="upper right",
            frameon=False,
            ncols=3,
            fontsize=8,
        )
    else:
        rate_ax.set_xlabel("Time (UTC)")

    if show_early_late_comparison and "step_distance_yd_from_speed" in df.columns:
        midpoint = df["ts"].iloc[0] + ((df["ts"].iloc[-1] - df["ts"].iloc[0]) / 2)
        early = df[df["ts"] <= midpoint]
        late = df[df["ts"] > midpoint]
        if not early.empty and not late.empty:
            early_min = max(float(early["dt_s"].sum()) / 60.0, 1e-9)
            late_min = max(float(late["dt_s"].sum()) / 60.0, 1e-9)
            early_rate = float(early["step_distance_yd_from_speed"].sum()) / early_min
            late_rate = float(late["step_distance_yd_from_speed"].sum()) / late_min
            delta_pct = ((late_rate / early_rate) - 1.0) * 100.0 if early_rate > 0 else 0.0
            rate_ax.text(
                0.985,
                0.95,
                (
                    "Early vs Late rate\n"
                    f"{early_rate:.1f} -> {late_rate:.1f} yd/min ({delta_pct:+.1f}%)"
                ),
                transform=rate_ax.transAxes,
                va="top",
                ha="right",
                fontsize=8.3,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9, "edgecolor": "#d1d5db"},
            )

    if qc_status == "QC FAILED":
        speed_ax.text(
            0.015,
            0.97,
            qc_status,
            transform=speed_ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.8,
            weight="bold",
            color="#b91c1c",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "#fee2e2", "edgecolor": "#ef4444", "alpha": 0.95},
        )

    speed_ax.set_title("Session Intensity Timeline", fontsize=13, weight="bold")
    speed_ax.set_ylabel("Speed (mph)")
    speed_upper = float(display_df["speed_mph"].max()) if not display_df.empty else 0.0
    speed_ax.set_ylim(0.0, max(15.5, speed_upper * 1.2))
    speed_ax.grid(alpha=0.2)

    rate_ax.set_ylabel("Distance rate (yd/min)")
    rate_upper = float(rate_df["distance_rate_yd_min"].max()) if not rate_df.empty else 0.0
    rate_ax.set_ylim(0.0, max(20.0, rate_upper * 1.3))
    rate_ax.grid(alpha=0.22)

    speed_handles, speed_labels = speed_ax.get_legend_handles_labels()
    if top_window_patch is not None:
        speed_handles.append(top_window_patch)
        speed_labels.append(top_window_patch.get_label())
    speed_unique: dict[str, object] = {}
    for handle, label in zip(speed_handles, speed_labels, strict=False):
        if label not in speed_unique:
            speed_unique[label] = handle
    speed_ax.legend(speed_unique.values(), speed_unique.keys(), loc="upper left", frameon=True, fontsize=8.2)

    rate_handles, rate_labels = rate_ax.get_legend_handles_labels()
    if cumulative_ax is not None:
        cumulative_handles, cumulative_labels = cumulative_ax.get_legend_handles_labels()
        rate_handles.extend(cumulative_handles)
        rate_labels.extend(cumulative_labels)
    rate_unique: dict[str, object] = {}
    for handle, label in zip(rate_handles, rate_labels, strict=False):
        if label not in rate_unique:
            rate_unique[label] = handle
    rate_ax.legend(rate_unique.values(), rate_unique.keys(), loc="upper left", frameon=True, fontsize=8.2)
    return fig, (speed_ax, phase_ax)


def plot_session_structure_map(
    structure_map_table: pd.DataFrame,
    *,
    x_col: str = "distance_per_min_yd",
    y_col: str = "hsr_distance_per_min_yd",
    size_col: str = "duration_min",
    hue_col: str = "dominant_field_zone",
) -> tuple[plt.Figure, plt.Axes]:
    """Supplemental visual: drill taxonomy map for phase-level work structure."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)

    if structure_map_table.empty:
        ax.text(0.5, 0.5, "No phase structure data available", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    plot_df = structure_map_table.copy().sort_values(size_col)
    sns.scatterplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        size=size_col,
        sizes=(120, 850),
        hue=hue_col,
        palette="Set2",
        alpha=0.85,
        edgecolor="#111111",
        linewidth=0.5,
        ax=ax,
    )

    for row in plot_df.itertuples(index=False):
        ax.annotate(
            str(row.coach_phase_name),
            xy=(float(getattr(row, x_col)), float(getattr(row, y_col))),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="#111111",
        )

    ax.set_title("Session Structure Map: Work Rate vs HSR Density", fontsize=13, weight="bold")
    ax.set_xlabel("Distance per minute (yd/min)")
    ax.set_ylabel("HSR distance per minute (yd/min)")
    ax.grid(alpha=0.25)
    return fig, ax


def plot_phase_speed_band_stack(
    df: pd.DataFrame,
    *,
    phase_col: str = "coach_phase_label",
    speed_col: str = "speed_mph",
    distance_col: str = "step_distance_yd_from_speed",
    speed_bands: Sequence[tuple[str, float, float | None]] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Bonus visual: stacked speed-band distance share by coach phase."""
    _require_columns(df, [phase_col, speed_col, distance_col], name="phase speed dataframe")
    if speed_bands is None:
        speed_bands = [
            ("Walk", 0.0, 3.0),
            ("Cruise", 3.0, 9.0),
            ("Run", 9.0, 13.0),
            ("HSR", 13.0, 16.0),
            ("Sprint", 16.0, None),
        ]

    work = df[[phase_col, speed_col, distance_col]].copy()
    work["speed_band"] = "Out of range"
    speed = pd.to_numeric(work[speed_col], errors="coerce")
    for band_name, lower, upper in speed_bands:
        if upper is None:
            mask = speed >= lower
        else:
            mask = (speed >= lower) & (speed < upper)
        work.loc[mask, "speed_band"] = band_name

    grouped = (
        work.groupby([phase_col, "speed_band"], dropna=False)[distance_col]
        .sum()
        .reset_index(name="distance_yd")
    )
    pivot = grouped.pivot(index=phase_col, columns="speed_band", values="distance_yd").fillna(0.0)
    order = [name for name, *_ in speed_bands if name in pivot.columns]
    pivot = pivot[[col for col in order if col in pivot.columns]]
    pct = pivot.div(pivot.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0) * 100.0

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12.8, 5.8), constrained_layout=True)
    colors = ["#9ecae1", "#6baed6", "#3182bd", "#fd8d3c", "#e31a1c"]
    pct.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=colors[: len(pct.columns)],
        width=0.75,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_title("Phase Build: Speed-Band Distance Share", fontsize=12.5, weight="bold")
    ax.set_ylabel("Distance share (%)")
    ax.set_xlabel("Coach phase")
    ax.set_ylim(0.0, 100.0)
    ax.legend(title="Speed band", bbox_to_anchor=(1.0, 1.0), loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    return fig, ax


def plot_peak_demand_summary(
    distance_table: pd.DataFrame,
    extrema_table: pd.DataFrame,
    *,
    peak_windows: pd.DataFrame | None = None,
    qc_status: str | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Template 3: peak intensity curve plus actionable top-window context."""
    _require_columns(distance_table, ["window_label", "best_distance_yd"], name="distance_table")
    _require_columns(extrema_table, ["metric", "value", "ts_utc"], name="extrema_table")
    if peak_windows is not None and not peak_windows.empty:
        required_peak_cols = {"window_rank", "window_start_utc", "window_end_utc"}
        if not required_peak_cols.issubset(peak_windows.columns):
            raise ValueError(
                f"peak_windows is missing required columns: {sorted(required_peak_cols - set(peak_windows.columns))}"
            )

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.7),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.8, 1.2]},
    )

    ax_left, ax_right = axes
    chart_df = distance_table.copy()
    if "window_s" in chart_df.columns:
        chart_df["window_s"] = pd.to_numeric(chart_df["window_s"], errors="coerce")
        chart_df = chart_df.sort_values("window_s")
    if "best_intensity_yd_per_min" not in chart_df.columns:
        chart_df["best_intensity_yd_per_min"] = (
            pd.to_numeric(chart_df["best_distance_yd"], errors="coerce")
            * (60.0 / pd.to_numeric(chart_df["window_s"], errors="coerce").replace(0.0, np.nan))
        ).fillna(0.0)

    x_labels = chart_df["window_label"].astype(str).tolist()
    x_pos = np.arange(len(chart_df))
    intensities = pd.to_numeric(chart_df["best_intensity_yd_per_min"], errors="coerce").fillna(0.0).to_numpy()
    distances = pd.to_numeric(chart_df["best_distance_yd"], errors="coerce").fillna(0.0).to_numpy()

    ax_left.plot(
        x_pos,
        intensities,
        color="#1d4ed8",
        linewidth=2.4,
        marker="o",
        markersize=7,
        label="Peak intensity (yd/min)",
    )
    ax_left.fill_between(x_pos, 0.0, intensities, color="#bfdbfe", alpha=0.45, zorder=1)
    for idx, (rate_val, dist_val) in enumerate(zip(intensities, distances, strict=False)):
        ax_left.annotate(
            f"{rate_val:.0f}",
            (x_pos[idx], rate_val),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 6),
            textcoords="offset points",
            color="#1e3a8a",
        )
        ax_left.annotate(
            f"{dist_val:.0f} yd",
            (x_pos[idx], max(rate_val * 0.08, 0.5)),
            ha="center",
            va="bottom",
            fontsize=8.4,
            color="#334155",
        )

    ax_left.set_title("Peak Running Intensity by Window", fontsize=12, weight="bold")
    ax_left.set_xlabel("Window duration")
    ax_left.set_ylabel("Peak intensity (yd/min)")
    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels(x_labels)
    peak_upper = float(np.nanmax(intensities)) if len(intensities) else 0.0
    ax_left.set_ylim(0.0, max(40.0, peak_upper * 1.25))
    ax_left.grid(axis="y", alpha=0.25)
    ax_left.legend(loc="upper right", frameon=True, fontsize=8.5)

    ax_right.axis("off")
    ax_right.set_title("Session Peaks and Top Windows", fontsize=12, weight="bold", pad=12)

    lines = ["Session extrema"]
    for row in extrema_table.itertuples(index=False):
        ts_text = pd.Timestamp(row.ts_utc).strftime("%H:%M:%S")
        metric = str(row.metric)
        value = float(row.value)
        if metric == "Max speed (mph)":
            lines.append(f"Max speed {value:.1f} @ {ts_text}")
        elif metric == "Max accel (m/s^2)":
            lines.append(f"Max accel {value:+.1f} @ {ts_text}")
        elif metric == "Max decel (m/s^2)":
            lines.append(f"Max decel {value:.1f} @ {ts_text}")
        else:
            lines.append(f"{metric} {value:.1f} @ {ts_text}")

    if peak_windows is not None and not peak_windows.empty:
        lines.append("")
        lines.append("Top windows")
        windows_df = peak_windows.copy()
        if "window_s" in windows_df.columns:
            windows_df["window_s"] = pd.to_numeric(windows_df["window_s"], errors="coerce")
            windows_df = windows_df.sort_values(["window_s", "window_rank"])
            for window_s, chunk in windows_df.groupby("window_s", sort=True):
                if pd.isna(window_s):
                    continue
                lines.append(f"{_window_label(int(window_s))}:")
                for row in chunk.head(3).itertuples(index=False):
                    start = pd.Timestamp(row.window_start_utc).strftime("%H:%M")
                    end = pd.Timestamp(row.window_end_utc).strftime("%H:%M")
                    distance_yd = (
                        float(getattr(row, "distance_yd"))
                        if hasattr(row, "distance_yd") and pd.notna(getattr(row, "distance_yd"))
                        else float(getattr(row, "value"))
                    )
                    rate_yd_min = (
                        float(distance_yd) * (60.0 / float(window_s))
                        if float(window_s) > 0
                        else 0.0
                    )
                    hsr_count = int(getattr(row, "hsr_event_count")) if hasattr(row, "hsr_event_count") else 0
                    accel_count = int(getattr(row, "accel_event_count")) if hasattr(row, "accel_event_count") else 0
                    decel_count = int(getattr(row, "decel_event_count")) if hasattr(row, "decel_event_count") else 0
                    lines.append(
                        f"W{int(row.window_rank)} {start}-{end} | {rate_yd_min:.0f} yd/min ({distance_yd:.0f} yd) | "
                        f"HSR{hsr_count} | A/D {accel_count}/{decel_count}"
                    )
        else:
            for row in windows_df.sort_values("window_rank").head(3).itertuples(index=False):
                start = pd.Timestamp(row.window_start_utc).strftime("%H:%M")
                end = pd.Timestamp(row.window_end_utc).strftime("%H:%M")
                distance_yd = (
                    float(getattr(row, "distance_yd"))
                    if hasattr(row, "distance_yd") and pd.notna(getattr(row, "distance_yd"))
                    else float(getattr(row, "value"))
                )
                window_s = int(getattr(row, "window_s")) if hasattr(row, "window_s") and pd.notna(getattr(row, "window_s")) else 60
                rate_yd_min = float(distance_yd) * (60.0 / float(window_s)) if window_s > 0 else 0.0
                hsr_count = int(getattr(row, "hsr_event_count")) if hasattr(row, "hsr_event_count") else 0
                accel_count = int(getattr(row, "accel_event_count")) if hasattr(row, "accel_event_count") else 0
                decel_count = int(getattr(row, "decel_event_count")) if hasattr(row, "decel_event_count") else 0
                lines.append(
                    f"W{int(row.window_rank)} {start}-{end} | {rate_yd_min:.0f} yd/min ({distance_yd:.0f} yd) | "
                    f"HSR{hsr_count} | A/D {accel_count}/{decel_count}"
                )

    ax_right.text(
        0.03,
        0.97,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=10.5,
        family="monospace",
    )

    if qc_status is not None and qc_status == "QC FAILED":
        ax_left.text(
            0.98,
            0.98,
            qc_status,
            transform=ax_left.transAxes,
            va="top",
            ha="right",
            fontsize=8.8,
            weight="bold",
            color="#b91c1c",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "#fee2e2", "edgecolor": "#ef4444", "alpha": 0.95},
        )

    fig.suptitle("Peak Demand Summary", fontsize=14, weight="bold")
    return fig, (ax_left, ax_right)


def plot_qc_overview(
    df: pd.DataFrame,
    *,
    gap_threshold_s: float = 0.15,
    qc_status: str | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Compact QC plot with cadence/gaps and speed-vs-distance diagnostics."""
    _require_columns(df, ["ts", "dt_s", "speed_mph", "step_distance_yd_from_speed"], name="qc dataframe")
    sns.set_theme(style="whitegrid")
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(13.5, 6.1),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    ax_top.plot(df["ts"], df["dt_s"], color="#1f77b4", linewidth=0.9, alpha=0.9, label="dt (s)")
    ax_top.axhline(gap_threshold_s, color="#d62728", linestyle="--", linewidth=1.2, label=f"Gap threshold {gap_threshold_s:.2f}s")
    gap_rows = df[df["dt_s"] > gap_threshold_s]
    if not gap_rows.empty:
        ax_top.scatter(gap_rows["ts"], gap_rows["dt_s"], color="#d62728", s=12, alpha=0.7, label="Flagged gaps")
    ax_top.set_ylabel("Cadence dt (s)")
    ax_top.set_title("QC Overview: Cadence, Gaps, and Distance Diagnostics", fontsize=12, weight="bold")
    ax_top.legend(loc="upper left", frameon=True)

    speed_smooth = df.set_index("ts")["speed_mph"].rolling("60s", min_periods=1).mean()
    ax_bottom.plot(speed_smooth.index, speed_smooth.values, color="#ff6b00", linewidth=1.8, label="Speed 60s mean (mph)")
    ax_bottom.set_ylabel("Speed (mph)")
    ax_bottom.set_xlabel("Time (UTC)")

    if "step_distance_yd_vendor_dis" in df.columns and df["step_distance_yd_vendor_dis"].notna().any():
        speed_roll = df.set_index("ts")["step_distance_yd_from_speed"].rolling("60s", min_periods=1).sum()
        dis_roll = df.set_index("ts")["step_distance_yd_vendor_dis"].rolling("60s", min_periods=1).sum()
        denom = speed_roll.clip(lower=1e-6)
        delta_pct = ((dis_roll - speed_roll).abs() / denom) * 100.0
        right_ax = ax_bottom.twinx()
        right_ax.plot(delta_pct.index, delta_pct.values, color="#2a9d8f", linewidth=1.2, alpha=0.85, label="`dis` vs `s*dt` delta (%)")
        right_ax.set_ylabel("Distance delta (%)")
        right_ax.grid(False)

        left_handles, left_labels = ax_bottom.get_legend_handles_labels()
        right_handles, right_labels = right_ax.get_legend_handles_labels()
        ax_bottom.legend(left_handles + right_handles, left_labels + right_labels, loc="upper left", frameon=True)
    else:
        ax_bottom.legend(loc="upper left", frameon=True)

    if qc_status is not None and qc_status != "QC PASS":
        ax_top.text(
            0.98,
            0.95,
            qc_status,
            transform=ax_top.transAxes,
            ha="right",
            va="top",
            fontsize=9.0,
            weight="bold",
            color="#b91c1c",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "#fee2e2", "edgecolor": "#ef4444", "alpha": 0.95},
        )

    return fig, (ax_top, ax_bottom)


def _contiguous_spans(
    df: pd.DataFrame,
    phase_col: str,
    *,
    continuity_col: str | None = None,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    _require_columns(df, [phase_col], name="phase dataframe")
    if df.empty:
        return []
    spans: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    phase_change = df[phase_col].ne(df[phase_col].shift(fill_value=df[phase_col].iloc[0]))
    continuity_change = pd.Series(False, index=df.index, dtype=bool)
    if continuity_col is not None:
        _require_columns(df, [continuity_col], name="phase dataframe")
        continuity_change = df[continuity_col].ne(
            df[continuity_col].shift(fill_value=df[continuity_col].iloc[0])
        )
    marker = (phase_change | continuity_change).cumsum()
    for _, chunk in df.groupby(marker, sort=True):
        label = str(chunk[phase_col].iloc[0])
        spans.append((label, chunk["ts"].iloc[0], chunk["ts"].iloc[-1]))
    return spans


def _phase_color(label: str, idx: int) -> tuple[float, float, float]:
    if "High" in label:
        return sns.color_palette("Reds", n_colors=4)[2]
    if "Moderate" in label:
        return sns.color_palette("YlOrBr", n_colors=4)[2]
    if "Low" in label:
        return sns.color_palette("Blues", n_colors=4)[1]
    return sns.color_palette("Set2", n_colors=8)[idx % 8]


def _window_label(window_s: int) -> str:
    if window_s % 60 == 0:
        mins = window_s // 60
        return f"{mins}m" if mins > 1 else "1m"
    return f"{window_s}s"


def save_figure(fig: plt.Figure, output_path: str | Path, dpi: int = 300) -> None:
    """Save a figure with a transparent-friendly white background."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")


def close_figures(figures: Sequence[plt.Figure]) -> None:
    """Close a batch of figures to keep notebook/script memory stable."""
    for fig in figures:
        plt.close(fig)


def _iter_continuous_chunks(
    df: pd.DataFrame,
    continuity_col: str,
) -> list[tuple[int, pd.DataFrame]]:
    if df.empty:
        return []
    _require_columns(df, [continuity_col], name="continuous dataframe")
    return [
        (int(block_id), chunk)
        for block_id, chunk in df.groupby(continuity_col, sort=True)
        if not chunk.empty
    ]


def _resample_within_blocks(
    df: pd.DataFrame,
    *,
    resample_seconds: int,
    phase_col: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        columns = ["ts", "speed_mph", "step_distance_yd_from_speed", "continuous_block_id"]
        if phase_col is not None:
            columns.append(phase_col)
        return pd.DataFrame(columns=columns)

    agg: dict[str, object] = {
        "speed_mph": "mean",
        "step_distance_yd_from_speed": "sum",
    }
    if phase_col is not None and phase_col in df.columns:
        agg[phase_col] = (
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.dropna().iloc[-1] if s.notna().any() else ""
        )

    rows: list[pd.DataFrame] = []
    window = f"{int(max(1, resample_seconds))}s"
    for block_id, chunk in df.groupby("continuous_block_id", sort=True):
        if chunk.empty:
            continue
        resampled = chunk.set_index("ts").resample(window).agg(agg)
        resampled["continuous_block_id"] = int(block_id)
        resampled = resampled.dropna(subset=["speed_mph"])
        if resampled.empty:
            continue
        rows.append(resampled.reset_index())

    if not rows:
        columns = ["ts", "speed_mph", "step_distance_yd_from_speed", "continuous_block_id"]
        if phase_col is not None:
            columns.append(phase_col)
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True).sort_values("ts").reset_index(drop=True)


def _require_columns(df: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
