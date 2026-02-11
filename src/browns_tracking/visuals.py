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
) -> tuple[plt.Figure, plt.Axes]:
    """Template 1: density map + neutral path + highlighted peak-demand windows/phases."""
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

    density_df = df[df[spatial_ok_col].astype(bool)]
    if density_df.empty:
        raise ValueError(f"No rows marked True in `{spatial_ok_col}`; cannot render density map.")

    hb = ax.hexbin(
        density_df[x_col],
        density_df[y_col],
        gridsize=65,
        bins="log",
        mincnt=1,
        cmap="Greys",
        alpha=0.35,
        zorder=1,
    )
    cbar = fig.colorbar(hb, ax=ax, pad=0.01)
    cbar.set_label("Log point density")

    for _, chunk in _iter_continuous_chunks(df, continuity_col):
        if len(chunk) < 2:
            continue
        ax.plot(
            chunk[x_col],
            chunk[y_col],
            linewidth=0.7,
            alpha=0.20,
            color="#6b7280",
            zorder=2,
        )

    window_colors = ("#00A676", "#FF7F11", "#3A86FF")
    windows = highlight_windows.head(highlight_top_n).reset_index(drop=True)
    window_notes: list[str] = []
    for i, row in windows.iterrows():
        start = pd.to_datetime(row["window_start_utc"], utc=True)
        end = pd.to_datetime(row["window_end_utc"], utc=True)
        window_df = df[(df["ts"] >= start) & (df["ts"] <= end)]
        if window_df.empty:
            raise ValueError(
                f"Top window {i + 1} ({start} to {end}) has no matching rows in movement dataframe."
            )

        color = window_colors[i % len(window_colors)]
        for _, chunk in _iter_continuous_chunks(window_df, continuity_col):
            if len(chunk) < 2:
                continue
            ax.plot(
                chunk[x_col],
                chunk[y_col],
                linewidth=2.6,
                alpha=0.95,
                color=color,
                zorder=4,
            )

        start_pt = window_df.iloc[0]
        end_pt = window_df.iloc[-1]
        ax.scatter(float(start_pt[x_col]), float(start_pt[y_col]), color=color, s=36, zorder=5)
        ax.scatter(
            float(end_pt[x_col]),
            float(end_pt[y_col]),
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
                f"W{i + 1}",
                xy=(float(mid[x_col]), float(mid[y_col])),
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

        start_str = start.strftime("%H:%M")
        end_str = end.strftime("%H:%M")
        window_notes.append(
            f"W{i + 1} {start_str}-{end_str} | {float(row['distance_yd']):.0f} yd | "
            f"{_compact_phase_label(str(row['dominant_phase']))}"
        )

    start_x = float(df[x_col].iloc[0])
    start_y = float(df[y_col].iloc[0])
    end_x = float(df[x_col].iloc[-1])
    end_y = float(df[y_col].iloc[-1])
    ax.scatter(start_x, start_y, color="#2ca02c", s=50, zorder=5)
    ax.scatter(end_x, end_y, color="#d62728", s=50, zorder=5)
    ax.annotate("Start", xy=(start_x, start_y), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.annotate("End", xy=(end_x, end_y), xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.text(
        0.02,
        0.02,
        "Top windows\n" + "\n".join(window_notes),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.92, "edgecolor": "#d1d5db"},
    )

    ax.set_title("Spatial Usage: Density + Peak Demand Windows", fontsize=13, weight="bold")
    ax.set_xlabel("X position (yards)")
    ax.set_ylabel("Y position (yards)")
    ax.set_xlim(-65, 25)
    ax.set_ylim(0, 130)
    ax.set_aspect("equal", adjustable="box")
    return fig, ax


def _compact_phase_label(label: str) -> str:
    if ":" not in label:
        return label
    left, right = label.split(":", maxsplit=1)
    return f"{left.strip()} ({right.strip().replace(' Intensity', '')})"


def plot_intensity_timeline(
    df: pd.DataFrame,
    *,
    top_windows: pd.DataFrame | None = None,
    hsr_threshold_mph: float | None = None,
    phase_col: str = "coach_phase_label",
    show_phase_strip: bool = True,
    show_cumulative_distance: bool = False,
    show_hsr_ticks: bool = True,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes | None]]:
    """Template 2: intensity timeline with phase strip, HSR ticks, and cumulative distance."""
    _require_columns(
        df,
        ["ts", "speed_mph", "continuous_block_id"],
        name="intensity dataframe",
    )
    if show_phase_strip:
        _require_columns(df, [phase_col], name="intensity dataframe")
        if not df[phase_col].notna().any():
            raise ValueError(f"`{phase_col}` has no non-null values; cannot render phase strip.")

    sns.set_theme(style="whitegrid")
    use_phase_strip = show_phase_strip
    y_max = 15.5

    if use_phase_strip:
        fig, (ax, phase_ax) = plt.subplots(
            2,
            1,
            figsize=(13.5, 6.4),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [5.0, 1.0]},
        )
    else:
        fig, ax = plt.subplots(figsize=(13.5, 5.6), constrained_layout=True)
        phase_ax = None

    for chunk_idx, (_, chunk) in enumerate(_iter_continuous_chunks(df, "continuous_block_id")):
        if chunk.empty:
            continue
        ax.plot(
            chunk["ts"],
            chunk["speed_mph"],
            color="#1f77b4",
            alpha=0.25,
            linewidth=0.6,
            label="Speed (mph)" if chunk_idx == 0 else None,
        )
        smooth = chunk.set_index("ts")["speed_mph"].rolling("60s", min_periods=1).mean()
        ax.plot(
            smooth.index,
            smooth.values,
            color="#ff6b00",
            linewidth=2.3,
            label="60s rolling mean" if chunk_idx == 0 else None,
        )

    if hsr_threshold_mph is not None:
        ax.axhline(
            hsr_threshold_mph,
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label=f"HSR threshold ({hsr_threshold_mph:.1f} mph)",
        )
        if show_hsr_ticks:
            hsr_mask = df["speed_mph"] >= hsr_threshold_mph
            block_break = df["continuous_block_id"].ne(
                df["continuous_block_id"].shift(fill_value=df["continuous_block_id"].iloc[0])
            )
            hsr_onset = hsr_mask & (~hsr_mask.shift(fill_value=False) | block_break)
            if hsr_onset.any():
                ax.plot(
                    df.loc[hsr_onset, "ts"],
                    np.full(hsr_onset.sum(), -0.02),
                    "|",
                    transform=ax.get_xaxis_transform(),
                    markersize=8,
                    color="#d62728",
                    alpha=0.45,
                    label="HSR onsets",
                )

    if top_windows is not None and not top_windows.empty:
        _require_columns(top_windows, ["window_start_utc", "window_end_utc"], name="top_windows")
        window_colors = ("#00A676", "#FF7F11", "#3A86FF")
        for i, row in top_windows.head(3).reset_index(drop=True).iterrows():
            color = window_colors[i % len(window_colors)]
            start = pd.to_datetime(row["window_start_utc"], utc=True)
            end = pd.to_datetime(row["window_end_utc"], utc=True)
            ax.axvspan(
                start,
                end,
                color=color,
                alpha=0.20,
                label="Top 3 windows" if i == 0 else None,
            )
            midpoint = start + ((end - start) / 2)
            ax.text(
                midpoint,
                15.2,
                f"W{i + 1}",
                ha="center",
                va="top",
                fontsize=8.5,
                color="#111111",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.85, "edgecolor": color},
            )

    right_ax = None
    if show_cumulative_distance and "step_distance_yd_from_speed" in df.columns:
        right_ax = ax.twinx()
        cumulative = df["step_distance_yd_from_speed"].cumsum()
        right_ax.plot(
            df["ts"],
            cumulative,
            color="#2a9d8f",
            linewidth=1.2,
            alpha=0.85,
            label="Cumulative distance (yd)",
        )
        right_ax.set_ylabel("Cumulative distance (yd)")
        right_ax.grid(False)

    if phase_ax is not None:
        spans = _contiguous_spans(df, phase_col, continuity_col="continuous_block_id")
        for span_idx, (label, start, end) in enumerate(spans):
            color = _phase_color(label, span_idx)
            phase_ax.axvspan(start, end, color=color, alpha=0.85)
            duration_min = max((end - start).total_seconds() / 60.0, 0.0)
            if duration_min >= 6.0:
                midpoint = start + ((end - start) / 2)
                phase_ax.text(
                    midpoint,
                    0.5,
                    _compact_phase_label(str(label)),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#111111",
                )
        phase_ax.set_ylim(0.0, 1.0)
        phase_ax.set_yticks([])
        phase_ax.set_ylabel("Phases")
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

    ax.set_title("Session Intensity Timeline", fontsize=13, weight="bold")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Speed (mph)")
    ax.set_ylim(0.0, y_max)

    peak_idx = int(df["speed_mph"].idxmax())
    peak_ts = df.loc[peak_idx, "ts"]
    peak_speed = float(df.loc[peak_idx, "speed_mph"])
    if peak_speed > y_max:
        ax.annotate(
            f"Peak {peak_speed:.1f} mph (clipped)",
            xy=(peak_ts, y_max - 0.05),
            xytext=(12, -16),
            textcoords="offset points",
            fontsize=8.5,
            color="#111111",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.9, "edgecolor": "#9ca3af"},
            arrowprops={"arrowstyle": "-", "color": "#6b7280", "lw": 0.9},
        )

    handles, labels = ax.get_legend_handles_labels()
    if right_ax is not None:
        right_handles, right_labels = right_ax.get_legend_handles_labels()
        handles.extend(right_handles)
        labels.extend(right_labels)
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=False):
        if label not in unique:
            unique[label] = handle
    ax.legend(unique.values(), unique.keys(), loc="upper left", frameon=True, ncols=2)
    return fig, (ax, phase_ax)


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


def plot_peak_demand_summary(
    distance_table: pd.DataFrame,
    extrema_table: pd.DataFrame,
    *,
    peak_windows: pd.DataFrame | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Template 3: peak-demand bars plus actionable top-window context."""
    _require_columns(distance_table, ["window_label", "best_distance_yd"], name="distance_table")
    _require_columns(extrema_table, ["metric", "value", "ts_utc"], name="extrema_table")
    if peak_windows is not None and not peak_windows.empty:
        _require_columns(
            peak_windows,
            [
                "window_rank",
                "window_start_utc",
                "window_end_utc",
                "distance_yd",
                "hsr_event_count",
                "accel_event_count",
                "decel_event_count",
            ],
            name="peak_windows",
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
    bar = sns.barplot(
        data=distance_table,
        x="window_label",
        y="best_distance_yd",
        color="#2A9D8F",
        ax=ax_left,
    )
    for label, patch in zip(distance_table["window_label"].astype(str), bar.patches, strict=False):
        if label == "1m":
            patch.set_edgecolor("#1B5E57")
            patch.set_linewidth(2.2)
        value = patch.get_height()
        ax_left.annotate(
            f"{value:.0f}",
            (patch.get_x() + patch.get_width() / 2.0, value),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 4),
            textcoords="offset points",
        )
    ax_left.set_title("Best Rolling Distance by Window", fontsize=12, weight="bold")
    ax_left.set_xlabel("Window")
    ax_left.set_ylabel("Distance (yards)")
    ax_left.set_ylim(0.0, 460.0)
    ax_left.set_yticks(np.arange(0.0, 461.0, 100.0))

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
        for row in peak_windows.sort_values("window_rank").head(3).itertuples(index=False):
            start = pd.Timestamp(row.window_start_utc).strftime("%H:%M")
            end = pd.Timestamp(row.window_end_utc).strftime("%H:%M")
            lines.append(
                f"W{int(row.window_rank)} {start}-{end} | {float(row.distance_yd):.0f} yd | "
                f"HSR{int(row.hsr_event_count)} | A/D {int(row.accel_event_count)}/{int(row.decel_event_count)}"
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

    fig.suptitle("Peak Demand Summary", fontsize=14, weight="bold")
    return fig, (ax_left, ax_right)


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


def _require_columns(df: pd.DataFrame, columns: Sequence[str], *, name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
