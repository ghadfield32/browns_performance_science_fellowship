"""Visualization templates for coach/staff communication."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
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
) -> tuple[plt.Figure, plt.Axes]:
    """Template 1: movement map with top-phase highlights and direct annotations."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)

    hb = ax.hexbin(
        df[x_col],
        df[y_col],
        gridsize=70,
        bins="log",
        mincnt=1,
        cmap="Greys",
        alpha=0.35,
    )
    cbar = fig.colorbar(hb, ax=ax, pad=0.01)
    cbar.set_label("Log point density")

    ax.plot(
        df[x_col],
        df[y_col],
        linewidth=1.0,
        alpha=0.5,
        color="#4c566a",
        zorder=2,
    )

    if segment_col in df.columns and highlight_top_n > 0:
        if "step_distance_yd_from_speed" in df.columns:
            ranked = (
                df.groupby(segment_col, dropna=False)["step_distance_yd_from_speed"]
                .sum()
                .sort_values(ascending=False)
            )
        else:
            ranked = df.groupby(segment_col, dropna=False).size().sort_values(ascending=False)

        highlight_labels = [label for label in ranked.index[:highlight_top_n].tolist() if pd.notna(label)]
        palette = sns.color_palette("Set2", n_colors=max(1, len(highlight_labels)))
        for color, label in zip(palette, highlight_labels, strict=False):
            phase = df[df[segment_col] == label]
            ax.plot(
                phase[x_col],
                phase[y_col],
                linewidth=2.2,
                alpha=0.95,
                color=color,
                zorder=3,
            )
            if annotate_highlights and not phase.empty:
                mid = phase.iloc[len(phase) // 2]
                ax.annotate(
                    _compact_phase_label(str(label)),
                    xy=(float(mid[x_col]), float(mid[y_col])),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=8,
                    color="#111111",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "white",
                        "alpha": 0.85,
                        "edgecolor": color,
                    },
                    zorder=5,
                )

    start_x = float(df[x_col].iloc[0])
    start_y = float(df[y_col].iloc[0])
    end_x = float(df[x_col].iloc[-1])
    end_y = float(df[y_col].iloc[-1])
    ax.scatter(start_x, start_y, color="#2ca02c", s=55, zorder=4)
    ax.scatter(end_x, end_y, color="#d62728", s=55, zorder=4)
    ax.annotate("Start", xy=(start_x, start_y), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.annotate("End", xy=(end_x, end_y), xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_title("Player Movement Map (Density + Key Phases)", fontsize=13, weight="bold")
    ax.set_xlabel("X position (yards)")
    ax.set_ylabel("Y position (yards)")
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
) -> tuple[plt.Figure, plt.Axes]:
    """Template 2: speed timeline with optional peak-window overlays."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)

    ax.plot(df["ts"], df["speed_mph"], color="#1f77b4", alpha=0.55, linewidth=0.8, label="Speed (mph)")
    smooth = df.set_index("ts")["speed_mph"].rolling("60s", min_periods=1).mean()
    ax.plot(smooth.index, smooth.values, color="#ff7f0e", linewidth=2.2, label="60s rolling mean")

    if hsr_threshold_mph is not None:
        ax.axhline(
            hsr_threshold_mph,
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            label=f"HSR threshold ({hsr_threshold_mph:.1f} mph)",
        )

    if top_windows is not None and not top_windows.empty:
        palette = sns.color_palette("Set2", n_colors=min(4, len(top_windows)))
        for i, row in top_windows.reset_index(drop=True).iterrows():
            color = palette[i % len(palette)]
            ax.axvspan(
                row["window_start_utc"],
                row["window_end_utc"],
                color=color,
                alpha=0.22,
                label=f"Top window {i + 1}",
            )

    ax.set_title("Session Intensity Timeline", fontsize=13, weight="bold")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Speed (mph)")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper right", frameon=True)
    return fig, ax


def plot_peak_demand_summary(
    distance_table: pd.DataFrame,
    extrema_table: pd.DataFrame,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Template 3: best rolling distances plus session extrema text panel."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.5), constrained_layout=True, gridspec_kw={"width_ratios": [1.8, 1]}
    )

    ax_left, ax_right = axes
    bar = sns.barplot(
        data=distance_table,
        x="window_label",
        y="best_distance_yd",
        color="#2a9d8f",
        ax=ax_left,
    )
    for patch in bar.patches:
        value = patch.get_height()
        ax_left.annotate(
            f"{value:.1f}",
            (patch.get_x() + patch.get_width() / 2.0, value),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 4),
            textcoords="offset points",
        )
    ax_left.set_title("Best Rolling Distance", fontsize=12, weight="bold")
    ax_left.set_xlabel("Window")
    ax_left.set_ylabel("Distance (yards)")

    ax_right.axis("off")
    ax_right.set_title("Session Peak Metrics", fontsize=12, weight="bold", pad=12)
    lines = []
    for row in extrema_table.itertuples(index=False):
        ts_text = pd.Timestamp(row.ts_utc).strftime("%H:%M:%S")
        lines.append(f"{row.metric}: {row.value:.2f}\n@ {ts_text} UTC")
    text = "\n\n".join(lines)
    ax_right.text(
        0.03,
        0.95,
        text,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    fig.suptitle("Peak Demands Summary", fontsize=14, weight="bold")
    return fig, (ax_left, ax_right)


def save_figure(fig: plt.Figure, output_path: str | Path, dpi: int = 300) -> None:
    """Save a figure with a transparent-friendly white background."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")


def close_figures(figures: Sequence[plt.Figure]) -> None:
    """Close a batch of figures to keep notebook/script memory stable."""
    for fig in figures:
        plt.close(fig)
