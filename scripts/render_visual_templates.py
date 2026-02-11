"""Run one-pass analysis and export canonical visual/tables/results contract artifacts."""

from __future__ import annotations

import argparse

from browns_tracking.config import resolve_data_file, resolve_output_dir
from browns_tracking.pipeline import load_tracking_data
from browns_tracking.presets import preferred_performance_model
from browns_tracking.results_contract import (
    run_session_analysis,
    write_results_contract,
    write_results_tables,
)
from browns_tracking.visuals import (
    close_figures,
    plot_intensity_timeline,
    plot_movement_map,
    plot_peak_demand_summary,
    plot_session_structure_map,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export 3 coach-facing visuals, canonical tables, and outputs/results.json "
            "from one validated analysis pass."
        )
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to tracking_data.csv (default: auto-resolve from project config).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for figure and table exports (default: auto-resolve from project config).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_data_file(args.input)
    output_dir = resolve_output_dir(args.output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_tracking_data(input_path)
    model = preferred_performance_model()
    results = run_session_analysis(df, model=model)

    table_paths = write_results_tables(results, output_dir)

    fig1, _ = plot_movement_map(
        results.coach_df,
        segment_col="coach_phase_label",
        highlight_top_n=3,
        highlight_windows=results.peak_windows,
    )
    fig2, _ = plot_intensity_timeline(
        results.coach_df,
        top_windows=results.peak_windows,
        hsr_threshold_mph=model.peak_demand_config.hsr_threshold_mph,
        phase_col="coach_phase_label",
        show_phase_strip=True,
        show_hsr_ticks=True,
        show_cumulative_distance=False,
    )
    fig3, _ = plot_peak_demand_summary(
        results.distance_table,
        results.extrema_table,
        peak_windows=results.peak_windows,
    )
    fig4, _ = plot_session_structure_map(results.structure_map_summary)

    canonical_figures = {
        "space": fig_dir / "01_space.png",
        "time": fig_dir / "02_time.png",
        "peaks": fig_dir / "03_peaks.png",
    }
    supplemental_figures = {
        "structure_map": fig_dir / "04_structure.png",
    }
    legacy_figures = {
        "movement_map_legacy": fig_dir / "movement_map.png",
        "intensity_timeline_legacy": fig_dir / "intensity_timeline.png",
        "peak_demand_summary_legacy": fig_dir / "peak_demand_summary.png",
    }

    save_figure(fig1, canonical_figures["space"])
    save_figure(fig2, canonical_figures["time"])
    save_figure(fig3, canonical_figures["peaks"])
    save_figure(fig4, supplemental_figures["structure_map"])
    save_figure(fig1, legacy_figures["movement_map_legacy"])
    save_figure(fig2, legacy_figures["intensity_timeline_legacy"])
    save_figure(fig3, legacy_figures["peak_demand_summary_legacy"])
    close_figures([fig1, fig2, fig3, fig4])

    figure_paths = {**canonical_figures, **supplemental_figures, **legacy_figures}
    contract_path = write_results_contract(
        results,
        input_path=input_path,
        output_dir=output_dir,
        table_paths=table_paths,
        figure_paths=figure_paths,
    )

    print("Export complete")
    print(f"Session rows: {results.session_summary['rows']}")
    print(f"Input data: {input_path}")
    print(f"Results contract: {contract_path}")
    print(f"Canonical figures: {canonical_figures['space']}, {canonical_figures['time']}, {canonical_figures['peaks']}")
    print(f"Supplemental figure: {supplemental_figures['structure_map']}")
    print(f"Canonical tables: {output_dir / 'phase_table.csv'}, {output_dir / 'peak_windows.csv'}")


if __name__ == "__main__":
    main()
