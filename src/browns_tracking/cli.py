"""CLI entrypoints for the tracking project."""

from __future__ import annotations

import argparse
import json

from .config import resolve_data_file
from .pipeline import load_tracking_data, summarize_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a 10 Hz tracking CSV.")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to tracking_data.csv (default: auto-resolve from project config).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    input_path = resolve_data_file(args.input)
    df = load_tracking_data(input_path)
    summary = summarize_session(df)
    summary["input_path"] = str(input_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
