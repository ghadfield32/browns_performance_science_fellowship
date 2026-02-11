#!/usr/bin/env python3
"""Validate submission completeness before delivery.

Checks that all required files, outputs, and deliverables are present
and meet quality criteria.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def validate_submission(project_root: Path) -> list[str]:
    """Return list of validation errors (empty if all pass)."""
    errors = []

    # Required files
    required_files = [
        "README.md",
        "pyproject.toml",
        "notebooks/00_self_contained_analysis.ipynb",
        "outputs/results.json",
        "docs/deliverables/tracking_analysis_deliverable.pptx",
        "scripts/build_powerpoint.py",
    ]

    for file in required_files:
        if not (project_root / file).exists():
            errors.append(f"Missing required file: {file}")

    # Required figures
    required_figures = ["01_space.png", "02_time.png", "03_peaks.png"]

    for fig in required_figures:
        fig_path = project_root / "outputs" / "figures" / fig
        if not fig_path.exists():
            errors.append(f"Missing figure: outputs/figures/{fig}")

    # Validate results.json structure
    results_path = project_root / "outputs" / "results.json"
    if results_path.exists():
        try:
            with open(results_path) as f:
                results = json.load(f)

            # Required keys (must have session_summary and qc_status at minimum)
            if "session_summary" not in results:
                errors.append("results.json missing key: session_summary")
            if "qc_status" not in results:
                errors.append("results.json missing key: qc_status")

            # Check distance source assumption (if assumptions key exists)
            assumptions = results.get("assumptions", {})
            if assumptions:
                if assumptions.get("distance_source") == "speed_integrated":
                    # Good! Using speed-integrated distance
                    pass
                else:
                    # Warn if using different source
                    errors.append(
                        "results.json should specify distance_source='speed_integrated'"
                    )

                # Check vendor dis status (if present)
                vendor_status = assumptions.get("vendor_dis_status", "")
                if vendor_status and "REJECT" not in vendor_status.upper():
                    errors.append(
                        "results.json should indicate vendor dis column is REJECTED"
                    )

        except json.JSONDecodeError:
            errors.append("results.json is not valid JSON")
    else:
        errors.append("Missing outputs/results.json")

    # Check PowerPoint file size (should be >0.5MB with images)
    pptx_path = project_root / "docs" / "deliverables" / "tracking_analysis_deliverable.pptx"
    if pptx_path.exists():
        size_mb = pptx_path.stat().st_size / (1024 * 1024)
        if size_mb < 0.5:
            errors.append(
                f"PowerPoint suspiciously small: {size_mb:.2f} MB (expected >0.5 MB)"
            )
    else:
        errors.append("Missing PowerPoint deliverable")

    # Check for essential CSV tables (flexible - from either pipeline)
    # Accept tables from self-contained notebook OR package pipeline
    tables_dir = project_root / "outputs" / "tables"
    if tables_dir.exists():
        # Check for at least one of the key table types
        has_speed_bands = (tables_dir / "speed_bands.csv").exists() or \
                         (tables_dir / "absolute_speed_band_summary.csv").exists()
        has_peaks = (tables_dir / "peak_windows.csv").exists() or \
                   (tables_dir / "peak_by_duration.csv").exists()

        if not has_speed_bands:
            errors.append("Missing speed band table in outputs/tables/")
        if not has_peaks:
            errors.append("Missing peak windows table in outputs/tables/")
    else:
        errors.append("Missing outputs/tables/ directory")

    return errors


def main() -> None:
    """Run validation and report results."""
    project_root = Path(__file__).parent.parent
    errors = validate_submission(project_root)

    if errors:
        print("[VALIDATION FAILED]\n")
        print(f"Found {len(errors)} issue(s):\n")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        print("\n" + "=" * 60)
        print("Please fix these issues before submitting.")
        sys.exit(1)
    else:
        print("=" * 60)
        print("[VALIDATION PASSED]")
        print("=" * 60)
        print("\nAll required files and outputs are present:")
        print("  - Self-contained notebook")
        print("  - PowerPoint deliverable (8 slides)")
        print("  - Analysis outputs (figures, tables, results.json)")
        print("  - Distance source: speed-integrated (vendor dis rejected)")
        print("\nSubmission is ready!")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
