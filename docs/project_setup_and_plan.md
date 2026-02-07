# Project Setup and Execution Plan

## Source brief requirements (from provided PowerPoint)

- Analyze one anonymized player single-session 10 Hz tracking data.
- Deliver reproducible code and either slides and/or notebook.
- Report:
  - Distance in yards
  - Speed in mph
  - Acceleration/deceleration in m/s^2
- Required analysis themes:
  - Session-level workload metrics
  - Peak demands and most demanding scenarios
  - Distinct periods/drills/events if possible
  - 2-3 visuals readable in under 30 seconds

## Data contract observed in `data/tracking_data.csv`

- 65,410 data rows (65,411 including header)
- Columns:
  - `ts,x,y,z,dir,dis,s,a,sa,latLoad,dirLoad`
- Raw units are aligned with brief:
  - Position/distance: yards
  - Speed: yards/second
  - Acceleration: yards/second^2

## Locked conversion constants

- `speed_mph = s_yd_per_s * 2.0454545454545454`
- `accel_ms2 = accel_yd_per_s2 * 0.9144`

## Tuned performance-model thresholds (current default preset)

- Speed bands (mph): `Walk 0-3`, `Cruise 3-9`, `Run 9-13`, `HSR 13-16`, `Sprint >=16`
- HSR threshold: `13.0 mph`
- Accel/decel event thresholds: `>= +3.0 m/s^2`, `<= -3.0 m/s^2`
- Rest segmentation threshold: `<1.2 mph` for at least `25 s`

## Recommended implementation order

1. Data QA and audit
   - Parse timestamp as UTC and validate 10 Hz cadence.
   - Count sampling gaps and flag outlier jumps.
2. Core metrics
   - Total distance (primary from speed integration, cross-check from XY step distance).
   - Mean/peak speed, mean/peak accel, peak decel.
   - Distance by speed bands (absolute and relative-to-max).
3. Peak demands
   - Rolling 30 s / 1 min / 3 min / 5 min distance.
   - Rolling high-speed distance and accel/decel event counts.
4. Session segmentation
   - Hard gaps + rest/work thresholding + intensity regime shifts.
   - Block-level summary table (Block A/B/C naming).
5. Communication outputs
   - Movement map (X-Y)
   - Intensity timeline with annotated peak windows
   - Peak demand summary visual/table

## Environment commands

```bash
uv venv
uv sync --group dev
uv run browns-tracking-summary
uv run python scripts/build_notebook.py
uv run python scripts/build_presentation_notebook.py
uv run python scripts/render_presentation_assets.py
```
