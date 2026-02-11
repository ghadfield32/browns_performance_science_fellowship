"""
Browns Performance Science Fellow — Tracking Analysis Pipeline
==============================================================
Single-player anonymized practice session (10 Hz tracking data).

Units (hard requirement):
  - Distance: yards
  - Speed: miles per hour (mph)
  - Acceleration/Deceleration: m/s²

Conversion factors:
  - yd/s → mph:  multiply by 3600/1760 ≈ 2.045454545
  - yd/s² → m/s²: multiply by 0.9144

Author: Geoff
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────────────────
DATA_PATH = Path("/mnt/user-data/uploads/tracking_data.csv")
OUT_DIR = Path("/home/claude/outputs")
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"
for d in [FIG_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ────────────────────────────────────────────────────────────
YDS_TO_MPH = 3600.0 / 1760.0        # 2.045454545...
YDS2_TO_MS2 = 0.9144
EXPECTED_CADENCE_S = 0.1
GAP_THRESHOLD_S = 1.0                # flag gaps > 1 second
TELEPORT_SPEED_YDS = 25.0            # yd/s (~51 mph) – impossible for humans

# Speed band definitions (mph) — justified by NFL tracking conventions
SPEED_BANDS = [
    {"name": "Standing",    "lower": 0.0,  "upper": 1.0},
    {"name": "Walking",     "lower": 1.0,  "upper": 4.0},
    {"name": "Jogging",     "lower": 4.0,  "upper": 8.0},
    {"name": "Running",     "lower": 8.0,  "upper": 13.0},
    {"name": "High-Speed",  "lower": 13.0, "upper": 16.0},
    {"name": "Sprint",      "lower": 16.0, "upper": None},
]

HSR_THRESHOLD_MPH = 13.0
SPRINT_THRESHOLD_MPH = 16.0
ACCEL_THRESHOLD_MS2 = 2.0     # ≥2.0 m/s² = high accel event
DECEL_THRESHOLD_MS2 = -2.0   # ≤-2.0 m/s² = high decel event

# Rolling window durations for peak demand analysis
PEAK_WINDOWS_S = [15, 30, 60, 120, 300]

# Phase detection parameters
PHASE_BIN_S = 120            # 2-minute bins for phase detection
PHASE_MERGE_GAP_MIN = 2.0   # merge rest blocks shorter than 2 min into adjacent active
MIN_ACTIVE_SPEED_MPH = 1.5  # speed threshold for "active" classification

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD AND QC
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Load raw data and quality control")
print("=" * 70)

raw = pd.read_csv(DATA_PATH)
raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
raw = raw.sort_values("ts").reset_index(drop=True)

# Compute dt (time delta between consecutive samples)
raw["dt"] = raw["ts"].diff().dt.total_seconds()
raw.loc[0, "dt"] = EXPECTED_CADENCE_S  # first row has no predecessor

# ── QC: Timestamp gaps ───────────────────────────────────────────────────
gap_mask = raw["dt"] > GAP_THRESHOLD_S
gap_count = gap_mask.sum()
max_gap_s = raw["dt"].max()
pct_on_cadence = ((raw["dt"] - EXPECTED_CADENCE_S).abs() < 0.02).mean() * 100

print(f"  Rows: {len(raw):,}")
print(f"  Time span: {raw['ts'].iloc[0]} → {raw['ts'].iloc[-1]}")
print(f"  Duration: {(raw['ts'].iloc[-1] - raw['ts'].iloc[0]).total_seconds():.1f}s "
      f"({(raw['ts'].iloc[-1] - raw['ts'].iloc[0]).total_seconds()/60:.1f} min)")
print(f"  Cadence: {pct_on_cadence:.1f}% of samples at expected 0.1s")
print(f"  Gaps > {GAP_THRESHOLD_S}s: {gap_count} (max: {max_gap_s:.1f}s)")

# ── QC: Flag gaps and teleports ──────────────────────────────────────────
raw["is_gap"] = gap_mask

# XY displacement check
raw["xy_step"] = np.sqrt(raw["x"].diff()**2 + raw["y"].diff()**2)
raw["xy_speed_yds"] = raw["xy_step"] / raw["dt"].replace(0, np.nan)
teleport_mask = raw["xy_speed_yds"] > TELEPORT_SPEED_YDS
raw["is_teleport"] = teleport_mask
print(f"  Teleport samples (>{TELEPORT_SPEED_YDS} yd/s): {teleport_mask.sum()}")

# ── QC: Speed-vs-XY consistency ──────────────────────────────────────────
normal = ~raw["is_gap"] & ~raw["is_teleport"] & (raw["dt"] > 0)
if normal.sum() > 100:
    speed_xy_corr = np.corrcoef(
        raw.loc[normal, "s"],
        raw.loc[normal, "xy_speed_yds"].fillna(0)
    )[0, 1]
    print(f"  Speed vs XY-derived speed correlation: {speed_xy_corr:.3f}")

# ── QC: dis column validation ────────────────────────────────────────────
dist_speed = (raw["s"] * raw["dt"]).sum()
dist_xy = raw["xy_step"].sum()
dist_dis = raw["dis"].sum()
print(f"  Distance from speed*dt: {dist_speed:.1f} yd")
print(f"  Distance from XY displacement: {dist_xy:.1f} yd")
print(f"  Distance from 'dis' column: {dist_dis:.1f} yd (vendor — unreliable, not used)")

# Assign continuity block IDs (reset at gaps)
raw["block_id"] = raw["is_gap"].cumsum()

# ── QC Summary table ─────────────────────────────────────────────────────
qc_summary = {
    "total_rows": len(raw),
    "session_start_utc": str(raw["ts"].iloc[0]),
    "session_end_utc": str(raw["ts"].iloc[-1]),
    "duration_s": round((raw["ts"].iloc[-1] - raw["ts"].iloc[0]).total_seconds(), 1),
    "pct_at_expected_cadence": round(pct_on_cadence, 2),
    "gap_count": int(gap_count),
    "max_gap_s": round(max_gap_s, 2),
    "teleport_count": int(teleport_mask.sum()),
    "continuity_blocks": int(raw["block_id"].nunique()),
    "distance_speed_yd": round(dist_speed, 1),
    "distance_xy_yd": round(dist_xy, 1),
    "distance_dis_yd": round(dist_dis, 1),
    "speed_xy_correlation": round(speed_xy_corr, 3) if normal.sum() > 100 else None,
    "qc_status": "PASS" if gap_count <= 10 and pct_on_cadence > 95 else "WARN",
}
print(f"\n  QC Status: {qc_summary['qc_status']}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: UNIT CONVERSIONS AND DERIVED COLUMNS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Unit conversions and derived columns")
print("=" * 70)

df = raw.copy()

# Speed: yd/s → mph
df["speed_mph"] = df["s"] * YDS_TO_MPH

# Acceleration: yd/s² → m/s² (using signed sa for accel/decel direction)
df["accel_ms2"] = df["sa"] * YDS2_TO_MS2
df["accel_mag_ms2"] = df["a"] * YDS2_TO_MS2

# Step distance (speed-derived, yards)
df["step_dist_yd"] = df["s"] * df["dt"]
# Zero out step distance at gaps (don't accumulate distance across discontinuities)
df.loc[df["is_gap"], "step_dist_yd"] = 0.0

# Elapsed time (seconds and minutes from session start)
df["elapsed_s"] = (df["ts"] - df["ts"].iloc[0]).dt.total_seconds()
df["elapsed_min"] = df["elapsed_s"] / 60.0

# Speed band classification
def classify_speed_band(speed_mph):
    for band in SPEED_BANDS:
        upper = band["upper"] if band["upper"] is not None else np.inf
        if band["lower"] <= speed_mph < upper:
            return band["name"]
    return SPEED_BANDS[-1]["name"]

df["speed_band"] = df["speed_mph"].apply(classify_speed_band)

# HSR / Sprint flags
df["is_hsr"] = df["speed_mph"] >= HSR_THRESHOLD_MPH
df["is_sprint"] = df["speed_mph"] >= SPRINT_THRESHOLD_MPH
df["is_accel"] = df["accel_ms2"] >= ACCEL_THRESHOLD_MS2
df["is_decel"] = df["accel_ms2"] <= DECEL_THRESHOLD_MS2

print(f"  Conversion: yd/s → mph (factor: {YDS_TO_MPH:.6f})")
print(f"  Conversion: yd/s² → m/s² (factor: {YDS2_TO_MS2})")
print(f"  Speed bands: {', '.join(b['name'] for b in SPEED_BANDS)}")
print(f"  HSR threshold: {HSR_THRESHOLD_MPH} mph | Sprint: {SPRINT_THRESHOLD_MPH} mph")
print(f"  Accel threshold: {ACCEL_THRESHOLD_MS2} m/s² | Decel: {DECEL_THRESHOLD_MS2} m/s²")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: SESSION-LEVEL METRICS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Session-level workload metrics")
print("=" * 70)

# ── 3a: Total distance ───────────────────────────────────────────────────
total_distance_yd = df["step_dist_yd"].sum()
print(f"  Total distance: {total_distance_yd:.1f} yd")

# ── 3b: Distance by speed band ───────────────────────────────────────────
speed_band_dist = df.groupby("speed_band").agg(
    distance_yd=("step_dist_yd", "sum"),
    sample_count=("step_dist_yd", "count"),
    time_s=("dt", "sum"),
).reindex([b["name"] for b in SPEED_BANDS]).fillna(0)

speed_band_dist["distance_pct"] = (speed_band_dist["distance_yd"] / total_distance_yd * 100).round(1)
speed_band_dist["time_min"] = (speed_band_dist["time_s"] / 60).round(1)
speed_band_dist["time_pct"] = (speed_band_dist["time_s"] / df["dt"].sum() * 100).round(1)

print("\n  Speed Band Distance Breakdown:")
for band_name, row in speed_band_dist.iterrows():
    print(f"    {band_name:12s}: {row['distance_yd']:7.1f} yd ({row['distance_pct']:5.1f}%)  "
          f"| {row['time_min']:5.1f} min ({row['time_pct']:5.1f}%)")

# ── 3c: Speed and acceleration statistics ────────────────────────────────
metrics = {
    "total_distance_yd": round(total_distance_yd, 1),
    "mean_speed_mph": round(df["speed_mph"].mean(), 2),
    "median_speed_mph": round(df["speed_mph"].median(), 2),
    "p95_speed_mph": round(df["speed_mph"].quantile(0.95), 2),
    "max_speed_mph": round(df["speed_mph"].max(), 2),
    "mean_accel_ms2": round(df["accel_ms2"].mean(), 3),
    "peak_accel_ms2": round(df["accel_ms2"].max(), 2),
    "peak_decel_ms2": round(df["accel_ms2"].min(), 2),
    "p95_accel_ms2": round(df["accel_ms2"].quantile(0.99), 2),
    "p05_decel_ms2": round(df["accel_ms2"].quantile(0.01), 2),
    "hsr_distance_yd": round(df.loc[df["is_hsr"], "step_dist_yd"].sum(), 1),
    "sprint_distance_yd": round(df.loc[df["is_sprint"], "step_dist_yd"].sum(), 1),
    "hsr_time_s": round(df.loc[df["is_hsr"], "dt"].sum(), 1),
    "sprint_time_s": round(df.loc[df["is_sprint"], "dt"].sum(), 1),
}

print(f"\n  Mean speed: {metrics['mean_speed_mph']} mph")
print(f"  Median speed: {metrics['median_speed_mph']} mph")
print(f"  P95 speed: {metrics['p95_speed_mph']} mph")
print(f"  Max speed: {metrics['max_speed_mph']} mph")
print(f"  Peak accel: {metrics['peak_accel_ms2']} m/s²")
print(f"  Peak decel: {metrics['peak_decel_ms2']} m/s²")
print(f"  HSR distance: {metrics['hsr_distance_yd']} yd ({metrics['hsr_time_s']}s)")
print(f"  Sprint distance: {metrics['sprint_distance_yd']} yd ({metrics['sprint_time_s']}s)")

# ── 3d: Event detection (contiguous threshold exposure >= 1s) ─────────────
def count_events(mask, dt_series, min_duration_s=1.0):
    """Count contiguous True runs lasting >= min_duration_s."""
    events = []
    in_event = False
    event_start = 0
    event_duration = 0.0
    event_distance = 0.0
    for i in range(len(mask)):
        if mask.iloc[i]:
            if not in_event:
                in_event = True
                event_start = i
                event_duration = 0.0
                event_distance = 0.0
            event_duration += dt_series.iloc[i]
            event_distance += df["step_dist_yd"].iloc[i]
        else:
            if in_event:
                if event_duration >= min_duration_s:
                    events.append({
                        "start_idx": event_start,
                        "end_idx": i - 1,
                        "duration_s": round(event_duration, 2),
                        "distance_yd": round(event_distance, 1),
                    })
                in_event = False
    # Handle event running to end
    if in_event and event_duration >= min_duration_s:
        events.append({
            "start_idx": event_start,
            "end_idx": len(mask) - 1,
            "duration_s": round(event_duration, 2),
            "distance_yd": round(event_distance, 1),
        })
    return events

hsr_events = count_events(df["is_hsr"], df["dt"])
sprint_events = count_events(df["is_sprint"], df["dt"])
accel_events = count_events(df["is_accel"], df["dt"])
decel_events = count_events(df["is_decel"], df["dt"])

event_counts = {
    "hsr_event_count": len(hsr_events),
    "sprint_event_count": len(sprint_events),
    "accel_event_count": len(accel_events),
    "decel_event_count": len(decel_events),
    "hsr_total_distance_yd": round(sum(e["distance_yd"] for e in hsr_events), 1),
    "sprint_total_distance_yd": round(sum(e["distance_yd"] for e in sprint_events), 1),
}
print(f"\n  HSR events (≥1s): {event_counts['hsr_event_count']}")
print(f"  Sprint events (≥1s): {event_counts['sprint_event_count']}")
print(f"  Accel events (≥{ACCEL_THRESHOLD_MS2} m/s², ≥1s): {event_counts['accel_event_count']}")
print(f"  Decel events (≤{DECEL_THRESHOLD_MS2} m/s², ≥1s): {event_counts['decel_event_count']}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: PEAK DEMAND — ROLLING WINDOWS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: Peak demand rolling windows")
print("=" * 70)

# For each window duration, find the top-1 window by distance (intensity = yd/min)
# Windows must not span a gap boundary.
peak_windows = []

for window_s in PEAK_WINDOWS_S:
    window_samples = int(window_s / EXPECTED_CADENCE_S)
    best_dist = 0
    best_start = 0

    # Rolling sum of step distance
    roll_dist = df["step_dist_yd"].rolling(window_samples, min_periods=window_samples).sum()
    # Rolling count of gaps within window (invalidate windows that span gaps)
    roll_gaps = df["is_gap"].astype(int).rolling(window_samples, min_periods=window_samples).sum()

    valid = (roll_gaps == 0) & roll_dist.notna()
    if valid.any():
        best_idx = roll_dist[valid].idxmax()
        best_dist = roll_dist[best_idx]
        window_start_idx = best_idx - window_samples + 1

        peak_windows.append({
            "window_s": window_s,
            "window_label": f"{window_s // 60}min" if window_s >= 60 else f"{window_s}s",
            "distance_yd": round(best_dist, 1),
            "intensity_yd_min": round(best_dist / (window_s / 60), 1),
            "start_idx": int(window_start_idx),
            "end_idx": int(best_idx),
            "start_utc": str(df.loc[window_start_idx, "ts"]),
            "end_utc": str(df.loc[best_idx, "ts"]),
            "start_elapsed_min": round(df.loc[window_start_idx, "elapsed_min"], 1),
            "end_elapsed_min": round(df.loc[best_idx, "elapsed_min"], 1),
            "mean_speed_mph": round(df.loc[window_start_idx:best_idx, "speed_mph"].mean(), 2),
            "max_speed_mph": round(df.loc[window_start_idx:best_idx, "speed_mph"].max(), 2),
        })

        label = f"{window_s // 60}min" if window_s >= 60 else f"{window_s}s"
        print(f"  Best {label:4s}: {best_dist:7.1f} yd ({best_dist / (window_s / 60):6.1f} yd/min) "
              f"@ min {df.loc[window_start_idx, 'elapsed_min']:.0f}–{df.loc[best_idx, 'elapsed_min']:.0f}")

peak_windows_df = pd.DataFrame(peak_windows)

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: SESSION PHASE SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: Session phase segmentation")
print("=" * 70)

# Resample into 2-minute bins for phase detection (reduces noise)
bin_s = PHASE_BIN_S
df["bin_phase"] = (df["elapsed_s"] // bin_s).astype(int)
bin_stats = df.groupby("bin_phase").agg(
    mean_speed_mph=("speed_mph", "mean"),
    max_speed_mph=("speed_mph", "max"),
    distance_yd=("step_dist_yd", "sum"),
    elapsed_min=("elapsed_min", "first"),
    sample_count=("speed_mph", "count"),
).reset_index()

# Classify each bin as active or rest
bin_stats["is_active"] = bin_stats["mean_speed_mph"] > MIN_ACTIVE_SPEED_MPH

# Merge adjacent bins of same type into phases
phases = []
current_type = None
current_start_min = 0
current_bins = []

for _, brow in bin_stats.iterrows():
    btype = "Active" if brow["is_active"] else "Rest"
    if btype != current_type:
        if current_type is not None and current_bins:
            phases.append({
                "type": current_type,
                "start_min": current_start_min,
                "end_min": current_bins[-1]["elapsed_min"] + bin_s / 60.0,
                "bins": current_bins,
            })
        current_type = btype
        current_start_min = brow["elapsed_min"]
        current_bins = [brow.to_dict()]
    else:
        current_bins.append(brow.to_dict())

if current_bins:
    phases.append({
        "type": current_type,
        "start_min": current_start_min,
        "end_min": current_bins[-1]["elapsed_min"] + bin_s / 60.0,
        "bins": current_bins,
    })

# Merge short rest phases (< PHASE_MERGE_GAP_MIN) into adjacent active
merged_phases = []
for p in phases:
    duration_min = p["end_min"] - p["start_min"]
    if (p["type"] == "Rest" and duration_min < PHASE_MERGE_GAP_MIN
            and merged_phases and merged_phases[-1]["type"] == "Active"):
        # Absorb into previous active phase
        merged_phases[-1]["end_min"] = p["end_min"]
        merged_phases[-1]["bins"].extend(p["bins"])
    else:
        merged_phases.append(p)

# Second pass: merge adjacent Active phases that are now next to each other
final_phases = []
for p in merged_phases:
    if final_phases and final_phases[-1]["type"] == p["type"]:
        final_phases[-1]["end_min"] = p["end_min"]
        final_phases[-1]["bins"].extend(p["bins"])
    else:
        final_phases.append(p)
merged_phases = final_phases

# Build phase summary table
phase_summary = []
for i, p in enumerate(merged_phases):
    phase_df = df[(df["elapsed_min"] >= p["start_min"]) & (df["elapsed_min"] < p["end_min"])]
    if len(phase_df) == 0:
        continue

    dist = phase_df["step_dist_yd"].sum()
    dur_s = phase_df["dt"].sum()
    max_spd = phase_df["speed_mph"].max()

    # Intensity label
    if p["type"] == "Rest":
        intensity = "Rest"
    elif max_spd >= SPRINT_THRESHOLD_MPH:
        intensity = "High"
    elif max_spd >= HSR_THRESHOLD_MPH:
        intensity = "Moderate-High"
    elif max_spd >= 8.0:
        intensity = "Moderate"
    else:
        intensity = "Low"

    phase_summary.append({
        "phase": i + 1,
        "type": p["type"],
        "intensity": intensity,
        "start_min": round(p["start_min"], 1),
        "end_min": round(p["end_min"], 1),
        "duration_min": round(dur_s / 60, 1),
        "distance_yd": round(dist, 1),
        "distance_rate_yd_min": round(dist / max(dur_s / 60, 0.01), 1),
        "max_speed_mph": round(max_spd, 1),
        "hsr_distance_yd": round(phase_df.loc[phase_df["is_hsr"], "step_dist_yd"].sum(), 1),
        "sprint_distance_yd": round(phase_df.loc[phase_df["is_sprint"], "step_dist_yd"].sum(), 1),
    })

phase_df_summary = pd.DataFrame(phase_summary)

print(f"  Detected {len(phase_summary)} session phases:")
for _, p in phase_df_summary.iterrows():
    print(f"    Phase {int(p['phase']):2d} [{p['intensity']:13s}]: "
          f"min {p['start_min']:5.1f}–{p['end_min']:5.1f} "
          f"({p['duration_min']:4.1f} min, {p['distance_yd']:6.1f} yd, "
          f"max {p['max_speed_mph']:.1f} mph)")

# Assign phase labels back to main df
df["phase_id"] = 0
df["phase_intensity"] = "Unassigned"
for _, p in phase_df_summary.iterrows():
    mask = (df["elapsed_min"] >= p["start_min"]) & (df["elapsed_min"] < p["end_min"])
    df.loc[mask, "phase_id"] = int(p["phase"])
    df.loc[mask, "phase_intensity"] = p["intensity"]

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: EARLY VS LATE COMPARISON
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: Early vs late half comparison")
print("=" * 70)

midpoint_s = df["elapsed_s"].iloc[-1] / 2
early = df[df["elapsed_s"] <= midpoint_s]
late = df[df["elapsed_s"] > midpoint_s]

early_late = []
for label, subset in [("Early Half", early), ("Late Half", late)]:
    d = {
        "period": label,
        "duration_min": round(subset["dt"].sum() / 60, 1),
        "distance_yd": round(subset["step_dist_yd"].sum(), 1),
        "distance_rate_yd_min": round(
            subset["step_dist_yd"].sum() / max(subset["dt"].sum() / 60, 0.01), 1
        ),
        "mean_speed_mph": round(subset["speed_mph"].mean(), 2),
        "max_speed_mph": round(subset["speed_mph"].max(), 2),
        "hsr_distance_yd": round(subset.loc[subset["is_hsr"], "step_dist_yd"].sum(), 1),
        "sprint_distance_yd": round(subset.loc[subset["is_sprint"], "step_dist_yd"].sum(), 1),
        "hsr_events": len(count_events(subset["is_hsr"].reset_index(drop=True), subset["dt"].reset_index(drop=True))),
        "sprint_events": len(count_events(subset["is_sprint"].reset_index(drop=True), subset["dt"].reset_index(drop=True))),
        "accel_events": len(count_events(subset["is_accel"].reset_index(drop=True), subset["dt"].reset_index(drop=True))),
        "decel_events": len(count_events(subset["is_decel"].reset_index(drop=True), subset["dt"].reset_index(drop=True))),
    }
    early_late.append(d)

early_late_df = pd.DataFrame(early_late)

# Add delta row
early_d = early_late_df.iloc[0]
late_d = early_late_df.iloc[1]
if early_d["distance_yd"] > 0:
    dist_delta_pct = round((late_d["distance_yd"] - early_d["distance_yd"]) / early_d["distance_yd"] * 100, 1)
else:
    dist_delta_pct = 0

print(f"  Early half: {early_d['distance_yd']:.1f} yd ({early_d['distance_rate_yd_min']:.1f} yd/min)")
print(f"  Late half:  {late_d['distance_yd']:.1f} yd ({late_d['distance_rate_yd_min']:.1f} yd/min)")
print(f"  Distance delta: {dist_delta_pct:+.1f}%")
print(f"  HSR events: early={early_d['hsr_events']}, late={late_d['hsr_events']}")
print(f"  Sprint events: early={early_d['sprint_events']}, late={late_d['sprint_events']}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: POSITION / ROLE INFERENCE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: Position / role inference")
print("=" * 70)

# Key indicators for position inference:
# 1. Max speed (18.7 mph) — consistent with skill positions (WR, RB, DB, LB)
# 2. Total distance for ~112 min session
# 3. Movement pattern — do they move laterally or in straight lines?
# 4. Field coverage area

x_range = df["x"].max() - df["x"].min()
y_range = df["y"].max() - df["y"].min()
hsr_pct = metrics["hsr_distance_yd"] / total_distance_yd * 100
sprint_pct = metrics["sprint_distance_yd"] / total_distance_yd * 100

# Directional analysis — how often does player change direction?
dir_changes = df["dir"].diff().abs()
# Normalize to [0, 180] (direction changes wrap around 360)
dir_changes = dir_changes.where(dir_changes <= 180, 360 - dir_changes)
mean_dir_change = dir_changes[normal].mean()

print(f"  Field coverage: X={x_range:.0f} yd, Y={y_range:.0f} yd")
print(f"  HSR % of total distance: {hsr_pct:.1f}%")
print(f"  Sprint % of total distance: {sprint_pct:.1f}%")
print(f"  Mean direction change per sample: {mean_dir_change:.1f}°")
print(f"  Max speed: {metrics['max_speed_mph']} mph")
print(f"  Assessment: Skill position (likely WR/DB) — high max speed with")
print(f"              frequent direction changes and broad field coverage.")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8: FIGURE 1 — SPATIAL MOVEMENT MAP
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 8: Generating Figure 1 — Spatial Movement Map")
print("=" * 70)

fig1, axes1 = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1.2, 1]})

# ── Left panel: Movement trace colored by speed band ─────────────────────
ax = axes1[0]

# Only plot continuous segments (skip gaps)
band_colors = {
    "Standing": "#d4d4d4",
    "Walking": "#93c5fd",
    "Jogging": "#60a5fa",
    "Running": "#f59e0b",
    "High-Speed": "#ef4444",
    "Sprint": "#dc2626",
}

# Draw field outline (approximate)
field_rect = plt.Rectangle(
    (df["x"].quantile(0.005), df["y"].quantile(0.005)),
    x_range * 0.99, y_range * 0.99,
    linewidth=1.5, edgecolor="#9ca3af", facecolor="#2d5016", alpha=0.15, zorder=0
)
ax.add_patch(field_rect)

# Plot movement by speed band (low speed first, high speed on top)
for band_name in ["Standing", "Walking", "Jogging", "Running", "High-Speed", "Sprint"]:
    band_mask = df["speed_band"] == band_name
    if band_mask.sum() > 0:
        alpha = 0.15 if band_name in ["Standing", "Walking"] else 0.7
        size = 0.3 if band_name in ["Standing", "Walking"] else 1.5
        ax.scatter(
            df.loc[band_mask, "x"], df.loc[band_mask, "y"],
            c=band_colors[band_name], s=size, alpha=alpha,
            label=band_name, zorder=1 if band_name in ["Standing", "Walking"] else 2,
            rasterized=True,
        )

# Highlight peak windows on the map
for pw in peak_windows:
    if pw["window_s"] <= 60:  # Only show shorter windows to avoid clutter
        pw_slice = df.iloc[pw["start_idx"]:pw["end_idx"]+1]
        ax.plot(pw_slice["x"], pw_slice["y"],
                color="#facc15", linewidth=2.5, alpha=0.9, zorder=5,
                solid_capstyle="round")

ax.set_xlabel("X (yards)", fontsize=11)
ax.set_ylabel("Y (yards)", fontsize=11)
ax.set_title("Movement Trace — Colored by Speed Band", fontsize=13, fontweight="bold", pad=12)
ax.set_aspect("equal")
ax.grid(True, alpha=0.15)

# Legend
legend_patches = [mpatches.Patch(color=band_colors[b["name"]], label=b["name"]) for b in SPEED_BANDS]
legend_patches.append(plt.Line2D([0], [0], color="#facc15", linewidth=2.5, label="Peak Window"))
ax.legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.9)

# ── Right panel: Heatmap (density) ──────────────────────────────────────
ax2 = axes1[1]

# Create 2D histogram for position density
x_bins = np.linspace(df["x"].min() - 2, df["x"].max() + 2, 80)
y_bins = np.linspace(df["y"].min() - 2, df["y"].max() + 2, 80)
heatmap, xedges, yedges = np.histogram2d(df["x"], df["y"], bins=[x_bins, y_bins])
heatmap = gaussian_filter(heatmap.T, sigma=2)

# Custom green-to-red colormap
colors_heat = ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#ffd700"]
cmap = LinearSegmentedColormap.from_list("field_heat", colors_heat, N=256)

im = ax2.imshow(
    heatmap, origin="lower", aspect="equal",
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    cmap=cmap, interpolation="gaussian",
)
plt.colorbar(im, ax=ax2, label="Time Spent (relative)", shrink=0.8)

ax2.set_xlabel("X (yards)", fontsize=11)
ax2.set_ylabel("Y (yards)", fontsize=11)
ax2.set_title("Position Density Heatmap", fontsize=13, fontweight="bold", pad=12)
ax2.grid(True, alpha=0.1, color="white")

fig1.suptitle(
    "WHERE: Spatial Usage and Role Signature\n"
    f"Assessment: Skill position (WR/DB) — {x_range:.0f}×{y_range:.0f} yd coverage, "
    f"max {metrics['max_speed_mph']:.1f} mph",
    fontsize=14, fontweight="bold", y=0.98
)
fig1.tight_layout(rect=[0, 0, 1, 0.93])
fig1.savefig(FIG_DIR / "01_space.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig1)
print("  Saved: 01_space.png")

# ══════════════════════════════════════════════════════════════════════════
# STEP 9: FIGURE 2 — INTENSITY TIMELINE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 9: Generating Figure 2 — Intensity Timeline")
print("=" * 70)

fig2 = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 0.4], hspace=0.08)

# ── Top panel: Speed over time ───────────────────────────────────────────
ax_speed = fig2.add_subplot(gs[0])

# Resample to 1-second bins for cleaner plotting
df["sec_bin"] = (df["elapsed_s"] // 1).astype(int)
sec_agg = df.groupby("sec_bin").agg(
    speed_mph=("speed_mph", "mean"),
    elapsed_min=("elapsed_min", "mean"),
    step_dist_yd=("step_dist_yd", "sum"),
    max_speed_mph=("speed_mph", "max"),
).reset_index()

ax_speed.fill_between(
    sec_agg["elapsed_min"], sec_agg["speed_mph"],
    color="#3b82f6", alpha=0.4, linewidth=0
)
ax_speed.plot(sec_agg["elapsed_min"], sec_agg["speed_mph"],
              color="#1d4ed8", linewidth=0.5, alpha=0.7)

# HSR threshold line
ax_speed.axhline(HSR_THRESHOLD_MPH, color="#ef4444", linestyle="--", linewidth=1, alpha=0.7, label=f"HSR ({HSR_THRESHOLD_MPH} mph)")
ax_speed.axhline(SPRINT_THRESHOLD_MPH, color="#dc2626", linestyle=":", linewidth=1, alpha=0.5, label=f"Sprint ({SPRINT_THRESHOLD_MPH} mph)")

# Highlight peak windows
pw_colors = ["#f59e0b", "#10b981", "#8b5cf6", "#ec4899", "#06b6d4"]
for i, pw in enumerate(peak_windows):
    start_min = pw["start_elapsed_min"]
    end_min = pw["end_elapsed_min"]
    ax_speed.axvspan(start_min, end_min, alpha=0.15, color=pw_colors[i % len(pw_colors)], zorder=0)
    ax_speed.text(
        (start_min + end_min) / 2, ax_speed.get_ylim()[1] * 0.95 if i == 0 else ax_speed.get_ylim()[1] * (0.95 - 0.06 * i),
        pw["window_label"], ha="center", fontsize=7, fontweight="bold",
        color=pw_colors[i % len(pw_colors)], alpha=0.9
    )

ax_speed.set_ylabel("Speed (mph)", fontsize=11)
ax_speed.set_xlim(0, df["elapsed_min"].max())
ax_speed.set_ylim(0, metrics["max_speed_mph"] * 1.1)
ax_speed.legend(loc="upper right", fontsize=8)
ax_speed.set_title(
    "WHEN: Intensity Over Time — Speed, Distance Rate, and Session Phases",
    fontsize=14, fontweight="bold", pad=12
)
ax_speed.tick_params(labelbottom=False)
ax_speed.grid(True, alpha=0.15)

# ── Middle panel: Distance rate (rolling 60s yd/min) ─────────────────────
ax_dist = fig2.add_subplot(gs[1], sharex=ax_speed)

# 60-second rolling distance rate
roll_60 = sec_agg["step_dist_yd"].rolling(60, min_periods=30).sum()  # yd per 60s = yd/min
ax_dist.fill_between(sec_agg["elapsed_min"], roll_60.fillna(0),
                      color="#10b981", alpha=0.4, linewidth=0)
ax_dist.plot(sec_agg["elapsed_min"], roll_60, color="#059669", linewidth=0.8)
ax_dist.set_ylabel("Distance Rate\n(yd/min, 60s window)", fontsize=9)
ax_dist.set_ylim(0, roll_60.max() * 1.2 if roll_60.max() > 0 else 10)
ax_dist.tick_params(labelbottom=False)
ax_dist.grid(True, alpha=0.15)

# ── Bottom panel: Phase strip ────────────────────────────────────────────
ax_phase = fig2.add_subplot(gs[2], sharex=ax_speed)

intensity_colors = {
    "Rest": "#e5e7eb",
    "Low": "#93c5fd",
    "Moderate": "#60a5fa",
    "Moderate-High": "#f59e0b",
    "High": "#ef4444",
}

for _, p in phase_df_summary.iterrows():
    color = intensity_colors.get(p["intensity"], "#d1d5db")
    ax_phase.axvspan(p["start_min"], p["end_min"], color=color, alpha=0.8)

ax_phase.set_xlabel("Elapsed Time (minutes)", fontsize=11)
ax_phase.set_yticks([])
ax_phase.set_ylabel("Phase", fontsize=9, rotation=0, labelpad=35)

# Phase legend
phase_patches = [mpatches.Patch(color=c, label=l) for l, c in intensity_colors.items()]
ax_phase.legend(handles=phase_patches, loc="upper right", fontsize=7, ncol=5, framealpha=0.9)

fig2.tight_layout()
fig2.savefig(FIG_DIR / "02_time.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print("  Saved: 02_time.png")

# ══════════════════════════════════════════════════════════════════════════
# STEP 10: FIGURE 3 — PEAK DEMAND PROFILE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 10: Generating Figure 3 — Peak Demand Profile")
print("=" * 70)

fig3 = plt.figure(figsize=(16, 8))
gs3 = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1], wspace=0.3)

# ── Left panel: Peak intensity curve (yd/min vs duration) ────────────────
ax_peak = fig3.add_subplot(gs3[0])

if len(peak_windows_df) > 0:
    ax_peak.plot(
        peak_windows_df["window_s"],
        peak_windows_df["intensity_yd_min"],
        color="#1d4ed8", linewidth=2.5, marker="o", markersize=8,
        markerfacecolor="#3b82f6", markeredgecolor="white", markeredgewidth=2,
        zorder=3,
    )
    # Annotate each point
    for _, pw in peak_windows_df.iterrows():
        ax_peak.annotate(
            f"{pw['intensity_yd_min']:.0f}\nyd/min",
            (pw["window_s"], pw["intensity_yd_min"]),
            textcoords="offset points", xytext=(0, 15),
            ha="center", fontsize=9, fontweight="bold", color="#1e3a5f",
        )

ax_peak.set_xlabel("Window Duration (seconds)", fontsize=12)
ax_peak.set_ylabel("Peak Intensity (yd/min)", fontsize=12)
ax_peak.set_title("Peak Demand Curve", fontsize=13, fontweight="bold")
ax_peak.set_xticks([pw["window_s"] for pw in peak_windows])
ax_peak.set_xticklabels([pw["window_label"] for pw in peak_windows])
ax_peak.grid(True, alpha=0.2)
ax_peak.set_ylim(0, peak_windows_df["intensity_yd_min"].max() * 1.3 if len(peak_windows_df) > 0 else 100)

# ── Right panel: Speed band distance breakdown ──────────────────────────
ax_bands = fig3.add_subplot(gs3[1])

band_names = [b["name"] for b in SPEED_BANDS]
band_dists = [speed_band_dist.loc[bn, "distance_yd"] if bn in speed_band_dist.index else 0 for bn in band_names]
band_cols = [band_colors.get(bn, "#999") for bn in band_names]

bars = ax_bands.barh(band_names, band_dists, color=band_cols, edgecolor="white", linewidth=0.5)

for bar, dist in zip(bars, band_dists):
    if dist > 0:
        pct = dist / total_distance_yd * 100
        ax_bands.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                      f"{dist:.0f} yd ({pct:.1f}%)",
                      va="center", fontsize=9, fontweight="bold")

ax_bands.set_xlabel("Distance (yards)", fontsize=12)
ax_bands.set_title("Distance by Speed Band", fontsize=13, fontweight="bold")
ax_bands.grid(True, axis="x", alpha=0.2)
ax_bands.invert_yaxis()

fig3.suptitle(
    f"WHAT: Peak Demands — Total Distance {total_distance_yd:.0f} yd | "
    f"HSR {metrics['hsr_distance_yd']} yd | Sprint {metrics['sprint_distance_yd']} yd",
    fontsize=14, fontweight="bold", y=0.98,
)
fig3.tight_layout(rect=[0, 0, 1, 0.93])
fig3.savefig(FIG_DIR / "03_peaks.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig3)
print("  Saved: 03_peaks.png")

# ══════════════════════════════════════════════════════════════════════════
# STEP 11: EXPORT TABLES AND RESULTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 11: Exporting tables and results contract")
print("=" * 70)

# Speed band table
speed_band_export = speed_band_dist.reset_index()
speed_band_export.columns = ["Zone", "Distance (yd)", "Samples", "Time (s)", "Distance (%)", "Time (min)", "Time (%)"]
speed_band_export.to_csv(TABLE_DIR / "speed_band_summary.csv", index=False)
print("  Saved: speed_band_summary.csv")

# Peak windows table
peak_windows_df.to_csv(TABLE_DIR / "peak_windows.csv", index=False)
print("  Saved: peak_windows.csv")

# Phase summary table
phase_df_summary.to_csv(TABLE_DIR / "phase_summary.csv", index=False)
print("  Saved: phase_summary.csv")

# Early vs late table
early_late_df.to_csv(TABLE_DIR / "early_vs_late.csv", index=False)
print("  Saved: early_vs_late.csv")

# Event counts table
pd.DataFrame([event_counts]).to_csv(TABLE_DIR / "event_counts.csv", index=False)
print("  Saved: event_counts.csv")

# Session metrics
pd.DataFrame([metrics]).to_csv(TABLE_DIR / "session_metrics.csv", index=False)
print("  Saved: session_metrics.csv")

# QC summary
pd.DataFrame([qc_summary]).to_csv(TABLE_DIR / "qc_summary.csv", index=False)
print("  Saved: qc_summary.csv")

# Results contract (JSON)
contract = {
    "session_summary": {**qc_summary, **metrics, **event_counts},
    "thresholds": {
        "speed_bands_mph": SPEED_BANDS,
        "hsr_threshold_mph": HSR_THRESHOLD_MPH,
        "sprint_threshold_mph": SPRINT_THRESHOLD_MPH,
        "accel_threshold_ms2": ACCEL_THRESHOLD_MS2,
        "decel_threshold_ms2": DECEL_THRESHOLD_MS2,
    },
    "units": {
        "distance": "yards",
        "speed": "mph",
        "acceleration": "m/s²",
        "speed_conversion": {"from": "yd/s", "to": "mph", "factor": YDS_TO_MPH},
        "accel_conversion": {"from": "yd/s²", "to": "m/s²", "factor": YDS2_TO_MS2},
    },
    "peak_windows": peak_windows,
    "early_vs_late_delta_pct": dist_delta_pct,
    "figures": {
        "01_space": str(FIG_DIR / "01_space.png"),
        "02_time": str(FIG_DIR / "02_time.png"),
        "03_peaks": str(FIG_DIR / "03_peaks.png"),
    },
}

with open(OUT_DIR / "results.json", "w") as f:
    json.dump(contract, f, indent=2, default=str)
print("  Saved: results.json")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
print(f"\nAll outputs in: {OUT_DIR}")
print(f"Figures: {list(FIG_DIR.glob('*.png'))}")
print(f"Tables: {list(TABLE_DIR.glob('*.csv'))}")
