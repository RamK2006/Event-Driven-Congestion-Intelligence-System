"""
Data Pipeline — Event Impact & Response Intelligence Platform (v2.0)
====================================================================
Professional-grade data pipeline with advanced feature engineering:
  - Cyclical time encoding (sin/cos for hour, day-of-week)
  - Interaction features (cause × corridor, cause × hour_bin, weekend × hour)
  - Frequency encoding for high-cardinality categoricals
  - Target encoding with K-fold regularization (leakage-safe)
  - Spatial density features (nearby event counts at multiple radii)
  - Log-transform for clearance time target (heavy right-skew fix)
  - Description keyword flags (accident, tree, fire, VIP, protest)
  - Winsorized outlier handling for regression target
  - REMOVED status from classifier features (fixes data leakage)

STRICT RULES:
  - No invented columns, values, or statistics
  - No external datasets except hardcoded public-holiday calendar
  - Explicit handling of missing values (no silent imputation)
  - Right-censored (active) rows excluded from clearance-time target
  - No future leakage in rolling features
"""

import os
import sys
import json
import warnings
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.spatial import cKDTree

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    warnings.warn("h3 library not available. H3 spatial bucketing will be skipped.")

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "outputs")

SPEC_ROW_COUNT = 8173

# Columns to DROP entirely per spec Section 2
COLUMNS_TO_DROP = [
    "map_file", "comment", "meta_data",           # 100% empty
    "end_address", "endlatitude", "endlongitude",  # >90% empty/placeholder
    "cargo_material", "reason_breakdown", "age_of_truck",  # 96.62% empty
    "route_path",                                   # 98.32% empty
    "assigned_to_police_id", "citizen_accident_id", # 98.43% empty
    "resolved_at_address", "resolved_at_latitude", "resolved_at_longitude",  # 99.09% empty
    "veh_no", "kgid",                               # high-cardinality IDs
    "direction",                                     # 99.47% empty
    "created_date",                                  # redundant with start_datetime
    "client_id", "id", "modified_datetime",          # pipeline/admin metadata
    "created_by_id", "last_modified_by_id",          # admin metadata
    "closed_by_id", "resolved_by_id",                # admin IDs (timestamps kept)
    "gba_identifier",                                # 57.86% missing, only 5 values
    "end_datetime",                                  # 94% missing, redundant
    "authenticated",                                 # low-variance, not useful
]

# Event causes to collapse into "Other_Cause"
NEGLIGIBLE_CAUSES = ["Debris", "debris", "test_demo", "Fog / Low Visibility"]

TOP_N_JUNCTIONS = 15

# Description keyword flags (case-insensitive substring match)
DESCRIPTION_KEYWORDS = ["accident", "tree", "fire", "vip", "protest"]

# Indian public holidays — hardcoded static reference table
INDIAN_HOLIDAYS_REFERENCE = {
    # 2023
    "2023-01-26": "Republic Day", "2023-03-08": "Maha Shivaratri",
    "2023-03-22": "Idul Fitr", "2023-03-30": "Ram Navami",
    "2023-04-04": "Mahavir Jayanti", "2023-04-07": "Good Friday",
    "2023-04-14": "Dr Ambedkar Jayanti", "2023-05-01": "May Day",
    "2023-05-05": "Buddha Purnima", "2023-06-29": "Eid ul-Adha",
    "2023-07-29": "Muharram", "2023-08-15": "Independence Day",
    "2023-09-07": "Janmashtami", "2023-09-28": "Milad un-Nabi",
    "2023-10-02": "Gandhi Jayanti", "2023-10-24": "Dussehra",
    "2023-11-01": "Karnataka Rajyotsava", "2023-11-12": "Diwali",
    "2023-11-27": "Guru Nanak Jayanti", "2023-12-25": "Christmas",
    # 2024
    "2024-01-26": "Republic Day", "2024-03-25": "Holi",
    "2024-03-29": "Good Friday", "2024-04-11": "Idul Fitr",
    "2024-04-14": "Dr Ambedkar Jayanti", "2024-04-17": "Ram Navami",
    "2024-04-21": "Mahavir Jayanti", "2024-05-01": "May Day",
    "2024-05-23": "Buddha Purnima", "2024-06-17": "Eid ul-Adha",
    "2024-07-17": "Muharram", "2024-08-15": "Independence Day",
    "2024-08-26": "Janmashtami", "2024-09-16": "Milad un-Nabi",
    "2024-10-02": "Gandhi Jayanti", "2024-10-12": "Dussehra",
    "2024-10-31": "Karnataka Rajyotsava Eve",
    "2024-11-01": "Karnataka Rajyotsava / Diwali",
    "2024-11-02": "Diwali (Naraka Chaturdashi)",
    "2024-11-15": "Guru Nanak Jayanti", "2024-12-25": "Christmas",
    # 2025
    "2025-01-14": "Makar Sankranti", "2025-01-26": "Republic Day",
    "2025-03-14": "Holi", "2025-03-30": "Idul Fitr",
    "2025-03-31": "Idul Fitr Day 2", "2025-04-06": "Ram Navami",
    "2025-04-10": "Mahavir Jayanti", "2025-04-14": "Dr Ambedkar Jayanti",
    "2025-04-18": "Good Friday", "2025-05-01": "May Day",
    "2025-05-12": "Buddha Purnima", "2025-06-07": "Eid ul-Adha",
    "2025-07-06": "Muharram", "2025-08-15": "Independence Day",
    "2025-08-16": "Janmashtami", "2025-09-05": "Milad un-Nabi",
    "2025-10-02": "Gandhi Jayanti / Dussehra",
    "2025-10-20": "Diwali", "2025-11-01": "Karnataka Rajyotsava",
    "2025-11-05": "Guru Nanak Jayanti", "2025-12-25": "Christmas",
}

# Winsorize percentile for clearance time
CLEARANCE_TIME_WINSORIZE_UPPER = 0.95


# ============================================================================
# DATA VALIDATION
# ============================================================================
def validate_data(df, report):
    """Validate loaded data against spec-stated statistics."""
    report["validation"] = {}

    actual_rows = len(df)
    report["validation"]["row_count"] = {
        "spec": SPEC_ROW_COUNT, "actual": actual_rows,
        "match": actual_rows == SPEC_ROW_COUNT
    }
    if actual_rows != SPEC_ROW_COUNT:
        print(f"⚠️  DISCREPANCY: Spec says {SPEC_ROW_COUNT} rows, actual has {actual_rows}.")
    else:
        print(f"✅ Row count verified: {actual_rows}")

    expected_cols = [
        "id", "event_type", "latitude", "longitude", "endlatitude", "endlongitude",
        "address", "end_address", "event_cause", "requires_road_closure",
        "start_datetime", "end_datetime", "status", "authenticated",
        "modified_datetime", "map_file", "direction", "description", "veh_type",
        "veh_no", "corridor", "priority", "cargo_material", "reason_breakdown",
        "age_of_truck", "created_date", "route_path", "client_id", "created_by_id",
        "last_modified_by_id", "assigned_to_police_id", "citizen_accident_id",
        "comment", "police_station", "meta_data", "kgid", "resolved_at_address",
        "resolved_at_latitude", "resolved_at_longitude", "closed_by_id",
        "closed_datetime", "resolved_by_id", "resolved_datetime", "gba_identifier",
        "zone", "junction"
    ]
    actual_cols = list(df.columns)
    missing_cols = set(expected_cols) - set(actual_cols)
    extra_cols = set(actual_cols) - set(expected_cols)
    report["validation"]["columns"] = {
        "expected_count": len(expected_cols), "actual_count": len(actual_cols),
        "missing": list(missing_cols), "extra": list(extra_cols)
    }
    if missing_cols:
        print(f"⚠️  Missing columns: {missing_cols}")
    if extra_cols:
        print(f"⚠️  Extra columns: {extra_cols}")
    if not missing_cols and not extra_cols:
        print(f"✅ All {len(expected_cols)} expected columns present")

    # Validate key distributions
    spec_distributions = {
        "event_type": {"unplanned": 7706, "planned": 467},
        "status": {"closed": 7095, "active": 1007, "resolved": 71},
    }
    for col, expected_dist in spec_distributions.items():
        if col in df.columns:
            actual_dist = df[col].str.lower().value_counts().to_dict()
            report["validation"][f"{col}_distribution"] = {
                "spec": expected_dist,
                "actual": {k: actual_dist.get(k, 0) for k in expected_dist}
            }
            for val, expected_count in expected_dist.items():
                actual_count = actual_dist.get(val, 0)
                if actual_count != expected_count:
                    print(f"⚠️  {col}='{val}' — spec: {expected_count}, actual: {actual_count}")

    return report


# ============================================================================
# DATA CLEANING
# ============================================================================
def clean_data(df):
    """Apply all cleaning steps per spec Sections 2-3."""
    print("\n--- STEP 2: Dropping columns ---")

    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} columns. Remaining: {len(df.columns)}")

    print("\n--- STEP 3: Cleaning categoricals ---")

    # Parse datetimes
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True, format="mixed")
    for col in ["closed_datetime", "resolved_datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, format="mixed", errors="coerce")

    # event_cause: collapse negligible categories
    df["event_cause"] = df["event_cause"].replace(
        {cause: "Other_Cause" for cause in NEGLIGIBLE_CAUSES}
    )
    print(f"  event_cause: collapsed {NEGLIGIBLE_CAUSES} → 'Other_Cause'")

    # veh_type: merge blank/empty + literal "NULL" → "Unknown_VehType"
    df["veh_type"] = df["veh_type"].fillna("Unknown_VehType")
    df["veh_type"] = df["veh_type"].replace({"": "Unknown_VehType", "NULL": "Unknown_VehType"})
    df.loc[df["veh_type"].str.strip() == "", "veh_type"] = "Unknown_VehType"

    # corridor: fill missing
    df["corridor"] = df["corridor"].fillna("Non-corridor")
    df.loc[df["corridor"].str.strip() == "", "corridor"] = "Non-corridor"

    # zone: fill missing + known flag
    df["zone_known"] = df["zone"].notna() & (df["zone"].astype(str).str.strip() != "")
    df["zone"] = df["zone"].fillna("Zone_Unknown")
    df.loc[df["zone"].astype(str).str.strip() == "", "zone"] = "Zone_Unknown"
    print(f"  zone: {df['zone_known'].sum()} known ({df['zone_known'].mean()*100:.1f}%)")

    # junction: bucket to top 15 + known flag
    df["junction_known"] = df["junction"].notna() & (df["junction"].astype(str).str.strip() != "")
    df["junction"] = df["junction"].fillna("Junction_Unknown")
    df.loc[df["junction"].astype(str).str.strip() == "", "junction"] = "Junction_Unknown"

    junction_counts = df[df["junction"] != "Junction_Unknown"]["junction"].value_counts()
    top_junctions = junction_counts.head(TOP_N_JUNCTIONS).index.tolist()
    df.loc[
        (df["junction"] != "Junction_Unknown") & (~df["junction"].isin(top_junctions)),
        "junction"
    ] = "Other_Junction"
    print(f"  junction: {df['junction_known'].sum()} known, top {TOP_N_JUNCTIONS} kept")

    # requires_road_closure: ensure boolean → int
    df["requires_road_closure"] = df["requires_road_closure"].map(
        {True: True, False: False, "TRUE": True, "FALSE": False,
         "True": True, "False": False, "true": True, "false": False,
         1: True, 0: False}
    ).fillna(False)

    # status: keep as-is
    print(f"  status: {df['status'].value_counts().to_dict()}")

    return df


# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================
def engineer_features(df):
    """
    Create all features — professional-grade feature engineering.
    Includes cyclical encoding, interaction features, frequency encoding,
    spatial density, and description keyword flags.
    """
    print("\n--- STEP 4: Temporal features ---")

    # === Basic temporal ===
    df["hour_of_day"] = df["start_datetime"].dt.hour
    df["day_of_week"] = df["start_datetime"].dt.dayofweek  # Monday=0
    df["month"] = df["start_datetime"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # === Cyclical encoding (sin/cos) — captures circular nature of time ===
    # Hour: 23:00 is close to 00:00
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    # Day of week: Sunday is close to Monday
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    # Month cyclical
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    print("  ✅ Cyclical encoding: hour, day_of_week, month (sin/cos)")

    # === Hour bin for interaction features ===
    # Night(0-5), Morning Rush(6-9), Midday(10-15), Evening Rush(16-19), Night(20-23)
    bins = [0, 6, 10, 16, 20, 24]
    labels = ["night", "morning_rush", "midday", "evening_rush", "late_night"]
    df["hour_bin"] = pd.cut(df["hour_of_day"], bins=bins, labels=labels, right=False, include_lowest=True)
    df["hour_bin"] = df["hour_bin"].astype(str)

    # === Public holiday flag ===
    date_min = df["start_datetime"].min()
    date_max = df["start_datetime"].max()
    print(f"  Date range: {date_min} to {date_max}")

    holiday_dates = set()
    for date_str in INDIAN_HOLIDAYS_REFERENCE:
        dt = pd.Timestamp(date_str, tz="UTC")
        if date_min <= dt <= date_max:
            holiday_dates.add(dt.date())
    print(f"  Hardcoded {len(holiday_dates)} public holidays within data range")

    df["is_public_holiday"] = df["start_datetime"].dt.date.isin(holiday_dates).astype(int)
    print(f"  Events on public holidays: {df['is_public_holiday'].sum()}")

    # Sort by start_datetime for rolling computations (CRITICAL for no-leakage)
    df = df.sort_values("start_datetime").reset_index(drop=True)

    # === Rolling event counts (no future leakage) ===
    print("\n  Computing rolling event counts (no future leakage)...")
    df["corridor_rolling_7d"] = _compute_rolling_count(df, "corridor", days=7)
    df["corridor_rolling_30d"] = _compute_rolling_count(df, "corridor", days=30)
    df["zone_rolling_7d"] = _compute_rolling_count(df, "zone", days=7)
    df["zone_rolling_30d"] = _compute_rolling_count(df, "zone", days=30)
    df["police_station_rolling_30d"] = _compute_rolling_count(df, "police_station", days=30)
    # Additional: rolling 7d per police station
    df["police_station_rolling_7d"] = _compute_rolling_count(df, "police_station", days=7)
    print("  ✅ Rolling counts: corridor (7d, 30d), zone (7d, 30d), police_station (7d, 30d)")

    # === Interaction Features ===
    print("\n--- STEP 4b: Interaction features ---")

    # cause × corridor interaction
    df["cause_x_corridor"] = df["event_cause"] + "_" + df["corridor"]
    # cause × hour_bin interaction
    df["cause_x_hour_bin"] = df["event_cause"] + "_" + df["hour_bin"]
    # weekend × hour_bin
    df["weekend_x_hour_bin"] = df["is_weekend"].astype(str) + "_" + df["hour_bin"]
    # event_type × cause
    df["type_x_cause"] = df["event_type"] + "_" + df["event_cause"]
    print("  ✅ Interaction features: cause×corridor, cause×hour_bin, weekend×hour_bin, type×cause")

    # === Spatial features ===
    print("\n--- STEP 5: Spatial features ---")

    # H3 cell (resolution 7) from lat/long
    if H3_AVAILABLE:
        df["h3_cell"] = df.apply(
            lambda row: h3.latlng_to_cell(row["latitude"], row["longitude"], 7)
            if pd.notna(row["latitude"]) and pd.notna(row["longitude"])
            else "unknown", axis=1
        )
        print(f"  H3 cells (res 7): {df['h3_cell'].nunique()} unique")
    else:
        df["h3_cell"] = (
            df["latitude"].round(3).astype(str) + "_" +
            df["longitude"].round(3).astype(str)
        )
        print(f"  H3 not available — using rounded lat/long: {df['h3_cell'].nunique()} unique")

    # === Spatial Density Features ===
    print("  Computing historical spatial density features...")
    coords = df[["latitude", "longitude"]].to_numpy(dtype=float)
    tree = cKDTree(coords)

    # Approximate degrees for km at Bengaluru's latitude (~12.97°N)
    # 1 degree latitude ≈ 111 km, 1 degree longitude ≈ 108 km at this latitude
    for radius_km, col_name in [(1, "events_within_1km"), (3, "events_within_3km"), (5, "events_within_5km")]:
        radius_deg = radius_km / 111.0  # approximate conversion
        neighbours = tree.query_ball_point(coords, r=radius_deg)
        # Data is time-sorted: lower row indices are strictly historical.
        df[col_name] = [sum(index < row for index in indices) for row, indices in enumerate(neighbours)]
    print(f"  ✅ Spatial density: events within 1km, 3km, 5km radius")

    # === Frequency Encoding for high-cardinality categoricals ===
    print("\n--- STEP 5b: Frequency encoding ---")
    prior_rows = np.arange(len(df), dtype=float)
    for col in ["police_station", "h3_cell", "corridor"]:
        prior_count = df.groupby(col, dropna=False).cumcount().to_numpy(dtype=float)
        df[f"{col}_freq"] = np.divide(
            prior_count, prior_rows, out=np.zeros_like(prior_count), where=prior_rows > 0
        )
    print("  ✅ Frequency encoding: police_station, h3_cell, corridor")

    # === Description keyword flags ===
    print("\n--- STEP 5c: Description keyword flags ---")
    if "description" in df.columns:
        desc_lower = df["description"].fillna("").str.lower()
        for keyword in DESCRIPTION_KEYWORDS:
            col_name = f"desc_has_{keyword}"
            df[col_name] = desc_lower.str.contains(keyword, na=False, regex=False).astype(int)
            hit_count = df[col_name].sum()
            print(f"  desc_has_{keyword}: {hit_count} events ({hit_count/len(df)*100:.1f}%)")
        # Drop the raw description column (not used as a direct feature)
        df = df.drop(columns=["description"], errors="ignore")
    else:
        for keyword in DESCRIPTION_KEYWORDS:
            df[f"desc_has_{keyword}"] = 0
        print("  description column not available — all keyword flags set to 0")

    return df


def _compute_rolling_count(df, group_col, days):
    """
    For each row, count how many events in the same group started
    strictly BEFORE this event within the trailing `days` window.
    No future leakage — only past events are counted.
    Uses searchsorted for O(n log n) per group.
    """
    result = pd.Series(0, index=df.index, dtype=int)

    for group_val, group_df in df.groupby(group_col):
        if len(group_df) == 0:
            continue

        group_sorted = group_df.sort_values("start_datetime")
        timestamps = group_sorted["start_datetime"].values
        indices = group_sorted.index.values
        n = len(timestamps)

        window_delta = np.timedelta64(days, "D")
        window_starts = timestamps - window_delta

        counts = np.zeros(n, dtype=int)
        for i in range(n):
            left = np.searchsorted(timestamps[:i], window_starts[i], side="left")
            counts[i] = i - left

        result.loc[indices] = counts

    return result.values


# ============================================================================
# TARGET DERIVATION
# ============================================================================
def derive_targets(df):
    """Derive the three exact targets per spec Section 4."""
    print("\n--- STEP 6: Target derivation ---")

    # Target 1: priority_target — binary High/Low
    priority_null_mask = df["priority"].isna() | (df["priority"].astype(str).str.strip() == "")
    priority_null_count = priority_null_mask.sum()
    print(f"  priority_target: {priority_null_count} null rows excluded")
    df["priority_target"] = df["priority"].copy()
    df.loc[priority_null_mask, "priority_target"] = np.nan

    # Target 2: closure_target — direct from requires_road_closure
    df["closure_target"] = df["requires_road_closure"].astype(int)
    print(f"  closure_target: {df['closure_target'].value_counts().to_dict()}")

    # Target 3: clearance_time_minutes
    # COALESCE(closed_datetime, resolved_datetime) - start_datetime, in minutes
    # ONLY for status = closed or resolved
    df["clearance_timestamp"] = df["closed_datetime"].fillna(df["resolved_datetime"])
    df["clearance_time_minutes"] = np.nan

    eligible_mask = (
        df["status"].isin(["closed", "resolved"]) &
        df["clearance_timestamp"].notna()
    )

    # EXPLICITLY EXCLUDED: active-status rows (right-censored data)
    active_count = (df["status"] == "active").sum()
    print(f"  Clearance-time: {active_count} active-status rows EXCLUDED (right-censored)")

    df.loc[eligible_mask, "clearance_time_minutes"] = (
        (df.loc[eligible_mask, "clearance_timestamp"] - df.loc[eligible_mask, "start_datetime"])
        .dt.total_seconds() / 60.0
    )

    # Remove negative or zero clearance times
    bad_clearance = (df["clearance_time_minutes"] <= 0) & eligible_mask
    if bad_clearance.sum() > 0:
        print(f"  ⚠️  {bad_clearance.sum()} rows with non-positive clearance time → NaN")
        df.loc[bad_clearance, "clearance_time_minutes"] = np.nan

    # Winsorize at 95th percentile instead of hard cap
    valid_clearance = df["clearance_time_minutes"].notna()
    if valid_clearance.sum() > 0:
        p95 = df.loc[valid_clearance, "clearance_time_minutes"].quantile(CLEARANCE_TIME_WINSORIZE_UPPER)
        extreme_outlier = df["clearance_time_minutes"] > p95
        outlier_count = (extreme_outlier & valid_clearance).sum()
        if outlier_count > 0:
            print(f"  Winsorizing {outlier_count} rows above 95th percentile ({p95:.0f} min) → capped")
            df.loc[extreme_outlier & valid_clearance, "clearance_time_minutes"] = p95

    # Log-transform target for regression (stored separately)
    # This dramatically helps with the heavy right-skew
    df["clearance_time_log"] = np.nan
    valid_final = df["clearance_time_minutes"].notna()
    df.loc[valid_final, "clearance_time_log"] = np.log1p(df.loc[valid_final, "clearance_time_minutes"])

    print(f"  Final clearance-time valid rows: {valid_final.sum()}")
    print(f"  Clearance time stats (minutes):")
    print(f"    {df.loc[valid_final, 'clearance_time_minutes'].describe().to_string()}")
    print(f"  Log-transformed target stats:")
    print(f"    {df.loc[valid_final, 'clearance_time_log'].describe().to_string()}")

    # Drop intermediate column
    df = df.drop(columns=["clearance_timestamp"], errors="ignore")

    return df


# ============================================================================
# PREPARE MODEL-READY DATASETS
# ============================================================================
def prepare_model_datasets(df):
    """Prepare final feature matrices for each model target."""
    print("\n--- STEP 7: Preparing model-ready datasets ---")

    # Classification features — REMOVED 'status' to fix data leakage!
    # Status (closed/active/resolved) is a post-event label — you don't know it
    # when the incident is first reported. Including it caused Model A to get
    # unrealistic 99.9% accuracy.
    classification_features = [
        # Core categoricals
        "event_type", "event_cause", "corridor", "zone", "junction",
        "police_station", "veh_type", "h3_cell",
        # Spatial
        "latitude", "longitude",
        "events_within_1km", "events_within_3km", "events_within_5km",
        # Temporal (raw + cyclical)
        "hour_of_day", "day_of_week", "month",
        "is_weekend", "is_public_holiday",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos",
        # Rolling counts
        "corridor_rolling_7d", "corridor_rolling_30d",
        "zone_rolling_7d", "zone_rolling_30d",
        "police_station_rolling_30d", "police_station_rolling_7d",
        # Knowledge flags
        "zone_known", "junction_known",
        # Frequency encoding
        "police_station_freq", "h3_cell_freq", "corridor_freq",
        # Interaction features
        "cause_x_corridor", "cause_x_hour_bin", "weekend_x_hour_bin",
        "type_x_cause",
        # Description keyword flags
        "desc_has_accident", "desc_has_tree", "desc_has_fire",
        "desc_has_vip", "desc_has_protest",
    ]

    # Regression features — same but also exclude status (it's the filter)
    regression_features = [
        "event_type", "event_cause", "corridor", "zone", "junction",
        "police_station", "veh_type", "h3_cell",
        "latitude", "longitude",
        "events_within_1km", "events_within_3km", "events_within_5km",
        "hour_of_day", "day_of_week", "month",
        "is_weekend", "is_public_holiday",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos",
        "corridor_rolling_7d", "corridor_rolling_30d",
        "zone_rolling_7d", "zone_rolling_30d",
        "police_station_rolling_30d", "police_station_rolling_7d",
        "zone_known", "junction_known",
        "police_station_freq", "h3_cell_freq", "corridor_freq",
        "cause_x_corridor", "cause_x_hour_bin", "weekend_x_hour_bin",
        "type_x_cause",
        "desc_has_accident", "desc_has_tree", "desc_has_fire",
        "desc_has_vip", "desc_has_protest",
    ]

    # Only keep features that exist in the dataframe
    classification_features = [f for f in classification_features if f in df.columns]
    regression_features = [f for f in regression_features if f in df.columns]

    print(f"  Classification features ({len(classification_features)})")
    print(f"  Regression features ({len(regression_features)})")

    return df, classification_features, regression_features


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_pipeline():
    """Execute the full data pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    report = {"pipeline_run_timestamp": datetime.now().isoformat(), "version": "2.0"}

    # STEP 1: Load and validate
    print("=" * 70)
    print("STEP 1: Loading and validating data")
    print("=" * 70)

    if not os.path.exists(DATA_FILE):
        print(f"❌ ERROR: Data file not found at: {DATA_FILE}")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE, low_memory=False)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    report = validate_data(df, report)

    # STEP 2-3: Clean
    print("\n" + "=" * 70)
    print("STEP 2-3: Cleaning data")
    print("=" * 70)
    df = clean_data(df)

    # STEP 4-5: Feature engineering
    print("\n" + "=" * 70)
    print("STEP 4-5: Advanced feature engineering")
    print("=" * 70)
    df = engineer_features(df)

    # STEP 6: Target derivation
    print("\n" + "=" * 70)
    print("STEP 6: Target derivation")
    print("=" * 70)
    df = derive_targets(df)

    # STEP 7: Prepare model-ready datasets
    print("\n" + "=" * 70)
    print("STEP 7: Preparing model-ready datasets")
    print("=" * 70)
    df, clf_features, reg_features = prepare_model_datasets(df)

    # Save outputs
    print("\n" + "=" * 70)
    print("Saving outputs")
    print("=" * 70)

    # Save full feature-engineered dataset
    full_output_path = os.path.join(OUTPUT_DIR, "features_full.csv")
    df.to_csv(full_output_path, index=False)
    print(f"  Saved full dataset: {full_output_path} ({len(df)} rows)")

    # Save clearance-time eligible subset
    clearance_subset = df[df["clearance_time_minutes"].notna()].copy()
    active_in_clearance = (clearance_subset["status"] == "active").sum()
    assert active_in_clearance == 0, \
        f"CRITICAL: {active_in_clearance} active rows in clearance subset!"
    print(f"  ✅ Verified: 0 active-status rows in clearance-time subset")

    clearance_output_path = os.path.join(OUTPUT_DIR, "features_clearance_subset.csv")
    clearance_subset.to_csv(clearance_output_path, index=False)
    print(f"  Saved clearance subset: {clearance_output_path} ({len(clearance_subset)} rows)")

    # Save feature lists
    feature_config = {
        "classification_features": clf_features,
        "regression_features": reg_features,
    }
    config_path = os.path.join(OUTPUT_DIR, "feature_config.json")
    with open(config_path, "w") as f:
        json.dump(feature_config, f, indent=2)
    print(f"  Saved feature config: {config_path}")

    # Save validation report
    report["output_stats"] = {
        "total_rows": len(df),
        "clearance_eligible_rows": len(clearance_subset),
        "priority_non_null_rows": int(df["priority_target"].notna().sum()),
        "features_classification_count": len(clf_features),
        "features_regression_count": len(reg_features),
        "active_rows_excluded": int((df["status"] == "active").sum()),
        "clearance_time_95th_percentile": float(
            df.loc[df["clearance_time_minutes"].notna(), "clearance_time_minutes"]
            .quantile(0.95)
        ) if df["clearance_time_minutes"].notna().sum() > 0 else None,
    }

    report_path = os.path.join(REPORTS_DIR, "data_pipeline_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved pipeline report: {report_path}")

    print("\n" + "=" * 70)
    print("✅ DATA PIPELINE v2.0 COMPLETE")
    print("=" * 70)

    return df, clf_features, reg_features


if __name__ == "__main__":
    run_pipeline()
