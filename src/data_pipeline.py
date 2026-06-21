"""
Data Pipeline — Event Impact & Response Intelligence Platform
=============================================================
Loads, validates, cleans, and feature-engineers the Astram event dataset
exactly per the finalized project specification. Every inclusion/exclusion
decision is documented in code comments.

STRICT RULES FOLLOWED:
- No invented columns, values, or statistics
- No external datasets except hardcoded public-holiday calendar
- Explicit handling of missing values (no silent imputation)
- Right-censored (active) rows excluded from clearance-time target
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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

# Expected row count from spec (will be validated, not enforced)
SPEC_ROW_COUNT = 8173

# Columns to DROP entirely per spec Section 2
COLUMNS_TO_DROP = [
    "map_file", "comment", "meta_data",           # 100% empty
    "end_address", "endlatitude", "endlongitude",  # >90% empty/placeholder
    "cargo_material", "reason_breakdown", "age_of_truck",  # 96.62% empty
    "route_path",                                   # 98.32% empty
    "assigned_to_police_id", "citizen_accident_id", # 98.43% empty
    "resolved_at_address", "resolved_at_latitude", "resolved_at_longitude",  # 99.09% empty
    "veh_no", "kgid",                               # high-cardinality IDs, no generalizable signal
    "direction",                                     # 99.47% empty
    "created_date",                                  # redundant with start_datetime
    "client_id", "id", "modified_datetime",          # pipeline/admin metadata
    "created_by_id", "last_modified_by_id",          # admin metadata
    "closed_by_id", "resolved_by_id",                # admin IDs (timestamps kept for target derivation)
    "gba_identifier",                                # 57.86% missing, only 5 values
    "end_datetime",                                  # 94% missing, redundant with closed/resolved datetime
]

# Event causes to collapse into "Other_Cause" per spec Section 5
NEGLIGIBLE_CAUSES = ["Debris", "debris", "test_demo", "Fog / Low Visibility"]

# Top N junctions to keep as individual categories per spec
TOP_N_JUNCTIONS = 15

# Indian public holidays — hardcoded static reference table per spec Section 5.
# Date range will be determined from the data and only matching holidays included.
INDIAN_HOLIDAYS_REFERENCE = {
    # 2024 holidays (national + Karnataka state)
    "2024-01-26": "Republic Day",
    "2024-03-25": "Holi",
    "2024-03-29": "Good Friday",
    "2024-04-11": "Idul Fitr (Eid ul-Fitr)",
    "2024-04-14": "Dr Ambedkar Jayanti",
    "2024-04-17": "Ram Navami",
    "2024-04-21": "Mahavir Jayanti",
    "2024-05-01": "May Day",
    "2024-05-23": "Buddha Purnima",
    "2024-06-17": "Eid ul-Adha (Bakrid)",
    "2024-07-17": "Muharram",
    "2024-08-15": "Independence Day",
    "2024-08-26": "Janmashtami",
    "2024-09-16": "Milad un-Nabi",
    "2024-10-02": "Gandhi Jayanti",
    "2024-10-12": "Dussehra (Vijayadashami)",
    "2024-10-31": "Halloween / Karnataka Rajyotsava Eve",
    "2024-11-01": "Karnataka Rajyotsava / Diwali",
    "2024-11-02": "Diwali (Naraka Chaturdashi)",
    "2024-11-15": "Guru Nanak Jayanti",
    "2024-12-25": "Christmas",
    # 2025 holidays
    "2025-01-14": "Makar Sankranti",
    "2025-01-26": "Republic Day",
    "2025-03-14": "Holi",
    "2025-03-30": "Idul Fitr (Eid ul-Fitr)",
    "2025-03-31": "Idul Fitr (Eid ul-Fitr) Day 2",
    "2025-04-06": "Ram Navami",
    "2025-04-10": "Mahavir Jayanti",
    "2025-04-14": "Dr Ambedkar Jayanti",
    "2025-04-18": "Good Friday",
    "2025-05-01": "May Day",
    "2025-05-12": "Buddha Purnima",
    "2025-06-07": "Eid ul-Adha (Bakrid)",
    "2025-07-06": "Muharram",
    "2025-08-15": "Independence Day",
    "2025-08-16": "Janmashtami",
    "2025-09-05": "Milad un-Nabi",
    "2025-10-02": "Gandhi Jayanti",
    "2025-10-02": "Dussehra (Vijayadashami)",
    "2025-10-20": "Diwali",
    "2025-11-01": "Karnataka Rajyotsava",
    "2025-11-05": "Guru Nanak Jayanti",
    "2025-12-25": "Christmas",
    # 2023 holidays (in case data goes back)
    "2023-01-26": "Republic Day",
    "2023-03-08": "Maha Shivaratri",
    "2023-03-22": "Idul Fitr",
    "2023-03-30": "Ram Navami",
    "2023-04-04": "Mahavir Jayanti",
    "2023-04-07": "Good Friday",
    "2023-04-14": "Dr Ambedkar Jayanti",
    "2023-05-01": "May Day",
    "2023-05-05": "Buddha Purnima",
    "2023-06-29": "Eid ul-Adha",
    "2023-07-29": "Muharram",
    "2023-08-15": "Independence Day",
    "2023-09-07": "Janmashtami",
    "2023-09-28": "Milad un-Nabi",
    "2023-10-02": "Gandhi Jayanti",
    "2023-10-24": "Dussehra",
    "2023-11-01": "Karnataka Rajyotsava",
    "2023-11-12": "Diwali",
    "2023-11-27": "Guru Nanak Jayanti",
    "2023-12-25": "Christmas",
}


# ============================================================================
# DATA VALIDATION
# ============================================================================
def validate_data(df, report):
    """Validate loaded data against spec-stated statistics."""
    report["validation"] = {}

    # Row count
    actual_rows = len(df)
    report["validation"]["row_count"] = {
        "spec": SPEC_ROW_COUNT,
        "actual": actual_rows,
        "match": actual_rows == SPEC_ROW_COUNT
    }
    if actual_rows != SPEC_ROW_COUNT:
        print(f"⚠️  DISCREPANCY: Spec says {SPEC_ROW_COUNT} rows, actual file has {actual_rows} rows.")
    else:
        print(f"✅ Row count verified: {actual_rows}")

    # Column count and names
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
        "expected_count": len(expected_cols),
        "actual_count": len(actual_cols),
        "missing_from_file": list(missing_cols),
        "extra_in_file": list(extra_cols)
    }
    if missing_cols:
        print(f"⚠️  DISCREPANCY: Missing columns: {missing_cols}")
    if extra_cols:
        print(f"⚠️  DISCREPANCY: Extra columns: {extra_cols}")
    if not missing_cols and not extra_cols:
        print(f"✅ All {len(expected_cols)} expected columns present")

    # Key categorical distributions
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
                    print(f"⚠️  DISCREPANCY: {col}='{val}' — spec says {expected_count}, actual is {actual_count}")

    # Missing value check for key columns
    key_missing_checks = {
        "event_type": 0, "latitude": 0, "longitude": 0, "event_cause": 0,
        "requires_road_closure": 0, "start_datetime": 0, "status": 0,
        "police_station": 0, "corridor": 0.24, "priority": 0.02
    }
    report["validation"]["missing_percentages"] = {}
    for col, spec_pct in key_missing_checks.items():
        if col in df.columns:
            # Handle various forms of "missing": NaN, empty string, "NULL" string
            null_mask = df[col].isna()
            if df[col].dtype == object:
                null_mask = null_mask | (df[col].str.strip() == "") | (df[col].str.upper() == "NULL")
                # For priority, only NaN counts as missing (not "NULL" string which doesn't appear)
                if col == "priority":
                    null_mask = df[col].isna() | (df[col].str.strip() == "")
            actual_pct = round(null_mask.sum() / len(df) * 100, 2)
            report["validation"]["missing_percentages"][col] = {
                "spec_pct": spec_pct,
                "actual_pct": actual_pct
            }

    return report


# ============================================================================
# DATA CLEANING
# ============================================================================
def clean_data(df):
    """Apply all cleaning steps per spec Sections 2-3."""
    print("\n--- STEP 2: Dropping columns ---")

    # Only drop columns that actually exist in the dataframe
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    cols_not_found = [c for c in COLUMNS_TO_DROP if c not in df.columns]
    if cols_not_found:
        print(f"  Note: These columns were already absent: {cols_not_found}")

    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} columns. Remaining: {len(df.columns)} columns")
    print(f"  Remaining columns: {list(df.columns)}")

    print("\n--- STEP 3: Cleaning categoricals ---")

    # Parse start_datetime early (needed for later steps)
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True, format="mixed")

    # Parse closed_datetime and resolved_datetime for target derivation
    for col in ["closed_datetime", "resolved_datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, format="mixed", errors="coerce")

    # event_cause: collapse negligible categories into "Other_Cause"
    df["event_cause"] = df["event_cause"].replace(
        {cause: "Other_Cause" for cause in NEGLIGIBLE_CAUSES}
    )
    print(f"  event_cause: collapsed {NEGLIGIBLE_CAUSES} → 'Other_Cause'")
    print(f"  event_cause distribution:\n{df['event_cause'].value_counts().to_string()}\n")

    # veh_type: merge blank/empty + literal "NULL" → "Unknown_VehType"
    df["veh_type"] = df["veh_type"].fillna("Unknown_VehType")
    df["veh_type"] = df["veh_type"].replace({
        "": "Unknown_VehType",
        "NULL": "Unknown_VehType",
    })
    # Also handle whitespace-only
    df.loc[df["veh_type"].str.strip() == "", "veh_type"] = "Unknown_VehType"
    print(f"  veh_type distribution:\n{df['veh_type'].value_counts().to_string()}\n")

    # corridor: fill missing → "Non-corridor"
    df["corridor"] = df["corridor"].fillna("Non-corridor")
    df.loc[df["corridor"].str.strip() == "", "corridor"] = "Non-corridor"

    # zone: fill missing → "Zone_Unknown" + add zone_known flag
    df["zone_known"] = df["zone"].notna() & (df["zone"].str.strip() != "")
    df["zone"] = df["zone"].fillna("Zone_Unknown")
    df.loc[df["zone"].str.strip() == "", "zone"] = "Zone_Unknown"
    zone_known_pct = df["zone_known"].sum() / len(df) * 100
    print(f"  zone: {df['zone_known'].sum()} known ({zone_known_pct:.1f}%), rest → 'Zone_Unknown'")

    # junction: bucket to top 15 + "Other_Junction" + "Junction_Unknown" + flag
    df["junction_known"] = df["junction"].notna() & (df["junction"].str.strip() != "")
    df["junction"] = df["junction"].fillna("Junction_Unknown")
    df.loc[df["junction"].str.strip() == "", "junction"] = "Junction_Unknown"

    # Find top 15 most frequent junctions (excluding "Junction_Unknown")
    junction_counts = df[df["junction"] != "Junction_Unknown"]["junction"].value_counts()
    top_junctions = junction_counts.head(TOP_N_JUNCTIONS).index.tolist()
    df.loc[
        (df["junction"] != "Junction_Unknown") & (~df["junction"].isin(top_junctions)),
        "junction"
    ] = "Other_Junction"
    junction_known_pct = df["junction_known"].sum() / len(df) * 100
    print(f"  junction: {df['junction_known'].sum()} known ({junction_known_pct:.1f}%), "
          f"top {TOP_N_JUNCTIONS} kept, rest → 'Other_Junction'/'Junction_Unknown'")
    print(f"  Top junctions: {top_junctions}")

    # status: keep as-is (closed/active/resolved)
    print(f"  status distribution:\n{df['status'].value_counts().to_string()}\n")

    # requires_road_closure: ensure boolean
    df["requires_road_closure"] = df["requires_road_closure"].map(
        {True: True, False: False, "TRUE": True, "FALSE": False,
         "True": True, "False": False, "true": True, "false": False,
         1: True, 0: False}
    )
    print(f"  requires_road_closure distribution:\n{df['requires_road_closure'].value_counts().to_string()}\n")

    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def engineer_features(df):
    """Create all features per spec Section 5. No future leakage."""
    print("\n--- STEP 4: Temporal features ---")

    # Core temporal features from start_datetime
    df["hour_of_day"] = df["start_datetime"].dt.hour
    df["day_of_week"] = df["start_datetime"].dt.dayofweek  # Monday=0, Sunday=6
    df["month"] = df["start_datetime"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Public holiday flag
    date_min = df["start_datetime"].min()
    date_max = df["start_datetime"].max()
    print(f"  Date range: {date_min} to {date_max}")

    # Filter holidays to only those within the dataset's date range
    holiday_dates = set()
    for date_str, name in INDIAN_HOLIDAYS_REFERENCE.items():
        dt = pd.Timestamp(date_str, tz="UTC")
        if date_min <= dt <= date_max:
            holiday_dates.add(dt.date())
    print(f"  Hardcoded {len(holiday_dates)} public holidays within data range")

    df["is_public_holiday"] = df["start_datetime"].dt.date.isin(holiday_dates).astype(int)
    print(f"  Events on public holidays: {df['is_public_holiday'].sum()}")

    # Sort by start_datetime for rolling computations (CRITICAL for no-leakage)
    df = df.sort_values("start_datetime").reset_index(drop=True)

    print("\n  Computing rolling event counts (no future leakage)...")
    # Rolling 7-day and 30-day event counts per corridor
    df["corridor_rolling_7d"] = _compute_rolling_count(df, "corridor", days=7)
    df["corridor_rolling_30d"] = _compute_rolling_count(df, "corridor", days=30)

    # Rolling 7-day and 30-day event counts per zone
    df["zone_rolling_7d"] = _compute_rolling_count(df, "zone", days=7)
    df["zone_rolling_30d"] = _compute_rolling_count(df, "zone", days=30)

    print("\n--- STEP 5: Spatial features ---")

    # H3 cell (resolution 7) from lat/long
    if H3_AVAILABLE:
        df["h3_cell"] = df.apply(
            lambda row: h3.latlng_to_cell(row["latitude"], row["longitude"], 7)
            if pd.notna(row["latitude"]) and pd.notna(row["longitude"])
            else "unknown",
            axis=1
        )
        print(f"  H3 cells (res 7): {df['h3_cell'].nunique()} unique cells")
    else:
        # Fallback: simple lat/long rounding as spatial bucket
        df["h3_cell"] = (
            df["latitude"].round(3).astype(str) + "_" +
            df["longitude"].round(3).astype(str)
        )
        print(f"  H3 not available — using rounded lat/long buckets: {df['h3_cell'].nunique()} unique")

    # Police-station historical load: rolling 30-day count per police_station
    df["police_station_rolling_30d"] = _compute_rolling_count(df, "police_station", days=30)

    return df


def _compute_rolling_count(df, group_col, days):
    """
    For each row, count how many events in the same group started
    strictly BEFORE this event within the trailing `days` window.
    No future leakage — only past events are counted.

    Uses vectorized pandas operations instead of nested Python loops
    for performance on 8K+ rows.
    """
    result = pd.Series(0, index=df.index, dtype=int)

    for group_val, group_df in df.groupby(group_col):
        if len(group_df) == 0:
            continue

        # Sort by timestamp within group (should already be sorted globally)
        group_sorted = group_df.sort_values("start_datetime")
        timestamps = group_sorted["start_datetime"].values
        indices = group_sorted.index.values
        n = len(timestamps)

        # Use binary search (searchsorted) for O(n log n) instead of O(n²)
        window_delta = np.timedelta64(days, "D")
        window_starts = timestamps - window_delta

        # For each event i, count events in [window_start_i, timestamp_i)
        # = (position of i) - (position of first event >= window_start_i)
        counts = np.zeros(n, dtype=int)
        for i in range(n):
            # Number of events before position i that are within the window
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

    # Target 1: priority_target — binary High/Low from priority column
    # Drop 2 null rows from this target only
    priority_null_mask = df["priority"].isna() | (df["priority"].str.strip() == "")
    priority_null_count = priority_null_mask.sum()
    print(f"  priority_target: {priority_null_count} null rows (will be excluded for this target)")
    df["priority_target"] = df["priority"].copy()
    df.loc[priority_null_mask, "priority_target"] = np.nan

    # Target 2: closure_target — direct from requires_road_closure
    df["closure_target"] = df["requires_road_closure"].astype(int)
    closure_dist = df["closure_target"].value_counts()
    print(f"  closure_target distribution:\n{closure_dist.to_string()}")

    # Target 3: clearance_time_minutes — COALESCE(closed_datetime, resolved_datetime) - start_datetime
    # ONLY for status = closed or resolved, with non-null timestamp
    df["clearance_timestamp"] = df["closed_datetime"].fillna(df["resolved_datetime"])
    df["clearance_time_minutes"] = np.nan  # default: not eligible

    eligible_mask = (
        df["status"].isin(["closed", "resolved"]) &
        df["clearance_timestamp"].notna()
    )

    # EXPLICITLY EXCLUDED: active-status rows (right-censored data)
    # These represent ongoing/unresolved events and must NOT be included in training,
    # imputed with a placeholder duration, or dropped silently without disclosure.
    active_count = (df["status"] == "active").sum()
    print(f"  Clearance-time: {active_count} active-status rows EXCLUDED (right-censored)")

    df.loc[eligible_mask, "clearance_time_minutes"] = (
        (df.loc[eligible_mask, "clearance_timestamp"] - df.loc[eligible_mask, "start_datetime"])
        .dt.total_seconds() / 60.0
    )

    eligible_count = eligible_mask.sum()
    print(f"  Clearance-time eligible rows: {eligible_count}")

    # Remove negative or zero clearance times (data quality issue if any)
    bad_clearance = (df["clearance_time_minutes"] <= 0) & eligible_mask
    if bad_clearance.sum() > 0:
        print(f"  ⚠️  {bad_clearance.sum()} rows have non-positive clearance time — setting to NaN")
        df.loc[bad_clearance, "clearance_time_minutes"] = np.nan

    # Remove extreme outliers (>10,000 minutes = ~7 days — likely data errors)
    extreme_outlier = (df["clearance_time_minutes"] > 10000) & df["clearance_time_minutes"].notna()
    if extreme_outlier.sum() > 0:
        print(f"  ⚠️  {extreme_outlier.sum()} rows have clearance time >10,000 min — capping at 10,000")
        df.loc[extreme_outlier, "clearance_time_minutes"] = 10000

    valid_clearance = df["clearance_time_minutes"].notna()
    print(f"  Final clearance-time valid rows: {valid_clearance.sum()}")
    print(f"  Clearance time stats (minutes):\n{df.loc[valid_clearance, 'clearance_time_minutes'].describe().to_string()}")

    # Drop intermediate column
    df = df.drop(columns=["clearance_timestamp"])

    return df


# ============================================================================
# PREPARE MODEL-READY DATASETS
# ============================================================================
def prepare_model_datasets(df):
    """
    Prepare final feature matrices for each model target.
    Returns the full dataframe with all features and targets.
    """
    print("\n--- STEP 7: Preparing model-ready datasets ---")

    # Define feature columns for classification models (A and B)
    classification_features = [
        "event_type", "event_cause", "corridor", "zone", "junction",
        "police_station", "veh_type", "h3_cell",
        "latitude", "longitude",
        "hour_of_day", "day_of_week", "month", "is_weekend", "is_public_holiday",
        "corridor_rolling_7d", "corridor_rolling_30d",
        "zone_rolling_7d", "zone_rolling_30d",
        "police_station_rolling_30d",
        "zone_known", "junction_known",
        "status",  # Used as feature for classifiers only, per spec
    ]

    # For clearance-time model, status is NOT a feature (it's the eligibility filter)
    regression_features = [
        "event_type", "event_cause", "corridor", "zone", "junction",
        "police_station", "veh_type", "h3_cell",
        "latitude", "longitude",
        "hour_of_day", "day_of_week", "month", "is_weekend", "is_public_holiday",
        "corridor_rolling_7d", "corridor_rolling_30d",
        "zone_rolling_7d", "zone_rolling_30d",
        "police_station_rolling_30d",
        "zone_known", "junction_known",
    ]

    # Only keep features that exist in the dataframe
    classification_features = [f for f in classification_features if f in df.columns]
    regression_features = [f for f in regression_features if f in df.columns]

    print(f"  Classification features ({len(classification_features)}): {classification_features}")
    print(f"  Regression features ({len(regression_features)}): {regression_features}")

    return df, classification_features, regression_features


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_pipeline():
    """Execute the full data pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    report = {"pipeline_run_timestamp": datetime.now().isoformat()}

    # STEP 1: Load and validate
    print("=" * 70)
    print("STEP 1: Loading and validating data")
    print("=" * 70)

    if not os.path.exists(DATA_FILE):
        print(f"❌ ERROR: Data file not found at: {DATA_FILE}")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE, low_memory=False)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {os.path.basename(DATA_FILE)}")

    report = validate_data(df, report)

    # STEP 2-3: Clean
    print("\n" + "=" * 70)
    print("STEP 2-3: Cleaning data")
    print("=" * 70)
    df = clean_data(df)

    # STEP 4-5: Feature engineering
    print("\n" + "=" * 70)
    print("STEP 4-5: Feature engineering")
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
    # VERIFY: no active rows in clearance subset
    active_in_clearance = (clearance_subset["status"] == "active").sum()
    assert active_in_clearance == 0, \
        f"CRITICAL ERROR: {active_in_clearance} active-status rows found in clearance-time subset!"
    print(f"  ✅ Verified: 0 active-status rows in clearance-time subset")

    clearance_output_path = os.path.join(OUTPUT_DIR, "features_clearance_subset.csv")
    clearance_subset.to_csv(clearance_output_path, index=False)
    print(f"  Saved clearance subset: {clearance_output_path} ({len(clearance_subset)} rows)")

    # Save feature lists
    feature_config = {
        "classification_features": clf_features,
        "regression_features": reg_features
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
        "active_rows_excluded_from_clearance": int((df["status"] == "active").sum()),
    }

    report_path = os.path.join(REPORTS_DIR, "data_pipeline_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved pipeline report: {report_path}")

    print("\n" + "=" * 70)
    print("✅ DATA PIPELINE COMPLETE")
    print("=" * 70)

    return df, clf_features, reg_features


if __name__ == "__main__":
    run_pipeline()
