"""
Diversion Heuristic — Event Impact & Response Intelligence Platform
====================================================================
Co-occurrence-based diversion lookup table per spec Section 8.

Method: For each corridor, identify which other corridors historically had
concurrent events (overlapping start_datetime -> closed_datetime windows
within the same zone or police_station). Then recommend the 1-3 nearby
corridors/zones that historically did NOT have simultaneous incidents.

MANDATORY DISCLOSURE: This is a heuristic layer, NOT a learned model,
routing algorithm, shortest-path calculation, or anything implying real
road-network analysis. No road-network graph data exists in this dataset.
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Time window for "near-concurrent" events (in hours)
CONCURRENCY_WINDOW_HOURS = 4

# Number of diversion suggestions per corridor
MAX_DIVERSIONS = 3


# ============================================================================
# BUILD CO-OCCURRENCE TABLE
# ============================================================================
def build_diversion_table():
    """Build the co-occurrence-based diversion lookup table."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Load data
    df = pd.read_csv(os.path.join(DATA_DIR, "features_full.csv"), low_memory=False)
    print(f"Loaded {len(df)} rows")

    # Parse timestamps
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True, format="mixed")
    for col in ["closed_datetime", "resolved_datetime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, format="mixed", errors="coerce")

    # Compute end time: COALESCE(closed_datetime, resolved_datetime, start_datetime + window)
    df["event_end"] = df["closed_datetime"].fillna(
        df.get("resolved_datetime", pd.NaT)
    )
    # For events with no end time, assume they last for the concurrency window
    fallback_end = df["start_datetime"] + pd.Timedelta(hours=CONCURRENCY_WINDOW_HOURS)
    df["event_end"] = df["event_end"].fillna(fallback_end)

    # Focus on corridors
    corridors = df["corridor"].unique().tolist()
    named_corridors = [c for c in corridors if c != "Non-corridor"]
    print(f"Named corridors: {len(named_corridors)}")
    print(f"All corridors (including Non-corridor): {len(corridors)}")

    # Build co-occurrence using vectorized approach per police_station group
    print("\nBuilding co-occurrence matrix (optimized)...")
    co_occurrence = defaultdict(lambda: defaultdict(int))

    # Sort by start time
    df_sorted = df.sort_values("start_datetime").reset_index(drop=True)

    # Convert to numpy arrays for fast access
    starts = df_sorted["start_datetime"].values
    ends = df_sorted["event_end"].values
    corridors_arr = df_sorted["corridor"].values
    stations_arr = df_sorted["police_station"].values if "police_station" in df_sorted.columns else None
    zones_arr = df_sorted["zone"].values if "zone" in df_sorted.columns else None

    n = len(df_sorted)
    window_td = np.timedelta64(CONCURRENCY_WINDOW_HOURS, "h")

    # Process using numpy arrays (much faster than .loc access)
    for i in range(n):
        if i % 2000 == 0:
            print(f"  Processing event {i}/{n}...")

        corridor_i = corridors_arr[i]
        start_i = starts[i]
        end_i = ends[i]
        station_i = stations_arr[i] if stations_arr is not None else None
        zone_i = zones_arr[i] if zones_arr is not None else None

        # Look at subsequent events within the concurrency window
        for j in range(i + 1, n):
            start_j = starts[j]

            # If event j starts after event i's end + window, stop
            if start_j > end_i + window_td:
                break

            corridor_j = corridors_arr[j]

            # Skip same-corridor pairs
            if corridor_i == corridor_j:
                continue

            # Check same jurisdiction
            same_jurisdiction = False
            if zones_arr is not None:
                zone_j = zones_arr[j]
                if (zone_i and zone_j and zone_i == zone_j
                        and zone_i != "Zone_Unknown"
                        and str(zone_i) != "nan"):
                    same_jurisdiction = True
            if stations_arr is not None and not same_jurisdiction:
                station_j = stations_arr[j]
                if station_i and station_j and station_i == station_j:
                    same_jurisdiction = True

            if same_jurisdiction:
                # Check temporal overlap
                end_j = ends[j]
                if start_i <= end_j and start_j <= end_i:
                    co_occurrence[corridor_i][corridor_j] += 1
                    co_occurrence[corridor_j][corridor_i] += 1

    print(f"\nCo-occurrence pairs found: {sum(len(v) for v in co_occurrence.values()) // 2}")

    # Build diversion recommendations
    diversion_table = {}

    for corridor in corridors:
        concurrent_corridors = co_occurrence.get(corridor, {})

        corridor_events = df[df["corridor"] == corridor]
        nearby_zones = set(corridor_events["zone"].unique()) - {"Zone_Unknown"} if "zone" in df.columns else set()
        nearby_stations = set(corridor_events["police_station"].unique()) if "police_station" in df.columns else set()

        candidates = []
        for other_corridor in named_corridors:
            if other_corridor == corridor:
                continue

            other_events = df[df["corridor"] == other_corridor]
            other_zones = set(other_events["zone"].unique()) - {"Zone_Unknown"} if "zone" in df.columns else set()
            other_stations = set(other_events["police_station"].unique()) if "police_station" in df.columns else set()

            shared_zones = nearby_zones & other_zones
            shared_stations = nearby_stations & other_stations

            if shared_zones or shared_stations:
                concurrent_count = concurrent_corridors.get(other_corridor, 0)
                total_events = len(other_events)
                co_occurrence_ratio = concurrent_count / max(total_events, 1)

                candidates.append({
                    "corridor": other_corridor,
                    "concurrent_events": concurrent_count,
                    "total_events": total_events,
                    "co_occurrence_ratio": round(co_occurrence_ratio, 4),
                    "shared_zones": list(shared_zones),
                    "shared_stations": list(shared_stations),
                })

        # Sort by co-occurrence ratio ascending (prefer least concurrent)
        candidates.sort(key=lambda x: x["co_occurrence_ratio"])
        best_diversions = candidates[:MAX_DIVERSIONS]

        diversion_table[corridor] = {
            "corridor": corridor,
            "total_events": int(len(corridor_events)),
            "suggested_diversions": best_diversions,
            "concurrent_corridors_avoided": [
                {"corridor": c, "concurrent_count": cnt}
                for c, cnt in sorted(concurrent_corridors.items(), key=lambda x: x[1], reverse=True)
            ][:5],
        }

    # Save output
    output_path = os.path.join(OUTPUTS_DIR, "diversion_lookup_table.json")
    with open(output_path, "w") as f:
        json.dump({
            "method": "Co-occurrence-based heuristic (NOT a routing algorithm)",
            "disclosure": (
                "This module provides Heuristic Diversion Suggestions based on "
                "historical event co-occurrence patterns. It is NOT a routing algorithm, "
                "shortest-path calculation, or anything implying real road-network analysis. "
                "No road-network graph data exists in this dataset."
            ),
            "concurrency_window_hours": CONCURRENCY_WINDOW_HOURS,
            "max_diversions_per_corridor": MAX_DIVERSIONS,
            "corridors": diversion_table,
        }, f, indent=2)

    print(f"\n[OK] Diversion lookup table saved to: {output_path}")

    # Print summary
    print(f"\nDiversion Table Summary:")
    print(f"{'Corridor':<25}{'Events':<8}{'Diversions':<12}{'Avoided':<10}")
    print("-" * 55)
    for corridor, info in sorted(diversion_table.items(), key=lambda x: x[1]["total_events"], reverse=True):
        n_div = len(info["suggested_diversions"])
        n_avoid = len(info["concurrent_corridors_avoided"])
        print(f"{corridor:<25}{info['total_events']:<8}{n_div:<12}{n_avoid:<10}")

    return diversion_table


if __name__ == "__main__":
    build_diversion_table()
