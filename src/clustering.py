"""
Hotspot Clustering — Event Impact & Response Intelligence Platform
==================================================================
HDBSCAN on lat/long pairs to identify genuine geographic incident hotspots.
Output: ranked cluster list with center coords, event count, dominant cause,
dominant priority, associated police_station/zone.
Per spec Section 7 — unsupervised, no external validation.
"""

import os
import json
import pandas as pd
import numpy as np

try:
    from hdbscan import HDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    from sklearn.cluster import DBSCAN
    HDBSCAN_AVAILABLE = False
    print("⚠️  hdbscan not available, falling back to sklearn.cluster.DBSCAN")

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# HDBSCAN parameters
MIN_CLUSTER_SIZE = 30  # Minimum cluster size for hotspot
MIN_SAMPLES = 10


# ============================================================================
# CLUSTERING
# ============================================================================
def run_clustering():
    """Run HDBSCAN/DBSCAN on lat/long to find hotspots."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # Load data
    df = pd.read_csv(os.path.join(DATA_DIR, "features_full.csv"), low_memory=False)
    print(f"Loaded {len(df)} rows for clustering")

    # Prepare coordinates
    coords = df[["latitude", "longitude"]].dropna().values
    valid_mask = df[["latitude", "longitude"]].notna().all(axis=1)
    df_valid = df[valid_mask].copy()
    print(f"Valid coordinate rows: {len(df_valid)}")

    # Run clustering
    if HDBSCAN_AVAILABLE:
        print("\nRunning HDBSCAN...")
        clusterer = HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES,
            metric="haversine",
        )
        # Convert to radians for haversine
        coords_rad = np.radians(df_valid[["latitude", "longitude"]].values)
        labels = clusterer.fit_predict(coords_rad)
        algo_name = "HDBSCAN"
    else:
        print("\nRunning DBSCAN (fallback)...")
        from sklearn.cluster import DBSCAN
        clusterer = DBSCAN(
            eps=0.005,  # ~500m in lat/long degrees
            min_samples=MIN_SAMPLES,
            metric="haversine",
            algorithm="ball_tree",
        )
        coords_rad = np.radians(df_valid[["latitude", "longitude"]].values)
        labels = clusterer.fit_predict(coords_rad)
        algo_name = "DBSCAN"

    df_valid["cluster"] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"\nAlgorithm: {algo_name}")
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")

    # Analyze each cluster
    hotspots = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue  # Skip noise

        cluster_mask = df_valid["cluster"] == cluster_id
        cluster_df = df_valid[cluster_mask]

        center_lat = cluster_df["latitude"].mean()
        center_lon = cluster_df["longitude"].mean()
        event_count = len(cluster_df)

        # Dominant event_cause
        dominant_cause = cluster_df["event_cause"].mode().iloc[0] if "event_cause" in cluster_df.columns else "unknown"
        cause_pct = (cluster_df["event_cause"] == dominant_cause).sum() / event_count * 100 if "event_cause" in cluster_df.columns else 0

        # Dominant priority
        dominant_priority = "unknown"
        priority_pct = 0
        if "priority_target" in cluster_df.columns:
            priority_series = cluster_df["priority_target"].dropna()
            if len(priority_series) > 0:
                dominant_priority = priority_series.mode().iloc[0]
                priority_pct = (priority_series == dominant_priority).sum() / len(priority_series) * 100

        # Associated police_station
        dominant_station = cluster_df["police_station"].mode().iloc[0] if "police_station" in cluster_df.columns else "unknown"

        # Associated zone
        dominant_zone = "Zone_Unknown"
        if "zone" in cluster_df.columns:
            zone_valid = cluster_df[cluster_df["zone"] != "Zone_Unknown"]["zone"]
            if len(zone_valid) > 0:
                dominant_zone = zone_valid.mode().iloc[0]

        # Associated corridor
        dominant_corridor = "Non-corridor"
        if "corridor" in cluster_df.columns:
            corr_valid = cluster_df[cluster_df["corridor"] != "Non-corridor"]["corridor"]
            if len(corr_valid) > 0:
                dominant_corridor = corr_valid.mode().iloc[0]

        hotspot = {
            "cluster_id": int(cluster_id),
            "center_latitude": round(center_lat, 6),
            "center_longitude": round(center_lon, 6),
            "event_count": int(event_count),
            "dominant_event_cause": dominant_cause,
            "dominant_cause_pct": round(cause_pct, 1),
            "dominant_priority": dominant_priority,
            "dominant_priority_pct": round(priority_pct, 1),
            "associated_police_station": dominant_station,
            "associated_zone": dominant_zone,
            "associated_corridor": dominant_corridor,
        }
        hotspots.append(hotspot)

    # Sort by event count (descending) — ranked list per spec
    hotspots.sort(key=lambda x: x["event_count"], reverse=True)

    # Add rank
    for i, h in enumerate(hotspots):
        h["rank"] = i + 1

    # Save output
    output_path = os.path.join(OUTPUTS_DIR, "hotspot_clusters.json")
    with open(output_path, "w") as f:
        json.dump({
            "algorithm": algo_name,
            "parameters": {
                "min_cluster_size": MIN_CLUSTER_SIZE,
                "min_samples": MIN_SAMPLES,
            },
            "total_clusters": n_clusters,
            "total_noise_points": int(n_noise),
            "hotspots": hotspots,
        }, f, indent=2)
    print(f"\n✅ Hotspot clusters saved to: {output_path}")

    # Print top 10
    print(f"\nTop 10 Hotspot Clusters:")
    print(f"{'Rank':<6}{'Events':<8}{'Cause':<20}{'Priority':<10}{'Station':<25}{'Corridor':<20}")
    print("-" * 90)
    for h in hotspots[:10]:
        print(f"{h['rank']:<6}{h['event_count']:<8}{h['dominant_event_cause']:<20}"
              f"{h['dominant_priority']:<10}{h['associated_police_station']:<25}"
              f"{h['associated_corridor']:<20}")

    return hotspots


if __name__ == "__main__":
    run_clustering()
