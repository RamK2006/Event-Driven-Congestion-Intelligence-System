"""
Flask API Server — Event Impact & Response Intelligence Platform
================================================================
Serves:
  - The Event Command Center dashboard (static HTML/CSS/JS)
  - REST API endpoints for model inference, hotspot data, diversion lookup,
    historical analytics
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

app = Flask(__name__, static_folder=DASHBOARD_DIR)

# CORS: allow all origins in dev, or restrict via CORS_ORIGINS env var in production
cors_origins = os.environ.get("CORS_ORIGINS", "*")
CORS(app, origins=cors_origins.split(","))

# ============================================================================
# LOAD MODELS AND DATA AT STARTUP
# ============================================================================
def load_assets():
    """Load all models, encoders, and data at startup."""
    assets = {}

    # Load models
    try:
        assets["model_a"] = joblib.load(os.path.join(MODELS_DIR, "model_a_priority.pkl"))
        assets["model_a_encoders"] = joblib.load(os.path.join(MODELS_DIR, "model_a_encoders.pkl"))
        assets["model_a_features"] = joblib.load(os.path.join(MODELS_DIR, "model_a_features.pkl"))

        assets["model_b"] = joblib.load(os.path.join(MODELS_DIR, "model_b_closure.pkl"))
        assets["model_b_encoders"] = joblib.load(os.path.join(MODELS_DIR, "model_b_encoders.pkl"))
        assets["model_b_features"] = joblib.load(os.path.join(MODELS_DIR, "model_b_features.pkl"))

        assets["model_c"] = joblib.load(os.path.join(MODELS_DIR, "model_c_clearance.pkl"))
        assets["model_c_encoders"] = joblib.load(os.path.join(MODELS_DIR, "model_c_encoders.pkl"))
        assets["model_c_features"] = joblib.load(os.path.join(MODELS_DIR, "model_c_features.pkl"))
        print("[OK] Models loaded successfully")
    except Exception as e:
        print(f"[WARN] Error loading models: {e}")
        print("   Dashboard will work but predictions will be unavailable")

    # Load data
    try:
        assets["df"] = pd.read_csv(os.path.join(DATA_DIR, "features_full.csv"), low_memory=False)
        assets["df"]["start_datetime"] = pd.to_datetime(assets["df"]["start_datetime"], utc=True, format="mixed")
        print(f"[OK] Data loaded: {len(assets['df'])} rows")
    except Exception as e:
        print(f"[WARN] Error loading data: {e}")

    # Load hotspot clusters
    try:
        with open(os.path.join(OUTPUTS_DIR, "hotspot_clusters.json"), "r") as f:
            assets["hotspots"] = json.load(f)
        print(f"[OK] Hotspots loaded: {assets['hotspots']['total_clusters']} clusters")
    except Exception as e:
        print(f"[WARN] Error loading hotspots: {e}")
        assets["hotspots"] = {"hotspots": [], "total_clusters": 0}

    # Load diversion table
    try:
        with open(os.path.join(OUTPUTS_DIR, "diversion_lookup_table.json"), "r") as f:
            assets["diversions"] = json.load(f)
        print(f"[OK] Diversion table loaded")
    except Exception as e:
        print(f"[WARN] Error loading diversions: {e}")
        assets["diversions"] = {"corridors": {}}

    # Load model evaluation report
    try:
        with open(os.path.join(OUTPUTS_DIR, "model_evaluation_report.json"), "r") as f:
            assets["eval_report"] = json.load(f)
        print(f"[OK] Evaluation report loaded")
    except Exception as e:
        print(f"[WARN] Error loading eval report: {e}")
        assets["eval_report"] = {}

    return assets


# Global assets dict — loaded at module level for gunicorn workers
ASSETS = load_assets()


# ============================================================================
# HEALTH CHECK (for Render / deployment monitoring)
# ============================================================================
@app.route("/api/health")
def health_check():
    """Health check endpoint for deployment platform monitoring."""
    models_loaded = all(k in ASSETS for k in ["model_a", "model_b", "model_c"])
    data_loaded = "df" in ASSETS
    return jsonify({
        "status": "healthy",
        "models_loaded": models_loaded,
        "data_loaded": data_loaded,
        "total_events": len(ASSETS["df"]) if data_loaded else 0,
    })


# ============================================================================
# DASHBOARD ROUTES
# ============================================================================
@app.route("/")
def serve_dashboard():
    return send_from_directory(DASHBOARD_DIR, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(DASHBOARD_DIR, filename)


# ============================================================================
# API: Historical Events Data
# ============================================================================
@app.route("/api/events")
def get_events():
    """Return all historical events for map display."""
    df = ASSETS.get("df")
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500

    # Return essential fields only for performance
    cols = ["latitude", "longitude", "event_type", "event_cause", "corridor",
            "police_station", "zone", "priority_target", "status",
            "closure_target", "hour_of_day", "day_of_week", "month",
            "start_datetime", "clearance_time_minutes", "veh_type"]
    available_cols = [c for c in cols if c in df.columns]

    events = df[available_cols].copy()
    events["start_datetime"] = events["start_datetime"].astype(str)
    events = events.fillna("N/A")

    return jsonify(events.to_dict(orient="records"))


@app.route("/api/events/summary")
def get_events_summary():
    """Return aggregated summary statistics."""
    df = ASSETS.get("df")
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500

    summary = {
        "total_events": int(len(df)),
        "event_type_dist": df["event_type"].value_counts().to_dict(),
        "priority_dist": df["priority_target"].value_counts().to_dict() if "priority_target" in df.columns else {},
        "status_dist": df["status"].value_counts().to_dict(),
        "event_cause_dist": df["event_cause"].value_counts().to_dict(),
        "corridor_dist": df["corridor"].value_counts().to_dict(),
        "date_range": {
            "min": str(df["start_datetime"].min()),
            "max": str(df["start_datetime"].max()),
        }
    }
    return jsonify(summary)


# ============================================================================
# API: Police Station Workload
# ============================================================================
@app.route("/api/workload")
def get_workload():
    """Return police station workload data, filterable by time window."""
    df = ASSETS.get("df")
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500

    # Optional time window filter
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")

    filtered = df.copy()
    if start_date:
        filtered = filtered[filtered["start_datetime"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        filtered = filtered[filtered["start_datetime"] <= pd.Timestamp(end_date, tz="UTC")]

    workload = filtered.groupby("police_station").agg(
        total_events=("police_station", "count"),
        high_priority=("priority_target", lambda x: (x == "High").sum()),
        low_priority=("priority_target", lambda x: (x == "Low").sum()),
        road_closures=("closure_target", "sum"),
    ).reset_index()

    workload = workload.sort_values("total_events", ascending=False)
    return jsonify(workload.to_dict(orient="records"))


# ============================================================================
# API: Historical Trends
# ============================================================================
@app.route("/api/trends")
def get_trends():
    """Return event counts over time, filterable by cause/corridor/zone."""
    df = ASSETS.get("df")
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500

    # Filters
    cause = request.args.get("cause")
    corridor = request.args.get("corridor")
    zone = request.args.get("zone")

    filtered = df.copy()
    if cause and cause != "all":
        filtered = filtered[filtered["event_cause"] == cause]
    if corridor and corridor != "all":
        filtered = filtered[filtered["corridor"] == corridor]
    if zone and zone != "all":
        filtered = filtered[filtered["zone"] == zone]

    # Group by month
    filtered["year_month"] = filtered["start_datetime"].dt.to_period("M").astype(str)
    trends = filtered.groupby("year_month").size().reset_index(name="count")
    trends = trends.sort_values("year_month")

    # Also group by day of week and hour
    hourly = filtered.groupby("hour_of_day").size().reset_index(name="count")
    daily = filtered.groupby("day_of_week").size().reset_index(name="count")

    return jsonify({
        "monthly": trends.to_dict(orient="records"),
        "hourly": hourly.to_dict(orient="records"),
        "daily": daily.to_dict(orient="records"),
    })


# ============================================================================
# API: Hotspot Clusters
# ============================================================================
@app.route("/api/hotspots")
def get_hotspots():
    """Return hotspot cluster data for map overlay."""
    return jsonify(ASSETS.get("hotspots", {"hotspots": []}))


# ============================================================================
# API: Diversion Suggestions
# ============================================================================
@app.route("/api/diversions")
def get_diversions():
    """Return diversion lookup table."""
    return jsonify(ASSETS.get("diversions", {"corridors": {}}))


@app.route("/api/diversions/<corridor>")
def get_diversion_for_corridor(corridor):
    """Return diversion suggestions for a specific corridor."""
    diversions = ASSETS.get("diversions", {"corridors": {}})
    corridor_data = diversions.get("corridors", {}).get(corridor)
    if corridor_data:
        return jsonify(corridor_data)
    return jsonify({"error": f"No diversion data for corridor: {corridor}"}), 404


# ============================================================================
# API: Model Evaluation Report
# ============================================================================
@app.route("/api/evaluation")
def get_evaluation():
    """Return model evaluation metrics."""
    return jsonify(ASSETS.get("eval_report", {}))


# ============================================================================
# API: Filter Options
# ============================================================================
@app.route("/api/filters")
def get_filters():
    """Return available filter options for the dashboard."""
    df = ASSETS.get("df")
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500

    return jsonify({
        "causes": sorted(df["event_cause"].unique().tolist()),
        "corridors": sorted(df["corridor"].unique().tolist()),
        "zones": sorted(df["zone"].unique().tolist()) if "zone" in df.columns else [],
        "police_stations": sorted(df["police_station"].unique().tolist()),
        "veh_types": sorted(df["veh_type"].unique().tolist()) if "veh_type" in df.columns else [],
    })


# ============================================================================
# API: Predict (New Incident)
# ============================================================================
@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accept a new incident report and return predictions from all 3 models.
    Expected JSON body: {latitude, longitude, event_cause, veh_type, datetime_str}
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        lat = float(data.get("latitude", 0))
        lon = float(data.get("longitude", 0))
        event_cause = data.get("event_cause", "vehicle_breakdown")
        veh_type = data.get("veh_type", "Unknown_VehType")
        datetime_str = data.get("datetime_str", datetime.utcnow().isoformat())
        event_type = data.get("event_type", "unplanned")

        # Parse datetime
        dt = pd.Timestamp(datetime_str, tz="UTC")

        # Find nearest corridor, zone, police_station from historical data
        df = ASSETS.get("df")
        if df is not None:
            distances = np.sqrt(
                (df["latitude"] - lat) ** 2 + (df["longitude"] - lon) ** 2
            )
            nearest_idx = distances.idxmin()
            corridor = df.loc[nearest_idx, "corridor"]
            police_station = df.loc[nearest_idx, "police_station"]
            zone = df.loc[nearest_idx, "zone"] if "zone" in df.columns else "Zone_Unknown"
            junction = df.loc[nearest_idx, "junction"] if "junction" in df.columns else "Junction_Unknown"
        else:
            corridor = "Non-corridor"
            police_station = "unknown"
            zone = "Zone_Unknown"
            junction = "Junction_Unknown"

        # Build feature vector
        try:
            import h3 as h3lib
            h3_cell = h3lib.latlng_to_cell(lat, lon, 7)
        except Exception:
            h3_cell = f"{round(lat, 3)}_{round(lon, 3)}"

        feature_dict = {
            "event_type": event_type,
            "event_cause": event_cause,
            "corridor": corridor,
            "zone": zone,
            "junction": junction,
            "police_station": police_station,
            "veh_type": veh_type if veh_type else "Unknown_VehType",
            "h3_cell": h3_cell,
            "latitude": lat,
            "longitude": lon,
            "hour_of_day": dt.hour,
            "day_of_week": dt.dayofweek,
            "month": dt.month,
            "is_weekend": 1 if dt.dayofweek >= 5 else 0,
            "is_public_holiday": 0,  # Simplified for inference
            "corridor_rolling_7d": 0,
            "corridor_rolling_30d": 0,
            "zone_rolling_7d": 0,
            "zone_rolling_30d": 0,
            "police_station_rolling_30d": 0,
            "zone_known": 1 if zone != "Zone_Unknown" else 0,
            "junction_known": 1 if junction != "Junction_Unknown" else 0,
            "status": "active",  # New incident is active
        }

        results = {
            "input": {
                "latitude": lat,
                "longitude": lon,
                "event_cause": event_cause,
                "veh_type": veh_type,
                "datetime": str(dt),
                "inferred_corridor": corridor,
                "inferred_police_station": police_station,
                "inferred_zone": zone,
            }
        }

        # Model A: Priority prediction
        try:
            model_a = ASSETS["model_a"]
            encoders_a = ASSETS["model_a_encoders"]
            features_a = ASSETS["model_a_features"]

            X_a = _encode_features(feature_dict, features_a, encoders_a)
            pred_a = model_a.predict(X_a)[0]
            prob_a = model_a.predict_proba(X_a)[0]

            target_le = encoders_a.get("__target__")
            priority_label = target_le.inverse_transform([pred_a])[0] if target_le else str(pred_a)

            results["priority"] = {
                "prediction": priority_label,
                "confidence": round(float(max(prob_a)), 4),
                "probabilities": {
                    target_le.classes_[i]: round(float(p), 4)
                    for i, p in enumerate(prob_a)
                } if target_le else {}
            }
        except Exception as e:
            results["priority"] = {"error": str(e)}

        # Model B: Closure prediction
        try:
            model_b = ASSETS["model_b"]
            encoders_b = ASSETS["model_b_encoders"]
            features_b = ASSETS["model_b_features"]

            X_b = _encode_features(feature_dict, features_b, encoders_b)
            pred_b = model_b.predict(X_b)[0]
            prob_b = model_b.predict_proba(X_b)[0]

            target_le_b = encoders_b.get("__target__")
            closure_label = target_le_b.inverse_transform([pred_b])[0] if target_le_b else str(pred_b)

            results["closure"] = {
                "prediction": "Yes" if str(closure_label) == "1" else "No",
                "confidence": round(float(max(prob_b)), 4),
                "probability": round(float(prob_b[1]) if len(prob_b) > 1 else float(prob_b[0]), 4),
            }
        except Exception as e:
            results["closure"] = {"error": str(e)}

        # Model C: Clearance time prediction
        try:
            model_c = ASSETS["model_c"]
            encoders_c = ASSETS["model_c_encoders"]
            features_c = ASSETS["model_c_features"]

            # Remove 'status' from features for regression model per spec
            feature_dict_c = {k: v for k, v in feature_dict.items() if k != "status"}
            X_c = _encode_features(feature_dict_c, features_c, encoders_c)
            pred_c = model_c.predict(X_c)[0]

            results["clearance_time"] = {
                "predicted_minutes": round(float(max(pred_c, 0)), 1),
                "predicted_hours": round(float(max(pred_c, 0)) / 60, 2),
            }
        except Exception as e:
            results["clearance_time"] = {"error": str(e)}

        # Nearest hotspot context
        try:
            hotspots = ASSETS.get("hotspots", {}).get("hotspots", [])
            if hotspots:
                min_dist = float("inf")
                nearest_hotspot = None
                for h in hotspots:
                    dist = np.sqrt(
                        (h["center_latitude"] - lat) ** 2 +
                        (h["center_longitude"] - lon) ** 2
                    )
                    if dist < min_dist:
                        min_dist = dist
                        nearest_hotspot = h
                results["nearest_hotspot"] = nearest_hotspot
            else:
                results["nearest_hotspot"] = None
        except Exception as e:
            results["nearest_hotspot"] = {"error": str(e)}

        # Diversion suggestion
        try:
            diversions = ASSETS.get("diversions", {}).get("corridors", {})
            corridor_diversions = diversions.get(corridor, {})
            suggestions = corridor_diversions.get("suggested_diversions", [])
            results["diversion"] = {
                "corridor": corridor,
                "suggestions": suggestions[:MAX_DIVERSIONS] if suggestions else [],
                "disclosure": (
                    "Heuristic Diversion Suggestion (historical co-occurrence based). "
                    "This is NOT a routing algorithm or shortest-path calculation."
                ),
            }
        except Exception as e:
            results["diversion"] = {"error": str(e)}

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


MAX_DIVERSIONS = 3


def _encode_features(feature_dict, feature_list, encoders):
    """Encode a single feature vector using the trained label encoders."""
    X = pd.DataFrame([feature_dict])

    # Only keep the features the model expects
    available = [f for f in feature_list if f in X.columns]
    missing = [f for f in feature_list if f not in X.columns]
    for col in missing:
        X[col] = 0

    X = X[feature_list]

    # Apply label encoding
    for col in X.select_dtypes(include=["object", "bool"]).columns:
        le = encoders.get(col)
        if le is not None:
            val = str(X[col].iloc[0])
            if val in le.classes_:
                X[col] = le.transform([val])[0]
            else:
                # Unknown category — use the most common class index
                X[col] = 0
        else:
            X[col] = 0

    # Convert booleans
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)

    X = X.fillna(0)
    return X


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print("=" * 60)
    print("Event Impact & Response Intelligence Platform")
    print("Event Command Center - Starting Server")
    print("=" * 60)
    print(f"\nServer running at http://localhost:{port}")
    print("=" * 60)

    app.run(host="0.0.0.0", port=port, debug=False)
