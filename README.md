# 🛡️ Event Impact & Response Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-02569B?logo=microsoft&logoColor=white)](https://lightgbm.readthedocs.io)
[![Leaflet](https://img.shields.io/badge/Leaflet-1.9-199900?logo=leaflet&logoColor=white)](https://leafletjs.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Real-time event severity, road-closure likelihood, and clearance-time prediction for Bengaluru traffic-disrupting incidents.**

<p align="center">
  <strong>
    <a href="#live-demo">Live Demo</a> •
    <a href="#features">Features</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#api-reference">API Reference</a> •
    <a href="#local-development">Local Development</a> •
    <a href="#deployment">Deployment</a>
  </strong>
</p>

---

## 📢 Honest Reframing (Required Disclosure)

> **This system does not predict traffic congestion, vehicle counts, or delay duration in minutes of traffic**, because the dataset contains no traffic flow, speed, or volume measurements. Instead, it predicts three real, label-backed outcomes — **event severity**, **road-closure likelihood**, and **clearance time** — and uses historical spatio-temporal patterns to recommend resource deployment and diversion zones. This is a deliberate, disclosed scope decision, not a limitation discovered after the fact.

---

## Live Demo

| Component | URL |
|-----------|-----|
| 🖥️ **Frontend Dashboard** | [dashboard-tan-ten-51.vercel.app](https://dashboard-tan-ten-51.vercel.app) |
| ⚙️ **Backend API** | [event-driven-congestion-intelligence.onrender.com](https://event-driven-congestion-intelligence.onrender.com) |
| 💚 **Health Check** | [`/api/health`](https://event-driven-congestion-intelligence.onrender.com/api/health) |

---

## Features

### 🗺️ Incident Map
All historical events plotted by lat/long, color-coded by priority (High = red, Low = green), with HDBSCAN hotspot cluster overlays and marker clustering.

### 🔮 New Incident Prediction
Submit location, cause, vehicle type, and datetime → get predicted priority, closure likelihood, clearance time, and heuristic diversion suggestions in real-time.

### 👮 Police Station Workload
Interactive bar chart + ranked table of event load per station, filterable by time window. Identifies overburdened stations for resource rebalancing.

### 📊 Historical Trends
Event counts over time with monthly, hourly, and day-of-week views. Filterable by cause, corridor, and zone.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Data Layer                        │
│  Cleaned CSV → Feature Engineering → Target Derivation│
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────┐
│                  Modeling Layer                       │
│  LightGBM × 3  │  HDBSCAN Clustering  │  Heuristic │
│  (Priority,     │  (Hotspot Detection) │  Diversion │
│   Closure,      │                      │  Lookup    │
│   Clearance)    │                      │  Table     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────┐
│          Flask REST API (Backend)                     │
│  /api/predict  │ /api/events │ /api/workload │ ...  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────┐
│       Event Command Center Dashboard (Frontend)      │
│  Leaflet Map │ Prediction Form │ Workload │ Trends  │
│              │                 │ Panel    │ Panel   │
│        Professional four-panel operations UI         │
└─────────────────────────────────────────────────────┘
```

---

## Models

| Model | Target | Algorithm | Key Metric |
|-------|--------|-----------|------------|
| **A** | Priority (High/Low) | LightGBM Classifier | Precision, Recall, F1, ROC-AUC |
| **B** | Road Closure (Yes/No) | LightGBM Classifier | Precision, Recall, F1, ROC-AUC |
| **C** | Clearance Time (minutes) | LightGBM Regressor | MAE, RMSE, Median AE |

- **Tuning**: Optuna hyperparameter search with configurable trial budget (`--trials`)
- **Feature engineering**: temporal/cyclical features, location density, H3 cells, historical frequency signals, keyword flags, interaction features, and leakage-safe target encoding
- **Split**: 80/20, stratified for classifiers
- **Class balancing**: LightGBM class weighting / positive-class scaling where applicable
- **Artifacts**: fitted model bundles in `models/`, feature contracts in `data/feature_config.json`, and diagnostics in `outputs/model_diagnostics/`
- **Full metrics**: See `outputs/model_evaluation_report.json`

---

## API Reference

All endpoints are served from the Flask backend.

### Health & Status

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check — returns model/data load status |
| `GET` | `/api/filters` | Available filter options (causes, corridors, zones) |
| `GET` | `/api/evaluation` | Model evaluation metrics |

### Data Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/events` | All historical events (for map display) |
| `GET` | `/api/events/summary` | Aggregated summary statistics |
| `GET` | `/api/hotspots` | HDBSCAN hotspot cluster data |
| `GET` | `/api/diversions` | Full diversion lookup table |
| `GET` | `/api/diversions/:corridor` | Diversion suggestions for a specific corridor |

### Analytics Endpoints

| Method | Endpoint | Query Params | Description |
|--------|----------|-------------|-------------|
| `GET` | `/api/workload` | `start_date`, `end_date` | Police station workload data |
| `GET` | `/api/trends` | `cause`, `corridor`, `zone` | Historical trend data (monthly, hourly, daily) |

### Prediction

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/api/predict` | `{latitude, longitude, event_cause, veh_type, event_type, datetime_str}` | Returns priority, closure, clearance-time predictions + diversion suggestions |

<details>
<summary><strong>Example prediction request</strong></summary>

```bash
curl -X POST https://your-backend-url/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 12.9716,
    "longitude": 77.5946,
    "event_cause": "accident",
    "veh_type": "heavy_vehicle",
    "event_type": "unplanned",
    "datetime_str": "2025-06-15T14:30:00"
  }'
```

</details>

---

## Local Development

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create and activate virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full offline pipeline (features + models + hotspots + diversions)
# Cheap smoke run:
python run.py --trials 10

# Final training run:
python run.py --trials 100

# Or retrain only the models with an explicit tuning budget
python src/train_models.py --trials 100

# 5. Start the server (serves both API + dashboard)
python src/server.py

# 6. Open browser
# Navigate to http://localhost:5000
```

> **Note**: If models are already trained (files exist in `models/`), you can skip step 4 and go directly to step 5.

---

## Deployment

### Backend — Render (Free Tier)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Web Service**
3. Connect your GitHub repository
4. Configure:
   - **Root Directory**: _(leave empty)_
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
   - **Instance Type**: Free
5. Deploy → copy the generated URL (e.g., `https://your-app.onrender.com`)
6. Verify: `curl https://your-app.onrender.com/api/health`

### Frontend — Vercel (Free Tier)

1. Go to [vercel.com](https://vercel.com) → **New Project**
2. Import the same GitHub repository
3. Configure:
   - **Root Directory**: `dashboard`
   - **Framework Preset**: Other
4. Before deploying, add the backend API config in `dashboard/index.html`:
   ```html
   <script>
     window.APP_CONFIG = { API_BASE: 'https://your-app.onrender.com' };
   </script>
   ```
5. Deploy → your dashboard is live with a global CDN

### Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `5000` | Server port (set automatically by Render) |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins for CORS |

---

## Dataset

- **Source**: Astram Event Data (Anonymized) — Bengaluru traffic-disrupting events
- **Rows**: 8,173 incident-lifecycle records
- **Coverage**: 94.3% unplanned, 5.7% planned events
- **No external data used** (holiday calendar is a static hardcoded reference table)

---

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Backend | Python, Flask, Flask-CORS | 3.11+, 3.x |
| ML | LightGBM, scikit-learn, HDBSCAN | 4.x, 1.9+, 0.8+ |
| Spatial | H3, Leaflet.js | 4.x, 1.9.4 |
| Charts | Chart.js | 4.4.0 |
| Explainability | SHAP | 0.52+ |
| Production | Gunicorn | 21.2+ |

---

## File Structure

```
├── dashboard/                    # Frontend (static HTML/CSS/JS)
│   ├── index.html               # Event Command Center
│   ├── style.css                # Premium dark theme
│   ├── app.js                   # Dashboard logic
│   └── vercel.json              # Vercel deployment config
├── src/
│   ├── __init__.py              # Package init
│   ├── data_pipeline.py         # Data cleaning + feature engineering
│   ├── train_models.py          # LightGBM training + evaluation
│   ├── clustering.py            # HDBSCAN hotspot detection
│   ├── diversion.py             # Co-occurrence diversion heuristic
│   └── server.py                # Flask API server
├── data/                         # Generated: cleaned datasets
├── models/                       # Trained model files (.pkl)
├── outputs/                      # Evaluation reports, clusters, diversions
├── run.py                        # Pipeline orchestrator
├── wsgi.py                       # WSGI entry point (gunicorn)
├── requirements.txt              # Python dependencies
├── Procfile                      # Render/Heroku process file
├── render.yaml                   # Render Blueprint
├── runtime.txt                   # Python version spec
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## Known Limitations

1. **Diversion suggestions** are heuristic and co-occurrence-based — they surface historically correlated alternative corridors, not shortest-path routing. No road-network graph is used.
2. **Render free tier** may experience cold starts (~30s spin-up on the first request after a period of inactivity).
3. **Clearance-time prediction** applies only to events that have a recorded resolution (closed/resolved status). Events still marked as active are excluded from this model's training by design, since they lack a ground-truth clearance duration.

---

## License

This project was built for the GridLock Hackathon (Flipkart). See the repository for license details.
