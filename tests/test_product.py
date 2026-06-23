"""End-to-end smoke tests for the trained API product."""

import os
import sys
import unittest
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from server import app  # noqa: E402


class ProductSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_health_and_assets(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["models_loaded"])
        self.assertEqual(payload["total_events"], 8173)

    def test_prediction_contract(self):
        response = self.client.post("/api/predict", json={
            "latitude": 12.9716,
            "longitude": 77.5946,
            "event_cause": "vehicle_breakdown",
            "veh_type": "car",
            "event_type": "unplanned",
            "description": "Vehicle breakdown blocking one lane",
            "datetime_str": "2026-06-21T12:00:00+05:30",
        })
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertIn("prediction", payload["priority"])
        self.assertIn("probability", payload["closure"])
        self.assertGreaterEqual(payload["clearance_time"]["predicted_minutes"], 0)

    def test_invalid_coordinates(self):
        response = self.client.post("/api/predict", json={"latitude": 120, "longitude": 77})
        self.assertEqual(response.status_code, 400)

    def test_feature_contract_has_no_status_leakage(self):
        config_path = os.path.join(ROOT, "data", "feature_config.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)

        classification_features = config["classification_features"]
        regression_features = config["regression_features"]
        self.assertNotIn("status", classification_features)
        self.assertNotIn("status", regression_features)

        expected_advanced_features = {
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "events_within_1km",
            "events_within_3km",
            "events_within_5km",
            "police_station_freq",
            "h3_cell_freq",
            "cause_x_corridor",
            "cause_x_hour_bin",
            "desc_has_accident",
            "desc_has_tree",
            "desc_has_fire",
            "desc_has_vip",
            "desc_has_protest",
        }
        self.assertTrue(expected_advanced_features.issubset(set(classification_features)))

    def test_clearance_subset_excludes_active_rows(self):
        import pandas as pd

        subset_path = os.path.join(ROOT, "data", "features_clearance_subset.csv")
        subset = pd.read_csv(subset_path, usecols=["status", "clearance_time_minutes"])
        self.assertFalse((subset["status"] == "active").any())
        self.assertTrue((subset["clearance_time_minutes"] > 0).all())

    def test_dashboard_navigation_excludes_methodology(self):
        index_path = os.path.join(ROOT, "dashboard", "index.html")
        with open(index_path, "r", encoding="utf-8") as handle:
            html = handle.read().lower()

        self.assertNotIn("methodology", html)
        self.assertIn("predict", html)
        self.assertIn("workload", html)


if __name__ == "__main__":
    unittest.main()
