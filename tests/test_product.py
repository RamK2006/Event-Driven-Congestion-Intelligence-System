"""End-to-end smoke tests for the trained API product."""

import os
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
