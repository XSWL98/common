import unittest
import numpy as np
import pandas as pd
from power_ad import PowerAnomalyDetector

class SystemSimulationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        n = 300
        self.data = pd.DataFrame({
            "timestamp": pd.date_range("2026-09-15", periods=n, freq="min"),
            "power_kw": 400 + rng.normal(0, 2, n),
            "current_a": 1100 + rng.normal(0, 4, n),
            "voltage_v": 380 + rng.normal(0, 0.4, n),
            "frequency_hz": 50 + rng.normal(0, 0.01, n),
        })
        self.data.loc[100:107, ["power_kw", "current_a"]] += [250, 700]
        self.data.loc[210:217, ["power_kw", "current_a"]] -= [220, 600]
        self.detector = PowerAnomalyDetector(window=30, z_threshold=4, change_threshold=12, min_duration=2)

    def test_detects_increase_and_decrease(self):
        result = self.detector.predict(self.data)
        self.assertTrue(result.loc[101:107, "is_anomaly"].any())
        self.assertTrue(result.loc[211:217, "is_anomaly"].any())

    def test_normal_region_is_mostly_quiet(self):
        result = self.detector.predict(self.data)
        normal = result.loc[40:90, "is_anomaly"]
        self.assertLess(normal.mean(), 0.1)

    def test_event_summary(self):
        result = self.detector.predict(self.data)
        events = self.detector.events(result)
        self.assertGreaterEqual(len(events), 2)
        self.assertIn("driver", events.columns)

if __name__ == "__main__":
    unittest.main()
