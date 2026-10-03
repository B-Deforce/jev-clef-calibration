"""Small checks for the calibration calculations (no network or data files)."""

import unittest

from calibration_metrics import score


class CalibrationMetricsTests(unittest.TestCase):
    def test_equal_scores_have_no_row_order_effect(self):
        probabilities = [0.2, 0.2, 0.2, 0.8, 0.8]
        first = score([1, 0, 0, 1, 0], probabilities)
        permuted = score([0, 0, 1, 0, 1], probabilities)
        self.assertAlmostEqual(first["ecce_r"], permuted["ecce_r"])
        self.assertAlmostEqual(first["ecce_mad"], permuted["ecce_mad"])

    def test_perfect_binary_predictions(self):
        metrics = score([0, 1], [0.0, 1.0])
        self.assertEqual(metrics["accuracy_at_0_5"], 1.0)
        self.assertEqual(metrics["brier"], 0.0)
        self.assertEqual(metrics["ecce_r"], 0.0)


if __name__ == "__main__":
    unittest.main()
