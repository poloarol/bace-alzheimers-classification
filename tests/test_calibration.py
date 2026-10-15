import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration import (  # noqa: E402
    apply_temperature,
    calibrate_and_compare,
    expected_calibration_error,
    fit_temperature,
    load_temperature,
    save_temperature,
)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(0)
        n = 600
        true_logit = torch.randn(n, generator=generator) * 1.5
        labels = (torch.rand(n, generator=generator) < torch.sigmoid(true_logit)).long()
        # Overconfident model: logits inflated 3x.
        self.logits = torch.stack([torch.zeros(n), true_logit * 3.0], dim=1)
        self.labels = labels

    def test_temperature_is_positive_and_softens_overconfidence(self):
        temperature = fit_temperature(self.logits, self.labels)
        self.assertGreater(temperature, 1.5)

    def test_scaling_reduces_ece(self):
        temperature = fit_temperature(self.logits, self.labels)
        labels = self.labels.numpy()
        before = expected_calibration_error(labels, apply_temperature(self.logits, 1.0))
        after = expected_calibration_error(labels, apply_temperature(self.logits, temperature))
        self.assertLess(after, before)

    def test_compare_writes_diagram_and_round_trips_temperature(self):
        with tempfile.TemporaryDirectory() as directory:
            result = calibrate_and_compare(
                self.logits, self.labels, self.logits, self.labels, directory
            )
            self.assertTrue((Path(directory) / "reliability_diagram.png").exists())
            self.assertLess(result["calibrated"]["NLL"], result["uncalibrated"]["NLL"])
            path = Path(directory) / "m.calibration.json"
            save_temperature(result["temperature"], path)
            self.assertAlmostEqual(load_temperature(path), result["temperature"])


    def test_saved_predictions_round_trip_through_cli_reader(self):
        from calibration import _read_predictions, save_predictions

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "validation_predictions.csv"
            save_predictions(self.labels.numpy(), self.logits, path)
            logits, labels = _read_predictions(str(path))
            self.assertTrue(torch.allclose(logits, self.logits.double()))
            self.assertEqual(labels.tolist(), self.labels.tolist())


if __name__ == "__main__":
    unittest.main()
