import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibration import (  # noqa: E402
    apply_temperature,
    bootstrap_metric_intervals,
    brier_score,
    calibrate_and_compare,
    calibration_metrics,
    check_split_filename,
    _read_predictions,
    class_counts,
    expected_calibration_error,
    fit_temperature,
    load_calibration,
    load_temperature,
    negative_log_likelihood,
    probabilities_to_logits,
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
            self.assertEqual(result["n_test_samples"], len(self.labels))
            with open(Path(directory) / "reliability_table.csv", encoding="utf-8") as handle:
                table = list(csv.DictReader(handle))
            self.assertEqual(len(table), 10)
            for kind in ("uncalibrated", "calibrated"):
                self.assertEqual(sum(int(r[f"{kind}_count"]) for r in table), len(self.labels))
            self.assertLess(result["calibrated"]["NLL"], result["uncalibrated"]["NLL"])
            path = Path(directory) / "m.calibration.json"
            save_temperature(result["temperature"], path)
            self.assertAlmostEqual(load_temperature(path), result["temperature"])


    def test_fit_rejects_invalid_inputs(self):
        good = self.logits
        cases = {
            "nan logits": (torch.full_like(good, float("nan")), self.labels),
            "label out of range": (good, self.labels + 2),
            "fractional labels": (good, self.labels.float() + 0.5),
            "single class": (good, torch.zeros_like(self.labels)),
            "length mismatch": (good, self.labels[:-1]),
            "one column": (good[:, :1], self.labels),
            "constant logits": (torch.zeros_like(good), self.labels),
            "integer logits": (good.long(), self.labels),
        }
        for name, (logits, labels) in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                fit_temperature(logits, labels)
        with self.assertRaises(ValueError):
            fit_temperature(good, self.labels, max_iter=0)

    def test_apply_rejects_bad_temperature(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(bad), self.assertRaises(ValueError):
                apply_temperature(self.logits, bad)

    def test_apply_validates_logits_and_handles_grad(self):
        for name, bad in {
            "three columns": torch.zeros(4, 3),
            "one dim": torch.zeros(4),
            "nan": torch.full((4, 2), float("nan")),
            "inf": torch.full((4, 2), float("inf")),
        }.items():
            with self.subTest(name), self.assertRaises(ValueError):
                apply_temperature(bad, 1.0)
        with_grad = self.logits.clone().requires_grad_(True)
        probabilities = apply_temperature(with_grad, 2.0)
        self.assertEqual(probabilities.shape, (self.logits.shape[0],))

    def test_probabilities_to_logits_validation(self):
        from calibration import probabilities_to_logits

        good = np.array([[0.2, 0.8], [0.5, 0.5]])
        logits = probabilities_to_logits(good)
        self.assertEqual(tuple(logits.shape), (2, 2))
        # softmax of the log-probabilities recovers the input
        self.assertTrue(np.allclose(torch.softmax(logits, dim=1).numpy(), good))

        cases = {
            "empty": np.empty((0, 2)),
            "one dim": np.array([0.2, 0.8]),
            "three columns": np.full((2, 3), 1 / 3),
            "nan": np.array([[np.nan, 0.5]]),
            "inf": np.array([[np.inf, 0.0]]),
            "negative": np.array([[-0.1, 1.1]]),
            "above one": np.array([[1.2, -0.2]]),
            "rows not summing to one": np.array([[0.3, 0.3]]),
        }
        for name, bad in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                probabilities_to_logits(bad)

    def test_binary_metrics_validate_inputs(self):
        from calibration import brier_score, negative_log_likelihood

        labels = np.array([0, 1, 1, 0])
        probabilities = np.array([0.1, 0.9, 0.8, 0.3])
        for metric in (brier_score, negative_log_likelihood, expected_calibration_error):
            cases = {
                "length mismatch": (labels, probabilities[:-1]),
                "empty": (np.array([]), np.array([])),
                "nan probability": (labels, np.array([0.1, np.nan, 0.8, 0.3])),
                "probability above one": (labels, np.array([0.1, 1.5, 0.8, 0.3])),
                "negative probability": (labels, np.array([-0.1, 0.9, 0.8, 0.3])),
                "non-binary labels": (np.array([0, 2, 1, 0]), probabilities),
                "fractional labels": (np.array([0, 0.5, 1, 0]), probabilities),
                "string labels": (np.array(["a", "b", "a", "b"]), probabilities),
                "2-D probabilities": (labels, np.stack([1 - probabilities, probabilities], axis=1)),
            }
            for name, (y, p) in cases.items():
                with self.subTest(metric=metric.__name__, case=name), self.assertRaises(ValueError):
                    metric(y, p)

    def test_binary_metric_values(self):
        from calibration import brier_score, negative_log_likelihood

        labels = np.array([0, 1])
        probabilities = np.array([0.2, 0.6])
        self.assertAlmostEqual(brier_score(labels, probabilities), (0.2**2 + 0.4**2) / 2)
        self.assertAlmostEqual(
            negative_log_likelihood(labels, probabilities),
            -(np.log(0.8) + np.log(0.6)) / 2,
        )
        # Confident wrong predictions stay finite thanks to clipping.
        self.assertTrue(np.isfinite(negative_log_likelihood(np.array([1]), np.array([0.0]))))
        with self.assertRaises(ValueError):
            expected_calibration_error(labels, probabilities, n_bins=0)

    def test_saved_predictions_round_trip_through_cli_reader(self):
        from calibration import _read_predictions, save_predictions

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "validation_predictions.csv"
            save_predictions(self.labels.numpy(), self.logits, path)
            logits, labels = _read_predictions(str(path))
            self.assertTrue(torch.allclose(logits, self.logits.double()))
            self.assertEqual(labels.tolist(), self.labels.tolist())


class CalibrationEdgeCaseTests(unittest.TestCase):
    def test_temperature_one_is_identity(self):
        logits = torch.tensor([[0.0, 2.0], [1.0, -1.0], [0.3, 0.1]], dtype=torch.float64)
        np.testing.assert_allclose(
            apply_temperature(logits, 1.0), torch.softmax(logits, dim=1)[:, 1].numpy(), rtol=1e-9
        )

    def test_already_calibrated_logits_fit_near_one(self):
        generator = torch.Generator().manual_seed(1)
        n = 4000
        score = torch.randn(n, generator=generator) * 1.5
        labels = (torch.rand(n, generator=generator) < torch.sigmoid(score)).long()
        logits = torch.stack([torch.zeros(n), score], dim=1)
        self.assertAlmostEqual(fit_temperature(logits, labels), 1.0, delta=0.15)

    def test_overconfident_logits_give_t_above_one_and_lower_nll(self):
        generator = torch.Generator().manual_seed(2)
        n = 1000
        score = torch.randn(n, generator=generator)
        labels = (torch.rand(n, generator=generator) < torch.sigmoid(score)).long()
        logits = torch.stack([torch.zeros(n), score * 5.0], dim=1)
        temperature = fit_temperature(logits, labels)
        self.assertGreater(temperature, 2.0)
        before = negative_log_likelihood(labels.numpy(), apply_temperature(logits, 1.0))
        after = negative_log_likelihood(labels.numpy(), apply_temperature(logits, temperature))
        self.assertLess(after, before)

    def test_invalid_probabilities_rejected(self):
        labels = np.array([0, 1, 1])
        for bad in ([0.2, 1.2, 0.5], [0.2, -0.1, 0.5], [0.2, np.nan, 0.5], [0.2, np.inf, 0.5]):
            with self.assertRaises(ValueError):
                brier_score(labels, np.array(bad))
            with self.assertRaises(ValueError):
                negative_log_likelihood(labels, np.array(bad))
            with self.assertRaises(ValueError):
                expected_calibration_error(labels, np.array(bad))
        with self.assertRaises(ValueError):
            probabilities_to_logits(np.array([[0.5, 0.6]]))

    def test_empty_inputs_rejected(self):
        empty = np.array([])
        for function in (brier_score, negative_log_likelihood, expected_calibration_error):
            with self.assertRaises(ValueError):
                function(empty, empty)
        with self.assertRaises(ValueError):
            fit_temperature(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
        with self.assertRaises(ValueError):
            apply_temperature(torch.empty(0, 3), 1.0)
        with self.assertRaises(ValueError):
            probabilities_to_logits(np.empty((0, 2)))
        with self.assertRaises(ValueError):
            bootstrap_metric_intervals(empty, empty, empty)

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            brier_score(np.array([0, 1]), np.array([0.2, 0.4, 0.9]))
        with self.assertRaises(ValueError):
            negative_log_likelihood(np.array([0, 1]), np.array([0.2]))
        with self.assertRaises(ValueError):
            fit_temperature(torch.randn(4, 2), torch.tensor([0, 1, 0]))

    def test_single_class_labels(self):
        logits = torch.randn(10, 2)
        with self.assertRaises(ValueError):
            fit_temperature(logits, torch.ones(10, dtype=torch.long))
        # Metrics remain defined when a (small) test set holds one class.
        metrics = calibration_metrics(np.zeros(5, dtype=int), np.full(5, 0.1))
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_invalid_json_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for text in ("{not json", "[1, 2]", "", '{"temperature": "2"}',
                         '{"temperature": true}', '{"schema_version": 99, "temperature": 1.0}',
                         '{"temperature": NaN}', '{"temperature": Infinity}'):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError, msg=text):
                    load_calibration(path)
            with self.assertRaises(FileNotFoundError):
                load_calibration(Path(directory) / "missing.json")

    def test_model_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "GCN.calibration.json"
            save_temperature(1.7, path, model="GCN")
            self.assertEqual(load_temperature(path, expected_model="GCN"), 1.7)
            with self.assertRaisesRegex(ValueError, "GCN"):
                load_calibration(path, expected_model="GAT")
            # An artifact without model identity cannot be checked, so it still loads.
            anonymous = Path(directory) / "anon.json"
            save_temperature(1.7, anonymous)
            self.assertEqual(load_temperature(anonymous, expected_model="GAT"), 1.7)

    def test_bootstrap_intervals(self):
        generator = np.random.default_rng(3)
        labels = generator.integers(0, 2, 200)
        score = (labels * 2 - 1) * 1.0 + generator.normal(0, 2, 200)
        logits = torch.as_tensor(np.stack([np.zeros(200), score * 3], axis=1))
        temperature = fit_temperature(logits, torch.as_tensor(labels))
        uncal = apply_temperature(logits, 1.0)
        cal = apply_temperature(logits, temperature)
        first = bootstrap_metric_intervals(labels, uncal, cal, n_resamples=200, seed=5)
        second = bootstrap_metric_intervals(labels, uncal, cal, n_resamples=200, seed=5)
        self.assertEqual(first, second)
        point = calibration_metrics(labels, cal)
        for metric in ("Brier", "NLL", "ECE"):
            bounds = first["calibrated"][metric]
            self.assertLessEqual(bounds["lower"], bounds["upper"])
            self.assertLessEqual(bounds["lower"], point[metric] + 0.05)
        self.assertLess(first["delta"]["NLL"]["upper"], 0.0)
        with self.assertRaises(ValueError):
            bootstrap_metric_intervals(labels, uncal, cal, n_resamples=0)
        with self.assertRaises(ValueError):
            bootstrap_metric_intervals(labels, uncal, cal, confidence=1.0)


class CalibrationArtifactTests(unittest.TestCase):
    def test_save_creates_parent_and_records_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a" / "b" / "GCN.calibration.json"
            save_temperature(1.5, path, model="GCN", checkpoint="GCN.pth")
            payload = load_calibration(path, expected_model="GCN")
            self.assertEqual(payload["schema_version"], 1)
            self.assertIn("BACE-1 inhibitor", payload["positive_class"])
            self.assertEqual(payload["checkpoint"], "GCN.pth")
            self.assertEqual(load_temperature(path), 1.5)

    def test_rejects_bad_temperatures_and_model_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.json"
            for bad in (0.0, -1.0, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    save_temperature(bad, path)
            save_temperature(2.0, path, model="GCN")
            with self.assertRaises(ValueError):
                load_temperature(path, expected_model="GAT")
            path.write_text('{"temperature": -1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_temperature(path)

    def test_csv_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p.csv"
            path.write_text("label,probability\n0,0.2\n2,0.4\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _read_predictions(str(path))
            path.write_text("label,logit_0,logit_1\n0,nan,1.0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                _read_predictions(str(path))
            path.write_text("label,probability\n0,0.2\n1,0.9\n", encoding="utf-8")
            logits, labels = _read_predictions(str(path))
            self.assertEqual(tuple(logits.shape), (2, 2))
            self.assertEqual(class_counts(labels), {"n": 2, "class_0": 1, "class_1": 1})


class CalibrationCliTests(unittest.TestCase):
    def test_cli_writes_artifact_and_report(self):
        import json
        import subprocess

        generator = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("validation", "test"):
                labels = generator.integers(0, 2, 80)
                logits = np.stack([-(labels * 2 - 1) * 3.0, (labels * 2 - 1) * 3.0], axis=1)
                logits = logits + generator.normal(0, 3, logits.shape)
                with open(root / f"{name}_predictions.csv", "w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["label", "logit_0", "logit_1"])
                    writer.writerows(zip(labels, logits[:, 0], logits[:, 1]))
            script = Path(__file__).resolve().parents[1] / "src" / "calibration.py"
            subprocess.run(
                [sys.executable, str(script), "--fit-predictions", str(root / "validation_predictions.csv"),
                 "--eval-predictions", str(root / "test_predictions.csv"), "--output-dir", str(root / "out"),
                 "--temperature-output", str(root / "m" / "GCN.calibration.json"),
                 "--model-name", "GCN", "--checkpoint", "GCN.pth", "--bins", "5"],
                check=True, capture_output=True,
            )
            report = json.loads((root / "out" / "calibration_report.json").read_text())
            self.assertEqual(report["binning"]["n_bins"], 5)
            self.assertEqual(report["eval_counts"]["n"], 80)
            self.assertEqual(report["fit_split"], "validation")
            self.assertEqual(report["eval_split"], "test")
            self.assertTrue(report["fit_predictions_file"].endswith("validation_predictions.csv"))
            self.assertTrue(report["eval_predictions_file"].endswith("test_predictions.csv"))
            artifact = load_calibration(root / "m" / "GCN.calibration.json", "GCN")
            self.assertEqual(artifact["fit_split"], "validation")
            self.assertEqual(artifact["eval_split"], "test")
            self.assertNotIn("fitted_on", artifact)
            self.assertIn("ECE", report["metrics"]["calibrated"])
            self.assertEqual(report["uncertainty"]["n_resamples"], 1000)
            self.assertIn("NLL", report["uncertainty"]["delta"])
            self.assertTrue((root / "out" / "reliability_diagram.png").is_file())
            self.assertGreater(load_temperature(root / "m" / "GCN.calibration.json", "GCN"), 0)
            fit, evaluate = str(root / "validation_predictions.csv"), str(root / "test_predictions.csv")
            for arguments in (
                ["--fit-predictions", fit, "--eval-predictions", evaluate, "--bins", "0"],
                ["--fit-predictions", evaluate, "--eval-predictions", fit],  # swapped splits
                ["--fit-predictions", fit, "--eval-predictions", fit],  # same file twice
            ):
                bad = subprocess.run(
                    [sys.executable, str(script), "--output-dir", str(root / "bad"), *arguments],
                    capture_output=True,
                )
                self.assertNotEqual(bad.returncode, 0, arguments)
            self.assertFalse((root / "bad").exists())
            # Legacy aliases still work.
            legacy = subprocess.run(
                [sys.executable, str(script), "--validation", fit, "--test", evaluate,
                 "--output-dir", str(root / "legacy"), "--bootstrap", "0"],
                capture_output=True,
            )
            self.assertEqual(legacy.returncode, 0, legacy.stderr)

    def test_check_split_filename(self):
        check_split_filename("out/validation_predictions.csv", "validation", "fit")
        check_split_filename("custom.csv", "validation", "fit")
        with self.assertRaises(ValueError):
            check_split_filename("out/test_predictions.csv", "validation", "fit")
        with self.assertRaises(ValueError):
            check_split_filename("calibration_predictions.csv", "test", "eval")

    def test_scaling_keeps_argmax_and_known_temperature(self):
        logits = torch.tensor([[0.0, 2.0], [1.0, -1.0], [0.3, 0.1]])
        for temperature in (0.5, 1.0, 4.0):
            scaled = apply_temperature(logits, temperature)
            self.assertEqual((scaled >= 0.5).astype(int).tolist(), logits.argmax(dim=1).tolist())
        expected = torch.softmax(logits / 2.0, dim=1)[:, 1].numpy()
        np.testing.assert_allclose(apply_temperature(logits, 2.0), expected, rtol=1e-6)
    def test_calibrated_inference_output(self):
        from inference import save_calibrated_predictions

        logits = torch.tensor([[0.0, 2.0], [1.0, -1.0], [0.3, 0.1]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_temperature(2.0, root / "GCN.calibration.json", model="GCN")
            path = save_calibrated_predictions(
                logits, np.array([1, 0, 1]), str(root / "GCN.calibration.json"), "GCN", root
            )
            rows = list(csv.DictReader(open(path)))
            expected = torch.softmax(logits / 2.0, dim=1)[:, 1].numpy()
            np.testing.assert_allclose([float(r["calibrated_probability"]) for r in rows], expected, rtol=1e-6)
            with self.assertRaises(ValueError):
                save_calibrated_predictions(
                    logits, np.array([1, 0, 1]), str(root / "GCN.calibration.json"), "GAT", root
                )


if __name__ == "__main__":
    unittest.main()
