import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from app import PipelinePaths, PipelineRunner, build_parser
from calibration import save_predictions
from conformal import MolecularConformalPredictor, main


class MolecularConformalPredictorTests(unittest.TestCase):
    def setUp(self):
        true_class_probabilities = [0.9] * 4 + [0.8] * 4 + [0.5]
        self.calibration_labels = np.array([0, 1] * 4 + [0])
        self.calibration_probabilities = np.array([
            [p, 1 - p] if label == 0 else [1 - p, p]
            for p, label in zip(true_class_probabilities, self.calibration_labels)
        ])

    def test_finite_sample_threshold_and_prediction_sets(self):
        predictor = MolecularConformalPredictor(confidence=0.75)
        predictor.fit(self.calibration_probabilities, self.calibration_labels)

        result = predictor.predict(np.array([[0.8, 0.2], [0.5, 0.5]]))

        self.assertAlmostEqual(result["threshold"], 0.2)
        self.assertEqual(result["prediction_sets"], [[0], []])
        self.assertEqual(result["set_sizes"].tolist(), [1, 0])

    def test_nonstandard_class_order_is_supported(self):
        predictor = MolecularConformalPredictor(classes=(1, 0), confidence=0.75)
        probabilities = self.calibration_probabilities[:, ::-1]
        predictor.fit(probabilities, self.calibration_labels)

        result = predictor.predict(np.array([[0.2, 0.8]]))

        self.assertEqual(result["prediction_sets"], [[0]])

    def test_evaluation_reports_class_and_group_coverage(self):
        predictor = MolecularConformalPredictor(confidence=0.75)
        predictor.fit(self.calibration_probabilities, self.calibration_labels)

        metrics = predictor.evaluate(
            np.array([[0.8, 0.2], [0.5, 0.5]]),
            np.array([0, 1]),
            groups=np.array(["scaffold-a", "scaffold-b"]),
        )

        self.assertEqual(metrics["n_calibration"], 9)
        self.assertEqual(metrics["empirical_coverage"], 0.5)
        self.assertEqual(metrics["coverage_by_group"]["scaffold-a"]["coverage"], 1.0)
        self.assertEqual(metrics["coverage_by_group"]["scaffold-b"]["coverage"], 0.0)

    def test_invalid_inputs_and_unfitted_prediction_are_rejected(self):
        predictor = MolecularConformalPredictor()
        with self.assertRaises(RuntimeError):
            predictor.predict(np.array([[0.5, 0.5]]))
        with self.assertRaises(ValueError):
            predictor.fit(np.array([[0.2, 0.2]]), np.array([0]))
        with self.assertRaises(ValueError):
            predictor.fit(self.calibration_probabilities, np.array([0, 1, 0]))


class ConformalCliTests(unittest.TestCase):
    def test_parser_exposes_model_inputs_temperature_and_validated_confidence(self):
        parser = build_parser()
        args = parser.parse_args([
            "conformal",
            "--model-type", "GCN",
            "--calibration-predictions", "calibration_predictions.csv",
            "--test-predictions", "test_predictions.csv",
            "--temperature-file", "GCN.calibration.json",
            "--confidence", "0.95",
        ])

        self.assertEqual(args.model_type, "GCN")
        self.assertEqual(args.temperature_file, "GCN.calibration.json")
        self.assertEqual(args.confidence, 0.95)
        with self.assertRaises(SystemExit):
            parser.parse_args(["conformal", "--confidence", "1.0"])

    def test_cli_reads_pipeline_predictions_and_writes_report_and_sets(self):
        logits = np.array([
            [-2.0, 2.0],
            [2.0, -2.0],
            [-1.5, 1.5],
            [1.5, -1.5],
        ])
        labels = np.array([1, 0, 1, 0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = root / "calibration_predictions.csv"
            test_path = root / "test_predictions.csv"
            groups_path = root / "test_groups.csv"
            output_directory = root / "conformal"
            save_predictions(labels, logits, calibration_path)
            save_predictions(labels, logits, test_path)
            np.savetxt(groups_path, ["scaffold-a", "scaffold-b", "scaffold-a", "scaffold-b"], fmt="%s")

            with patch("sys.argv", [
                "conformal.py",
                "--calibration-predictions", str(calibration_path),
                "--test-predictions", str(test_path),
                "--output-dir", str(output_directory),
                "--confidence", "0.75",
                "--groups", str(groups_path),
            ]):
                main()

            report_path = output_directory / "conformal_report.json"
            predictions_path = output_directory / "conformal_predictions.csv"
            self.assertTrue(report_path.is_file())
            self.assertTrue(predictions_path.is_file())
            with predictions_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(labels))
            self.assertEqual(rows[0]["prediction_set"], "[1]")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("scaffold-a", report["metrics"]["coverage_by_group"])

    def test_pipeline_runner_uses_prediction_and_temperature_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = PipelinePaths.from_dataset_root(directory)
            runner = PipelineRunner(paths, tracking_uri=None, dry_run=True)
            with patch.object(runner, "_run") as run_script:
                runner.run_conformal("GCN")

        export_script, export_arguments = run_script.call_args_list[0].args
        self.assertEqual(export_script, "train.py")
        self.assertIn("--export-calibration-predictions", export_arguments)
        self.assertIn(str(paths.model_directory / "GCN.pth"), export_arguments)
        script_name, arguments = run_script.call_args.args
        self.assertEqual(script_name, "conformal.py")
        self.assertIn(str(paths.predictions_directory / "GCN" / "calibration_predictions.csv"), arguments)
        self.assertIn(str(paths.predictions_directory / "GCN" / "test_predictions.csv"), arguments)
        self.assertIn(str(paths.model_directory / "GCN.calibration.json"), arguments)


if __name__ == "__main__":
    unittest.main()
