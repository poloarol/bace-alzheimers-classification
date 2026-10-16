import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import train
from app import PipelinePaths, PipelineRunner, build_parser
from audit_splits import build_report
from calibration import save_predictions
from compare_models import compare_models

def _write_dataset(root: Path) -> dict[str, np.ndarray]:
    (root / "raw").mkdir(parents=True)
    (root / "clean").mkdir()
    n = 40
    labels = np.array([0, 1] * (n // 2))
    frame = pd.DataFrame({"mol": ["C" * (i + 1) for i in range(n)], "Class": labels})
    frame["pIC50"] = np.where(labels == 1, 7.5, 5.5)
    frame.to_csv(root / "raw" / "bace.csv", index=False)
    frame[["mol", "Class"]].to_csv(root / "raw" / "bace_processed.csv", index=False)

    order = np.arange(n)
    split = {
        "train_idx": order[:24],
        "validation_idx": order[24:30],
        "calibration_idx": order[30:36],
        "test_idx": order[36:],
    }
    np.savez(root / "clean" / "train_test_split.npz", **split)
    return {name.replace("_idx", ""): labels[idx] for name, idx in split.items()}


def _write_model_artifacts(root: Path, name: str, labels_by_split: dict[str, np.ndarray]) -> None:
    import torch

    rng = np.random.default_rng(0)
    for split in ("validation", "calibration", "test"):
        labels = labels_by_split[split]
        logits = torch.tensor(np.column_stack((rng.normal(size=len(labels)), rng.normal(size=len(labels)) + labels)))
        save_predictions(labels, logits, root / "output" / "predictions" / name / f"{split}_predictions.csv")

    calibration = root / "output" / "calibration" / name
    calibration.mkdir(parents=True)
    (calibration / "calibration_report.json").write_text(json.dumps({
        "temperature": 1.5,
        "metrics": {
            "uncalibrated": {"Brier": 0.3, "NLL": 0.8, "ECE": 0.2},
            "calibrated": {"Brier": 0.25, "NLL": 0.7, "ECE": 0.1},
        },
    }), encoding="utf-8")
    conformal = root / "output" / "conformal" / name
    conformal.mkdir(parents=True)
    (conformal / "conformal_report.json").write_text(json.dumps({
        "metrics": {
            "confidence_target": 0.9,
            "empirical_coverage": 0.88,
            "coverage_by_class": {"0": {"coverage": 0.9}, "1": {"coverage": 0.86}},
            "average_set_size": 1.2,
            "singleton_rate": 0.8,
            "n_calibration": 6,
        },
    }), encoding="utf-8")


class AuditAndComparisonTests(unittest.TestCase):
    def test_audit_reports_class_counts_partitions_and_graph_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_dataset(root)
            report = build_report(root)

        splits = report["partitions"]["splits"]
        self.assertTrue(report["partitions"]["partitions_disjoint"])
        self.assertTrue(report["partitions"]["covers_all_molecules"])
        self.assertEqual({k: v["molecules"] for k, v in splits.items()},
                         {"train": 24, "validation": 6, "calibration": 6, "test": 4})
        self.assertEqual(splits["validation"]["class_counts"], {"0": 3, "1": 3})
        self.assertEqual(splits["test"]["invalid_smiles"], 0)
        self.assertEqual(report["labels"]["class_vs_pic50_threshold"]["total_disagreements"], 0)

    def test_comparison_scores_models_on_identical_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = _write_dataset(root)
            for name in ("M1", "M2"):
                _write_model_artifacts(root, name, labels)

            comparison = compare_models(root, ("M1", "M2"))
            self.assertEqual([row["model"] for row in comparison["results"]], ["M1", "M2"])
            self.assertIn("test_pr_auc", comparison["results"][0])
            self.assertEqual(comparison["results"][0]["conformal_coverage"], 0.88)

            wrong = labels["test"].copy()
            wrong[0] = 1 - wrong[0]
            _write_model_artifacts_with_wrong_test(root, "M3", labels, wrong)
            with self.assertRaises(ValueError):
                compare_models(root, ("M3",))

    def test_gnn_checkpoint_selection_uses_validation_pr_auc(self):
        source = inspect.getsource(train)
        self.assertIn('validation_metrics["PR-AUC"] > best_val_pr_auc', source)
        self.assertNotIn("best_val_acc", source)

    def test_parser_and_runner_expose_audit_and_compare(self):
        parser = build_parser()
        self.assertEqual(parser.parse_args(["audit"]).command, "audit")
        args = parser.parse_args(["compare", "--retrain", "--epochs", "5"])
        self.assertTrue(args.retrain)

        with tempfile.TemporaryDirectory() as directory:
            runner = PipelineRunner(PipelinePaths.from_dataset_root(directory), tracking_uri=None, dry_run=True)
            with patch.object(runner, "_run") as run_script:
                runner.run_comparison(epochs=5, retrain=True)

        scripts = [call.args[0] for call in run_script.call_args_list]
        self.assertEqual(scripts[0], "audit_splits.py")
        self.assertEqual(scripts[-1], "compare_models.py")
        self.assertGreaterEqual(scripts.count("train.py"), 3)


def _write_model_artifacts_with_wrong_test(root, name, labels, wrong_test):
    _write_model_artifacts(root, name, {**labels, "test": wrong_test})


if __name__ == "__main__":
    unittest.main()
