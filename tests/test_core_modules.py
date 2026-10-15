import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from app import PipelinePaths, build_parser
from baseline_model import (
    _generate_confusion_matrix,
    create_estimator,
    evaluate as evaluate_baseline,
    save_confusion_matrix_as_csv,
    save_confusion_matrix_as_image,
    train as train_baseline,
)
from build_fingerprints import build_morgan_fingerprint
from inference import (
    _output_directory,
    run_sklearn_inference,
    save_confusion_matrix,
)
from mlflow_utils import configure_mlflow, log_files, log_parameters
from model import GAT, GATGraphTransformer, GCN, build_model
from train import classification_metrics


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_shape_and_binary_values(self):
        fingerprint = build_morgan_fingerprint(
            "CCO",
            radius=2,
            n_bits=32,
        )

        self.assertEqual(fingerprint.shape, (32,))
        self.assertEqual(fingerprint.dtype, np.int8)
        self.assertTrue(set(np.unique(fingerprint)).issubset({0, 1}))
        self.assertGreater(fingerprint.sum(), 0)

    def test_invalid_smiles_is_rejected(self):
        with self.assertRaises(ValueError):
            build_morgan_fingerprint("not-a-smiles", n_bits=32)


class BaselineTests(unittest.TestCase):
    def test_estimator_factory_and_unknown_method(self):
        self.assertEqual(
            create_estimator("random_forest").__class__.__name__,
            "RandomForestClassifier",
        )
        self.assertEqual(
            create_estimator("logistic_regression").__class__.__name__,
            "LogisticRegression",
        )
        self.assertEqual(
            create_estimator("gradient_boosting").__class__.__name__,
            "GradientBoostingClassifier",
        )
        with self.assertRaises(ValueError):
            create_estimator("unknown")

    def test_training_and_evaluation(self):
        features = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.1, 0.0],
            [0.9, 1.0],
        ])
        labels = np.array([0, 0, 1, 1, 0, 1])
        estimator = create_estimator("logistic_regression")
        classifier = train_baseline(estimator, features, labels)
        metrics = evaluate_baseline(classifier, features, labels)

        self.assertEqual(
            set(metrics),
            {"ROC-AUC", "PR-AUC", "Balanced accuracy", "Inhibitor F1"},
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_confusion_matrix_outputs(self):
        labels = np.array([0, 0, 1, 1])
        predictions = np.array([0, 1, 1, 1])
        expected = np.array([[1, 1], [0, 2]])
        np.testing.assert_array_equal(
            _generate_confusion_matrix(labels, predictions),
            expected,
        )

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "matrix.csv"
            image_path = Path(directory) / "matrix.png"
            save_confusion_matrix_as_csv(labels, predictions, str(csv_path))
            save_confusion_matrix_as_image(labels, predictions, str(image_path))
            self.assertTrue(csv_path.is_file())
            self.assertTrue(image_path.is_file())
            saved = pd.read_csv(csv_path).to_numpy()
            np.testing.assert_array_equal(saved, expected)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.data = SimpleNamespace(
            num_classes=2,
            num_node_features=3,
            num_edge_features=2,
        )
        self.x = torch.randn(4, 3)
        self.edge_index = torch.tensor([
            [0, 1, 2, 3],
            [1, 2, 3, 0],
        ])
        self.edge_attr = torch.randn(4, 2)
        self.batch = torch.zeros(4, dtype=torch.long)

    def test_all_model_factories_produce_binary_logits(self):
        for model_name, model_class in (
            ("GCN", GCN),
            ("GAT", GAT),
            ("GATGraphTransformer", GATGraphTransformer),
        ):
            model = build_model(model_name, hidden_channels=8, data=self.data)
            self.assertIsInstance(model, model_class)
            output = model(
                self.x,
                self.edge_index,
                self.batch,
                self.edge_attr,
            )
            self.assertEqual(output.shape, (1, 2), model_name)

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            build_model("Unknown", hidden_channels=8, data=self.data)

    def test_attention_models_require_edge_attributes(self):
        for model_class in (GAT, GATGraphTransformer):
            model = model_class(hidden_channels=8, data=self.data)
            with self.assertRaises(ValueError):
                model(self.x, self.edge_index, self.batch)


class InferenceTests(unittest.TestCase):
    def test_output_directory_rejects_path_components(self):
        with tempfile.TemporaryDirectory() as directory:
            output = _output_directory(directory, "GCN")
            self.assertTrue(output.is_dir())
            with self.assertRaises(ValueError):
                _output_directory(directory, "../outside")
            with self.assertRaises(ValueError):
                _output_directory(directory, ".")

    def test_confusion_matrix_is_saved_with_fixed_class_order(self):
        labels = np.array([0, 1, 1])
        predictions = np.array([0, 0, 1])
        with tempfile.TemporaryDirectory() as directory:
            matrix = save_confusion_matrix(
                labels,
                predictions,
                Path(directory),
            )
            np.testing.assert_array_equal(matrix, np.array([[1, 0], [1, 1]]))
            self.assertTrue((Path(directory) / "confusion_matrix.txt").is_file())
            self.assertTrue((Path(directory) / "confusion_matrix.png").is_file())

    def test_sklearn_inference_loads_test_features_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pd.DataFrame([[1, 0], [0, 1]]).to_csv(
                directory / "test_fingerprints.csv",
                index=False,
            )
            pd.DataFrame([0, 1]).to_csv(
                directory / "test_labels.csv",
                index=False,
            )
            model = SimpleNamespace(
                predict=lambda features: np.array([0, 1]),
            )
            with patch(
                "inference.mlflow.sklearn.load_model",
                return_value=model,
            ) as load_model:
                labels, predictions = run_sklearn_inference(
                    "runs:/test/model",
                    str(directory),
                )

        load_model.assert_called_once_with("runs:/test/model")
        np.testing.assert_array_equal(labels, np.array([0, 1]))
        np.testing.assert_array_equal(predictions, np.array([0, 1]))


class MlflowUtilityTests(unittest.TestCase):
    def test_configure_mlflow_uses_explicit_tracking_uri(self):
        with patch("mlflow_utils.mlflow.set_tracking_uri") as set_uri, \
                patch("mlflow_utils.mlflow.set_experiment") as set_experiment:
            configure_mlflow("test-experiment", "sqlite:///test.db")

        set_uri.assert_called_once_with("sqlite:///test.db")
        set_experiment.assert_called_once_with("test-experiment")

    def test_log_parameters_stringifies_complex_values(self):
        with patch("mlflow_utils.mlflow.log_params") as log_params:
            log_parameters({"number": 1, "values": [1, 2]})

        log_params.assert_called_once_with({
            "number": 1,
            "values": "[1, 2]",
        })

    def test_log_files_requires_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.txt"
            artifact.write_text("artifact", encoding="utf-8")
            with patch("mlflow_utils.mlflow.log_artifact") as log_artifact:
                log_files([artifact])
            log_artifact.assert_called_once_with(str(artifact))

            with self.assertRaises(FileNotFoundError):
                log_files([Path(directory) / "missing.txt"])


class AppTests(unittest.TestCase):
    def test_pipeline_paths_are_resolved_consistently(self):
        paths = PipelinePaths.from_dataset_root(".")
        self.assertTrue(paths.dataset_root.is_absolute())
        self.assertEqual(paths.raw_csv.name, "bace_processed.csv")
        self.assertEqual(paths.split_file.name, "train_test_split.npz")

    def test_parser_supports_all_models_and_legacy_model(self):
        parser = build_parser()
        all_models = parser.parse_args(["gnn", "--all-models"])
        legacy = parser.parse_args(["gnn", "--model", "GAT"])
        self.assertTrue(all_models.all_models)
        self.assertEqual(legacy.model, "GAT")


class TrainingMetricTests(unittest.TestCase):
    def test_classification_metrics_contains_expected_values(self):
        metrics = classification_metrics(
            [0, 0, 1, 1],
            [0.1, 0.2, 0.8, 0.9],
            [0, 0, 1, 1],
        )
        self.assertEqual(metrics["Accuracy"], 1.0)
        self.assertEqual(metrics["Balanced accuracy"], 1.0)
        self.assertEqual(metrics["Inhibitor F1"], 1.0)


if __name__ == "__main__":
    unittest.main()
