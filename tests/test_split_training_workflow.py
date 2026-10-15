import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from dataset import split_file_signature
from stratified_scaffold_split import (
    SPLIT_NAMES,
    stratified_scaffold_split,
    validate_split_indices,
)
from app import GNN_MODELS, PipelinePaths, PipelineRunner
from train import (
    evaluate_after_model_selection,
    evaluate_for_model_selection,
)


class SplitTrainingWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.smiles = [
            "c1ccccc1", "c1ccncc1", "c1ccoc1", "c1ccsc1",
            "C1CCCCC1", "C1CCCC1", "C1CCC1", "C1CCNC1",
            "c1ccc2ccccc2c1", "c1ccc2ncccc2c1",
            "c1ccc2occc2c1", "c1ccc2sccc2c1",
            "C1CCCCC1O", "C1CCCCC1N", "C1CCCCC1C", "C1CCCCC1F",
        ]
        self.labels = np.array([0, 1] * 8)

    def test_split_file_contract_and_scaffold_isolation(self):
        split_values = stratified_scaffold_split(
            self.smiles,
            self.labels,
            n_trials=100,
        )
        split_indices = dict(zip(SPLIT_NAMES, split_values))
        validate_split_indices(
            self.smiles,
            self.labels,
            split_indices,
        )

        with tempfile.TemporaryDirectory() as directory:
            split_path = Path(directory) / "train_test_split.npz"
            np.savez(
                split_path,
                **{
                    f"{name}_idx": split_indices[name]
                    for name in SPLIT_NAMES
                },
            )
            with np.load(split_path) as split:
                self.assertEqual(
                    set(split.files),
                    {f"{name}_idx" for name in SPLIT_NAMES},
                )
                validate_split_indices(
                    self.smiles,
                    self.labels,
                    {
                        name: split[f"{name}_idx"]
                        for name in SPLIT_NAMES
                    },
                )

    def test_split_change_invalidates_graph_cache_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            split_path = Path(directory) / "train_test_split.npz"
            np.savez(
                split_path,
                **{
                    f"{name}_idx": np.array([index])
                    for index, name in enumerate(SPLIT_NAMES)
                },
            )
            original_signature = split_file_signature(split_path)
            np.savez(
                split_path,
                **{
                    f"{name}_idx": np.array([index + 1])
                    for index, name in enumerate(SPLIT_NAMES)
                },
            )
            self.assertNotEqual(
                original_signature,
                split_file_signature(split_path),
            )

    def test_dataset_cache_marker_changes_with_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean_directory = root / "clean"
            clean_directory.mkdir()
            split_path = clean_directory / "train_test_split.npz"
            np.savez(
                split_path,
                **{
                    f"{name}_idx": np.array([index])
                    for index, name in enumerate(SPLIT_NAMES)
                },
            )

            from dataset import MoleculeDataset

            dataset = MoleculeDataset.__new__(MoleculeDataset)
            dataset.root = root
            dataset.split = "validation"
            first_marker = dataset.processed_file_names

            np.savez(
                split_path,
                **{
                    f"{name}_idx": np.array([index + 1])
                    for index, name in enumerate(SPLIT_NAMES)
                },
            )
            self.assertNotEqual(first_marker, dataset.processed_file_names)

    def test_invalid_split_contracts_are_rejected(self):
        valid = dict(zip(
            SPLIT_NAMES,
            stratified_scaffold_split(self.smiles, self.labels, n_trials=100),
        ))

        missing_key = dict(valid)
        missing_key.pop("test")
        with self.assertRaises(ValueError):
            validate_split_indices(self.smiles, self.labels, missing_key)

        duplicate_index = dict(valid)
        duplicate_index["train"] = np.append(
            duplicate_index["train"],
            duplicate_index["train"][0],
        )
        with self.assertRaises(ValueError):
            validate_split_indices(self.smiles, self.labels, duplicate_index)

        missing_index = dict(valid)
        missing_index["test"] = missing_index["test"][1:]
        with self.assertRaises(ValueError):
            validate_split_indices(self.smiles, self.labels, missing_index)

        one_class = dict(valid)
        one_class["test"] = np.array([one_class["test"][0]])
        with self.assertRaises(ValueError):
            validate_split_indices(self.smiles, self.labels, one_class)

    def test_split_parameter_and_dataset_api_validation(self):
        with self.assertRaises(ValueError):
            stratified_scaffold_split(
                self.smiles,
                self.labels,
                train_size=0.8,
                validation_size=0.1,
                calibration_size=0.1,
                test_size=0.1,
            )

        from dataset import MoleculeDataset

        with self.assertRaises(ValueError):
            MoleculeDataset(
                root=tempfile.gettempdir(),
                filename="missing.csv",
                split="unknown",
            )
        with self.assertRaises(ValueError):
            MoleculeDataset(
                root=tempfile.gettempdir(),
                filename="missing.csv",
                test=True,
                split="validation",
            )

    def test_all_gnn_dispatch_preserves_model_order(self):
        paths = PipelinePaths.from_dataset_root(tempfile.gettempdir())
        runner = PipelineRunner(paths, tracking_uri=None, dry_run=True)
        with patch.object(runner, "run_gnn_training") as run_model:
            runner.run_all_gnn_training(
                epochs=1,
                hidden_channels=8,
                learning_rate=0.01,
                batch_size=4,
            )

        self.assertEqual(
            [call.args[0] for call in run_model.call_args_list],
            list(GNN_MODELS),
        )
        self.assertEqual(run_model.call_count, len(GNN_MODELS))

    def test_model_selection_evaluates_only_train_and_validation(self):
        with patch("train.evaluate", side_effect=["train", "validation"]) as evaluate:
            result = evaluate_for_model_selection(
                "model",
                "train_loader",
                "validation_loader",
            )

        self.assertEqual(result, ("train", "validation"))
        self.assertEqual(
            evaluate.call_args_list[0].args,
            ("model", "train_loader"),
        )
        self.assertEqual(
            evaluate.call_args_list[1].args,
            ("model", "validation_loader"),
        )

    def test_calibration_and_test_evaluation_is_deferred(self):
        with patch("train.evaluate", side_effect=["calibration", "test"]) as evaluate:
            result = evaluate_after_model_selection(
                "model",
                "calibration_loader",
                "test_loader",
            )

        self.assertEqual(result, ("calibration", "test"))
        self.assertEqual(
            evaluate.call_args_list[0].args,
            ("model", "calibration_loader"),
        )
        self.assertEqual(
            evaluate.call_args_list[1].args,
            ("model", "test_loader"),
        )


if __name__ == "__main__":
    unittest.main()
