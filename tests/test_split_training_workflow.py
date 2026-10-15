import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from dataset import split_file_signature
from stratified_scaffold_split import (
    SPLIT_NAMES,
    stratified_scaffold_split,
    validate_split_indices,
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


if __name__ == "__main__":
    unittest.main()
