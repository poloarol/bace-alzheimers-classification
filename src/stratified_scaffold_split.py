import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import mlflow

from mlflow_utils import configure_mlflow, log_parameters

SPLIT_NAMES = ("train", "validation", "calibration", "test")


def validate_split_indices(smiles, labels, split_indices):
    """Validate split coverage, class representation, and scaffold isolation."""
    smiles = list(smiles)
    y = np.asarray(labels).reshape(-1)
    if set(split_indices) != set(SPLIT_NAMES):
        raise ValueError(
            f"Split indices must contain exactly {SPLIT_NAMES}."
        )

    seen_indices = []
    scaffold_to_split = {}
    for split_name in SPLIT_NAMES:
        indices = np.asarray(split_indices[split_name], dtype=int)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError(f"{split_name} split must be a non-empty 1D array.")
        if np.any(indices < 0) or np.any(indices >= len(smiles)):
            raise ValueError(f"{split_name} split contains invalid indices.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{split_name} split contains duplicate indices.")

        split_labels = y[indices]
        if set(np.unique(split_labels)) != {0, 1}:
            raise ValueError(
                f"{split_name} split must contain both binary classes."
            )

        for index in indices:
            mol = Chem.MolFromSmiles(smiles[index])
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol,
                includeChirality=False,
            )
            previous_split = scaffold_to_split.setdefault(scaffold, split_name)
            if previous_split != split_name:
                raise ValueError(
                    f"Scaffold {scaffold!r} appears in both "
                    f"{previous_split} and {split_name} splits."
                )
        seen_indices.extend(indices.tolist())

    if len(seen_indices) != len(smiles) or set(seen_indices) != set(range(len(smiles))):
        raise ValueError("Splits must cover every sample exactly once.")


def stratified_scaffold_split(
    smiles,
    labels,
    train_size=0.7,
    test_size=0.1,
    random_state=42,
    size_weight=1.0,
    balance_weight=1.0,
    n_trials=500,
    validation_size=0.1,
    calibration_size=0.1,
):
    """
    Greedy four-way scaffold-group split optimized for molecule count and
    molecule-level class balance in train, validation, calibration, and test.

    All molecules sharing a Bemis-Murcko scaffold stay together.

    Supports binary classification with labels encoded as 0 and 1.

    Returns:
        train_idx, validation_idx, calibration_idx, test_idx: original
        sample indices.
    """
    smiles = list(smiles)
    y = np.asarray(labels).reshape(-1)

    if len(smiles) != len(y):
        raise ValueError("smiles and labels must have equal lengths.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary and encoded as 0/1.")
    split_names = SPLIT_NAMES
    split_sizes = np.array(
        [train_size, validation_size, calibration_size, test_size],
        dtype=float,
    )
    if not np.isclose(split_sizes.sum(), 1.0):
        raise ValueError("Split sizes must sum to 1.")
    if np.any(split_sizes <= 0):
        raise ValueError("All split sizes must be positive.")

    # Group molecules by Bemis-Murcko scaffold.
    scaffold_to_indices = defaultdict(list)

    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smi}")

        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        )
        scaffold_to_indices[scaffold].append(i)

    groups = list(scaffold_to_indices.values())
    if len(groups) < len(split_names):
        raise ValueError("At least four distinct scaffold groups are required.")

    # Each row contains [total molecules, positive molecules].
    group_stats = np.array([
        [len(idx), int(y[idx].sum())]
        for idx in groups
    ], dtype=float)

    n = len(y)
    total_positive = float(y.sum())
    total_negative = n - total_positive

    if total_positive == 0 or total_negative == 0:
        raise ValueError("Both classes must be present.")

    target_stats = np.column_stack((
        split_sizes * n,
        split_sizes * total_positive,
        split_sizes * total_negative,
    ))

    rng = np.random.default_rng(random_state)
    best_split_groups = None
    best_score = np.inf

    # Repeated randomized greedy assignments; retain the best objective.
    for _ in range(n_trials):
        # Large groups first, with randomized tie-breaking.
        tie_break = rng.random(len(groups))
        order = sorted(
            range(len(groups)),
            key=lambda j: (
                group_stats[j, 0],
                tie_break[j],
            ),
            reverse=True,
        )

        split_group_ids = [[] for _ in split_names]
        current_stats = np.zeros((len(split_names), 3), dtype=float)

        for group_position, j in enumerate(order):
            size, positives = group_stats[j]
            negatives = size - positives

            candidate_scores = []
            empty_split_ids = [
                split_id
                for split_id, group_ids in enumerate(split_group_ids)
                if not group_ids
            ]
            remaining_groups = len(order) - group_position
            allowed_split_ids = (
                empty_split_ids
                if remaining_groups == len(empty_split_ids)
                else range(len(split_names))
            )

            for split_id in allowed_split_ids:
                candidate_stats = current_stats.copy()
                candidate_stats[split_id] += [size, positives, negatives]
                candidate_scores.append((
                    _split_score(
                        candidate_stats,
                        target_stats,
                        n,
                        total_positive,
                        total_negative,
                        size_weight,
                        balance_weight,
                    ),
                    split_id,
                ))

            _, selected_split_id = min(candidate_scores)
            current_stats[selected_split_id] += [size, positives, negatives]
            split_group_ids[selected_split_id].append(j)

        if any(not group_ids for group_ids in split_group_ids):
            continue

        score = _split_score(
            current_stats,
            target_stats,
            n,
            total_positive,
            total_negative,
            size_weight,
            balance_weight,
        )

        if score < best_score:
            best_score = score
            best_split_groups = split_group_ids

    if best_split_groups is None:
        raise ValueError(
            "Could not find a valid split. Try increasing n_trials."
        )

    split_indices = [
        np.array(sorted(
            i for j in group_ids for i in groups[j]
        ), dtype=int)
        for group_ids in best_split_groups
    ]
    result = dict(zip(split_names, split_indices))
    validate_split_indices(smiles, y, result)
    return tuple(result[split_name] for split_name in split_names)


def _split_score(
    current_stats,
    target_stats,
    total_samples,
    total_positive,
    total_negative,
    size_weight,
    balance_weight,
):
    normalized_size_error = (
        (current_stats[:, 0] - target_stats[:, 0]) / total_samples
    ) ** 2
    normalized_positive_error = (
        (current_stats[:, 1] - target_stats[:, 1]) / total_positive
    ) ** 2
    normalized_negative_error = (
        (current_stats[:, 2] - target_stats[:, 2]) / total_negative
    ) ** 2
    return (
        size_weight * normalized_size_error.sum()
        + balance_weight * (
            normalized_positive_error.sum() + normalized_negative_error.sum()
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified scaffold split")
    parser.add_argument("--input", type=str, required=True, help="Path to the SMILES file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the split indices")
    parser.add_argument("--train_size", type=float, default=0.7, help="Proportion of the dataset to include in the train split")
    parser.add_argument("--validation_size", type=float, default=0.1, help="Proportion of the dataset to include in the validation split")
    parser.add_argument("--calibration_size", type=float, default=0.1, help="Proportion of the dataset to include in the calibration split")
    parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of the dataset to include in the test split")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for finding the best split")
    parser.add_argument("--mlflow-experiment", default="bace-preprocessing", help="MLflow experiment name.")
    parser.add_argument("--mlflow-tracking-uri", help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI or ./mlruns.")
    args = parser.parse_args()

    configure_mlflow(args.mlflow_experiment, args.mlflow_tracking_uri)
    
    df = pd.read_csv(args.input)
    smiles = df["mol"].tolist()
    labels = df["Class"].values

    with mlflow.start_run(run_name="stratified-scaffold-split"):
        log_parameters({
            "input_csv": args.input,
            "output_file": args.output,
            "train_size": args.train_size,
            "validation_size": args.validation_size,
            "calibration_size": args.calibration_size,
            "test_size": args.test_size,
            "n_trials": args.n_trials,
            "samples": len(df),
        })
        train_idx, validation_idx, calibration_idx, test_idx = stratified_scaffold_split(
            smiles,
            labels,
            train_size=args.train_size,
            validation_size=args.validation_size,
            calibration_size=args.calibration_size,
            test_size=args.test_size,
            n_trials=args.n_trials,
        )

        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.savez(
            args.output,
            train_idx=train_idx,
            validation_idx=validation_idx,
            calibration_idx=calibration_idx,
            test_idx=test_idx,
        )
        with np.load(args.output) as split:
            expected_keys = {f"{name}_idx" for name in SPLIT_NAMES}
            if set(split.files) != expected_keys:
                raise ValueError(
                    f"Split file must contain exactly {sorted(expected_keys)}."
                )
            validate_split_indices(
                smiles,
                labels,
                {
                    name: split[f"{name}_idx"]
                    for name in SPLIT_NAMES
                },
            )
        mlflow.log_metrics({
            "train_samples": len(train_idx),
            "validation_samples": len(validation_idx),
            "calibration_samples": len(calibration_idx),
            "test_samples": len(test_idx),
            "train_positive_rate": float(labels[train_idx].mean()),
            "validation_positive_rate": float(labels[validation_idx].mean()),
            "calibration_positive_rate": float(labels[calibration_idx].mean()),
            "test_positive_rate": float(labels[test_idx].mean()),
        })
        mlflow.log_artifact(args.output)
    