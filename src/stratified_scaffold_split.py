import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import mlflow

from mlflow_utils import configure_mlflow, log_parameters

def stratified_scaffold_split(
    smiles,
    labels,
    train_size=0.8,
    test_size=0.2,
    random_state=42,
    size_weight=1.0,
    balance_weight=1.0,
    n_trials=500,
):
    """
    Greedy scaffold-group split optimized for molecule count and
    molecule-level class balance.

    All molecules sharing a Bemis-Murcko scaffold stay together.

    Supports binary classification with labels encoded as 0 and 1.

    Returns:
        train_idx, test_idx: original sample indices.
    """
    smiles = list(smiles)
    y = np.asarray(labels).reshape(-1)

    if len(smiles) != len(y):
        raise ValueError("smiles and labels must have equal lengths.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be binary and encoded as 0/1.")
    if not np.isclose(train_size + test_size, 1.0):
        raise ValueError("train_size and test_size must sum to 1.")
    if train_size <= 0 or test_size <= 0:
        raise ValueError("Both split sizes must be positive.")

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
    if len(groups) < 2:
        raise ValueError("At least two distinct scaffold groups are required.")

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

    target_test_n = test_size * n
    target_test_pos = test_size * total_positive
    target_test_neg = test_size * total_negative

    rng = np.random.default_rng(random_state)
    best_test_groups = None
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

        test_group_ids = []
        current = np.zeros(2, dtype=float)  # total, positives

        for j in order:
            size, positives = group_stats[j]
            negatives = size - positives

            # Evaluate adding this group to test versus train.
            candidate = current + [size, positives]

            # Normalize errors so size and class balance contribute
            # on comparable scales.
            candidate_neg = candidate[0] - candidate[1]

            test_size_error = (
                (candidate[0] - target_test_n) / max(n, 1)
            ) ** 2

            test_pos_error = (
                (candidate[1] - target_test_pos)
                / max(total_positive, 1)
            ) ** 2

            test_neg_error = (
                (candidate_neg - target_test_neg)
                / max(total_negative, 1)
            ) ** 2

            # Also consider the assignment if this group stays in train.
            current_neg = current[0] - current[1]

            current_score = (
                size_weight * ((current[0] - target_test_n) / n) ** 2
                + balance_weight * (
                    ((current[1] - target_test_pos)
                     / total_positive) ** 2
                    + ((current_neg - target_test_neg)
                       / total_negative) ** 2
                )
            )

            candidate_score = (
                size_weight * test_size_error
                + balance_weight * (test_pos_error + test_neg_error)
            )

            # Greedy choice: add if it improves the objective.
            if candidate_score < current_score:
                current = candidate
                test_group_ids.append(j)

        # Ensure neither split is empty.
        if not test_group_ids or len(test_group_ids) == len(groups):
            continue

        test_indices = [
            i for j in test_group_ids for i in groups[j]
        ]
        test_y = y[test_indices]

        # Final objective uses actual molecule counts and class balance.
        n_test = len(test_indices)
        n_test_pos = int(test_y.sum())
        n_test_neg = n_test - n_test_pos

        score = (
            size_weight * ((n_test - target_test_n) / n) ** 2
            + balance_weight * (
                ((n_test_pos - target_test_pos)
                 / total_positive) ** 2
                + ((n_test_neg - target_test_neg)
                   / total_negative) ** 2
            )
        )

        if score < best_score:
            best_score = score
            best_test_groups = set(test_group_ids)

    if best_test_groups is None:
        raise ValueError(
            "Could not find a valid split. Try increasing n_trials."
        )

    test_idx = np.array(sorted(
        i for j in best_test_groups for i in groups[j]
    ), dtype=int)

    train_idx = np.array(sorted(
        i for j, group in enumerate(groups)
        if j not in best_test_groups
        for i in group
    ), dtype=int)

    return train_idx, test_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified scaffold split")
    parser.add_argument("--input", type=str, required=True, help="Path to the SMILES file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the split indices")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split")
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
            "test_size": args.test_size,
            "n_trials": args.n_trials,
            "samples": len(df),
        })
        train_idx, test_idx = stratified_scaffold_split(
            smiles, labels, test_size=args.test_size, n_trials=args.n_trials
        )

        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        np.savez(args.output, train_idx=train_idx, test_idx=test_idx)
        mlflow.log_metrics({
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "train_positive_rate": float(labels[train_idx].mean()),
            "test_positive_rate": float(labels[test_idx].mean()),
        })
        mlflow.log_artifact(args.output)
    