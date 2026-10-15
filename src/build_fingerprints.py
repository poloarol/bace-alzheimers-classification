import argparse
import os

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator

import mlflow
from mlflow_utils import configure_mlflow, log_parameters


def build_morgan_fingerprint(smiles: str, 
                             radius: int = 2, 
                             n_bits: int = 2048) -> np.ndarray:   
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits, includeChirality=True
    ).GetFingerprint(mol)
    array = np.zeros((n_bits,), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fingerprint, array)
    return array

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Morgan fingerprints from a CSV file containing SMILES strings.")
    parser.add_argument("--input_csv", required=True, type=str, help="Path to the input CSV file containing SMILES strings.")
    parser.add_argument("--train_test_split_npz", required=True, type=str, help="Path to the train/test split NPZ file.")
    parser.add_argument("--output_dir", required=True, type=str, help="Path to the output directory where fingerprints will be saved.")
    parser.add_argument("--radius", type=int, default=2, help="Radius of the Morgan fingerprint.")
    parser.add_argument("--n_bits", type=int, default=2048, help="Number of bits in the Morgan fingerprint.")
    parser.add_argument("--mlflow-experiment", default="bace-preprocessing", help="MLflow experiment name.")
    parser.add_argument("--mlflow-tracking-uri", help="MLflow tracking URI. Defaults to MLFLOW_TRACKING_URI or ./mlruns.")
    
    args = parser.parse_args()

    configure_mlflow(args.mlflow_experiment, args.mlflow_tracking_uri)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the input CSV file
    df = pd.read_csv(args.input_csv)
    
    split = np.load(args.train_test_split_npz)
    split_indices = {
        split_name: split[f"{split_name}_idx"]
        for split_name in ("train", "validation", "calibration", "test")
    }
    fingerprints = {
        split_name: np.array([
            build_morgan_fingerprint(
                smiles,
                radius=args.radius,
                n_bits=args.n_bits,
            )
            for smiles in df["mol"].iloc[indices]
        ])
        for split_name, indices in split_indices.items()
    }
    labels = {
        split_name: df["Class"].iloc[indices].values
        for split_name, indices in split_indices.items()
    }

    output_files = [
        *(os.path.join(args.output_dir, f"{split_name}_fingerprints.csv")
          for split_name in split_indices),
        *(os.path.join(args.output_dir, f"{split_name}_labels.csv")
          for split_name in split_indices),
    ]
    with mlflow.start_run(run_name="build-morgan-fingerprints"):
        log_parameters({
            "input_csv": args.input_csv,
            "split_file": args.train_test_split_npz,
            "output_dir": args.output_dir,
            "radius": args.radius,
            "n_bits": args.n_bits,
            **{
                f"{split_name}_samples": len(indices)
                for split_name, indices in split_indices.items()
            },
        })
        arrays = [
            *(fingerprints[split_name] for split_name in split_indices),
            *(labels[split_name] for split_name in split_indices),
        ]
        for array, output_file in zip(arrays, output_files):
            pd.DataFrame(array).to_csv(output_file, index=False)
        mlflow.log_artifacts(args.output_dir, artifact_path="fingerprints")


# Example usage:
# python src/build_fingerprints.py --input_csv data/molecules.csv --train_test_split_npz data/split.npz --output_dir data/fingerprints --radius 2 --n_bits 2048