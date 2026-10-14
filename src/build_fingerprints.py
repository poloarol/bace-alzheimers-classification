import argparse
import os

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator


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
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the input CSV file
    df = pd.read_csv(args.input_csv)
    
    # Load the train/test split
    split = np.load(args.train_test_split_npz)
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    
    train_fp = np.array([build_morgan_fingerprint(smiles, radius=args.radius, n_bits=args.n_bits) for smiles in df["mol"].iloc[train_idx]])
    test_fp = np.array([build_morgan_fingerprint(smiles, radius=args.radius, n_bits=args.n_bits) for smiles in df["mol"].iloc[test_idx]])
    
    train_labels = df["Class"].iloc[train_idx].values
    test_labels = df["Class"].iloc[test_idx].values

    # Save the fingerprints
    pd.DataFrame(train_fp).to_csv(os.path.join(args.output_dir, "train_fingerprints.csv"), index=False)
    pd.DataFrame(test_fp).to_csv(os.path.join(args.output_dir, "test_fingerprints.csv"), index=False)
    
    pd.DataFrame(train_labels).to_csv(os.path.join(args.output_dir, "train_labels.csv"), index=False)
    pd.DataFrame(test_labels).to_csv(os.path.join(args.output_dir, "test_labels.csv"), index=False)


# Example usage:
# python src/build_fingerprints.py --input_csv data/molecules.csv --train_test_split_npz data/split.npz --output_dir data/fingerprints --radius 2 --n_bits 2048