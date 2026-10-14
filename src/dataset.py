import argparse
import os

import numpy as np
import deepchem as dc
import pandas as pd
import torch
import torch_geometric

from rdkit import Chem
from torch_geometric.data import Dataset
from tqdm import tqdm


print("Dataset module loaded successfully.")
print("Torch version:", torch.__version__)
print("Torch Geometric version:", torch_geometric.__version__)
print("DeepChem version:", dc.__version__)
print("CUDA available:", torch.cuda.is_available())

class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        self.filename = filename
        self.test = test

        super().__init__(
            root,
            transform,
            pre_transform
        )
        

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        if self.test:
            return "test_complete.txt"
        else:
            return "train_complete.txt"

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0]).reset_index(drop=True)

        featurizer = dc.feat.MolGraphConvFeaturizer(
            use_edges=True,
            use_chirality=True,
            use_partial_charge=True
        )

        os.makedirs(self.processed_dir, exist_ok=True)

        processed_index = 0
        skipped = 0
        
        split = np.load(os.path.join(self.root, "clean/train_test_split.npz"))
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        
        if self.test:
            df = df.loc[test_idx].reset_index(drop=True)
        else:
            df = df.loc[train_idx].reset_index(drop=True)

        for csv_index, row in tqdm(
            df.iterrows(),
            total=df.shape[0],
            desc="Processing molecules"
        ):

            smiles = str(row["mol"]).strip()

            try:
                # Convert SMILES to molecule
                mol = Chem.MolFromSmiles(smiles)

                if mol is None:
                    print(
                        f"\nSkipping invalid SMILES "
                        f"(CSV index {csv_index}): {smiles}"
                    )
                    skipped += 1
                    continue

                # Featurize
                features = featurizer._featurize(mol)

                # Convert to PyG
                data = features.to_pyg_graph()

                # Label
                data.y = self._get_label(
                    row["Class"]
                )

                # Store SMILES
                data.smiles = smiles

                # Store original CSV index for reference
                data.original_index = csv_index

                # IMPORTANT:
                # Use a contiguous processed index.
                if self.test:
                    filename = f"data_test_{processed_index}.pt"
                else:
                    filename = f"data_{processed_index}.pt"

                filepath = os.path.join(
                    self.processed_dir,
                    filename
                )

                torch.save(data, filepath)

                processed_index += 1

            except Exception as e:

                print("\n" + "=" * 70)
                print("ERROR PROCESSING MOLECULE")
                print(f"CSV index: {csv_index}")
                print(f"Processed index: {processed_index}")
                print(f"SMILES: {smiles}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error: {e}")
                print("=" * 70)

                raise

    def len(self):
        marker = ("test_complete.txt" if self.test else "train_complete.txt")
        marker_path = os.path.join(self.processed_dir, marker)

        if not os.path.exists(marker_path):
            return 0

        with open(marker_path, "r") as f:
            return int(f.read().strip())

    def get(self, idx):
        if self.test:
            filename = f"data_test_{idx}.pt"
        else:
            filename = f"data_{idx}.pt"

        filepath = os.path.join(
            self.processed_dir,
            filename
        )

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Processed graph does not exist: {filepath}"
            )

        return torch.load(
            filepath,
            weights_only=False
        )
    
    @staticmethod
    def _get_label(label):
        return torch.tensor(
            [label],
            dtype=torch.int64
        )

def str2bool(value):
    if isinstance(value, bool):
        return value

    if value.lower() in ("true", "1", "yes", "y"):
        return True

    if value.lower() in ("false", "0", "no", "n"):
        return False

    raise argparse.ArgumentTypeError(
        "Expected True/False"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BACE dataset")
    parser.add_argument("--root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--filename", type=str, required=True, help="molecule CSV filename")
    parser.add_argument("--test", type=str2bool, default=False, help="Whether to process the test dataset")
    args = parser.parse_args()

    dataset = MoleculeDataset(
        root=args.root,
        filename=args.filename,
        test=args.test
    )
    
    print("Number of molecules:", len(dataset))
    