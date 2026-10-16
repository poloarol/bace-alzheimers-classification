"""Audit BACE labels, split partitions, and processed graph counts.

Read-only: nothing here trains a model or touches predictions. The report
documents what each partition contains before any comparison is run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from dataset import split_file_signature

SPLITS = ("train", "validation", "calibration", "test")
# MoleculeNet BACE "Class": 1 = inhibitor, 0 = non-inhibitor. It is a source label,
# not a strict pIC50 cutoff, so disagreements with pIC50 >= 7 are reported, not asserted.
INHIBITOR_PIC50_THRESHOLD = 7.0


def audit_label_semantics(raw_csv: Path, processed_csv: Path) -> dict[str, Any]:
    processed = pd.read_csv(processed_csv)
    result: dict[str, Any] = {
        "class_values": sorted(int(v) for v in processed["Class"].unique()),
        "class_1_meaning": "inhibitor (source-provided MoleculeNet label; pIC50 is mostly >= 7)",
        "class_counts": {
            str(k): int(v) for k, v in processed["Class"].value_counts().sort_index().items()
        },
        "n_molecules": int(len(processed)),
        "duplicate_smiles": int(processed["mol"].duplicated().sum()),
    }

    if raw_csv.is_file():
        raw = pd.read_csv(raw_csv)
        if {"Class", "pIC50", "mol"} <= set(raw.columns):
            derived = (raw["pIC50"] >= INHIBITOR_PIC50_THRESHOLD).astype(int)
            mismatch = derived != raw["Class"]
            result["class_vs_pic50_threshold"] = {
                "class_0_with_pic50_at_least_7": int(((raw["Class"] == 0) & (derived == 1)).sum()),
                "class_1_with_pic50_below_7": int(((raw["Class"] == 1) & (derived == 0)).sum()),
                "total_disagreements": int(mismatch.sum()),
            }
            result["raw_pic50_by_class"] = {
                str(label): {
                    "min": float(group.min()),
                    "max": float(group.max()),
                }
                for label, group in raw.groupby("Class")["pIC50"]
            }
            result["processed_labels_match_raw"] = bool(
                len(raw) == len(processed)
                and (raw["mol"].to_numpy() == processed["mol"].to_numpy()).all()
                and (raw["Class"].to_numpy() == processed["Class"].to_numpy()).all()
            )
    return result


def _count_valid_graphs(smiles: pd.Series) -> int:
    RDLogger.DisableLog("rdApp.*")
    return int(sum(Chem.MolFromSmiles(str(s).strip()) is not None for s in smiles))


def audit_splits(dataset_root: Path, filename: str = "bace_processed.csv") -> dict[str, Any]:
    frame = pd.read_csv(dataset_root / "raw" / filename)
    split_path = dataset_root / "clean" / "train_test_split.npz"
    signature = split_file_signature(split_path)

    with np.load(split_path) as archive:
        indices = {name: archive[f"{name}_idx"] for name in SPLITS}

    all_indices = np.concatenate(list(indices.values()))
    report: dict[str, Any] = {
        "split_file": str(split_path),
        "split_signature": signature,
        "total_assigned": int(len(all_indices)),
        "unique_assigned": int(len(np.unique(all_indices))),
        "partitions_disjoint": bool(len(all_indices) == len(np.unique(all_indices))),
        "covers_all_molecules": bool(len(np.unique(all_indices)) == len(frame)),
        "splits": {},
    }

    processed_dir = dataset_root / "processed"
    for name, idx in indices.items():
        subset = frame.iloc[idx]
        counts = subset["Class"].value_counts().sort_index()
        marker = processed_dir / f"{name}_complete_{signature}.txt"
        processed_graphs = int(marker.read_text().strip()) if marker.is_file() else None
        expected_graphs = _count_valid_graphs(subset["mol"])
        report["splits"][name] = {
            "molecules": int(len(idx)),
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "inhibitor_fraction": float((subset["Class"] == 1).mean()),
            "valid_smiles": expected_graphs,
            "invalid_smiles": int(len(idx) - expected_graphs),
            "processed_graphs": processed_graphs,
            "graphs_match_valid_smiles": (
                None if processed_graphs is None else processed_graphs == expected_graphs
            ),
        }
    return report


def build_report(dataset_root: Path, filename: str = "bace_processed.csv") -> dict[str, Any]:
    root = Path(dataset_root)
    return {
        "labels": audit_label_semantics(root / "raw" / "bace.csv", root / "raw" / filename),
        "partitions": audit_splits(root, filename),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BACE labels and split counts.")
    parser.add_argument("--dataset-path", default="data/MoleculeNet/bace")
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args()

    report = build_report(Path(args.dataset_path))
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

# Example usage:
# python src/audit_splits.py --dataset-path data/MoleculeNet/bace --output data/MoleculeNet/bace/output/split_audit.json
