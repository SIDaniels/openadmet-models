#!/usr/bin/env python
"""
Train Thyroid Receptor (TR) binding model using OpenADMET.

Uses ToxCast TR assay data (ATG_THRa1_TRANS, ATG_THRb_TRANS2) with hitcall
as binary classification target.

Trains a LightGBM model with Morgan fingerprint features.
"""

import csv
import pickle
from pathlib import Path
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from openadmet.models.architecture.lgbm import LGBMClassifierModel


def load_tr_data(tr_csv_path):
    """Load TR assay data with SMILES."""
    print(f"Loading TR data from {tr_csv_path}...")

    data = []
    with open(tr_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row.get('smiles')
            hitc = row.get('hitc')

            if smiles and hitc:
                try:
                    hitc_val = float(hitc)
                    # Binary classification: hitc > 0.5 = active
                    label = 1 if hitc_val > 0.5 else 0
                    data.append({
                        'smiles': smiles,
                        'hitc': hitc_val,
                        'label': label,
                        'casn': row.get('casn'),
                        'chnm': row.get('chnm'),
                        'aeid': row.get('aeid'),
                    })
                except ValueError:
                    pass

    print(f"Loaded {len(data)} compounds with SMILES")

    # Aggregate by compound (multiple assays per compound)
    compound_data = {}
    for entry in data:
        smiles = entry['smiles']
        if smiles not in compound_data:
            compound_data[smiles] = {
                'smiles': smiles,
                'casn': entry['casn'],
                'chnm': entry['chnm'],
                'labels': [],
            }
        compound_data[smiles]['labels'].append(entry['label'])

    # Consensus label: active if majority of assays are active
    for smiles, comp in compound_data.items():
        comp['label'] = 1 if sum(comp['labels']) > len(comp['labels']) / 2 else 0

    print(f"Aggregated to {len(compound_data)} unique compounds")
    actives = sum(1 for c in compound_data.values() if c['label'] == 1)
    print(f"  Active: {actives}, Inactive: {len(compound_data) - actives}")

    return list(compound_data.values())


def compute_fingerprints(smiles_list, radius=2, n_bits=2048):
    """Compute Morgan fingerprints for SMILES list."""
    print(f"Computing Morgan fingerprints (radius={radius}, bits={n_bits})...")

    fps = []
    valid_idx = []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(np.array(fp))
            valid_idx.append(i)

    print(f"Successfully featurized {len(valid_idx)}/{len(smiles_list)} compounds")
    return np.array(fps), valid_idx


def train_tr_model(tr_data, output_dir):
    """Train TR binding model."""
    print("\nTraining TR binding model...")

    smiles_list = [d['smiles'] for d in tr_data]
    labels = [d['label'] for d in tr_data]

    # Compute features
    X, valid_idx = compute_fingerprints(smiles_list)
    y = np.array([labels[i] for i in valid_idx])

    # Train/test split (80/20)
    n_train = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Training set: {len(X_train)} compounds")
    print(f"Test set: {len(X_test)} compounds")

    # Train LightGBM classifier
    model = LGBMClassifierModel(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
    )
    model.build()
    model.train(X_train, y_train)

    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_acc = np.mean(train_preds == y_train)
    test_acc = np.mean(test_preds == y_test)

    # Additional metrics
    test_positives = y_test == 1
    test_negatives = y_test == 0
    pred_positives = test_preds == 1

    true_positives = np.sum(test_positives & pred_positives)
    false_positives = np.sum(test_negatives & pred_positives)
    false_negatives = np.sum(test_positives & ~pred_positives)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nResults:")
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy: {test_acc:.3f}")
    print(f"  Test precision: {precision:.3f}")
    print(f"  Test recall: {recall:.3f}")
    print(f"  Test F1: {f1:.3f}")

    # Save model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "tr_model.pkl"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Save metadata
    metadata = {
        'target': 'Thyroid Receptor',
        'data_source': 'ToxCast',
        'n_compounds': len(tr_data),
        'n_features': X.shape[1],
        'feature_type': 'Morgan fingerprint',
        'model_type': 'LightGBM Classifier',
        'test_accuracy': test_acc,
        'test_f1': f1,
    }

    with open(output_dir / "tr_model_metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)

    return model, metadata


if __name__ == "__main__":
    # Paths
    data_dir = Path.home() / "openadmet-models" / "data" / "toxcast"
    tr_csv = data_dir / "toxcast_tr_assays_smiles.csv"
    output_dir = Path.home() / "openadmet-models" / "models" / "tr_model"

    if not tr_csv.exists():
        print(f"ERROR: {tr_csv} not found!")
        print("Run add_smiles_to_toxcast_tr.py first to get SMILES.")
        exit(1)

    # Load data
    tr_data = load_tr_data(tr_csv)

    # Train model
    model, metadata = train_tr_model(tr_data, output_dir)

    print("\n" + "=" * 60)
    print("TR Model Training Complete!")
    print(f"Model: {output_dir / 'tr_model.pkl'}")
    print(f"Metadata: {output_dir / 'tr_model_metadata.pkl'}")
