#!/usr/bin/env python
"""
Predict Thyroid Receptor (TR) binding for CHDS chemicals.

Tests our TR model on DDT, PCBs, and other CHDS exposome chemicals
that were studied in the CHDS Thyroid project.
"""

import pickle
import joblib
from pathlib import Path
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

# Add openadmet to path
import sys
sys.path.insert(0, str(Path.home() / "openadmet-models"))

from openadmet.chds.data import CHDSExposomeLoader


def compute_fingerprint(smiles, radius=2, n_bits=2048):
    """Compute Morgan fingerprint for a single SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    return None


def load_tr_model(model_dir):
    """Load trained TR model and metadata."""
    model_dir = Path(model_dir)

    # Model is saved with joblib (via OpenADMET's PickleableModelBase.save)
    model = joblib.load(model_dir / "tr_model.pkl")

    with open(model_dir / "tr_model_metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)

    return model, metadata


def main():
    print("=" * 70)
    print("TR Binding Predictions for CHDS Chemicals")
    print("=" * 70)

    # Load model
    model_dir = Path.home() / "openadmet-models" / "models" / "tr_model"
    model, metadata = load_tr_model(model_dir)

    print(f"\nModel info:")
    print(f"  Target: {metadata['target']}")
    print(f"  Training compounds: {metadata['n_compounds']}")
    print(f"  Test accuracy: {metadata['test_accuracy']:.3f}")
    print(f"  Test F1: {metadata['test_f1']:.3f}")

    # Load CHDS chemicals
    loader = CHDSExposomeLoader()
    chemicals = loader.get_all_chemicals()

    print(f"\nPredicting TR binding for {len(chemicals)} CHDS chemicals...")
    print("-" * 70)

    # Predict for each chemical
    results = []
    for chem in chemicals:
        fp = compute_fingerprint(chem.smiles)
        if fp is not None:
            # Get probability of being active (class 1)
            proba = model.predict_proba(fp.reshape(1, -1))[0]
            pred_class = model.predict(fp.reshape(1, -1))[0]

            results.append({
                'name': chem.name,
                'smiles': chem.smiles,
                'assay': chem.chds_assay,
                'tr_prob': proba[1] if len(proba) > 1 else proba[0],
                'tr_active': pred_class == 1,
                'disease_assoc': chem.disease_associations[:2] if chem.disease_associations else []
            })

    # Sort by TR probability (most likely active first)
    results.sort(key=lambda x: x['tr_prob'], reverse=True)

    # Display results
    print(f"\n{'Chemical':<30} {'TR Prob':>8} {'Active':>7}  Disease Associations")
    print("-" * 80)

    for r in results:
        active_str = "YES" if r['tr_active'] else "no"
        assoc = ', '.join(r['disease_assoc']) if r['disease_assoc'] else '-'
        print(f"{r['name']:<30} {r['tr_prob']:>8.3f} {active_str:>7}  {assoc}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    active_count = sum(1 for r in results if r['tr_active'])
    print(f"Chemicals predicted as TR-active: {active_count}/{len(results)}")

    # Highlight organochlorines (DDT, PCBs) - key CHDS thyroid study chemicals
    print("\n" + "-" * 70)
    print("CHDS Thyroid Study Chemicals (DDT/PCBs):")
    print("-" * 70)

    oc_results = [r for r in results if
                  'DDT' in r['name'].upper() or 'PCB' in r['name'].upper() or
                  'DDE' in r['name'].upper()]

    if oc_results:
        for r in sorted(oc_results, key=lambda x: x['tr_prob'], reverse=True):
            status = "ACTIVE" if r['tr_active'] else "inactive"
            print(f"  {r['name']:<25} TR prob: {r['tr_prob']:.3f} ({status})")
    else:
        print("  (No DDT/PCBs found in chemical list)")

    # Also show PFAS which CHDS linked to thyroid disruption
    print("\n" + "-" * 70)
    print("PFAS (also linked to thyroid in CHDS):")
    print("-" * 70)

    pfas_results = [r for r in results if 'PF' in r['name'].upper()]

    if pfas_results:
        for r in sorted(pfas_results, key=lambda x: x['tr_prob'], reverse=True):
            status = "ACTIVE" if r['tr_active'] else "inactive"
            print(f"  {r['name']:<25} TR prob: {r['tr_prob']:.3f} ({status})")


if __name__ == "__main__":
    main()
