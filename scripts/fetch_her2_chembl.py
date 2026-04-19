#!/usr/bin/env python
"""
Fetch HER2 (ERBB2) bioactivity data from ChEMBL.

HER2/ERBB2 (CHEMBL1824) is a key target for:
- Breast cancer (HER2+ subtype)
- CHDS research on gestational exposures and cancer risk

This script fetches:
- IC50, Ki, EC50 data for ERBB2
- Structures (SMILES) for each compound
- Creates a dataset for model training
"""

import csv
import json
import urllib.request
import urllib.parse
import time
from pathlib import Path


def fetch_chembl_activities(target_chembl_id, activity_types=None, limit=5000):
    """
    Fetch bioactivity data from ChEMBL for a target.

    Args:
        target_chembl_id: e.g., "CHEMBL1824" for ERBB2/HER2
        activity_types: List of activity types (IC50, Ki, etc.)
        limit: Maximum number of results

    Returns:
        List of activity records
    """
    if activity_types is None:
        activity_types = ["IC50", "Ki", "EC50", "Kd"]

    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

    all_activities = []

    for activity_type in activity_types:
        offset = 0
        batch_size = 1000

        while len(all_activities) < limit:
            params = {
                "target_chembl_id": target_chembl_id,
                "standard_type": activity_type,
                "limit": batch_size,
                "offset": offset,
            }

            url = f"{base_url}?{urllib.parse.urlencode(params)}"

            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    data = json.loads(response.read().decode('utf-8'))

                activities = data.get("activities", [])
                if not activities:
                    break

                all_activities.extend(activities)
                print(f"  Fetched {len(activities)} {activity_type} records (total: {len(all_activities)})")

                offset += batch_size
                time.sleep(0.3)  # Rate limiting

            except Exception as e:
                print(f"  Error fetching {activity_type}: {e}")
                break

    return all_activities


def get_compound_smiles(molecule_chembl_id):
    """Get SMILES for a compound from ChEMBL."""
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{molecule_chembl_id}.json"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            structures = data.get("molecule_structures")
            if structures:
                return structures.get("canonical_smiles")
    except Exception:
        pass

    return None


def process_activities(activities, output_path):
    """Process activities and save to CSV."""
    print(f"\nProcessing {len(activities)} activity records...")

    # Group by compound
    compound_activities = {}
    for act in activities:
        mol_id = act.get("molecule_chembl_id")
        if not mol_id:
            continue

        if mol_id not in compound_activities:
            compound_activities[mol_id] = {
                "molecule_chembl_id": mol_id,
                "pref_name": act.get("molecule_pref_name"),
                "activities": []
            }

        # Parse activity value
        value = act.get("standard_value")
        units = act.get("standard_units")
        activity_type = act.get("standard_type")
        relation = act.get("standard_relation")

        if value and units == "nM":
            compound_activities[mol_id]["activities"].append({
                "type": activity_type,
                "value_nM": float(value),
                "relation": relation,
                "assay_chembl_id": act.get("assay_chembl_id"),
            })

    print(f"Found {len(compound_activities)} unique compounds")

    # Get SMILES for each compound
    print("Fetching SMILES for compounds...")
    results = []
    for i, (mol_id, data) in enumerate(compound_activities.items()):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(compound_activities)}")

        smiles = get_compound_smiles(mol_id)
        if not smiles:
            continue

        # Calculate aggregate activity (use most potent IC50)
        ic50_values = [a["value_nM"] for a in data["activities"]
                      if a["type"] == "IC50" and a["relation"] in ["=", "<"]]

        if ic50_values:
            best_ic50 = min(ic50_values)
            # Convert to pIC50 for ML
            import math
            pIC50 = -math.log10(best_ic50 * 1e-9) if best_ic50 > 0 else None
        else:
            best_ic50 = None
            pIC50 = None

        results.append({
            "smiles": smiles,
            "molecule_chembl_id": mol_id,
            "pref_name": data["pref_name"],
            "ic50_nM": best_ic50,
            "pIC50": pIC50,
            "n_activities": len(data["activities"]),
        })

        time.sleep(0.1)  # Rate limiting

    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        fieldnames = ["smiles", "molecule_chembl_id", "pref_name", "ic50_nM", "pIC50", "n_activities"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([r for r in results if r.get("pIC50")])

    valid = sum(1 for r in results if r.get("pIC50"))
    print(f"\nSaved {valid} compounds with pIC50 to {output_path}")

    return results


if __name__ == "__main__":
    # HER2/ERBB2 target
    target_id = "CHEMBL1824"  # ERBB2

    print(f"Fetching HER2 (ERBB2) bioactivity data from ChEMBL...")
    print(f"Target: {target_id}")
    print("=" * 60)

    activities = fetch_chembl_activities(target_id)

    output_path = Path.home() / "openadmet-models" / "data" / "chembl" / "chembl_her2_activities.csv"

    process_activities(activities, output_path)

    print("\n" + "=" * 60)
    print("HER2 data extraction complete!")
    print(f"Output: {output_path}")
