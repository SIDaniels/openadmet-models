#!/usr/bin/env python
"""
Add SMILES to ToxCast TR assay data using PubChem API.

Uses CAS numbers to look up SMILES via PubChem PUG-REST.
"""

import csv
import time
import urllib.request
import urllib.error
from pathlib import Path


def get_smiles_from_cas(cas_number, retries=3):
    """Get SMILES from PubChem using CAS number."""
    if not cas_number or cas_number == "None":
        return None

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas_number}/property/CanonicalSMILES/TXT"

    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                smiles = response.read().decode('utf-8').strip()
                return smiles
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(0.5)
        except Exception:
            time.sleep(0.5)

    return None


def process_tr_file(input_path, output_path):
    """Add SMILES to a ToxCast TR CSV file."""
    print(f"Processing {input_path}...")

    # Read all rows
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Get unique CAS numbers
    cas_to_smiles = {}
    unique_cas = set(row['casn'] for row in rows if row.get('casn'))

    print(f"Looking up SMILES for {len(unique_cas)} unique CAS numbers...")

    for i, cas in enumerate(unique_cas):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(unique_cas)}")

        smiles = get_smiles_from_cas(cas)
        cas_to_smiles[cas] = smiles
        time.sleep(0.2)  # Rate limiting

    found = sum(1 for s in cas_to_smiles.values() if s)
    print(f"Found SMILES for {found}/{len(unique_cas)} chemicals")

    # Write output with SMILES
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['smiles', 'dsstox_substance_id', 'casn', 'chnm', 'aeid', 'hitc', 'modl', 'coff']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        written = 0
        for row in rows:
            smiles = cas_to_smiles.get(row['casn'])
            if smiles:
                writer.writerow({
                    'smiles': smiles,
                    'dsstox_substance_id': row['dsstox_substance_id'],
                    'casn': row['casn'],
                    'chnm': row['chnm'],
                    'aeid': row['aeid'],
                    'hitc': row['hitc'],
                    'modl': row['modl'],
                    'coff': row['coff']
                })
                written += 1

    print(f"Wrote {written} entries to {output_path}")
    return written


if __name__ == "__main__":
    data_dir = Path.home() / "openadmet-models" / "data" / "toxcast"

    # Process TR assays
    tr_input = data_dir / "toxcast_tr_assays.csv"
    tr_output = data_dir / "toxcast_tr_assays_smiles.csv"
    process_tr_file(tr_input, tr_output)
