#!/usr/bin/env python
"""
Extract Thyroid Receptor (TR) assay data from ToxCast invitrodb MySQL dump.

This script parses the SQL dump and extracts:
1. TR-related assay endpoints (ATG_THRa1_TRANS, ATG_THRb_TRANS2)
2. mc5 fitted dose-response results (hitc, AC50)
3. Chemical information (DTXSID for SMILES lookup)

Output: CSV with chemical info and TR activity for model training
"""

import re
import csv
import sys
from pathlib import Path

# TR-related assay endpoint IDs from assay_component_endpoint table
# ATG_THRa1_TRANS: 112, 143 (TR alpha)
# ATG_THRb_TRANS2: 936, 1369 (TR beta)
TR_AEIDS = [112, 143, 936, 1369]


def parse_insert_values(line, table_name):
    """Parse INSERT INTO ... VALUES (...) statements."""
    pattern = rf"INSERT INTO `{table_name}` VALUES "
    if not line.startswith(pattern):
        return []

    values_str = line[len(pattern):]

    rows = []
    current_row = []
    current_val = ""
    in_string = False
    escape_next = False
    paren_depth = 0

    for char in values_str:
        if escape_next:
            current_val += char
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            current_val += char
            continue

        if char == "'" and not in_string:
            in_string = True
            continue
        elif char == "'" and in_string:
            in_string = False
            continue

        if in_string:
            current_val += char
            continue

        if char == '(':
            paren_depth += 1
            if paren_depth == 1:
                current_row = []
                current_val = ""
            continue
        elif char == ')':
            paren_depth -= 1
            if paren_depth == 0:
                if current_val:
                    current_row.append(current_val if current_val != 'NULL' else None)
                rows.append(current_row)
            continue
        elif char == ',' and paren_depth == 1:
            current_row.append(current_val if current_val != 'NULL' else None)
            current_val = ""
            continue
        elif paren_depth == 1:
            current_val += char

    return rows


def extract_tr_data(mysql_dump_path, output_dir):
    """Extract TR assay data from MySQL dump."""

    print(f"Reading MySQL dump from {mysql_dump_path}...")

    # Storage for parsed data
    chemicals = {}  # chid -> {casn, chnm, dsstox_substance_id}
    mc5_tr = []     # List of mc5 rows for TR assays

    # Linking tables
    mc4_to_spid = {}  # m4id -> spid
    spid_to_chid = {} # spid -> chid

    with open(mysql_dump_path, 'r', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f):
            if line_num % 100000 == 0:
                print(f"  Processing line {line_num}...")

            line = line.strip()

            if line.startswith("INSERT INTO `chemical`"):
                rows = parse_insert_values(line, "chemical")
                for row in rows:
                    if len(row) >= 4:
                        chid = row[0]
                        casn = row[1]
                        chnm = row[2]
                        dsstox = row[3]
                        chemicals[chid] = {
                            'casn': casn,
                            'chnm': chnm,
                            'dsstox_substance_id': dsstox
                        }

            elif line.startswith("INSERT INTO `sample`"):
                rows = parse_insert_values(line, "sample")
                for row in rows:
                    if len(row) >= 3:
                        spid = row[0]
                        chid = row[1]
                        spid_to_chid[spid] = chid

            elif line.startswith("INSERT INTO `mc4`"):
                rows = parse_insert_values(line, "mc4")
                for row in rows:
                    if len(row) >= 3:
                        m4id = row[0]
                        spid = row[2]  # mc4 structure: m4id, aeid, spid, ...
                        mc4_to_spid[m4id] = spid

            elif line.startswith("INSERT INTO `mc5`"):
                rows = parse_insert_values(line, "mc5")
                for row in rows:
                    if len(row) >= 6:
                        m5id = row[0]
                        m4id = row[1]
                        aeid = int(row[2]) if row[2] else None
                        modl = row[3]
                        hitc = float(row[4]) if row[4] else None
                        fitc = row[5]
                        coff = float(row[6]) if len(row) > 6 and row[6] else None

                        if aeid in TR_AEIDS:
                            mc5_tr.append({
                                'm5id': m5id,
                                'm4id': m4id,
                                'aeid': aeid,
                                'modl': modl,
                                'hitc': hitc,
                                'fitc': fitc,
                                'coff': coff
                            })

    print(f"\nExtracted:")
    print(f"  {len(chemicals)} chemicals")
    print(f"  {len(mc4_to_spid)} mc4 entries")
    print(f"  {len(spid_to_chid)} sample entries")
    print(f"  {len(mc5_tr)} TR mc5 entries")

    # Link mc5 -> chemical
    results = []
    for entry in mc5_tr:
        m4id = entry['m4id']
        spid = mc4_to_spid.get(m4id)
        if not spid:
            continue
        chid = spid_to_chid.get(spid)
        if not chid:
            continue
        chem = chemicals.get(chid)
        if not chem:
            continue

        results.append({
            'dsstox_substance_id': chem['dsstox_substance_id'],
            'casn': chem['casn'],
            'chnm': chem['chnm'],
            'aeid': entry['aeid'],
            'hitc': entry['hitc'],
            'modl': entry['modl'],
            'coff': entry['coff']
        })

    print(f"  Linked {len(results)} TR entries to chemicals")

    # Save to CSV
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tr_path = output_dir / "toxcast_tr_assays.csv"

    with open(tr_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'dsstox_substance_id', 'casn', 'chnm', 'aeid', 'hitc', 'modl', 'coff'
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} TR entries to {tr_path}")

    # Summary by assay
    aeid_counts = {}
    for r in results:
        aeid = r['aeid']
        aeid_counts[aeid] = aeid_counts.get(aeid, 0) + 1

    print("\nEntries per assay endpoint:")
    aeid_names = {
        112: 'ATG_THRa1_TRANS (gain)',
        143: 'ATG_THRa1_TRANS (loss)',
        936: 'ATG_THRb_TRANS2 (gain)',
        1369: 'ATG_THRb_TRANS2 (loss)'
    }
    for aeid, count in sorted(aeid_counts.items()):
        name = aeid_names.get(aeid, f'Unknown ({aeid})')
        print(f"  {name}: {count}")

    return results


if __name__ == "__main__":
    mysql_dump = Path.home() / "Downloads" / "invitrodb_v4_1_mysql"
    output_dir = Path.home() / "openadmet-models" / "data" / "toxcast"

    extract_tr_data(mysql_dump, output_dir)
