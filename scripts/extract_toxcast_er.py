#!/usr/bin/env python
"""
Extract Estrogen Receptor assay data from ToxCast invitrodb MySQL dump.

This script parses the SQL dump and extracts:
1. ER-related assay endpoints (ATG_ERa_TRANS, ATG_ERE_CIS, NVS_NR_hER, etc.)
2. mc5 fitted dose-response results (hitc, AC50)
3. Chemical information (DTXSID for SMILES lookup)

Output: CSV with SMILES, hitcall, and potency for ER model training
"""

import re
import csv
import sys
from pathlib import Path

# ER-related assay endpoint IDs from assay_component_endpoint.csv
# Found earlier: 2 (ACEA_ER_80hr), 75 (ATG_ERE_CIS), 117 (ATG_ERa_TRANS),
# 714 (NVS_NR_hER), 725 (NVS_NR_mERa), 742-747 (OT_ER_*), 750-751 (OT_ERa_*)
ER_AEIDS = [2, 75, 117, 714, 725, 742, 743, 744, 745, 746, 747, 750, 751]

# AR-related assay endpoint IDs
AR_AEIDS = [115, 710, 711, 726, 739, 740, 741, 759, 760]


def parse_insert_values(line, table_name):
    """Parse INSERT INTO ... VALUES (...) statements."""
    # Match the pattern: INSERT INTO `table` VALUES (val1,val2,...),(val1,val2,...)...
    pattern = rf"INSERT INTO `{table_name}` VALUES "
    if not line.startswith(pattern):
        return []

    values_str = line[len(pattern):]

    # Parse comma-separated tuples
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


def extract_er_data(mysql_dump_path, output_dir):
    """Extract ER assay data from MySQL dump."""

    print(f"Reading MySQL dump from {mysql_dump_path}...")

    # Storage for parsed data
    chemicals = {}  # chid -> {casn, chnm, dsstox_substance_id}
    mc5_er = []     # List of mc5 rows for ER assays
    mc5_ar = []     # List of mc5 rows for AR assays

    # We need to link: mc5 -> m4id -> spid -> chid -> dsstox_substance_id -> SMILES
    # The mc5 table has: m5id, m4id, aeid, modl, hitc, fitc, coff, actp, ...
    # The mc4 table has: m4id, spid, aeid, ...
    # The sample table has: spid, chid, ...

    mc4_to_spid = {}  # m4id -> spid
    spid_to_chid = {} # spid -> chid

    current_table = None

    with open(mysql_dump_path, 'r', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f):
            if line_num % 100000 == 0:
                print(f"  Processing line {line_num}...")

            line = line.strip()

            # Track current table for context
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
                        # mc4 structure: m4id, aeid, spid, ...
                        spid = row[2]
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

                        if aeid in ER_AEIDS:
                            mc5_er.append({
                                'm5id': m5id,
                                'm4id': m4id,
                                'aeid': aeid,
                                'modl': modl,
                                'hitc': hitc,
                                'fitc': fitc,
                                'coff': coff
                            })
                        elif aeid in AR_AEIDS:
                            mc5_ar.append({
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
    print(f"  {len(mc5_er)} ER mc5 entries")
    print(f"  {len(mc5_ar)} AR mc5 entries")

    # Link mc5 -> chemical
    def link_to_chemical(mc5_list, name):
        results = []
        for entry in mc5_list:
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

        print(f"  Linked {len(results)} {name} entries to chemicals")
        return results

    er_results = link_to_chemical(mc5_er, "ER")
    ar_results = link_to_chemical(mc5_ar, "AR")

    # Save to CSV
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    er_path = output_dir / "toxcast_er_assays.csv"
    ar_path = output_dir / "toxcast_ar_assays.csv"

    for results, path, name in [(er_results, er_path, "ER"), (ar_results, ar_path, "AR")]:
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'dsstox_substance_id', 'casn', 'chnm', 'aeid', 'hitc', 'modl', 'coff'
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {len(results)} {name} entries to {path}")

    return er_results, ar_results


if __name__ == "__main__":
    mysql_dump = Path.home() / "Downloads" / "invitrodb_v4_1_mysql"
    output_dir = Path.home() / "openadmet-models" / "data" / "toxcast"

    extract_er_data(mysql_dump, output_dir)
