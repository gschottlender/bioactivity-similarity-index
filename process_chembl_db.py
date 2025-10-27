#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process ChEMBL SQLite → prot_ligs_db.csv (+ smiles/properties/fps/scaffolds/decoys)
==============================================================================

This script generates a database of ChEMBL ligands with their targets and corresponding activity information. 
It uses a SQL query, clamps pChEMBL to [3,10], fills 3 for
explicit negatives without pChEMBL, labels activity with thresholds, drops
undefined, and writes `prot_ligs_db.csv` as output. Also calculates Fingerprints, Bemis Murcko Scaffolds for the ChEMBL compounds.
Finally obtains decoys for each active ChEMBL compound.

Exports (by default; use --skip-* to disable):
- prot_ligs_db.csv              (columns: lig, prot, pchembl, comment, pfam, activity)
- smiles.csv                    (ligand_id, smiles)
- props.csv                     (basic physchem per ligand)
- fps.npy + fps_index.csv + fps.pkl  (ECFP4)
- scaffolds.pkl                 (Bemis–Murcko)
- decoys.pkl                    (per-active decoy lists)

CLI accepts both dashed and underscored flags for convenience
(e.g., --chembl-sqlite or --chembl_sqlite).
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors

from src.ligand_clustering_functions import (
    get_ligand_scaffolds,
    generate_decoys_from_properties,
)

log = logging.getLogger("process_chembl_db")

# --------------------------- Query (single SQL) ---------------------------
QUERY_PROT_LIGS = '''
select distinct molecule_dictionary.chembl_id as ligand_id,
component_sequences.accession as uniprot_id,
activities.pchembl_value as pchembl,
activities.activity_comment as comment,
domains.source_domain_id as pfam,
compound_structures.canonical_smiles as smiles
            from activities join assays on activities.assay_id = assays.assay_id
            join molecule_dictionary on activities.molregno = molecule_dictionary.molregno 
			join compound_structures on molecule_dictionary.molregno = compound_structures.molregno
            join target_dictionary on assays.tid = target_dictionary.tid
			join target_components on target_dictionary.tid = target_components.tid
			join component_sequences on target_components.component_id = component_sequences.component_id
            join site_components on target_components.component_id = site_components.component_id
			join domains on site_components.domain_id = domains.domain_id
            where assays.assay_type = 'B' and
            target_dictionary.target_type = "SINGLE PROTEIN"'''

# --------------------------- Helpers ---------------------------

def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')


def sql_fetch(query: str, chembl_sqlite: Path) -> list[tuple]:
    with sqlite3.connect(str(chembl_sqlite)) as conn:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
    return rows

# --------------------------- Notebook-equivalent build ---------------------------

def build_prot_ligs_table(chembl_sqlite: Path,
                          positive_threshold: float = 6.5,
                          negative_threshold: float = 4.5) -> pd.DataFrame:
    """Builds database of of ChEMBL ligands with their targets and corresponding activity information.

    Note: we also fetch `smiles` in the same query to avoid a second DB hit.
    The returned DataFrame therefore includes a `smiles` column, which we
    exclude when writing `prot_ligs_db.csv`, but reuse to emit `smiles.csv` and
    downstream databases.
    """
    rows = sql_fetch(QUERY_PROT_LIGS, chembl_sqlite)
    df = pd.DataFrame(rows, columns=['lig','prot','pchembl','comment','pfam','smiles'])

    # Fill explicit negatives without pChEMBL with 3
    df.loc[(df['comment'].isin(['Not Active','inactive','No significant effect','No Activity','No binding'])) &
           (df['pchembl'].isna()), 'pchembl'] = 3
    # Clamp to [3, 10]
    df.loc[df['pchembl'] < 3, 'pchembl'] = 3
    df.loc[df['pchembl'] > 10, 'pchembl'] = 10

    # Label activity as in the notebook
    df['activity'] = df['pchembl'].apply(lambda x: 1 if x > positive_threshold else (0 if x < negative_threshold else np.nan))

    # Drop undefined
    df = df.dropna(subset=['activity']).reset_index(drop=True)
    return df



# --------------------------- Other databases ---------------------------

def compute_properties(smiles_map: Dict[str,str]) -> pd.DataFrame:
    rows: list[dict] = []
    for mol_id, smiles in smiles_map.items():
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue
        rows.append({'compound_id': mol_id,
                     'mw': Descriptors.MolWt(mol),
                     'logP': Descriptors.MolLogP(mol),
                     'rot_bonds': Descriptors.NumRotatableBonds(mol),
                     'h_acceptors': Descriptors.NumHAcceptors(mol),
                     'h_donors': Descriptors.NumHDonors(mol),
                     'charge': Chem.rdmolops.GetFormalCharge(mol)})
    df = pd.DataFrame(rows).set_index('compound_id').sort_index()
    return df


def compute_ecfp4(smiles_map: Dict[str,str], n_bits: int = 256) -> dict[str, np.ndarray]:
    out: dict[str,np.ndarray] = {}
    for lig, smi in smiles_map.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(bv, arr)
        out[lig] = arr
    return out


def compute_scaffolds(smiles_map: Dict[str,str]) -> dict[str,str]:
    return get_ligand_scaffolds(smiles_map)


def _arr_to_bv(arr: np.ndarray):
    from rdkit.DataStructs.cDataStructs import CreateFromBitString
    return CreateFromBitString(''.join('1' if float(x) >= 0.5 else '0' for x in arr.tolist()))


def generate_decoys_for_actives(actives: Sequence[str], props: pd.DataFrame,
                                fps: dict[str,np.ndarray], scaffolds: dict[str,str],
                                threshold: float = 0.30) -> dict[str,list[str]]:
    fps_bv = {cid: _arr_to_bv(v) for cid, v in fps.items()}
    decoys: dict[str,list[str]] = {}
    for lig in actives:
        try:
            decoys[lig] = generate_decoys_from_properties(lig, props, fps_bv, scaffolds, threshold=threshold)
        except Exception:
            decoys[lig] = []
    return decoys

# --------------------------- CLI ---------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(
        'Build database of ligands and targets from ChEMBL with corresponding activity, plus SMILES, fingerprints, Bemis-Murcko scaffolds and decoys.'
    ))

    # Accept both dashed and underscored variants
    ap.add_argument('--chembl-sqlite','--chembl_sqlite', required=True, type=Path, help='Path to chembl_XX.db (SQLite)')
    ap.add_argument('--out-dir','--out_dir', required=True, type=Path, help='Output directory')

    # Export policy: everything ON by default; allow skipping
    ap.add_argument('--skip-props', action='store_true', help='Do not export properties CSV')
    ap.add_argument('--skip-fps', action='store_true', help='Do not export ECFP4')
    ap.add_argument('--skip-scaffolds', action='store_true', help='Do not export scaffolds')
    ap.add_argument('--skip-decoys', action='store_true', help='Do not export decoys')

    # Thresholds/sizes
    ap.add_argument('--positive-threshold', type=float, default=6.5, help='pChEMBL > t => active (default 6.5)')
    ap.add_argument('--negative-threshold', type=float, default=4.5, help='pChEMBL < t => inactive (default 4.5)')
    ap.add_argument('--fp-bits', type=int, default=256, help='ECFP4 bit-length (default 256)')

    # Logging
    ap.add_argument('-v','--verbose', action='count', default=1, help='Verbosity (-v, -vv)')

    return ap.parse_args()

# --------------------------- Main ---------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(
    'Build database of ligands and targets from ChEMBL with corresponding activity, plus SMILES, fingerprints (ECFP4 .pkl), Bemis–Murcko scaffolds, and decoys. All databases are produced.'
    ))


    # Accept both dashed and underscored variants
    ap.add_argument('--out-dir','--out_dir', required=True, type=Path, help='Output directory')

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--download_chembl", action="store_true", help="Download ChEMBL database automatically (from EBI FTP)")
    group.add_argument("--chembl-sqlite", "--chembl_sqlite", type=Path, help="Path to local chembl_XX.db (SQLite)")

    # Thresholds/sizes
    ap.add_argument('--positive-threshold', type=float, default=6.5, help='pChEMBL > t => active (default 6.5)')
    ap.add_argument('--negative-threshold', type=float, default=4.5, help='pChEMBL < t => inactive (default 4.5)')
    ap.add_argument('--fp-bits', type=int, default=256, help='ECFP4 bit-length (default 256)')
    
    ap.add_argument('--chembl_version',type=int, default=35)



    # Logging
    ap.add_argument('-v','--verbose', action='count', default=1, help='Verbosity (-v, -vv)')


    return ap.parse_args()


# --------------------------- Main ---------------------------

def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.download_chembl:
        # Downloads chembl with the specified version

        chembl_file = out / f"chembl_{args.chembl_version}_sqlite.tar.gz"
        
        if chembl_file.exists():
            chembl_file.unlink()
        
        os.system(
            f"wget -P {out} "
            f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/"
            f"chembl_{args.chembl_version}/chembl_{args.chembl_version}_sqlite.tar.gz"
        )
        os.system(f"tar -xvzf {chembl_file} -C {out}")
        chembl_db_path = next(out.rglob(f"chembl_{args.chembl_version}.db"))
    else:
        chembl_db_path = args.chembl_sqlite

    # 1) Build prot_ligs db (with activity) from a single SQL query
    prot_ligs = build_prot_ligs_table(chembl_db_path, args.positive_threshold, args.negative_threshold)

    # 2) Export prot_ligs_db.csv (without smiles) and smiles.csv (from the same result)
    prot_cols = ['lig','prot','pchembl','comment','pfam','activity']
    prot_ligs.loc[:, prot_cols].to_csv(out / 'prot_ligs_db.csv', index=False)
    log.info('Wrote prot_ligs_db.csv → %s', out / 'prot_ligs_db.csv')

    smiles_map = (prot_ligs[['lig','smiles']]
                .dropna(subset=['smiles'])
                .drop_duplicates('lig')
                .rename(columns={'lig':'ligand_id'}))
    smiles_map.to_csv(out / 'smiles.csv', index=False)
    log.info('Wrote smiles.csv → %s', out / 'smiles.csv')

    # 3) Always compute and export properties, ECFP4 (pickle), scaffolds, and decoys
    smiles_dict: Dict[str,str] = dict(smiles_map.values)

    # Properties
    props = compute_properties(smiles_dict)
    props.to_csv(out / 'props.csv')
    log.info('Wrote props.csv → %s', out / 'props.csv')

    # Fingerprints (ECFP4) → only .pkl
    fps = compute_ecfp4(smiles_dict, n_bits=args.fp_bits)
    with open(out / 'fps.pkl','wb') as f:
        pickle.dump(fps, f)
    log.info('Wrote ECFP4 (pickle dict) → %s', out / 'fps.pkl')

    # Scaffolds
    scaff = compute_scaffolds(smiles_dict)
    with open(out / 'scaffolds.pkl','wb') as f:
        pickle.dump(scaff, f)
    log.info('Wrote scaffolds.pkl → %s', out / 'scaffolds.pkl')

    # Decoys (threshold fixed at 0.30 as in previous logic)
    actives = prot_ligs.loc[prot_ligs['activity'] == 1, 'lig'].unique().tolist()
    decoys = generate_decoys_for_actives(actives, props, fps, scaff, threshold=0.30)
    with open(out / 'decoys.pkl','wb') as f:
        pickle.dump(decoys, f)
    log.info('Wrote decoys.pkl → %s', out / 'decoys.pkl')

    log.info('All databases generated. ✅')
    return 0

if __name__ == "__main__":
    main()
