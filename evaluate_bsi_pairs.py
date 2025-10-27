#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate SMILES pairs with a trained BSI model
==============================================
Loads a trained BSI model (.pth) and its minimal PyTorch params from the
adjacent JSON file with the same basename and suffix '.params.json', e.g.:

  /path/to/BSI_Large.pth
  /path/to/BSI_Large.pth.params.json

If --hidden-layers or --dropout are provided via CLI, they override
the values loaded from JSON.

Input CSV must have: l1,l2  (SMILES)
Output CSV appends model predictions.

Notes:
- Model class and preprocessing live in src.model_training_functions
- Optional Tanimoto filtering to match the train regime
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from src.model_training_functions import (
    NeuralNetworkModel,
    prepare_and_evaluate_pairs,
)

# ========================= Logging & utils =========================

log = logging.getLogger("evaluate_bsi")

def setup_logging(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

def parse_hidden_layers(s: str) -> List[int]:
    s = s.strip()
    if s.startswith("["):
        return list(map(int, json.loads(s)))
    return [int(x) for x in s.split(",") if x.strip()]

# ========================= Fingerprints & Tanimoto =========================

def ecfp4_from_smiles(smiles: str, n_bits: int) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr

def tanimoto_smiles(smi1: str, smi2: str, n_bits: int) -> float | None:
    v1 = ecfp4_from_smiles(smi1, n_bits)
    v2 = ecfp4_from_smiles(smi2, n_bits)
    if v1 is None or v2 is None:
        return None
    from rdkit.DataStructs.cDataStructs import CreateFromBitString
    bv1 = CreateFromBitString("".join("1" if x >= 0.5 else "0" for x in v1.tolist()))
    bv2 = CreateFromBitString("".join("1" if x >= 0.5 else "0" for x in v2.tolist()))
    return float(DataStructs.FingerprintSimilarity(bv1, bv2))

# ========================= Params loader =========================

def load_model_params_from_json(model_path: Path) -> dict:
    """
    Load minimal PyTorch model params from <model>.params.json
    Expected keys: hidden_layers (list[int]), dropout (float).
    """
    model_path = Path(model_path)
    params_path = model_path.with_suffix(".params.json")
    if not params_path.exists():
        raise SystemExit(f"Missing params JSON: {params_path} (expected next to {model_path})")

    with open(params_path) as f:
        params = json.load(f)

    if "hidden_layers" not in params or "dropout" not in params:
        raise SystemExit(
            f"Params JSON must contain 'hidden_layers' and 'dropout'. Got: {list(params.keys())} "
            f"at {params_path}"
        )
    if not isinstance(params["hidden_layers"], list) or not all(isinstance(x, int) for x in params["hidden_layers"]):
        raise SystemExit("'hidden_layers' must be a list of ints in params JSON.")
    if not isinstance(params["dropout"], (int, float)):
        raise SystemExit("'dropout' must be numeric in params JSON.")

    return params

# ========================= CLI =========================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate a CSV of SMILES pairs with a trained BSI model; optional Tanimoto filtering to match training."
        )
    )

    # IO
    ap.add_argument("--model-path", type=Path, required=True, help="Path to trained model weights (.pth)")
    ap.add_argument("--input-csv", type=Path, required=True, help="CSV with columns l1,l2 containing SMILES pairs")
    ap.add_argument("--output-csv", type=Path, required=True, help="Output CSV with predictions appended")

    # Similarity filtering
    ap.add_argument("--tanimoto-threshold", type=float, default=0.40,
                    help="Filter out pairs with Tanimoto >= threshold (default: 0.40)")
    ap.add_argument("--no-tanimoto-filter", action="store_true", help="Do not filter pairs by Tanimoto")
    ap.add_argument("--fp-bits", type=int, default=256, help="ECFP4 bit-length used for compound encoding (default: 256)")

    # Logging
    ap.add_argument("-v", "--verbose", action="count", default=1, help="Verbosity (-v, -vv)")

    return ap.parse_args()

# ========================= Main =========================

def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    # 1) Load input pairs
    df = pd.read_csv(args.input_csv)
    for col in ("l1", "l2"):
        if col not in df.columns:
            raise SystemExit(f"Input CSV must contain column '{col}'")

    # 2) Optional Tanimoto filtering (recommended to mirror training)
    if not args.no_tanimoto_filter:
        log.info("Computing Tanimoto to filter pairs with similarity >= %.2f", args.tanimoto_threshold)
        tanims: list[float | None] = []
        keep_mask = []
        for smi1, smi2 in df.itertuples(index=False, name=None):
            t = tanimoto_smiles(smi1, smi2, args.fp_bits)
            tanims.append(t)
            keep_mask.append((t is not None) and (t < args.tanimoto_threshold))
        df["Tanimoto"] = tanims
        kept = int(sum(keep_mask))
        dropped = len(df) - kept
        if dropped > 0:
            log.info("Dropped %d/%d pairs (Tanimoto >= %.2f or invalid SMILES)", dropped, len(df), args.tanimoto_threshold)
        df = df.loc[keep_mask].reset_index(drop=True)

    if df.empty:
        log.warning("No pairs left to score after filtering. Exiting.")
        out = pd.DataFrame(columns=["l1", "l2", "predicted_score"])
        out.to_csv(args.output_csv, index=False)
        return 0

    # 3) Load model params from JSON next to the .pth, then apply CLI overrides
    base_params = load_model_params_from_json(args.model_path)  # loads hidden_layers + dropout
    hidden_layers = base_params["hidden_layers"]
    dropout = float(base_params["dropout"])

    log.info("Model params → hidden_layers=%s | dropout=%.3f", hidden_layers, dropout)

    # 4) Construct model and load weights
    model = NeuralNetworkModel(hidden_layers=hidden_layers, dropout_prob=dropout, input_size=args.fp_bits, output_size=1)
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # 5) Score pairs via project helper (ensures same preprocessing as training)
    scored = prepare_and_evaluate_pairs(df[["l1", "l2"]].copy(), model, fp_size=args.fp_bits)

    # 6) Save predictions
    if {"l1", "l2"}.issubset(scored.columns):
        out = scored
    else:
        out = df.join(scored)
    out.to_csv(args.output_csv, index=False)
    log.info("Wrote predictions → %s", args.output_csv)

    return 0

if __name__ == "__main__":
    main()
