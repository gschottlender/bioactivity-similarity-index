# BSI: Binding Similarity Inference – From ChEMBL to Trained Models

A reproducible pipeline to:
1) extract & label ligand–protein activity data from a local ChEMBL SQLite dump,  
2) derive cheminformatics artifacts (SMILES, physchem props, ECFP4, Bemis–Murcko scaffolds, decoys),  
3) assemble pairwise training sets (S/N pairs with Tanimoto filtering), train a neural model, and  
4) **evaluate** new compound pairs with the trained model.

This repo includes three command-line scripts and companion notebooks that mirror the workflow end-to-end. Also includes a pre-trained BSI-Large model, available to make predictions and fine-tuning.

---

## Table of Contents
- [Project Goals](#project-goals)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Data & Outputs](#data--outputs)
- [Workflow](#workflow)
  - [1) Obtain & label ChEMBL data](#1-obtain--label-chembl-data)
  - [2) Assemble pairwise datasets & train](#2-assemble-pairwise-datasets--train)
  - [3) Evaluate new SMILES pairs](#3-evaluate-new-smiles-pairs)
- [Configuration & Hyperparameters](#configuration--hyperparameters)
- [Notebooks](#notebooks)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Project Goals
Predict whether two ligands are **binding-similar** (positive) or **non-similar** (negative) with respect to protein targets, using data derived from ChEMBL and ECFP4-based similarity constraints. The training set is built by combining:
- positive **S** pairs: active–active for the same target,
- negative **N** pairs: active vs inactive/decoy for the same target,
while filtering by Tanimoto to avoid trivial similarity.

---

## Repository Structure
```
.
├── process_chembl_db.py        # Build labeled tables + cheminformatics artifacts from ChEMBL (SQLite)
├── train_BSI_model.py          # Streamed pair generation + chunked CSVs + model training
├── evaluate_bsi_pairs.py       # Evaluate new compound pairs (from SMILES) with a trained model (.pth + .params.json)
├── fine-tune_model.py          # Fine-tune pre-trained model on new data
notebooks/
│   ├── 1_data_obtaining.ipynb      # Notebook version for interactive step-by-step data extraction from chembl
│   ├── 2_dataset_assembly.ipynb    # Notebook version of interactive step-by-step dataset active and inactive pairs assembly
│   └── 3_train_models.ipynb        # Notebook version of interactive step-by-step model training, including evaluations on test sets and fine-tuning
src/
│   ├── ligand_clustering_functions.py  # Ligand filtering, scaffolds, clustering, decoys, Tanimoto
│   └── model_training_functions.py     # Fingerprint conversion, NN model, training, fine‑tuning, inference
trained_models 
    ├── BSI_Large.pth         # Pre-trained BSI-Large model, usable for predictions and fine-tuning
    └── BSI_Large.params.json # BSI-Large model parameters, loadable for predictions and fine-tuning
```

---

## Instalation

```bash
# 1. Clone
git clone git@github.com:gschottlender/bioactivity-similarity-index.git && cd bioactivity-similarity-index

# 2. Create environment (exact versions from paper)
conda env create -f environment.yml
conda activate bsi_env

# 3. Register Jupyter kernel (once)
python -m ipykernel install --user --name bsi_env --display-name "Python (bsi_env)"
```

## Quick Examples

```bash
# Extract data from ChEMBL Databases (optional, only for training models from scratch)
python process_chembl_db.py \
  --chembl-sqlite /path/to/chembl_XX.db \
  --out-dir out/db_data -vv

# Build training pairs (streamed to CSV chunks) + train model (optional, only for training models from scratch)
python train_BSI_model.py \
  --data-dir out/db_data \
  --train-dir out/train_data \
  --model-out out/models/bsi_large.pth \
  --model-type BSI_Large_MPG

# Fine-tune model on new data
python fine-tune_model.py \
  --input_csv data/fine-tuning_data.csv \
  --model_path /trained_models/BSI_Large.pth 
  --train_dir out/train_data_ft \
  --model_out out/models_ft/ft_model.pth

# Score new SMILES pairs with trained models
python evaluate_bsi_pairs.py \
  --model-path trained_models/BSI_large.pth \
  --input-csv data/pairs_to_score.csv \
  --output-csv out/predictions.csv -vv
```

---

## Installation
**Core dependencies**
- Python ≥ 3.10
- RDKit (with minimal features for ECFP4, scaffolds, descriptors)
- NumPy, pandas
- PyTorch (CPU is fine; GPU optional)
- scikit-learn (KMeans)
- SQLite (to read the ChEMBL dump)

If you don’t have a `requirements.txt`, start with:
```txt
numpy
pandas
rdkit-pypi
torch
scikit-learn
```

> RDKit wheels (`rdkit-pypi`) are convenient on many platforms; on others you may prefer conda (`conda install -c conda-forge rdkit`).

---

## Data & Outputs
### Inputs
- **ChEMBL SQLite**: `chembl_XX.db` (download from EMBL-EBI).

### Key outputs by stage
**Step 1 (database processing)**
- `prot_ligs_db.csv` – ligand–protein activity table with labels (`lig, prot, pchembl, comment, pfam, activity`)
- `smiles.csv` – unique `(ligand_id, smiles)`
- `props.csv` – basic physchem properties per ligand
- `fps.pkl` – dict of ECFP4 (float32) per ligand
- `scaffolds.pkl` – Bemis–Murcko scaffold per ligand
- `decoys.pkl` – per-active lists of decoys (property-matched, similarity-screened)

**Step 2 (pair assembly & training)**
- **Chunked CSVs** under `--train-dir`:
  - `chunk_*.csv` with columns: `prot, l1, l2, Tanimoto, y`
- Model weights:
  - `bsi_large.pth`
  - `bsi_large.pth.params.json` (minimal architecture params saved alongside the model)

**Step 3 (inference)**
- Scored CSV with predictions appended to your pairs.

---

## Workflow

### 1) Obtain & label ChEMBL data
Run:
```bash
python process_chembl_db.py \
  --chembl-sqlite /path/to/chembl_XX.db \
  --out-dir out/db_data -vv
```
What it does:
- Single SQL joins fetch **ligand_id**, **UniProt**, **pChEMBL**, **PFAM**, **SMILES**.
- pChEMBL is clamped to **[3,10]**; explicit textual “inactive/no binding” with missing pChEMBL are set to **3**.
- Activity labels: `activity = 1` if pChEMBL > **6.5**, `0` if pChEMBL < **4.5**; mid-zone rows are dropped.
- Emits the full set of artifacts (SMILES, props, **ECFP4**, **scaffolds**, **decoys**).

Flags of note:
- `--positive-threshold` (default 6.5)
- `--negative-threshold` (default 4.5)
- `--fp-bits` (default 256)

---

### 2) Assemble pairwise datasets & train
Run:
```bash
python train_BSI_model.py \
  --data-dir out/db_data \
  --train-dir out/train_datasets \
  --model-out out/models/bsi_large.pth \
  --model-type BSI_Large_MPG \
  --tanimoto-threshold 0.40 --butina-threshold 0.40 \
  --kmeans-representatives 100 --min-positives 25 \
  --n-decoys-per-lig 25 --decoys-proportion 2.0 \
  --num-chunks 30 --chunk-prefix chunk \
  --hidden-layers "512,256,128,64" --dropout 0.30 --epochs 10 -vv
```

What it does:
- Loads `prot_ligs_db.csv`, `fps.pkl`, `scaffolds.pkl`, `decoys.pkl`.
- **Protein selection**:
  - `BSI_Large`: all proteins,
  - `BSI_Large_MPG`: proteins within a curated **MPG** PFAM list,
  - `Group`: restrict to a **single PFAM** via `--pfam-id`.
  - You can further restrict with `--protein-list-file` or exclude using `--exclude-prots-file`.
- **Ligand pruning** (per class): **Bemis–Murcko → Butina → KMeans**.
- Construct **S (active–active)** and **N (active–inactive/decoy)** pairs; compute ECFP4-Tanimoto and keep only pairs with `Tanimoto < threshold` (default **0.40**).
- **Global shuffle** and **split** pairs into `chunk_*.csv` to keep memory bounded.
- Train the neural network from chunked CSVs using provided **hidden layers**, **dropout**, and **epochs**; save weights to `--model-out` (and a small JSON with the essential params next to it).

---

### 3) Evaluate new SMILES pairs
Prepare a CSV with columns:
```csv
l1,l2
CCO,CN(C)C(=O)...
CC(C)N...,c1ccccc1...
...
```

Run:
```bash
python evaluate_bsi_pairs.py \
  --model-path out/models/bsi_large.pth \
  --input-csv data/pairs_to_score.csv \
  --output-csv out/predictions.csv \
  --tanimoto-threshold 0.40 \
  --fp-bits 256 -vv
```

What it does:
- Reads `model_path.pth` **and** `model_path.pth.params.json` (same basename).
- Optionally override `--hidden-layers` / `--dropout` from CLI.
- (Default) computes ECFP4 (radius=2) and **filters** pairs with `Tanimoto ≥ threshold` to mirror the training regime. Disable with `--no-tanimoto-filter`.
- Appends predictions to your CSV and writes results to `--output-csv`.

---

## Configuration & Hyperparameters
- **ECFP4**: `fp-bits` default **256** for both processing and evaluation.
- **Tanimoto threshold**: default **0.40** (both training pair generation and evaluation).
- **Clustering**: Butina `0.40`, KMeans representatives `100` by default; Bemis–Murcko toggled **on** in the pruning chain.
- **Model**: MLP with user-defined `--hidden-layers`, `--dropout`; training for `--epochs` epochs.

---

## Notebooks
- **1_data_obtaining.ipynb** – mirrors `process_chembl_db.py` steps: SQL extraction, labeling, and artifact generation.
- **2_dataset_assembly.ipynb** – demonstrates ligand pruning, pair generation, and chunking.
- **3_train_models.ipynb** – focuses on training from chunks, monitoring loss/metrics, and saving the model.

> The scripts are the source of truth for automation; the notebooks are great for exploration and sanity checks.

---

## Examples

### Minimal end-to-end (default settings)
```bash
# Step 1
python process_chembl_db.py --chembl-sqlite chembl_34.db --out-dir out/db_data -v
# Step 2
python train_BSI_model.py --data-dir out/db_data --train-dir out/train --model-out out/models/bsi.pth -v
# Step 3
python evaluate_bsi_pairs.py --model-path out/models/bsi.pth \
  --input-csv data/pairs.csv --output-csv out/preds.csv -v
```

### Train on a single PFAM group
```bash
python train_BSI_model.py \
  --data-dir out/db_data \
  --train-dir out/group_train \
  --model-out out/models/bsi_group.pth \
  --model-type Group --pfam-id PF00069 -vv
```

### Use custom architecture at inference
```bash
python evaluate_bsi_pairs.py \
  --model-path out/models/bsi_large.pth \
  --input-csv data/pairs.csv \
  --output-csv out/preds.csv \
  --hidden-layers "[768,256,64]" --dropout 0.25 -vv
```

---

## Troubleshooting
- **RDKit import errors**: prefer conda to install RDKit; otherwise ensure `rdkit-pypi` matches your Python version.
- **No training pairs generated**: lower `--min-positives`, relax pruning (increase `--kmeans-representatives`), or adjust activity thresholds in Step 1.
- **Empty output after evaluation**: you likely filtered all pairs by Tanimoto; rerun with `--no-tanimoto-filter` or raise `--tanimoto-threshold`.
- **Missing `.params.json` at inference**: make sure the params JSON is saved next to your `.pth` (same basename). If absent, supply `--hidden-layers` and `--dropout` explicitly.

---

## Dependencies
- Python ≥ 3.10
- RDKit
- NumPy, pandas
- scikit-learn
- PyTorch
- SQLite (library only; no server required)

