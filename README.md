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
- [Installation](#Installation)
- [Examples](#examples)
- [Data & Outputs](#data--outputs)
- [Description and Usage](#Description-and-Usage)
- [Configuration & Hyperparameters](#configuration--hyperparameters)
- [End-to-end example](#End-to-end-Workflow-Example)
- [Notebooks](#notebooks)
- [Dependencies](#dependencies)

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
trained_models/ 
│   ├── BSI_Large.pth         # Pre-trained BSI-Large model, usable for predictions and fine-tuning
│   └── BSI_Large.params.json # BSI-Large model parameters, loadable for predictions and fine-tuning
example_inputs
    ├── fine_tuning_data.csv  # Example csv file that details the format for fine-tuning data
    └── test_data.csv         # Example csv file that details the format for testing data
```

---

## Installation

```bash
# 1. Clone
git clone git@github.com:gschottlender/bioactivity-similarity-index.git && cd bioactivity-similarity-index

# 2. Create ONE of the follwing environments:

# GPU (recommended) — requires NVIDIA driver ~550+ (CUDA 12.4 compatible)
conda env create -f environment_gpu.yml

# CPU-only — works everywhere
conda env create -f environment_cpu.yml

# 3. Activate environment
conda activate bsi_env

# 4. Register Jupyter kernel (once)
python -m ipykernel install --user --name bsi_env --display-name "Python (bsi_env)"
```

## Examples

```bash
# Extract data from ChEMBL Databases (optional, only for training models from scratch)
python process_chembl_db.py \
  --chembl-sqlite /path/to/chembl_XX.db \
  --out-dir out/db_data

# Build training pairs (streamed to CSV chunks) + train model (optional, only for training models from scratch)
python train_BSI_model.py \
  --data-dir out/db_data \
  --train-dir out/train_data \
  --model-out out/models/bsi_large.pth \
  --model-type BSI_Large_MPG

# Train on a single PFAM group (requires the outputs from process_chembl_db.py)
python train_BSI_model.py \
  --data-dir out/db_data \
  --train-dir out/group_train \
  --model-out out/models/PF00069_model.pth \
  --model-type Group --pfam-id PF00069

# Fine-tune model on new data (example of fine-tuning data format provided in example_inputs/fine_tuning_data.csv)
python fine-tune_model.py \
  --input_csv data/fine-tuning_data.csv \
  --model_path /trained_models/BSI_Large.pth 
  --train_dir out/train_data_ft \
  --model_out out/models_ft/ft_model.pth

# Score new SMILES pairs with trained models (can be used also with fine-tuned models, example of test data format provided in example_inputs/test_data.csv)
python evaluate_bsi_pairs.py \
  --model-path trained_models/BSI_large.pth \
  --input-csv data/pairs_to_score.csv \
  --output-csv out/predictions.csv

```

---

## Data & Outputs
### Inputs
- **ChEMBL SQLite**: `chembl_XX.db` (download from EMBL-EBI).

### Key outputs by stage
**database processing**
- `prot_ligs_db.csv` – ligand–protein activity table with labels (`lig, prot, pchembl, comment, pfam, activity`)
- `smiles.csv` – unique `(ligand_id, smiles)`
- `props.csv` – basic physchem properties per ligand
- `fps.pkl` – dict of ECFP4 (float32) per ligand
- `scaffolds.pkl` – Bemis–Murcko scaffold per ligand
- `decoys.pkl` – per-active lists of decoys (property-matched, similarity-screened)

**pair assembly & training**
- **Chunked CSVs** under `--train-dir`:
  - `chunk_*.csv` with columns: `prot, l1, l2, Tanimoto, y`
- Model weights:
  - `bsi_large.pth`
  - `bsi_large.pth.params.json` (minimal architecture params saved alongside the model)

**inference**
- Scored CSV with predictions appended to your pairs.

---

## Description and Usage

### Obtain & label ChEMBL data
Run:
```bash
python process_chembl_db.py \
  --chembl-sqlite /path/to/chembl_XX.db \
  --out-dir out/db_data
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

### Assemble pairwise datasets & train
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
  --hidden-layers "512,256,128,64" --dropout 0.30 --epochs 10
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

### Fine-tune a pre-trained model
Requires a fine_tuning_data.csv file with columns 'l1','l2','y'. Where 'l1' and 'l2' correspond to the pair of compounds in SMILES format, and 'y' is the similarity label (binary, 1 or 0): 

```csv
l1,l2,y
CCO,CN(C)C(=O),1
CC(C)N,c1ccccc1,0
...
```
Example of the required format provided in example_inputs/fine_tuning_data.csv



Run:
```bash
python fine-tune_model.py \
  --input_csv path/to/fine_tuning_data.csv \
  --model_path out/models/bsi_large.pth \
  --model_out out/models/bsi_large_finetuned.pth \
  --train_dir out/train_chunks_finetune \
  --chunk_prefix chunk --num_chunks 10 \
  --freeze_until_layer 1 \
  --n_epochs 5 --dropout_prob 0.50 \
  --lr 1e-5 --batch_size 32 \
  --fp-bits 256 --random_seed 42
```

What it does:
- **Loads labeled ligand–ligand pairs** from `--input_csv` and validates required columns `l1`, `l2`, `y`.
- **Computes ECFP4 fingerprints (256‑bit)** for every unique SMILES appearing in `l1` or `l2` (cached and reused across chunks).
- **Shuffles and splits the pairs into chunked CSVs** under `--train_dir` (bounded‑memory training), with files prefixed by `--chunk_prefix`.
- **Loads base model hyperparameters** (hidden layers & dropout) from `--model_path` and **reconstructs the network** accordingly; then **loads pre‑trained weights**.
- **Fine‑tunes the model** on the chunked data, with options to freeze initial layers (`--freeze_until_layer`), set epochs, dropout during training, learning rate, and batch size.
- **Saves the fine‑tuned weights to `--model_out`** and writes a small JSON next to it (`.params.json`) capturing essential training parameters (hidden layers, dropout, seed).

Notes:
- Fingerprints are computed once per unique SMILES to avoid recomputation across chunks.
- The script preserves the base network architecture (hidden layers) from the pre‑trained model; training‑time options are controlled via CLI.


---

### Evaluate new SMILES pairs
Requires a CSV with columns 'l1', and 'l2', both in SMILES format:
```csv
l1,l2
CCO,CN(C)C(=O)
CC(C)N,c1ccccc1
...
```
Example of the required format provided in example_inputs/test_data.csv

Run:
```bash
python evaluate_bsi_pairs.py \
  --model-path out/models/bsi_large.pth \
  --input-csv data/pairs_to_score.csv \
  --output-csv out/predictions.csv \
  --tanimoto-threshold 0.40 \
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

## End-to-end Workflow Example 
### Training and fine-tuning a BSI-Large using higher dimension fingerprints (fp-bits = 512)
We fine-tune on example data involving S pairs of compounds represented by two drug-like compounds, and N pairs corresponding to one drug-like compound and another non drug-like.
We predict with the fine-tuned model on related data from drug-like and non-drug like compounds.

#### Step 1: Download and process ChEMBL Database

```bash
python process_chembl_db.py \
  --download_chembl \
  --out-dir out/db_data \
  --fp-bits 512
```

#### Step 2: Train the model

```bash
python train_BSI_model.py \
  --data-dir out/db_data \
  --train-dir out/train_data \
  --model-out out/models/bsi_large.pth \
  --hidden-layers "512,256,128,64" \
  --model-type BSI_Large_MPG
```

#### Step 3: Fine-tune the model

```bash
python fine-tune_model.py \
  --input_csv /example_inputs/fine_tuning_data.csv \
  --model_path out/models/bsi_large.pth \
  --model_out out/models/bsi_large_finetuned.pth \
  --train_dir out/train_data_ft \
  --freeze_until_layer 1 \
  --n_epochs 5
```

#### Step 4: Predictions on example data

Predict using BSI-Large model

```bash
python evaluate_bsi_pairs.py \
  --model-path out/models/bsi_large.pth \
  --input-csv example_inputs/test_data.csv \
  --output-csv out/test_data_preds.csv \
  --tanimoto-threshold 0.40
```

Predict using fine-tuned BSI-Large model

```bash
python evaluate_bsi_pairs.py \
  --model-path out/models/bsi_large_finetuned.pth \
  --input-csv data/pairs_to_score.csv \
  --output-csv out/test_data_preds_ft.csv \
  --tanimoto-threshold 0.40
```

---

## Notebooks
- **1_data_obtaining.ipynb** – mirrors `process_chembl_db.py` steps: SQL extraction, labeling, and artifact generation.
- **2_dataset_assembly.ipynb** – demonstrates ligand pruning, pair generation, and chunking.
- **3_train_models.ipynb** – focuses on training from chunks, monitoring loss/metrics, and saving the model.

> The scripts are the source of truth for automation; the notebooks are great for exploration, workflow modifications and sanity checks.

---

## Dependencies
- Python ≥ 3.10
- RDKit
- NumPy, pandas
- scikit-learn
- PyTorch
- SQLite (library only; no server required)

