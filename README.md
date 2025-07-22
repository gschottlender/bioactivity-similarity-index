# BSI Molecular Similarity

This repository accompanies the manuscript describing the **Bioactivity Similarity Index (BSI)** models trained to detect functionally similar ligands across protein targets when conventional Tanimoto similarity is low. The code and notebooks provided here reproduce the core data preparation, dataset assembly, model training, and basic evaluation workflows used in the study.

---

## Repository Structure

```
bioactivity-similarity-index/
├── environment.yml                # Conda environment (exact versions used in the study)
├── 1_data_obtaining.ipynb         # Retrieve & normalize ligand–protein activity data
├── 2_dataset_assembly.ipynb       # Build S/N pair datasets; clustering & decoys
├── 3_train_models.ipynb           # Train BSI group models & BSI-Large; evaluation
├── src/
│   ├── ligand_clustering_functions.py  # Ligand filtering, scaffolds, clustering, decoys, Tanimoto
│   └── model_training_functions.py     # Fingerprint conversion, NN model, training, fine‑tuning, inference
├── data/                          # (generated) user-provided input tables go here (not versioned)
├── train_datasets/                # (generated) chunked training CSVs
├── test_datasets/                 # (generated) chunked test CSVs / evaluation splits
└── README.md                      # You are here
```

---

## Quick Start (TL;DR)

> **Requires:** Conda (recommended), Linux or Linux‑like environment, GPU optional.

```bash
# 1. Clone
git clone git@github.com:gschottlender/bioactivity-similarity-index.git && cd bioactivity-similarity-index

# 2. Create environment (exact versions from paper)
conda env create -f environment.yml
conda activate bsi_env

# 3. Register Jupyter kernel (once)
python -m ipykernel install --user --name bsi_env --display-name "Python (bsi_env)"

# 4. Launch notebooks
jupyter notebook
```

---

## Data Sources

The notebooks **query a local ChEMBL SQLite file** and build all intermediate tables automatically—no manual CSV preparation is required.

### Required file

- `chembl_33.db` (or any `chembl_XX.db`) placed in `./data/` **or** another path you set in `1_data_obtaining.ipynb`.

The file must contain the standard ChEMBL schema (`molecule_dictionary`, `activities`, `assays`, `component_sequences`, `drug_mechanism`, `molecule_structures`, …). You can obtain it from the official EBI FTP:

```bash
# Example for ChEMBL 33
wget ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBL33/chembl_33_sqlite.tar.gz
tar -xzf chembl_33_sqlite.tar.gz -C data/
```

> **Tip:** Any recent ChEMBL release (≥ v29) works, but the manuscript was produced with **ChEMBL 33** (models) and **ChEMBL 35** (held‑out validation).

---

## Preprocessing Rules (as used in the manuscript)

These steps are implemented across the notebooks and helper functions:

1. **Impute pChEMBL for negatives lacking numeric values** – assign `pchembl = 3` to compounds annotated as inactive (e.g., "Not Active", "No Activity", "No binding"). Establishes a minimum activity floor.
2. **Clip pChEMBL range** – floor at 3, cap at 10.
3. **Binary activity thresholds** –
   - Positive: `pchembl > 6.5`
   - Negative: `pchembl < 4.5` (plus decoys where applicable)
   - Intermediate values (4.5–6.5) are dropped when building classification sets.
4. **Fingerprinting** – Morgan/ECFP‑like circular fingerprints (radius = 2, length = 256 bits by default).
5. **Pair Encoding** – Compound pairs are represented by the *elementwise sum* of their fingerprints (not concatenation) unless otherwise specified.
6. **Training Domain** – Models were trained only on pairs with **Tanimoto < 0.4**. Predictions for higher‑Tanimoto pairs should be interpreted with caution.

---

## S vs N Pair Construction

We build pairwise datasets per protein (and later aggregate by protein group):

- **S pairs (similar/positive label = 1):** all unique unordered combinations of active ligands for the same protein.
- **N pairs (non‑similar/negative label = 0):** all combinations of one active compound with an inactive/decoy for the same protein.
- **Tanimoto Filter:** retain only pairs with Tanimoto < 0.4 (training domain constraint).

Pair tables minimally require columns: `prot`, `l1`, `l2`, `Tanimoto`, `y`.

---

## Ligand Diversity Reduction & Decoy Generation

Implemented in src/ligand_clustering_functions.py:

- DUD‑E‑style physicochemical filtering relative to a reference ligand.
- Bemis–Murcko scaffold clustering (select one representative scaffold).
- Butina clustering (Tanimoto‑based diversity reduction; adjustable threshold).
- Optional K‑means clustering on fingerprint vectors for further downselection.
- Multi‑ligand decoy generation with scaffold de‑duplication.

Use `cluster_ligands()` to apply Bemis–Murcko → Butina → K‑means sequentially.

---

## Model Architecture & Training

Implemented in src/model_training_functions.py:

- `NeuralNetworkModel` – shallow fully connected network with configurable hidden layers and dropout.
- Input dimensionality inferred from fingerprint length.
- Sigmoid output for binary similarity (S vs N pairs).
- `train_model_on_chunks()` – stream training over many CSV “chunks” to control memory use.
- `fine_tune_model_on_chunks()` – resume from a pretrained model; optional layer freezing and head replacement.
- `evaluate_test_data()` – convert pair table → model inputs → run inference.
- `prepare_and_evaluate_pairs()` – convenience wrapper: generate fingerprints from SMILES, compute Tanimoto, score with model.

### Memory‑Aware Training by Chunks

Large datasets are split into manageable CSV blocks (see `shuffle_and_save_chunks()` pattern). Each epoch iterates over all chunk files; each file yields its own mini‑epoch (train + quick val).

---

## Running the Notebooks

### 1_data_obtaining.ipynb

- Load raw activity data (ChEMBL or other sources).
- Normalize protein & ligand IDs.
- Impute pChEMBL for annotated negatives.
- Clip range, derive binary activity labels.
- Export cleaned tables.

### 2_dataset_assembly.ipynb

- Load cleaned data.
- Generate SMILES fingerprints.
- Cluster ligands (Bemis–Murcko, Butina, optional K‑means).
- Build S/N pair tables per protein.
- Apply Tanimoto < 0.4 filter.
- Chunk and write training/test CSVs by protein group.

### 3_train_models.ipynb

- Load chunked datasets & fingerprint db.
- Train BSI group‑specific models and the global BSI‑Large.
- Fine‑tune variants where data are sparse.
- Evaluate (ROC AUC and PR AUC) on held‑out proteins.

---


## Environment & Reproducibility

The included environment.yml pins the versions actually used for the study (Python 3.9, RDKit 2021.09.1, PyTorch 2.5.1 CUDA 12.4, etc.). Create the environment with:

```bash
conda env create -f environment.yml
conda activate bsi_env
python -m ipykernel install --user --name bsi_env --display-name "Python (bsi_env)"
```

Note:
The provided environment.yml installs PyTorch with CUDA 12.4 (the version used for all experiments in this manuscript).
If your GPU or system requires a different CUDA version,
please install the appropriate matching version of PyTorch.
See: https://pytorch.org/get-started/locally/

