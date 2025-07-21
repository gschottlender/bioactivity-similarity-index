import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import numpy as np
import gc

import os
import pandas as pd

import random

from src.ligand_clustering_functions import compute_tanimoto

from rdkit.Chem import MolFromSmiles, AllChem

def convert_compound_pairs(pairs, ligs_fps_db, sim='y'):
    """
    Converts compound pairs into the sum of their corresponding fingerprints.

    Parameters
    ----------
    pairs : pd.DataFrame
        DataFrame containing compound pairs with columns ['l1', 'l2'] and a label column.
    ligs_fps_db : dict
        Dictionary mapping each compound to its fingerprint vector.
    sim : str, optional
        Name of the column with the similarity label (default: 'y').

    Returns
    -------
    X : np.ndarray
        Matrix where each row is the sum of the fingerprints of a compound pair.
    y : np.ndarray
        Array of labels (e.g., 1 for similar, 0 for non-similar).
    """
    n_rows = len(pairs)
    n_features = len(next(iter(ligs_fps_db.values())))  # Size of the fingerprint vector
    X = np.zeros((n_rows, n_features), dtype=np.float32)  # Preallocated matrix
    y = np.array(pairs[sim], dtype=np.float32)

    # Process each row
    for i, (l1, l2) in enumerate(zip(pairs['l1'], pairs['l2'])):
        X[i] = ligs_fps_db[l1] + ligs_fps_db[l2]

    return X, y

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_prob):
        super(NeuralNetworkModel, self).__init__()
        
        layers = []
        
        # First hidden layer
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_prob))
        
        # Intermediate hidden layers
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
        
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        layers.append(nn.Sigmoid())
        
        # Combine all layers into a Sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def create_sum_model_inputs(X, y, batch_size=32, max_ram_gb=10, valid_size=0.05):
    """
    Given a (sub)dataset X, y (as NumPy arrays), split and return DataLoaders for training and validation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (NumPy array).
    y : np.ndarray
        Target labels (NumPy array).
    batch_size : int, optional
        Batch size for DataLoaders (default: 32).
    max_ram_gb : float, optional
        Maximum allowed RAM usage in GB (default: 10). [Currently unused: sampling is commented out.]
    valid_size : float, optional
        Fraction of the data to use for validation (default: 0.05).

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training set.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation set.
    """

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=valid_size, random_state=10
    )

    # Free original arrays
    del X, y
    gc.collect()

    # Ensure float32, contiguous arrays for PyTorch
    X_train = np.ascontiguousarray(X_train.astype(np.float32, copy=False))
    y_train = np.ascontiguousarray(y_train.astype(np.float32, copy=False))
    X_val   = np.ascontiguousarray(X_val.astype(np.float32, copy=False))
    y_val   = np.ascontiguousarray(y_val.astype(np.float32, copy=False))

    # Convert to tensors
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    X_val_tensor   = torch.from_numpy(X_val)
    y_val_tensor   = torch.from_numpy(y_val)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset   = TensorDataset(X_val_tensor, y_val_tensor)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Clean up intermediate variables
    del X_train, y_train, X_val, y_val
    del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
    gc.collect()

    return train_loader, val_loader

def train_model_on_chunks(
    dataset_dir,
    lig_fps_db,               # or any resource needed for conv_sum
    hidden_layers=[256, 256],
    dropout_prob=0.5,
    n_epochs=10,
    batch_size=32
):
    """
    Creates and trains a model by iterating over multiple chunk files.
    Each epoch consists of going through all the files, training mini-epochs on each chunk.
    Returns the trained model at the end.
    
    Parameters
    ----------
    dataset_dir : str
        Directory containing chunked training files (e.g., CSVs).
    lig_fps_db : dict
        Dictionary of ligand fingerprints (used in conv_sum).
    hidden_layers : list of int, optional
        Sizes of the hidden layers (default: [256, 256]).
    dropout_prob : float, optional
        Dropout probability (default: 0.5).
    n_epochs : int, optional
        Number of epochs to train (default: 10).
    batch_size : int, optional
        Batch size for DataLoader (default: 32).
    
    Returns
    -------
    model : nn.Module
        Trained neural network model.
    """

    # 1) Initialize model, criterion, optimizer

    # To determine input_size, you can read or generate a small sample, or assume you know it.
    # Example: read a small chunk to get X, y
    # df_temp = pd.read_csv(file_list[0])
    # X_temp, y_temp = conv_sum(df_temp, lig_fps_db)
    input_size = len(next(iter(lig_fps_db.values())))

    model = NeuralNetworkModel(input_size, hidden_layers, 1, dropout_prob)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    gc.collect()

    # 2) Training loop
    for epoch in range(n_epochs):
        print(f"\n=== Epoch {epoch+1}/{n_epochs} ===")
        
        epoch_running_loss = 0.0
        epoch_size = 0  # For average loss over the whole epoch

        # Get list of chunk files in random order
        file_list = os.listdir(dataset_dir)
        random.shuffle(file_list)
        
        for filename in file_list:
            # 2.1) Read the chunk and convert to model inputs
            df_chunk = pd.read_csv(f'{dataset_dir}/{filename}')
            X_chunk, y_chunk = convert_compound_pairs(df_chunk, lig_fps_db)

            # 2.2) Create DataLoaders for train/val for this chunk
            train_loader, val_loader = create_sum_model_inputs(
                X_chunk,
                y_chunk,
                batch_size=batch_size
            )

            # 2.3) Train (mini-epoch) on this chunk
            model.train()
            running_loss_chunk = 0.0
            samples_in_chunk = 0

            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                outputs = outputs.view(-1)   # [batch_size]
                targets = targets.view(-1)   # [batch_size]

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                actual_batch_size = inputs.size(0)
                running_loss_chunk += loss.item() * actual_batch_size
                samples_in_chunk += actual_batch_size

            # Mean loss for this chunk
            chunk_loss = running_loss_chunk / samples_in_chunk if samples_in_chunk > 0 else 0.0

            # 2.4) Validation loss for this chunk (optional quick tracking)
            model.eval()
            val_loss_chunk = 0.0
            val_samples_chunk = 0
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)

                    val_outputs = model(val_inputs).view(-1)
                    val_targets = val_targets.view(-1)

                    loss_val = criterion(val_outputs, val_targets)
                    bs_val = val_inputs.size(0)
                    val_loss_chunk += loss_val.item() * bs_val
                    val_samples_chunk += bs_val

            val_loss_chunk = val_loss_chunk / val_samples_chunk if val_samples_chunk > 0 else 0.0

            print(f" - {filename} | Train Loss: {chunk_loss:.4f}, Val Loss: {val_loss_chunk:.4f}")

            # Accumulate for global epoch average
            epoch_running_loss += running_loss_chunk
            epoch_size += samples_in_chunk

            # Free chunk memory
            del df_chunk, X_chunk, y_chunk, train_loader, val_loader
            gc.collect()

        # At the end of all files, complete 1 epoch over all data
        epoch_loss = epoch_running_loss / epoch_size if epoch_size > 0 else 0.0
        print(f"=== End of epoch {epoch+1}/{n_epochs} | Avg Training Loss: {epoch_loss:.4f} ===")

    return model

def fine_tune_model_on_chunks(
    dataset_dir,
    lig_fps_db,                     # fingerprint dictionary used by convert_compound_pairs
    pretrained_model,               # path to .pt file or an nn.Module instance
    freeze_until_layer=None,        # index up to which layers are frozen (None = no freezing)
    new_hidden_layers=None,         # if you want to replace/add new classifier layers
    dropout_prob=0.5,
    lr=1e-5,
    n_epochs=3,
    batch_size=32
):
    """
    Fine‑tune a pre‑trained model on chunked datasets.

    Parameters
    ----------
    dataset_dir : str
        Directory containing chunked CSV files for the new task.
    lig_fps_db : dict
        Dictionary mapping ligand IDs to fingerprint vectors.
    pretrained_model : str or nn.Module
        File path to a saved model (.pt/.pth) or an already‑loaded nn.Module.
    freeze_until_layer : int or None, optional
        Index of the last layer to freeze (0‑based). Use None for no freezing (default).
    new_hidden_layers : list of int or None, optional
        If provided, replaces the classifier head with a new MLP of these sizes.
    dropout_prob : float, optional
        Dropout probability for a new head (if created).
    lr : float, optional
        Learning rate for fine‑tuning (default: 1e‑5).
    n_epochs : int, optional
        Number of fine‑tuning epochs (default: 3).
    batch_size : int, optional
        Batch size (default: 32).

    Returns
    -------
    model : nn.Module
        Fine‑tuned model.
    """

    # ------------------------------------------------------------------
    # 1. Load / prepare the model
    # ------------------------------------------------------------------
    if isinstance(pretrained_model, str):
        model = torch.load(pretrained_model, map_location="cpu")
    else:
        model = pretrained_model  # already an nn.Module

    # Optionally freeze layers
    if freeze_until_layer is not None:
        for idx, (name, param) in enumerate(model.named_parameters()):
            param.requires_grad = idx > freeze_until_layer

    # Optionally replace / extend the classifier head
    if new_hidden_layers is not None:
        input_size = len(next(iter(lig_fps_db.values())))
        new_head = NeuralNetworkModel(
            input_size=input_size,
            hidden_layers=new_hidden_layers,
            output_size=1,
            dropout_prob=dropout_prob
        )
        model.model = new_head  # assumes the original model has .model as final head
        # All new parameters require grad by default

    # Loss & optimiser (tune only parameters with requires_grad = True)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gc.collect()

    # ------------------------------------------------------------------
    # 2. Fine‑tuning loop over epochs and chunks
    # ------------------------------------------------------------------
    for epoch in range(n_epochs):
        print(f"\n=== Fine‑tuning Epoch {epoch+1}/{n_epochs} ===")
        epoch_running_loss, epoch_size = 0.0, 0

        file_list = os.listdir(dataset_dir)
        random.shuffle(file_list)

        for filename in file_list:
            df_chunk = pd.read_csv(f"{dataset_dir}/{filename}")
            X_chunk, y_chunk = convert_compound_pairs(df_chunk, lig_fps_db)

            train_loader, val_loader = create_sum_model_inputs(
                X_chunk, y_chunk, batch_size=batch_size
            )

            # ---------- train mini‑epoch on current chunk ----------
            model.train()
            running_loss_chunk, samples_in_chunk = 0.0, 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()

                outputs = model(inputs).view(-1)
                loss = criterion(outputs, targets.view(-1))
                loss.backward()
                optimizer.step()

                bs = inputs.size(0)
                running_loss_chunk += loss.item() * bs
                samples_in_chunk += bs

            chunk_loss = running_loss_chunk / samples_in_chunk if samples_in_chunk else 0.0

            # ---------- quick validation on this chunk ----------
            model.eval()
            val_loss_chunk, val_samples = 0.0, 0
            with torch.no_grad():
                for v_inputs, v_targets in val_loader:
                    v_inputs, v_targets = v_inputs.to(device), v_targets.to(device)
                    v_outputs = model(v_inputs).view(-1)
                    v_loss = criterion(v_outputs, v_targets.view(-1))
                    val_loss_chunk += v_loss.item() * v_inputs.size(0)
                    val_samples += v_inputs.size(0)

            val_loss_chunk = val_loss_chunk / val_samples if val_samples else 0.0
            print(f" - {filename} | Train Loss: {chunk_loss:.4f} | Val Loss: {val_loss_chunk:.4f}")

            epoch_running_loss += running_loss_chunk
            epoch_size += samples_in_chunk

            # Clean up
            del df_chunk, X_chunk, y_chunk, train_loader, val_loader
            gc.collect()

        epoch_loss = epoch_running_loss / epoch_size if epoch_size else 0.0
        print(f"=== End of Epoch {epoch+1}/{n_epochs} | Avg Train Loss: {epoch_loss:.4f} ===")

    return model

def evaluate_test_data(model,test_data,db_ligs):
    model.eval()
    X_test,y_test = convert_compound_pairs(test_data,db_ligs)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    device = next(model.parameters()).device
    X_test = X_test.to(device)
    with torch.no_grad():
        preds = model(X_test).flatten().cpu().numpy()
    return preds

def prepare_and_evaluate_pairs(new_pairs, model, fp_size=256):
    """
    Generate ligand fingerprints from SMILES, compute Tanimoto similarity,
    and evaluate predictions for compound pairs.

    Parameters
    ----------
    new_pairs : pd.DataFrame
        DataFrame containing at least columns ['l1', 'l2'] with SMILES strings.
    model : nn.Module or callable
        Trained model used in evaluate_test_data.
    fp_size : int, optional
        Size of the Morgan fingerprint (default: 256).

    Returns
    -------
    new_pairs : pd.DataFrame
        DataFrame with Tanimoto similarity and model predictions added.
    """

    # 1. Collect all unique SMILES in both 'l1' and 'l2'
    unique_smiles = set(new_pairs['l1']).union(set(new_pairs['l2']))

    # 2. Generate fingerprint dictionary for each unique SMILES
    fps_dict = {
        smi: AllChem.GetMorganFingerprintAsBitVect(MolFromSmiles(smi), 2, fp_size)
        for smi in unique_smiles
    }
    # 3. Compute Tanimoto similarity for all pairs
    new_pairs['Tanimoto'] = compute_tanimoto(list(zip(new_pairs['l1'], new_pairs['l2'])), fps_dict)

    # 4. Set default label to 0 (provisory)
    new_pairs['y'] = 0

    # 5. Generate predictions using the model
    fps_dict = {l:np.array(fps_dict[l], dtype=np.float32) for l in fps_dict}
    preds = evaluate_test_data(model, new_pairs, fps_dict)
    new_pairs['pred'] = preds

    # 6. Remove the temporary 'y' column
    new_pairs = new_pairs.drop(columns=['y'])

    return new_pairs