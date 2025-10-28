import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch

from evaluate_bsi_pairs import ecfp4_from_smiles, load_model_params_from_json, parse_hidden_layers
from train_BSI_model import shuffle_and_save_chunks
from src.model_training_functions import fine_tune_model_on_chunks, NeuralNetworkModel


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help='Input data to perform fine-tuning. Must have columns l1|l2|y')
    parser.add_argument("--model_path", type=str, required=True, help='Directory of the pre-trained model to perform fine-tuning')
    parser.add_argument("--model_out", type=str, required=True, help='Directory to save fine-tuned model and parameters')
    parser.add_argument("--train_dir", type=str, required=True, help='Directory to save training provisional chunked data')
    parser.add_argument("--chunk_prefix", type=str, default="chunk")
    parser.add_argument("--num_chunks", type=int, default=5, help="Number of chunks to divide dataset after converting to features vector (default: 5)")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--freeze_until_layer", type=int, default=1,
                        help="Numbers of layers to freeze during training")
    parser.add_argument("--n_epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--dropout_prob", type=float, default=0.5,
                        help="Dropout probability during training")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate during training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    return parser.parse_args()


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    global log
    log = logging.getLogger(__name__)


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    # 1) Load model parameters
    base_params = load_model_params_from_json(args.model_path)
    hidden_layers = base_params["hidden_layers"]
    dropout = float(base_params["dropout"])
    fp_bits = int(base_params["fp_bits"])

    log.info(f"Model params → hidden_layers={hidden_layers} | dropout={dropout} | fp_bits={fp_bits}")

    # 2) Load input pairs
    pairs = pd.read_csv(args.input_csv)
    for col in ("l1", "l2", "y"):
        if col not in pairs.columns:
            raise SystemExit(f"Input CSV must contain column '{col}'")

    tot_ligs = pd.unique(pairs[['l1', 'l2']].values.ravel())
    db_ligs = {s: ecfp4_from_smiles(s,n_bits=fp_bits) for s in tot_ligs}

    desired_chunks = args.num_chunks if args.num_chunks and args.num_chunks > 0 else 30
    num_chunks = min(desired_chunks, len(pairs))

    shuffle_and_save_chunks(
        df=pairs,
        output_folder=args.train_dir,
        num_chunks=num_chunks,
        prefix=args.chunk_prefix,
        random_state=args.random_seed,
    )

    # 3) Load model weights
    model = NeuralNetworkModel(hidden_layers=hidden_layers, dropout_prob=dropout, input_size=fp_bits, output_size=1)
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)

    # 4) Fine-tune model
    log.info(f"Fine tuning model")
    ft_model = fine_tune_model_on_chunks(
        args.train_dir,
        db_ligs,
        model,
        freeze_until_layer=args.freeze_until_layer,
        n_epochs=args.n_epochs,
        dropout_prob=args.dropout_prob,
        lr=args.lr,
        batch_size=args.batch_size
    )

    # 5) Save model and parameters
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ft_model.state_dict(), str(args.model_out))

    model_params = {
        "hidden_layers": hidden_layers,
        "dropout": args.dropout_prob,
        "random_seed": args.random_seed,
        "fp_bits": fp_bits
    }

    params_path = Path(args.model_out).with_suffix(".params.json")
    with open(params_path, "w") as f:
        json.dump(model_params, f, indent=4)

    log.info(f"Model parameters saved → {params_path}")
    return 0


if __name__ == "__main__":
    main()
    

