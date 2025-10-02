import os
import argparse
import torch

from config import *
from efficient_pruning.model import ModelTrain
from efficient_pruning.model.utils import set_benchmark, set_seed, setup_optimizer
from efficient_pruning.model.net import NetFactory
from efficient_pruning.dataset import DatasetFactory

set_seed(42)
set_benchmark(False)

def resume_training(model_name, dataset_name, checkpoint_path, checkpoints_folder):
    config = PRUNING_DEFAULT_CONFIG

    # Controlla che il checkpoint esista
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} non trovato!")

    # Carica dataset
    dataset = DatasetFactory(dataset_name=dataset_name, batch_size=config[model_name][dataset_name]["batch_size"], valsplit=config["val_split"], num_workers=config["num_workers"]).get_dataset()
    num_classes = dataset.get_output_shape()[-1]

    # Ricrea il modello identico a quello originale
    model_train = ModelTrain.from_scratch(
        model_name=model_name,
        dataset_name=dataset_name,
        batch_size=config[model_name][dataset_name]["batch_size"],
        valsplit=config["val_split"],
        num_workers=config["num_workers"],
        d_model=config[model_name][dataset_name]["features"],
        d_state=64,
        depth=config[model_name][dataset_name]["depth"],
        dropout=config[model_name][dataset_name]["dropout"],
        norm=config[model_name][dataset_name]["norm"],
        pre_norm=config[model_name][dataset_name]["pre-norm"]
    )

    # Carica il checkpoint nello stato del modello
    model_train.model.load_state_dict(torch.load(checkpoint_path))
    model_train.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"), non_blocking=True)

    # Ricrea ottimizzatore
    optimizer, scheduler = setup_optimizer(
        model_train.model,
        lr=config[model_name][dataset_name]["lr"],
        weight_decay=config[model_name][dataset_name]["wd"],
        epochs=config[model_name][dataset_name]["epochs"]
    )
    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Riprendi il training
    model_train.run(optimizer, scheduler, criterion, config[model_name][dataset_name]["epochs"], checkpoints_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Training from Checkpoint")
    parser.add_argument("--model", "-m", required=True, help="Model name")
    parser.add_argument("--dataset", "-d", required=True, help="Dataset name")
    parser.add_argument("--checkpoint", "-c", required=True, help="Checkpoint path to resume from")
    parser.add_argument("--checkpoints-folder", "-f", default="./checkpoints", help="Folder to save checkpoints")
    args = parser.parse_args()

    resume_training(
        args.model,
        args.dataset,
        args.checkpoint,
        args.checkpoints_folder
    )
