import os
import argparse
import torch
if torch.cuda.is_available() and not torch.cuda.is_initialized():
    torch.cuda.current_device()

from models_config import *
from models_datasets_and_profiling_implementation.model import ModelTrain, ModelTest, ModelInfo, ModelProfile
from models_datasets_and_profiling_implementation.model.utils import set_benchmark, set_seed, get_device
from models_datasets_and_profiling_implementation.model.net import NetFactory

set_seed(42)
set_benchmark(False)

def main(model_name, dataset_name, checkpoints_folder, pruned_model_name):

    # Caricamento configurazione
    config = MODELS_CONFIG

    # Creazione path modello
    if pruned_model_name is None:
        model_path = os.path.join(f"./{checkpoints_folder}", f"{model_name}_{dataset_name}_best.pth")
    else:
        model_path = os.path.join(f"./{checkpoints_folder}", f"{pruned_model_name}.pth")

    # Metadati in base al dataset
    if dataset_name == "imdb":
        vocab_size = 30522
        seq_len = 4096
        num_classes = 2
    elif dataset_name == "listops":
        vocab_size = 18
        seq_len = 5995
        num_classes = 10
    elif dataset_name == "pathfinder":
        vocab_size = 1
        seq_len = 1024
        num_classes = 2
    elif dataset_name == "ecg":
        vocab_size = 12
        seq_len = 4096
        num_classes = 6
    elif dataset_name == "image":
        vocab_size = 1
        seq_len = 1024
        num_classes = 10
    elif dataset_name == "retrieval":
        vocab_size = 256
        seq_len = 4096
        num_classes = 2
    else:
        raise FileNotFoundError(f"Dataset {dataset_name} does not exist.")

    # Se il modello non esiste
    if (not os.path.exists(model_path)) and (pruned_model_name is None):
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    
    # Se il modello esiste ed è stato specificato un modello prunato
    elif pruned_model_name is not None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pruned model path {model_path} does not exist.")

        print(f"\nLoading pruned model from path {model_path}...")

        ckpt = torch.load(model_path, map_location=get_device())
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            active_idx_layers = ckpt["active_idx_layers"]
        else:
            active_idx_layers = None

        model = NetFactory(
            model_name=model_name,
            dataset_name=dataset_name,
            vocab_size=vocab_size,
            d_model=config[f"{model_name}"][f"{dataset_name}"]["features"], 
            d_state=config[f"{model_name}"][f"{dataset_name}"]["d_state"],
            depth=config[f"{model_name}"][f"{dataset_name}"]["depth"],
            dropout=config[f"{model_name}"][f"{dataset_name}"]["dropout"],
            num_classes=num_classes,
            norm=config[f"{model_name}"][f"{dataset_name}"]["norm"],
            pre_norm=config[f"{model_name}"][f"{dataset_name}"]["pre-norm"],
            active_idx_layers=active_idx_layers
        ).get_net()

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        else:    
            model.load_state_dict(torch.load(model_path, map_location=get_device()), strict=True)

        print("\nStarting model profiling...")
        # Model Info
        model_info = ModelInfo(
            model=model, 
            vocab_size=vocab_size,
            seq_len=seq_len, 
            batch_size=100, 
            dataset_name=dataset_name
        )

        # Model Profile
        with torch.no_grad():
            model_profile = ModelProfile(iterations=100)
            model_profile.add("prova")
            model_profile.run("prova", model, model_info.get_example_input, iterations=100)
            model_profile.info("prova")
    
    # Se il modello esiste (e non è stato specificato un modello prunato)
    else: 
        print(f"\nLoading base model from path {model_path}...")

        ckpt = torch.load(model_path, map_location=get_device())
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            active_idx_layers = ckpt["active_idx_layers"]
        else:
            active_idx_layers = None

        # Testing
        model = NetFactory(
            model_name=model_name,
            dataset_name=dataset_name,
            vocab_size=vocab_size,
            d_model=config[f"{model_name}"][f"{dataset_name}"]["features"], 
            d_state=config[f"{model_name}"][f"{dataset_name}"]["d_state"],
            depth=config[f"{model_name}"][f"{dataset_name}"]["depth"],
            dropout=config[f"{model_name}"][f"{dataset_name}"]["dropout"],
            num_classes=num_classes,
            norm=config[f"{model_name}"][f"{dataset_name}"]["norm"],
            pre_norm=config[f"{model_name}"][f"{dataset_name}"]["pre-norm"],
            active_idx_layers=active_idx_layers
        ).get_net()

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        else:    
            model.load_state_dict(torch.load(model_path, map_location=get_device()), strict=True)

        print("\nStarting model profiling...")
        # Model Info
        model_info = ModelInfo(
            model=model, 
            vocab_size=vocab_size,
            seq_len=seq_len, 
            batch_size=100, 
            dataset_name=dataset_name
        )

        # Model Profile    
        with torch.no_grad():    
            model_profile = ModelProfile(iterations=100)
            model_profile.add("prova")
            model_profile.run("prova", model, model_info.get_example_input, iterations=100)
            model_profile.info("prova")

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument(
        "--model", "-m", 
        dest="model_name", 
        required=True,
        help="Model name")
    parser.add_argument(
        "--dataset", "-d", 
        dest="dataset_name", 
        required=True,
        help="Dataset name")
    parser.add_argument(
        "--checkpoints-folder", "-f", 
        dest="checkpoints_folder", 
        required=True, default=os.path.join(".", "checkpoints"),
        help="Checkpoints folder")
    parser.add_argument(
        "--pruned-model", "-p",
        dest="pruned_model_name",
        required=False, default=None,
        help="Pruned model index")
    args = parser.parse_args()

    main(args.model_name, args.dataset_name, args.checkpoints_folder, args.pruned_model_name)