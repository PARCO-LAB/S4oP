import argparse
import os
import torch
import random

from efficient_pruning.prune.prune_test2 import prune_random_channels
from efficient_pruning.model import FineTuning, ModelTest, ModelInfo, ModelProfile
from efficient_pruning.model.utils import get_device
from efficient_pruning.model.utils import set_benchmark, set_seed
from pruning_test_config import *
from config import *

set_seed(42)
set_benchmark(False)

def prune_and_finetune(model_name, dataset_name, checkpoint_folder):
    
    path = os.path.join(checkpoint_folder, f"{model_name}_{dataset_name}_pruned.pth")
    if os.path.exists(path):
        raise ValueError(f"Il file {path} esiste già. Scegliere un'altra cartella o un altro nome per il file.")
    
    config = PRUNING_DEFAULT_CONFIG
    ptc= PRUNING_CONFIG[dataset_name]
    model_test = ModelTest.from_pth(
        model_path=f"./checkpoints/{model_name}_{dataset_name}_best.pth",
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
    model = model_test.model
    dataset = model_test.dataset

    # Individua tutti i LayerS4D
    s4_layers = [m for m in model.modules() if m.__class__.__name__ == "LayerS4D" or m.__class__.__name__ == "S4Block"]
    n_layers = len(s4_layers)
    print(f"Trovati {n_layers} layer s4/s4d nel modello.")

    for i, layer in enumerate(s4_layers):

        H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model

        # Calcola il numero di canali da prunare
        n_pruned = int(H * 0.999)
        n_active = H - n_pruned

        # Selezione dei nuovi canali da prunare solo tra quelli attivi
        idx_to_remove = random.sample(range(H), n_pruned)
        if layer.__class__.__name__ == "LayerS4D":
            layer.pruning_mask[idx_to_remove] = 0  
        else:
            layer.layer.pruning_mask[idx_to_remove] = 0

        print(f"[Layer {i}]: canali rimanenti {n_active}/{H}")

    print("=== FINE-TUNING ===")
    trainer = FineTuning(
        model=model,
        dataset=dataset,
        epochs=ptc["finetune_epochs"],
        lr=ptc["lr"],
        weight_decay=ptc["weight_decay"],
        checkpoint_folder=checkpoint_folder
    )
    trainer.run()

    print("=== TESTING FINALE ===")
    model_test.run()

    # Model Info
    model_info = ModelInfo(
        model=model_test.model, 
        vocab_size=model_test.dataset.vocab_size if hasattr(model_test.dataset, 'vocab_size') else None,
        seq_len=model_test.dataset.input_shape[1], 
        batch_size=config[model_name][dataset_name]["batch_size"], 
        dataset_name=dataset_name
    )
    if not os.path.exists(f"./model_info_ppruned/{model_name}_{dataset_name}_torchinfo_pruned.txt"):
        model_info.torchinfo(output_dir="model_info_pruned", mode="pruned")
        print("\nModel info salvato nella cartella 'model_info_pruned'")
    else:
        print("\nModel info già esistente nella cartella 'model_info_pruned', salto la creazione")

    # Model Profile
    print("\nStarting model profiling...")
    
    model_profile = ModelProfile(iterations=100)
    model_profile.add("prova")
    model_profile.run("prova", model_test.model, model_info.get_example_input, iterations=100)
    model_profile.info("prova")

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
        "--checkpoint_folder", "-c",
        dest="checkpoint_folder",
        required=False, default=os.path.join(".", "checkpoints_pruned2"),
        help="Cartella per salvare i checkpoint"
    )
    args = parser.parse_args()

    prune_and_finetune(args.model_name, args.dataset_name, args.checkpoint_folder)