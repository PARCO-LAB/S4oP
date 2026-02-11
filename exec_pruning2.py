""" import argparse
import os
import copy
import random
import torch
import gc
if torch.cuda.is_available() and not torch.cuda.is_initialized():
    torch.cuda.current_device()

from models_datasets_and_profiling_implementation.model import FineTuning, ModelTest
from models_datasets_and_profiling_implementation.model.utils import set_benchmark, set_seed
from pruning_config import *
from models_config import *

def prune_and_finetune(model_name, dataset_name, base_model_folder, checkpoint_folder):

    config = MODELS_CONFIG
    pc = PRUNING_CONFIG[model_name][dataset_name]
    pc2 = PRUNING_CONFIG
    
    os.makedirs(checkpoint_folder, exist_ok=True)
    path = os.path.join(f"./{checkpoint_folder}", f"{model_name}_{dataset_name}_pruned_{int(pc2['perc'][0]*100)}%.pth")
    if os.path.exists(path):
        raise ValueError(f"Il file {path} esiste già. Scegliere un'altra cartella o un altro nome per il file.")

    model_checkpoint = f"./{base_model_folder}/{model_name}_{dataset_name}_best.pth"

    for seed in pc2["seeds"]:
        set_seed(seed)
        set_benchmark(False)
        print(f"\n--- SEED {seed} ---")

        idx_to_remove = None

        for perc in pc2["perc"]:
            print(f"\n=== PRUNING {perc*100}% ===")
            model_test = ModelTest.from_pth(
                model_path=model_checkpoint,
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

            s4_layers = [m for m in model.modules() if m.__class__.__name__ in ["LayerS4D", "S4Block"]]
            n_layers = len(s4_layers)
            # Lista globale dei canali prunabili
            global_channels = []
            layer_channel_sizes = []

            for i, layer in enumerate(s4_layers):
                H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
                layer_channel_sizes.append(H)
                for c in range(H):
                    global_channels.append((i, c))

            total_channels = len(global_channels)

            # Numero totale di canali da prunare
            n_pruned_total = int(perc * total_channels)

            # Campionamento random globale
            pruned_channels = random.sample(global_channels, n_pruned_total)

            # Reset delle maschere
            for layer in s4_layers:
                if layer.__class__.__name__ == "LayerS4D":
                    layer.pruning_mask[:] = 1
                else:
                    layer.layer.pruning_mask[:] = 1

            # Applicazione pruning
            per_layer_count = [0] * len(s4_layers)

            for layer_idx, ch_idx in pruned_channels:
                layer = s4_layers[layer_idx]
                if layer.__class__.__name__ == "LayerS4D":
                    layer.pruning_mask[ch_idx] = 0
                else:
                    layer.layer.pruning_mask[ch_idx] = 0
                per_layer_count[layer_idx] += 1

            # Log
            for i, H in enumerate(layer_channel_sizes):
                print(f"[Layer {i}]: canali rimanenti {H - per_layer_count[i]}/{H}")

            # Fine-tuning
            trainer = FineTuning(
                model=model,
                dataset=dataset,
                epochs=pc["finetune_epochs"],
                lr=pc["lr"],
                weight_decay=pc["weight_decay"],
                checkpoint_folder=os.path.join(f"./{checkpoint_folder}", f"{model_name}_{dataset_name}_seed{seed}_pruned_{int(perc*100)}%.pth"),
                patience=pc["early_stopping"]
            )
            trainer.run()

            model_test.run()

            # Pulizia memoria
            del model_test
            del model
            del dataset
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

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
        required=True,
        help="Nome cartella dove salvare i checkpoint dei modelli prunati")
    parser.add_argument(
        "--base_model_folder", "-b",
        dest="base_model_folder",
        required=True,
        help="Nome cartella da dove caricare i modelli base")
    args = parser.parse_args()

    prune_and_finetune(args.model_name, args.dataset_name, args.base_model_folder, args.checkpoint_folder) """

import argparse
import os
import copy
import random
import torch
import gc
if torch.cuda.is_available() and not torch.cuda.is_initialized():
    torch.cuda.current_device()

from models_datasets_and_profiling_implementation.model import FineTuning, ModelTest
from models_datasets_and_profiling_implementation.model.utils import set_benchmark, set_seed
from pruning_config import *
from models_config import *

def prune_and_finetune(model_name, dataset_name, base_model_folder, checkpoint_folder):

    config = MODELS_CONFIG
    pc = PRUNING_CONFIG[model_name][dataset_name]
    pc2 = PRUNING_CONFIG
    
    os.makedirs(checkpoint_folder, exist_ok=True)
    path = os.path.join(f"./{checkpoint_folder}", f"{model_name}_{dataset_name}_pruned_{int(pc2['perc'][0]*100)}%.pth")
    if os.path.exists(path):
        raise ValueError(f"Il file {path} esiste già. Scegliere un'altra cartella o un altro nome per il file.")

    model_checkpoint = f"./{base_model_folder}/{model_name}_{dataset_name}_best.pth"

    for seed in pc2["seeds"]:
        set_seed(seed)
        set_benchmark(False)
        print(f"\n--- SEED {seed} ---")

        idx_to_remove = None
        model_test = ModelTest.from_pth(
            model_path=model_checkpoint,
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

        s4_layers = [m for m in model.modules() if m.__class__.__name__ in ["LayerS4D", "S4Block"]]
        n_layers = len(s4_layers)

        for i, layer in enumerate(s4_layers):
            H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
            model_test = ModelTest.from_pth(
                model_path=model_checkpoint,
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
            
            for j, layer2 in enumerate(s4_layers):

                if j == i:
                    n_pruned = H - 1
                else:
                    n_pruned = 0

                idx = random.sample(set(range(H)), n_pruned)
                if layer.__class__.__name__ == "LayerS4D":
                    layer.pruning_mask[idx] = 0
                else:
                    layer.layer.pruning_mask[idx] = 0    
                print(f"[Layer {i}]: canali rimanenti {H - n_pruned}/{H}")
                

            # Fine-tuning
            trainer = FineTuning(
                model=model,
                dataset=dataset,
                epochs=pc["finetune_epochs"],
                lr=pc["lr"],
                weight_decay=pc["weight_decay"],
                checkpoint_folder=os.path.join(f"./{checkpoint_folder}", f"{model_name}_{dataset_name}_seed{seed}_pruned%.pth"),
                patience=pc["early_stopping"]
            )
            trainer.run()

            model_test.run()

            # Pulizia memoria
            del model_test
            del model
            del dataset
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

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
        required=True,
        help="Nome cartella dove salvare i checkpoint dei modelli prunati")
    parser.add_argument(
        "--base_model_folder", "-b",
        dest="base_model_folder",
        required=True,
        help="Nome cartella da dove caricare i modelli base")
    args = parser.parse_args()

    prune_and_finetune(args.model_name, args.dataset_name, args.base_model_folder, args.checkpoint_folder)