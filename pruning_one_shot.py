import argparse
import os
import copy
import random
import torch
import gc
if torch.cuda.is_available() and not torch.cuda.is_initialized():
    torch.cuda.current_device()

from efficient_pruning.model import FineTuning, ModelTest, ModelInfo, ModelProfile
from efficient_pruning.model.utils import set_benchmark, set_seed
from pruning_test_config import *
from config import *

def prune_and_finetune(model_name, dataset_name, checkpoint_folder):
    
    path = os.path.join(checkpoint_folder, f"{model_name}_{dataset_name}_pruned.pth")
    if os.path.exists(path):
        raise ValueError(f"Il file {path} esiste già. Scegliere un'altra cartella o un altro nome per il file.")
    
    config = PRUNING_DEFAULT_CONFIG
    ptc= PRUNING_CONFIG[model_name][dataset_name]

    idx_to_remove_per_layer_global = None
    base_checkpoint = f"./checkpoints/{model_name}_{dataset_name}_best.pth"

    for perc in [0.1, 0.3, 0.5, 0.7, 0.9]:

        print(f"\n=== PRUNING {perc*100}% ===")

        best_acc = -1
        best_checkpoint = None
        best_idx_to_remove = None

        for seed in [7, 42, 123, 2024, 314159]:
            print(f"\n--- SEED {seed} ---")
            set_seed(seed)
            set_benchmark(False)

            # Ricarica sempre dal checkpoint base
            model_test = ModelTest.from_pth(
                model_path=base_checkpoint,
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

            # Copia pruning precedente
            if idx_to_remove_per_layer_global is None:
                idx_to_remove_seed = [[] for _ in s4_layers]
            else:
                idx_to_remove_seed = copy.deepcopy(idx_to_remove_per_layer_global)

            # Pruning incrementale
            for i, layer in enumerate(s4_layers):
                H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model

                # Numero di canali da prunare 
                if perc == 0.1:
                    H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
                    if i != 0:
                        perc2 = perc / (sum([2**j for j in range(len(s4_layers) - 1)]))
                        n_pruned = max(1, int(perc2 * H * len(s4_layers) * (2**(i-1)) + 0.5))
                    else:
                        n_pruned = 0
                elif perc == 0.3:
                    H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
                    if i == 0:
                        n_pruned = 0
                    elif i == 1:
                        n_pruned = 9
                    elif i == 2:
                        n_pruned = 19
                    elif i == 3:
                        n_pruned = 39
                    elif i == 4:
                        n_pruned = 79
                    elif i == 5:
                        n_pruned = 160
                elif perc == 0.5:
                    H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
                    if i == 0:
                        n_pruned = 0
                    elif i == 1:
                        n_pruned = 22
                    elif i == 2:
                        n_pruned = 57
                    elif i == 3:
                        n_pruned = 127
                    elif i == 4:
                        n_pruned = 102
                    elif i == 5:
                        n_pruned = 0
                elif perc == 0.7:
                    H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
                    if i == 0:
                        n_pruned = 0
                    elif i == 1:
                        n_pruned = 113
                    elif i == 2:
                        n_pruned = 117
                    elif i == 3:
                        n_pruned = 44
                    elif i == 4:
                        n_pruned = 22
                    elif i == 5:
                        n_pruned = 11
                elif perc == 0.9:
                    H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
                    if i == 0:
                        n_pruned = 118
                    elif i == 1:
                        n_pruned = 99
                    elif i == 2:
                        n_pruned = 49
                    elif i == 3:
                        n_pruned = 24
                    elif i == 4:
                        n_pruned = 12
                    elif i == 5:
                        n_pruned = 5

                available = list(set(range(H)) - set(idx_to_remove_seed[i]))
                new_idx = random.sample(available, n_pruned)
                idx_to_remove_seed[i].extend(new_idx)
                print(f"{idx_to_remove_seed[i]}")

                if layer.__class__.__name__ == "LayerS4D":
                    layer.pruning_mask[idx_to_remove_seed[i]] = 0
                else:
                    layer.layer.pruning_mask[idx_to_remove_seed[i]] = 0
                print(f"[Layer {i}]: canali rimanenti {H - len(idx_to_remove_seed[i])}/{H}")

            # Fine-tuning
            trainer = FineTuning(
                model=model,
                dataset=dataset,
                epochs=ptc["finetune_epochs"],
                lr=ptc["lr"],
                weight_decay=ptc["weight_decay"],
                checkpoint_folder=checkpoint_folder
            )
            trainer.run()

            acc = model_test.run()

            if acc > best_acc:
                best_acc = acc
                best_idx_to_remove = idx_to_remove_seed
                best_checkpoint = os.path.join(
                    checkpoint_folder,
                    f"{model_name}_{dataset_name}_pruned_{int(perc*100)}%.pth"
                )
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "pruning_masks": [
                        (layer.pruning_mask.clone().cpu()
                        if layer.__class__.__name__ == "LayerS4D"
                        else layer.layer.pruning_mask.clone().cpu())
                        for layer in s4_layers
                    ],
                    "seed": seed,
                }
                torch.save(checkpoint, best_checkpoint)

        # Pulizia memoria
        del trainer
        del model
        del model_test
        del dataset
        gc.collect()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Aggiorna stato globale
        base_checkpoint = best_checkpoint
        idx_to_remove_per_layer_global = best_idx_to_remove

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
        required=False, default=os.path.join(".", "checkpoints_pruned"),
        help="Cartella per salvare i checkpoint"
    )
    args = parser.parse_args()

    prune_and_finetune(args.model_name, args.dataset_name, args.checkpoint_folder)

#################################################################################

""" import argparse
import os
import random
import torch
if torch.cuda.is_available() and not torch.cuda.is_initialized():
    torch.cuda.current_device()

from efficient_pruning.model import FineTuning, ModelTest, ModelInfo, ModelProfile
from efficient_pruning.model.utils import set_benchmark, set_seed
from pruning_test_config import *
from config import *

set_seed(123)
set_benchmark(False)

def prune_and_finetune(model_name, dataset_name, checkpoint_folder):
    
    path = os.path.join(checkpoint_folder, f"{model_name}_{dataset_name}_pruned.pth")
    if os.path.exists(path):
        raise ValueError(f"Il file {path} esiste già. Scegliere un'altra cartella o un altro nome per il file.")
    
    config = PRUNING_DEFAULT_CONFIG
    ptc= PRUNING_CONFIG[model_name][dataset_name]
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
    idx_to_remove_per_layer = [[] for _ in range(n_layers)]
    for perc in ([0.1, 0.3, 0.5, 0.7, 0.9]):
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

        print(f"\n=== PRUNING {perc*100}% DEI CANALI ===\n")
        for i, layer in enumerate(s4_layers):

            # Numero di canali da prunare 
            if perc == 0.1:
                H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
                if i != 0:
                    perc2 = perc / (sum([2**j for j in range(n_layers - 1)]))
                    n_pruned = max(1, int(perc2 * H * len(s4_layers) * (2**(i-1)) + 0.5))
                else:
                    n_pruned = 0
            elif perc == 0.3:
                H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
                if i == 0:
                    n_pruned = 0
                elif i == 1:
                    n_pruned = 6
                elif i == 2:
                    n_pruned = 11
                elif i == 3:
                    n_pruned = 21
                elif i == 4:
                    n_pruned = 39
                elif i == 5:
                    n_pruned = 76
            elif perc == 0.5:
                H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
                if i == 0:
                    n_pruned = 0
                elif i == 1:
                    n_pruned = 10
                elif i == 2:
                    n_pruned = 24
                elif i == 3:
                    n_pruned = 29
                elif i == 4:
                    n_pruned = 60
                elif i == 5:
                    n_pruned = 11
            elif perc == 0.7:
                H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
                if i == 0:
                    n_pruned = 0
                elif i == 1:
                    n_pruned = 53
                elif i == 2:
                    n_pruned = 61
                elif i == 3:
                    n_pruned = 36
                elif i == 4:
                    n_pruned = 4
                elif i == 5:
                    n_pruned = 0
            elif perc == 0.9:
                H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
                if i == 0:
                    n_pruned = 67
                elif i == 1:
                    n_pruned = 49
                elif i == 2:
                    n_pruned = 23
                elif i == 3:
                    n_pruned = 10
                elif i == 4:
                    n_pruned = 4
                elif i == 5:
                    n_pruned = 0
            
            idx_to_remove = random.sample(list(set(range(H)) - set(idx_to_remove_per_layer[i])), n_pruned)
            idx_to_remove_per_layer[i].extend(idx_to_remove)
            print(f"{idx_to_remove_per_layer[i]}")

            # Selezione dei nuovi canali da prunare solo tra quelli attivi
            if layer.__class__.__name__ == "LayerS4D":
                layer.pruning_mask[idx_to_remove_per_layer[i]] = 0
            else:
                layer.layer.pruning_mask[idx_to_remove_per_layer[i]] = 0

            n_active = layer.pruning_mask.sum().item() if layer.__class__.__name__ == "LayerS4D" else layer.layer.pruning_mask.sum().item()
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
            vocab_size=model_test.dataset.vocab_size if hasattr(model_test.dataset, 'vocab_size') else model_test.dataset.input_shape[-1],
            seq_len=model_test.dataset.input_shape[1], 
            batch_size=config[model_name][dataset_name]["batch_size"], 
            dataset_name=dataset_name
        )
        if not os.path.exists(f"./model_info_pruned/{model_name}_{dataset_name}_torchinfo_pruned.txt"):
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
        required=False, default=os.path.join(".", "checkpoints_pruned"),
        help="Cartella per salvare i checkpoint"
    )
    args = parser.parse_args()

    prune_and_finetune(args.model_name, args.dataset_name, args.checkpoint_folder) """

# Pathfinder:
"""elif perc == 0.3:
                    H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
                    if i == 0:
                        n_pruned = 0
                    elif i == 1:
                        n_pruned = 9
                    elif i == 2:
                        n_pruned = 19
                    elif i == 3:
                        n_pruned = 39
                    elif i == 4:
                        n_pruned = 79
                    elif i == 5:
                        n_pruned = 160
                elif perc == 0.5:
                    H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
                    if i == 0:
                        n_pruned = 0
                    elif i == 1:
                        n_pruned = 22
                    elif i == 2:
                        n_pruned = 57
                    elif i == 3:
                        n_pruned = 127
                    elif i == 4:
                        n_pruned = 102
                    elif i == 5:
                        n_pruned = 0
                elif perc == 0.7:
                    H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
                    if i == 0:
                        n_pruned = 0
                    elif i == 1:
                        n_pruned = 113
                    elif i == 2:
                        n_pruned = 117
                    elif i == 3:
                        n_pruned = 44
                    elif i == 4:
                        n_pruned = 22
                    elif i == 5:
                        n_pruned = 11
                elif perc == 0.9:
                    H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
                    if i == 0:
                        n_pruned = 118
                    elif i == 1:
                        n_pruned = 99
                    elif i == 2:
                        n_pruned = 49
                    elif i == 3:
                        n_pruned = 24
                    elif i == 4:
                        n_pruned = 12
                    elif i == 5:
                        n_pruned = 5"""