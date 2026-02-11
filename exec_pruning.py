import argparse
import os
import copy
import random
import torch
import gc
import time
if torch.cuda.is_available() and not torch.cuda.is_initialized():
    torch.cuda.current_device()

from models_datasets_and_profiling_implementation.model import FineTuning, ModelTest
from models_datasets_and_profiling_implementation.model.utils import set_benchmark, set_seed
from pruning_config import *
from models_config import *

def pretty_time(time_s):
    time_s = int(round(time_s))

    if time_s < 60:
        return f"{time_s} sec"
    elif time_s < 3600:
        minutes = time_s // 60
        seconds = time_s % 60
        return f"{minutes} min {seconds} sec"
    else:
        hours = time_s // 3600
        minutes = (time_s % 3600) // 60
        seconds = time_s % 60
        return f"{hours} hr {minutes} min {seconds} sec"

def prune_and_finetune(model_name, dataset_name, base_model_folder, checkpoint_folder):

    config = MODELS_CONFIG
    pc = PRUNING_CONFIG[model_name][dataset_name]
    pc2 = PRUNING_CONFIG
    
    os.makedirs(checkpoint_folder, exist_ok=True)
    path = os.path.join(f"./{checkpoint_folder}", f"{model_name}_{dataset_name}_pruned_{int(pc2['perc'][0]*100)}%.pth")
    if os.path.exists(path):
        raise ValueError(f"Il file {path} esiste già. Scegliere un'altra cartella o un altro nome per il file.")

    idx_to_remove_per_layer_global = None
    model_checkpoint = f"./{base_model_folder}/{model_name}_{dataset_name}_best.pth"

    accuracies = {}
    best_seeds = {}
    timers = {}

    for perc in pc2["perc"]:

        print(f"\n=== PRUNING {perc*100}% ===")

        best_acc = -1
        best_seed = None
        best_checkpoint = None
        best_idx_to_remove = None
        timers[perc] = {}

        for seed in pc2["seeds"]:
            t1 = time.time()
            print(f"\n--- SEED {seed} ---")
            set_seed(seed)
            set_benchmark(False)

            # Ricarica sempre dal checkpoint base
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

            # Copia pruning precedente
            if idx_to_remove_per_layer_global is None:
                idx_to_remove_seed = [[] for _ in s4_layers]
            else:
                idx_to_remove_seed = copy.deepcopy(idx_to_remove_per_layer_global)

            # Pruning incrementale
            for i, layer in enumerate(s4_layers):
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
                epochs=pc["finetune_epochs"],
                lr=pc["lr"],
                weight_decay=pc["weight_decay"],
                checkpoint_folder=os.path.join(f"./{checkpoint_folder}", f"{model_name}_{dataset_name}_seed{seed}_pruned_{int(perc*100)}%.pth"),
                patience=pc["early_stopping"]
            )
            trainer.run()

            acc = model_test.run()

            if acc > best_acc:
                best_seed = seed
                best_acc = acc
                best_idx_to_remove = idx_to_remove_seed
                best_checkpoint = os.path.join(f"./{checkpoint_folder}", f"{model_name}_{dataset_name}_pruned_{int(perc*100)}%.pth")
                torch.save(model.state_dict(), best_checkpoint)

            t2 = time.time()
            timers[perc][seed] = t2 - t1
            # Pulizia memoria
            del trainer
            del model
            del model_test
            del dataset
            gc.collect()

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Aggiorna stato globale
        accuracies[perc] = best_acc
        best_seeds[perc] = best_seed
        model_checkpoint = best_checkpoint
        idx_to_remove_per_layer_global = best_idx_to_remove

        # Rimuovi checkpoint intermedi
        seed_folder = os.path.join(f"./{checkpoint_folder}")
        for filename in os.listdir(seed_folder):
            if "seed" in filename:
                file_path = os.path.join(seed_folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    raise ValueError(f"Il percorso {file_path} non è un file.")

    print(f"\n=== ACCURACY MIGLIORI PER OGNI STEP ===")
    for perc, acc in accuracies.items():
        print(f"    - Pruning {int(perc*100)}%: {acc:.2f}%. Best seed: {best_seeds[perc]}")
    print(f"\n=== TEMPI DI ESECUZIONE ===")
    for perc, seed_times in timers.items():
        tot = sum(seed_times.values())
        print(f"= Pruning {int(perc*100)}% =")
        print(f"Tempo totale: {pretty_time(tot)}")
        print(f"Tempo medio: {pretty_time(tot / len(seed_times))}")
        for seed, t in seed_times.items():
            print(f"    - Seed {seed}: {pretty_time(t)}")
        

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

    t1 = time.time()
    prune_and_finetune(args.model_name, args.dataset_name, args.base_model_folder, args.checkpoint_folder)
    t2 = time.time()
    print(f"\nTempo totale di esecuzione: {pretty_time(t2 - t1)}")

"""import argparse
import os
import copy
import random
import torch
import gc
import time
if torch.cuda.is_available() and not torch.cuda.is_initialized():
    torch.cuda.current_device()

from models_datasets_and_profiling_implementation.model import FineTuning, ModelTest
from models_datasets_and_profiling_implementation.model.utils import set_benchmark, set_seed
from pruning_config import *
from models_config import *

def pretty_time(time_s):
    time_s = int(round(time_s))

    if time_s < 60:
        return f"{time_s} sec"
    elif time_s < 3600:
        minutes = time_s // 60
        seconds = time_s % 60
        return f"{minutes} min {seconds} sec"
    else:
        hours = time_s // 3600
        minutes = (time_s % 3600) // 60
        seconds = time_s % 60
        return f"{hours} hr {minutes} min {seconds} sec"
    
def exponential_pruning_allocation(perc, num_layers, channels_per_layer, base=2, p_unlock=0.8):
    alpha = min(1.0, perc / p_unlock)

    weights = [alpha if i == 0 else base ** i for i in range(num_layers)]

    total_channels = num_layers * channels_per_layer
    total_pruned = int(round(perc * total_channels))

    weight_sum = sum(weights)

    pruned_per_layer = [min(channels_per_layer - 1, int(round(total_pruned * weights[i] / weight_sum))) for i in range(num_layers)]

    return pruned_per_layer

def prune_and_finetune(model_name, dataset_name, base_model_folder, checkpoint_folder):

    config = MODELS_CONFIG
    pc = PRUNING_CONFIG[model_name][dataset_name]
    pc2 = PRUNING_CONFIG
    
    os.makedirs(checkpoint_folder, exist_ok=True)
    path = os.path.join(f"./{checkpoint_folder}", f"{model_name}_{dataset_name}_pruned_{int(pc2['perc'][0]*100)}%.pth")
    if os.path.exists(path):
        raise ValueError(f"Il file {path} esiste già. Scegliere un'altra cartella o un altro nome per il file.")

    idx_to_remove_per_layer_global = None
    base_checkpoint = f"./{base_model_folder}/{model_name}_{dataset_name}_best.pth"

    accuracies = {}
    best_seeds = {}
    timers = {}

    for perc in pc2["perc"]:

        print(f"\n=== PRUNING {perc*100}% ===")

        best_acc = -1
        best_seed = None
        best_checkpoint = None
        best_idx_to_remove = None

        # Inizializzazione numero di canali da rimuovere per layer
        L = config[model_name][dataset_name]["depth"]
        H = config[model_name][dataset_name]["features"]
        pruned_per_layer = exponential_pruning_allocation(perc, L, H, base=2, p_unlock=0.8)

        for seed in pc2["seeds"]:
            t1 = time.time()
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
                pruned_per_layer_copy = copy.deepcopy(pruned_per_layer)
            else:
                idx_to_remove_seed = copy.deepcopy(idx_to_remove_per_layer_global)
                pruned_per_layer_copy = [pruned_per_layer[i] - len(idx_to_remove_seed[i]) for i in range(len(s4_layers))]

            # Pruning incrementale
            for i, layer in enumerate(s4_layers):
                H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
                n_pruned = pruned_per_layer_copy[i]
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
                epochs=pc["finetune_epochs"],
                lr=pc["lr"],
                weight_decay=pc["weight_decay"],
                checkpoint_folder=os.path.join(f"./{checkpoint_folder}", f"{model_name}_{dataset_name}_seed{seed}_pruned_{int(perc*100)}%.pth"),
                patience=pc["early_stopping"]
            )
            trainer.run()

            acc = model_test.run()

            if acc > best_acc:
                best_seed = seed
                best_acc = acc
                best_idx_to_remove = idx_to_remove_seed
                best_checkpoint = os.path.join(f"./{checkpoint_folder}", f"{model_name}_{dataset_name}_pruned_{int(perc*100)}%.pth")
                torch.save(model.state_dict(), best_checkpoint)

            t2 = time.time()
            timers[perc][seed] = t2 - t1
            # Pulizia memoria
            del trainer
            del model
            del model_test
            del dataset
            gc.collect()

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Aggiorna stato globale
        accuracies[perc] = best_acc
        best_seeds[perc] = best_seed
        base_checkpoint = best_checkpoint
        idx_to_remove_per_layer_global = best_idx_to_remove

        # Rimuovi checkpoint intermedi
        seed_folder = os.path.join(f"./{checkpoint_folder}")
        for filename in os.listdir(seed_folder):
            if "seed" in filename:
                file_path = os.path.join(seed_folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                else:
                    raise ValueError(f"Il percorso {file_path} non è un file.")

    print(f"\n=== ACCURACY MIGLIORI PER OGNI STEP ===")
    for perc, acc in accuracies.items():
        print(f"    - Pruning {int(perc*100)}%: {acc}. Best seed: {best_seeds[perc]}")
    print(f"\n=== TEMPI DI ESECUZIONE ===")
    for perc, seed_times in timers.items():
        tot = sum(seed_times.values())
        print(f"Tempo totale per pruning {int(perc*100)}%: {pretty_time(tot)}")
        print(f"Tempo medio per seed: {pretty_time(tot / len(seed_times))}")
        for seed, time in seed_times.items():
            print(f"    - Pruning {int(perc*100)}%, Seed {seed}: {pretty_time(time)}")
        

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

    t1 = time.time()
    prune_and_finetune(args.model_name, args.dataset_name, args.base_model_folder, args.checkpoint_folder)
    t2 = time.time()
    print(f"\nTempo totale di esecuzione: {pretty_time(t2 - t1)}")"""