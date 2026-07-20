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
from models_datasets_and_profiling_implementation.model.utils import set_benchmark, set_seed, get_device
from models_datasets_and_profiling_implementation.model.net import NetFactory
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
    
def get_pruning_idx_exponential(perc, n_layers, H):

    N_target = int(round(perc * H * n_layers))

    tot_pruned = [0] * n_layers
    remaining_target = N_target
    remaining_layers = list(range(1, n_layers))

    while remaining_layers and remaining_target > 0:

        weights = [2**i for i in remaining_layers]
        Z = sum(weights)
        weights = [w / Z for w in weights]
        new_remaining_layers = []

        for idx, w in zip(remaining_layers, weights):
            alloc = int(round(w * remaining_target))

            if tot_pruned[idx] + alloc >= H - 1:
                tot_pruned[idx] = H - 1
            else:
                tot_pruned[idx] += alloc
                new_remaining_layers.append(idx)

        used = sum(tot_pruned)
        remaining_target = N_target - used
        remaining_layers = new_remaining_layers

        # Se tutti i layer sono saturi, abilitiamo il layer 0
        if not remaining_layers and tot_pruned[0] < H - 1 and remaining_target > 0:
            remaining_layers = [0]

    return tot_pruned

def prune_parameter(param, idx):
    if isinstance(idx, list):
        idx = torch.tensor(idx, dtype=torch.long, device=param.device)
    return param.index_select(0, idx).contiguous()

def prune_kernel_param(param, idx, axis=0):
    # idx può essere lista o tensor, lo convertiamo sempre in LongTensor
    if isinstance(idx, list):
        idx = torch.tensor(idx, dtype=torch.long, device=param.device)
    return param.index_select(axis, idx).contiguous()


def convert_layer_s4d(masked_layer, structural_layer, local_idx=None):

    # === D ===
    if local_idx is not None:
        structural_layer.D.data.copy_(
            prune_parameter(masked_layer.D.data, local_idx)
        )
    else:
        structural_layer.D.data.copy_(masked_layer.D.data)

    # === KERNEL PARAMETERS ===
    kernel_m = masked_layer.kernel
    kernel_s = structural_layer.kernel

    if local_idx is not None:
        kernel_s.C.data.copy_(
            prune_parameter(kernel_m.C.data, local_idx)
        )
    else:
        kernel_s.C.data.copy_(kernel_m.C.data)

    if local_idx is not None:
        kernel_s.log_dt.data.copy_(
            prune_parameter(kernel_m.log_dt.data, local_idx)
        )
    else:
        kernel_s.log_dt.data.copy_(kernel_m.log_dt.data)
    
    if local_idx is not None:
        kernel_s.log_A_real.data.copy_(
            prune_parameter(kernel_m.log_A_real.data, local_idx)
        )
    else:
        kernel_s.log_A_real.data.copy_(kernel_m.log_A_real.data)
    if local_idx is not None:
        kernel_s.A_imag.data.copy_(
            prune_parameter(kernel_m.A_imag.data, local_idx)
        )
    else:
        kernel_s.A_imag.data.copy_(kernel_m.A_imag.data)

    # === OUTPUT LINEAR (SHARED SHAPE) ===
    structural_layer.output_linear.load_state_dict(
        masked_layer.output_linear.state_dict()
    )

def convert_layer_s4(masked_layer, structural_layer, local_idx=None):

    # === KERNEL PARAMETERS ===
    kernel_m = masked_layer.layer
    kernel_s = structural_layer.layer  

    if local_idx is not None:
        kernel_s.D.data.copy_(
            prune_kernel_param(kernel_m.D.data, local_idx, axis=1)
        )
    else:
        kernel_s.D.data.copy_(kernel_m.D.data)

    if local_idx is not None:
        kernel_s.kernel.P.data.copy_(
            prune_kernel_param(kernel_m.kernel.P.data, local_idx, axis=1)
        )
    else:
        kernel_s.kernel.P.data.copy_(kernel_m.kernel.P.data)

    if local_idx is not None:
        kernel_s.kernel.inv_dt.data.copy_(
            prune_kernel_param(kernel_m.kernel.inv_dt.data, local_idx, axis=0)
        )
    else:
        kernel_s.kernel.inv_dt.data.copy_(kernel_m.kernel.inv_dt.data)

    if local_idx is not None:
        kernel_s.kernel.A_real.data.copy_(
            prune_kernel_param(kernel_m.kernel.A_real.data, local_idx, axis=0)
        )
    else:
        kernel_s.kernel.A_real.data.copy_(kernel_m.kernel.A_real.data)

    if local_idx is not None:
        kernel_s.kernel.A_imag.data.copy_(
            prune_kernel_param(kernel_m.kernel.A_imag.data, local_idx, axis=0)
        )
    else:
        kernel_s.kernel.A_imag.data.copy_(kernel_m.kernel.A_imag.data)

    if local_idx is not None:
        kernel_s.kernel.B.data.copy_(
            prune_kernel_param(kernel_m.kernel.B.data, local_idx, axis=1)
        )
    else:
        kernel_s.kernel.B.data.copy_(kernel_m.kernel.B.data)

    if local_idx is not None:
        kernel_s.kernel.C.data.copy_(
            prune_kernel_param(kernel_m.kernel.C.data, local_idx, axis=1)
        )
    else:
        kernel_s.kernel.C.data.copy_(kernel_m.kernel.C.data)   

    kernel_s.kernel.l_kernel.data.copy_(kernel_m.kernel.l_kernel.data)

    # === OUTPUT LINEAR (SHARED SHAPE) ===
    structural_layer.output_linear.load_state_dict(
        masked_layer.output_linear.state_dict()
    )

def convert_layer_mamba(masked_layer, structural_layer, local_idx=None):
    if local_idx is None:
        structural_layer.load_state_dict(masked_layer.state_dict())
        return

    device = masked_layer.in_proj.weight.device
    idx = torch.as_tensor(local_idx, dtype=torch.long, device=device)
    d_inner_base = masked_layer.d_inner

    # in_proj: righe idx (metà x) + (d_inner_base + idx) (metà z)
    rows_in = torch.cat([idx, d_inner_base + idx], dim=0)
    structural_layer.in_proj.weight.data.copy_(
        masked_layer.in_proj.weight.data.index_select(0, rows_in))
    if masked_layer.in_proj.bias is not None:
        structural_layer.in_proj.bias.data.copy_(
            masked_layer.in_proj.bias.data.index_select(0, rows_in))

    # conv1d depthwise: canali idx
    structural_layer.conv1d.weight.data.copy_(
        masked_layer.conv1d.weight.data.index_select(0, idx))
    if masked_layer.conv1d.bias is not None:
        structural_layer.conv1d.bias.data.copy_(
            masked_layer.conv1d.bias.data.index_select(0, idx))

    # x_proj: colonne idx (l'input è d_inner)
    structural_layer.x_proj.weight.data.copy_(
        masked_layer.x_proj.weight.data.index_select(1, idx))

    # dt_proj: righe idx (l'output è d_inner) + bias
    structural_layer.dt_proj.weight.data.copy_(
        masked_layer.dt_proj.weight.data.index_select(0, idx))
    structural_layer.dt_proj.bias.data.copy_(
        masked_layer.dt_proj.bias.data.index_select(0, idx))

    # A_log [d_inner, d_state] e D [d_inner]: righe idx
    structural_layer.A_log.data.copy_(masked_layer.A_log.data.index_select(0, idx))
    structural_layer.D.data.copy_(masked_layer.D.data.index_select(0, idx))

    # out_proj: colonne idx (l'input è d_inner); l'output resta d_model (bias intero)
    structural_layer.out_proj.weight.data.copy_(
        masked_layer.out_proj.weight.data.index_select(1, idx))
    if masked_layer.out_proj.bias is not None:
        structural_layer.out_proj.bias.data.copy_(masked_layer.out_proj.bias.data)

def prune_and_finetune(model_name, dataset_name, base_model_folder, checkpoint_folder):

    config = MODELS_CONFIG
    pc = PRUNING_CONFIG[model_name][dataset_name]
    pc2 = PRUNING_CONFIG
    
    os.makedirs(checkpoint_folder, exist_ok=True)
    path = os.path.join(f"./{checkpoint_folder}", f"{model_name}_{dataset_name}_pruned_{int(pc2['perc'][0]*100)}%.pth")
    if os.path.exists(path):
        raise ValueError(f"Il file {path} esiste già. Scegliere un'altra cartella o un altro nome per il file.")

    model_checkpoint = f"./{base_model_folder}/{model_name}_{dataset_name}_best.pth"

    H = config[model_name][dataset_name]["features"]
    if model_name == "mamba":
        H = 2 * H   # d_inner = expand * d_model (expand=2)
    n_layers = config[model_name][dataset_name]["depth"]

    accuracies = {}
    best_seeds = {}
    timers = {}

    for perc in pc2["perc"]:

        print(f"\n=== PRUNING {perc*100}% ===")

        best_acc = -1
        best_val_acc = -1
        best_seed = None
        best_checkpoint = None
        timers[perc] = {}

        prev_model = ModelTest.from_pth(
            model_path=model_checkpoint,
            batch_size=config[model_name][dataset_name]["batch_size"],
            valsplit=config["val_split"],
            num_workers=config["num_workers"],
            d_model=config[model_name][dataset_name]["features"],
            d_state=config[model_name][dataset_name]["d_state"],
            depth=config[model_name][dataset_name]["depth"],
            dropout=config[model_name][dataset_name]["dropout"],
            norm=config[model_name][dataset_name]["norm"],
            pre_norm=config[model_name][dataset_name]["pre-norm"]
        )
        ckpt = torch.load(model_checkpoint, map_location=get_device())
        prev_active_idx = ckpt["active_idx_layers"] 
        prev_active = [list(idx) for idx in prev_active_idx]

        for seed in pc2["seeds"]:
            t1 = time.time()
            print(f"\n--- SEED {seed} ---")
            set_seed(seed)
            set_benchmark(False)

            # Pruning incrementale
            n_pruned_perc = get_pruning_idx_exponential(perc, n_layers, H)
            n_pruned = [n_pruned_perc[i] - (H - len(prev_active[i])) for i in range(n_layers)]

            active_idx = [[] for _ in range(n_layers)]
            for i in range(n_layers):
                available = prev_active[i]
                idx_to_prune = random.sample(available, n_pruned[i])
                new_active = list(set(available) - set(idx_to_prune))
                active_idx[i] = new_active
                print(f"[Layer {i}]: canali rimanenti {len(active_idx[i])}/{H}")
        
            # Costruzione modello strutturale
            model = NetFactory(
                model_name=model_name,
                dataset_name=dataset_name,
                vocab_size=prev_model.dataset.vocab_size if hasattr(prev_model.dataset, 'vocab_size') else prev_model.dataset.input_shape[-1],
                d_model=config[f"{model_name}"][f"{dataset_name}"]["features"],
                d_state=config[f"{model_name}"][f"{dataset_name}"]["d_state"],
                depth=config[f"{model_name}"][f"{dataset_name}"]["depth"],
                dropout=config[f"{model_name}"][f"{dataset_name}"]["dropout"],
                num_classes=prev_model.dataset.get_output_shape()[-1],
                norm=config[f"{model_name}"][f"{dataset_name}"]["norm"],
                pre_norm=config[f"{model_name}"][f"{dataset_name}"]["pre-norm"] ,
                active_idx_layers=active_idx
            ).get_net()

            # === COPY EMBEDDING ===
            model.embedding.load_state_dict(
                prev_model.model.embedding.state_dict()
            )   

            # === COPY LAYERS ===
            if model_name == "s4d":
                for i, (lm, ls) in enumerate(zip(prev_model.model.s4d_layers, model.s4d_layers)):
                    mapping = {int(g):i for i,g in enumerate(prev_active_idx[i])}
                    local_idx = [mapping[int(g)] for g in active_idx[i]]
                    convert_layer_s4d(lm, ls, local_idx)
            elif model_name == "mamba":
                for i, (lm, ls) in enumerate(zip(prev_model.model.mamba_layers, model.mamba_layers)):
                    mapping = {int(g):j for j,g in enumerate(prev_active_idx[i])}
                    local_idx = [mapping[int(g)] for g in active_idx[i]]
                    convert_layer_mamba(lm, ls, local_idx)
            else:
                for i, (lm, ls) in enumerate(zip(prev_model.model.s4_layers, model.s4_layers)):
                    mapping = {int(g):i for i,g in enumerate(prev_active_idx[i])}
                    local_idx = [mapping[int(g)] for g in active_idx[i]]
                    convert_layer_s4(lm, ls, local_idx)

            # === COPY NORMS ===
            for nm, ns in zip(prev_model.model.norms, model.norms):
                ns.load_state_dict(nm.state_dict())

            if model_name == "mamba" and hasattr(model, "norm_f"):
                model.norm_f.load_state_dict(prev_model.model.norm_f.state_dict())

            # === COPY CLASSIFIER ===
            if model_name == "mamba" and hasattr(model, "fc"):
                model.fc.load_state_dict(
                    prev_model.model.fc.state_dict()
                )

            # === COPY MLP MATCH FOR RETRIEVAL DATASET ===
            if model_name == "mamba" and hasattr(model, "match"):
                model.match.load_state_dict(
                    prev_model.model.match.state_dict()
                )

            model_test = ModelTest(model=model, dataset=prev_model.dataset)
            model = model_test.model
            dataset = model_test.dataset

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
            val_acc = trainer.run()

            acc = model_test.run()

            if val_acc > best_val_acc:
                best_acc = acc
                best_seed = seed
                best_val_acc = val_acc
                best_checkpoint = os.path.join(f"./{checkpoint_folder}", f"{model_name}_{dataset_name}_pruned_{int(perc*100)}%.pth")
                torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "active_idx_layers": active_idx
                },
                best_checkpoint
                )

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