import argparse
import os
import torch
import time

from efficient_pruning.prune.prune_test import prune_random_channels, remove_all_masks
from efficient_pruning.model import FineTuning, ModelTest, ModelInfo
from efficient_pruning.model.utils import get_device
from pruning_test_config import *
from config import *

def prune_and_finetune(model_name, 
                       dataset_name, 
                       mode,
                       checkpoint_folder):
    
    path = os.path.join(checkpoint_folder, f"{model_name}_{dataset_name}_{mode}.pth")
    if os.path.exists(path):
        raise ValueError(f"Il file {path} esiste già. Scegliere un'altra cartella o un altro nome per il file.")
    
    config1 = PRUNING_DEFAULT_CONFIG
    config2 = PRUNING_CONFIG[mode]

    model_test = ModelTest.from_pth(
        model_path=f"./checkpoints/{model_name}_{dataset_name}_best.pth",
        batch_size=config1[model_name][dataset_name]["batch_size"],
        valsplit=config1["val_split"],
        num_workers=config1["num_workers"],
        d_model=config1[model_name][dataset_name]["features"],
        d_state=64,
        depth=config1[model_name][dataset_name]["depth"],
        dropout=config1[model_name][dataset_name]["dropout"],
        norm=config1[model_name][dataset_name]["norm"],
        pre_norm=config1[model_name][dataset_name]["pre-norm"]
    )
    model = model_test.model
    dataset = model_test.dataset

    print(f"=== PRUNING MODE: {mode} ===")

    if mode.startswith("one-shot"):
        pruned_model = prune_random_channels(model, perc_channels=config2["perc_channels"])

        print("=== FINE-TUNING FINALE ===")
        trainer = FineTuning(
            model=pruned_model,
            dataset=dataset,
            epochs=config2["finetune_epochs"],
            lr=config2["lr"],
            weight_decay=config2["weight_decay"],
            checkpoint_folder=checkpoint_folder
        )
        trainer.run()

    elif mode.startswith("iterative"):
        pruned_model = model
        for it in range(config2["iterations"]):
            print(f"\n--- Iterazione {it+1}/{config2['iterations']} ---")
            pruned_model = prune_random_channels(pruned_model, perc_channels=config2["perc_channels"])

            print("=== MINI FINE-TUNING ===")
            trainer = FineTuning(
                model=pruned_model,
                dataset=dataset,
                epochs=config2["finetune_epochs"],
                lr=config2["lr"],
                weight_decay=config2["weight_decay"],
                checkpoint_folder=checkpoint_folder
            )
            trainer.run()

    else:
        raise ValueError(f"Modo di pruning non valido: {mode}")
    
    return pruned_model, dataset

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
        "--mode", "-M",
        dest="mode",
        required=True,
        default="one-shot",
        help="Modalità di pruning (one-shot o iterative)")
    parser.add_argument(
        "--checkpoint_folder", "-c",
        dest="checkpoint_folder",
        required=False, default=os.path.join(".", "checkpoints_pruned"),
        help="Cartella per salvare i checkpoint"
    )
    args = parser.parse_args()

    model, dataset = prune_and_finetune(
        args.model_name,
        args.dataset_name,
        args.mode,
        args.checkpoint_folder
    )

    model_test = ModelTest(
        model=model,
        dataset=dataset
    )

    model_test.run()

    # Model Info
    config = PRUNING_DEFAULT_CONFIG
    """ model_info = ModelInfo(
        model=model_test.model, 
        vocab_size=model_test.dataset.vocab_size, 
        seq_len=model_test.dataset.input_shape[1], 
        batch_size=config[args.model_name][args.dataset_name]["batch_size"], 
        dataset_name=args.dataset_name
    )
    model_info.torchinfo(output_dir="model_info_pruned", mode=args.mode) """

    device = get_device()
    torch.cuda.empty_cache() if device.type=='cuda' else None
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)

    dataloader = model_test.model_train.dataset.get_testloader()
    batch_size=config[args.model_name][args.dataset_name]["batch_size"]
    timings = []

    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= batch_size:
                break
            inputs = inputs.to(device, non_blocking=True)
            torch.cuda.synchronize() if device.type=='cuda' else None
            t0 = time.time()
            _ = model_test.model_train.model(inputs)
            torch.cuda.synchronize() if device.type=='cuda' else None
            t1 = time.time()
            timings.append(t1 - t0)

    peak_mem = torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0

    avg_time = sum(timings)/len(timings)
    print("\n=== Timing Results ===")
    print(f"Tempo medio di inferenza per batch: {avg_time*1000:.2f} ms")
    print(f"Tempo totale di inferenza: {sum(timings)*1000:.2f} ms")
    # Non ha senso perchè tanto la funzione di hook sballa tutto
    """ if device.type=='cuda':
        print("\n=== CUDA Memory Usage ===")
        print(f"Picco di memoria usata: {peak_mem/1e6:.2f} MB") """
    