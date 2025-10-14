import torch
import time
import argparse
import os
from efficient_pruning.model import ModelTest
from efficient_pruning.dataset import DatasetFactory
from efficient_pruning.model.net import NetFactory
from config import *
from efficient_pruning.model.utils import *

set_seed(42)
set_benchmark(False)

def print_model_size(model):
    print("\n=== Model Size ===")

    # Print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # Print parameters per layer
    print("\nParameters per layer:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel():,}")

def profile_model(model_path):
    config = PRUNING_DEFAULT_CONFIG
    device = get_device()

    basename = os.path.basename(model_path)
    basename_split = basename.split("_")
    model_name, dataset_name = basename_split[0], basename_split[1]

    model_test = ModelTest.from_pth(model_path=model_path, 
                                    batch_size=config[f"{model_name}"][f"{dataset_name}"]["batch_size"], 
                                    valsplit=config["val_split"],
                                    num_workers=config["num_workers"], 
                                    d_model=config[f"{model_name}"][f"{dataset_name}"]["features"],
                                    d_state=64,
                                    depth=config[f"{model_name}"][f"{dataset_name}"]["depth"],
                                    dropout=config[f"{model_name}"][f"{dataset_name}"]["dropout"],
                                    norm=config[f"{model_name}"][f"{dataset_name}"]["norm"],
                                    pre_norm=config[f"{model_name}"][f"{dataset_name}"]["pre-norm"]
                                    )

    torch.cuda.empty_cache() if device.type=='cuda' else None
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)

    dataloader = model_test.model_train.dataset.get_testloader()
    batch_size=config[f"{model_name}"][f"{dataset_name}"]["batch_size"]
    timings = []

    # Warm-up CUDA
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
    if device.type=='cuda':
        print("\n=== CUDA Memory Usage ===")
        print(f"Picco di memoria usata: {peak_mem/1e6:.2f} MB")

    print_model_size(model_test.model_train.model)

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument(
        "--path", "-p", 
        dest="model_path", 
        required=False, default=os.path.join(".", "checkpoints"),
        help="Model path")
    args = parser.parse_args()

    profile_model(args.model_path)