import argparse
import os
from efficient_pruning.prune.prune_test import prune_random_channels
from efficient_pruning.model import FineTuning, ModelTest
from pruning_test_config import *
from config import *

def prune_and_finetune(model_name, 
                       dataset_name, 
                       perc_channels, 
                       mode,
                       checkpoint_folder):
    
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

    if mode == "one-shot":
        pruned_model = prune_random_channels(model, perc_channels=perc_channels)

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

    elif mode == "iterative":
        pruned_model = model
        for it in range(config2["iterations"]):
            print(f"\n--- Iterazione {it+1}/{config2['iterations']} ---")
            pruned_model = prune_random_channels(pruned_model, perc_channels=perc_channels)

            print("=== MINI FINE-TUNING ===")
            trainer = FineTuning(
                model=pruned_model,
                dataset=dataset,
                epochs=max(1, config2["finetune_epochs"] // config2["iterations"]),
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
        "--perc_channels", "-p", 
        type=float,
        dest="perc_channels",
        required=True, 
        default=1,
        help="Numero di canali H da prunare per step")
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
        args.perc_channels,
        args.mode,
        args.checkpoint_folder
    )

    model_test = ModelTest(
        model=model,
        dataset=dataset
    )

    model_test.run()