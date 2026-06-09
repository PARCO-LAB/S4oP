import os
import argparse
import torch
if torch.cuda.is_available() and not torch.cuda.is_initialized():
    torch.cuda.current_device()

from models_config import *
from models_datasets_and_profiling_implementation.model import ModelTrain, ModelTest, ModelInfo, ModelProfile
from models_datasets_and_profiling_implementation.model.utils import set_benchmark, set_seed, setup_optimizer

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
    
    # Se il modello non esiste, lo alleno
    if (not os.path.exists(model_path)) and (pruned_model_name is None):

        # Creazione modello
        model_train = ModelTrain.from_scratch(
            model_name=model_name, 
            dataset_name=dataset_name, 
            batch_size=config[f"{model_name}"][f"{dataset_name}"]["batch_size"], 
            valsplit=config["val_split"], 
            num_workers=config["num_workers"], 
            d_model=config[f"{model_name}"][f"{dataset_name}"]["features"],
            d_state=64,
            depth=config[f"{model_name}"][f"{dataset_name}"]["depth"],
            dropout=config[f"{model_name}"][f"{dataset_name}"]["dropout"],
            norm=config[f"{model_name}"][f"{dataset_name}"]["norm"],
            pre_norm=config[f"{model_name}"][f"{dataset_name}"]["pre-norm"],
            active_idx_layers=[list(range(config[f"{model_name}"][f"{dataset_name}"]["features"])) for _ in range(config[f"{model_name}"][f"{dataset_name}"]["depth"])]
        )
        print(f"\nModello: {model_name}, Dataset: {dataset_name}")
        print(model_train.model)
        criterion = torch.nn.CrossEntropyLoss() if dataset_name in ["imdb", "listops", "pathfinder"] else torch.nn.BCEWithLogitsLoss()
        optimizer, scheduler = setup_optimizer(
            model_train.model,
            lr=config[f"{model_name}"][f"{dataset_name}"]["lr"],
            weight_decay=config[f"{model_name}"][f"{dataset_name}"]["wd"],
            epochs=config[f"{model_name}"][f"{dataset_name}"]["epochs"],
        )

        print(f"\nStarting training for {config[f'{model_name}'][f'{dataset_name}']['epochs']} epochs with batch size {config[f'{model_name}'][f'{dataset_name}']['batch_size']}...")
        # Training
        model_train.run(optimizer, scheduler, criterion, config[f"{model_name}"][f"{dataset_name}"]["epochs"], checkpoints_folder=os.path.join(".", f"{checkpoints_folder}"), is_pruned=False)

        # Testing
        model_test = ModelTest.from_pth(model_path=model_path, 
                                        batch_size=config[f"{model_name}"][f"{dataset_name}"]["batch_size"], 
                                        valsplit=config["val_split"],
                                        num_workers=config["num_workers"], 
                                        d_model=config[f"{model_name}"][f"{dataset_name}"]["features"],
                                        d_state=64,
                                        depth=config[f"{model_name}"][f"{dataset_name}"]["depth"],
                                        dropout=config[f"{model_name}"][f"{dataset_name}"]["dropout"],
                                        norm=config[f"{model_name}"][f"{dataset_name}"]["norm"],
                                        pre_norm=config[f"{model_name}"][f"{dataset_name}"]["pre-norm"],
                                        )
        print("\nStarting testing...")
        model_test.run()

        # Model Info
        model_info = ModelInfo(
            model=model_test.model, 
            vocab_size=model_test.dataset.vocab_size if hasattr(model_test.dataset, 'vocab_size') else model_test.dataset.input_shape[-1],
            seq_len=model_test.dataset.input_shape[1], 
            batch_size=config[f"{model_name}"][f"{dataset_name}"]["batch_size"],
            dataset_name=dataset_name
        )
        os.makedirs("model_info", exist_ok=True)
        if not os.path.exists(f"./model_info/{model_name}_{dataset_name}_torchinfo.txt"):
            model_info.torchinfo(output_dir="model_info", name=f"{model_name}_{dataset_name}")
            print("\nModel info salvato nella cartella 'model_info'")
        else:
            print("\nModel info già esistente nella cartella 'model_info', salto la creazione")

        # Model Profile
        model_profile = ModelProfile(iterations=100)
        model_profile.add("prova")
        model_profile.run("prova", model_test.model, model_info.get_example_input, iterations=100)
        model_profile.info("prova")
 
    # Se il modello esiste ed è stato specificato un modello prunato
    elif pruned_model_name is not None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pruned model path {model_path} does not exist.")

        print(f"\nLoading pruned model from path {model_path}...")

        # Testing
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
        print("\nStarting testing...")
        model_test.run()

        print("\nStarting model profiling...")
        # Model Info
        model_info = ModelInfo(
            model=model_test.model, 
            vocab_size=model_test.dataset.vocab_size if hasattr(model_test.dataset, 'vocab_size') else model_test.dataset.input_shape[-1],
            seq_len=model_test.dataset.input_shape[1], 
            batch_size=config[f"{model_name}"][f"{dataset_name}"]["batch_size"], 
            dataset_name=dataset_name
        )
        os.makedirs("model_info_pruned_structural", exist_ok=True)
        if not os.path.exists(f"./model_info_pruned_structural/{pruned_model_name}_torchinfo.txt"):
            model_info.torchinfo(output_dir="model_info_pruned_structural", name=pruned_model_name)
            print("\nModel info salvato nella cartella 'model_info_pruned_structural'")
        else:
            print("\nModel info già esistente nella cartella 'model_info_pruned_structural', salto la creazione")

        # Model Profile
        model_profile = ModelProfile(iterations=100)
        model_profile.add("prova")
        model_profile.run("prova", model_test.model, model_info.get_example_input, iterations=100)
        model_profile.info("prova")
    
    # Se il modello esiste (e non è stato specificato un modello prunato), salto l'addestramento
    else: 
        print(f"\nSkipping training because model path {model_path} already exists")

        # Testing
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
        print("\nStarting testing...")
        model_test.run()

        print("\nStarting model profiling...")
        # Model Info
        model_info = ModelInfo(
            model=model_test.model, 
            vocab_size=model_test.dataset.vocab_size if hasattr(model_test.dataset, 'vocab_size') else model_test.dataset.input_shape[-1],
            seq_len=model_test.dataset.input_shape[1], 
            batch_size=config[f"{model_name}"][f"{dataset_name}"]["batch_size"], 
            dataset_name=dataset_name
        )
        os.makedirs("model_info", exist_ok=True)
        if not os.path.exists(f"./model_info/{model_name}_{dataset_name}_torchinfo.txt"):
            model_info.torchinfo(output_dir="model_info", name=f"{model_name}_{dataset_name}")
            print("\nModel info salvato nella cartella 'model_info'")
        else:
            print("\nModel info già esistente nella cartella 'model_info', salto la creazione")

        # Model Profile        
        model_profile = ModelProfile(iterations=100)
        model_profile.add("prova")
        model_profile.run("prova", model_test.model, model_info.get_example_input, iterations=100)
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