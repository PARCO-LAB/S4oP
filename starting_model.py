import os
import argparse
import torch

from config import *
from efficient_pruning.model import ModelTrain, ModelTest, ModelInfo
from efficient_pruning.model.utils import set_benchmark, set_seed, setup_optimizer

set_seed(42)
set_benchmark(False)

def main(model_name, dataset_name, epochs, batch_size, valsplit, checkpoints_folder):

    # Caricamento configurazione
    config = PRUNING_DEFAULT_CONFIG

    # Creazione path modello
    model_path = os.path.join(checkpoints_folder, "{}_{}_best.pth".format(model_name, dataset_name))

    # Caricamento iperparametri se non impostati dall'utente
    if epochs is None:
        epochs = config[f"{model_name}"][f"{dataset_name}"]["epochs"]
    if batch_size is None:
        batch_size = config[f"{model_name}"][f"{dataset_name}"]["batch_size"]
    if valsplit is None:
        valsplit = config["val_split"]
    
    # Se il modello non esiste, lo alleno
    if not os.path.exists(model_path):

        # Creazione modello
        model_train = ModelTrain.from_scratch(
            model_name=model_name, 
            dataset_name=dataset_name, 
            batch_size=batch_size, 
            valsplit=valsplit, 
            num_workers=config["num_workers"], 
            d_model=config[f"{model_name}"][f"{dataset_name}"]["features"],
            d_state=64,
            depth=config[f"{model_name}"][f"{dataset_name}"]["depth"],
            dropout=config[f"{model_name}"][f"{dataset_name}"]["dropout"],
            norm=config[f"{model_name}"][f"{dataset_name}"]["norm"],
            pre_norm=config[f"{model_name}"][f"{dataset_name}"]["pre-norm"]
        )
        print(f"Modello: {model_name}, Dataset: {dataset_name}")
        print(model_train.model)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer, scheduler = setup_optimizer(
            model_train.model,
            lr=config[f"{model_name}"][f"{dataset_name}"]["lr"],
            weight_decay=config[f"{model_name}"][f"{dataset_name}"]["wd"],
            epochs=epochs
        )

        # Training
        model_train.run(optimizer, scheduler, criterion, epochs, checkpoints_folder)
    else: 
        print("Skipping training because model path {} already exists".format(model_path))

    # Testing
    model_test = ModelTest.from_pth(model_path=model_path, 
                                    batch_size=batch_size, 
                                    valsplit=valsplit,
                                    num_workers=config["num_workers"], 
                                    d_model=config[f"{model_name}"][f"{dataset_name}"]["features"],
                                    d_state=64,
                                    depth=config[f"{model_name}"][f"{dataset_name}"]["depth"],
                                    dropout=config[f"{model_name}"][f"{dataset_name}"]["dropout"],
                                    norm=config[f"{model_name}"][f"{dataset_name}"]["norm"],
                                    pre_norm=config[f"{model_name}"][f"{dataset_name}"]["pre-norm"]
                                    )
    model_test.run()

    # Model Info
    print("=== MODEL INFO ===")
    model_info = ModelInfo(
        model=model_test.model, 
        vocab_size=model_test.dataset.vocab_size, 
        seq_len=model_test.dataset.input_shape[1], 
        batch_size=batch_size, 
        dataset_name=dataset_name
    )
    model_info.torchinfo(output_dir="model_info")
    model_info.torchviz(output_dir="model_info")

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
        "--epochs", "-e", 
        dest="epochs", 
        required=False,
        help="Epochs amount")
    parser.add_argument(
        "--batch-size", "-b", 
        dest="batch_size", 
        required=False,
        help="Batch size")
    parser.add_argument(
        "--valsplit", "-v", 
        dest="valsplit", 
        required=False, default=0.2,
        help="Val split percentage: [0, 1]")
    parser.add_argument(
        "--checkpoints-folder", "-f", 
        dest="checkpoints_folder", 
        required=False, default=os.path.join(".", "checkpoints"),
        help="Checkpoints folder")
    args = parser.parse_args()

    if args.epochs is None and args.batch_size is None:
        main(args.model_name, args.dataset_name, None, None, float(args.valsplit), args.checkpoints_folder)
    elif args.epochs is None:
        main(args.model_name, args.dataset_name, None, int(args.batch_size), float(args.valsplit), args.checkpoints_folder)
    elif args.batch_size is None:
        main(args.model_name, args.dataset_name, int(args.epochs), None, float(args.valsplit), args.checkpoints_folder)
    else:
        main(args.model_name, args.dataset_name, int(args.epochs), int(args.batch_size), float(args.valsplit), args.checkpoints_folder)