import torch
import os
import argparse
if torch.cuda.is_available() and not torch.cuda.is_initialized():
    torch.cuda.current_device()

from models_config import *
from models_datasets_and_profiling_implementation.model import ModelTest, ModelInfo
from models_datasets_and_profiling_implementation.model.utils import set_benchmark, set_seed, get_device

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

    # Se il modello esiste, lo carico e lo metto in eval mode, altrimenti lancio un errore
    if (os.path.exists(model_path)):

        # Caricamento modello
        model_test = ModelTest.from_pth(
            model_path=model_path,
            batch_size=config[f"{model_name}"][f"{dataset_name}"]["batch_size"],
            valsplit=config["val_split"],
            num_workers=0, # Niente workers per evitare problemi di multiprocessing
            d_model=config[f"{model_name}"][f"{dataset_name}"]["features"],
            d_state=64,
            depth=config[f"{model_name}"][f"{dataset_name}"]["depth"],
            dropout=config[f"{model_name}"][f"{dataset_name}"]["dropout"],
            norm=config[f"{model_name}"][f"{dataset_name}"]["norm"],
            pre_norm=config[f"{model_name}"][f"{dataset_name}"]["pre-norm"]
        )
        model = model_test.model

        # Model Info
        model_info = ModelInfo(
            model=model_test.model, 
            vocab_size=model_test.dataset.vocab_size if hasattr(model_test.dataset, 'vocab_size') else model_test.dataset.input_shape[-1],
            seq_len=model_test.dataset.input_shape[1], 
            batch_size=config[f"{model_name}"][f"{dataset_name}"]["batch_size"], 
            dataset_name=dataset_name
        )

        example_input = model_info.get_example_input()
        device = get_device()
        model.to(device)
        model.eval()

        # warmup
        for _ in range(10):
            _ = model(example_input)

        torch.cuda.synchronize()

        # forward da profilare
        _ = model(example_input)

        torch.cuda.synchronize()
    # Il modello non esiste, lancio un errore
    else:
        raise FileNotFoundError(f"Model path {model_path} does not exist.")


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