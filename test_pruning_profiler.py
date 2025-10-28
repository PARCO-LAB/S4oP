import torch
import argparse
import os

from pathlib import Path
from efficient_pruning.prune.progressive_profile_pruning.pruning_profiler import PruningProfiler
from efficient_pruning.model import ModelTest
from config import PRUNING_DEFAULT_CONFIG

def load_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    model = checkpoint['model'] if 'model' in checkpoint else checkpoint
    # Assicurati che sia un LayerS4D per il test
    if isinstance(model, torch.nn.Module):
        return model
    raise Exception(f"Checkpoint non contiene un nn.Module valido: {path}")

def main(model_name, dataset_name, epochs, batch_size, valsplit, checkpoints_folder, pruned_model_index):

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

    if os.path.exists(model_path):
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
        # Creo il profiler
        profiler = PruningProfiler(seq_len=model_test.dataset.input_shape[1], batch_size=batch_size, iterations=50)

        # Ottengo le configurazioni per pruning
        configs = profiler.get_config(model_test.model.s4_layers[0])  # Profilo il primo layer S4D come esempio

        print(f"Trovate {len(configs)} configurazioni da profilare:")
        for c in configs[:10]:  # Stampa le prime 10 configurazioni
            print(c)

        # Profiling con one_shot_profile = True
        profiler.one_shot_profile = True
        print("\nEseguendo profiling...")
        results_one_shot = profiler.run(configs)
        print("Risultati:", results_one_shot[:10])  # Stampa i primi 10 risultati

        # Pulizia
        profiler.stop()
        print("\nProfiling completato con successo")
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
    parser.add_argument(
        "--pruned-model", "-p",
        dest="pruned_model_index",
        required=False, default=None,
        help="Pruned model index")
    args = parser.parse_args()

    if args.epochs is None and args.batch_size is None:
        main(args.model_name, args.dataset_name, None, None, float(args.valsplit), args.checkpoints_folder, args.pruned_model_index)
    elif args.epochs is None:
        main(args.model_name, args.dataset_name, None, int(args.batch_size), float(args.valsplit), args.checkpoints_folder, args.pruned_model_index)
    elif args.batch_size is None:
        main(args.model_name, args.dataset_name, int(args.epochs), None, float(args.valsplit), args.checkpoints_folder, args.pruned_model_index)
    else:
        main(args.model_name, args.dataset_name, int(args.epochs), int(args.batch_size), float(args.valsplit), args.checkpoints_folder, args.pruned_model_index)
