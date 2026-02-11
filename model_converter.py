import os
import argparse
import torch
if torch.cuda.is_available() and not torch.cuda.is_initialized():
    torch.cuda.current_device()

from models_config import *
from models_datasets_and_profiling_implementation.model import ModelTest, ModelInfo
from models_datasets_and_profiling_implementation.model.utils import set_benchmark, set_seed
from models_datasets_and_profiling_implementation.model.net import NetFactory

set_seed(42)
set_benchmark(False)

def extract_active_idx(mask: torch.Tensor):

    return mask.nonzero(as_tuple=True)[0]


def prune_parameter(param, idx):

    return param.index_select(0, idx).contiguous()

def prune_kernel_param(param, idx, axis=0):

    return param.index_select(axis, idx).contiguous()


def convert_layer_s4d(masked_layer, structural_layer):

    active_idx = structural_layer.active_idx

    # === D ===
    structural_layer.D.data.copy_(
        prune_parameter(masked_layer.D.data, active_idx)
    )

    # === KERNEL PARAMETERS ===
    kernel_m = masked_layer.kernel
    kernel_s = structural_layer.kernel

    kernel_s.C.data.copy_(
        prune_parameter(kernel_m.C.data, active_idx)
    )
    kernel_s.log_dt.data.copy_(
        prune_parameter(kernel_m.log_dt.data, active_idx)
    )
    kernel_s.log_A_real.data.copy_(
        prune_parameter(kernel_m.log_A_real.data, active_idx)
    )
    kernel_s.A_imag.data.copy_(
        prune_parameter(kernel_m.A_imag.data, active_idx)
    )

    # === OUTPUT LINEAR (SHARED SHAPE) ===
    structural_layer.output_linear.load_state_dict(
        masked_layer.output_linear.state_dict()
    )

def convert_layer_s4(masked_layer, structural_layer):

    active_idx = structural_layer.active_idx

    # === KERNEL PARAMETERS ===
    kernel_m = masked_layer.layer
    kernel_s = structural_layer.layer  

    kernel_s.D.data.copy_(
        prune_kernel_param(kernel_m.D.data, active_idx, axis=1)
    )
    kernel_s.kernel.P.data.copy_(
        prune_kernel_param(kernel_m.kernel.P.data, active_idx, axis=1)
    )
    kernel_s.kernel.inv_dt.data.copy_(
        prune_kernel_param(kernel_m.kernel.inv_dt.data, active_idx, axis=0)
    )
    kernel_s.kernel.A_real.data.copy_(
        prune_kernel_param(kernel_m.kernel.A_real.data, active_idx, axis=0)
    )
    kernel_s.kernel.A_imag.data.copy_(
        prune_kernel_param(kernel_m.kernel.A_imag.data, active_idx, axis=0)
    )
    kernel_s.kernel.B.data.copy_(
        prune_kernel_param(kernel_m.kernel.B.data, active_idx, axis=1)
    )
    kernel_s.kernel.C.data.copy_(
        prune_kernel_param(kernel_m.kernel.C.data, active_idx, axis=1)
    )
    kernel_s.kernel.l_kernel.data.copy_(kernel_m.kernel.l_kernel.data)

    # === OUTPUT LINEAR (SHARED SHAPE) ===
    structural_layer.output_linear.load_state_dict(
        masked_layer.output_linear.state_dict()
    )

def main(model_name, dataset_name, checkpoints_folder, pruned_model_name):

    # Caricamento configurazione
    config = MODELS_CONFIG

    # Creazione path modello
    model_path = os.path.join(f"./{checkpoints_folder}", f"{pruned_model_name}.pth")
 
    # Se il modello prunato esiste 
    if os.path.exists(model_path):

        print(f"\nLoading pruned model from path {model_path}...")

        # Testing
        masked_model = ModelTest.from_pth(model_path=model_path, 
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
        
        print("\nStarting removing channels...")
        # === BUILD ACTIVE IDX PER LAYER ===
        if model_name == "s4d":
            active_indices = []
            for layer in masked_model.model.s4d_layers:
                mask = layer.pruning_mask
                active_idx = extract_active_idx(mask)
                active_indices.append(active_idx)
        else:
            active_indices = []
            for layer in masked_model.model.s4_layers:
                mask = layer.layer.pruning_mask
                active_idx = extract_active_idx(mask)
                active_indices.append(active_idx)

        # === BUILD STRUCTURAL MODEL ===
        structural_model = NetFactory(
            model_name=model_name,
            dataset_name=dataset_name,
            vocab_size=masked_model.dataset.vocab_size if hasattr(masked_model.dataset, 'vocab_size') else masked_model.dataset.input_shape[-1],
            d_model=config[f"{model_name}"][f"{dataset_name}"]["features"],
            d_state=64,
            depth=config[f"{model_name}"][f"{dataset_name}"]["depth"],
            dropout=config[f"{model_name}"][f"{dataset_name}"]["dropout"],
            num_classes=masked_model.dataset.get_output_shape()[-1],
            norm=config[f"{model_name}"][f"{dataset_name}"]["norm"],
            pre_norm=config[f"{model_name}"][f"{dataset_name}"]["pre-norm"] ,
            active_idx_layers=active_indices
        ).get_net()

        # === COPY EMBEDDING ===
        structural_model.embedding.load_state_dict(
            masked_model.model.embedding.state_dict()
        )

        # === COPY LAYERS ===
        if model_name == "s4d":
            for lm, ls in zip(masked_model.model.s4d_layers, structural_model.s4d_layers):
                convert_layer_s4d(lm, ls)
        else:
            for lm, ls in zip(masked_model.model.s4_layers, structural_model.s4_layers):
                convert_layer_s4(lm, ls)

        # === COPY NORMS ===
        for nm, ns in zip(masked_model.model.norms, structural_model.norms):
            ns.load_state_dict(nm.state_dict())

        # === COPY CLASSIFIER ===
        structural_model.fc.load_state_dict(
            masked_model.model.fc.state_dict()
        )

        print("\nStarting model profiling...")
        # Model Info
        model_info = ModelInfo(
            model=structural_model, 
            vocab_size=masked_model.dataset.vocab_size if hasattr(masked_model.dataset, 'vocab_size') else masked_model.dataset.input_shape[-1],
            seq_len=masked_model.dataset.input_shape[1], 
            batch_size=config[f"{model_name}"][f"{dataset_name}"]["batch_size"], 
            dataset_name=dataset_name
        )
        os.makedirs("model_info_pruned_structural", exist_ok=True)
        if not os.path.exists(f"./model_info_pruned_structural/{pruned_model_name}_torchinfo.txt"):
            model_info.torchinfo(output_dir="model_info_pruned_structural", name=pruned_model_name)
            print("\nModel info salvato nella cartella 'model_info_pruned_structural'")
        else:
            print("\nModel info già esistente nella cartella 'model_info_pruned_structural', salto la creazione")

        # === SAVE ===
        torch.save(
            {
                "model_state_dict": structural_model.state_dict(),
                "active_idx_layers": active_indices
            },
            os.path.join(f"./checkpoints_pruned_structural", f"{pruned_model_name}.pth")
        )

        print(f"Structural checkpoint saved to: {os.path.join(f'./checkpoints_pruned_structural', f'{pruned_model_name}.pth')}")
    
    # Se il modello prunato non esiste
    else: 
        raise FileNotFoundError(f"Pruned model file not found at path: {model_path}")

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
        required=False, default=os.path.join(".", "checkpoints_pruned"),
        help="Checkpoints folder")
    parser.add_argument(
        "--pruned-model", "-p",
        dest="pruned_model_name",
        required=True, default=None,
        help="Pruned model index")
    args = parser.parse_args()

    main(args.model_name, args.dataset_name, args.checkpoints_folder, args.pruned_model_name)