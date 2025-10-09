import torch
import random
import math

def prune_random_channels(model, perc_channels):

    # Individua tutti i LayerS4D
    s4_layers = [m for m in model.modules() if m.__class__.__name__ == "LayerS4D" or m.__class__.__name__ == "S4Block"]
    n_layers = len(s4_layers)
    print(f"Trovati {n_layers} layer s4/s4d nel modello.")
    remove_all_masks(model)

    for i, layer in enumerate(s4_layers):
        # Inizializza active_idx se non presente
        if not hasattr(layer, "active_idx"):
            H = layer.h if layer.__class__.__name__ == "LayerS4D" else layer.d_model
            layer.active_idx = list(range(H))
        else:
            H = len(layer.active_idx)

        # Percentuale crescente di pruning layer-wise
        #k = 2.0
        #factor = (1 - math.exp(-k * (i + 1))) / (1 - math.exp(-k * n_layers))
        factor = (i + 1) / n_layers
        perc = perc_channels * factor
        n_pruned = int(H * perc)

        # Selezione dei nuovi canali da prunare solo tra quelli attivi
        idx_to_remove = random.sample(layer.active_idx, n_pruned)
        layer.active_idx = [idx for idx in layer.active_idx if idx not in idx_to_remove]

        # Maschera aggiornata come buffer
        mask = torch.zeros(layer.h) if layer.__class__.__name__ == "LayerS4D" else torch.zeros(layer.d_model)
        mask[layer.active_idx] = 1
        layer.register_buffer("mask", mask)

        # Applica hook per mascheramento in forward
        apply_mask_hooks(layer)

        print(f"[Layer {i}]: canali rimanenti {len(layer.active_idx)}/{layer.h if layer.__class__.__name__ == 'LayerS4D' else layer.d_model}")

    return model

def apply_mask_hooks(layer):
    if hasattr(layer, "mask"):
        # Registra l’hook sul layer
        layer._mask_hook_handle = layer.register_forward_hook(mask_hook)

def mask_hook(module, input, output):
    # Gestione del caso output = (main_output, *rest)
    if isinstance(output, tuple):
        main_output, *rest = output
    else:
        main_output, rest = output, None

    # Applica solo se è nel formato (B, H, L)
    if main_output.dim() == 3:
        mask = module.mask.to(main_output.device)
        input_tensor = input[0]  # input del layer S4D (B, H, L)

        # Indici dei canali attivi e inattivi
        active_idx = torch.nonzero(mask).view(-1)
        inactive_idx = torch.nonzero(mask == 0).view(-1)

        # Seleziona i valori di input corrispondenti ai canali prunati
        bypass_values = torch.index_select(input_tensor, dim=1, index=inactive_idx)

        # Scatter: sostituisce i canali prunati con i valori di input
        main_output.scatter_(
            dim=1,
            index=inactive_idx.unsqueeze(0).unsqueeze(-1).expand(
                main_output.size(0), -1, main_output.size(2)
            ),
            src=bypass_values
        )

    # Ricostruisce la tupla se necessario
    if rest:
        output = (main_output, *rest)
    else:
        output = main_output

def remove_all_masks(model):
    for layer in model.modules():
        if hasattr(layer, "_mask_hook_handle"):
            layer._mask_hook_handle.remove()
            del layer._mask_hook_handle
        if hasattr(layer, "mask"):
            del layer.mask
