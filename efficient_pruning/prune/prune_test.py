import torch
import torch.nn as nn
import random

def prune_random_channels(model, perc_channels):
    
    def prune_layer_s4d(layer, keep_idx):
        with torch.no_grad():
            layer.D = nn.Parameter(layer.D[keep_idx])

            # Parametri del kernel
            if hasattr(layer.kernel, "C") and layer.kernel.C is not None:
                layer.kernel.C = nn.Parameter(layer.kernel.C[keep_idx, :])
            if hasattr(layer.kernel, "log_dt") and layer.kernel.log_dt is not None:
                layer.kernel.log_dt = nn.Parameter(layer.kernel.log_dt[keep_idx])
            if hasattr(layer.kernel, "log_A_real") and layer.kernel.log_A_real is not None:
                layer.kernel.log_A_real = nn.Parameter(layer.kernel.log_A_real[keep_idx, :])
            if hasattr(layer.kernel, "A_imag") and layer.kernel.A_imag is not None:
                layer.kernel.A_imag = nn.Parameter(layer.kernel.A_imag[keep_idx, :])

            # Output Linear + GLU
            if hasattr(layer, "output_linear"):
                conv = layer.output_linear[0]
                W, b = conv.weight, conv.bias
                # input pruning
                W = W[:, keep_idx, :]
                # output pruning (GLU)
                out_keep_idx = []
                for i in keep_idx:
                    out_keep_idx.extend([2*i, 2*i+1])
                W = W[out_keep_idx, :, :]
                b = b[out_keep_idx]

                conv.in_channels = len(keep_idx)
                conv.out_channels = 2*len(keep_idx)
                conv.weight = nn.Parameter(W)
                conv.bias = nn.Parameter(b)
                layer.output_linear[1] = nn.GLU(dim=-2)

            layer.h = len(keep_idx)
            layer.d_output = layer.h

    def prune_norm(norm_layer, keep_idx):
        with torch.no_grad():
            if isinstance(norm_layer, nn.BatchNorm1d):
                bn = norm_layer
                bn.num_features = len(keep_idx)
                bn.weight = nn.Parameter(bn.weight.data[keep_idx].clone())
                bn.bias   = nn.Parameter(bn.bias.data[keep_idx].clone())
                bn.running_mean = bn.running_mean[keep_idx].clone()
                bn.running_var  = bn.running_var[keep_idx].clone()
            elif isinstance(norm_layer, nn.LayerNorm):
                ln = norm_layer
                ln.normalized_shape = (len(keep_idx),)
                ln.weight = nn.Parameter(ln.weight.data[keep_idx].clone())
                ln.bias   = nn.Parameter(ln.bias.data[keep_idx].clone())

    # Individua tutti i LayerS4D
    s4_layers = [m for m in model.modules() if m.__class__.__name__ == "LayerS4D"]

    # Numero di canali da mantenere
    H = s4_layers[0].h
    n_remove = int(H * perc_channels)

    # Selezione globale degli indici da tenere (uguale per tutti i layer)
    idx_to_remove = random.sample(range(H), n_remove)
    keep_idx = [j for j in range(H) if j not in idx_to_remove]

    print(f"Pruning globale: keeping {len(keep_idx)} / {H} channels")

    for i, layer in enumerate(s4_layers):

        # Embedding
        if i == 0:
            emb_w = model.embedding.weight.data
            new_emb = nn.Embedding(
                num_embeddings=model.embedding.num_embeddings,
                embedding_dim=len(keep_idx),
                padding_idx=model.embedding.padding_idx
            ).to(emb_w.device)
            new_emb.weight.data.copy_(emb_w[:, keep_idx].clone())
            model.embedding = new_emb
            print(f"Shape after pruning Embedding: ({new_emb.num_embeddings}, {new_emb.embedding_dim})")

        # Prune LayerS4D
        prune_layer_s4d(layer, keep_idx)
        print(f"Shape after pruning LayerS4D {i}: ({layer.h}, {layer.d_output})")

        # Prune norm
        prune_norm(model.norms[i], keep_idx)
        print(f"Shape after pruning Norm {i}: ({len(keep_idx)})")

        # Ultimo layer: fc
        if i == len(s4_layers) - 1:
            fc_old = model.fc
            new_fc = nn.Linear(len(keep_idx), fc_old.out_features).to(fc_old.weight.device)
            new_fc.weight.data.copy_(fc_old.weight.data[:, keep_idx].clone())
            new_fc.bias.data.copy_(fc_old.bias.data.clone())
            model.fc = new_fc
            print(f"Shape after pruning Final FC: ({new_fc.in_features}, {new_fc.out_features})")

    return model
