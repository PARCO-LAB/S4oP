import torch
import torch.nn as nn
from .mamba_block import Mamba as MambaBlock


def dropout_fn(p):
    if p > 0.0:
        return nn.Dropout(p)
    return nn.Identity()

class Mamba(nn.Module):

    def __init__(
        self,
        vocab_size,
        dataset_name,
        d_model,
        depth,
        dropout,
        num_classes,
        active_idx_layers,
        norm,
        pre_norm,
        d_state=16,
        d_conv=4,
        expand=2,
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.active_idx_layers = active_idx_layers
        self.dataset_name = dataset_name

        if dataset_name in ["pathfinder", "ecg"]:
            # input continuo -> embedding lineare: [B, L, C] -> [B, L, H]
            self.embedding = nn.Linear(vocab_size, d_model)
        else:
            # input discreto -> embedding token: [B, L] -> [B, L, H]
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        self.mamba_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(depth):
            self.mamba_layers.append(
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            )
            if norm.upper() == "LN":
                self.norms.append(nn.LayerNorm(d_model))
            elif norm.upper() == "BN":
                self.norms.append(nn.BatchNorm1d(d_model))
            else:
                raise ValueError(f"Norma non supportata: {norm}")

            self.dropouts.append(dropout_fn(dropout))

        self.fc = nn.Linear(d_model, num_classes)

    def _apply_norm(self, norm, z):
        # z: [B, L, H]
        if isinstance(norm, nn.BatchNorm1d):
            z = z.transpose(1, 2)   # [B, H, L]  (BatchNorm1d normalizza sul canale H)
            z = norm(z)
            z = z.transpose(1, 2)   # [B, L, H]
        else:                       # LayerNorm sull'ultima dim
            z = norm(z)
        return z

    def forward(self, x):
        # Input: [B, L] (token) oppure [B, L, C] (pathfinder/ecg)
        x = self.embedding(x)        # -> [B, L, H]

        for layer, norm, dropout in zip(self.mamba_layers, self.norms, self.dropouts):
            z = x
            if self.pre_norm:
                z = self._apply_norm(norm, z)

            z = layer(z)             # blocco Mamba: [B, L, H] -> [B, L, H]
            z = dropout(z)
            x = z + x                # residual connection

            if not self.pre_norm:
                x = self._apply_norm(norm, x)

        x = x.mean(dim=1)            # pooling sulla sequenza -> [B, H]
        out = self.fc(x)            # -> [B, num_classes]
        return out