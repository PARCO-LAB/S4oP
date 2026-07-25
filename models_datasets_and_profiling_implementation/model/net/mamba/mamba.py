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
        vocab_size, # None -> input continuo
        input_size, # feature dim, usato solo quando vocab_size is None
        dataset_name,
        d_model,
        depth,
        dropout,
        num_classes,
        active_idx_layers,
        norm,
        pre_norm,
        d_state,
        d_conv=4,
        expand=2,
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.active_idx_layers = active_idx_layers
        self.dataset_name = dataset_name

        if vocab_size is None:
            # input continuo -> embedding lineare: [B, L, C] -> [B, L, H]
            self.embedding = nn.Linear(input_size, d_model)
        else:
            # input discreto -> embedding token: [B, L] -> [B, L, H]
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        self.mamba_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(depth):
            self.mamba_layers.append(
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    active_idx=active_idx_layers[i]
                )
            )
            if norm.upper() == "LN":
                self.norms.append(nn.LayerNorm(d_model))
            elif norm.upper() == "BN":
                self.norms.append(nn.BatchNorm1d(d_model))
            else:
                raise ValueError(f"Norma non supportata: {norm}")

            self.dropouts.append(dropout_fn(dropout))

        self.norm_f = nn.LayerNorm(d_model)

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

    def encode(self, x):
        # Input: [B, L] (token) oppure [B, L, C] (pathfinder/ecg)
        has_padding = (x.dim() == 2)
        pad_mask = (x != 0) if has_padding else None
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

        # Pooling mean
        x = self.norm_f(x)           # normalizzazione finale -> [B, L, H]
        if pad_mask is not None:
            m = pad_mask.unsqueeze(-1).to(x.dtype)            # [B, L, 1]
            x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            x = x.mean(dim=1)        # [B, H]

        return x

    def forward(self, x):                      # classificazione normale
        return self.fc(self.encode(x))

class RetrievalMamba(Mamba):
    def __init__(self, *args, d_model, num_classes=2, **kwargs):
        super().__init__(*args, d_model=d_model, num_classes=num_classes, **kwargs)
        H = d_model
        del self.fc 
        self.match = torch.nn.Sequential(
            torch.nn.Linear(4 * H, H), torch.nn.GELU(),
            torch.nn.Linear(H, num_classes),
        )

    def forward(self, x):                  # x: [B, 2, L]
        d0, d1 = x[:, 0, :], x[:, 1, :]    # i due documenti
        v0, v1 = self.encode(d0), self.encode(d1)          # encoder CONDIVISO
        feat = torch.cat([v0, v1, (v0 - v1).abs(), v0 * v1], dim=-1)   # [B, 4H]
        return self.match(feat)            # [B, num_classes]