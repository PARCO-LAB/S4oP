import torch
import torch.nn as nn
from .layer_s4d import LayerS4D

class S4D(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        depth,
        dropout,
        num_classes,
        norm,       # "LN" (LayerNorm) oppure "BN" (BatchNorm)
        pre_norm,   # se normalizzare prima o dopo il blocco
        d_state=64, # stato interno di S4D
    ):
        super().__init__()

        # Embedding layer: [B, L] -> [B, L, H]
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Stack di S4D layers
        self.layers = nn.ModuleList([
            LayerS4D(d_model, d_state=d_state, dropout=dropout)
            for _ in range(depth)
        ])

        # Normalizzazione finale (in base alla config)
        if norm.upper() == "LN":
            self.norm = nn.LayerNorm(d_model)
        elif norm.upper() == "BN":
            # BatchNorm1d aspetta input [B, C, L], quindi va applicata prima del pooling
            self.norm = nn.BatchNorm1d(d_model)
        else:
            raise ValueError(f"Norma non supportata: {norm}")

        self.pre_norm = pre_norm

        # Classificazione finale
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Input: [B, L]
        x = self.embedding(x)     # -> [B, L, H]
        x = x.transpose(1, 2)     # -> [B, H, L]

        # Passa attraverso gli S4D layer
        for layer in self.layers:
            if self.pre_norm:
                # Normalizzazione prima del layer
                if isinstance(self.norm, nn.BatchNorm1d):
                    x = self.norm(x)       # BN accetta [B, C, L]
                else:
                    x = x.transpose(1, 2)  # [B, L, H]
                    x = self.norm(x)
                    x = x.transpose(1, 2)  # torna a [B, H, L]

            x, _ = layer(x)  # ogni S4D restituisce (out, state)

        # Pooling (media sulle posizioni della sequenza)
        x = x.mean(dim=-1)        # -> [B, H]

        if not self.pre_norm:
            # Normalizzazione dopo il pooling
            x = self.norm(x)

        out = self.fc(x)          # -> [B, n_classes]
        return out
