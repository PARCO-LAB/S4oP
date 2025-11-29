import torch
import torch.nn as nn
from .layer_s4 import S4Block   

def dropout_fn(p):
    if p > 0.0:
        return nn.Dropout(p)
    return nn.Identity()

class S4(nn.Module):
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

        self.pre_norm = pre_norm

        if vocab_size <= 12:
            # Dataset PathFinder → embedding continuo
            self.embedding = nn.Linear(vocab_size, d_model)
        else:
            # Dataset discreto → embedding token
            # Embedding layer: [B, L] -> [B, L, H]
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Stack S4D layers + normalizzazione + dropout
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(depth):
            self.s4_layers.append(
                S4Block(d_model, d_state=d_state, dropout=dropout)
            )
            if norm.upper() == "LN":
                self.norms.append(nn.LayerNorm(d_model))
            elif norm.upper() == "BN":
                self.norms.append(nn.BatchNorm1d(d_model))
            else:
                raise ValueError(f"Norma non supportata: {norm}")

            self.dropouts.append(dropout_fn(dropout))

        # Decoder finale
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Input: [B, L]
        x = self.embedding(x)     # -> [B, L, H]
        x = x.transpose(1, 2)     # -> [B, H, L]

        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.pre_norm:
                # Prenorm
                if isinstance(norm, nn.BatchNorm1d):
                    z = norm(z)                
                else:
                    z = z.transpose(1, 2)     
                    z = norm(z)
                    z = z.transpose(1, 2)

            # Apply S4D block
            z, _ = layer(z)

            # Dropout
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.pre_norm:
                # Postnorm
                if isinstance(norm, nn.BatchNorm1d):
                    x = norm(x)
                else:
                    x = x.transpose(1, 2)
                    x = norm(x)
                    x = x.transpose(1, 2)

        # Pooling: media sulle posizioni della sequenza
        x = x.mean(dim=-1)        # -> [B, H]

        out = self.fc(x)          # -> [B, num_classes]
        return out
