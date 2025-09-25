import torch
import torch.nn as nn
from .layer_s4d import LayerS4D

class S4D(nn.Module):
    def __init__(self, vocab_size, d_model, d_state=64, n_layers=4, dropout=0.1, num_classes=2):
        super().__init__()
        # Embedding layer: [B, L] -> [B, L, H]
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Stack di S4D layers
        self.layers = nn.ModuleList([
            LayerS4D(d_model, d_state=d_state, dropout=dropout) 
            for _ in range(n_layers)
        ])

        # Normalizzazione
        self.norm = nn.LayerNorm(d_model)

        # Classificazione finale
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # Input: [B, L]
        x = self.embedding(x)     # -> [B, L, H]
        x = x.transpose(1, 2)     # -> [B, H, L]

        # Passa attraverso gli S4D layer
        for layer in self.layers:
            x, _ = layer(x)       # ogni S4D restituisce (out, state)

        # Pooling (media sulle posizioni della sequenza)
        x = x.mean(dim=-1)        # -> [B, H]

        # Normalizzazione + classificazione
        x = self.norm(x)          # -> [B, H]
        out = self.fc(x)          # -> [B, n_classes]

        return out