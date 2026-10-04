import torch
import torch.nn as nn
from .layer_s4d import LayerS4D

def dropout_fn(p):
    return nn.Dropout(p) if p > 0.0 else nn.Identity()

class S4D(nn.Module):
    def __init__(
        self,
        vocab_size, # None -> input continuo
        input_size, # feature dim, usato solo quando vocab_size is None
        dataset_name,
        d_model,
        depth,
        dropout,
        num_classes,
        norm,
        pre_norm,
        d_state,
        active_idx_layers,
        pool="mean",
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.pool = pool
        self.active_idx_layers = active_idx_layers
        self.dataset_name = dataset_name

        if vocab_size is None:
            # input continuo -> embedding lineare: [B, L, C] -> [B, L, H]
            self.embedding = nn.Linear(input_size, d_model)
        else:
            # input discreto -> embedding token: [B, L] -> [B, L, H]
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Stack S4D layers + normalizzazione + dropout
        self.s4d_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(depth):
            self.s4d_layers.append(
                LayerS4D(d_model,
                         active_idx=active_idx_layers[i],
                         d_state=d_state, dropout=dropout)
            )
            if norm.upper() == "LN":
                self.norms.append(nn.LayerNorm(d_model))
            elif norm.upper() == "BN":
                self.norms.append(nn.BatchNorm1d(d_model))
            else:
                raise ValueError(f"Norma non supportata: {norm}")
            self.dropouts.append(dropout_fn(dropout))

        self.fc = nn.Linear(d_model, num_classes)

    def _apply_norm(self, norm, x):
        # x: [B, H, L]
        if isinstance(norm, nn.BatchNorm1d):
            return norm(x) # BN normalizza sul canale H
        x = x.transpose(1, 2) # [B, L, H]
        x = norm(x)
        return x.transpose(1, 2) # [B, H, L]

    def encode(self, x):
        # Input: [B, L] (token) oppure [B, L, C] (continuo)
        #pad_mask = (x != 0) if x.dim() == 2 else None # [B, L]
 
        x = self.embedding(x) # -> [B, L, H]
        x = x.transpose(1, 2) # -> [B, H, L]

        for layer, norm, dropout in zip(self.s4d_layers, self.norms, self.dropouts):
            z = x
            if self.pre_norm:
                z = self._apply_norm(norm, z)
            z, _ = layer(z)
            z = dropout(z)
            x = z + x
            if not self.pre_norm:
                x = self._apply_norm(norm, x)

        # Pooling mean (mascherato sui PAD se input tokenizzato)
        # if pad_mask is not None:
        #     m = pad_mask.unsqueeze(1).to(x.dtype) # [B, 1, L]
        #     x = (x * m).sum(dim=-1) / m.sum(dim=-1).clamp(min=1.0)
        # else:
        #     x = x.mean(dim=-1) # [B, H]
        if self.pool == "causal_half":
            x = x[:, :, x.shape[-1] // 2:]
        x = x.mean(dim=-1) # [B, H]
        return x

    def forward(self, x):
        return self.fc(self.encode(x))


class RetrievalS4D(S4D):
    def __init__(self, *args, d_model, num_classes=2, **kwargs):
        super().__init__(*args, d_model=d_model, num_classes=num_classes, **kwargs)
        H = d_model
        del self.fc
        self.match = nn.Sequential(
            nn.Linear(4 * H, H), nn.GELU(),
            nn.Linear(H, num_classes),
        )

    def forward(self, x): # x: [B, 2, L]
        d0, d1 = x[:, 0, :], x[:, 1, :]
        v0, v1 = self.encode(d0), self.encode(d1) 
        feat = torch.cat([v0, v1, (v0 - v1).abs(), v0 * v1], dim=-1) # [B, 4H]
        return self.match(feat)