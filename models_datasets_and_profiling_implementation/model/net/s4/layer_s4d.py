"""Minimal version of S4D with operator pruning support."""

import math
import torch
import torch.nn as nn
from einops import rearrange, repeat

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.tie = tie
        self.transposed = transposed

    def forward(self, X):
        if self.training:
            if not self.transposed:
                X = rearrange(X, 'b ... d -> b d ...')
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            mask = torch.rand(mask_shape, device=X.device) < (1. - self.p)
            X = X * mask * (1.0 / (1. - self.p))
            if not self.transposed:
                X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X

class S4DKernel(nn.Module):
    def __init__(self, H_active, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        self.H = H_active
        self.N = N

        log_dt = torch.rand(self.H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        C = torch.randn(self.H, N // 2, dtype=torch.cfloat)

        self.C = nn.Parameter(torch.view_as_real(C))
        log_A_real = torch.log(0.5 * torch.ones(self.H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=self.H)

        self.register("log_dt", log_dt, lr)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def register(self, name, tensor, lr=None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            getattr(self, name)._optim = optim

    def forward(self, L):
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag

        dtA = A * dt.unsqueeze(-1)
        arange_L = torch.arange(L, device=C.device)

        Ktmp = dtA.unsqueeze(-1) * arange_L
        Cmod = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum('hn, hnl -> hl', Cmod, torch.exp(Ktmp)).real
        return K

class LayerS4D(nn.Module):
    def __init__(
        self,
        d_model,
        active_idx,          # lista o tensor di indici attivi
        d_state=64,
        dropout=0.0,
        transposed=True,
        **kernel_args,
    ):
        super().__init__()
        self.h = d_model
        self.transposed = transposed

        if not isinstance(active_idx, torch.Tensor):
            active_idx = torch.tensor(active_idx, dtype=torch.long)

        self.register_buffer("active_idx", active_idx)

        self.h_active = active_idx.numel()

        # === PARAMETRI SOLO PER CANALI ATTIVI ===
        self.kernel = S4DKernel(self.h_active, N=d_state, **kernel_args)
        self.D = nn.Parameter(torch.randn(self.h_active))

        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout) if dropout > 0 else nn.Identity()

        # Output mixing invariato (shape [B, H, L])
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):
        if not self.transposed:
            u = u.transpose(-1, -2)

        B, H, L = u.shape
        y = u.clone()  # passthrough di default

        if self.h_active > 0:
            u_act = u.index_select(1, self.active_idx)

            k = self.kernel(L)
            kf = torch.fft.rfft(k, n=2 * L)
            uf = torch.fft.rfft(u_act, n=2 * L)

            y_act = torch.fft.irfft(uf * kf, n=2 * L)[..., :L]
            y_act = y_act + u_act * self.D.view(1, -1, 1)

            y.index_copy_(1, self.active_idx, y_act)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)

        if not self.transposed:
            y = y.transpose(-1, -2)

        return y, None
