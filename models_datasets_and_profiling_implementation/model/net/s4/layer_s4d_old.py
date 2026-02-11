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
    """Generate convolution kernel from diagonal SSM parameters.
       Supports computing only a subset of channels (idx).
    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)

        self.register("log_dt", log_dt, lr)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L, idx=None):
        if idx is None:
            dt = torch.exp(self.log_dt)
            C = torch.view_as_complex(self.C)
            log_A_real = self.log_A_real
            A_imag = self.A_imag
        else:
            if idx.numel() == 0:
                return torch.empty((0, L))
            dt = torch.exp(self.log_dt[idx])
            C = torch.view_as_complex(self.C)[idx]
            log_A_real = self.log_A_real[idx]
            A_imag = self.A_imag[idx]

        A = -torch.exp(log_A_real) + 1j * A_imag
        dtA = A * dt.unsqueeze(-1)
        arange_L = torch.arange(L, device=self.C.device)
        Ktmp = dtA.unsqueeze(-1) * arange_L
        Cmod = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum('hn, hnl -> hl', Cmod, torch.exp(Ktmp)).real
        return K

    def register(self, name, tensor, lr=None):
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            getattr(self, name)._optim = optim

class LayerS4D_old(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.register_buffer("pruning_mask", torch.ones(d_model, dtype=torch.bool))
        self.D = nn.Parameter(torch.randn(self.h))
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)
        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout) if dropout > 0 else nn.Identity()

        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def set_pruning_mask(self, mask):
        if isinstance(mask, torch.Tensor) and mask.dtype == torch.bool:
            if mask.numel() != self.h:
                raise ValueError("mask length must equal d_model")
            self.pruning_mask = mask.clone()
        else:
            new_mask = torch.zeros(self.h, dtype=torch.bool)
            new_mask[mask] = True
            self.pruning_mask = new_mask

    def forward(self, u, **kwargs):
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        mask = self.pruning_mask
        active_idx = mask.nonzero(as_tuple=True)[0]
        inactive_idx = (~mask).nonzero(as_tuple=True)[0]

        y = torch.zeros_like(u)

        if active_idx.numel() > 0:
            k_act = self.kernel(L=L, idx=active_idx)
            kf = torch.fft.rfft(k_act, n=2 * L)
            u_act = u.index_select(1, active_idx)
            uf = torch.fft.rfft(u_act, n=2 * L)
            y_act = torch.fft.irfft(uf * kf, n=2 * L)[..., :L]
            D_act = self.D.index_select(0, active_idx).view(1, -1, 1)
            y_act = y_act + u_act * D_act
            y = y.index_copy(1, active_idx, y_act)

        if inactive_idx.numel() > 0:
            u_inact = u.index_select(1, inactive_idx)
            y = y.index_copy(1, inactive_idx, u_inact)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)

        if not self.transposed:
            y = y.transpose(-1, -2)
        return y, None