"""CPU-friendly deep anomaly models for multivariate power time series."""
from __future__ import annotations

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class PatchTSTAutoencoder(nn.Module):
    """Channel-independent patch Transformer with sequence reconstruction head."""
    def __init__(self, n_features, window=64, patch_len=8, d_model=64, nhead=4, layers=2):
        super().__init__()
        if window % patch_len:
            raise ValueError("window must be divisible by patch_len")
        self.n_features, self.window, self.patch_len = n_features, window, patch_len
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos = PositionalEncoding(d_model, window // patch_len)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.decode = nn.Linear(d_model, patch_len)

    def forward(self, x):
        b, w, c = x.shape
        patches = x.transpose(1, 2).reshape(b * c, w // self.patch_len, self.patch_len)
        z = self.encoder(self.pos(self.patch_embed(patches)))
        return self.decode(z).reshape(b, c, w).transpose(1, 2)


class TranAD(nn.Module):
    """Two-phase self-conditioning Transformer reconstruction model."""
    def __init__(self, n_features, window=64, d_model=64, nhead=4, layers=2):
        super().__init__()
        self.n_features, self.window = n_features, window
        self.input = nn.Linear(n_features * 2, d_model)
        self.pos = PositionalEncoding(d_model, window)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.output = nn.Linear(d_model, n_features)

    def _phase(self, x, condition):
        return self.output(self.encoder(self.pos(self.input(torch.cat([x, condition], dim=-1)))))

    def forward(self, x):
        first = self._phase(x, torch.zeros_like(x))
        focus = (x - first).pow(2).detach()
        second = self._phase(x, focus)
        return first, second


def build_model(name, n_features, **kwargs):
    if name == "patchtst":
        return PatchTSTAutoencoder(n_features, **kwargs)
    if name == "tranad":
        return TranAD(n_features, **kwargs)
    raise ValueError(f"unknown model: {name}")
